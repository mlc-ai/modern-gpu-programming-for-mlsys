CUDA C++/PTX intrinsics
=======================

当没有 tile primitive 覆盖你需要的操作时，有两种 escape hatch 可以直接触达硬件：**调用 backend intrinsic**（来自 ``tvm.backend.cuda`` 的 ``T.cuda.*`` / ``T.ptx.*`` namespace），或者 **inline raw CUDA** 源码。

Calling backend intrinsics
--------------------------

``T.cuda.*`` 和 ``T.ptx.*`` 直接暴露 CUDA backend 的 device intrinsic，包括 synchronization、mbarrier、reduction，以及 PTX data-movement / MMA 指令族：

.. code-block:: python

    T.cuda.cta_sync()                    # block barrier (__syncthreads)
    T.cuda.warp_sync()                   # __syncwarp
    T.cuda.warpgroup_sync(8)             # warpgroup barrier
    T.cuda.cta_sum(val, num_warps, scratch.ptr_to([0]))   # block-level reduction

    bar = T.alloc_shared((1,), "uint64")
    T.ptx.mbarrier.init(bar.data, 1)     # mbarrier for async completion
    T.ptx.mbarrier.try_wait(bar.data, phase)

一个完整可运行示例：通过 ``T.tvm_warp_shuffle_xor`` 做 warp all-reduce：

.. code-block:: python

    @T.prim_func
    def warp_reduce(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (32,), "float32", align=16)
        T.device_entry()
        cta_id = T.cta_id([1]); warp_id = T.warp_id([1]); lane_id = T.lane_id([32])
        v = T.alloc_local((1,), "float32"); i = T.alloc_local((1,), "int32")
        v[0] = T.float32(31 - lane_id)
        i[0] = 16
        while i[0] >= 1:
            v[0] += T.tvm_warp_shuffle_xor(0xFFFFFFFF, v[0], i[0], 32, 32)
            i[0] = i[0] // 2
        A[lane_id] = v[0]

Shuffle 会直接 lower 成 ``__shfl_xor_sync``：

.. code-block:: c++

    v_ptr[0] = v_ptr[0] + __shfl_xor_sync(0xFFFFFFFF, v_ptr[0], i_ptr[0], 32);

``T.ptx.*`` / ``T.cuda.*`` 下还有其他指令族：``cp_async``（LDGSTS）、``cp_async.bulk.tensor``（TMA）、``ldmatrix`` / ``stmatrix``、``tcgen05.*``（Blackwell MMA）、``atomic_add``、``fence`` 等。完整 ``tvm.backend.cuda`` reference 请参阅 backend API reference。

Synchronization semantics
-------------------------

GEMM 和 Flash Attention kernel 中反复出现四种同步机制。由于它们控制异步引擎和并行线程组，误用任何一种通常都会导致 silent corruption 或 deadlock。

**Mbarrier Phase。** Mbarrier 用一个内部 phase bit 跟踪 arrival。``T.ptx.mbarrier.try_wait(bar, phase)`` 会阻塞，直到 barrier 的内部 phase *不同于* 调用方提供的 ``phase`` 参数。因此，跨 loop iteration 复用 barrier 时，调用方必须在每次 wait 后翻转自己的本地 phase tracker（``phase ^= 1``）。如果不这样做，后续 wait 会立即返回，允许 engine 读取半写入的 memory。:ref:`chap_gemm_basics` 给出了完整 phase-tracking 表。

**Election。** ``T.ptx.elect_sync()`` 会在 *一个 warp 内的 active lane* 中选出一个，不一定是 lane 0，也不是每个 CTA 一个线程。要把 issuer 缩窄到恰好一个线程，必须配合 warp-level guard。:ref:`chap_gemm_basics` 中发起 ``Tx.gemm_async`` 和 ``tcgen05.commit`` 时使用的 pattern 是先 ``if warp_id == 0:``，再 ``if T.ptx.elect_sync():``。

**Named Warpgroup Barrier。** ``T.cuda.cta_sync()`` 映射到 ``__syncthreads()``，需要 *每个* CTA thread arrive。一旦 warpgroup specialization 到不同 code path，把 ``cta_sync()`` 放在 warpgroup branch 内就会 deadlock，因为其他 warpgroup 永远到不了。硬件提供 16 个 named barrier（ID 0 到 15）；``T.cuda.warpgroup_sync(10)`` 只同步一个 warpgroup 的线程。不同 warpgroup 使用不同 ID（例如 ``warpgroup_sync(wg_id + 10)``），避免撞到同一个硬件 barrier。见 :ref:`chap_gemm_advanced`。

**Fence。** Fence 会把 producer 的写入排在 consumer（通常是异步 engine）读取之前：

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Fence
     - 排序内容
   * - ``T.ptx.fence.proxy_async("shared::cta")``
     - 线程写入的 shared memory 在 async proxy（TMA store / MMA）读取之前可见
   * - ``T.ptx.fence.mbarrier_init()``
     - mbarrier initialization 在后续 arrival 或 wait 使用 barrier 之前完成
   * - ``T.ptx.tcgen05.fence.after_thread_sync()``
     - ``tcgen05`` writeback edge 上的保守 ordering fence（Steps 8 和 9 添加；TMA-to-MMA 路径不需要）

Inlining raw CUDA
-----------------

对于完全没有 intrinsic 的操作，可以用 ``T.cuda.func_call(name, *args, source_code=..., return_type=...)`` 从源码字符串注入一个 ``__device__`` function：

.. code-block:: python

    SRC = r"""
    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    """

    @T.prim_func
    def k(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        B = T.match_buffer(B_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
        B[tx] = T.cuda.func_call("my_relu", A[tx], source_code=SRC, return_type="float32")

源代码会原样 emit，调用也会连上：

.. code-block:: c++

    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    // ...
    B_ptr[tx] = my_relu(A_ptr[tx]);
