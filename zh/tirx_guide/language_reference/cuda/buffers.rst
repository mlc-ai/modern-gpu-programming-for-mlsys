Buffer 与 memory
================

参数 buffer 使用 ``T.match_buffer`` 绑定；临时 scratch buffer 在函数体中使用下面两类声明 API 创建。用 ``A[i, j]`` 索引 buffer，用 ``A[m0:m0+BM, 0:BK]`` 切片（得到 ``BufferRegion``），并用 ``A.ptr_to([i, j])`` 取得 pointer，或用 ``A.data`` 取得原始 data pointer。

声明 buffer
-----------

有两个基础 API 会创建 buffer：

- ``T.alloc_buffer(shape, dtype, scope=..., ...)`` — 分配新的 storage（生成 ``AllocBuffer`` 节点），并返回 ``Buffer``。``T.alloc_shared`` / ``T.alloc_local`` 只是带有 ``scope="shared"`` / ``scope="local"`` 的 ``alloc_buffer``。
- ``T.decl_buffer(shape, dtype, data=..., ...)`` — 在已有 pointer ``data`` 上声明一个 view（不分配）；用于给 storage 起别名或重新解释 storage，例如 pool 的某个子区域，或一个 tensor-memory address。当 ``data=None`` 时，它会像 ``alloc_buffer`` 一样分配。

Buffer 的 ``data`` pointer 是一个 immutable ``Var``（``alloc_buffer`` 会定义它；``decl_buffer`` 会接收它）。如果要用一个 pointer *expression* 作为 buffer backing，需要先把它绑定起来，参见 :doc:`data_types`。

二者共用同一类 descriptor；最重要的参数如下：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 参数
     - 含义
   * - ``dtype``
     - element type，例如 ``"float32"``、``"float16"``、``"float4_e2m1fn"`` 等
   * - ``shape``
     - 逻辑 shape（一组 extent）
   * - ``layout``
     - 物理映射（:ref:`TileLayout <chap_tirx_layout_api>`）；``"default"`` 表示 dense row-major
   * - ``elem_offset`` / ``allocated_addr``
     - ``elem_offset``（或 ``byte_offset``）把一个 *view* 放到 ``data`` 中的某个 offset；``allocated_addr`` 携带预先分配的 address（tensor memory）
   * - ``align``
     - data pointer 的对齐，以 byte 为单位

``scope`` 参数选择 memory space：

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - Scope
     - 简写
     - Memory
   * - ``"global"``
     - （默认）
     - device global memory
   * - ``"shared"``
     - ``T.alloc_shared``
     - static shared memory（``__shared__``）
   * - ``"shared.dyn"``
     - （pool）
     - dynamic shared memory（pooled，见下文）
   * - ``"local"``
     - ``T.alloc_local``
     - per-thread register
   * - ``"tmem"``
     - （TMEM pool）
     - Blackwell tensor memory（见下文）

.. code-block:: python

    A = T.match_buffer(A_ptr, (M, K), "float16", align=16)   # parameter buffer
    As = T.alloc_shared((BM, BK), "float16")                 # new shared tile
    acc = T.alloc_local((4,), "float32")                     # register accumulator
    view = T.decl_buffer((BM, BK), "float16", data=As.data)  # a view over As

**基于 pointer 的 buffer 只是 pointer 之上的 metadata。** 对任何非 tmem buffer 而言，声明就是一个 pointer 加一个 layout，索引会解析成 address::

    addr(buffer[coord]) = buffer.data + elem_offset + layout.apply(coord, shape=shape)["m"]

（``layout.apply`` 返回逐轴映射；其中 ``"m"`` 分量是 element offset。）因此，同一个逻辑访问会完全根据 buffer metadata 编译成不同的 address arithmetic。在 4×8 区域上写 ``B[i, j] = A[i, j] + 1``，如果用四种方式声明 ``B``：

.. code-block:: python

    from tvm.tirx.layout import TileLayout, S

    B = T.match_buffer(p, (4, 8), "float32")                                       # row-major
    B = T.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(1, 4)]))  # column-major
    B = T.match_buffer(p, (4, 8), "float32", elem_offset=64)                       # shifted view
    B = T.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(16, 1)])) # row stride 16

那么每一种都会让 ``B[i, j]`` lower 成生成 CUDA 中不同的 index（``A[i, j]`` load 仍然是 ``i*8 + j``，只有 ``B`` 的 metadata 发生了变化）：

.. code-block:: c++

    B_ptr[((i * 8) + j)]        = ...;   // row-major:        i*8 + j
    B_ptr[((j * 4) + i)]        = ...;   // column-major:     j*4 + i
    B_ptr[(((i * 8) + j) + 64)] = ...;   // elem_offset=64:   i*8 + j + 64
    B_ptr[((i * 16) + j)]       = ...;   // row stride 16:    i*16 + j

Shared memory
-------------

Shared memory 有两种形式：static（编译时固定）和 dynamic（launch 时指定大小），此外还有一个 pool helper 用于管理 dynamic 情况。

Static
~~~~~~

最简单的 shared buffer 是 **static** buffer，也就是 ``T.alloc_shared``（即 ``scope="shared"``），其大小在编译时确定。把数据 stage 到其中，执行 ``cta_sync`` 让整个 block 看到写入，然后再读回：

.. code-block:: python

    @T.prim_func
    def smem_demo(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (128,), "float32")
        B = T.match_buffer(B_ptr, (128,), "float32")
        T.device_entry()
        bx = T.cta_id([1])
        tx = T.thread_id([128])
        sm = T.alloc_shared((128,), "float32")   # static shared memory
        sm[tx] = A[tx]
        T.cuda.cta_sync()
        B[tx] = sm[tx] * T.float32(2.0)

它会 lower 成普通的 ``__shared__`` array（生成 CUDA，省略 boilerplate）：

.. code-block:: c++

    extern "C" __global__ void __launch_bounds__(128)
    smem_demo_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr) {
      int tx = ((int)threadIdx.x);
      __shared__ alignas(64) float sm_ptr[128];      // T.alloc_shared
      sm_ptr[tx] = A_ptr[tx];
      __syncthreads();                               // T.cuda.cta_sync()
      B_ptr[tx] = sm_ptr[tx] * 2.0f;
    }

Dynamic
~~~~~~~

**Dynamic** shared memory（``scope="shared.dyn"``）的大小按 launch 设置（``sharedMemBytes`` launch parameter），而不是在编译时固定。一个 kernel 只能有 **一个** dynamic-shared allocation，也就是 *arena*。因此需要分配一次 arena，再用 ``T.decl_buffer`` 和 arena pointer 加 ``elem_offset`` 把每个 buffer 声明成其中的一个 view：

.. code-block:: python

    arena = T.alloc_buffer((128,), "float32", scope="shared.dyn")   # the one arena
    As = T.decl_buffer((64,), "float32", data=arena.data, scope="shared.dyn")                 # offset 0
    Bs = T.decl_buffer((64,), "float32", data=arena.data, elem_offset=64, scope="shared.dyn") # offset 64
    As[tx] = A[tx]
    Bs[tx] = B[tx]
    T.cuda.cta_sync()
    C[tx] = As[tx] + Bs[tx]

两个 view 共享同一个 ``extern __shared__`` arena（生成 CUDA，省略 boilerplate；这里为清晰起见把 arena 命名为 ``smem``）：

.. code-block:: c++

    extern __shared__ __align__(64) float smem[];   // the one dynamic-shared arena
    smem[tx]      = A_ptr[tx];                       // As — view at offset 0
    smem[tx + 64] = B_ptr[tx];                       // Bs — view at offset 64
    __syncthreads();
    C_ptr[tx] = smem[tx] + smem[tx + 64];

（两个独立的 ``alloc_buffer(scope="shared.dyn")`` 调用是错误的，**只允许一次 dynamic shared memory allocation**。）所以，static shared memory 的大小在编译时确定（``__shared__ T x[N];``）；dynamic shared memory 则是这个按 launch 指定大小的唯一 arena，其他 buffer 是在其中以 offset 声明出来的 view。

.. note::

   **TVM 如何标注 dynamic-shared 大小。** Arena 的大小在编译时已知（这里是 ``128`` 个 float，即 ``512`` byte）。lowering 时，TVM 会向 device kernel 的 ``tirx.kernel_launch_params`` 追加一个 ``"tirx.use_dyn_shared_memory"`` tag，host launcher 会计算总 byte 数，并作为最后一个 launch argument 传入：

   .. code-block:: python

       # device kernel attribute:
       "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"]

       # host-side launch call  (..., gridDim.x, blockDim.x, dyn_shared_bytes):
       T.call_packed("dyn_kernel", A.data, B.data, C.data, 1, 64, 512)

   运行时这个 ``512`` 会变成 ``cuLaunchKernelEx`` 调用中的 ``config.sharedMemBytes``。你不需要手动设置它；它由 ``shared.dyn`` allocation 的大小推导出来。

Pool sugar
~~~~~~~~~~

``T.SMEMPool`` 会自动处理 arena bookkeeping：它以 bump allocation 的方式分配 offset，因此不需要手写 ``decl`` view。除了 ``alloc`` / ``commit`` 之外，它还提供每个 buffer 的 ``align=``、``alloc_mma`` helper（自动构造 MMA 兼容的 swizzle layout），以及 ``move_base_to``，用于回退 cursor 并复用空间：

.. code-block:: python

    pool = T.SMEMPool()                          # bump allocator over shared.dyn
    As = pool.alloc((BM, BK), "float16", align=128)   # carve a tile
    Bs = pool.alloc((BK, BN), "float16", align=128)
    Cs = pool.alloc_mma((BM, BN), "float16")     # MMA-compatible, swizzle inferred
    pool.commit()                                 # finalize the pool's size
    # pool.move_base_to(offset) rewinds the cursor to reuse space

TMEM pool（见下文 `Tensor memory`_）建立在 ``SMEMPool`` 之上。

Registers
---------

Per-thread scratch 位于 register 中。用 ``T.alloc_local(shape, dtype)``（即 ``scope="local"``）分配它：它对每个 thread 私有，并 lower 成保存在 register 中的 local array。

.. code-block:: python

    r = T.alloc_local((4,), "float32")   # per-thread register array
    for k in T.unroll(4):
        r[k] = A[tx, k]
    # ... compute on r[0..3] ...

.. code-block:: c++

    alignas(64) float r_ptr[4];          // per-thread, register-resident
    r_ptr[0] = A_ptr[tx * 4 + 0];
    r_ptr[1] = A_ptr[tx * 4 + 1];
    // ...

.. note::

   ``alignas(64)`` 是 *默认* buffer alignment：buffer 的 ``data_alignment`` 默认是 ``runtime::kAllocAlignment``（64 byte），CUDA codegen 会把它标到每个 allocation 上，包括 per-thread ``local`` array，即使这里没有实际意义。对这些 register-resident array 来说，它 **没有性能影响**：带有静态可解析 index 的 thread-local array 会被 nvcc/ptxas 提升到 register 中（scalar replacement of aggregates, SROA），因此它永远不会存在于可寻址的 local memory 中，alignment 也就是 no-op。（如果动态索引 array spill 到 local memory，它确实会带上这个过度对齐，但这不是常见情况。）register local 的这种过度对齐是一个已知粗糙点，我们计划修复（对 ``local`` scope 使用 dtype 的自然对齐）。

Scalar
~~~~~~

Scalar 只是只有 **一个 element** 的 register array；严格来说，不需要单独的概念。你可以分配一个大小为 1 的 ``local`` buffer 并索引 ``[0]``：

.. code-block:: python

    phase = T.alloc_local((1,), "int32")   # 1-element register array
    phase[0] = 0
    while phase[0] < 4:
        acc = acc + A[tx, phase[0]]
        phase[0] += 1

但到处写 ``phase[0]`` 很笨重，所以 **scalar** 正是这件事的语法糖：一个单元素 register buffer，可以 **按名字** 读写：

.. code-block:: python

    phase: T.int32 = 0                 # mutable scalar (sugar for the above)
    while phase < 4:
        acc = acc + A[tx, phase]
        phase += 1

    s = T.local_scalar("int32")        # explicit form; assign by name (s = ..., not s[0])
    acc: T.float32 = 0.0               # a type-annotated assignment also makes one

二者不只是相似，而是会 parse 成 **结构完全相同的 TIRx**。这个语法糖完全在 parser 中消解：``phase: T.int32`` *就是* 那个单元素 ``local`` buffer，``phase`` / ``phase += 1`` *就是* ``phase[0]`` / ``phase[0] += 1``。对两个 kernel 调用 ``tvm.ir.assert_structural_equal`` 会通过，printer 甚至会把显式的 ``alloc_local`` + ``[0]`` 形式 **重新打印** 成 scalar 形式。因此，一旦 parsing 完成，二者完全没有区别。二者都会 lower 成同一个 ``alignas(64) int phase_ptr[1];``；scalar 只是让你省掉 ``[0]``。（``T.local_scalar`` / ``T.shared_scalar`` / ``T.alloc_scalar`` 会显式选择 scope。）

.. note::

   为什么不用 ``Var`` ？TIRx ``Var`` 是 *immutable* 的，也就是一次性的静态绑定（正是下面 ``T.let`` 产生的东西）。Scalar 需要是 *mutable* 的：你会在 loop 和 accumulator 中重新赋值。因此它必须由可以反复 store 的单元素 buffer backing，而不是由 ``Var`` backing。

``let``
~~~~~~~

``T.let`` binding 是 **immutable** 的，也就是一个 ``LetStmt`` （一个具名值，不是 buffer）。用它表示派生常量：

.. code-block:: python

    n: T.let = M * K               # immutable binding (LetStmt)
    half: T.let[T.int32] = N // 2  # ... with an explicit type

它会 lower 成 **普通的 scalar C variable**，而不是 buffer（没有 array，也没有 ``[0]``）。例如 ``half: T.let = m * 2``（其中 ``m`` 是 runtime 值）：

.. code-block:: c++

    int half = m * 2;     // the `let` -> a const-like local

由于值是 immutable 的，simplifier 可以自由传播它并对它做 CSE，所以在使用处你经常会看到 ``m * 2`` 被直接替换进去（或通过 common-subexpression 临时变量共享），而不是看到对 ``half`` 的引用。

.. note::

   **为什么需要 immutable binding？** 因为值不能改变，arithmetic analyzer 会把 var 绑定到该值（它简化 ``LetStmt`` 时会执行 ``analyzer.Bind(var, value)``），所以关于该值证明出来的事实，包括常量边界、modular set（divisibility / alignment）和 range，都会 **传播到每一次使用**。这会帮助 index simplification、bounds-check elimination，以及 alignment/vectorization 决策。*Mutable* scalar 是一次 memory load（``buf[0]``）：analyzer 不能假设它保持不变，因此这些属性都无法传递。``let`` 也是一个纯 value：没有 allocation，并且可以自由 inline、substitute 或 CSE；而 scalar 是带有 load/store 语义的单元素 buffer。

Tensor memory
-------------

Blackwell *tensor memory* 不是普通的 scratch scope：它必须通过 warp-uniform 的 ``T.ptx.tcgen05.alloc`` / ``tcgen05.dealloc`` intrinsic 显式 reserve 和 free；每个 tensor 都是其中的一个 view，通过 ``T.decl_buffer(..., scope="tmem", allocated_addr=<column>, layout=<tmem layout>)`` 声明。``allocated_addr``（column offset）是必需的，tensor-core dispatch 会断言它存在，因此 ``T.alloc_buffer(scope="tmem")``（它 **不会** 设置该字段）不能工作。与 shared memory 不同，tensor memory 不能被直接寻址：只能通过 ``tcgen05`` ``mma`` / ``ld`` / ``st`` / ``cp`` 读写。

手写时，一个 warp 把 allocation 发到 shared slot 中，你用 column offset 把每个 tensor ``decl`` 成一个 view，最后由一个 warp free 它：

.. code-block:: python

    addr = T.alloc_shared((1,), "uint32")             # slot for the allocated base
    if warp_id == alloc_warp:                         # tcgen05.alloc is warp-uniform
        T.ptx.tcgen05.alloc(T.address_of(addr), n_cols=512, cta_group=cta_group)
    acc = T.decl_buffer((CTA_M, 512), "float32", scope="tmem",
                        allocated_addr=0, layout=tmem_layout)   # view at column 0
    # ... use acc as a gemm_async / copy_async operand ...
    if warp_id == alloc_warp:
        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
        T.ptx.tcgen05.dealloc(addr, n_cols=512, cta_group=cta_group)

你需要自己管理 column offset 和 ``tmem_layout`` （一个 datapath D/F layout）。这正是下面 pool 会生成的序列。

Pool
~~~~

``T.TMEMPool`` 会封装上述所有工作：warp-uniform alloc/dealloc、column bump-allocation，以及 datapath layout：

.. code-block:: python

    tmem_addr = pool.alloc((1,), "uint32")          # pool = the kernel's smem pool
    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=cta_group,
                           tmem_addr=tmem_addr)
    acc = tmem_pool.alloc((CTA_M, 512), "float32")  # allocated_addr set for you
    tmem_pool.commit()                               # emits tcgen05.alloc (one warp)
    # ... use acc ...
    tmem_pool.dealloc()                              # emits tcgen05.dealloc (one warp)

完整示例见第三部分的 GEMM kernel。

Buffer API
----------

``Buffer`` 是 pointer 之上的 metadata（见上文 *声明 buffer*），因此它的大部分方法都是 *compile-time* reshape/reinterpret：它们改变 index arithmetic，或给你一个 pointer，本身不会生成运行时操作。常用方法如下：

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 方法
     - 含义
   * - ``B.data``
     - 原始 data pointer（一个 ``Var``）；打印为 ``B_ptr``
   * - ``B.ptr_to([i, j])``
     - 指向某个 element 的 typed pointer（``address_of``）；打印为 ``&B_ptr[…]``
   * - ``B.vload([i], dtype="float32x4")`` / ``B.vstore([i], v)``
     - vectorized load / store；打印为 ``*(float4*)(B_ptr + …)``
   * - ``B.view(*shape, layout=…)``
     - 在新的 shape/layout 下重新解释同一块 storage（无 copy）
   * - ``B.local(*shape, layout=…)``
     - ``local`` buffer 中属于调用 thread 的私有 register slice
   * - ``B.permute(*dims)``
     - 轴被置换后的 view（transposed layout）
   * - ``B.access_ptr(mask, …)``
     - masked access pointer（``tvm_access_ptr`` builtin），用于把一个 region 传给 intrinsic

**Pointer：``ptr_to`` / ``data``。** ``ptr_to`` 用于把 element address 传给 intrinsic 或 inline function；``data`` 是 base pointer：

.. code-block:: python

    B[tx] = T.cuda.func_call("ld", A.ptr_to([tx]), source_code=SRC, return_type="float32")

.. code-block:: c++

    B_ptr[tx] = ld(&A_ptr[tx]);          // ptr_to([tx]) -> &A_ptr[tx];  A.data -> A_ptr

**Vectorized access：``vload`` / ``vstore``。** 用一次 wide transfer 移动多个 element（另见 :doc:`data_types`）：

.. code-block:: python

    B.vstore([tx * 4], A.vload([tx * 4], dtype="float32x4"))

.. code-block:: c++

    *(float4*)(B_ptr + tx * 4) = *(float4*)(A_ptr + tx * 4);

**Reshape / reinterpret：``view`` / ``permute``。** 二者都是纯 metadata；data pointer 不变，只是 index arithmetic 不同。``A.view(64, 4)`` 会把 256-element buffer 看作 ``64×4``；``A.permute(1, 0)`` 会转置轴：

.. code-block:: python

    A2 = A.view(64, 4);     y = A2[tx, 0] + A2[tx, 3]   # A2[tx, j] -> A_ptr[tx*4 + j]
    At = A.permute(1, 0);   z = At[i, j]                # At[i, j]  -> A_ptr[j*4 + i]

.. code-block:: c++

    A2_ptr[tx * 4]  /* +3 */                 // view: row-major 64x4 index
    At_ptr[(j * 4) + i]                       // permute: swapped strides

**Register：``local``。** 将 thread-axis ``local`` layout 分解成调用 thread 自己的扁平 register bundle（tile primitive 中大量使用）：

.. code-block:: python

    R  = T.alloc_buffer((32, 8), "float32", scope="local", layout=TileLayout(S[(32, 8) : (1 @ laneid, 1)]))
    Rl = R.local(8)          # this lane's 8 registers

.. code-block:: c++

    alignas(64) float Rl_ptr[8];             // the lane's private registers
