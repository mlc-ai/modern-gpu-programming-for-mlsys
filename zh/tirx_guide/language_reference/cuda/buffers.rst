..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Buffer 与内存
=============

Parameter buffer 通过 ``T.match_buffer`` 绑定，kernel body 中的临时 buffer
则由下面介绍的两种 declaration API 创建。使用 ``A[i, j]`` 访问元素，
``A[m0:m0+BM, 0:BK]`` 取得 ``BufferRegion``，``A.ptr_to([i, j])`` 取得
元素 pointer，``A.data`` 取得原始 data pointer。

声明 Buffer
-----------

创建 buffer 的两个基本 API 是：

- ``T.alloc_buffer(shape, dtype, scope=..., ...)``：**分配新的 storage**，
  生成一个 ``AllocBuffer`` node，并返回对应的 ``Buffer``。
  ``T.alloc_shared`` 和 ``T.alloc_local`` 分别是
  ``scope="shared"`` 和 ``scope="local"`` 的简写。
- ``T.decl_buffer(shape, dtype, data=..., ...)``：在已有 pointer ``data``
  上**声明一个 view**，不会分配 storage。它可以为 pool 的一段区域或
  tensor-memory address 创建 alias，也可以重新解释已有 storage。
  ``data=None`` 时，它和 ``alloc_buffer`` 一样会分配 storage。

Buffer 的 ``data`` pointer 是 immutable ``Var``：``alloc_buffer`` 创建它，
``decl_buffer`` 接收它。如果已有的是 pointer 表达式，需要先将表达式绑定为
``Var``，详见 :doc:`data_types`。

两种 API 使用同一个 buffer descriptor，主要参数如下：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 参数
     - 含义
   * - ``dtype``
     - 元素类型，例如 ``"float32"``、``"float16"``、``"float4_e2m1fn"``
   * - ``shape``
     - 逻辑 shape，也就是各维 extents 组成的 tuple
   * - ``layout``
     - 物理映射，详见 :ref:`TileLayout <chap_tirx_layout_api>`；
       ``"default"`` 表示 dense row-major
   * - ``elem_offset`` / ``allocated_addr``
     - ``elem_offset``（或 ``byte_offset``）将 view 放在 ``data`` 内的某个
       offset；``allocated_addr`` 保存预先分配的地址，用于 tensor memory
   * - ``align``
     - Data pointer 的 byte alignment

``scope`` 参数选择 memory space：

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - Scope
     - 简写
     - Memory
   * - ``"global"``
     - 默认值
     - Device global memory
   * - ``"shared"``
     - ``T.alloc_shared``
     - Static shared memory（``__shared__``）
   * - ``"shared.dyn"``
     - pool
     - Dynamic shared memory，使用 pool 管理
   * - ``"local"``
     - ``T.alloc_local``
     - Per-thread registers
   * - ``"tmem"``
     - TMEM pool
     - Blackwell tensor memory

.. code-block:: python

    A = T.match_buffer(A_ptr, (M, K), "float16", align=16)   # parameter buffer
    As = T.alloc_shared((BM, BK), "float16")                 # new shared tile
    acc = T.alloc_local((4,), "float32")                     # register accumulator
    view = T.decl_buffer((BM, BK), "float16", data=As.data)  # a view over As

对于非 TMEM buffer，基于 pointer 的 buffer 只是 pointer 加上一组 metadata。
访问元素时，编译器根据 layout 计算地址：

::

    addr(buffer[coord]) = buffer.data + elem_offset + layout.apply(coord, shape=shape)["m"]

``layout.apply`` 返回各个 physical axis 的映射，其中 ``"m"`` 分量是 element
offset。因此，同一个逻辑访问会因为 buffer metadata 不同而生成不同的地址
计算。下面都在 ``4×8`` 区域上执行 ``B[i, j] = A[i, j] + 1``，但用四种
方式声明 ``B``：

.. code-block:: python

    from tvm.tirx.layout import TileLayout, S

    B = T.match_buffer(p, (4, 8), "float32")                                       # row-major
    B = T.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(1, 4)]))  # column-major
    B = T.match_buffer(p, (4, 8), "float32", elem_offset=64)                       # shifted view
    B = T.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(16, 1)])) # row stride 16

生成的 CUDA 会为 ``B[i, j]`` 使用不同的 index。``A[i, j]`` 的 load 始终
保持 ``i*8 + j``，变化的只有 ``B`` 的 metadata：

.. code-block:: c++

    B_ptr[((i * 8) + j)]        = ...;   // row-major:        i*8 + j
    B_ptr[((j * 4) + i)]        = ...;   // column-major:     j*4 + i
    B_ptr[(((i * 8) + j) + 64)] = ...;   // elem_offset=64:   i*8 + j + 64
    B_ptr[((i * 16) + j)]       = ...;   // row stride 16:    i*16 + j

Shared Memory
-------------

Shared memory 分为两种：**static shared memory** 的大小在编译期固定，
**dynamic shared memory** 的大小在 launch 时确定。TIRx 还提供 pool helper
管理 dynamic shared memory。

Static Shared Memory
~~~~~~~~~~~~~~~~~~~~

``T.alloc_shared``（也就是 ``scope="shared"``）创建最简单的 static shared
buffer，其大小在编译期确定。下面先将数据写入 shared memory，通过
``cta_sync`` 确保整个 block 都能看到写入，再读出结果：

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

它会生成普通的 ``__shared__`` array。下面省略了无关的 CUDA 代码：

.. code-block:: c++

    extern "C" __global__ void __launch_bounds__(128)
    smem_demo_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr) {
      int tx = ((int)threadIdx.x);
      __shared__ alignas(64) float sm_ptr[128];      // T.alloc_shared
      sm_ptr[tx] = A_ptr[tx];
      __syncthreads();                               // T.cuda.cta_sync()
      B_ptr[tx] = sm_ptr[tx] * 2.0f;
    }

Dynamic Shared Memory
~~~~~~~~~~~~~~~~~~~~~

**Dynamic shared memory**（``scope="shared.dyn"``）的大小由 launch parameter
``sharedMemBytes`` 指定，而不是在编译期写死。一个 kernel 只能有**一个**
dynamic-shared allocation，也就是 arena。因此需要先分配一次 arena，再用
``T.decl_buffer`` 在 arena pointer 的不同 ``elem_offset`` 上声明 views：

.. code-block:: python

    arena = T.alloc_buffer((128,), "float32", scope="shared.dyn")   # the one arena
    As = T.decl_buffer((64,), "float32", data=arena.data, scope="shared.dyn")                 # offset 0
    Bs = T.decl_buffer((64,), "float32", data=arena.data, elem_offset=64, scope="shared.dyn") # offset 64
    As[tx] = A[tx]
    Bs[tx] = B[tx]
    T.cuda.cta_sync()
    C[tx] = As[tx] + Bs[tx]

两个 views 共用同一个 ``extern __shared__`` arena。下面为便于阅读，将 arena
命名为 ``smem``，并省略无关代码：

.. code-block:: c++

    extern __shared__ __align__(64) float smem[];   // the one dynamic-shared arena
    smem[tx]      = A_ptr[tx];                       // As — view at offset 0
    smem[tx + 64] = B_ptr[tx];                       // Bs — view at offset 64
    __syncthreads();
    C_ptr[tx] = smem[tx] + smem[tx + 64];

分别调用两次 ``alloc_buffer(scope="shared.dyn")`` 会报错，因为只允许一个
dynamic shared memory allocation。Static shared memory 的大小在编译期
确定，例如 ``__shared__ T x[N];``；dynamic shared memory 则是 launch 时
指定大小的单个 arena，各个 buffer 是 arena 内不同 offset 上的 views。

.. note::

   **TVM 如何记录 dynamic-shared 大小。** Arena 的大小在编译期已知，本例中
   是 ``128`` 个 float，也就是 ``512`` bytes。Lowering 时，TVM 会在 device
   kernel 的 ``tirx.kernel_launch_params`` 中加入
   ``"tirx.use_dyn_shared_memory"`` tag。Host launcher 计算总 byte 数，并将
   它作为最后一个 launch argument：

   .. code-block:: python

       # device kernel 属性：
       "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"]

       # host 侧 launch 调用（..., gridDim.x, blockDim.x, dyn_shared_bytes）：
       T.call_packed("dyn_kernel", A.data, B.data, C.data, 1, 64, 512)

   运行时，这里的 ``512`` 会成为 ``cuLaunchKernelEx`` 调用中的
   ``config.sharedMemBytes``。用户不需要手工设置，它由 ``shared.dyn``
   allocation 的大小推导得到。

SMEMPool
~~~~~~~~

``T.SMEMPool`` 自动管理 arena。它使用 bump allocation 计算 offset，因此
不需要手工 ``decl`` 每个 view。除了 ``alloc`` 和 ``commit``，它还支持
per-buffer ``align=``、创建 MMA-compatible swizzle layout 的 ``alloc_mma``
helper，以及将 cursor 回退以复用空间的 ``move_base_to``：

.. code-block:: python

    pool = T.SMEMPool()                          # shared.dyn 上的 bump allocator
    As = pool.alloc((BM, BK), "float16", align=128)   # 分配一个 tile
    Bs = pool.alloc((BK, BN), "float16", align=128)
    Cs = pool.alloc_mma((BM, BN), "float16")     # 自动推导 MMA-compatible swizzle
    pool.commit()                                 # 确定 pool 的最终大小
    # pool.move_base_to(offset) 将 cursor 回退到可复用的位置

下方的 TMEM pool 建立在 ``SMEMPool`` 之上。

Registers
---------

Per-thread 临时数据使用 ``local`` scope。通过
``T.alloc_local(shape, dtype)`` 分配后，这些数据只属于当前 thread。使用
静态索引的 local arrays 通常会被 scalarize 到 registers；使用动态索引，或
register pressure 较高时，也可能进入 local memory。

.. code-block:: python

    r = T.alloc_local((4,), "float32")   # 每个 thread 私有的 register array
    for k in T.unroll(4):
        r[k] = A[tx, k]
    # ... 使用 r[0..3] 计算 ...

.. code-block:: c++

    alignas(64) float r_ptr[4];          // per-thread, register-resident
    r_ptr[0] = A_ptr[tx * 4 + 0];
    r_ptr[1] = A_ptr[tx * 4 + 1];
    // ...

.. note::

   ``alignas(64)`` 来自 buffer 的默认 alignment。``data_alignment`` 默认为
   ``runtime::kAllocAlignment``，也就是 64 bytes；CUDA codegen 会把它加到
   每个 allocation 上，包括 per-thread ``local`` array。对于保存在
   registers 中的 array，这个 alignment **不会影响性能**：只要 index 能在
   编译期解析，nvcc/ptxas 就会通过 scalar replacement of aggregates
   （SROA）将 thread-local array 提升为 registers，因而不会使用可寻址的
   local memory。只有使用动态 index 并 spill 到 local memory 的 array
   才会实际受到 over-alignment 影响，这种情况并不常见。Register local 的
   over-alignment 是一个已知问题，未来应为 ``local`` scope 使用 dtype 的
   natural alignment。

Scalar
~~~~~~

Scalar 本质上是只有**一个元素**的 register array。可以直接分配大小为 1
的 ``local`` buffer，并通过 ``[0]`` 访问：

.. code-block:: python

    phase = T.alloc_local((1,), "int32")   # 单元素 register array
    phase[0] = 0
    while phase[0] < 4:
        acc = acc + A[tx, phase[0]]
        phase[0] += 1

每次都写 ``phase[0]`` 比较繁琐，因此 TIRx 提供 scalar 语法，表示同一个
单元素 register buffer，并允许直接通过名称读写：

.. code-block:: python

    phase: T.int32 = 0                 # mutable scalar，是上一种写法的语法糖
    while phase < 4:
        acc = acc + A[tx, phase]
        phase += 1

    s = T.local_scalar("int32")        # 显式形式；通过名称赋值，而不是 s[0]
    acc: T.float32 = 0.0               # 带类型注解的赋值也会创建 scalar

两种写法在 parse 后会得到结构完全相同的 TIRx。Parser 会将
``phase: T.int32`` 解析为单元素 ``local`` buffer，将 ``phase`` 和
``phase += 1`` 解析为 ``phase[0]`` 和 ``phase[0] += 1``。对两个 kernels
调用 ``tvm.ir.assert_structural_equal`` 会通过；printer 甚至会把显式的
``alloc_local`` 加 ``[0]`` 重新输出为 scalar 语法。因此，parse 完成后两者
没有区别，都会生成 ``alignas(64) int phase_ptr[1];``。Scalar
只是省去了 ``[0]``。``T.local_scalar``、``T.shared_scalar`` 和
``T.alloc_scalar`` 可以显式选择 scope。

.. note::

   **为什么不使用** ``Var`` **？** TIRx ``Var`` 是 immutable 的静态绑定，
   与下面的 ``T.let`` 相同。Scalar 必须是 mutable 的，例如在 loop 或
   accumulator 中重复赋值，因此需要由能够反复 store 的单元素 buffer
   支撑，不能使用 ``Var``。

``let``
~~~~~~~

``T.let`` 是 **immutable** binding，对应一个 ``LetStmt``。它表示命名后的
值，不是 buffer，适合保存派生常量：

.. code-block:: python

    n: T.let = M * K               # immutable binding（LetStmt）
    half: T.let[T.int32] = N // 2  # 显式指定类型

它会生成普通的 C scalar variable，而不是 array，也不需要 ``[0]``。例如，
运行时变量 ``m`` 上的 ``half: T.let = m * 2`` 会生成：

.. code-block:: c++

    int half = m * 2;     // `let` 生成类似 const 的 local variable

由于值不会改变，simplifier 可以自由执行 propagation 和 common
subexpression elimination。因此在使用位置可能直接看到 ``m * 2``，也可能
看到共享的 common-subexpression temporary，而不一定保留 ``half``。

.. note::

   **为什么需要 immutable binding？** 因为值不会改变，arithmetic analyzer
   在简化 ``LetStmt`` 时可以调用 ``analyzer.Bind(var, value)``，将关于这个
   值的结论传播到所有使用位置，包括 constant bounds、表示 divisibility 和
   alignment 的 modular set，以及 ranges。这些信息可用于简化 index、
   消除 bounds check，以及决定 alignment 和 vectorization。Mutable scalar
   是一次 memory load（``buf[0]``），analyzer 不能假设它保持不变，因此无法
   传播这些性质。``let`` 还是一个不需要 allocation 的 pure value，可以自由
   inline、substitute 或 CSE；scalar 则是具有 load/store 语义的单元素 buffer。

Tensor Memory
-------------

Blackwell **tensor memory** 不能像普通 scratch scope 那样直接分配。Kernel
必须通过 warp-uniform ``T.ptx.tcgen05.alloc`` 和 ``tcgen05.dealloc``
intrinsics 显式申请和释放，再用
``T.decl_buffer(..., scope="tmem", allocated_addr=<column>, layout=<tmem layout>)``
在其中声明 tensor view。

``allocated_addr`` 表示 column offset，是必需参数；Tensor Core dispatch
会检查它。因此，未设置 ``allocated_addr`` 的
``T.alloc_buffer(scope="tmem")`` 无法使用。Tensor memory 也不能直接寻址，
只能通过 ``tcgen05`` 的 ``mma``、``ld``、``st`` 和 ``cp`` 访问。

手工管理时，一个 warp 将 allocation 地址写入 shared slot，再在不同 column
offset 上 ``decl`` tensor views，结束时由一个 warp 释放：

.. code-block:: python

    addr = T.alloc_shared((1,), "uint32")             # 保存 allocation base 的 slot
    if warp_id == alloc_warp:                         # tcgen05.alloc 是 warp-uniform
        T.ptx.tcgen05.alloc(T.address_of(addr), n_cols=512, cta_group=cta_group)
    acc = T.decl_buffer((CTA_M, 512), "float32", scope="tmem",
                        allocated_addr=0, layout=tmem_layout)   # column 0 处的 view
    # ... 将 acc 用作 gemm_async / copy_async operand ...
    if warp_id == alloc_warp:
        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
        T.ptx.tcgen05.dealloc(addr[0], n_cols=512, cta_group=cta_group)

此时 column offsets 和 ``tmem_layout`` （datapath D/F layout）都需要手工
管理。下面的 pool 会自动生成同样的步骤。

TMEMPool
~~~~~~~~

``T.TMEMPool`` 封装 warp-uniform alloc/dealloc、column bump allocation 和
datapath layout：

.. code-block:: python

    tmem_addr = pool.alloc((1,), "uint32")          # pool 是 kernel 的 SMEM pool
    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=cta_group,
                           tmem_addr=tmem_addr)
    acc = tmem_pool.alloc((CTA_M, 512), "float32")  # 自动设置 allocated_addr
    tmem_pool.commit()                               # 由一个 warp 发出 tcgen05.alloc
    # ... 使用 acc ...
    tmem_pool.dealloc()                              # 由一个 warp 发出 tcgen05.dealloc

完整示例见第三部分的 GEMM kernels。

Buffer API
----------

``Buffer`` 是 pointer 上的一组 metadata，详见上方 `声明 Buffer`_。因此，
多数 methods 都是在编译期改变 shape、layout 或 index arithmetic，或者返回
pointer，本身不会生成运行时操作。常用 methods 如下：

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Method
     - 作用
   * - ``B.data``
     - 原始 data pointer（``Var``），输出为 ``B_ptr``
   * - ``B.ptr_to([i, j])``
     - 指向某个元素的 typed pointer（``address_of``），输出为 ``&B_ptr[…]``
   * - ``B.vload([i], dtype="float32x4")`` / ``B.vstore([i], v)``
     - Vectorized load/store，输出为 ``*(float4*)(B_ptr + …)``
   * - ``B.view(*shape, layout=…)``
     - 用新的 shape/layout 重新解释同一 storage，不复制数据
   * - ``B.local(*shape, layout=…)``
     - ``local`` buffer 中属于当前 thread 的 private register slice
   * - ``B.permute(*dims)``
     - 交换 axes 后的 view，也就是 transposed layout
   * - ``B.access_ptr(mask, …)``
     - Masked access pointer，即 ``tvm_access_ptr`` builtin，用于将一个 region 传给 intrinsic

**Pointer：``ptr_to`` / ``data``。** ``ptr_to`` 将某个元素的地址传给
intrinsic 或 inline function；``data`` 则是 base pointer：

.. code-block:: python

    B[tx] = T.cuda.func_call("ld", A.ptr_to([tx]), source_code=SRC, return_type="float32")

.. code-block:: c++

    B_ptr[tx] = ld(&A_ptr[tx]);          // ptr_to([tx]) -> &A_ptr[tx];  A.data -> A_ptr

**Vectorized access：``vload`` / ``vstore``。** 一次 wide transfer 搬运多个
元素，另见 :doc:`data_types`：

.. code-block:: python

    B.vstore([tx * 4], A.vload([tx * 4], dtype="float32x4"))

.. code-block:: c++

    *(float4*)(B_ptr + tx * 4) = *(float4*)(A_ptr + tx * 4);

**Reshape / reinterpret：``view`` / ``permute``。** 两者都只修改 metadata，
data pointer 保持不变，变化的是 index arithmetic。``A.view(64, 4)`` 将包含
256 个元素的 buffer 看作 ``64×4``；``A.permute(1, 0)`` 交换两个 axes：

.. code-block:: python

    A2 = A.view(64, 4);     y = A2[tx, 0] + A2[tx, 3]   # A2[tx, j] -> A_ptr[tx*4 + j]
    At = A.permute(1, 0);   z = At[i, j]                # At[i, j]  -> A_ptr[j*4 + i]

.. code-block:: c++

    A2_ptr[tx * 4]  /* +3 */                 // view: row-major 64x4 index
    At_ptr[(j * 4) + i]                       // permute: swapped strides

**Register：``local``。** 对带 thread axis 的 ``local`` layout 做分解，返回
属于当前 thread 的 flat register bundle。Tile primitives 会频繁使用它：

.. code-block:: python

    R  = T.alloc_buffer((32, 8), "float32", scope="local", layout=TileLayout(S[(32, 8) : (1 @ laneid, 1)]))
    Rl = R.local(8)          # this lane's 8 registers

.. code-block:: c++

    alignas(64) float Rl_ptr[8];             // the lane's private registers
