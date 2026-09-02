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

.. _chap_tirx_tile_layout_lowering:

Tile Primitive 与 Layout Lowering
=================================

上一页 :ref:`chap_tirx_lowering_pipeline` 给出了完整 pipeline。本页只深入
``LowerTIRx``，并沿一条 Blackwell ``Tx.gemm_async`` 解释 layout、算子静态分派
和直接 PTX 之间的关系。

.. admonition:: 三个最容易混淆的结论

   1. 默认 row-major layout 的地址可能与传统连续数组完全相同，但 **“映射结果
      相同”不等于“IR 中没有 layout metadata”**。
   2. ``Tx.gemm_async`` 不靠 layout 执行矩阵乘法；它读取 layout 以匹配/验证
      operand 的物理约定，并推导 descriptor、offset 和硬件指令参数。
   3. 直接 ``T.ptx.*`` 省掉的是对应 ``Tx.*`` 的算子级 lowering。周围的 scope
      lowering、普通地址展开、合法化、host/device split 和 codegen 仍然存在。

贯穿示例：一条 ``Tx.gemm_async``
--------------------------------

下面抽取 :ref:`chap_tirx_primer` 中单-tile GEMM 的关键部分。完整 kernel 还
包含 SMEM/TMEM allocation、barrier 初始化、等待和释放；这里仅保留与 lowering
有关的声明与 tile operations：

.. code-block:: python

    BLK_M, BLK_N, BLK_K = 128, 128, 64

    A_layout = mma_shared_layout(
        "float16", SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K)
    )
    B_layout = mma_shared_layout(
        "float16", SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K)
    )

    Asmem = pool.alloc((BLK_M, BLK_K), "float16", layout=A_layout)
    Bsmem = pool.alloc((BLK_N, BLK_K), "float16", layout=B_layout)

    tmem = T.decl_buffer(
        (128, 512),
        "float32",
        scope="tmem",
        allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]),
    )

    if warp_id == 0:
        if T.ptx.elect_sync():
            Tx.gemm_async(
                tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                accum=False, dispatch="tcgen05", cta_group=1,
            )

    Dreg = T.alloc_local((BLK_N,), "float32")
    Dreg_wg = Dreg.view(
        128,
        BLK_N,
        layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]),
    )
    Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])

这些语句已经给 lowering 提供了大部分 schedule：

.. list-table::
   :header-rows: 1
   :widths: 29 71

   * - 输入信息
     - 它表达什么
   * - A/B/C 的 regions
     - 本次 GEMM 的逻辑 ``M``、``N``、``K`` 范围
   * - A/B 的 SMEM layouts
     - shared-memory 中的字节排列，以及选定的 128-byte swizzle
   * - C 的 TMEM layout
     - 声明期望的 accumulator datapath；dispatcher 可从特定 layout 推断
       ``.ws``，随后验证它与最终 tcgen05 datapath 一致并提取 slice offsets
   * - ``warp_id`` 和 ``elect_sync``
     - 将 issuing scope 限定到一个被选中的 thread
   * - ``dispatch="tcgen05"``
     - 强制选择 Blackwell tcgen05 variant
   * - ``Dreg_wg`` layout
     - 声明 readback 后每个逻辑元素的 thread ownership 和局部 slot

其他 copy、reduce 等 tile primitive 也使用相同的 dispatcher 框架，但各自读取的
layout 字段、约束和 lowering 结果并不相同。本例不能代表所有算子的具体硬件规则。

``TilePrimitiveDispatch`` 如何选择实现
------------------------------------------------

前端把 ``Tx.gemm_async(...)`` 表示为 ``TilePrimitiveCall``。一次 dispatch 的输入
由两部分合成：

- ``TilePrimitiveCall`` 携带 operator、operand ``BufferRegion``、config 和
  ``dispatch=``；
- ``DispatchContext`` 提供 target、当前 execution scope、launch parameters、
  variable ranges 和插入初始化/分配语句所需的 callbacks。

候选实现按 ``(operator name, target kind)`` 注册，再按固定 priority 和 variant
名称排序。显式 ``dispatch="tcgen05"`` 只保留该 variant；没有显式指定时，
dispatcher 依次检查 predicates，并使用第一个成功返回 ``PrimFunc`` 的实现。
这是 **静态规则选择**，不是 cost model，也不是 autotuning。

选中的 ``PrimFunc`` body 替换原 ``TilePrimitiveCall``。实现还可以通过 callbacks
请求 private allocation、device initialization、host initialization，或紧跟
某个 buffer definition 的语句。因此，一次 lowering 不一定只在调用位置插入
几条 PTX；它也可能准备 descriptor 或其他依赖资源。

这个 pass 还解析 ``T.device_entry()`` 内的抽象 scope IDs，生成 launch parameters、
``Bind`` 和 ``thread_extent``。例如：

.. code-block:: text

    T.cta_id([grid_x]) / T.thread_id([block_x])
                         ↓
                 blockIdx.x / threadIdx.x

其中 ``grid_x`` 和 ``block_x`` 来自作者声明的 execution hierarchy，dispatcher
不会替 kernel 搜索 block size。

``Tx.gemm_async`` 的四步 lowering
---------------------------------------------

第一步：slice layout 并验证 operands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dispatcher 从三个 ``BufferRegion`` 取得 extents，并对各 buffer layout 执行
``slice`` 与 ``canonicalize``。随后检查：

- C 是否位于 TMEM，A/B 是否位于该 variant 支持的 memory scope；
- operand dtype 和逻辑 ``M/N/K`` 是否满足指令约束；
- A/B 的 sliced layout 是否包含受支持的 SMEM atom、swizzle 和 alignment；
- C 的 sliced layout 是否匹配受支持的 TMEM datapath。

这里不会把不兼容的 layout 自动“优化正确”。无法证明 layout、shape 或
alignment 合法时，该 variant 会被拒绝；没有其他候选成功时，dispatch 失败。

第二步：A/B layout 变成 matrix descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tcgen05`` 从 SMEM 读取 A/B。Dispatcher 将 sliced layout 与硬件支持的 K-major
和 MN-major swizzle atoms 匹配，从匹配结果取得：

.. code-block:: text

    swizzle mode
    leading-dimension offset (ldo)
    stride-dimension offset (sdo)
    K-major / MN-major
    当前 MMA tile 相对 buffer 原点的 16-byte offset

这些字段与 shared-memory base address 一起构成 matrix descriptor。前面的
``Tx.cta.copy`` 或 ``Tx.copy_async`` 与后面的 MMA 因而通过同一个 layout contract
解释 SMEM 中的字节排列，kernel 作者不必手写 descriptor bit fields。

第三步：验证 C datapath 并取得 TMEM 目标
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

示例 C layout 的正向映射是：

.. code-block:: text

    C[m, n] → { TLane: m, TCol: n }

它声明期望怎样解释 accumulator 的硬件坐标。Dispatcher 确定最终 tcgen05
instruction datapath，验证 C layout 与它相容，并提取 sliced region 的
``TLane`` / ``TCol`` offsets；若 region 从非零 column 开始，slice 会产生相应的
``TCol`` offset。最后再与 ``allocated_addr`` 组合成目标 TMEM address。

这里不能理解为“任意 C layout 都能改变硬件 accumulator 排列”。Layout-E
等受支持形式可以影响 ``.ws`` 推断，但最终仍必须匹配 tcgen05 能表达的 datapath。

第四步：shape、dtype 和 config 决定指令分解
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Tx.gemm_async`` 表示整个逻辑 tile，不等于“一行 DSL 固定对应一条 PTX”。
Dispatcher 主要根据逻辑 ``M/N/K``、dtype、``cta_group`` 和 operator config 选择
合法 instruction shape；已有 layout 用来验证 operand 约定并取得每一步的
descriptor offsets。

对于示例中的 fp16 ``128×128×64`` GEMM，tcgen05 每个 K step 处理 16 个 fp16
K elements，因此会产生 4 个 MMA iterations。去掉函数签名和大量常量细节后，
dispatch 后的结构可以概括为：

.. code-block:: text

    # lowering sketch：尖括号内容不是可调用的 Python API
    desc_a = <由 Asmem base、ldo、sdo、swizzle 组成的 TIR expression>
    desc_b = <由 Bsmem base、ldo、sdo、swizzle 组成的 TIR expression>
    desc_i = T.uint32(<dispatcher 在 dispatch 时编码出的常量>)

    for ki in T.unroll(4):
        T.ptx.tcgen05.mma(
            <tmem base + TLane/TCol slice offset>,
            <desc_a + A 的第 ki 个 16-byte offset>,
            <desc_b + B 的第 ki 个 16-byte offset>,
            desc_i,
            enable_input_d=(ki != 0),
            ...
        )

``desc_i`` 对 dense tcgen05 路径是 dispatcher 在编译期间算出的 ``uint32``
常量，不是 runtime 再调用某个 descriptor encoder。``T.unroll(4)`` 则由后面的
``UnrollLoop`` 展开，随后的 ``StmtSimplify`` 会化简各次迭代的常量。由此可以看出：
instruction decomposition 来自 tcgen05 operator lowering，而 loop 展开属于
后续通用 pass。

Readback 延续同一个物理约定
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

tcgen05 按最终确定的 datapath 把结果写入 TMEM；C layout 用于验证并解释这个
物理结果。随后的 ``Tx.wg.copy_async`` 同时读取 C layout 和 ``Dreg_wg`` layout，
选择匹配的 ``tcgen05.ld`` form，并把逻辑 ``(m, n)`` 分配给
``tid_in_wg=m`` 的 thread 及其局部 slot ``n``：

.. code-block:: text

    GMEM
      │  Tx.cta.copy / Tx.copy_async：按 A/B layout 写入
      ▼
    swizzled SMEM
      │  Tx.gemm_async：descriptor 按同一 layout 读取
      ▼
    TMEM (TLane, TCol)
      │  Tx.wg.copy_async：按 C 与 register layouts 解释和读取
      ▼
    per-thread local slots (tid_in_wg, m)

Layout 没有执行 copy 或 MMA；它是 producer 与 consumer 对“同一个逻辑元素
位于哪里”的共同约定。

Layout 的完整生命周期
----------------------

同一个 layout 从创建到消失会经过下面几个阶段：

.. code-block:: text

    parser 默认构造 / helper synthesis / 作者显式构造
                         │
                         ▼
                  attach 到 Buffer
                         │
                         ▼
             Buffer view / region slice
                         │
                         ▼
                  canonicalize / match
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
    operator dispatcher          LowerTIRxCleanup
    消费硬件语义与 offsets        物化剩余的 memory offset
             └───────────┬───────────┘
                         ▼
               layout metadata 被清除
                         │
                         ▼
             backend 只看到地址和 intrinsics

Dispatcher 和 cleanup 不是互斥的二选一。同一个 shared-memory swizzle 可以被
GEMM dispatcher 用来构造 descriptor，同时 cleanup 仍会把这个 buffer 上残留的
普通 ``BufferLoad`` / ``BufferStore`` 展开成物理地址。

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Layout 类型
     - Operator dispatcher 怎样使用
     - Cleanup 怎样使用
   * - 单一 memory axis ``m``，可含 swizzle
     - 若 buffer 参与 tile primitive，可读取它以推导 descriptor、vector width
       或 offsets
     - 将剩余直接 access 物化为一个线性 offset
   * - ``laneid`` / ``tid_in_wg`` 加局部 ``m``
     - Register-aware operator 将它解释为 thread ownership 与局部 slot
     - 不能直接遗留；必须先通过理解它的 operator，或用 ``.view()`` 后再以
       ``.local()`` 取得当前 thread 的 storage view
   * - ``TLane`` / ``TCol``
     - TMEM-aware operator 验证 datapath 并取得硬件地址
     - 不能被普通 TIR ``BufferLoad`` 压成一个 pointer offset

默认 row-major 不等于“没有 layout”
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在本章讨论的 CUDA TIRx ``PrimFunc`` 中，``T.match_buffer``、``T.decl_buffer``
等 buffer APIs 的 ``layout`` 参数默认值就是 ``"default"``。因此作者省略
``layout=`` 时，parser 会自动构造：

.. code-block:: text

    TileLayout(S[shape])

它是 dense row-major layout。三种写法的差别如下：

.. list-table::
   :header-rows: 1
   :widths: 24 35 41

   * - 写法
     - 普通连续 access 的结果
     - IR 中保留的信息
   * - 省略 ``layout=``，或写 ``layout="default"``
     - cleanup 后与传统 row-major 地址相同
     - 有一个可供 slice、match 和 operator 检查的 ``TileLayout``
   * - ``layout=None``
     - 通过普通 shape/stride 规则也可得到相同地址
     - 明确不附带 layout metadata；operator 无法从它取得专用映射
   * - 显式硬件 layout
     - 地址可能包含 padding/swizzle，或映射到 thread/TMEM axes
     - 携带 operator-specific 的物理约定

所以，对普通连续 buffer 来说，“默认 layout”和“没有 layout”可能产生完全
相同的最终地址；差别在于编译器前半程有没有一份统一、可检查的映射契约。
它本身不会凭空带来性能收益，更不等于编译器已经选出了最佳 layout。

Layout 自动到什么程度
~~~~~~~~~~~~~~~~~~~~~~

“自动 layout”常混用三种含义：

1. **默认构造。** Parser 在省略参数时补 dense row-major layout。这只是默认
   语义，不是性能搜索。
2. **Helper synthesis。** ``mma_shared_layout``、``tmem_datapath_layout``、
   ``tcgen05_atom_layout`` 等 helper 根据显式 dtype、shape 和 mode 构造已知
   硬件 layout。选择哪个 helper 和 mode 仍由作者或上层生成器决定。
3. **Lowering-time parameter inference。** Dispatcher 结合既有 layout 与
   shape、dtype、``cta_group`` 和 config，推导 major mode、descriptor fields、
   instruction shape 与 offsets。它是从已给定约定推出底层参数，不是反向搜索
   最优 layout。

默认 TIRx pipeline 没有一个全局 ``InferOptimalLayout`` pass，也不会从任意
手写 PTX 反推出 lane/register、SMEM swizzle 或 TMEM layout。

三种 layout 怎样变成物理位置
-----------------------------

普通 memory layout：逻辑下标变成地址
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

考虑逻辑 shape ``(4, 8)``、row stride 为 16 的 padded layout：

.. code-block:: text

    layout = TileLayout(S[(4, 8) : (16, 1)])

    layout.apply(i, j)["m"] = i * 16 + j

    B[i, j]                  →  B_flat[i * 16 + j]

这是纯映射示例；实际 backing allocation 必须至少容纳
``layout.span() = 56`` 个 elements，而不是只分配逻辑元素数 ``4 * 8 = 32``。
若通过函数参数传入 B，caller 也必须满足这个物理容量约定。

``LowerTIRxCleanup`` 会先把 layout 产生的物理坐标写入 access index，并保留
buffer 的 ``elem_offset`` metadata；后续 ``FlattenBuffer`` 再把
``elem_offset`` 折入最终线性 index。所以上图表示的是最终 **有效地址语义**，
不是声称 cleanup 内某一个 AST 节点已经完成所有后续 folding。

如果使用 ``ComposeLayout``，物理 offset 还可能包含 shared-memory swizzle 的
XOR、shift 和 mask。完整 layout 代数见 :ref:`chap_tirx_layout_api`。

Distributed layout：逻辑元素变成 ownership 与局部 slot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Register-backed tile 通常需要两个物理坐标：哪个 thread 持有元素，以及它在该
thread 的第几个局部 slot。例如：

.. code-block:: python

    fragment_layout = TileLayout(
        S[(8, 4, 2) : (4@laneid, 1@laneid, 1)]
    )

把它解释成逻辑 ``8×8`` tile 时：

.. code-block:: text

    laneid = 4 * row + col // 2
    m      = col % 2

``laneid`` 表示 ownership，``m`` 表示 lane-local slot；``m`` 不是最终 PTX 中
某个固定寄存器编号。真实 register allocation 仍由 CUDA toolchain 完成。

普通 TIR ``BufferLoad`` 无法只凭一个 offset 验证“当前 thread 是否拥有这个
逻辑元素”。因此，含 thread axis 的 layout 如果直接遗留到 cleanup 会报错。
``.view()`` 先建立 distributed logical view，随后还必须通过 ``.local()`` 取得
当前 thread 的 storage view；``.view()`` 单独使用并不能让直接 load 合法。

TMEM layout：逻辑元素变成 ``TLane`` 与 ``TCol``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Blackwell TMEM 使用二维硬件坐标：

.. code-block:: text

    TileLayout(S[(128, N) : (1@TLane, 1@TCol)])

    C[m, n] → { TLane: m, TCol: n }

``TLane`` 是 TMEM 的物理 lane row，不是执行当前代码的 CUDA ``lane_id``。
``TLane`` 与 ``TCol`` 也不是普通 pointer 的两个 strides；tcgen05-aware operator
必须先验证并解释它们。若这种二维坐标直接留给普通 ``BufferLoad``，cleanup
无法把它降成所要求的单一 memory offset。

``LowerTIRxCleanup`` 的准确边界
---------------------------------------------

Dispatcher 完成后，``LowerTIRxCleanup`` 运行 ``LayoutApplier``。对剩余普通
``BufferLoad`` / ``BufferStore``，其核心工作是：

.. code-block:: text

    logical indices
          │
          ▼
    layout.canonicalize().apply(indices, shape)
          │
          ▼
    一个 symbolic physical coordinate
          │
          ▼
    flattened buffer access

对于 CUDA 的直接 memory access，有 layout 时最终必须只产生一个 physical
coordinate，通常命名为 ``m``；没有 layout 时则使用普通 shape/stride 规则。
``LayoutApplier`` 还把 layout-backed buffers 重建为 physical views并清空 layout
metadata。随后 ``BufferOffsetRemover`` 消除 ``tirx.buffer_offset(BufferLoad)``
wrapper，使其中的 offset 与已经展平的访问一致。

这里的 ``symbolic`` 表示编译器在 lowering 时构造、化简地址表达式；表达式中仍可
包含 runtime loop index、``threadIdx`` 或函数参数，并非所有地址都在编译期变成
常量。

完整 ``LowerTIRx`` 成功后，可以依赖：

- 所有 ``TilePrimitiveCall`` 已被具体实现替换，否则 pass 会失败；
- scope IDs 已被解析；
- layout metadata 已被 operator 消费或由 cleanup 物化并清除；
- ``T.ptx.*`` 可以继续存在，尚未变成最终 CUDA source/PTX assembly；
- 类型合法化、host/device split 和 ABI lowering 仍未完成。

高层 tile primitive 与直接 PTX
-------------------------------

``T.ptx.tcgen05.mma`` 是 target intrinsic ``Call``，不是 ``TilePrimitiveCall``。
它不会进入 ``gemm_async`` 的 registered variants，但 surrounding kernel
仍会经过 scope lowering、cleanup、通用 passes、host/device split 和 codegen。

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - 责任
     - ``Tx.gemm_async``
     - 直接 ``T.ptx.tcgen05.mma``
   * - Backend variant
     - Dispatcher 选择，或验证显式 ``dispatch=``
     - 作者已经选择
   * - Logical shape 到 instruction tiles
     - Dispatcher 根据 shape/dtype/config 分解
     - 作者手写每条 instruction
   * - Operand layout 合法性
     - Dispatcher 做 operator-specific matching 和检查
     - 主要由作者保证
   * - SMEM/TMEM descriptor 与 offsets
     - 从 layout、region 和 config 推导
     - 作者手写或调用低级 encoding intrinsics
   * - Lane/local-slot/TMEM operand 顺序
     - Layout 与 dispatcher 共同表达和检查
     - 作者编码在 lane 公式、地址与 operand 顺序中
   * - Simplify、legalize、split、codegen
     - 仍然执行
     - 仍然执行

“作者编码寄存器顺序”不是说作者指定 ``%r17`` 这样的最终物理寄存器号；作者
只是在 local arrays、lane formulas 和 PTX operands 中表达值的相对位置，真正的
register allocation 仍由 ptxas 等工具完成。区别在于：直接 PTX 的 IR 不再保留
“这个 operand 对应逻辑 ``C[m,n]``”的完整 tile-op contract，dispatcher 因而无法
替作者做同等级的结构匹配与 layout mismatch 诊断。

直接 PTX 也不必然绕过所有 memory layout：

- 若 intrinsic 参数经 ``buf.ptr_to(logical_indices)`` 构造，且 buffer layout
  能降低为 **单一 memory-axis offset**，cleanup 仍会把逻辑 indices 映射成
  物理地址。
- Distributed/thread-axis layout 或 ``TLane/TCol`` layout 不能把 ``ptr_to``
  当作通用逃生口；若它们仍以直接 ``BufferLoad`` 形式出现，cleanup 会报错。
- 一旦 intrinsic 只接收 raw base address 和作者已经算好的 offset，layout
  mapping 就已被绕过。``buf.data + raw_offset`` 是常见写法，但不是唯一方式。

因此，“直接 PTX 没有消灭 layout”的准确含义是：它只会让 **那个低级
instruction call** 不再经过 tile-op layout inference；周围 buffers 的普通地址
访问仍可能需要 layout。反过来，如果所有相关地址、lane mapping 和 operand
顺序都由作者以 raw expressions 写完，那么 layout 对这条指令当然不会再提供
额外推导。

失败表示检查，而不是自动修复
------------------------------

假设 C 位于 TMEM，却只给它一个普通 row-major ``m`` layout，然后强制
``dispatch="tcgen05"``。该 layout 无法证明逻辑 C region 与 ``TLane/TCol``
datapath 相容，tcgen05 implementation 会拒绝它；如果没有其他候选，编译器
报告 dispatch failure。

这类错误说明了 TIRx 的责任边界：dispatcher 会从 **已经声明的** layout 推导
底层参数并验证约束，但不会搜索一个新 layout，再悄悄重写 kernel 的 producer、
consumer 和 allocation 使它们全部匹配。

检查 dispatch 与 layout lowering
--------------------------------

调试时可以在 cleanup 删除 layout metadata 前，单独查看 dispatch 结果：

.. code-block:: python

    import tvm
    from tvm.tirx import transform as TT

    target = tvm.target.Target("cuda -arch=sm_100a").with_host("llvm")
    mod = tvm.IRModule({"main": kernel})
    bound = TT.BindTarget(target)(mod)

    print("=== authored TIRx ===")
    print(bound.script())

    print("=== after TilePrimitiveDispatch ===")
    dispatched = TT.TilePrimitiveDispatch()(bound)
    print(dispatched.script())

    print("=== after LowerTIRx ===")
    lowered = TT.LowerTIRx()(bound)
    print(lowered.script())

``TilePrimitiveDispatch`` 运行时必须能解析出 target；显式 ``BindTarget`` 最清晰，
也可以依赖已有的 PrimFunc target attribute 或 current target context。这里指定
``sm_100a`` 是因为示例使用 Blackwell ``tcgen05``；生成和运行最终代码还要求相应
CUDA toolkit 与硬件支持。

``LowerTIRx`` 也提供一个调试开关，在 dispatch 与 cleanup 之间打印 IR：

.. code-block:: bash

    TVM_PRINT_AFTER_TIRX_DISPATCH_OPS=1 python your_kernel.py

检查 ``Tx.gemm_async`` 时，建议依次确认：

1. 原始 C/A/B regions 和 layouts；
2. dispatch 选择了哪个 variant；
3. A/B descriptors 的 major、swizzle、``ldo/sdo`` 和 slice offsets；
4. C datapath 与 TMEM offsets；
5. 生成的 MMA iteration 数量与 ``enable_input_d``；
6. cleanup 后还剩哪些直接 physical accesses。

这样可以把问题定位到 kernel schedule、layout contract、operator dispatcher
或后端 codegen，而不是只比较 TIRx 源码和最终 assembly。

核心源码导航
------------

- `dispatcher.py`_：variant registry、priority、predicate 与失败报告；
- `tile_primitive_dispatch.cc`_：scope/launch context、body replacement 与
  callbacks；
- `lower_tirx_cleanup.cc`_：``LayoutApplier``、``BufferOffsetRemover`` 与
  physical-offset materialization；
- `tcgen05 gemm dispatcher`_：从 operand regions/layouts 推导 descriptors、
  TMEM address、instruction tiling 和 ``T.ptx.tcgen05.mma``。

.. _dispatcher.py: https://github.com/apache/tvm/blob/v0.26.0/python/tvm/tirx/operator/tile_primitive/dispatcher.py
.. _tile_primitive_dispatch.cc: https://github.com/apache/tvm/blob/v0.26.0/src/tirx/transform/tile_primitive_dispatch.cc
.. _lower_tirx_cleanup.cc: https://github.com/apache/tvm/blob/v0.26.0/src/tirx/transform/lower_tirx_cleanup.cc
.. _tcgen05 gemm dispatcher: https://github.com/apache/tvm/blob/v0.26.0/python/tvm/backend/cuda/tile_primitive/gemm_async/tcgen05.py
