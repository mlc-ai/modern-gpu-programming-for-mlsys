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

上一页 :ref:`chap_tirx_lowering_pipeline` 给出了完整的编译流水线。本页进入 ``LowerTIRx`` 内部，沿一条 Blackwell ``Tx.tile.gemm_async`` 说明 tile primitive 和 layout 怎样逐步变成 target intrinsic 与物理地址。

``Tx.tile.gemm_async(C, A, B)`` 记录一次逻辑 tile GEMM。它给出操作数、区域和配置，具体使用哪些硬件指令、每个操作数怎样编码，则由 ``LowerTIRx`` 结合 target 与 layout 确定。这里的 layout 是一张从 **逻辑坐标** 到 **物理坐标** 的映射：生产者用它摆放数据，消费者用它找到同一份数据。

``LowerTIRx`` 由两个顺序执行的 pass 组成：

.. code-block:: text

    TilePrimitiveCall + Buffer layouts + target
                         │
                         ▼
              TilePrimitiveDispatch
       选择实现，生成 Tx.ptx.* 与地址表达式
                         │
                         ▼
                LowerTIRxCleanup
       物化剩余普通访问，清除 layout metadata
                         │
                         ▼
    包含 Tx.ptx.* 与物理 Buffer 访问的 PrimFunc

前一个 pass 读取算子级语义，后一个 pass 处理仍留在普通 ``BufferLoad``、``BufferStore`` 和指针访问中的 layout。同一个 buffer 的 layout 可以依次服务于这两个阶段。

先读懂 Layout
-------------

一个逻辑 tile 可以落在普通内存、多个线程的局部存储或 TMEM 中。三类位置需要三种物理坐标：

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - 存储位置
     - Layout 映射
     - 物理坐标的含义
   * - GMEM / SMEM
     - ``(i, j) → m``
     - ``m`` 是一个线性存储位置，也可以包含 padding 或 swizzle
   * - 每线程局部存储
     - ``(i, j) → (thread axis, m)``
     - thread axis 指定持有元素的线程，``m`` 指定该线程内的局部位置
   * - Blackwell TMEM
     - ``(i, j) → (TLane, TCol)``
     - 两个坐标共同指定 TMEM 中的硬件位置

这些映射都由 ``TileLayout`` 表达。完整的 layout 代数、``S[...]`` 语法与组合规则见 :ref:`chap_tirx_layout_api`；这里关注它们在 lowering 中提供的信息。

普通内存：逻辑下标映射到地址
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

省略 ``layout=`` 时，buffer API 默认创建连续行优先（row-major）layout。例如 shape 为 ``(4, 8)`` 的 buffer 使用：

.. code-block:: text

    TileLayout(S[(4, 8)])

    B[i, j] → B_flat[i * 8 + j]

这个公式与普通二维连续数组一致。layout metadata 会在编译前半程保留，因此 dispatcher 仍可对它执行 slice、匹配与检查。显式 ``layout=None`` 会让该字段保持为空，普通访问随后沿 shape/stride 规则展开。

物理排列也可以带 padding 或 swizzle。例如：

.. code-block:: text

    TileLayout(S[(4, 8) : (16, 1)])

    B[i, j] → B_flat[i * 16 + j]

这里每一行占 16 个元素，逻辑 shape 仍是 ``4×8``。对应的 backing allocation 需要覆盖 ``layout.span() = 56`` 个元素。``ComposeLayout`` 还可以把 XOR、shift 和 mask 组成 shared-memory swizzle。

每线程局部存储：逻辑下标映射到持有者和局部位置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下面的 distributed layout 把第 ``m`` 行交给 workgroup 中的第 ``m`` 个线程，并把 ``n`` 作为该线程的局部位置：

.. code-block:: python

    Dreg_wg = Dreg.view(
        128,
        N,
        layout=TileLayout(S[(128, N) : (1@tid_in_wg, 1)]),
    )

.. code-block:: text

    Dreg_wg[m, n] → { tid_in_wg: m, local slot: n }

``tid_in_wg`` 表达元素归属，默认 memory axis 记录该线程内的局部位置。CUDA toolchain 随后为这些局部值分配具体物理寄存器。

TMEM：逻辑下标映射到两个硬件坐标
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Blackwell TMEM 使用 ``TLane`` 和 ``TCol``：

.. code-block:: text

    TileLayout(S[(128, N) : (1@TLane, 1@TCol)])

    C[m, n] → { TLane: m, TCol: n }

``TLane`` 表示 TMEM lane row，``TCol`` 表示 TMEM column。二者共同组成 tcgen05 指令使用的 TMEM 地址。这里的 ``TLane`` 属于存储坐标；CUDA ``lane_id`` 表示当前执行线程，两者含义不同。

Layout 从哪里来
~~~~~~~~~~~~~~~~

TIRx kernel 中的 layout 通常来自三处：

1. **默认构造。** 省略 ``layout=`` 或写 ``layout="default"`` 时，parser 根据 shape 构造连续行优先 layout。
2. **Helper 构造。** ``mma_shared_layout``、``tmem_datapath_layout`` 和 ``tcgen05_atom_layout`` 等 helper 根据 dtype、shape 与 mode 生成已知的硬件 layout。
3. **显式声明。** 作者或上层生成器直接使用 ``TileLayout``、``ComposeLayout`` 等接口写出目标映射。

这里的自动化包含默认 layout 的构造，以及 lowering 根据既有 layout 推导指令参数。tile size、swizzle mode、线程分工和性能 layout 的选择通常由 kernel 作者、上层生成器或搜索系统完成。``LowerTIRx`` 接收这些已经确定的映射，再完成局部推导与合法性检查。

沿一条 ``Tx.tile.gemm_async`` 看 lowering
-------------------------------------------

下面抽取 :ref:`chap_tirx_primer` 中单-tile GEMM 的核心语句。SMEM/TMEM allocation、barrier 和等待代码在这里省略：

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

    C = Tx.decl_buffer(
        (128, 512),
        "float32",
        scope="tmem",
        allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]),
    )

    if warp_id == 0:
        if Tx.ptx.elect_sync():
            Tx.tile.gemm_async(
                C[:, :BLK_N],
                Asmem[:, :],
                Bsmem[:, :],
                accum=False,
                dispatch="tcgen05",
                cta_group=1,
            )

这一条调用向 dispatcher 提供了四组信息：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 输入
     - Lowering 中的用途
   * - C/A/B 的 ``BufferRegion``
     - 确定本次调用的逻辑 ``M``、``N``、``K`` 范围及各 region 的起点
   * - A/B 的 SMEM layouts
     - 确定 shared-memory 排列、swizzle、matrix descriptor 参数和每个指令 tile 的偏移
   * - C 的 TMEM layout
     - 验证 accumulator datapath，并取得 region 的 ``TLane`` 与 ``TCol`` 偏移
   * - dtype、``cta_group`` 与 ``dispatch``
     - 约束候选实现、指令 shape 和 tcgen05 variant

选择实现
~~~~~~~~

Python 前端把 ``Tx.tile.gemm_async`` 保存为 ``TilePrimitiveCall``。候选实现以 ``(operator, target kind)`` 为键注册，并带有 variant 名称、优先级和适用条件。

显式 ``dispatch="tcgen05"`` 会筛选出对应 variant；省略 ``dispatch=`` 时，dispatcher 按优先级检查各候选的适用条件。选中的实现返回一个 ``PrimFunc``，其函数体替换原来的 ``TilePrimitiveCall``。整个过程发生在编译期，依据 target、操作数 region、layout、dtype 和 config 做规则匹配。

其他 copy、reduce 和同步类 tile primitive 也经过同一套选择流程。每个实现自行定义要读取哪些 layout 坐标、接受哪些 shape，以及最终生成哪些 target intrinsics。

切出本次调用的 Layout
~~~~~~~~~~~~~~~~~~~~~~

调用参数是 ``C[:, :BLK_N]``、``Asmem[:, :]`` 和 ``Bsmem[:, :]`` 这样的 ``BufferRegion``。Dispatcher 先对各 buffer layout 执行 ``slice``，把 region 起点折入 layout offset，再通过 ``canonicalize`` 得到便于匹配的等价形式。

tcgen05 实现随后检查：

- C 的 scope、dtype、shape 与 ``TLane/TCol`` 映射；
- A/B 的 scope、dtype、alignment 与 shared-memory atom；
- A/B 的 major mode 和 swizzle 是否落在该指令支持的组合中；
- ``M/N/K`` 与 ``cta_group`` 是否可以分解成合法的 tcgen05 指令 shape。

这些检查把 layout 当作调用约束。layout 与候选实现的硬件约定一致后，lowering 才继续生成 descriptor 和指令。

A/B Layout 推导 matrix descriptor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

tcgen05 从 SMEM 读取 A/B 时，硬件通过 matrix descriptor 解释 shared-memory 地址。Dispatcher 将 sliced layout 与受支持的 K-major 或 MN-major swizzle atom 匹配，并得到：

.. code-block:: text

    major mode
    swizzle mode
    leading-dimension offset (ldo)
    stride-dimension offset (sdo)
    当前指令 tile 相对 region 原点的 16-byte offset

``ldo``、``sdo`` 和 swizzle 等字段可在 dispatch 时确定；SMEM base address 属于运行时值。生成的 TIR 会调用 ``Tx.ptx.tcgen05.encode_matrix_descriptor``，把运行时地址与这些字段编码到一个局部 ``uint64`` descriptor 中。后续各个 MMA iteration 再加入以 16 bytes 为单位的 tile offset。

这也解释了 producer 与 consumer 为何要共享 layout。前面的 copy 按 A/B layout 把元素写入 swizzled SMEM，GEMM dispatcher 从同一份 layout 生成 descriptor，tcgen05 因而按照相同的字节排列读回元素。

C Layout 推导 TMEM 地址
~~~~~~~~~~~~~~~~~~~~~~~

示例中 C region 的映射为：

.. code-block:: text

    C[m, n] → { TLane: m, TCol: n }

对 layout 做 slice 后，dispatcher 分别取得 ``TLane`` offset 和 ``TCol`` offset。它们与 ``allocated_addr`` 一起传给 ``Tx.cuda.get_tmem_addr``，形成 tcgen05 使用的 TMEM 目标地址：

.. code-block:: text

    get_tmem_addr(allocated_addr, TLane offset, TCol offset)

当 C region 从非零行或非零列开始时，对应偏移会进入这两个坐标。Dispatcher 同时验证 sliced layout 与所选 accumulator datapath 一致。

Shape 与 dtype 决定指令分解
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

示例表示一次 fp16 ``128×128×64`` GEMM。tcgen05 的这个 dense 路径每个 K step 处理 16 个 fp16 元素，因此 K 方向被分成 4 次 MMA。省略具体 TIR 语法后，dispatch 结果的结构如下：

.. code-block:: text

    # 结构示意；尖括号中的内容代表生成的 TIR expression
    desc_a = encode_matrix_descriptor(
        <Asmem runtime base>, <A ldo>, <A sdo>, <A swizzle>
    )
    desc_b = encode_matrix_descriptor(
        <Bsmem runtime base>, <B ldo>, <B sdo>, <B swizzle>
    )
    desc_i = Tx.uint32(<dispatch 时生成的 dense instruction descriptor>)

    for ki in Tx.unroll(4):
        Tx.ptx.tcgen05.mma(
            get_tmem_addr(<C base>, <TLane offset>, <TCol offset>),
            add_16B_offset(desc_a, <A offset for ki>),
            add_16B_offset(desc_b, <B offset for ki>),
            desc_i,
            enable_input_d=(ki != 0),
            ...
        )

这里的 ``desc_a`` 和 ``desc_b`` 需要运行时 SMEM 地址，所以 descriptor encoding 保留在生成的 TIR 中。dense ``desc_i`` 只依赖指令 shape、dtype、major mode 与 ``cta_group``，dispatcher 会把它折叠为编译期 ``uint32`` 常量。

``Tx.unroll(4)`` 会由后续 ``UnrollLoop`` 展开，``StmtSimplify`` 再化简每次迭代中的常量表达式。指令数量与操作数分块来自 tile primitive 实现，循环展开和局部化简由流水线后段完成。

Readback 使用另一组 Layout
~~~~~~~~~~~~~~~~~~~~~~~~~~

GEMM 将结果写入 TMEM 后，readback 是另一条独立的 tile primitive：

.. code-block:: python

    Dreg = Tx.alloc_local((BLK_N,), "float32")
    Dreg_wg = Dreg.view(
        128,
        BLK_N,
        layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]),
    )
    Tx.tile.wg.copy_async(Dreg_wg[:, :], C[:, :BLK_N])

它同时读取 C 的 ``TLane/TCol`` layout 和 ``Dreg_wg`` 的 distributed layout，选择相应的 ``tcgen05.ld`` form，并把 TMEM 元素送到指定线程的局部位置。整个数据流为：

.. code-block:: text

    GMEM
      │  copy primitive 按 A/B layout 写入
      ▼
    swizzled SMEM
      │  gemm_async 按同一 layout 生成 descriptor 并读取
      ▼
    TMEM (TLane, TCol)
      │  wg.copy_async 同时解释 TMEM 与 distributed layouts
      ▼
    per-thread local slots (tid_in_wg, m)

Layout 在这里充当 producer 与 consumer 之间的物理映射契约；tile primitive 负责把这份契约翻译成具体硬件操作。

``LowerTIRxCleanup`` 处理剩余 Layout
------------------------------------

``TilePrimitiveDispatch`` 已将全部 ``TilePrimitiveCall`` 展开。普通 ``BufferLoad``、``BufferStore`` 和指针访问仍可能引用带 layout 的 buffer，``LowerTIRxCleanup`` 随后处理这些访问：

.. code-block:: text

    创建或附着 Layout
           │
           ▼
    Buffer view / region slice
           │
           ▼
    TilePrimitiveDispatch 读取并展开算子
           │
           ▼
    LowerTIRxCleanup 映射剩余普通访问
           │
           ▼
    重建物理 Buffer，清除 layout metadata
           │
           ▼
    后续 TIR passes 与 target codegen

对普通 memory layout，``LayoutApplier`` 的核心过程可以写成：

.. code-block:: text

    logical indices
          │
          ▼
    layout.canonicalize().apply(indices, shape)
          │
          ▼
    一个 symbolic memory coordinate
          │
          ▼
    physical buffer access

这里的 symbolic coordinate 可以包含运行时循环变量、``threadIdx`` 或函数参数。后续 ``FlattenBuffer`` 会把 buffer 的 ``elem_offset`` 合入线性 index，形成最终地址语义。``BufferOffsetRemover`` 则消除 ``tirx.buffer_offset(BufferLoad)`` wrapper，使其中的 offset 与物理访问保持一致。

普通指针访问需要 layout 最终产生一个 memory coordinate，通常命名为 ``m``。带有 ``laneid``、``tid_in_wg``、``TLane`` 或 ``TCol`` 的映射表达额外的硬件坐标；它们应先由理解这些坐标的 tile primitive 消费，或通过 ``.view()`` 与 ``.local()`` 转成当前线程的存储视图。若这类坐标以普通 ``BufferLoad`` 留到 cleanup，编译器会给出诊断。

``LowerTIRx`` 完成后，所有 ``TilePrimitiveCall`` 都已展开，layout metadata 已被 dispatcher 读取或由 cleanup 物化，``Tx.ptx.*`` 等 target intrinsics 则继续进入后续 TIR passes 与 CUDA codegen。

高层 Tile Primitive 与直接 PTX
-------------------------------

直接写 ``Tx.ptx.tcgen05.mma`` 时，这个 target intrinsic ``Call`` 从一开始就在输入 IR 中。``TilePrimitiveDispatch`` 的算子分派针对 ``TilePrimitiveCall``，作者写下的 PTX intrinsic 会原样保留。高层写法生成的 PTX intrinsic 与直接写入的 PTX intrinsic，随后都经过 cleanup 和流水线后段：

.. code-block:: text

    Tx.tile.gemm_async
            │
            ▼
    TilePrimitiveDispatch
            │
            ▼
    Tx.ptx.tcgen05.mma
            │
            ▼
    cleanup → 后续 TIR passes → CUDA codegen

    直接 Tx.ptx.tcgen05.mma
            │
            ▼
    TilePrimitiveDispatch 保留该 intrinsic
            │
            ▼
    cleanup → 后续 TIR passes → CUDA codegen

两种写法的差别集中在 PTX intrinsic 生成之前：

.. list-table::
   :header-rows: 1
   :widths: 27 37 36

   * - 编程责任
     - ``Tx.tile.gemm_async``
     - 直接 ``Tx.ptx.tcgen05.mma``
   * - 实现选择
     - Dispatcher 根据 target 选择或验证 ``dispatch=``
     - 作者已经选定具体 intrinsic
   * - 逻辑 tile 分解
     - 根据 shape、dtype 和 config 生成指令 tiles
     - 作者逐条写出指令
   * - 操作数 layout 检查
     - Dispatcher 执行算子专用的匹配与验证
     - 作者按照 PTX 约定组织操作数
   * - Descriptor 与 TMEM 地址
     - 从 layout、region 和 config 推导
     - 作者调用低级编码与地址接口
   * - Lane 与局部值顺序
     - Distributed layout 表达归属和局部位置
     - 作者用 lane 公式、局部数组和 intrinsic 参数表达

直接 PTX 仍可与 layout 共存，具体取决于地址怎样构造：

- PTX 参数来自 layout-backed memory buffer 的 ``ptr_to(logical_indices)`` 时，cleanup 可以把单一 memory-axis layout 映射成物理地址。
- 使用 ``Tx.ptr_byte_offset`` 或 ``Tx.handle_add_byte_offset`` 提供 byte offset 时，地址表达式直接携带作者选定的物理映射。
- TMEM 地址可以通过 ``Tx.cuda.get_tmem_addr(base, tlane, tcol)`` 明确给出，distributed 数据的 lane 归属与局部顺序也由作者明确组织。

因此，直接 PTX 省去了 tile primitive 提供的算子级推导与检查；layout 仍可负责周围 buffer 的地址映射。作者显式写出的 descriptor、TMEM 坐标、lane 公式和局部顺序承担同一组物理约定。ptxas 等工具继续负责 ``%r17`` 一类物理寄存器的最终分配。

Layout 约束怎样产生诊断
-----------------------

假设 C 位于 TMEM，却声明为只有 ``m`` 轴的连续行优先 layout，同时强制 ``dispatch="tcgen05"``。tcgen05 variant 期望 C region 映射到受支持的 ``TLane/TCol`` datapath；两组物理坐标发生冲突，该候选实现会被拒绝。

这类诊断来自 layout 的契约作用。Dispatcher 从已声明的 layout 推导底层参数并验证硬件约束，kernel 作者或上层生成器负责让 producer、consumer 与 allocation 使用一致的映射。

检查 Dispatch 与 Layout Lowering
--------------------------------

需要定位问题时，可以分别打印 authored TIRx、dispatch 结果和完整 ``LowerTIRx`` 结果：

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

示例使用 Blackwell ``tcgen05``，因此 target 设为 ``sm_100a``。检查输出时，可以沿同一条调用依次确认：

1. 原始 C/A/B regions 与 layouts；
2. dispatch 选中的 variant；
3. A/B descriptor 的 major、swizzle、``ldo/sdo`` 和 16-byte offsets；
4. C 的 ``TLane/TCol`` offsets 与 TMEM 地址；
5. MMA iteration 数量及 ``enable_input_d``；
6. cleanup 后生成的普通物理访问。

这样可以把问题定位到输入 schedule、layout 契约、算子 dispatcher 或后端 codegen 的具体阶段。

核心源码导航
------------

- `dispatcher.py`_：variant registry、priority、predicate 与失败报告；
- `tile_primitive_dispatch.cc`_：dispatch context、调用替换与 execution scope lowering；
- `lower_tirx_cleanup.cc`_：``LayoutApplier``、``BufferOffsetRemover`` 与物理地址物化；
- `tcgen05 gemm dispatcher`_：从操作数 regions/layouts 推导 descriptors、TMEM 地址、指令分块和 ``Tx.ptx.tcgen05.mma``。

.. _dispatcher.py: https://github.com/apache/tvm/blob/v0.26.0/python/tvm/tirx/operator/tile_primitive/dispatcher.py
.. _tile_primitive_dispatch.cc: https://github.com/apache/tvm/blob/v0.26.0/src/tirx/transform/tile_primitive_dispatch.cc
.. _lower_tirx_cleanup.cc: https://github.com/apache/tvm/blob/v0.26.0/src/tirx/transform/lower_tirx_cleanup.cc
.. _tcgen05 gemm dispatcher: https://github.com/apache/tvm/blob/v0.26.0/python/tvm/backend/cuda/tile_primitive/gemm_async/tcgen05.py
