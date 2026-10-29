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

Tile 指令与 Layout Lowering
================================================

:ref:`chap_tirx_lowering_pipeline` 给出完整 pipeline。本页分别观察 ``TileDispatch``
怎样展开显式 tile 指令，以及 ``LowerTensorLayouts`` 怎样降低其余 Tensor 访问。
这两项转换读取同一个 layout 约定，但职责不同。

.. code-block:: text

    Function + typed tensors + explicit PTX tile calls
              │ target/backend binding
              ▼
    TileDispatch
      ├─ 检查所选 opcode、scope、canonical operand views
      └─ 展开 group/repeat axes，推导 descriptor 与指令参数
              ▼
    LowerTensorLayouts
      └─ 将剩余 Tensor 访问与 view 转成底层存储访问
              ▼
    StmtSimplify → FlattenTensor → index/loop lowering → finalization

先读懂物理坐标
--------------

GMEM/SMEM 使用线性 memory axis ``m``；分布式局部存储还需要 thread ownership axes；
TMEM 使用 ``TLane`` 与 ``TCol``。这些坐标回答“元素在哪里”，不等同于发出当前指令的
CUDA thread coordinate。

.. code-block:: python

    from tvm_tirx.tirx.layout import TileLayout, S
    from tvm_tirx.backend.cuda.axis import tid_in_wg, TLane, TCol

    registers = TileLayout(S[(128, 4) : (1@tid_in_wg, 1)])
    accumulator = TileLayout(S[(128, 128) : (1@TLane, 1@TCol)])
    print(registers.apply(7, 3))
    print(accumulator.apply(7, 3))

第一个映射把第 7 行分配给 warpgroup thread 7，第 3 列位于该 thread 的局部 slot 3。
第二个映射得到 TMEM row 7、column 3。对 FP16 或 FP8 的 ``TCol``，多个元素会打包
进同一个 32-bit 物理 TMEM column。Layout 的 ``apply`` 只计算 base coordinate；
replica iters 仍由使用该 layout 的指令处理。

Layout 的来源
--------------------

作者可以直接指定 ``TileLayout`` 或 ``ComposeLayout``，也可以使用 CUDA backend 的
``mma_shared_layout``、``tmem_datapath_layout`` 等 helper。``T.Tensor`` 省略
``layout`` 或传入 ``None`` 时保留未指定状态；``layout="default"`` 才显式创建连续
row-major 映射。连续 GMEM view 可以在需要时推导有效 layout；分布式局部存储与
TMEM 则需要满足所选指令的显式 layout。

编译器检查这些已经选定的映射。Tile 大小、swizzle mode、协作范围和 PTX opcode
仍由作者或上层生成器决定。改变 layout 只改变元素位置的描述，不会搬运数据，
也不会让不兼容的 instruction 自动切换到其他实现。

一个可执行的 tile 展开例子
--------------------------------

下面用每线程四个寄存器演示 repeat axes。一个 ``tile.apply`` 把所选 ``mov.b32``
重复应用到四个局部元素，再通过普通 Tensor stores 写回输出。Operand view 的
最后一个维度显式表示一条 ``mov.b32`` 的 register group（大小为1），前面的维度
表示四次重复；局部 view 同时提供显式的线性 layout。

.. code-block:: python

    import tvm_tirx
    from tvm_tirx.script import tirx as T
    from tvm_tirx.tirx.layout import TileLayout, S
    from tvm_ffi import structural_walk

    @T.function
    def layout_fill(B: T.Tensor((128, 4), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=128))
        tx = T.cuda.thread_idx("x")
        values = T.alloc_local((4,), "float32")
        T.cuda.tile.apply(T.ptx["mov.b32"], values.view(4, 1, layout=TileLayout(S[(4, 1) : (1, 1)]))[:, :], T.float32(3), scope="thread")
        for j in T.unroll(4):
            B[tx, j] = values[j]

先单独运行 ``TileDispatch``，观察高层调用消失：

.. code-block:: python

    from tvm_tirx.tirx import transform as TT

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"})
    mod = tvm_tirx.IRModule({"main": layout_fill})
    mod = TT.BindTarget(target)(mod)
    dispatched = TT.TileDispatch()(mod)
    calls = []
    structural_walk(dispatched, (tvm_tirx.ir.Call, calls.append))
    assert all(call.op.name != "tirx.cuda.tile.apply"
               for call in calls if isinstance(call.op, tvm_tirx.ir.Op))
    print(dispatched.script())

``TileDispatch`` 已将 tile 请求展开为指定的底层调用，随后 ``LowerTensorLayouts``
处理剩余的局部 Tensor load/store 和 view。``FlattenTensor`` 再把普通多维存储访问
展平。底层 IR 仍需经过类型合法化、host/device 拆分、ABI 与代码生成。

沿 Blackwell MMA 看同一过程
------------------------------------------

入门 GEMM 使用 ``tcgen05.mma.cta_group::1.kind::f16``，并显式提供
``instruction_shape=(128,128,16)``。作者将 D、A、B 分别整理成下面的 canonical axes：

.. code-block:: text

    D: (repeat-M, repeat-N, M-atom, N-atom)
    A: (repeat-M, repeat-K, M-atom, K-atom)
    B: (repeat-N, repeat-K, N-atom, K-atom)

对于 ``128×128×64`` tile，K-atom 为16，因此展开四次 K steps。Accumulator 的首次
``accum=False`` 初始化结果，后续 steps 累加。Shared-memory descriptor 由布局与地址
约定导出；Tensor alignment、swizzle 和矩阵 major order 必须与指令相符。

每个 ring-buffer stage 使用二维 ``decl_tensor`` alias 传入固定 shape 和真实 alignment，
可以避免复杂动态 stage 切片的对齐推导。Alias 的底层 pointer 必须实际满足该对齐，
相关完整代码见 :ref:`chap_gemm_async`。

TMEM load/store 的 canonical axes 又不同。例如入门 TMEM readback 将寄存器和 TMEM
views 都变成 ``(4,32,128)``，再以 ``scope="warpgroup"`` 应用
``tcgen05.ld.sync.aligned.32x32b.x128.b32``。不能仅把通用 copy 改名为 tile.apply：
opcode、scope 与 operands 的形状必须作为同一个硬件约定一起匹配。

直接 PTX 与 tile 指令的边界
--------------------------------------

直接 ``T.ptx`` 调用指定一次指令及其操作数；``tile.apply`` 增加 canonical view
检查和 group/repeat 展开。前者没有 tile pattern 这层转换，两者都继续经过普通 IR
与代码生成 passes。Direct PTX 不会自动提供 barrier、wait 或 memory fence。

布局正确也不代表异步操作已完成。MMA completion、TMA byte count、TMEM loads/stores、
跨线程 handoff 与 SMEM proxy publication 都需要相应的同步协议。API 迁移时应同时检查
生成的 CUDA 和数值结果，使用本书 :ref:`chap_gemm_advanced` 的完整 barriers 作为例子。

源码与进一步检查
----------------

- `patterns <https://github.com/apache/tvm-tirx/blob/a821c08826b4721d47122ee9b1dced7f3b88e738/python/tvm_tirx/backend/cuda/tile/patterns/__init__.py>`_：tile.apply 与显式指令 pattern dispatch。
- `TileDispatch <https://github.com/apache/tvm-tirx/blob/a821c08826b4721d47122ee9b1dced7f3b88e738/src/tirx/tile/lower.cc>`_：tile 调用遍历。
- `LowerTensorLayouts <https://github.com/apache/tvm-tirx/blob/a821c08826b4721d47122ee9b1dced7f3b88e738/src/tirx/transform/lower_tensor_layouts.cc>`_：普通 Tensor layout lowering。
- `compilation_pipeline.py <https://github.com/apache/tvm-tirx/blob/a821c08826b4721d47122ee9b1dced7f3b88e738/python/tvm_tirx/tirx/compilation_pipeline.py>`_：pass 顺序与 finalization。
