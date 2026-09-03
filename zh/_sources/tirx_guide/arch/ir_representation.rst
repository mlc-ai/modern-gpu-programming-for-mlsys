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

.. _chap_tirx_ir_representation:

TIRx IR 的组织方式
==================

:ref:`chap_tirx_primer` 已经展示了 scope、layout 和 dispatch 在 kernel 中的写法。本页选取其中一条 tile primitive，展开 parser 生成的对象，观察每类信息落在哪个节点、节点之间如何引用，以及这些节点为什么需要分开表示。

``@Tx.prim_func`` 解析后返回 ``tvm.tirx.PrimFunc``。它以 ``Stmt`` 树保存程序结构，以 ``PrimExpr`` 保存索引和标量计算；``Buffer``、``Layout`` 与 ``ExecScope`` 等对象由树中的节点引用。进入模块级 pass 前，一个或多个 ``PrimFunc`` 会作为全局函数放入 ``IRModule``。

通用基础设施与 TIRx 方言
-------------------------

TIRx 使用 TVM 的 ``IRModule``、``BaseFunc``、表达式基类、类型系统、``Op`` 注册表和 pass 管理机制。函数和语句结构采用 TIRx 方言节点，``IntImm``、``FloatImm`` 与 ``Range`` 等叶子对象来自通用 IR：

.. code-block:: text

    TVM 通用基础设施
    ├─ IRModule
    ├─ BaseFunc
    ├─ ir.PrimExpr / IntImm / FloatImm / Range
    ├─ Op / Type / Target
    └─ PassContext 与 pass manager
             │
             ▼
    TIRx 方言
    ├─ tirx.PrimFunc
    ├─ tirx.Stmt
    ├─ tirx.Var / tirx.Call / tirx.BufferLoad
    ├─ tirx.Buffer / tirx.BufferRegion
    ├─ tirx.ScopeIdDefStmt / tirx.ExecScope
    └─ tirx.TilePrimitiveCall

模块基础设施通过 ``PrimFunc`` 的运行时类型区分 TIRx 函数，函数体则由 TIRx 的 visitor 或 mutator 遍历。TIRx 由此接入 TVM 的通用模块，同时为 tile、layout 和执行层级保留专门的节点类型。

沿一条 tile primitive 看对象关系
---------------------------------

以入门章计算阶段的 ``gemm_async`` tile primitive 和紧随其后的 ``tcgen05.commit`` intrinsic 为观察点。第一条调用在 parser 之后成为 ``TilePrimitiveCall``；第二条已经确定为 PTX 操作，保存为 ``Evaluate(Call)``。省略外围条件和其他语句后，局部对象关系如下：

.. code-block:: text

    tirx.PrimFunc.body
    └─ ...
       ├─ TilePrimitiveCall
       │  ├─ op = Op("tirx.tile.gemm_async")
       │  ├─ args[0:3]
       │  │  ├─ BufferRegion(tmem)
       │  │  ├─ BufferRegion(Asmem)
       │  │  └─ BufferRegion(Bsmem)
       │  ├─ args[3:6] = transpose_A, transpose_B, accum
       │  ├─ config = {"cta_group": 1}
       │  ├─ dispatch = "tcgen05"
       │  └─ scope = ExecScope("thread")
       └─ Evaluate
          └─ Call(op=tirx.ptx.*)

这张局部对象图包含两种关系。``Stmt`` 节点之间的包含关系确定执行顺序和控制流；同一个 ``Buffer`` 则可以被声明节点、``BufferRegion`` 和访问节点共同引用。图中的箭头表示共享同一个 ``ObjectRef``，pass 根据对象身份连接定义与使用，再沿这些引用读取 layout 和 scope 等信息。

``PrimFunc`` 给出函数边界
-------------------------

``tirx.PrimFunc`` 的核心字段是 ``params``、``buffer_map``、``body`` 和 ``ret_type``，函数级 ``attrs`` 来自 ``BaseFunc``。其中 ``params`` 保存 handle 与标量参数，``body`` 保存一条 ``Stmt``；多条顶层语句由 ``SeqStmt`` 组织。

Buffer 参数在函数边界上分成两部分。参数注解 ``A: Tx.Buffer((M, K), "float16")`` 会在 ``params`` 中产生一个 handle 变量，同时由 ``buffer_map`` 将这个 handle 映射到包含 shape、dtype 和存储域的 ``Buffer`` 对象。分析 pass 可以直接从 ``buffer_map`` 读取参数约束，无需从函数体中的声明重新恢复。

``Tx.device_entry()`` 在 ``body`` 中形成 ``AttrStmt(attr_key="tirx.device_entry", value=True, body=...)``。它以一段语句区域为边界，因此保存在函数体内。

声明节点为何平铺在 ``SeqStmt`` 中
----------------------------------

在各自所在的词法作用域内，变量绑定、buffer 声明和执行 ID 声明采用平铺形式，直接成为 ``SeqStmt`` 的成员：

.. code-block:: text

    SeqStmt
    ├─ ScopeIdDefStmt(tx = cta→thread, extent=128)
    ├─ AllocBuffer(S)
    ├─ Bind(i, ...)
    ├─ TilePrimitiveCall(... S ...)
    └─ BufferStore(... i ...)

``Bind``、``AllocBuffer``、``DeclBuffer`` 和 ``ScopeIdDefStmt`` 都是独立的 ``Stmt``。这些节点分别保存绑定、分配或声明信息，外围 ``SeqStmt`` 负责承载后续程序。声明产生的变量或 buffer 对该 ``SeqStmt`` 中的后续语句可见，pass 按顺序遍历时便能维护当前可用的变量、buffer 和执行上下文。

控制流本身仍按词法作用域嵌套，索引、条件和标量计算以 ``PrimExpr`` 嵌入各个语句。这样的主干兼顾了控制流结构与声明的源码顺序，普通循环、表达式和 buffer 访问也能继续使用统一的 visitor 与 mutator。

分配与视图共享底层存储
------------------------

``AllocBuffer`` 用一个 ``Buffer`` 描述新分配的存储，``DeclBuffer`` 则可以在已有数据指针上声明新的视图。入门 GEMM 中的 ``Dreg = Tx.alloc_local(...)`` 和 ``Dreg_wg = Dreg.view(...)`` 在 IR 中形成下面的关系：

.. code-block:: text

    AllocBuffer
    └─ buffer = Dreg
       ├─ data ──────────────┐
       ├─ shape = [BLK_N]    │  同一个 data Var
       └─ layout = ...       │
                             │
    DeclBuffer               │
    └─ buffer = Dreg_wg      │
       ├─ data ──────────────┘
       ├─ shape = [128, BLK_N]
       └─ layout = distributed layout

``AllocBuffer(Dreg)`` 记录每个 thread 的局部存储。``Dreg_wg`` 是另一个 ``Buffer`` 对象，由 ``DeclBuffer`` 加入语句序列；它保存自己的 shape 和 layout，同时与 ``Dreg`` 指向同一个 data ``Var``，dtype、strides 与 ``elem_offset`` 等属性从 ``Dreg`` 延续。入门示例中的 ``(128, BLK_N)`` 因而描述 warpgroup 使用的逻辑坐标系；在 parser 生成的 IR 中，实际分配节点仍是形状为 ``(BLK_N,)`` 的 ``AllocBuffer(Dreg)``。

``BufferRegion`` 连接 tile 与数据布局
-------------------------------------

``Buffer`` 是一份结构化数据视图，保存数据指针、dtype、shape、strides 和 ``elem_offset``，数据指针的类型携带 storage scope。TIRx 还在其中保存可选的 ``layout``，以及 TMEM 等专用存储使用的 ``allocated_addr``。参数的 ``buffer_map``、局部分配、数据访问和 tile 区域可以共同引用同一个 ``Buffer``。

单点访问形成 ``BufferLoad`` 或 ``BufferStore``，切片形成 ``BufferRegion``。``BufferRegion`` 只保存原 ``Buffer`` 和每一维的 ``Range(min, extent)``：

.. code-block:: text

    TilePrimitiveCall.args[i]
                │
                ▼
          BufferRegion
          ├─ region = [Range(...), Range(...)]
          └─ buffer ────────────────┐
                                    ▼
                                 Buffer
                                 ├─ dtype / shape / storage scope
                                 ├─ layout
                                 └─ allocated_addr

一条 tile primitive 因而可以同时引用本次操作覆盖的逻辑范围，以及底层数据视图中的 dtype、存储域、layout 和硬件地址。Layout 保存在 ``Buffer`` 上，因此各个 ``BufferRegion`` 都能访问同一份映射。

``layout`` 字段保存一个可选的 ``Layout`` 对象。``Layout`` 是统一的映射接口，当前有三种具体节点：

.. code-block:: text

    Buffer.layout: Optional<Layout>
                    │
                    ▼
                  Layout
                  ├─ TileLayout
                  │  ├─ shard:   [Iter(extent, stride, Axis), ...]
                  │  ├─ replica: [Iter(extent, stride, Axis), ...]
                  │  └─ offset:  {Axis: PrimExpr}
                  ├─ SwizzleLayout
                  │  ├─ per_element
                  │  ├─ swizzle_len / atom_len
                  │  └─ swizzle_inner
                  └─ ComposeLayout
                     ├─ swizzle: SwizzleLayout
                     └─ tile_layout: TileLayout

``TileLayout`` 用一组 ``Iter`` 表示普通存储或线程映射，每个 ``Iter`` 保存 extent、stride 和物理 ``Axis``；``SwizzleLayout`` 保存 XOR swizzle 的参数；``ComposeLayout`` 同时持有一项 swizzle 和一项 tile mapping。三者通过 ``Layout`` 接口提供 ``Apply``、``Slice`` 和 ``Canonicalize`` 等操作。

入门 GEMM 的 ``128×64`` A/B shared-memory layout 在 canonicalize 后是 ``SwizzleLayout``，TMEM accumulator 使用 ``TileLayout``；带有额外外层 tile mapping 的 swizzle 会保留为 ``ComposeLayout``。Layout 中的 thread axis 表示元素归属于哪个执行成员，``TilePrimitiveCall.scope`` 则记录整项操作的协作层级。这些映射的具体含义与 lowering 过程见 :ref:`chap_tirx_tile_layout_lowering`。

这里需要区分两类 scope：``Buffer`` 数据指针上的 storage scope 描述数据位于 global、shared、local 或 TMEM；下一节的 execution scope 描述一项操作由哪一级线程集合执行。

执行层级保存在三个位置
------------------------

执行层级在 IR 中分成区域边界、ID 声明和逐调用协作粒度：

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - 保存位置
     - 对应的 Python 写法
     - 节点中的信息
   * - device-entry ``AttrStmt``
     - ``Tx.device_entry()``
     - device region 的范围
   * - ``ScopeIdDefStmt.def``
     - ``tx = Tx.thread_id([128])``
     - ``def_ids``、``extents``、``scope`` 和可选的 ``preferred_extents``
   * - ``TilePrimitiveCall.scope``
     - ``Tx.tile.cta.copy(...)``
     - ``ExecScope.kind``，也就是这一条调用的协作层级

``ScopeIdDefStmt`` 记录当前区域可用的执行坐标及其范围，``ExecScope`` 记录当前调用的协作层级。同一组执行 ID 可以服务于多条具有不同 ``ExecScope`` 的 tile primitive，因此两部分信息各自保存。

``TilePrimitiveCall`` 保留 dispatch 所需信息
---------------------------------------------

``TilePrimitiveCall`` 自身是一条 ``Stmt``，包含六个字段：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 保存的内容
   * - ``op``
     - tile operator 在 ``Op`` 注册表中的标识
   * - ``args``
     - ``Array<Any>``，可容纳 ``BufferRegion``、标量表达式和其他算子参数
   * - ``workspace``
     - 算子使用的预分配 buffer
   * - ``config``
     - ``cta_group`` 等算子选项
   * - ``dispatch``
     - 可选的显式实现名称
   * - ``scope``
     - 保存协作粒度的 ``ExecScope``

Tile primitive 可能读取或改写多个 ``BufferRegion``，其结果和副作用通过这些区域传递，因此作为一条语句占据明确的执行位置。各操作按照自己的参数约定识别输入与输出；``TilePrimitiveCall`` 构造时还会检查 ``op`` 是否注册为 tile primitive，TIRx 的 visitor 也为它提供独立的分派入口。

``BufferRegion`` 的运行时类型保持为独立的引用对象；进入 ``PrimExpr`` 上下文时，可转换的定长区域会成为 ``BufferLoad``。``TilePrimitiveCall.args`` 使用 ``Array<Any>``，可以原样保留每一维的 ``Range(min, extent)``、底层 ``Buffer`` 引用及其 layout。

``tirx.Call`` 是 ``PrimExpr``，参数类型为 ``Array<PrimExpr>``，主要字段为 ``op``、``args`` 和 ``attrs``。``Tx.ptx.*`` 与 ``Tx.cuda.*`` 已经给出目标相关操作，parser 会用这种节点保存它们。返回标量的 ``Tx.ptx.elect_sync()`` 可以直接参与表达式；以副作用为主的调用则位于 ``Evaluate(Call)`` 中。

从打印结果验证这些关系
------------------------

取得入门示例返回的 ``kernel`` 后，可以直接检查函数边界与语句树：

.. code-block:: python

    print(type(kernel))
    print(kernel.params)
    print(kernel.buffer_map)
    print(type(kernel.body))
    print(next(iter(kernel.buffer_map.values())).layout)

    print(kernel.script(
        syntax_sugar=False,
        extra_config={"tirx.prefix": "Tx"},
    ))

关闭 syntax sugar 后，printer 会展开参数 handle 与 buffer 绑定，并从 ``kernel.body`` 进入 visitor。默认 layout 在 script 输出中仍会省略，因此上面的 ``buffer.layout`` 可用于区分默认 layout 与空值。编写分析或变换 pass 时，可以使用 ``tvm.tirx.stmt_functor`` 中的 visitor 和 mutator。

``TilePrimitiveCall`` 将 tile 级语义保留为一条完整语句，``tirx.Call`` 则记录已经明确的调用及其返回类型。这些节点随后的转换顺序见 :ref:`chap_tirx_lowering_pipeline`，tile primitive 与 layout 的具体展开见 :ref:`chap_tirx_tile_layout_lowering`。

核心源码导航
------------

- `function.h`_：``tirx.PrimFunc`` 的字段；
- `buffer.h`_：``Buffer``、layout 和 ``allocated_addr``；
- `buffer.py`_：``Buffer.view`` 如何构造共享 data pointer 的新视图；
- `layout.h`_：``Layout`` 层级、``Iter`` 与 ``Axis``；
- `stmt.h`_：平铺声明、``BufferRegion`` 与 ``ScopeIdDefStmt``；
- `tirx_stmt.h`_：``TilePrimitiveCall`` 的六个字段；
- `expr.h`_：``tirx.Call``、``tirx.Var`` 与 ``tirx.BufferLoad`` 等具体表达式节点；
- `ir_expr.h`_：通用 ``PrimExpr`` 基类与常量节点；
- `exec_scope.h`_：``ExecScope``、``ScopeIdDef`` 与 ``ScopeBinding``；
- `transform.cc`_：TIRx ``PrimFuncPass`` 的模块遍历边界；

.. _function.h: https://github.com/apache/tvm/blob/v0.26.0/include/tvm/tirx/function.h
.. _buffer.h: https://github.com/apache/tvm/blob/v0.26.0/include/tvm/tirx/buffer.h
.. _buffer.py: https://github.com/apache/tvm/blob/v0.26.0/python/tvm/tirx/buffer.py
.. _layout.h: https://github.com/apache/tvm/blob/v0.26.0/include/tvm/tirx/layout.h
.. _stmt.h: https://github.com/apache/tvm/blob/v0.26.0/include/tvm/tirx/stmt.h
.. _tirx_stmt.h: https://github.com/apache/tvm/blob/v0.26.0/include/tvm/tirx/tirx_stmt.h
.. _expr.h: https://github.com/apache/tvm/blob/v0.26.0/include/tvm/tirx/expr.h
.. _ir_expr.h: https://github.com/apache/tvm/blob/v0.26.0/include/tvm/ir/expr.h
.. _exec_scope.h: https://github.com/apache/tvm/blob/v0.26.0/include/tvm/tirx/exec_scope.h
.. _transform.cc: https://github.com/apache/tvm/blob/v0.26.0/src/tirx/ir/transform.cc
