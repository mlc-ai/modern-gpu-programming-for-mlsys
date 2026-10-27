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
==========================

:ref:`chap_tirx_primer` 展示了 scope、layout 与指令选择。本页沿一个完整程序，观察
这些信息分别保存在函数、类型、语句区域和调用的哪个位置。以下 API 对应
``apache-tvm-tirx==0.1.0a3``，Python namespace 为 ``tvm_tirx``。

通用 IR 与 TIRx 方言
------------------------------

``@T.function`` 返回 ``tvm_tirx.tirx.Function``。函数边界使用 ``BaseFunc`` 和
``IRModule``，函数体使用通用的 ``tvm_tirx.ir`` 语句与表达式；Tensor 的存储约定和
layout 则由 TIRx 类型保存。当前对象关系如下：

.. code-block:: text

    IRModule
    └─ tirx.Function
       ├─ params: ir.Var，参数类型位于 Var.ty
       ├─ ret_type / attrs / ty
       └─ body: ir.SeqStmt
          ├─ ir.Bind(var, value)
          ├─ ir.RegionStmt(op, args, attrs, body)
          ├─ ir.If / ir.For / ir.While
          ├─ ir.TensorStore
          └─ ir.Evaluate(ir.Call(...))

``ir.Call`` 保存 ``op``、``args``、``attrs``、``ty_args`` 和结果类型 ``ty``。
带返回值的调用可以绑定到 ``Var``；以副作用为主的调用由 ``Evaluate`` 承载。
``Function.params`` 中的 Tensor 参数直接携带 ``TensorType``，无需额外的
handle-to-buffer 映射。

沿一个完整程序看 IR
----------------------

以下 kernel 为每个 thread 分配四个局部值，用显式 ``mov.b32`` tile 指令清零，
再执行一次逐元素复制。它同时展示 allocation、tile 调用和普通 Tensor 访问。

.. code-block:: python

    import tvm_tirx
    from tvm_tirx.script import tirx as T
    from tvm_tirx.tirx.layout import TileLayout, S
    from tvm_ffi import structural_walk

    @T.function
    def ir_copy(A: T.Tensor((128, 4), "float32"), B: T.Tensor((128, 4), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=128))
        tx = T.cuda.thread_idx("x")
        values = T.alloc_local((4,), "float32")
        T.cuda.tile.apply(T.ptx["mov.b32"], values.view(4, 1, layout=TileLayout(S[(4, 1) : (1, 1)]))[:, :], T.float32(0), scope="thread")
        for j in T.unroll(4):
            B[tx, j] = A[tx, j] + values[j]

可以直接检查函数参数和调用节点：

.. code-block:: python

    kernel = ir_copy
    assert isinstance(kernel, tvm_tirx.tirx.Function)
    print(kernel.params[0].ty)
    print(type(kernel.body))
    calls = []
    structural_walk(kernel, (tvm_tirx.ir.Call, calls.append))
    names = {call.op.name for call in calls if isinstance(call.op, tvm_tirx.ir.Op)}
    assert "tirx.cuda.tile.apply" in names
    print(kernel.script())

类型、存储与视图
----------------

Tensor 是带 ``TensorType`` 的表达式。``TensorType`` 保存逻辑 ``shape``、元素
``dtype``、storage ``scope``、``data_alignment`` 和 ``layout``。指针本身不是
``TensorType`` 的字段；``tensor.data`` 从 Tensor 值取得数据指针。

局部分配表现为 ``Bind`` 的右侧 allocation Call，``T.decl_tensor(pointer, ...)``
在已有 pointer 上声明访问约定。``view``、``sub``、``permute`` 和 ``rearrange``
生成新的访问视图，不复制底层数据。不同视图可以共享存储，同时拥有不同的逻辑 shape
与 layout。``Dreg.view(128, N, layout=...)`` 因而可以把每个 thread 的 N 个局部值
解释为 warpgroup 分布式 tile。

.. code-block:: text

    tensor value ── ty ──▶ TensorType(shape, dtype, scope, alignment, layout)
         │
         ├─ data / ptr_to(...) ──▶ typed pointer
         └─ view / sub / rearrange ──▶ another tensor view over the same storage

单点访问使用 ``TensorLoad`` / ``TensorStore``。切片得到 ``ir.TensorRegion``，其中
保留 Tensor 表达式与逻辑范围；``sub`` 则返回可以继续变形的 Tensor view。整数索引
在普通 region 中保留长度为 1 的维度，在 ``sub`` 中移除该维度。因此，stage-indexed
SMEM 应先用 ``buffer.sub[stage, :, :]`` 取得二维 view，再交给二维 tile 指令。

Layout 的对象关系
------------------------

``TensorType.layout`` 保存 storage mapping：``TileLayout`` 用 shard、replica 和
固定 offset 描述命名轴上的仿射映射；``ComposeLayout`` 将 tile mapping 与
``SwizzleLayout`` 的地址置换组合。默认 memory axis ``m``、CUDA 执行轴和 TMEM 的
``TLane`` / ``TCol`` 都可以参与映射。Layout 描述元素位置，不决定调用者执行哪条指令。

执行层级与调用边界
------------------

``T.device_entry(launch=...)`` 形成 ``ir.RegionStmt``，operator 为
``tirx.device_entry``。LaunchConfig 的 grid/block/cluster 信息与区域一起保存；
``T.cuda.thread_idx`` 等操作读取硬件坐标。``T.cuda.tile.apply(..., scope=...)``
则指定该次 tile 调用的协作范围，支持 thread、warp、warpgroup、CTA 和 cluster。

Storage scope 与 execution scope 是两个维度：``shared.dyn`` 指明值的存储位置，
``scope="warpgroup"`` 指明哪些 threads 共同执行当前操作。Layout 中的
``tid_in_wg`` 又进一步描述元素归属于哪个执行成员。

显式 tile 调用
--------------------

当前 tile API 的 operator 是 ``tirx.cuda.tile.apply``，它使用普通 ``ir.Call``
保存显式 PTX 指令、operand regions 与参数；scalar ``tile.ewise`` 也有自己的调用
约定。``TileDispatch`` 验证并展开已指定的指令，不从通用 GEMM/copy 名称搜索实现。
``T.ptx`` 直接调用则已经表示特定指令。两条路径都继续经过 layout、普通表达式、
ABI 和代码生成 passes，详见 :ref:`chap_tirx_tile_layout_lowering`。

绑定与可变状态
--------------

``ir.Bind`` 在当前 ``SeqStmt`` 中声明一个值，其变量在后续语句可见。标量值 binding
不可变；可变计数器使用局部单元，更新会变成 Tensor store。控制流保留词法嵌套，
所以循环外绑定的 tile 坐标不会随 scheduler cursor 自动更新。Persistent kernel 应在
每轮任务内部重新计算坐标，详见 :ref:`chap_gemm_advanced`。

核心源码导航
------------

- `function.py <https://github.com/apache/tvm-tirx/blob/a821c08826b4721d47122ee9b1dced7f3b88e738/python/tvm_tirx/tirx/function.py>`_：Function 的参数与函数体。
- `tensor.py <https://github.com/apache/tvm-tirx/blob/a821c08826b4721d47122ee9b1dced7f3b88e738/python/tvm_tirx/tirx/tensor.py>`_：TensorType 与 view 操作。
- `stmt.py <https://github.com/apache/tvm-tirx/blob/a821c08826b4721d47122ee9b1dced7f3b88e738/python/tvm_tirx/ir/stmt.py>`_：Bind、RegionStmt 与 TensorStore。
- `expr.py <https://github.com/apache/tvm-tirx/blob/a821c08826b4721d47122ee9b1dced7f3b88e738/python/tvm_tirx/ir/expr.py>`_：Call、TensorLoad 与 TensorRegion。
- `layout.h <https://github.com/apache/tvm-tirx/blob/a821c08826b4721d47122ee9b1dced7f3b88e738/include/tvm/tirx/layout.h>`_：命名轴与 layout 对象。
- `patterns <https://github.com/apache/tvm-tirx/blob/a821c08826b4721d47122ee9b1dced7f3b88e738/python/tvm_tirx/backend/cuda/tile/patterns/__init__.py>`_：显式 tile.apply 的调用约定。
