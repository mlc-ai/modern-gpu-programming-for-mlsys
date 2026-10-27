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

.. _chap_tirx_lowering_pipeline:

TIRx 编译流水线
====================

``tvm_tirx.tirx.compile`` 接受 ``tirx.Function`` 或 ``IRModule``，返回 ``Executable``。默认 pipeline 处理 CUDA entry 和 CPU launcher，无需指定旧的 ``tir_pipeline`` 参数。先进行显式 tile dispatch 和 tensor layout lowering，再拆分 host/device 并降低 ABI。下面展示完整 kernel、中间 IR 和最终 CUDA source 的检查方式。

默认编译 Pipeline
--------------------------

``tvm_tirx.tirx.compile`` 依次完成 target/backend binding、dialect preparation、
module finalization 和 host/device finalization。每个 pass 处理一项 IR 转换：

.. code-block:: text

    Python source → tirx.Function + TensorType + ir.Call / ir.RegionStmt
                       │
                       ▼
    BindTarget → cuda.BindBackendConfig → cuda.SpecializeEntryHelpers
                       │
                       ▼
    TileDispatch → LowerTensorLayouts → StmtSimplify → FlattenTensor
                       │
                       ▼
    index narrowing / vectorization / unrolling / CSE / verification
                       │
                       ▼
    AnnotateEntryFunc → SplitHostDevice → CUDA helpers / dtype / packed ABI
                       ├─ host finalization → native LLVM / ORC JIT
                       └─ device finalization → CUDA code generation

Preparation passes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下顺序直接来自当前 ``compilation_pipeline.py``：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Pass
     - 职责
   * - ``TileDispatch``
     - 验证并展开显式 tile 指令的 canonical operands
   * - ``LowerTensorLayouts``
     - 降低剩余 Tensor 访问与 view 的物理位置
   * - ``StmtSimplify``、``FlattenTensor``
     - 简化表达式，并展平多维普通存储访问
   * - ``NarrowDataType(32)``
     - 能证明安全时缩窄索引类型
   * - ``VectorizeLoop``、``UnrollLoop``、``StmtSimplify``
     - 执行循环转换并再次简化
   * - ``CommonSubexprElim``
     - 提取重复表达式；``tirx.disable_cse_tir`` 可关闭
   * - ``VerifyMemory``、``LowerThreadAllreduce``
     - 检查内存访问，并降低跨线程 reduction

``tirx.disable_vectorize`` 控制 vectorization；显式指令 pattern 生成的 repeat loops
与普通用户循环的转换属于不同层次。``prepare_only=True`` 可以让 pipeline factory
停在 module finalization 之前。

Module finalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AnnotateEntryFunc`` 标记入口，``SplitHostDevice`` 拆分 CPU launcher 与 GPU function。
随后依次运行 ``cuda.LowerIket``、``cuda.SpecializeDeviceHelpers``、
``BF16ComputeLegalize``、``FP8ComputeLegalize``、``MakePackedAPI``、
``FP8StorageLegalize`` 与 ``BF16StorageLegalize``。Packed ABI 让 runtime 能传递参数；
它与 device launch ABI 是不同层次。

拆分后，host 运行 ``LowerTVMBuiltin`` 和 ``LowerIntrin``；device 运行
``StmtSimplify`` 和 ``LowerIntrin``，随后各自交给 target code generator。
Minimal wheel 的 native LLVM/ORC JIT 已足以处理本机 CPU launcher；ARM/Wasm 等对象
生成需要额外 LLVM targets。

显式 tile 与直接 PTX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tile.apply`` 已由作者指定 PTX opcode，``TileDispatch`` 负责验证 view 并展开 group
与 repeat axes。直接 ``T.ptx`` 调用不经过对应的 tile pattern 展开，但仍经过普通
layout、IR 和代码生成 passes，详见 :ref:`chap_tirx_tile_layout_lowering`。

完整 pass 顺序见 `compilation_pipeline.py <https://github.com/apache/tvm-tirx/blob/a821c08826b4721d47122ee9b1dced7f3b88e738/python/tvm_tirx/tirx/compilation_pipeline.py>`_。
用 ``PassContext(instruments=[...])`` 可以观察具体 build 的 passes，不应把某次输出的
行号、临时变量名或所有 IR 节点形状当作固定接口。

完整的 Device Entry
--------------------------------


.. code-block:: python

    import tvm_tirx
    from tvm_tirx.script import tirx as T

    @T.function
    def scale(A: T.Tensor((1024,), "float32"), B: T.Tensor((1024,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(4,), block=256))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        B[bx * 256 + tx] = A[bx * 256 + tx] * T.float32(2)

检查中间 IR Module
----------------------------


.. code-block:: python

    from tvm_tirx.tirx import transform as TT

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"})
    mod = tvm_tirx.IRModule({"main": scale})
    mod = TT.BindTarget(target)(mod)
    mod = TT.TileDispatch()(mod)
    print(mod.script())

编译并检查 CUDA
--------------------


.. code-block:: python

    exe = tvm_tirx.tirx.compile(tvm_tirx.IRModule({"main": scale}), target=target)
    print(exe.mod.imports[0].inspect_source())
