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

.. _chap_arch:

TIRx 编译器内部机制：编译与 Lowering 流水线
================================================

``tvm.compile(mod, target, tir_pipeline="tirx")`` 会把编写好的 TIRx module
转换成 host launcher 和 device code。这个过程并非一次完成：编译器先处理
TIRx 特有的结构，再用通用的 TIRx 规范化与合法化 passes 处理结果，最后拆分
module 并交给后端生成代码。

完整顺序定义在 `compilation_pipeline.py
<https://github.com/apache/tvm/blob/v0.26.0/python/tvm/tirx/compilation_pipeline.py>`_。
本页先说明这条流水线在 ``tvm.compile`` 中的位置，再按阶段解释各个 pass
改变了什么，以及 host 与 device 两条路径从哪里分开。

整体编译路径
------------

``tvm.compile`` 首先绑定 target，再运行下面的 module-level **tirx
pipeline**。随后，host 和 device functions 分别经过 finalization passes，
device function 最终交给 CUDA code generator：

.. code-block:: text

    authored TIRx  ──BindTarget──▶  tirx_pipeline  ──▶  host func  ──host finalize──▶  C/LLVM
                                          │
                                          └──────────▶  device func ──device finalize──▶  CUDA

``tirx_pipeline`` 的 Pass 执行顺序
-----------------------------------

``tirx_pipeline`` 按下表中的 19 个步骤组织。公共子表达式消除是可选项，
vectorization 和 unrolling 的行为也可以通过 ``PassContext`` 控制：

.. list-table::
   :header-rows: 1
   :widths: 6 24 24 46

   * - #
     - 阶段
     - Pass
     - 作用
   * - 1
     - TIRx lowering
     - ``LowerTIRx``
     - 完成 TIRx 的核心转换，详见下方 `LowerTIRx 内部做了什么`_
   * - 2
     - TIR 规范化
     - ``UnifyThreadBinding``
     - 合并等价的 thread-axis bindings，使每个 ``threadIdx`` / ``blockIdx``
       axis 只声明一次
   * - 3
     - TIR 规范化
     - ``StmtSimplify``
     - 使用 arithmetic analyzer 简化 statement 中的算术表达式
   * - 4
     - TIR 规范化
     - ``LowerTIRxOpaque``
     - 处理剩余的 opaque constructs，包括 thread-binding loops、unit loops
       和 pragma annotations
   * - 5
     - TIR 规范化
     - ``FlattenBuffer``
     - 将剩余的多维 TIR ``BufferLoad`` / ``BufferStore`` 展平为一维访问
   * - 6
     - 计算合法化
     - ``BF16ComputeLegalize``
     - 将 ``bfloat16`` 计算改写为合法形式，其中计算会提升到 f32
   * - 7
     - TIR 规范化
     - ``NarrowDataType(32)``
     - 在能够证明安全时，将 index 和 loop 的 scalar ``Expr`` type 缩窄为 32 bits
   * - 8
     - Loop lowering
     - ``VectorizeLoop``
     - 将 ``T.vectorized`` loops 转换为 vector operations；设置
       ``tir.disable_vectorize`` 时，改为将这些 loops scalarize
   * - 9
     - Loop lowering
     - ``UnrollLoop``
     - 展开标记为 ``T.unroll`` 的 loops；普通常量 loops 只有在相应配置或
       pragma 启用时才会自动展开
   * - 10
     - TIR 规范化
     - ``StmtSimplify``
     - Vectorize 和 unroll 暴露出更多常量后，再次执行简化
   * - 11
     - TIR 规范化
     - ``CommonSubexprElim``
     - 将重复的子表达式提取为临时变量；设置 ``tir.disable_cse_tir`` 时跳过
   * - 12
     - 计算合法化
     - ``FP8ComputeLegalize``
     - 将 ``float8`` 计算改写为合法形式
   * - 13
     - 校验与 ABI
     - ``VerifyMemory``
     - 检查 host 代码没有直接解引用 device memory
   * - 14
     - 校验与 ABI
     - ``AnnotateEntryFunc``
     - 将唯一 function 标记为入口；对于多 function module，则标记其中唯一
       对外可见的 PrimFunc
   * - 15
     - 校验与 ABI
     - ``SplitHostDevice``
     - 识别 device regions，拆分 host 与 device PrimFuncs，并将 host 侧调用
       转换为 kernel-launch ABI
   * - 16
     - 校验与 ABI
     - ``LowerIket``
     - 普通 build 中移除 frontend-only NVIDIA IKET annotations；IRModule 显式
       启用 IKET 时则生成 IKET metadata 和 placeholders
   * - 17
     - 校验与 ABI
     - ``MakePackedAPI``
     - 将 host function 改写为 TVM launcher 使用的 packed-function ABI
   * - 18
     - Storage 合法化
     - ``FP8StorageLegalize``
     - 将 ``float8`` storage 转换为 ``uint8`` container
   * - 19
     - Storage 合法化
     - ``BF16StorageLegalize``
     - 将 ``bfloat16`` storage 转换为 ``uint16`` container

Host 与 Device Finalization
---------------------------

上面列出的 19 个步骤组成 ``tirx_pipeline``。这条 module-level pipeline
结束后，``tvm.compile`` 会根据 function 类型分别执行 finalization：

- **host**：``LowerTVMBuiltin`` 处理 ``tvm_*`` builtins，``LowerIntrin``
  处理 target-specific intrinsics。
- **device**：``LowerWarpMemory`` 将 warp-scoped buffers 转换为
  shuffles，随后执行 ``StmtSimplify`` 和 ``LowerIntrin``。

``LowerTIRx`` 内部做了什么
---------------------------

正常编译时，``LowerTIRx`` 本身由两个 passes 组成，定义在
`lower_tirx.cc
<https://github.com/apache/tvm/blob/v0.26.0/src/tirx/transform/lower_tirx.cc>`_：

.. code-block:: text

    LowerTIRx = Sequential([ TilePrimitiveDispatch, LowerTIRxCleanup ])

- **``TilePrimitiveDispatch``** 根据 backend dispatch，为每个
  ``TilePrimitiveCall``（``copy``、``gemm``、``reduction`` 等）选择具体实现，
  同时把 ``T.cta_id``、``T.thread_id`` 等 execution-scope IDs 解析为 kernel
  launch parameters 和对应的 bindings。
- **``LowerTIRxCleanup``** 运行 ``LayoutApplier``，将使用
  ``TileLayout`` 的 buffer access 变成具体的物理地址计算
  （``addr = data + elem_offset + layout.apply(*coord, shape=shape)``），把带 layout
  的 buffer parameters 替换成物理 views，并移除显式的 buffer offsets。

完成 ``LowerTIRx`` 后，tile primitives 和 ``TileLayout`` 间接层已经消失，
execution-scope IDs 也已经解析。此时仍有少量 opaque TIRx constructs；后续的
``LowerTIRxOpaque`` 会先处理这些结构，后续的
``tirx.transform.FlattenBuffer`` pass 再展平普通 TIR 中的 buffer access。

端到端 IR 演化示例
------------------

以下面的 scale kernel 为例：

.. code-block:: python

    @T.prim_func
    def scale(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        B = T.match_buffer(B_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
        B[tx] = A[tx] * T.float32(2.0)

这个简单的一维 kernel 没有非平凡的 ``TileLayout``，主要用来展示
``LowerTIRx`` 如何将 scope IDs 转换成真实的 thread axes。下面只摘录核心
body，省略 buffer declarations 和未使用的 warp-ID binding；``A_1`` 与
``B_1`` 是生成的物理 views：

.. code-block:: python

    # 省略 match_buffer / decl_buffer declarations
    with T.launch_thread("blockIdx.x", 1) as blockIdx_x:
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        bx: T.let = blockIdx_x
        tx: T.let = threadIdx_x
        B_1[threadIdx_x] = A_1[threadIdx_x] * T.float32(2.0)

``SplitHostDevice`` 随后将单个 function 拆成 host launcher 和 device kernel，
``MakePackedAPI`` 再将 host launcher 转换为 TVM 的 packed-function ABI：

.. code-block:: text

    @I.ir_module
    class Module:
        def main(...):          # host: packed-API launcher (computes the grid/block, launches)
            ...
        def scale_kernel(...):  # device: the __global__ body, run on the GPU

CUDA backend 随后将 ``scale_kernel`` 生成 ``__global__`` function：
``B_ptr[threadIdx.x] = A_ptr[threadIdx.x] * 2.0f``。

检查中间 IR 与生成代码
----------------------

可以手动运行 pipeline 的任意前缀，检查某个阶段的 IR。本书中的 IR 片段也是
用这种方式生成的：

.. code-block:: python

    from tvm.tirx import transform as TT

    target = tvm.target.Target("cuda")
    mod = TT.BindTarget(target.with_host("llvm"))(tvm.IRModule({"main": scale}))
    mod = TT.LowerTIRx()(mod)         # tile primitives dispatched, layouts applied
    print(mod.script())               # inspect the lowered TIRx IR

也可以编译完整 module，再查看生成的 CUDA：

.. code-block:: python

    exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")
    print(exe.mod.imports[0].inspect_source())
