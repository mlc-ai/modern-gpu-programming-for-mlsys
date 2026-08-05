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

TIRx Lowering Pipeline
======================

调用 ``tvm.compile(mod, target, tir_pipeline="tirx")`` 时，编译器会将输入的
TIRx module 依次送入一组 TIR passes，这组 passes 称为 **tirx pipeline**。
它负责把 tile primitives、使用 ``TileLayout`` 的 buffers 和 execution-scope
ids 等高层结构逐步转换为彼此分离的 **host** 与 **device** functions，最后再
由 CUDA backend 生成源码。

Pipeline 定义在 ``python/tvm/tirx/compilation_pipeline.py`` 的
``tirx_pipeline`` 中。下面按执行顺序介绍其中的 passes。

Pipeline 在编译流程中的位置
---------------------------

``tvm.compile`` 首先绑定 target，再运行下面的 module-level **tirx
pipeline**。随后，host 和 device functions 分别经过 finalization passes，
device function 最终交给 CUDA code generator：

.. code-block:: text

    authored TIRx  ──BindTarget──▶  tirx_pipeline  ──▶  host func  ──host finalize──▶  C/LLVM
                                          │
                                          └──────────▶  device func ──device finalize──▶  CUDA

Pass 执行顺序
-------------

``tirx_pipeline`` 依次执行下表中的 passes，其中少数 pass 会由
``PassContext`` 配置控制是否启用：

.. list-table::
   :header-rows: 1
   :widths: 6 32 62

   * - #
     - Pass
     - 作用
   * - 1
     - ``LowerTIRx``
     - 完成 TIRx 的核心转换，详见下方 `LowerTIRx 内部做了什么`_
   * - 2
     - ``UnifyThreadBinding``
     - 合并等价的 thread-axis bindings，使每个 ``threadIdx`` / ``blockIdx``
       axis 只声明一次
   * - 3
     - ``StmtSimplify``
     - 使用 arithmetic analyzer 简化 statement 中的算术表达式
   * - 4
     - ``LowerTIRxOpaque``
     - 将剩余的 opaque TIRx constructs 转换为普通 TIR
   * - 5
     - ``FlattenBuffer``
     - 将多维 ``BufferLoad`` / ``BufferStore`` 展平为一维访问
   * - 6
     - ``BF16ComputeLegalize``
     - 将 ``bfloat16`` 计算改写为合法形式，其中计算会提升到 f32
   * - 7
     - ``NarrowDataType(32)``
     - 在能够证明安全时，将 index 和 loop 的 ``PrimExpr`` dtype 缩窄为 32 bits
   * - 8
     - ``VectorizeLoop``
     - 将 ``T.vectorized`` loops 改写为 vector operations；设置
       ``tir.disable_vectorize`` 时跳过
   * - 9
     - ``UnrollLoop``
     - 展开标记为 ``T.unroll`` 的 loops，以及较小的常量 loops
   * - 10
     - ``StmtSimplify``
     - Vectorize 和 unroll 暴露出更多常量后，再次执行简化
   * - 11
     - ``CommonSubexprElim``
     - 将重复的子表达式提取为临时变量；设置 ``tir.disable_cse_tir`` 时跳过
   * - 12
     - ``FP8ComputeLegalize``
     - 将 ``float8`` 计算改写为合法形式
   * - 13
     - ``VerifyMemory``
     - 检查 host 代码没有直接解引用 device memory
   * - 14
     - ``AnnotateEntryFunc``
     - 将 module 中唯一的 PrimFunc 标记为入口函数
   * - 15
     - ``SplitHostDevice``
     - 在 ``launch_thread`` 边界处，将每个 kernel 拆分为 **host** function
       和 **device** function
   * - 16
     - ``MakePackedAPI``
     - 将 host function 改写为 TVM launcher 使用的 packed-function ABI
   * - 17
     - ``FP8StorageLegalize``
     - 将 ``float8`` storage 打包为 backend 支持的容器类型
   * - 18
     - ``BF16StorageLegalize``
     - 将 ``bfloat16`` storage 改写为合法形式

之后，编译器会根据 function 类型分别执行 **finalization**：

- **host**：``LowerTVMBuiltin`` 处理 ``tvm_*`` builtins，``LowerIntrin``
  处理 target-specific intrinsics。
- **device**：``LowerWarpMemory`` 将 warp-scoped buffers 转换为
  shuffles，随后执行 ``StmtSimplify`` 和 ``LowerIntrin``。

LowerTIRx 内部做了什么
-----------------------

``LowerTIRx`` 本身由两个 passes 组成，定义在
``src/tirx/transform/lower_tirx.cc``：

.. code-block:: text

    LowerTIRx = Sequential([ TilePrimitiveDispatch, LowerTIRxCleanup ])

- **``TilePrimitiveDispatch``** 根据选中的 backend dispatch，将每个
  ``TilePrimitiveCall``（``copy``、``gemm``、``reduction`` 等）替换为对应
  的实现。
- **``LowerTIRxCleanup``** 运行 ``LayoutApplier``，将使用
  ``TileLayout`` 的 buffer access 变成具体的物理地址计算
  （``addr = data + elem_offset + layout.apply(coord)``），再展平 buffers，
  并将 execution-scope ids 转换为 thread axes，例如
  ``T.cta_id`` / ``T.thread_id`` 通过 ``launch_thread`` 变为
  ``blockIdx`` / ``threadIdx``。

完成 ``LowerTIRx`` 后，module 中只剩普通 TIR：tile primitives 已经展开，
``TileLayout`` 间接层已经消失，scope ids 也已经解析为 thread axes。

完整示例
--------

以下面的 scale kernel 为例：

.. code-block:: python

    @T.prim_func
    def scale(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        B = T.match_buffer(B_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
        B[tx] = A[tx] * T.float32(2.0)

执行 ``LowerTIRx`` 后，scope ids 已经变成真实的 thread axes，layout 也已经
应用到 buffer access 上。这里的 ``A_1`` 和 ``B_1`` 是展平后的一维 views：

.. code-block:: python

    with T.launch_thread("blockIdx.x", 1) as blockIdx_x:
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        bx: T.let = blockIdx_x
        tx: T.let = threadIdx_x
        B_1[threadIdx_x] = A_1[threadIdx_x] * T.float32(2.0)

经过 ``SplitHostDevice`` 和 ``MakePackedAPI`` 后，一个 function 会拆成 host
launcher 和 device kernel：

.. code-block:: python

    @I.ir_module
    class Module:
        def main(...):          # host: packed-API launcher (computes the grid/block, launches)
            ...
        def scale_kernel(...):  # device: the __global__ body, run on the GPU

CUDA backend 随后将 ``scale_kernel`` 生成 ``__global__`` function：
``B_ptr[threadIdx.x] = A_ptr[threadIdx.x] * 2.0f``。

手动检查中间结果
----------------

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
