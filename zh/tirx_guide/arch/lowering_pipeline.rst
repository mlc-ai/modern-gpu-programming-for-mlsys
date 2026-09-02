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
===============

调用 ``tvm.compile(mod, target, tir_pipeline="tirx")`` 时，输入并不是直接从
Python 翻译成 PTX。编译器先消费 TIRx 特有的 tile primitive、execution scope
和 layout，再进行通用 TIR 的正规化、局部优化、类型合法化、host/device
拆分与代码生成。这里的 pass 是对 IR 执行一类转换、检查或标注的编译步骤；
target 指定设备和代码生成后端。

本页回答两个问题：**pipeline 里有哪些 passes，以及优化主要发生在哪里**。
``Tx.gemm_async`` 如何使用 layout 推导 descriptor、直接 ``T.ptx.*`` 绕过什么，
单独放在 :ref:`chap_tirx_tile_layout_lowering` 中展开。

.. admonition:: 先给结论

   默认 TIRx 编译器可以概括为：**较薄的通用优化流水线，加上较厚的算子级
   lowering**。Kernel 作者或上层生成器负责 tile sizes、warp roles、同步、
   pipeline stages 和主要 layout；tile primitive 的实现负责合法性检查、
   descriptor/地址参数推导和指令分解；通用 passes 主要负责化简、合法化、
   模块拆分和 ABI。

.. note::

   本书固定使用 Apache TVM ``0.26.0``。下面的 pass 名称和顺序都对应这个版本，
   并且讨论的是显式指定 ``tir_pipeline="tirx"`` 的路径。开发分支中的 pipeline
   可能已经变化；名为 ``"default"`` 的 pipeline 也不是 ``"tirx"`` 的别名。

完整编译路径
------------

先按职责观察整条路径：

.. code-block:: text

    TIRx Python
        │  script parser
        ▼
    TIRx PrimFunc
      ├─ logical computation
      ├─ Tx.* TilePrimitiveCall
      ├─ Buffer + TileLayout
      └─ device / CTA / warpgroup / warp / thread scopes
        │
        │  BindTarget
        ▼
    LowerTIRx
      ├─ TilePrimitiveDispatch：Tx.* → target-specific TIR / T.ptx.*
      └─ LowerTIRxCleanup：剩余 layout/access → physical buffer access
        │
        ▼
    结构正规化 + 局部程序变换 + dtype 合法化
        │
        ▼
    校验 + SplitHostDevice + launcher ABI
        ├─ host PrimFunc   → host finalization   → host backend
        └─ device PrimFunc → device finalization → CUDA C++ / inline PTX
                                                        │
                                                        ▼
                                                PTX / cubin / runtime module

``PrimFunc`` 是 TIR 中的函数表示，finalization 是代码生成前面向具体 target
的最后一组转换。``BindTarget`` 在 ``tirx_pipeline`` 之前运行。Target 不只是决定最后使用哪个
code generator；``TilePrimitiveDispatch`` 在较早阶段就需要 target，才能查找
对应的算子实现。``SplitHostDevice`` 位于 module-level pipeline 后半段，拆分后
host 和 device functions 才分别进入各自的 finalization。

高层 tile primitive 和直接 PTX 从不同位置进入这条路径：

.. code-block:: text

    Tx.gemm_async ── TilePrimitiveDispatch ──▶ T.ptx.tcgen05.mma ──┐
                                                                    ├─▶ 后续通用 passes
    直接 T.ptx.tcgen05.mma ────────────────────────────────────────┘

因此，直接 PTX 绕过的是对应 tile primitive 的算子级选择、检查和参数推导，
不是整个 TIRx pipeline。两条路径的详细比较见
:ref:`chap_tirx_tile_layout_lowering`。

``LowerTIRx`` 的边界
------------------------------

默认情况下，``LowerTIRx`` 包含两个 transformation passes：

.. code-block:: text

    LowerTIRx = Sequential([
        TilePrimitiveDispatch,
        LowerTIRxCleanup,
    ])

若设置 ``TVM_PRINT_AFTER_TIRX_DISPATCH_OPS``，两者之间还会临时插入一个
``PrintIR``，但它只是调试 instrumentation，不是固定 transformation。

``TilePrimitiveDispatch`` 首先选择已注册的 target-specific 实现，把
``Tx.copy``、``Tx.gemm_async``、``Tx.reduce`` 等 ``TilePrimitiveCall`` 替换为
lower-level TIR 或 ``T.ptx.*``，并解析 device entry 内的 scope IDs。例如：

.. code-block:: text

    bx = T.cta_id([grid_x])       →  bx = blockIdx.x
    tx = T.thread_id([block_x])   →  tx = threadIdx.x

``LowerTIRxCleanup`` 随后对仍然存在的直接 ``BufferLoad`` / ``BufferStore`` 应用
memory layout，展平相应 buffers，清除已消费的 layout metadata，并移除
``tirx.buffer_offset`` wrappers。必须先 dispatch 后 cleanup：算子实现需要在
metadata 消失前读取完整的 region、shape、dtype、scope 和 layout。

``LowerTIRx`` 成功结束后：

- ``TilePrimitiveCall`` 已被具体实现替换；
- 抽象 scope IDs 已变成 launch parameters、``Bind`` 和 thread bindings；
- layout 已被算子 lowering 消费，或被 cleanup 物化为后端能处理的地址；
- ``T.ptx.*`` 等 target intrinsics 仍可存在；
- thread-binding loops 和 TIRx-specific loop annotations 仍可能存在，随后由
  ``LowerTIRxOpaque`` 规范化；
- host/device split、ABI lowering 和最终 code generation 尚未完成。

因此，这个阶段只能说 **TIRx 的核心高层语义已经消解**，不能说已经得到最终
PTX，也不宜把它描述成编译流程的终点。算子 dispatch 和 layout cleanup 的内部
过程见 :ref:`chap_tirx_tile_layout_lowering`。

最小示例：从 scope IDs 到 host/device
---------------------------------------

下面用一个不依赖 Tensor Core 的 scale kernel 串起这些阶段。它启动 4 个 CTAs，
每个 CTA 有 256 个 threads：

.. code-block:: python

    import tvm
    from tvm.script import tirx as T

    @T.prim_func
    def scale(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (1024,), "float32")
        B = T.match_buffer(B_ptr, (1024,), "float32")
        T.device_entry()
        bx = T.cta_id([4])
        tx = T.thread_id([256])
        i = bx * 256 + tx
        B[i] = A[i] * T.float32(2.0)

``T.device_entry()`` 标出 device region。``LowerTIRx`` 将抽象 IDs 解析成
``blockIdx.x`` 和 ``threadIdx.x``；省略 buffer declarations 后，核心结构可写成：

.. code-block:: python

    with T.launch_thread("blockIdx.x", 4) as bx:
        tx = T.launch_thread("threadIdx.x", 256)
        i = bx * 256 + tx
        B[i] = A[i] * T.float32(2.0)

这仍是 TIR 的结构摘要，不是 CUDA 源码，也不是完整的 printer output。
``SplitHostDevice`` 随后把原来的一个 ``PrimFunc`` 拆成两部分：

.. code-block:: text

    host launcher
      └─ 启动 scale_kernel，gridDim.x = 4，blockDim.x = 256

    device scale_kernel
      └─ 每个 GPU thread 将一个元素乘以 2

``MakePackedAPI`` 把 host entry 降低为 runtime 使用的统一 ABI（函数调用约定）。
Device function 则交给 CUDA backend，最终生成与下面代码等价的 kernel：

.. code-block:: cuda

    __global__ void scale_kernel(float* A, float* B) {
        int i = blockIdx.x * 256 + threadIdx.x;
        B[i] = A[i] * 2.0f;
    }

后续 passes：按职责理解
------------------------

``LowerTIRx`` 后的 passes 可以先分成四组；组内项目仍按源码定义的实际顺序
执行，不能因为教学上分组就任意交换。``PassContext`` 是控制 pass 行为的
编译配置对象。

.. list-table::
   :header-rows: 1
   :widths: 21 35 44

   * - 职责
     - 主要 passes
     - 做什么、不做什么
   * - 结构正规化
     - ``UnifyThreadBinding``、``StmtSimplify``、``LowerTIRxOpaque``、
       ``FlattenBuffer``
     - 统一 thread bindings，化简地址和条件，处理 thread-binding loops、
       unit loops 与 pragmas，并展平剩余 ``BufferLoad`` / ``BufferStore``
   * - 局部程序变换
     - ``NarrowDataType``、``VectorizeLoop``、``UnrollLoop``、第二次
       ``StmtSimplify``、``CommonSubexprElim``
     - 缩窄安全的 index；兑现已有 vectorize/unroll 标记或配置；执行局部代数
       与公共子表达式化简，而不是自动发现完整 schedule
   * - 类型合法化
     - BF16/FP8 compute legalization、BF16/FP8 storage legalization、
       final ``LowerIntrin``
     - 将当前 target/backend 不能直接表示的 compute、storage 和 intrinsic
       改写为受支持形式；部分 pass 可能因 target 能力而成为 no-op
   * - 校验、模块和 ABI lowering
     - ``VerifyMemory``、``AnnotateEntryFunc``、``SplitHostDevice``、
       ``LowerIket``、``MakePackedAPI``
     - 验证设备计算处于 thread environment，抽取 device kernel，降低
       kernel launch，并生成 runtime 可调用的 packed ABI

``VectorizeLoop`` 主要降低已经写成 ``T.vectorized`` 的 loops；它不会自己搜索
应该向量化哪一层；禁用 vectorization 时，相应循环按标量循环处理。
``UnrollLoop`` 默认主要处理显式 ``T.unroll``，也可以通过
``PassContext`` 或 pragma 设置自动展开阈值。两者都不等于自动 tiling 或
software-pipeline synthesis。

Host/device split 与最终 codegen
--------------------------------

在 Apache TVM 0.26.0 中，``SplitHostDevice`` 是一个组合 pass：它识别 device
regions，将其抽取为 device ``PrimFunc``，并把 host 侧调用降低为 kernel-launch
约定。随后 ``MakePackedAPI`` 将公开的 host entry 改写为 TVM runtime 使用的
packed-function ABI。

.. code-block:: text

    一个包含 device region 的 PrimFunc
        │
        │  SplitHostDevice
        ▼
    host launcher                         device PrimFunc
      ├─ 准备调用参数                      ├─ thread_extent
      ├─ grid/block launch parameters      ├─ physical buffer accesses
      └─ 调用 device kernel                └─ T.ptx.* / target intrinsics
        │                                         │
        │ host finalization                       │ device finalization
        ▼                                         ▼
    host target backend                    CUDA code generator
                                                  │
                                                  ▼
                                       CUDA C++ source + inline PTX/helpers
                                                  │
                                                  ▼
                                        CUDA frontend / assembler / runtime

Module-level pipeline 结束后，finalization 分别运行：

- **host**：``LowerTVMBuiltin``、``LowerIntrin``；
- **device**：``LowerWarpMemory``、``StmtSimplify``、``LowerIntrin``。

``LowerTVMBuiltin`` 处理 ``tvm_*`` builtins，``LowerIntrin`` 处理
target-specific intrinsics。
``LowerWarpMemory`` 将 warp-scoped buffers 降成 local storage 和 shuffle 等
形式。CUDA code generator 随后生成 CUDA C++；部分 ``T.ptx.*`` intrinsic 会被
打印为 inline PTX 或 helper code。之后 NVRTC/NVCC/ptxas 等 CUDA toolchain
组件才继续产生可加载的 PTX 或 binary。``LowerIntrin`` 本身并不等于“输出 PTX”。

编译器优化什么、不优化什么
----------------------------

默认 TIRx pipeline 的责任边界如下：

.. list-table::
   :header-rows: 1
   :widths: 27 73

   * - 层次
     - 主要责任
   * - Kernel 作者或生成 kernel 的 agent
     - 选择 tile sizes、pipeline stages、warp roles、execution scope、同步、
       主要 layout 和跨 tile 的整体 schedule
   * - Layout helpers 与 tile dispatcher
     - 根据显式 dtype/shape/mode 构造已知 layout；静态选择实现，检查合法性，
       推导 descriptor/物理参数，并将单个 tile operation 分解为硬件指令
   * - 通用 TIR passes
     - 局部化简、index narrowing、显式或配置驱动的 vectorize/unroll，以及
       dtype、module 和 ABI 合法化
   * - CUDA backend 与 toolchain
     - 生成 target code，进行更底层的 peephole、register allocation 和组装

所以，默认 pipeline 中确实有局部优化，但通常没有通用的自动 tiling、跨算子
fusion、software-pipeline synthesis、warp-specialization synthesis、全局
layout search、cost-model selection 或 autotuning。一个 dispatcher 根据 shape
将 tile operation 拆成多条硬件指令，是算子实现的一部分，不能与全程序
schedule search 混为一谈。

检查完整流水线
--------------

前面的 ``scale`` 例子也可以直接用于检查中间 IR。先绑定 CUDA device 与 LLVM
host target，再观察 ``LowerTIRx`` 前后的结果：

.. code-block:: python

    import tvm
    from tvm.tirx import transform as TT

    target = tvm.target.Target("cuda").with_host("llvm")
    mod = tvm.IRModule({"main": scale})
    bound = TT.BindTarget(target)(mod)

    print("=== authored TIRx ===")
    print(bound.script())

    print("=== after LowerTIRx ===")
    lowered = TT.LowerTIRx()(bound)
    print(lowered.script())

要检查完整 pipeline 生成的 CUDA source：

.. code-block:: python

    exe = tvm.compile(
        tvm.IRModule({"main": scale}),
        target=target,
        tir_pipeline="tirx",
    )
    print(exe.mod.imports[0].inspect_source())

这个例子的 host module 只导入一个 device module，因此 ``imports[0]`` 就是
CUDA module。如果问题发生在 ``Tx.*`` 到 ``T.ptx.*`` 之间，应进一步单独观察
``TilePrimitiveDispatch``；Blackwell target 的设置和具体方法见
:ref:`chap_tirx_tile_layout_lowering`。

Pass 顺序参考
-------------

Apache TVM 0.26.0 的 ``tirx_pipeline`` 在 **默认配置且 CSE 开启** 时按下面
顺序执行，共 19 步。若设置 ``tir.disable_cse_tir=True``，第 11 步会被省略，
后续 passes 依次前移。

.. list-table::
   :header-rows: 1
   :widths: 6 29 65

   * - #
     - Pass
     - 作用
   * - 1
     - ``LowerTIRx``
     - Dispatch tile primitives、解析 execution scope，并物化 layout
   * - 2
     - ``UnifyThreadBinding``
     - 合并等价的 thread-axis bindings
   * - 3
     - ``StmtSimplify``
     - 使用 arithmetic analyzer 化简 statements 和索引表达式
   * - 4
     - ``LowerTIRxOpaque``
     - 转换 thread-binding loops、消除无 annotation 的 unit loops，并规范化
       loop pragmas
   * - 5
     - ``FlattenBuffer``
     - 展平剩余 ``BufferLoad`` / ``BufferStore``
   * - 6
     - ``BF16ComputeLegalize``
     - target 不支持时将 BF16 compute 提升至 ``float32`` 并改写
   * - 7
     - ``NarrowDataType(32)``
     - 能够证明安全时，将 index/loop expressions 缩窄到 32 bits
   * - 8
     - ``VectorizeLoop``
     - Lower 已标记的 vectorized loops；禁用时将其作为标量 loops 处理
   * - 9
     - ``UnrollLoop``
     - 展开显式 ``T.unroll``，并执行配置或 pragma 允许的自动展开
   * - 10
     - ``StmtSimplify``
     - 在 vectorize/unroll 暴露常量后再次化简
   * - 11
     - ``CommonSubexprElim``
     - 执行可配置关闭的公共子表达式消除
   * - 12
     - ``FP8ComputeLegalize``
     - target 不支持时将 FP8 compute 提升至默认的 ``float32`` 并改写
   * - 13
     - ``VerifyMemory``
     - 检查 GPU target/default calling-convention function 中，参数 buffer 的
       load/store 是否处于 ``thread_extent`` 环境
   * - 14
     - ``AnnotateEntryFunc``
     - 单一 PrimFunc 时直接标记；多函数 module 中标记唯一的公开 PrimFunc
   * - 15
     - ``SplitHostDevice``
     - 标注/抽取 device functions，并 lowering host-to-device kernel calls
   * - 16
     - ``LowerIket``
     - IKET 未启用时移除相关 annotations；启用时生成所需 tracing 形式
   * - 17
     - ``MakePackedAPI``
     - 将 host entry 改写为 runtime packed-function ABI
   * - 18
     - ``FP8StorageLegalize``
     - target 需要 fallback 时，将 FP8 storage 改写为等宽 ``uint8`` 表示
   * - 19
     - ``BF16StorageLegalize``
     - target 需要 fallback 时，将 BF16 storage 改写为等宽 ``uint16`` 表示

不要仅凭 pass 名称推断默认会进行 aggressive auto-vectorization 或
auto-unrolling；还需要检查 loop annotations 和当前 ``PassContext`` 配置。

版本与核心源码
--------------

- `compilation_pipeline.py`_：module-level pipeline 与 host/device finalization；
- `lower_tirx.cc`_：``LowerTIRx`` 的两个 transformation passes 和调试
  ``PrintIR`` 插入点。

.. _compilation_pipeline.py: https://github.com/apache/tvm/blob/v0.26.0/python/tvm/tirx/compilation_pipeline.py
.. _lower_tirx.cc: https://github.com/apache/tvm/blob/v0.26.0/src/tirx/transform/lower_tirx.cc
