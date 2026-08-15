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

TIRx 编译流水线
===============

``tvm.compile(mod, target, tir_pipeline="tirx")`` 接收一个 TIRx module，最终生成两部分代码：CPU 端的启动函数负责准备参数并启动 GPU，GPU 端的 kernel 负责执行计算。编译器不是一步完成这项转换，而是依次运行多个编译步骤。每个步骤称为一个 pass，负责对 IR 做一类特定的转换、检查或标注。

TIRx 的完整 pass 顺序定义在 Apache TVM 源码中的 `python/tvm/tirx/compilation_pipeline.py
<https://github.com/apache/tvm/blob/v0.26.0/python/tvm/tirx/compilation_pipeline.py>`_。

整体编译路径
------------

``target`` 指定代码要在哪种硬件和后端上运行。下面的例子在 GPU 端使用 CUDA，在 CPU 端使用 LLVM：``tvm.compile`` 先将 target 信息写入 module，再运行模块级的 **tirx pipeline**。``tirx_pipeline`` 拆出 CPU 端的 host 函数和 GPU 端的 device 函数后，两者分别经过最后一轮面向具体 target 的转换，再交给相应的代码生成器：

.. code-block:: text

    编写好的 TIRx
          │ BindTarget
          ▼
    tirx_pipeline
    （SplitHostDevice 在其中拆分两条路径）
          ├── host PrimFunc   ──host finalization──▶ C/LLVM
          └── device PrimFunc ─device finalization─▶ CUDA

``PrimFunc`` 是 TIR 中的函数表示。上图中的 host PrimFunc 就是 CPU 端的启动函数，device PrimFunc 则是 GPU 上执行的 kernel；``finalization`` 表示代码生成之前针对具体 target 所做的最后几步转换。

``tirx_pipeline`` 的 pass 顺序
-------------------------------

下表按照实际执行顺序列出 ``tirx_pipeline`` 中的 19 个步骤。ABI 是函数之间的调用约定；表中的 ABI passes 负责把普通 TIR 函数改造成 runtime 能够调用的形式。``PassContext`` 是控制编译选项的配置对象：公共子表达式消除可以关闭，向量化和循环展开的行为也可以通过它调整。

.. list-table::
   :header-rows: 1
   :widths: 6 24 24 46

   * - #
     - 类别
     - Pass
     - 作用
   * - 1
     - TIRx lowering
     - ``LowerTIRx``
     - 完成 TIRx 的核心转换，详见下方 `LowerTIRx 的内部组成`_
   * - 2
     - TIR 规范化
     - ``UnifyThreadBinding``
     - 合并等价的 thread-axis bindings，使每个 ``threadIdx`` / ``blockIdx``
       axis 只声明一次
   * - 3
     - TIR 规范化
     - ``StmtSimplify``
     - 简化 IR 中的算术表达式
   * - 4
     - TIR 规范化
     - ``LowerTIRxOpaque``
     - 转换绑定到线程轴的循环，消除未标注的长度为 1 的循环，并规范化循环 pragma
   * - 5
     - TIR 规范化
     - ``FlattenBuffer``
     - 将剩余的多维 TIR ``BufferLoad`` / ``BufferStore`` 展平为一维访问
   * - 6
     - 计算合法化
     - ``BF16ComputeLegalize``
     - target 不原生支持 ``bfloat16`` 计算时，将其提升到 ``float32`` 并改写为合法形式
   * - 7
     - TIR 规范化
     - ``NarrowDataType(32)``
     - 在能够证明安全时，将索引表达式和循环变量缩窄至 32 位
   * - 8
     - 循环转换
     - ``VectorizeLoop``
     - 将 ``T.vectorized`` 循环转换为向量操作；设置 ``tir.disable_vectorize`` 时，则改写为普通标量循环
   * - 9
     - 循环转换
     - ``UnrollLoop``
     - 展开标记为 ``T.unroll`` 的循环；普通常量循环只有在相应配置或
       pragma 启用时才会自动展开
   * - 10
     - TIR 规范化
     - ``StmtSimplify``
     - 向量化和循环展开后会出现更多可简化的常量，再次执行简化
   * - 11
     - TIR 规范化
     - ``CommonSubexprElim``
     - 将重复的子表达式提取为临时变量；设置 ``tir.disable_cse_tir`` 时跳过
   * - 12
     - 计算合法化
     - ``FP8ComputeLegalize``
     - target 不原生支持 ``float8`` 计算时，将其提升为受支持的类型
       （默认为 ``float32``）
   * - 13
     - 校验与 ABI
     - ``VerifyMemory``
     - 确保 host 代码不会直接解引用 device memory
   * - 14
     - 校验与 ABI
     - ``AnnotateEntryFunc``
     - 只有一个 PrimFunc 时直接将其标记为入口；有多个 PrimFunc 时，则标记其中唯一对外可见的函数
   * - 15
     - 校验与 ABI
     - ``SplitHostDevice``
     - 识别 device regions，拆分 host 与 device PrimFuncs，并将 host 侧调用转换为 kernel-launch ABI
   * - 16
     - 校验与 ABI
     - ``LowerIket``
     - 普通 build 中移除 NVIDIA IKET annotations；启用 IKET 时则将其转换为 tracing 所需的形式
   * - 17
     - 校验与 ABI
     - ``MakePackedAPI``
     - 将 host function 改写为 TVM runtime 通过 packed-function ABI 调用的形式
   * - 18
     - 存储合法化
     - ``FP8StorageLegalize``
     - target 不原生支持 ``float8`` storage 时，改用 ``uint8`` container 保存
   * - 19
     - 存储合法化
     - ``BF16StorageLegalize``
     - target 不原生支持 ``bfloat16`` storage 时，改用 ``uint16`` container 保存

Host 与 Device 的后续处理
-------------------------

上面列出的 19 个步骤组成 ``tirx_pipeline``。这条模块级 pipeline
结束后，``tvm.compile`` 会根据函数类型分别执行 finalization：

- **host**：``LowerTVMBuiltin`` 处理 ``tvm_*`` builtins，``LowerIntrin``
  处理面向具体 target 的 intrinsics。
- **device**：``LowerWarpMemory`` 将 warp-scoped buffers 转换为
  shuffles，随后执行 ``StmtSimplify`` 和 ``LowerIntrin``。

``LowerTIRx`` 的内部组成
------------------------

``LowerTIRx`` 主要完成两个任务：为 tile-level 操作选择具体实现，以及把逻辑数据布局转换成实际的内存索引。它的核心转换由下面两个 passes 组成，定义在 Apache TVM 源码中的
`src/tirx/transform/lower_tirx.cc
<https://github.com/apache/tvm/blob/v0.26.0/src/tirx/transform/lower_tirx.cc>`_：

.. code-block:: text

    LowerTIRx = Sequential([ TilePrimitiveDispatch, LowerTIRxCleanup ])

- **``TilePrimitiveDispatch``** 为 tile 操作选择具体实现。TIRx 中的 ``copy``、
  ``gemm``、``reduction`` 等操作以 ``TilePrimitiveCall`` 表示；这个 pass 根据
  backend 选择对应实现。它还会把 ``T.cta_id``、``T.thread_id`` 等抽象的执行范围编号转换成 kernel launch 参数和线程绑定。
- **``LowerTIRxCleanup``** 将逻辑坐标转换为物理索引。它把支持的逻辑 layout 应用到 buffer access 上，使后续 passes 可以直接处理具体的索引表达式。

完成 ``LowerTIRx`` 后，tile 操作已经换成选定的底层实现，逻辑 layout 也已经落实为物理索引，``T.cta_id`` 和 ``T.thread_id`` 等抽象编号则变成了线程绑定。此时仍可能保留 thread-binding loops 和 TIRx 特有的 loop annotations；后续的
``LowerTIRxOpaque`` 会规范化这些结构，再由 ``tirx.transform.FlattenBuffer``
展平普通 TIR 中的 buffer access。

一个简单 Kernel 的编译过程
--------------------------

下面用一个 scale kernel 观察两件事：``T.cta_id`` 和 ``T.thread_id`` 怎样落实为具体的线程编号，以及一个 TIRx 函数如何拆成 CPU 端的启动函数与 GPU 上执行的 kernel。这个 kernel 处理 1,024 个元素，使用 4 个 CUDA thread blocks（CTA），每个 CTA 包含 256 个线程。

**1. TIRx 源码使用抽象的线程编号。**

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
        B[bx * 256 + tx] = A[bx * 256 + tx] * T.float32(2.0)

``T.device_entry()`` 标记 GPU 代码的入口。``LowerTIRx`` 根据这个标记建立线程绑定；后面的 ``SplitHostDevice`` 再从所得设备代码区域（device region）中拆出单独的 device kernel。``T.cta_id([4])`` 表示 x 方向有 4 个 CTA，
``T.thread_id([256])`` 表示每个 CTA 有 256 个线程。这里的 ``bx`` 和 ``tx``
仍是 TIRx 提供的抽象编号。

**2. ``LowerTIRx`` 将抽象编号转换为 TIR 线程绑定。** 它把 ``bx`` 和 ``tx``
分别绑定到 ``blockIdx.x`` 与 ``threadIdx.x``。省略 buffer declarations 后，核心计算等价于：

.. code-block:: python

    with T.launch_thread("blockIdx.x", 4) as bx:
        tx = T.launch_thread("threadIdx.x", 256)
        B[bx * 256 + tx] = A[bx * 256 + tx] * T.float32(2.0)

这里仍然是 TIR，还不是 CUDA 源码。这段代码只保留了关键映射，不是编译器输出的完整 IR；下一节会给出打印完整结果的命令。

**3. 后续 passes 拆分 host/device，并生成 CUDA。** 编译开始时只有一个 TIRx
函数。``LowerTIRx`` 生成线程绑定和 device region 后，``SplitHostDevice`` 将其拆成两个 TIR 函数（PrimFunc）：

.. code-block:: text

    host launcher（由 scale 生成）
      └── 启动 scale_kernel，gridDim.x = 4，blockDim.x = 256

    device scale_kernel
      └── 每个 GPU 线程将一个输入元素乘以 2

host 函数保存 kernel 的启动逻辑；device 函数保存真正的逐元素计算。
``MakePackedAPI`` 随后将 host 函数转换为 TVM runtime 使用的统一调用形式。Device 函数则交给 CUDA backend，生成与下面代码等价的 CUDA
kernel：

.. code-block:: cuda

    __global__ void scale_kernel(float* A, float* B) {
        int i = blockIdx.x * 256 + threadIdx.x;
        B[i] = A[i] * 2.0f;
    }

整个过程可以概括为：TIRx 描述线程组织和计算，``LowerTIRx`` 将抽象编号落实为
TIR 线程绑定，``SplitHostDevice`` 分开 CPU 端的启动逻辑与 GPU 端的计算，CUDA
backend 最后生成 CUDA 源码。

这里不需要边界判断，因为 ``4 * 256`` 恰好等于 1,024。处理一般长度 ``N``
时，需要向上取整得到 CTA 数量，并在 kernel 中判断 ``i < N``。

检查中间 IR 与生成代码
----------------------

为了查看中间 IR，可以只运行完整 pipeline 最前面的几步，然后停下来打印结果。下面先把 ``scale`` 以全局名 ``main`` 放入 ``IRModule``。CUDA target 指定 GPU
端生成 CUDA，``with_host("llvm")`` 则指定 CPU 端生成 LLVM 代码。
``BindTarget`` 将这组 target 信息写入 module，随后只运行 ``LowerTIRx``：

.. code-block:: python

    from tvm.tirx import transform as TT

    target = tvm.target.Target("cuda").with_host("llvm")
    mod = tvm.IRModule({"main": scale})
    mod = TT.BindTarget(target)(mod)
    mod = TT.LowerTIRx()(mod)         # 运行 LowerTIRx，转换抽象线程编号
    print(mod.script())               # 查看 LowerTIRx 之后的 IR

输出中应该能看到 ``blockIdx.x`` 和 ``threadIdx.x`` 对应的线程绑定，而原来的
``T.cta_id`` 与 ``T.thread_id`` 已经消失。

要查看最终 CUDA，可以运行完整 pipeline。这里的 host module 只导入了一个
device module，因此 ``imports[0]`` 就是生成的 CUDA module；
``inspect_source()`` 返回它的源码：

.. code-block:: python

    exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")
    cuda_mod = exe.mod.imports[0]
    print(cuda_mod.inspect_source())

生成的代码中应该能找到 ``blockIdx.x``、``threadIdx.x``，以及将每个输入元素乘以 2 的计算。
