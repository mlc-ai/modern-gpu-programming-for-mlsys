TIRx lowering pipeline
======================

``tvm.compile(mod, target, tir_pipeline="tirx")`` 会把作者写出的 TIRx module 送入 **tirx pipeline**。这是一组有序的 TIR pass，会把你写下的高层构造（tile primitive、带 ``TileLayout`` 的 buffer、execution-scope id）变成拆分后的 **host** + **device** function，然后由 CUDA backend 渲染成源码。Pipeline 定义在 ``python/tvm/tirx/compilation_pipeline.py``（``tirx_pipeline``）中；本页按顺序介绍这些 pass。

它位于哪里
----------

``tvm.compile`` 会先绑定 target，运行 **tirx pipeline**（下面这些 module-level pass），然后分别对 host 和 device function 应用 **finalization** pass，最后把每个 device function 交给 CUDA code generator：

.. code-block:: text

    authored TIRx  ──BindTarget──▶  tirx_pipeline  ──▶  host func  ──host finalize──▶  C/LLVM
                                          │
                                          └──────────▶  device func ──device finalize──▶  CUDA

这些 Pass
---------

``tirx_pipeline`` module pass 会应用下面这个精确序列（少数 pass 受 ``PassContext`` config 控制）：

.. list-table::
   :header-rows: 1
   :widths: 6 32 62

   * - #
     - Pass
     - 作用
   * - 1
     - ``LowerTIRx``
     - 核心 lowering，见下方 `Inside LowerTIRx`_
   * - 2
     - ``UnifyThreadBinding``
     - 合并等价的 thread-axis binding，使每个 ``threadIdx`` / ``blockIdx`` 轴只声明一次
   * - 3
     - ``StmtSimplify``
     - statement-level 算术简化（arith analyzer）
   * - 4
     - ``LowerTIRxOpaque``
     - 把剩余 opaque TIRx 构造 lower 成普通 TIR
   * - 5
     - ``FlattenBuffer``
     - 把多维 ``BufferLoad`` / ``BufferStore`` 展平成 1-D
   * - 6
     - ``BF16ComputeLegalize``
     - 把 ``bfloat16`` compute 重写成合法形式（f32 up-cast）
   * - 7
     - ``NarrowDataType(32)``
     - 在可证明安全时把 index/loop ``PrimExpr`` dtype 缩窄到 32-bit
   * - 8
     - ``VectorizeLoop``
     - 把 ``T.vectorized`` loop 变成 vector op（如果 ``tir.disable_vectorize`` 则跳过）
   * - 9
     - ``UnrollLoop``
     - 展开标记为 ``T.unroll`` 的 loop（以及小的 constant loop）
   * - 10
     - ``StmtSimplify``
     - 在 vectorize/unroll 暴露常量后再次简化
   * - 11
     - ``CommonSubexprElim``
     - 把重复 subexpression hoist 到临时变量（如果 ``tir.disable_cse_tir`` 则跳过）
   * - 12
     - ``FP8ComputeLegalize``
     - 把 ``float8`` compute 重写成合法形式
   * - 13
     - ``VerifyMemory``
     - 检查 host-side code 不会直接 dereference device memory（安全门）
   * - 14
     - ``AnnotateEntryFunc``
     - 把单个 PrimFunc 标记为 module entry point
   * - 15
     - ``SplitHostDevice``
     - 在 ``launch_thread`` 边界把每个 kernel 拆成 **host** function 和 **device** function
   * - 16
     - ``MakePackedAPI``
     - 把 host function 重写成 packed-func ABI（TVM 调用的 launcher）
   * - 17
     - ``FP8StorageLegalize``
     - legalize ``float8`` storage（打包进支持的 container type）
   * - 18
     - ``BF16StorageLegalize``
     - legalize ``bfloat16`` storage

**Finalization** 随后按 function kind 分别运行：

- **host**：``LowerTVMBuiltin`` （lower ``tvm_*`` builtin）、``LowerIntrin`` （target-specific intrinsic）
- **device**：``LowerWarpMemory`` （warp-scoped buffer -> shuffle）、``StmtSimplify``、``LowerIntrin``

Inside LowerTIRx
----------------

``LowerTIRx`` 本身是一个小序列（``src/tirx/transform/lower_tirx.cc``）：

.. code-block:: text

    LowerTIRx = Sequential([ TilePrimitiveDispatch, LowerTIRxCleanup ])

- **``TilePrimitiveDispatch``** 会把每个 ``TilePrimitiveCall``（``copy``、``gemm``、``reduction`` 等）替换成其选定 backend dispatch 生成的 body，也就是 variant selection 和 codegen。
- **``LowerTIRxCleanup``** 会运行 ``LayoutApplier``：把每个带 ``TileLayout`` 的 buffer access 解析成具体物理地址算术（``addr = data + elem_offset + layout.apply(coord)``），展平 buffer，并把 execution-scope id（``T.cta_id`` / ``T.thread_id`` 等）lower 成 ``launch_thread`` 上的 ``blockIdx`` / ``threadIdx``。

因此，``LowerTIRx`` 之后 module 已经是普通 TIR：没有 tile primitive，没有 ``TileLayout`` 间接层，scope id 已解析成 thread axis。

一个 worked example
-------------------

看一个一行 scale kernel：

.. code-block:: python

    @T.prim_func
    def scale(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        B = T.match_buffer(B_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
        B[tx] = A[tx] * T.float32(2.0)

**``LowerTIRx`` 之后**，scope id 变成真实 thread axis，layout 已经应用（``A_1`` / ``B_1`` 是展平后的 1-D view）：

.. code-block:: python

    with T.launch_thread("blockIdx.x", 1) as blockIdx_x:
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        bx: T.let = blockIdx_x
        tx: T.let = threadIdx_x
        B_1[threadIdx_x] = A_1[threadIdx_x] * T.float32(2.0)

**``SplitHostDevice`` + ``MakePackedAPI`` 之后**，一个 function 变成两个：host launcher 和 device kernel：

.. code-block:: python

    @I.ir_module
    class Module:
        def main(...):          # host: packed-API launcher (computes the grid/block, launches)
            ...
        def scale_kernel(...):  # device: the __global__ body, run on the GPU

CUDA backend 随后把 ``scale_kernel`` 渲染成 ``__global__`` function（``B_ptr[threadIdx.x] = A_ptr[threadIdx.x] * 2.0f``）。

自己复现
--------

你可以手动运行 pipeline 的任意前缀来检查某个阶段。这些文档中的 IR snippet 就是这样生成的：

.. code-block:: python

    from tvm.tirx import transform as TT

    target = tvm.target.Target("cuda")
    mod = TT.BindTarget(target.with_host("llvm"))(tvm.IRModule({"main": scale}))
    mod = TT.LowerTIRx()(mod)         # tile primitives dispatched, layouts applied
    print(mod.script())               # inspect the lowered TIRx IR

也可以编译整个 module 并读取生成的 CUDA：

.. code-block:: python

    exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")
    print(exe.mod.imports[0].inspect_source())
