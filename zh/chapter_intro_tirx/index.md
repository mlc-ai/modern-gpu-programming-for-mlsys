(chap_tirx_primer)=
# TIRx 入门

:::{admonition} 概览
:class: overview

- TIRx 是一个用于在 IR 层编写 GPU kernel 的 Python DSL：你会直接命名硬件，但通过结构化 IR 来表达。
- 每个 tile 操作都由三个设计要素控制：*scope*（哪些线程执行）、*layout*（tile 位于哪里）和 *dispatch*（走哪条硬件路径）。
- 一个可运行的 single-MMA GEMM 会同时展示这三者；本书后续内容就是把这些设计要素放大到真实规模。
:::

:::{admonition} 运行示例
:class: note

这些示例需要 Blackwell GPU（`sm_100a`，例如 B200）。TIRx 编译器作为 Apache TVM wheel 中的 `tvm.tirx` 模块发布；请和 CUDA 版本的 PyTorch 一起安装：

```bash
pip install apache-tvm
```

用 `python -c "import tvm, tvm.tirx; print(tvm.__version__)"` 确认它可以 import。同样的环境可以运行本书所有可运行示例。
:::

第一部分解释了硬件是什么。要让硬件真正计算，我们还需要一种编程方式。

我们可以直接写 CUDA 或 PTX，很多快速 kernel 也确实是这样写的。问题在于，真正决定 kernel 行为的决策在那里很难看清：哪些线程执行某个操作、每个数据 tile 位于哪里、以及由哪条硬件路径执行。这些选择会埋在 intrinsic 参数、地址计算和约定之中。

TIRx（Tensor IR neXt）是一个 Python DSL，它把这三个决策明确提升出来：**scope**（哪些线程执行操作）、**layout**（operand tile 位于哪里）和 **dispatch**（使用哪条硬件路径执行）。它仍然直接命名硬件概念，包括线程、shared memory、tensor memory、barrier 和 `tcgen05` MMA。区别是，这些选择现在变成结构化 IR，编译器可以 lower、检查和调度。

我们不会先抽象地介绍这些概念，而是从一个完整 kernel 开始：最小 single-MMA GEMM。我们先让它跑起来，然后逐行读回去，看 scope、layout 和 dispatch 分别如何塑造它，以及 kernel 如何被编译。Kernel 依赖的 tensor layout 模型会在 {ref}`chap_tirx_layout_api` 中单独展开，完整语言特性集合在 {ref}`chap_language_reference` 中介绍；这里我们聚焦这一个 kernel 和三个设计要素。

## 第一个 Kernel：Single-MMA GEMM

我们承诺的 kernel 是一个最小 GEMM，删减到仍然能使用 Tensor Core 的最小版本。它计算 `D = A B^T` 的单个 128 x 128 output tile，K = 64。整个计算从头到尾被表达成一次 `Tx.gemm_async` tile operation。（这一个 tile operation 并不映射到单条硬件指令：因为硬件 MMA 的 K atom 是 16，K=64 的 tile 会 lower 成沿 K 前进的一小段 `tcgen05.mma` 指令序列。DSL 的重点正是我们写 tile，而不是手写这段序列。）在这个操作周围，kernel 做常规工作：分配 shared memory（SMEM）和 tensor memory（TMEM），把 A 和 B 从 global copy 到 shared memory，发起 tile MMA 并把结果写入 TMEM accumulator，再把 accumulator 通过寄存器读回并 store 结果。虽然它很小，但这个 kernel 就是 {ref}`chap_gemm_basics` 中 GEMM 阶梯的 Step 1，那里会完整讲解它。

每个 TIRx kernel 都从同一组 import 开始，所以值得先看一次：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

我们把 kernel 包在一个小 builder `hgemm_v1(M, N, K)` 中，它接受问题 shape 并返回一个 `PrimFunc`。对于我们选择的 shape，`M=N=128, K=64`，launch 中刚好只有一个 output tile，这让第一个版本足够简单，可以一次读完：

```python
def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    # MMA_M/MMA_N/MMA_K document the underlying hardware MMA tile; they are not
    # passed to gemm_async (which derives the MMA shape from the operand and
    # accumulator tiles), so the later steps omit them.
    MMA_M, MMA_N, MMA_K = 128, 128, 16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # Step 1 is a single-tile kernel: M = BLK_M and N = BLK_N, so the grid
        # is 1x1. Starting with a 1x1 grid keeps the per-CTA tile offsets
        # (m_st, n_st) trivially zero; Steps 3+ generalise this to larger M / N.
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])      # single warpgroup, so wg_id is always 0 (unused below)
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        # --- Barrier + TMEM init (warp 0 only) ---
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_mma: T.int32 = 0

        # --- Load: all threads copy global -> shared (synchronous).
        # With M=BLK_M and N=BLK_N the slices below cover the full matrices;
        # the slice form is kept so the diff to Step 3 (multi-tile) is minimal.
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()

        # --- Compute: single elected thread issues MMA ---
        if warp_id == 0:
            if T.ptx.elect_sync():
                Tx.gemm_async(
                    tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                    accum=False, dispatch="tcgen05", cta_group=1
                )
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)

        # --- Writeback: TMEM -> RF -> GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

        # --- Deallocate TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

在读这个 kernel 之前，我们先确认它能工作。我们编译它，并用 torch reference 检查输出。这里不需要显式写出具体架构：arch（例如 `sm_100a`）会从设备自动检测，所以 target `"cuda"` 就足够了，`tir_pipeline="tirx"` 选择 TIRx lowering pipeline。编译完成后，`ex.mod(...)` 可以直接接受 torch tensor，中间不需要手动转换。

```python
import torch

target = tvm.target.Target("cuda")
device = torch.device('cuda')  # gpu(0)

M, N, K = 128, 128, 64
kernel = hgemm_v1(M, N, K)
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

torch.cuda.empty_cache()
torch.cuda.synchronize()
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

# ex.mod(...) takes torch tensors directly, the same call form used in every chapter.
ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")
```

## Scope、Layout、Dispatch

现在 kernel 已经能跑，我们可以回过头来读它，问每一行到底决定了什么。从这个角度看，整个 kernel 是围绕三个设计要素的一组选择。里面的每个操作都回答同样三个问题：*谁*执行它、它的数据*在哪里*、它*如何*执行；这三个答案就是 scope、layout 和 dispatch。本节会依次讨论这些设计要素；下面的交互 demo 可以看到每个设计要素控制了 kernel 的哪些行。

```{raw} html
<iframe src="../demo_zh/tirx_dispatch.html" title="TIRx: scope, layout, dispatch" loading="lazy"
        style="width:100%; min-width:960px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互图：点击 Scope / Layout / Dispatch，高亮 kernel 中由每个设计要素控制的行。*

使用 demo 时，请关注三个问题：

- **Scope：谁执行这个操作？** `Tx.cta.copy(...)` 是 CTA-scoped，因此全部 128 个线程都会帮助完成 GMEM -> SMEM copy。`Tx.gemm_async(...)` 由一个 elected thread 发起，因为 lowering 后的每条 `tcgen05.mma` 指令本身已经是一次 cooperative MMA launch。`Tx.wg.copy_async(...)` 是 warpgroup-scoped，因此 warpgroup 的 128 个线程会按行拆分 TMEM readback。
- **Layout：每个 tile 位于哪里？** A 和 B 使用 `tcgen05.mma` 期望的 swizzled SMEM layout。Accumulator 位于 TMEM 中，使用 `TLane`/`TCol` 布局。Register readback view 把 row 映射到 `tid_in_wg`，因此每个 warpgroup thread 拥有一个 row fragment。
- **Dispatch：哪条硬件路径执行它？** `Tx.gemm_async(..., dispatch="tcgen05", ...)` 选择 Blackwell Tensor Core 路径。Copy 操作也有 dispatch 选择：第一个 kernel 使用普通 thread copy，后续 GEMM step 会把这些 copy 换成 TMA，而不改变周围的 scope 或 layout。

**Try with your agent**：从第一个 kernel 里挑三行：一个 copy、一个 MMA、一个 TMEM readback。让它用 scope、layout 和 dispatch 标注每一行，然后检查答案是否与 guard、buffer layout 和 `dispatch=` 参数一致。

## 编译如何工作

我们已经在上面编译过 kernel 来测试它；现在稍微近距离看一下这个步骤做了什么。流程很短：把 `PrimFunc` 包进一个 `IRModule`，再交给 `tvm.compile(mod, target=..., tir_pipeline="tirx")`。这会运行 TIRx lowering pipeline，并返回一个可以直接调用的 `Executable`。

```python
target = tvm.target.Target("cuda")
ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
```

至少大致知道 `tir_pipeline="tirx"` 会触发什么，是有帮助的。Pipeline 的核心 pass 是 `LowerTIRx`，它会根据每个 tile primitive 的 scope / layout / dispatch contract 进行解析：这里正是我们刚才讨论的三个设计要素真正兑换成指令的地方。之后，常规 host/device split 和 finalize step 会产生可 launch 的 module。如果愿意，也可以在 `with target:` block 内编译，这样 kernel 可以获取外层 target context。

这个流程的一个好处是没有东西对你隐藏：结果可以在两个层级上检查。你可以用 `.show()` 或 `.script()` 读取 IR 本身，也可以从编译后的 module 中直接查看编译器最终生成的 CUDA C。

```python
kernel.show()                          # pretty-print the TIRx (TVMScript)
print(kernel.script())                 # ... the same, as a string

# the generated CUDA C source, from the compiled Executable:
print(ex.mod.imports[0].inspect_source())
```

这里只是一个概览。完整 lowering 过程，包括所有 pass、tile-primitive dispatch 如何解析，以及 host/device split 如何完成，请见 {ref}`chap_arch`。

## 下一步

一个 kernel 已经足够让我们认识 scope、layout 和 dispatch，并看到它们如何被编译和运行。这三个设计要素以及 kernel 本身，分别通向后续章节：

- {ref}`chap_tirx_layout_api`：tensor layout 模型（`TileLayout`、命名轴、swizzle），上面 operand 和 accumulator 的 placement 都建立在它之上。如果 layout 这个设计要素最让你困惑，可以从这里继续。
- {ref}`chap_language_reference`：完整语言特性集合，包括 parser utility、data type、buffer 和 memory、control flow，以及 thread synchronization；当你需要完整词汇表而不是导览时可以查这里。
- {ref}`chap_gemm_basics`：这个 kernel 作为 GEMM 优化路径的 Step 1，并进一步加入 K-loop accumulation、spatial tiling、TMA 和 warp specialization。如果你想看同样三个设计要素如何扩展到真实 kernel，这是自然的下一站。
