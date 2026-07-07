(chap_gemm_basics)=
# 构建 Tiled GEMM

:::{admonition} 概览
:class: overview

- 从 TIRx tile primitive 出发，构建一个正确的 tiled GEMM，起点是单个 output tile。
- Step 1 是 single-tile GEMM，Step 2 加入 K-loop accumulation，Step 3 在 CTA 之间做 spatial tiling 以覆盖完整矩阵。
- 先保证正确性；性能优化交给接下来的两章。
:::

GEMM 是本书围绕构建的核心 workload。它位于 linear layer、attention projection 和 convolution 的底层，而这些操作占据了 GPU 的大部分时间。因此，一个只“正确”的 GEMM 和一个“快速”的 GEMM 之间的差别，往往就是芯片大部分算力空转和接近饱和之间的差别。

这个差距太大，无法一步跨过去。一个接近饱和的 kernel 会让你同时调试内存搬运、accumulation、tiling 和 Tensor Core scheduling，而且一开始还没有可信的 baseline 可以对照。更稳妥的路径是从能产生正确答案的最小 kernel 开始，然后一次只增加一个设计决策。

本章会写出第一个正确的 tiled GEMM。前面章节以抽象形式介绍了 TIRx 的 scope / layout / dispatch 模型；这里我们把它应用到真实 kernel 上。我们从一个 128 x 128 output tile 开始，然后把它扩展成可以处理完整矩阵的 kernel：先加入 K 维度 accumulation，再加入跨多个 CTA 的 spatial tiling。

这是三章 GEMM 优化路径中的第一章。这条路径会从头到尾走完一个 GEMM kernel 的演化。本章只构建正确的 tiled kernel 并到此为止。下一章（{ref}`chap_gemm_async`）会把 thread copy 换成 TMA，并通过 pipelining 让数据搬运和计算重叠；{ref}`chap_gemm_advanced` 会进一步加入 warp specialization 和 CTA cluster。每章都建立在前一章之上，因此 kernel 会逐步积累功能，而不是每章重新开始。

阅读每个 step 时，可以把它看作对同一份三项 contract 的一次编辑：哪个 **scope** 执行操作，operand tile 使用哪个 **layout**，以及通过哪条 **dispatch** 路径执行。多数 step 都只有一个主要变化，因此我们会先用一个小卡片指出变化是什么，并标出让复用安全所需的同步细节。Step 1 会建立后续所有修改的 baseline。

## GEMM

GEMM 是 dense matrix multiply，位于 linear layer、attention projection 和许多 convolution 实现的底层，所以快速 GEMM kernel 几乎在任何地方都有收益。本教程中的例子使用 $D = A B^{\top}$：

- $A$ 的 shape 是 $M \times K$。
- $B$ 的 shape 是 $N \times K$。
- $D$ 的 shape 是 $M \times N$。
- $D[m,n] = \sum_k A[m,k] \cdot B[n,k]$。

这里的 transpose 不是我们额外选择执行的操作；它来自数据的存储方式。示例保持 $B$ 为 $N$ 行、每行长度 $K$，这也是 linear-layer weight 通常使用的布局。因此沿 $K$ contraction 时，自然就是在读取 $B^{\top}$，不需要实际重排。

整个教程中，我们用 TFLOPS 衡量 kernel 的吞吐，把每次 multiply-add 计为两个 floating-point operation，并除以 wall-clock time：

$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{t_{\text{seconds}} \times 10^{12}}$$

### GEMM 数据路径

本教程中的每个优化最终都归结为数据位于哪里、如何移动。因此在写代码前，先把这条路径画出来是值得的。一个 Blackwell GEMM kernel 的核心只有两类活动：在不同 memory 之间搬运 tile，以及在 tile 上计算。下图追踪一个 tile 从输入到输出会触碰的每一种 memory：

![*Memory Data Flow*](../../img/memory_dataflow.png)

上图展示了 baseline 路径。之后每个优化都会编辑这条路径，但不会替换它。从左到右读：operand tile 先从 GMEM 移到 SMEM；随后 `tcgen05.mma` 消费 SMEM operand，并把 accumulator 写入 TMEM；最后 epilogue 把 TMEM 读回寄存器，再把结果 store 到 GMEM。请记住这条链路，因为下面每一步都会改变其中某一跳*如何*发生，但不会改变这些跳本身。

## 优化路径

上面的朴素数据路径已经足以得到正确答案，但会让大部分硬件空闲。教程剩余部分会一次加入一个 Blackwell 特性来缩小这个差距，每个特性都通过 TIRx tile primitive 表达。我们将依次经过这些特性：

- **TMA async movement** 通过 Blackwell 的硬件 copy 路径移动 GMEM <-> SMEM tile，并用 barrier 跟踪完成。
- **Software pipelining** 使用多个 SMEM stage，让下一块 K tile 的数据搬运可以与当前 tile 上的 Tensor Core compute 重叠。
- **Persistent scheduling** 保持一组固定 CTA，让每个 CTA 通过 tile scheduler 处理多个 output tile，而不是每个 tile launch 一个 CTA。
- **Warp specialization** 把 producer、MMA consumer 和 writeback 角色拆分到不同 warpgroup 上。
- **CTA clusters** 让两个 CTA 协作处理一个更大的 Blackwell MMA tile。
- **Multi-consumer execution** 使用多个 consumer warpgroup 同时计算 tile 的不同部分，提高 compute density。

---

(chap_single_tile)=
## Step 1：顺序 Single-Tile GEMM

仍然能覆盖完整硬件路径的最简单 GEMM，是计算单个 output tile 的 GEMM。因此我们从这里开始。Step 1 计算一个 128 x 128 output tile，K = 64；这个规模小到不需要 loop，并且数据路径中的每个部分都只出现一次。没有重复结构时，我们可以先单独看清每一跳，然后再开始推理循环。

> **本 step 建立的内容：baseline**
> - Scope：一个 128 线程的 single warpgroup 按顺序走完整条路径，一阶段接一阶段。
> - Layout：A 和 B tile 位于 SMEM，accumulator 位于 TMEM，结果通过寄存器 staged out。
> - Dispatch：同步 `Tx.copy` 执行 load，`tcgen05` 执行 MMA。

### Single-Tile Dataflow

Baseline contract 固定后，下一件事是确定一个 tile 按什么顺序穿过它。第一个 kernel 会完整走一次核心 GEMM 数据路径，也就是 data-flow 图里的同一条 GMEM -> SMEM -> TMEM -> registers -> GMEM 链路，外面没有包任何 loop。它分配工作内存、加载 operand、计算乘积、写回结果，并清理自己使用的资源：

1. **Allocate**：SMEM（pool allocator）、TMEM（`tcgen05.alloc`）、mbarrier
2. **Load**：全部 128 个线程协作把 A 和 B tile 从 GMEM copy 到 SMEM（sync `Tx.copy`）
3. **Compute**：一个 elected thread 发起 `Tx.gemm_async` + `tcgen05.commit`；所有线程在 mbarrier 上等待
4. **Writeback**：Warpgroup 读取 TMEM -> registers；每个线程把 fp32 cast 到 fp16 并写入 GMEM
5. **Deallocate**：释放 TMEM

### 第一个 Kernel 的四个部分

完整 kernel 只有几十行，但分段读更容易消化。我们会按四部分阅读它：memory allocation、同步 load、MMA dispatch 和 writeback；之后再把它们拼成一个 kernel。沿途出现的 API 名称，是第二部分介绍过的 TIRx tile-primitive 词汇（{ref}`chap_tirx_primer`、{ref}`chap_tirx_layout_api`）。

**Memory allocation。** Kernel 首先从 shared memory 中切出 operand 所需空间，以及 TMEM address 和 mbarrier 的位置：

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")           # TMEM address (4 bytes)
mma_bar = pool.alloc((1,), "uint64", align=8)    # mbarrier (8 bytes)
pool.move_base_to(1024)                           # Skip to offset 1024
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)  # 128×64 fp16
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)  # 128×64 fp16
pool.commit()
```

这里有两个细节值得停一下。`pool.move_base_to(1024)` 会把 Asmem 和 Bsmem 推到 offset 1024，给前面较小的 metadata 预留低地址区域，使 bulky operand tile 落在干净边界上。`layout=A_layout` 会让 `tma_shared_layout` 提供一个 swizzled SMEM placement，这个 placement 能被 TMA 和 `tcgen05.mma` 直接读取，正是第二部分所说的 layout-as-contract。

**Synchronous load。** Buffer 到位后，operand 还需要到达 SMEM。在第一个版本中，我们让 CTA 自己的线程执行 copy：

```python
Tx.cta.copy(Asmem[:, :], A[:, :])
Tx.cta.copy(Bsmem[:, :], B[:, :])
T.cuda.cta_sync()
```

因为这里总共只有一个 tile（M=N=128, K=64），copy 完整 A 和 B 就是整个 load。`Tx.cta.copy(...)` 让 CTA 在这次 copy 上协作，每个线程负责自己那一片数据。后面的 `T.cuda.cta_sync()` 有双重作用：它等待每个线程完成，并发布这些线程对 shared memory 的写入，因此后续 MMA 读取 `Asmem` 和 `Bsmem` 时看到的是完整 tile，而不是半填充 buffer。这个 thread-driven copy 也是我们首先会替换的东西；下一章（{ref}`chap_gemm_async`）会把它换成 TMA。

**MMA dispatch。** Operand 已经位于 SMEM 中，现在可以发起 MMA，并且由一个 elected thread 来做：

```python
if warp_id == 0:
    if T.ptx.elect_sync():
        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                      accum=False, dispatch="tcgen05", cta_group=1)
        T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
```

两层 guard 会分两步把 issuer 缩小到一个线程。外层 `if warp_id == 0` 只保留 warpgroup 中的 warp 0，内层 `if T.ptx.elect_sync():` 再从这个 warp 的 active lane 中选出一个。合起来，只剩一个线程执行 `Tx.gemm_async` 和 `tcgen05.commit`。

这里需要明确说明单个线程意味着什么、不意味着什么，因为直觉读法很容易误导。单个 issuing thread 并不意味着单线程矩阵乘法。计算仍然是完整的 tile-level MMA：硬件会根据 SMEM operand layout 和 TMEM accumulator layout 描述的 tile 执行协作矩阵乘法。关键在于 `Tx.gemm_async` 是一个 *tile operation*，不是一条硬件指令。K = 64 tile 比硬件 MMA K-atom（`MMA_K = 16`）更宽，所以这个 tile op 会 lower 成沿 K 前进的一小段 raw `tcgen05.mma` 指令序列，而 warpgroup 会协作驱动每一条。之所以只有一个线程发起 tile op，是因为底层每条 `tcgen05.mma` 本身就是一条 cooperative op：一次 launch 驱动这个 K-atom 的 tile MMA。如果 128 个线程都发起同一段序列，同样的工作就会被 launch 128 次。最后，`accum=False` 告诉 MMA 覆盖 TMEM destination，而不是加到已有值上；这里没有先前 partial sum，所以这正是我们想要的。

**Writeback。** 乘积现在位于 TMEM 中，但调用方希望在 GMEM 中得到 fp16 结果。因此 epilogue 必须先把结果通过寄存器带下来，并在途中 cast：

```python
Dreg = T.alloc_local((BLK_N,), acc_type)        # per-thread fp32 register row
Dreg_f16 = T.alloc_local((BLK_N,), d_type)      # same row, cast to fp16
Dreg_wg = Dreg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
Tx.cast(Dreg_f16[:], Dreg[:])
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

MMA 会在 TMEM 中留下一个 128 x 128 fp32 accumulator tile。使用 fp32 是有意的：GEMM 会沿 K 累加很多乘积，用更高精度保存 running sum 可以降低累积的 rounding error。但 `D` 是 fp16，所以这些值不能直接写出。它们先进入寄存器，在那里 narrow 到 fp16，然后才到达 GMEM。

两个 register buffer 作用不同。`Dreg` 是每个线程自己的 `BLK_N` 元素 buffer，而 `Dreg_wg` 是同一组寄存器在所选 layout 下的 warpgroup-wide *view*：

```python
TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)])
```

这个 layout 把 tile 的第一维映射到 warpgroup 的线程上：thread 0 拥有 row 0，thread 1 拥有 row 1，一直到 row 127。第二维留在每个线程自己的 register buffer 中，因此单个线程持有自己那一行的所有列。Warpgroup 有 128 个线程，tile 有 128 行，所以 128 x 128 输出刚好分成每个线程一行。

在这个 view 下读取 accumulator，正是 `Tx.wg.copy_async(Dreg_wg, tmem)` 所做的事，它会 lower 到 Blackwell TMEM load 路径 `tcgen05.ld`。由于这个 load 是异步的，任何线程触碰 `Dreg` 之前都必须先完成 `T.ptx.tcgen05.wait.ld()`；否则线程可能读取尚未被 load 填好的寄存器。

Wait 返回后，每个线程私有的 `Dreg[:]` 保存自己那一条逻辑输出行的 fp32 值。线程把它们 narrow 到 `Dreg_f16` 中，计算自己负责的 global row：

```python
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
```

然后写入 `D[m_thr, n_st:n_st + BLK_N]`。这些 row 在四个 warp 之间整齐切分：warp 0 写 rows 0-31，warp 1 写 rows 32-63，warp 2 写 rows 64-95，warp 3 写 rows 96-127。

### 完整 Kernel

现在把四个部分拼回一个可运行 kernel（M=N=128, K=64）。Import 先出现：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

Kernel 包在后续 step 都会使用的 `hgemm_vX(M, N, K)` 风格中。Step 1 使用 `M=N=128, K=64`，因此 launch 中刚好有一个 output tile：

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

后面的每个 GEMM step 都会用同样方式编译、运行并检查自己，因此这个 scaffolding 只完整写一次。从此之后，我们只展示 kernel。要运行后续 step，把下面的 `hgemm_vX` 和匹配的问题规模换成对应版本即可。有一个注意事项：每次新的 Python session 只编译一个 step，尝试另一个 step 前先重启，因为这些示例会复用内部名字，而编译器持有 per-session state。

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
# Relative tolerance, like the warp-specialization and Flash Attention cells:
# output magnitude grows with K, so a fixed absolute bound would fail at larger K.
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")

# Optional timing for larger kernels.
ITERS = 10
for _ in range(3):
    ex.mod(A_tensor, B_tensor, D_tensor)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(ITERS):
    ex.mod(A_tensor, B_tensor, D_tensor)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / ITERS
tflops = 2 * M * N * K / ms / 1e9
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

Steps 1 到 3 会刻意使用较小规模（这里是 128×128，Step 3 是 256³），使最初几个 walkthrough 容易跟上。{ref}`chap_gemm_advanced` 末尾的跨 step *End-to-End Result* 表则采用相反策略：它把每个 step，包括这个 Step 1 算法，都放到统一的 M=N=K=4096 规模下测量，因此 speedup ratio 可以直接比较。

### Single-Tile Kernel 的限制

这个 kernel 是正确的，这正是 Step 1 的目标，但它只在非常窄的设定下正确。这里有四个限制是有意留下的，后续优化路径会逐个解除它们：

- 它只处理单个 K tile，因此不能对很大的 K 做 contraction。
- 它只处理单个 output tile，因此 M 和 N 被固定在 128。
- 它使用同步 GMEM -> SMEM copy，而不是 TMA。
- 它没有重叠数据搬运和计算，所以两者不会同时运行。

---

(chap_k_loop)=
## Step 2：K-Loop Accumulation

第一个要移除的是最小的限制。Step 1 只处理单个宽度为 64 的 K tile，但真实矩阵会沿远大于 64 的 K 做 contraction。在 Step 2 中，我们仍然只计算单个 output tile，但允许 K 跨越多个 64-wide chunk。

想法很直接：对每个 chunk 重复 load -> MMA -> wait 序列，并让每次 MMA 累加到同一个 TMEM slot 中。真正需要小心的地方是同步。跨 iteration 复用同一个 mbarrier，会引入本章第一个真实 correctness hazard。如果代码跟踪了错误 phase，某次 wait 可能在对应 MMA 真正完成*之前*返回，静默破坏结果。下面的机制会精确说明这个错误如何发生，以及如何避免。

> **本 step 改变的内容：Layout reuse**
> - Scope：不变，仍然是 single warpgroup。
> - Layout/reuse：同一对 SMEM tile 和同一个 TMEM accumulator slot 会在 K-loop 中复用。不分配新 storage；operand tile 流经固定的一对 buffer，accumulator state 保持在一个 TMEM slot 中。
> - Synchronization：复用的 MMA barrier 必须在每个 K chunk 上推进到正确 phase，否则后续 wait 可能观察到更早的 completion。
> - Dispatch：不变。

### K-Loop 机制

Step 1 只 contraction 了单个 64-wide K tile；这里我们保留它的 single output tile，但让 K 按矩阵需要的长度前进。为了覆盖大于 64 的 K，我们以 `BLK_K=64` 为 chunk 沿 K 迭代。每次 iteration 加载下一段 A 和 B 的 K-slice 到 SMEM，并发起 `Tx.gemm_async`。`accum` flag 把这些 chunk 拼成同一个 dot product：第一个 chunk 上 `accum=False` 初始化 TMEM accumulator，之后每个 chunk 上 `accum=True` 把该 chunk 的乘积加到 TMEM 中已经存在的 running sum 上。

同步是需要谨慎的地方。我们为每次 MMA completion 复用同一个 mbarrier，而安全复用的关键是跟踪正在等待哪个 barrier phase。一个 mbarrier 带有 1-bit phase，可以是 0 或 1；每当期望的 arrival 到达，它就翻转到另一个值。微妙之处在于 wait 条件本身：`try_wait(bar, phase)` 会阻塞，直到 barrier 的内部 phase *不同于* `phase` 参数。因此我们传入的参数必须命名我们期望离开的 phase，而不是等待抵达的 phase：

| K iteration | Wait 前本地 `phase_mma` | `try_wait` 等待什么 | Wait 后本地更新 |
|---|---:|---|---:|
| 0 | 0 | barrier flips to 1 | `phase_mma = 1` |
| 1 | 1 | barrier flips to 0 | `phase_mma = 0` |
| 2 | 0 | barrier flips to 1 | `phase_mma = 1` |

`phase_mma ^= 1` 这一行正是保持这张表正确的原因。去掉它后，第二次 iteration 仍然调用 `try_wait(bar, 0)`，但 barrier 在第一次 MMA 后已经翻到了 phase 1，因此 wait 看到 mismatch 就立即返回，而此时第二次 MMA 还没完成。Kernel 随后会读取半计算的 accumulator，并在没有任何 error 的情况下给出错误答案。这个 bug 可以完美编译和运行，这就是 phase flip 值得如此强调的原因。

### 完整 Kernel

下面的完整 kernel 只是 Step 1 加入 K-loop 和 phase flip。Import 和之前相同：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

它包在 `hgemm_v2(M, N, K)` 中。Grid 仍然是 `[1, 1]`，因为我们仍然只计算单个 output tile；增长的只是 K extent：

```python
def hgemm_v2(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])  # still one output tile (M=N=128)
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # === K-loop: iterate over K in chunks of BLK_K ===
        for i in T.serial(K_TILES):   # serial device loop (keeps the full-K A/B parameters correctly shaped)
            # Load the i-th K chunk
            Tx.cta.copy(Asmem[:, :], A[:, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[:, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            # MMA: accum=False for first tile, True for rest
            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            # Wait for MMA, then flip phase
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # === Writeback (same as Step 1) ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

---

(chap_spatial_tiling)=
## Step 3：Spatial Tiling（Multi-CTA）

K-loop 处理了 contraction dimension，但 M 和 N 仍然固定在单个 128 x 128 tile 上。真实输出远大于一个 tile，所以基础 kernel 的最后一块是用多个 tile 同时覆盖 M 和 N。Step 3 会 launch 一个 2D CTA grid，每个 output tile 一个 CTA，让 GPU 并行计算所有 tile。示例使用 M=N=K=256，对应 2x2 tile grid，刚好能让 indexing 不再平凡，又不至于把重点埋掉。

> **本 step 改变的内容：Scope**
> - Scope：一个 2D CTA grid，每个 CTA 拥有一个 128 x 128 output tile。
> - Layout：不变；在每个 CTA 内部，这仍然是 Step 2 的同一条 SMEM/TMEM/register 路径。
> - Dispatch：不变。

### Grid Mapping

Grid shape 直接来自 tiling：每个 128 x 128 output tile 一个 CTA，所以总共需要 `[M // BLK_M, N // BLK_N]` 个 CTA。相对 Step 2，唯一真正新增的工作，是让每个 CTA 知道矩阵中哪一片是*自己*要计算的 slice。

CTA `(bx, by)` 拥有这个 output region：

```text
D[bx * BLK_M : (bx + 1) * BLK_M,
  by * BLK_N : (by + 1) * BLK_N]
```

为了产生它，该 CTA 的 K-loop 会反复加载自己 A row band 和 B column band 对应的 K-slice：

```text
A[bx * BLK_M : (bx + 1) * BLK_M, k : k + BLK_K]
B[by * BLK_N : (by + 1) * BLK_N, k : k + BLK_K]
```

Indexing 直接来自 `D = A @ B.T` 约定：`bx` 选择 A 和 D 的行，而 `by` 选择 B 的行；transpose 应用之后，这些 B 行会变成 D 的列。

每个 CTA 一个 tile 是最简单可行的映射，但它也浪费。Row 中的每个 CTA 都会从 GMEM 重新加载同样的 A tile，column 中的每个 CTA 都会重新加载同样的 B tile，因此没有复用邻近 CTA 已经拉进来的数据。我们暂时保留这个浪费；persistent scheduling（{ref}`chap_gemm_async` 中的 Step 6）会回到这个问题，并让这些共享 operand 在 L2 中保持 hot。

**Try with your agent**：设 `M=N=K=256`、`BLK_M=BLK_N=128`、`BLK_K=64`，让它 trace CTA `(1, 0)` 和 CTA `(0, 1)`。对每个 CTA，列出 `m_st`、`n_st`、每次 K iteration 加载的 A/B slice，以及写入的 D region。哪些 B row 因为 kernel 计算 `D = A @ B.T` 而变成 D column？

### 完整 Kernel

这个 kernel 再次从 Step 2 发展而来，这次只有两个变化：grid shape 和 per-CTA offset。内部 K-loop 和 writeback 不变。Import 相同：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

Grid 从 `[1, 1]` 变成 `[M // BLK_M, N // BLK_N]`，load 和 store 现在都会加上 CTA 自己的 `m_st` 和 `n_st` offset：

```python
def hgemm_v3(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 2D grid: one CTA per 128x128 output tile
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0

        # Per-CTA tile offsets
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # K-loop with offset A and B slices
        for i in T.serial(K_TILES):   # serial device loop (keeps the full-K A/B parameters correctly shaped)
            Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # Writeback to the correct output tile
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st:n_st+BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

## 练习

1. 在 Steps 1-3 中，`Tx.copy` 会在 MMA 之前把 A 和 B tile 移入 SMEM。为什么 kernel 需要在 `Tx.gemm_async` 读取这些 SMEM tile 前执行 `T.cuda.cta_sync()`？
2. 在 Step 2 中，如果从 K-loop 中移除 `phase_mma ^= 1` 会发生什么？Kernel 是否会等待每一次 MMA，还是后续 wait 可能过早通过？
3. 对于 M=N=4096 且 BLK_M=BLK_N=128 的情况，Step 3 会 launch 多少个 CTA？哪些 operand tile 在逻辑上会被相邻 CTA 复用？Step 3 是否利用了这种复用？
