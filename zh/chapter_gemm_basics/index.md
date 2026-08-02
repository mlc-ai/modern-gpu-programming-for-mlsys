(chap_gemm_basics)=
# 构建 Tiled GEMM

:::{admonition} 本章概览
:class: overview

- 本章使用 TIRx tile primitives，从一个输出 tile 开始构建 tiled GEMM。
- 第 1 步完成单个 tile 的计算，第 2 步沿 K 维循环累加，第 3 步将完整输出矩阵划分为多个 tiles，并交给不同的 CTAs 计算。
- 本章先保证结果正确，后两章再逐步优化性能。
:::

GEMM 是本书后续章节反复使用的核心计算。Linear layer、attention projection 和许多 convolution 实现都以矩阵乘法为基础，而这些运算通常占据 GPU 的大部分执行时间。要进一步优化 GEMM，首先需要一个结果正确、结构清楚的基线 kernel。

如果一开始就同时加入数据搬运、K 维累加、tiling 和 Tensor Core 调度，一旦结果出错，很难判断问题来自哪一步。因此，本章采用逐步扩展的方式：每个版本只增加一个主要机制，并保留上一版作为对照。

我们先让一个 CTA 计算一个 `128×128` 输出 tile，再加入 K-loop 完成 K 维归约，最后沿 M、N 维划分输出矩阵，让多个 CTAs 共同覆盖整个问题。到本章结束时，kernel 已经可以处理完整矩阵，但暂不追求性能。

这些步骤也会把前面介绍的 TIRx 模型落实到具体代码中。阅读时可以关注三个问题：操作由哪个 **scope** 执行，operand tile 采用什么 **layout**，以及 tile operation 最终通过哪条 **dispatch** 路径执行。后两章会继续在这个基线 kernel 上加入异步搬运、流水线和其他性能优化。

## GEMM

GEMM 是稠密矩阵乘法，也是 linear layer、attention projection 和许多 convolution 实现的基础。本章使用下面的形式：

- $A$ 的 shape 为 $M \times K$。
- $B$ 的 shape 为 $N \times K$。
- $D$ 的 shape 为 $M \times N$。
- $D[m,n] = \sum_k A[m,k] \cdot B[n,k]$.

这里将 $B$ 按 $N \times K$ 存储，这是 linear-layer weights 常见的存储方式。计算时直接读取 $B[n,k]$；若写成矩阵形式，等价于 $D=AB^{\top}$，但 kernel 不会额外转置或重排 $B$。

本章使用 TFLOPS 衡量 kernel throughput。一次 multiply-add 计作两次浮点运算，因此：

$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{t_{\text{seconds}} \times 10^{12}}$$

### GEMM 的数据路径

后面的每项优化都与数据存放在哪里、如何移动有关，因此先看 Blackwell GEMM 的基本数据路径。Kernel 主要完成两类工作：在不同 memory space 之间搬运 tiles，以及使用这些 tiles 进行计算。下图展示了数据从输入到输出依次经过的 memory space：

![*Memory 数据流*](../../img/memory_dataflow.png)

从左向右看：operand tiles 先从 GMEM 进入 SMEM；`tcgen05.mma` 读取 SMEM 中的 operands，并把 accumulator 写入 TMEM；最后，epilogue 将 TMEM 中的结果读入 registers，再写回 GMEM。后续优化会改变其中某一步如何执行，但不会改变这条基本路径。

## 优化路线

这条基础数据路径足以得到正确结果，但还不能充分利用硬件。接下来会通过 TIRx tile primitives 依次加入以下机制：

- **TMA 异步搬运**：使用 Blackwell 的硬件 copy path 在 GMEM 与 SMEM 之间搬运 tiles，并通过 barrier 跟踪完成状态。
- **Software pipeline**：使用多个 SMEM stages，让下一块 K tile 的数据搬运与当前 tile 的 Tensor Core 计算重叠。
- **Persistent scheduling**：不再为每个 output tile 启动一个 CTA，而是让固定数量的 CTAs 通过 tile scheduler 反复处理多个 tiles。
- **Warp specialization**：把 producer、MMA consumer 和 writeback 分配给不同 warpgroups。
- **CTA cluster**：让两个 CTAs 协作计算一个更大的 Blackwell MMA tile。
- **Multi-consumer execution**：让多个 consumer warpgroups 同时计算 tile 的不同部分，提高计算密度。

---

(chap_single_tile)=
## 第 1 步：顺序执行的单 Tile GEMM

第 1 步沿用“TIRx 入门”中的 `hgemm_v1`，详细拆解其数据路径，并将它作为后续版本的正确性基线。这个 kernel 只计算一个 `128×128` output tile，并取 `K=64`；该规模不需要循环，数据路径中的每一步只出现一次，便于逐段理解。

> **这一步建立基线**
> - Scope：一个包含 128 个 threads 的 warpgroup 按顺序执行整条数据路径。
> - Layout：A、B tiles 位于 SMEM，accumulator 位于 TMEM，结果通过 registers 写出。
> - Dispatch：同步 `Tx.copy` 负责加载，`tcgen05` 执行 MMA。

### 单 Tile 数据流

这个 kernel 只沿 `GMEM -> SMEM -> TMEM -> registers -> GMEM` 路径执行一次，不包含循环。具体步骤如下：

1. **分配**：通过 pool allocator 分配 SMEM，通过 `tcgen05.alloc` 分配 TMEM，并准备 mbarrier。
2. **加载**：128 个 threads 使用同步 `Tx.copy`，协作将 A、B tiles 从 GMEM 搬到 SMEM。
3. **计算**：选出的一个 thread 发出 `Tx.gemm_async` 和 `tcgen05.commit`，所有 threads 等待 mbarrier。
4. **写回**：warpgroup 将 TMEM 读入 registers；每个 thread 把 fp32 转成 fp16，再写入 GMEM。
5. **释放**：释放 TMEM。

### Kernel 的四个部分

下面先分别介绍存储空间分配、operand 加载、MMA 发起和结果写回，再把它们组合成完整 kernel。相关 API 已在第二部分（{ref}`chap_tirx_primer`、{ref}`chap_tirx_layout_api`）中介绍。

**分配存储空间。** Kernel 先为 operands 分配 shared memory，并为 TMEM address 和 mbarrier 预留位置：

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")           # TMEM address (4 bytes)
mma_bar = pool.alloc((1,), "uint64", align=8)    # mbarrier (8 bytes)
pool.move_base_to(1024)                           # Skip to offset 1024
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)  # 128×64 fp16
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)  # 128×64 fp16
pool.commit()
```

`pool.move_base_to(1024)` 将 SMEM pool 的当前分配位置移动到 byte offset 1024。之后，`Asmem` 从这里开始分配，`Bsmem` 紧随其后；前面的区域留给 `tmem_addr`、`mma_bar` 等 metadata。

`A_layout` 和 `B_layout` 由 `tma_shared_layout(dtype, swizzle_mode, shape)` 生成。这个函数根据数据类型、swizzle mode 和 tile shape 构造 shared-memory layout；这里选择 128-byte swizzle，得到与当前 `tcgen05.mma` dispatch 匹配的 SMEM 排列。`layout=A_layout` 和 `layout=B_layout` 再将这两个 layout 分别绑定到 `Asmem` 和 `Bsmem`。

第 1 步由 `Tx.cta.copy` 按照这些 layout 写入数据，随后 `tcgen05.mma` 按照匹配的排列读取。

**加载 operand tiles。** Buffer 分配完成后，由 CTA 中的 threads 把 operands 搬入 SMEM：

```python
Tx.cta.copy(Asmem[:, :], A[:, :])
Tx.cta.copy(Bsmem[:, :], B[:, :])
T.cuda.cta_sync()
```

这里只有一个 tile（`M=N=128, K=64`），因此直接复制完整的 A 和 B。`Tx.cta.copy(...)` 让 CTA 中的 threads 协作完成 copy，每个 thread 负责其中一部分。随后执行的 `T.cuda.cta_sync()` 一方面等待所有 threads 完成，另一方面保证它们对 shared memory 的写入对后续 MMA 可见。这样，MMA 读取 `Asmem` 和 `Bsmem` 时看到的是完整 tile。下一章（{ref}`chap_gemm_async`）会首先用 TMA 替换这里的 thread-driven copy。

**发起 MMA。** Operands 已经位于 SMEM，接下来由一个选出的 thread 发起 MMA：

```python
if warp_id == 0:
    if T.ptx.elect_sync():
        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                      accum=False, dispatch="tcgen05", cta_group=1)
        T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
```

外层 `if warp_id == 0` 只保留 warpgroup 中的 warp 0；内层 `T.ptx.elect_sync()` 再从这个 warp 的 active lanes 中选出一个。最终只有一个 thread 执行 `Tx.gemm_async` 和 `tcgen05.commit`。

只有一个 thread 发出指令，并不表示矩阵乘法由这个 thread 单独完成。硬件仍然根据 SMEM operand layouts 和 TMEM accumulator layout，对整个 tile 执行 MMA。若让 128 个 threads 都发出同一操作，硬件反而会重复启动这次计算。

`Tx.gemm_async` 表示一个 tile operation，而不是一条硬件指令。这里 `K=64`，大于硬件 MMA 的 K-atom（`MMA_K=16`），因此 TIRx 会沿 K 维将它 lower 成一小段 `tcgen05.mma` 指令序列。

`tcgen05.mma` 是异步操作。`tcgen05.commit` 将前面发出的 MMA 与 `mma_bar` 关联；warpgroup 中的 threads 随后在外层执行 `mbarrier.try_wait`，等到 barrier 完成后才能读取 TMEM 中的结果。

`accum=False` 表示这次 `gemm_async` 从新的 accumulator 开始，不读取 TMEM 中原有的 partial sum。本步骤只执行一次 tile operation，因此使用 `False`；第 2 步加入 K-loop 后，后续 iterations 会改用 `accum=True`。

**写回结果。** 计算结果位于 TMEM，而输出 `D` 需要以 fp16 写回 GMEM。Epilogue 先将结果读入 registers，再完成类型转换：

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

MMA 在 TMEM 中留下一个 `128×128` fp32 accumulator tile。沿 K 维累加大量乘积时，使用较高精度的 fp32 可以减小累计舍入误差。由于输出 `D` 是 fp16，结果需要先进入 registers，在那里转换成 fp16，再写入 GMEM。

两个 register buffers 作用不同。`Dreg` 是每个 thread 私有的 `BLK_N` 元素 buffer；`Dreg_wg` 则使用指定 layout，为同一组 registers 建立一个 warpgroup-wide view：

```python
TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)])
```

这个 layout 将 tile 的第一维映射到 warpgroup 中的 threads：thread 0 持有 row 0，thread 1 持有 row 1，依次直到 row 127。第二维保留在各 thread 自己的 register buffer 中，因此每个 thread 持有一整行的所有 columns。Warpgroup 有 128 个 threads，tile 也有 128 行，正好每个 thread 一行。

`Tx.wg.copy_async(Dreg_wg, tmem)` 按照这个 view 读取 accumulator，并 lower 到 Blackwell 的 TMEM load 指令 `tcgen05.ld`。该 load 是异步的，因此必须先完成 `T.ptx.tcgen05.wait.ld()`，之后 threads 才能使用 `Dreg`；否则可能读取尚未填充完成的 registers。

等待完成后，每个 thread 的 `Dreg[:]` 保存其逻辑输出行对应的 fp32 值。每个 thread 将这些值转换到 `Dreg_f16`，并计算自己负责的全局输出行：

```python
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
```

然后写入 `D[m_thr, n_st:n_st + BLK_N]`。四个 warps 分别负责连续的 32 行：warp 0 写第 0–31 行，warp 1 写第 32–63 行，warp 2 写第 64–95 行，warp 3 写第 96–127 行。

### 完整 Kernel

下面将四个部分组合成可运行的 kernel（`M=N=128, K=64`）。首先导入相关模块：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

Kernel 使用后续步骤共同采用的 `hgemm_vX(M, N, K)` 形式。第 1 步取 `M=N=128, K=64`，因此 launch 中只有一个 output tile：

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

后续每个 GEMM 版本都使用相同方式编译、运行并检查结果，因此这里只完整给出一次测试代码，之后只展示 kernel。运行后续步骤时，将下面的 `hgemm_vX` 和问题规模替换成对应版本即可。每个新的 Python session 只编译一个步骤；这些示例会复用内部名称，而 compiler 会保存 session 内的状态，因此切换步骤前需要重启 session。

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

第 1 至第 3 步故意使用较小规模（本节为 `128×128`，第 3 步为 $256^3$），便于理解最初几个版本。{ref}`chap_gemm_advanced` 末尾的“端到端结果”表则统一使用 `M=N=K=4096` 测量所有步骤，因此各版本的 speedup 可以直接比较。

### 单 Tile Kernel 的限制

这个 kernel 已经能够算对，但适用范围很窄。当前有四项刻意保留的限制，后续步骤会逐一解决：

- 只处理一个 K tile，无法对较大的 K 做 contraction。
- 只处理一个 output tile，因此 M、N 固定为 128。
- 使用同步的 GMEM → SMEM copy，而不是 TMA。
- 数据搬运与计算不重叠，两者不能同时执行。

---

(chap_k_loop)=
## 第 2 步：K-Loop 累加

先解决 K 维的限制。第 1 步只处理一个宽度为 64 的 K tile，而真实矩阵的 K 往往远大于 64。第 2 步仍然只计算一个 output tile，但允许 K 由多个宽度为 64 的 chunks 组成。

基本做法是：对每个 chunk 重复一次 `load -> MMA -> wait`，并让所有 MMA 累加到同一个 TMEM 位置。需要特别注意的是同步。多个 iterations 复用同一个 mbarrier 时，如果 phase 跟踪错误，wait 可能在当前 MMA 真正完成之前返回，最终结果会在没有报错的情况下被破坏。

> **这一步改变 Layout 复用方式**
> - Scope：不变，仍然是一个 warpgroup。
> - Layout/复用：K-loop 始终复用同一对 SMEM tiles 和同一个 TMEM accumulator 位置。Operand tiles 依次流过固定 buffers，accumulator 则保留在同一 TMEM 位置。
> - 同步：复用的 MMA barrier 必须在每个 K chunk 后进入正确 phase，否则后续 wait 可能误把上一轮完成当作当前轮完成。
> - Dispatch：不变。

### K-Loop 如何工作

当 K 大于 64 时，kernel 以 `BLK_K=64` 为步长遍历 K。每个 iteration 将 A、B 的下一个 K-slice 加载到 SMEM，再执行 `Tx.gemm_async`。`accum` 参数把多个 chunks 连接成一次完整 dot product：第一个 chunk 使用 `accum=False` 初始化 TMEM accumulator；之后每个 chunk 使用 `accum=True`，将新的乘积加到 TMEM 中已有的 partial sum 上。

每次 MMA 都复用同一个 mbarrier，因此需要正确跟踪当前等待的 phase。mbarrier 的 phase 在 0 和 1 之间切换，每当预期的 arrival 到达后就进入另一 phase。`try_wait(bar, phase)` 会等待 barrier 的内部 phase 与参数 `phase` 不同。因此，传入的参数表示当前准备离开的 phase，而不是等待进入的 phase：

| K iteration | wait 前的 `phase_mma` | `try_wait` 等待的状态 | wait 后的本地更新 |
|---|---:|---|---:|
| 0 | 0 | barrier 切换到 1 | `phase_mma = 1` |
| 1 | 1 | barrier 切换到 0 | `phase_mma = 0` |
| 2 | 0 | barrier 切换到 1 | `phase_mma = 1` |

`phase_mma ^= 1` 用来在每轮之后翻转本地 phase。若删除这行，第二个 iteration 仍会调用 `try_wait(bar, 0)`；但 barrier 在第一次 MMA 后已经切换到 phase 1，因此 wait 立即观察到 phase 不同并返回，此时第二次 MMA 可能尚未完成。Kernel 随后会读取仍在更新的 accumulator，并得到错误结果。这个错误不会导致编译或运行失败，因此 phase flip 不能省略。

### 完整 Kernel

下面的完整 kernel 在第 1 步基础上加入 K-loop 和 phase flip。Imports 与前面相同：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

这个版本封装为 `hgemm_v2(M, N, K)`。由于仍然只计算一个 output tile，grid 仍为 `[1, 1]`；变化的只是 K extent。

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
## 第 3 步：空间 Tiling（Multi-CTA）

K-loop 已经解决 contraction dimension，但 M、N 仍然固定在一个 `128×128` tile。第 3 步使用多个 tiles 覆盖 M、N，并启动一个二维 CTA grid：每个 output tile 对应一个 CTA，GPU 可以并行计算这些 tiles。示例取 `M=N=K=256`，得到 `2×2` tile grid，足以展示索引关系，同时保持规模简单。

> **这一步改变 Scope**
> - Scope：二维 CTA grid，每个 CTA 负责一个 `128×128` output tile。
> - Layout：不变；CTA 内部仍使用第 2 步的 SMEM/TMEM/register 路径。
> - Dispatch：不变。

### Grid 映射

每个 `128×128` output tile 对应一个 CTA，因此 grid shape 为 `[M // BLK_M, N // BLK_N]`。与第 2 步相比，新增的工作只是确定每个 CTA 负责哪些矩阵 slices。

CTA `(bx, by)` 负责下面的输出区域：

```text
D[bx * BLK_M : (bx + 1) * BLK_M,
  by * BLK_N : (by + 1) * BLK_N]
```

为了计算这块区域，CTA 的 K-loop 会依次加载 A 对应 row band 和 B 对应 row band 中的 K-slices：

```text
A[bx * BLK_M : (bx + 1) * BLK_M, k : k + BLK_K]
B[by * BLK_N : (by + 1) * BLK_N, k : k + BLK_K]
```

索引直接来自 `D = A @ B.T`：`bx` 选择 A 和 D 的 rows；`by` 选择 B 的 rows，这些 rows 在乘以 `B.T` 后对应 D 的 columns。

每个 CTA 计算一个 tile 是最简单的映射，但会产生重复加载。同一 grid row 中的 CTAs 会从 GMEM 重复加载相同的 A tiles，同一 grid column 中的 CTAs 则会重复加载相同的 B tiles，邻近 CTAs 之间没有显式复用。第 6 步的 persistent scheduling（{ref}`chap_gemm_async`）会重新处理这个问题，使共享 operands 尽可能保留在 L2 中。

**使用你的 agent 练习**：取 `M=N=K=256`、`BLK_M=BLK_N=128`、`BLK_K=64`，分别追踪 CTA `(1, 0)` 和 CTA `(0, 1)`。列出每个 CTA 的 `m_st`、`n_st`，每次 K iteration 加载的 A、B slices，以及最终写入的 D 区域。由于 kernel 计算 `D = A @ B.T`，B 的哪些 rows 会成为 D 的 columns？

### 完整 Kernel

这个 kernel 只在第 2 步基础上修改两处：grid shape 和每个 CTA 的 offsets。内部 K-loop 与 writeback 保持不变。Imports 仍然相同：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

Grid 从 `[1, 1]` 改为 `[M // BLK_M, N // BLK_N]`，loads 和 stores 则加上当前 CTA 的 `m_st` 与 `n_st`：

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

1. 在第 1 至第 3 步中，`Tx.copy` 会在 MMA 之前将 A、B tiles 搬入 SMEM。为什么 `Tx.gemm_async` 读取这些 tiles 前必须执行 `T.cuda.cta_sync()`？
2. 在第 2 步中，如果从 K-loop 删除 `phase_mma ^= 1`，会发生什么？Kernel 仍会等待每次 MMA，还是后续 wait 可能提前通过？
3. 当 `M=N=4096`、`BLK_M=BLK_N=128` 时，第 3 步会启动多少 CTAs？邻近 CTAs 在逻辑上复用了哪些 operand tiles？第 3 步是否真正利用了这种复用？
