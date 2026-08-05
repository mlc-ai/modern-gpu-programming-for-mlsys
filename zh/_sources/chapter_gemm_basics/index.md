(chap_gemm_basics)=
# 构建 Tiled GEMM

:::{admonition} 本章概览
:class: overview

- 第 1 步从输出矩阵 D 中一个顺序计算的 $128\times128$ tile 开始，走通输入矩阵 A、B 从 GMEM 加载到 SMEM、`tcgen05` MMA 将结果写入 TMEM，以及 D 的读回与写出。
- 第 2 步沿 K 维分块，并在同一块 TMEM accumulator 中累加 partial sums；每次复用 MMA barrier 时，都要更新下一轮的等待状态。
- 第 3 步沿 M、N 维划分 output tiles，并让多个 CTAs 分别计算这些 tiles，从而覆盖完整的输出矩阵。
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

示例中的 $A$、$B$ 和 $D$ 都以 fp16 存储。MMA 沿 $K$ 维累加时使用 fp32 accumulator，以减小累计舍入误差。

Kernel 性能使用 TFLOPS 衡量。一次 multiply-add 计作两次浮点运算，因此：

$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{t_{\text{seconds}} \times 10^{12}}$$

### GEMM 的数据路径

后面的每项优化都与数据存放在哪里、如何移动有关，因此先看 Blackwell GEMM 的基本数据路径。Kernel 主要完成两类工作：在不同 memory space 之间搬运 tiles，以及使用这些 tiles 进行计算。下图展示了数据从输入到输出依次经过的 memory space：

![*Memory 数据流*](../../img/memory_dataflow.png)

从左向右看：operand tiles 先从 GMEM 进入 SMEM；`tcgen05.mma` 读取 SMEM 中的 operands，并把 accumulator 写入 TMEM；最后的结果写回阶段称为 epilogue，它将 TMEM 中的结果读入 registers，再写回 GMEM。后续优化会改变其中某一步如何执行，但不会改变这条基本路径。

## 优化路线

这条基础数据路径足以得到正确结果，但还不能充分利用硬件。接下来会通过 TIRx tile primitives 依次加入以下机制：

- **TMA 异步搬运**：使用 Blackwell 的硬件 copy path 在 GMEM 与 SMEM 之间搬运 tiles，并通过 barrier 跟踪完成状态。
- **Software pipeline**：使用多个 SMEM stages，让下一块 K tile 的数据搬运与当前 tile 的 Tensor Core 计算重叠。
- **Persistent scheduling**：不再为每个 output tile 启动一个 CTA，而是让固定数量的 CTAs 通过 tile scheduler 反复处理多个 tiles。
- **Warp specialization**：把 producer、MMA consumer 和 writeback 分配给专门的 warps 或 warpgroups。
- **CTA cluster**：让两个 CTAs 协作计算一个更大的 Blackwell MMA tile。
- **Multi-consumer execution**：使用多个 MMA consumer warps 分别计算不同的 output rows，并为每个 consumer 配置对应的 writeback warpgroup；这些 consumers 共用同一份 staged B tile。

---

(chap_single_tile)=
## 第 1 步：顺序执行的单 Tile GEMM

第 1 步沿用 {ref}`chap_tirx_primer` 中的 `hgemm_v1`，详细拆解其数据路径，并将它作为后续版本的正确性基线。这个 kernel 只计算一个 `128×128` output tile，并取 `K=64`；该规模不需要循环，数据路径中的每一步只出现一次，便于逐段理解。

> **第 1 步的执行结构**
> - Scope：一个包含 128 个 threads 的 warpgroup 按顺序执行整条数据路径。
> - Layout：A、B tiles 位于 SMEM，accumulator 位于 TMEM，结果通过 registers 写出。
> - Dispatch：同步 `Tx.cta.copy` 负责加载，`tcgen05` 执行 MMA。

### 单 Tile 数据流

这个 kernel 只沿 `GMEM -> SMEM -> TMEM -> registers -> GMEM` 路径执行一次，不包含循环。具体步骤如下：

1. **分配**：通过 pool allocator 分配 SMEM，通过 `tcgen05.alloc` 分配 TMEM，并准备等待 MMA 完成的 mbarrier。
2. **加载**：128 个 threads 使用同步 `Tx.cta.copy`，协作将 A、B tiles 从 GMEM 搬到 SMEM。
3. **计算**：选出的一个 thread 发出 `Tx.gemm_async` 和 `tcgen05.commit`，所有 threads 等待 mbarrier。
4. **写回**：warpgroup 将 TMEM 读入 registers；每个 thread 把 fp32 转成 fp16，再写入 GMEM。
5. **释放**：释放 TMEM。

### Kernel 的四个部分

下面先分别介绍存储空间分配、operand 加载、MMA 发起和结果写回，再把它们组合成完整 kernel。相关 API 已在第二部分（{ref}`chap_tirx_primer`、{ref}`chap_tirx_layout_api`）中介绍。本节固定使用 `BLK_M=BLK_N=128`、`BLK_K=64`；`m_st` 和 `n_st` 表示当前 output tile 在 D 中的行、列起点，在这个单 tile kernel 中都为 0。

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

`pool.move_base_to(1024)` 将 SMEM pool 的当前分配位置移动到 byte offset 1024。之后，`Asmem` 从这里开始分配，`Bsmem` 紧随其后；前面的区域留给 `tmem_addr`、`mma_bar` 等少量管理数据。

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

`Tx.gemm_async` 表示一个 tile operation，而不是一条硬件指令。这里 tile 的 K 维大小为 64，而底层每条 MMA 指令处理 16 个 K 元素，因此 TIRx 会将它 lower 成一小段 `tcgen05.mma` 指令序列。

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

Kernel 使用后续步骤共同采用的 `hgemm_vX(M, N, K)` 形式。一次 kernel launch 中的所有 CTAs 构成 grid；第 1 步取 `M=N=128, K=64`，只需要一个 CTA，因此 grid shape 为 `1×1`：

```python
def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    # MMA_M/MMA_N/MMA_K 记录底层硬件 MMA tile 的 shape。gemm_async 会根据
    # operands 和 accumulator tiles 推导该 shape，因此后续步骤不再保留这些常量。
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
        # 第 1 步只计算一个 tile：M=BLK_M、N=BLK_N，因此 grid shape 为 1x1。
        # 此时每个 CTA 的 tile offsets（m_st、n_st）都为 0；第 3 步再扩展到更大的 M、N。
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

        # --- Load：所有 threads 同步完成 global -> shared copy ---
        # M=BLK_M、N=BLK_N 时，下面的 slices 覆盖完整矩阵；保留 slice 写法，
        # 便于与第 3 步的 multi-tile 版本比较。
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()

        # --- Compute：由一个 elected thread 发起 MMA ---
        if warp_id == 0:
            if T.ptx.elect_sync():
                Tx.gemm_async(
                    tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                    accum=False, dispatch="tcgen05", cta_group=1
                )
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)

        # --- Writeback：TMEM -> RF -> GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

        # --- 释放 TMEM ---
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

# ex.mod(...) 可以直接接收 torch tensors；后续章节沿用相同的调用方式。
ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
# 与 warp specialization 和 Flash Attention 的示例一样，这里使用相对容差：
# output magnitude 会随 K 增长，固定的绝对误差上限不适用于较大的 K。
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")

# 对更大 kernel 进行可选计时。
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

### 单 Tile Kernel 的限制

这个 kernel 已经能够算对，但适用范围很窄。当前仍有以下限制：

- 只处理一个 K tile，无法沿更大的 K 维完成分块累加。
- 只处理一个 output tile，因此 M、N 固定为 128。
- 使用同步的 GMEM → SMEM copy，而不是 TMA。
- 数据搬运与计算不重叠，两者不能同时执行。

---

(chap_k_loop)=
## 第 2 步：K-Loop 累加

先解决 K 维的限制。第 1 步只处理一个宽度为 64 的 K tile，而真实矩阵的 K 往往远大于 64。第 2 步仍然只计算一个 output tile，但允许 K 由多个宽度为 64 的 chunks 组成。

基本做法是：对每个 chunk 重复一次 `load -> MMA -> wait`，并让所有 MMA 累加到同一个 TMEM 位置。`Tx.gemm_async` 只负责发起异步 MMA；它返回时，Tensor Core 可能仍在更新 TMEM。随后执行的 `tcgen05.commit` 将本轮 MMA 的完成通知关联到 `mma_bar`，硬件写完 accumulator 后才会向这个 barrier 报告 arrival。`try_wait` 等待的正是这次完成通知，返回后才能确认当前 chunk 的结果已经写入 TMEM。

所有 iterations 都复用同一个 `mma_bar`。Barrier 每完成一轮就进入下一个 phase，因此 kernel 还要用 `phase_mma` 指明当前等待的是哪一轮。若 phase 跟踪错误，wait 可能把上一轮的完成状态当成当前 MMA 已经完成，最终在没有报错的情况下破坏结果。

> **第 2 步的执行结构**
> - Scope：不变，仍然是一个 warpgroup。
> - Layout/复用：K-loop 始终复用同一对 SMEM tiles 和同一个 TMEM accumulator 位置。Operand tiles 依次流过固定 buffers，accumulator 则保留在同一 TMEM 位置。
> - 同步：复用的 MMA barrier 必须在每个 K chunk 后进入正确 phase，否则后续 wait 可能误把上一轮完成当作当前轮完成。
> - Dispatch：不变。

### 沿 K 维分块累加

当 `K > 64` 时，kernel 将 K 维切成多个宽度为 `BLK_K=64` 的 chunks。每个 iteration 加载 A、B 的一个 K-slice，然后执行 `Tx.gemm_async`。

`accum` 决定是否读取 TMEM 中已有的 accumulator。第一个 chunk 使用 `accum=False`，直接写入第一份 partial sum；后续 chunks 使用 `accum=True`，把新的乘积累加到已有结果上。

代码中，每轮选出的 thread 都在 `Tx.gemm_async` 后执行 `tcgen05.commit(mma_bar)`。MMA 完成并报告 arrival 后，barrier 才会离开当前 phase。`phase_mma` 记录当前 iteration 要等待的 phase：

| K iteration | 传给 `try_wait` 的 `phase_mma` | MMA 完成后的 barrier phase |
|---|---:|---:|
| 0 | 0 | 1 |
| 1 | 1 | 0 |
| 2 | 0 | 1 |

`try_wait(bar, phase_mma)` 会等到 barrier 离开指定 phase 后返回。每次等待完成后，kernel 执行：

```python
phase_mma ^= 1
```

为下一次 MMA 更新等待值。

如果不翻转 `phase_mma`，第二个 iteration 仍会等待 phase 0。但 barrier 在第一次 MMA 完成后已经进入 phase 1，这次等待可能立即返回，导致 kernel 在第二次 MMA 尚未完成时继续读取 accumulator。

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
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])  # 仍然只有一个 output tile（M=N=128）
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

        # === K-loop：以 BLK_K 为单位遍历 K ===
        for i in T.serial(K_TILES):   # device 侧串行 loop；A、B parameters 仍保留完整 K 维
            # 加载第 i 个 K chunk
            Tx.cta.copy(Asmem[:, :], A[:, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[:, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            # 第一个 tile 使用 accum=False，后续 tiles 使用 accum=True
            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            # 等待 MMA 完成，再翻转 phase
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # === Writeback（与第 1 步相同）===
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

第 2 步允许 K 大于 64，但仍要求 `M=N=128`，因此只能计算一个 `128×128` output tile。实际 GEMM 的 M、N 往往更大。第 3 步将 `M×N` 输出矩阵切成多个 `128×128` tiles，并为每个 tile 启动一个 CTA。

前两步的 grid 只有一个 CTA。现在 output tiles 沿 M、N 两个方向排列，因此第 3 步使用二维 grid；CTA 的 coordinate `(bx, by)` 表示它负责第几行、第几列的 output tile。

例如，取 `M=N=256, K=256` 时，输出矩阵被切成 `2×2` 个 tiles，因此 grid shape 为 `2×2`，共包含 4 个 CTAs。每个 CTA 负责一个 output tile，并在内部执行第 2 步的 K-loop。

> **第 3 步的执行结构**
> - Scope：二维 CTA grid，每个 CTA 计算一个 `128×128` output tile。
> - Layout：不变，CTA 内部仍使用第 2 步的 SMEM、TMEM 和 register layouts。
> - Dispatch：不变。

### Grid 映射

Grid shape 为：

```text
[M // BLK_M, N // BLK_N]
```

对于 CTA `(bx, by)`，定义：

```text
m_st = bx * BLK_M
n_st = by * BLK_N
```

它负责的输出区域为：

```text
D[m_st : m_st + BLK_M, n_st : n_st + BLK_N]
```

每次 K iteration 加载：

```text
A[m_st : m_st + BLK_M, k : k + BLK_K]
B[n_st : n_st + BLK_N, k : k + BLK_K]
```

这些索引来自 `D = A @ B.T`：`bx` 选择 A 和 D 的行，`by` 选择 B 的行；经过 `B.T` 后，这些 B rows 对应 D 的 columns。

具有相同 `bx` 的 CTAs 会读取相同的 A tiles，具有相同 `by` 的 CTAs 则会读取相同的 B tiles。当前版本没有显式实现跨 CTA 的数据复用。

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

        # 当前 CTA 的 tile offsets
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # K-loop：加载带 offset 的 A、B slices
        for i in T.serial(K_TILES):   # device 侧串行 loop；A、B parameters 仍保留完整 K 维
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

        # 写回当前 CTA 对应的 output tile
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

1. 在第 1 至第 3 步中，`Tx.cta.copy` 会在 MMA 之前将 A、B tiles 搬入 SMEM。为什么 `Tx.gemm_async` 读取这些 tiles 前必须执行 `T.cuda.cta_sync()`？
2. 在第 2 步中，如果从 K-loop 删除 `phase_mma ^= 1`，会发生什么？Kernel 仍会等待每次 MMA，还是后续 wait 可能提前通过？
3. 当 `M=N=4096`、`BLK_M=BLK_N=128` 时，第 3 步的 grid shape 是多少，共启动多少个 CTAs？对于 CTA `(bx, by)`，哪些 CTAs 会独立读取相同的 A tiles，哪些会独立读取相同的 B tiles？当前 kernel 是否显式共享了这些数据？
