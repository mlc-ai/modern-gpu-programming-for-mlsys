(chap_gemm_basics)=
# Building a Tiled GEMM

:::{admonition} Overview
:class: overview

- Step 1 computes one $128\times128$ output tile, with load, MMA, and writeback performed in sequence. A and B move from GMEM to SMEM, `tcgen05` writes the accumulator to TMEM, and the result is read back and stored.
- Step 2 partitions K and accumulates the partial sums in one TMEM accumulator. Reusing the MMA barrier requires the phase to advance in step with the loop.
- Step 3 assigns the output tiles along M and N to a two-dimensional CTA grid, producing a correctness baseline that covers the full matrix.
:::

GEMM is a core computation used throughout this book. Linear layers, attention projections, and many convolution implementations are built on matrix multiplication, and these operations often account for much of a GPU's execution time. Before optimizing GEMM, we need a correct baseline kernel whose structure is easy to follow.

Introducing data movement, K-dimension accumulation, tiling, and Tensor Core scheduling all at once would make failures difficult to localize. We therefore extend the kernel one mechanism at a time and keep each earlier version as a reference.

We begin with one CTA computing a $128\times128$ output tile. We then add a K-loop to complete the reduction and partition the output along M and N so that multiple CTAs cover the full problem. By the end of the chapter, the kernel handles complete matrices, but performance is not yet the goal.

These steps also apply the TIRx model from the preceding chapters to concrete code. As you read, keep track of three questions: which **scope** executes an operation, which **layout** an operand tile uses, and which **dispatch** path implements the tile operation. The next two chapters continue from this baseline with asynchronous data movement, pipelining, and further performance optimizations.

## GEMM

GEMM is dense matrix multiplication, the operation behind linear layers, attention projections, and many convolution implementations. This chapter uses the following convention:

- $A$ has shape $M \times K$.
- $B$ has shape $N \times K$.
- $D$ has shape $M \times N$.
- $D[m,n] = \sum_k A[m,k] \cdot B[n,k]$.

We store $B$ with shape $N\times K$, a common layout for linear-layer weights. The kernel reads $B[n,k]$ directly. In matrix notation, this is equivalent to $D=AB^{\top}$, but the kernel does not transpose or rearrange $B$ at runtime.

We measure kernel throughput in TFLOPS. Each multiply-add counts as two floating-point operations, so:

$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{t_{\text{seconds}} \times 10^{12}}$$

### GEMM Data Path

Every later optimization depends on where data lives and how it moves, so we begin with the basic Blackwell GEMM data path. The kernel performs two kinds of work: moving tiles between memory spaces and computing with those tiles. The figure below follows the data from input to output:

![*Memory Data Flow*](../img/memory_dataflow.png)

Read the figure from left to right. Operand tiles first move from GMEM to SMEM. `tcgen05.mma` reads the operands from SMEM and writes the accumulator to TMEM. The epilogue then reads the result from TMEM into registers and stores it to GMEM. Later optimizations change how a step in this path executes, while the path itself remains the same.

## Optimization Path

The basic data path is sufficient for correctness, but it does not yet use the hardware efficiently. The following chapters add these mechanisms through TIRx tile primitives:

- **TMA async movement** moves GMEM <-> SMEM tiles through Blackwell's hardware copy path, with barriers tracking completion.
- **Software pipelining** uses multiple SMEM stages so that the data movement for the next K tile can overlap Tensor Core compute on the current one.
- **Persistent scheduling** keeps a fixed pool of CTAs, each processing many output tiles through a tile scheduler, instead of launching one CTA per tile.
- **Warp specialization** assigns the producer, MMA consumer, and writeback roles to dedicated warps or warpgroups.
- **CTA clusters** let two CTAs cooperate on a single, larger Blackwell MMA tile.
- **Multi-consumer execution** uses multiple MMA consumer warps, each paired with a writeback warpgroup, to compute different row ranges while sharing a staged B tile.

---

(chap_single_tile)=
## Step 1: Sequential Single-Tile GEMM

Step 1 reuses `hgemm_v1` from {ref}`chap_tirx_primer`. Rather than introducing a new kernel, this section walks through its data path in detail and uses it as the correctness baseline for every later version. The kernel computes one $128\times128$ output tile with $K=64$. At this size no loop is needed, so each part of the path appears only once.

> **This step establishes the baseline**
> - Scope: one warpgroup of 128 threads executes the full path sequentially.
> - Layout: A and B tiles reside in SMEM, the accumulator resides in TMEM, and the result is written through registers.
> - Dispatch: synchronous `Tx.copy` performs the loads, and `tcgen05` performs the MMA.

### Single-Tile Dataflow

This kernel follows the `GMEM -> SMEM -> TMEM -> registers -> GMEM` path once, without a loop:

1. **Allocate**: allocate SMEM through the pool allocator, allocate TMEM with `tcgen05.alloc`, and prepare an mbarrier.
2. **Load**: all 128 threads cooperatively copy A and B tiles from GMEM to SMEM with synchronous `Tx.copy`.
3. **Compute**: one elected thread issues `Tx.gemm_async` and `tcgen05.commit`; the warpgroup waits on the mbarrier.
4. **Write back**: the warpgroup reads TMEM into registers; each thread converts fp32 to fp16 and stores its row to GMEM.
5. **Release**: release the TMEM allocation.

### Four Pieces of the First Kernel

We first examine allocation, operand loading, MMA issue, and writeback separately, then combine them into the complete kernel. The relevant APIs were introduced in Part II ({ref}`chap_tirx_primer`, {ref}`chap_tirx_layout_api`).

**Allocate storage.** The kernel allocates shared memory for the operands and reserves space for the TMEM address and mbarrier:

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")           # TMEM address (4 bytes)
mma_bar = pool.alloc((1,), "uint64", align=8)    # mbarrier (8 bytes)
pool.move_base_to(1024)                           # Skip to offset 1024
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)  # 128×64 fp16
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)  # 128×64 fp16
pool.commit()
```

`pool.move_base_to(1024)` moves the current SMEM allocation point to byte offset 1024. `Asmem` begins there and `Bsmem` follows it; the lower addresses remain available for metadata such as `tmem_addr` and `mma_bar`.

`tma_shared_layout(dtype, swizzle_mode, shape)` constructs a shared-memory layout from the data type, swizzle mode, and tile shape. Here it produces a 128-byte-swizzled arrangement that matches the current `tcgen05.mma` dispatch. Passing `layout=A_layout` and `layout=B_layout` binds those layouts to `Asmem` and `Bsmem`.

In Step 1, `Tx.cta.copy` writes data according to these layouts and `tcgen05.mma` reads the matching arrangement.

**Load the operand tiles.** Once the buffers are allocated, the CTA threads move the operands into SMEM:

```python
Tx.cta.copy(Asmem[:, :], A[:, :])
Tx.cta.copy(Bsmem[:, :], B[:, :])
T.cuda.cta_sync()
```

There is only one tile (`M=N=128, K=64`), so the kernel copies all of A and B. `Tx.cta.copy(...)` distributes the copy across the CTA threads. `T.cuda.cta_sync()` then waits for every thread and makes their shared-memory writes visible to the later MMA, ensuring that `Asmem` and `Bsmem` are complete before they are read. The next chapter ({ref}`chap_gemm_async`) replaces this thread-driven copy with TMA.

**Issue the MMA.** With both operands in SMEM, one elected thread issues the MMA:

```python
if warp_id == 0:
    if T.ptx.elect_sync():
        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                      accum=False, dispatch="tcgen05", cta_group=1)
        T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
```

The outer `if warp_id == 0` keeps only warp 0, and `T.ptx.elect_sync()` selects one active lane from that warp. Exactly one thread therefore executes `Tx.gemm_async` and `tcgen05.commit`.

Having one issuing thread does not make this a single-threaded matrix multiplication. The hardware still performs the tile-level MMA described by the SMEM operand layouts and the TMEM accumulator layout. If all 128 threads issued the same operation, the computation would be launched 128 times.

`Tx.gemm_async` represents a tile operation rather than one hardware instruction. Because `K=64` is larger than the hardware K-atom (`MMA_K=16`), TIRx lowers this operation to a short sequence of `tcgen05.mma` instructions along K.

`tcgen05.mma` is asynchronous. `tcgen05.commit` associates the issued MMA operations with `mma_bar`, and the warpgroup waits on the mbarrier before reading the result from TMEM.

`accum=False` starts a new accumulator instead of reading a previous partial sum from TMEM. This step performs only one tile operation, so no earlier sum exists. Step 2 uses `accum=True` for the later K iterations.

**Write back the result.** The result is in TMEM, while `D` must be stored to GMEM as fp16. The epilogue first reads the accumulator into registers and performs the conversion there:

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

The MMA leaves a $128\times128$ fp32 accumulator tile in TMEM. Accumulating along K in fp32 reduces rounding error. Because `D` is fp16, the result must pass through registers, where it is converted before being stored to GMEM.

The two register buffers serve different purposes. `Dreg` is a private `BLK_N`-element buffer for each thread. `Dreg_wg` applies a layout to those same registers and exposes them as a warpgroup-wide view:

```python
TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)])
```

The layout maps the first tile dimension to the warpgroup threads: thread 0 owns row 0, thread 1 owns row 1, and so on through row 127. The second dimension remains within each thread's register buffer, so every thread holds one complete row. The 128 threads therefore match the 128 output rows one-to-one.

`Tx.wg.copy_async(Dreg_wg, tmem)` reads the accumulator through this view and lowers to the Blackwell TMEM load instruction `tcgen05.ld`. The load is asynchronous, so `T.ptx.tcgen05.wait.ld()` must complete before any thread uses `Dreg`.

After the wait, each thread's `Dreg[:]` contains the fp32 values for its logical output row. The thread converts them into `Dreg_f16` and computes its global row:

```python
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
```

It then writes `D[m_thr, n_st:n_st + BLK_N]`. The four warps cover consecutive groups of 32 rows: warp 0 writes rows 0-31, warp 1 writes rows 32-63, warp 2 writes rows 64-95, and warp 3 writes rows 96-127.

### Complete Kernel

Now we stitch the four pieces back together into one runnable kernel (M=N=128, K=64). The imports come first:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

The kernel is wrapped in the same `hgemm_vX(M, N, K)` style that the later steps use. Step 1 runs with `M=N=128, K=64`, so the launch contains exactly one output tile:

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

Every GEMM step that follows compiles, runs, and checks itself in the same way, so we spell that scaffolding out in full just once, here, and from then on show only the kernel. To run a later step, drop in its `hgemm_vX` and the matching problem size in place of the ones below. One caveat is worth remembering: compile a single step per fresh Python session and restart before trying another, since the examples reuse inner names and the compiler holds per-session state.

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

### Limits of the Single-Tile Kernel

The kernel is correct, but it still has a narrow operating range:

- It handles only a single K tile, so it cannot contract over a large K.
- It handles only a single output tile, so M and N are pinned to 128.
- It uses synchronous GMEM -> SMEM copies rather than TMA.
- It does not overlap data movement with compute, so the two never run at once.

---

(chap_k_loop)=
## Step 2: K-Loop Accumulation

We first remove the restriction on K. Step 1 handles only one K tile of width 64, while real matrices often have a much larger K dimension. Step 2 still computes one output tile, but K may now span several 64-wide chunks.

For each chunk, the kernel repeats `load -> MMA -> wait` and accumulates every MMA into the same TMEM location. Synchronization now requires more care: when several iterations reuse one mbarrier, incorrect phase tracking can let a wait return before the current MMA has finished, silently corrupting the result.

> **What this step changes: Layout reuse**
> - Scope: unchanged, still a single warpgroup.
> - Layout/reuse: the same SMEM tile pair and TMEM accumulator slot are reused across the K-loop. No new storage is allocated; the operand tiles stream through one fixed pair of buffers, and the accumulator state stays in one TMEM slot.
> - Synchronization: the reused MMA barrier must advance through the right phase on every K chunk, or a later wait can observe an earlier completion.
> - Dispatch: unchanged.

### Accumulating Along K

When `K > 64`, the kernel divides K into chunks of width `BLK_K=64`. Each iteration loads one K-slice from A and B, then issues `Tx.gemm_async`.

The `accum` argument controls whether the MMA reads the existing accumulator from TMEM. The first chunk uses `accum=False` to write the initial partial sum. Every later chunk uses `accum=True` to add its product to the running result.

Every MMA reports completion through the same mbarrier. `phase_mma` records the barrier phase that the current iteration must wait for:

| K iteration | `phase_mma` passed to `try_wait` | Barrier phase after MMA completes |
|---|---:|---:|
| 0 | 0 | 1 |
| 1 | 1 | 0 |
| 2 | 0 | 1 |

`try_wait(bar, phase_mma)` returns after the barrier leaves the specified phase. After each successful wait, the kernel executes:

```python
phase_mma ^= 1
```

This updates the wait value for the next MMA.

Without the flip, the second iteration would still wait for phase 0. Because the barrier entered phase 1 after the first MMA, that wait could return immediately and allow the kernel to read the accumulator before the second MMA finishes.

### Complete Kernel

The Step 2 kernel keeps the structure of Step 1 and adds the K-loop and phase update described above. The imports are unchanged:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

It is wrapped in `hgemm_v2(M, N, K)`. The grid is still `[1, 1]`, since we are still computing a single output tile; all that has grown is its K extent:

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
## Step 3: Spatial Tiling (Multi-CTA)

Step 2 allows K to exceed 64, but it still requires `M=N=128` and therefore computes only one $128\times128$ output tile. Real GEMMs usually have larger M and N dimensions. Step 3 partitions the $M\times N$ output into $128\times128$ tiles and launches one CTA for each tile.

All CTAs launched by one kernel form a **grid**. Because the output tiles are arranged along both M and N, this kernel uses a two-dimensional grid. A CTA's coordinate `(bx, by)` identifies the output tile row and column that it computes.

For example, with `M=N=256, K=256`, the output contains a $2\times2$ array of tiles. The grid therefore has shape $2\times2$ and contains four CTAs. Each CTA computes one output tile and runs the K-loop from Step 2 internally.

> **What this step changes: Scope**
> - Scope: a 2D grid of CTAs, with each CTA owning one 128 x 128 output tile.
> - Layout: unchanged; within a CTA, this is the same SMEM/TMEM/register path as Step 2.
> - Dispatch: unchanged.

### Grid Mapping

The grid shape is:

```text
[M // BLK_M, N // BLK_N]
```

For CTA `(bx, by)`, define:

```text
m_st = bx * BLK_M
n_st = by * BLK_N
```

The CTA computes:

```text
D[m_st : m_st + BLK_M, n_st : n_st + BLK_N]
```

At each K iteration, it loads:

```text
A[m_st : m_st + BLK_M, k : k + BLK_K]
B[n_st : n_st + BLK_N, k : k + BLK_K]
```

These indices follow from `D = A @ B.T`: `bx` selects rows of A and D, while `by` selects rows of B. After `B.T`, those rows correspond to columns of D.

CTAs with the same `bx` read the same A tiles, while CTAs with the same `by` read the same B tiles. This version does not explicitly share those tiles across CTAs.

### Complete Kernel

The kernel is once again Step 2, this time with just two changes: the grid shape and the per-CTA offsets. The inner K-loop and the writeback are untouched. The imports are the same:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

The grid becomes `[M // BLK_M, N // BLK_N]` rather than `[1, 1]`, and the loads and stores are now offset by the CTA's own `m_st` and `n_st`:

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

## Exercises

1. In Steps 1-3, `Tx.copy` moves A and B tiles into SMEM before MMA. Why does the kernel need `T.cuda.cta_sync()` before `Tx.gemm_async` reads those SMEM tiles?
2. In Step 2, what happens if `phase_mma ^= 1` is removed from the K-loop? Does the kernel wait for every MMA, or can a later wait pass too early?
3. For `M=N=4096` and `BLK_M=BLK_N=128`, what is the Step 3 grid shape, and how many CTAs does it launch? For CTA `(bx, by)`, which other CTAs independently read the same A tiles, and which independently read the same B tiles? Does the current kernel explicitly share that data?
