(chap_gemm_async)=
# Pipelining GEMM with TMA

:::{admonition} Overview
:class: overview

- Step 4 uses TMA to move tiles between GMEM and SMEM. Load completion is tracked with an mbarrier, while stores use async commit and wait groups.
- Step 5 turns the A and B buffers into a double-buffered SMEM ring and introduces prefetching, stage reuse, and phase management, establishing the buffer structure needed to overlap TMA loads with MMA.
- Step 6 adds a tile scheduler and turns the kernel into a persistent kernel, allowing a fixed number of CTAs to process multiple output tiles while improving L2 locality.
:::

The kernel from the previous chapter processes each K tile in a fixed order: threads copy A and B into shared memory and wait for every write to finish, then issue the MMA and wait for the computation to complete before loading the next tile. This sequence is easy to follow and produces the correct result, but data movement and Tensor Core computation cannot overlap.

This chapter continues from the three kernels built so far. Step 4 replaces the thread-driven A and B copies with TMA. Step 5 provides two shared-memory stages for prefetching and later overlap. Step 6 adds a tile scheduler so that resident CTAs can process several output tiles in succession. By the end of the chapter, the kernel has asynchronous tile movement, reusable SMEM stages, and persistent scheduling. The next chapter assigns these stages to separate warp roles so that they can run concurrently.

(chap_tma_async)=
## Step 4: TMA Async Load

Steps 1 through 3 use `Tx.cta.copy` to move A and B tiles: the CTA threads compute their addresses and execute the corresponding loads and stores. Step 4 switches to TMA. One thread issues the operation, and the TMA engine performs the remaining address generation and tile transfer. From this point onward, the examples use the full `M=N=K=4096` problem size.

> **What this step changes: Dispatch**
> - Scope: unchanged, one warpgroup.
> - Layout: unchanged, same SMEM/TMEM/register tiles.
> - Dispatch: GMEM → SMEM loads move from sync `Tx.copy` to the TMA engine.

### Issuing a TMA Load

First compare the code in Steps 3 and 4.

**Before (Step 3)**: all 128 threads participate in the copy, then `cta_sync` makes the shared-memory writes visible:
```python
Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])   # all 128 threads
Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])
T.cuda.cta_sync()
```

**After (Step 4)**: one thread issues the TMA load, and the mbarrier tracks when the hardware transfer is complete:
```python
tid = warp_id * 32 + lane_id                 # 0..127 within the warpgroup
if tid == 0:  # exactly one thread starts TMA
    Tx.copy_async(Asmem, A[...], dispatch="tma")
    Tx.copy_async(Bsmem, B[...], dispatch="tma")
    T.ptx.mbarrier.arrive.expect_tx(tma_bar, byte_count)  # bytes expected from TMA
T.ptx.mbarrier.try_wait(tma_bar, phase)                  # wait before MMA reads SMEM
```

`tid` combines the warp ID and lane ID into a thread ID within the warpgroup, so `tid == 0` selects exactly one thread. If all four warps called `elect_sync()` directly, each warp would select one active lane and four threads would issue the TMA load. The code could instead guard on `warp_id == 0` before calling `elect_sync()`; using `tid == 0` is more direct here.

Step 4 still waits immediately after every TMA load, so load and compute do not yet overlap. The change at this stage is that address generation and tile movement move from the CTA threads to the TMA engine, reducing the number of copy instructions those threads execute. Step 5 adds a second SMEM stage for prefetching; full role-level overlap arrives in Step 7.

### TMA Load and Store Synchronization

After a TMA load is issued, the transfer continues on the TMA engine. `cta_sync()` synchronizes only the CTA threads and cannot determine whether this asynchronous transfer has finished. Before MMA reads the SMEM tile, it must therefore wait for the TMA load through an mbarrier.

The figure below presents this handoff as a top-to-bottom timeline. Its four lifelines represent the issuing thread, the TMA engine, the mbarrier, and the MMA that consumes the data. The diagram uses a simplified example in which the A and B tiles are 2048 bytes each, for a total transfer of 4096 bytes.

![TMA Async Load: Synchronization Flow](../img/tma_sync_flow.svg)

Steps 1 and 2 in the figure occur on the issuing thread. It launches one `copy_async` for A and one for B, then executes `arrive.expect_tx(4096)`. This operation reports one thread arrival to the mbarrier and registers the 4096 bytes of asynchronous transfer that remain. The pending arrival count is now zero, but the pending byte count is still 4096, so the barrier is not complete.

Step 3 is performed by the TMA engine. As A and B arrive in SMEM, the hardware uses `complete_tx` to reduce the pending byte count. Once both transfers finish, the byte count reaches zero. The consumer's `try_wait(phase)` can then pass in Step 4, and the MMA starts reading the completed A and B tiles in Step 5.

The kernel uses the same protocol with larger tiles. A and B each contain `128×64` fp16 elements and occupy 16384 bytes, so `arrive.expect_tx` registers 32768 bytes in total.

TMA stores use a different completion mechanism. After the threads write the result to `Dsmem` and synchronize, the `tid == 0` thread starts the asynchronous copy from `Dsmem` to GMEM. The kernel commits the store with `cp_async.bulk.commit_group()` and waits for it with `cp_async.bulk.wait_group(0)`. `Dsmem` cannot be overwritten or reused until that wait returns.

### Complete Kernel

The complete kernel folds the TMA load and store into the Step 3 structure, leaving the rest of that structure untouched. The imports are the same as before:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

It is wrapped in `hgemm_v4(M, N, K)`, a pattern we follow throughout: the wrapper keeps the shape-dependent constants and layouts right next to the kernel that uses them.

```python
def hgemm_v4(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    F16_SIZE = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation (now includes Dsmem for TMA store) ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((1,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # --- Barrier + TMEM init ---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
        if warp_id == 0:
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
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0

        # --- Inline helpers ---
        @T.inline
        def tma_load(k_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([0])
            })
            Tx.copy_async(Asmem[:, :],
                          A[m_st : m_st + BLK_M, k_st : k_st + BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[:, :],
                          B[n_st : n_st + BLK_N, k_st : k_st + BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([0]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE
            )

        @T.inline
        def mma(accum):
            Tx.gemm_async(
                tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                accum=accum, dispatch="tcgen05", cta_group=1
            )
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        # --- K-loop with TMA async ---
        tid = T.meta_var(warp_id * 32 + lane_id)
        for k in range(K_TILES):
            k_st = T.meta_var(k * BLK_K)

            # Single thread issues TMA load
            if tid == 0:
                tma_load(k_st)

            # Wait for TMA to finish; the mbarrier release carries SMEM
            # visibility to the subsequent MMA, so no extra fence is needed.
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([0]), phase_tma)

            # Single thread issues MMA
            if tid == 0:
                mma(accum=k != 0)

            # Wait for MMA to finish
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_tma ^= 1
            phase_mma ^= 1

        # --- TMA Store Writeback ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        # Read TMEM -> registers (async; wait.ld then cta_sync to ensure read completes)
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        # Cast fp32 -> fp16
        Tx.cast(Dreg_f16[:], Dreg[:])
        # Write registers -> Dsmem, flush, then sync
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        # TMA store: Dsmem -> GMEM. One selected thread starts the store and drains the
        # store group before Dsmem is reused.
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)

        # --- Deallocate TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

### TMA Configuration in the Kernel

Most of this kernel comes directly from Step 3. The following five settings define the TMA behavior:

- **TMA config**: `{"dispatch": "tma", "cta_group": 1, "mbar": tma_bar.ptr_to([0])}` tells `Tx.copy_async` to use TMA and to report load completion through `tma_bar`.

- **Byte count**: `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` is the number of bytes loaded by the two fp16 operand tiles. `arrive.expect_tx(...)` gives this count to the mbarrier.

- **mbarrier initialization**: `init(tma_bar.ptr_to([0]), 1)` creates the completion barrier used by the TMA load.

- **`@T.inline`**: `tma_load(...)` and `mma(...)` are helper functions. They are expanded into the kernel body at compile time and can use variables from the surrounding kernel.

- **TMA store synchronization**: The epilogue first writes fp16 rows into `Dsmem`. `fence.proxy_async` and `warpgroup_sync` make those thread-written SMEM values ready for the TMA store path. The store then uses `commit_group()` and `wait_group(0)` to wait for the SMEM-to-GMEM transfer to finish.

The data-movement path is now correct, but the schedule remains sequential: each load completes before its MMA begins, so the two engines still take turns. The next step keeps the TMA load and store paths and introduces reusable SMEM stages for prefetching.

(chap_software_pipeline)=
## Step 5: Software Pipeline (PIPE_DEPTH=2)

Step 4 cannot overlap load and compute because SMEM contains only one operand-tile pair. The next load has no independent destination; starting it early would overwrite data that the current MMA is still reading. Step 5 removes this storage conflict by double-buffering shared memory. The single-warpgroup loop still waits for each MMA before issuing the next TMA load, but it now has separate stages that can be prefetched and reused in a ring.

> **What this step changes: Layout**
> - Scope: unchanged, one warpgroup.
> - Layout: the single SMEM tile pair becomes a `PIPE_DEPTH`-stage ring buffer.
> - Dispatch: unchanged, TMA load and `tcgen05` MMA; this step adds prefetch and stage reuse, while full load/compute overlap arrives in Step 7.

### Pipeline Walkthrough

With `PIPE_DEPTH=2`, the kernel allocates two SMEM stages, giving the load and MMA paths different slots. This separation is required before data movement and computation can overlap, but the current single-warpgroup kernel still waits for the MMA before issuing the next TMA load. The figure shows the target schedule supported by the double buffer. Step 7 reaches this schedule after assigning TMA and MMA to separate roles.

![*Target schedule with `PIPE_DEPTH=2`*](../img/pipe_depth2.png)

At startup, two TMA loads fill both stages. The loop then waits for the current stage, runs MMA on it, and loads tile `k + PIPE_DEPTH` into the stage that has just become reusable. This establishes the ring buffer and prefetches the first two tiles.

Concretely, the code differs from Step 4 in four places:

1. `Asmem` and `Bsmem` gain a leading `PIPE_DEPTH` dimension, so each stage has its own SMEM storage.
2. `tma_bar` becomes an array with one mbarrier per stage.
3. Before the main K loop, the kernel prefetches the first two stages.
4. The K loop uses `stage = k % PIPE_DEPTH`: wait for the current stage, run MMA on it, then reuse that stage for `k + PIPE_DEPTH`.

### Pipeline Mechanics

**1. Prefetch**: before the main loop begins, load the first `PIPE_DEPTH` stages so that data is ready for the first iteration:
```python
for s in range(min(PIPE_DEPTH, K_TILES)):
    tma_load(s, s * BLK_K)
```

**2. Main loop**: for each K tile, wait for its stage, run MMA, then reuse the released stage to load the tile `PIPE_DEPTH` positions ahead:
```python
stage = k % PIPE_DEPTH
wait(tma_bar[stage], phase_tma)
mma(stage, accum)
wait(mma_bar[0], phase_mma)
phase_mma ^= 1
tma_load(stage, next_k * BLK_K)
```

**3. Phase management**: as described in the asynchronous synchronization chapter, an mbarrier changes phase after each completed round. The two local phase variables advance at different rates because one tracks a single MMA accumulator while the other tracks several SMEM stages.

All K iterations use `mma_bar.ptr_to([0])` to track the same TMEM accumulator, so `phase_mma` flips on every iteration. TMA assigns one barrier to each SMEM stage. A stage's barrier begins a new round only when the ring returns to that stage, so `phase_tma` flips after the stage index reaches the end of the ring:
```python
if stage == PIPE_DEPTH - 1:
    phase_tma ^= 1
```

**Trace the pipeline**: for `PIPE_DEPTH=2` and `K_TILES=5`, trace the main loop. For each `k`, record `stage`, the `phase_tma` and `phase_mma` values passed to the waits, and whether the iteration issues another prefetch. Where does `phase_tma` flip, and why do the final two iterations issue no prefetch?

### Complete Kernel

The complete kernel retains the Step 4 TMA load and store path, then adds the staged buffers and phase logic described above. The imports are unchanged:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

It is wrapped in `hgemm_v5(M, N, K)`. The `PIPE_DEPTH=2` constant sets the number of pipeline stages (two of them here, which is exactly double buffering):

```python
PIPE_DEPTH = 2

def hgemm_v5(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    # Double-buffered layouts: first dimension is pipeline stage
    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        # Double-buffered TMA barriers (one per stage), single MMA barrier
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # Initialize barriers: PIPE_DEPTH for TMA, 1 for MMA
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                for s in range(PIPE_DEPTH):
                    T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0

        @T.inline
        def tma_load(stage, k_offset):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        tid = T.meta_var(warp_id * 32 + lane_id)

        # === Prefetch: load first PIPE_DEPTH stages ===
        if tid == 0:
            for s in range(min(PIPE_DEPTH, K_TILES)):
                tma_load(s, s * BLK_K)

        # === Main loop ===
        for k in range(K_TILES):
            stage = k % PIPE_DEPTH

            # Wait for TMA to finish loading this stage
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)

            # MMA on this stage's data
            if tid == 0:
                mma(stage, accum=(k != 0))

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

            # Issue next prefetch load (k + PIPE_DEPTH)
            next_k = k + PIPE_DEPTH
            if next_k < K_TILES:
                if tid == 0:
                    tma_load(stage, next_k * BLK_K)

            # TMA phase flips when stage wraps around
            if stage == PIPE_DEPTH - 1:
                phase_tma ^= 1

        # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.cast(Dreg_f16[:], Dreg[:])
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)

        # Deallocate TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

(chap_persistent_kernel)=
## Step 6: Persistent Kernel + Tile Scheduler

The preceding steps optimize execution within one output tile. Step 6 turns to scheduling across tiles.

Step 5 launches one CTA per $128\times128$ output tile. A $4096\times4096$ output therefore requires 1024 CTAs. Each CTA performs its own initialization and exits after computing one tile.

A persistent kernel instead launches a fixed number of CTAs and lets each one process several tiles in sequence. This amortizes initialization across multiple tiles and moves tile assignment into the kernel, where the scheduler can choose an order that improves operand locality.

> **What this step changes: Scope**
> - Scope: a fixed pool of persistent CTAs, each looping over many output tiles via the scheduler.
> - Layout: unchanged, the same per-tile SMEM/TMEM/register path.
> - Dispatch: unchanged.

### Persistent Scheduling

A persistent kernel uses a smaller one-dimensional grid. This example sets `SM_COUNT=148` and launches 148 persistent CTAs. Each CTA obtains an output tile from the scheduler, computes it, and requests another until no tiles remain. `SM_COUNT` determines how many persistent CTAs the kernel launches. Occupancy and hardware scheduling determine how many can be resident at once and which SMs execute them; no CTA is permanently bound to a particular SM.

Because one CTA processes several tiles, it allocates TMEM, initializes its barriers, and creates scheduler state only once. Those resources remain in place until the CTA finishes all of its assigned work.

The scheduler also changes the logical tile order. `l2_group_size=8` groups eight consecutive output-tile rows along M. Within each group, tile IDs run down those eight rows for one N tile column before advancing to the next column. This places work that shares a B tile close together in the schedule and revisits the same group of A tiles over a short span. The CTAs still move their data independently, and the hardware may execute them in a different order, but this numbering makes L2 reuse more likely.

```python
bx = T.cta_id([SM_COUNT])  # 1D persistent grid

tile_scheduler = ClusterPersistentScheduler2D(
    "ts",
    num_m_tiles=M // BLK_M,
    num_n_tiles=N // BLK_N,
    l2_group_size=8,
    num_clusters=SM_COUNT
)
tile_scheduler.init(bx)
```

When a CTA starts its next output tile, it reuses the same TMA and MMA barriers. The locally stored phase parity must therefore match each barrier's current state.

With the current parameters, each output tile contains 64 K iterations. `mma_bar` is used 64 times, while each of the two TMA stage barriers is used 32 times. Because all of these counts are even, every barrier returns to its initial parity at the end of a tile, and the next tile may start its local phase values at zero:

```python
while tile_scheduler.valid():
    phase_tma: T.int32 = 0
    phase_mma: T.int32 = 0
    ...
```

If changing `K`, `BLK_K`, or `PIPE_DEPTH` makes any barrier run an odd number of rounds per output tile, its phase parity cannot simply be reset to zero. The wrapper below uses an assertion to restrict the parameter combinations supported by this implementation.

### Complete Kernel

Step 6 keeps the staged K-loop from Step 5 and places it inside an outer loop over output tiles. The only new dependency is the scheduler:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D
```

The launch grid now has `SM_COUNT` CTAs rather than one CTA for every `(M, N)` output tile. A `ClusterPersistentScheduler2D` assigns tiles to those persistent CTAs:

```python
SM_COUNT = 148  # Number of SMs on NVIDIA B200 GPU
PIPE_DEPTH = 2

def hgemm_v6(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    assert K % BLK_K == 0, "K must be divisible by BLK_K"
    K_TILES = K // BLK_K
    assert K_TILES % (2 * PIPE_DEPTH) == 0, (
        "K_TILES must be divisible by 2 * PIPE_DEPTH"
    )

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 1D grid: one CTA per SM (not a 2D grid anymore!)
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation (same as Step 5) ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # --- Barrier + TMEM init (same as Step 5) ---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            for s in range(PIPE_DEPTH):
                T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        # Tile scheduler: assigns tiles to CTAs in L2-friendly order
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts",
            num_m_tiles=M // BLK_M,
            num_n_tiles=N // BLK_N,
            l2_group_size=8,
            num_clusters=SM_COUNT
        )
        tile_scheduler.init(bx)

        tid = T.meta_var(warp_id * 32 + lane_id)

        @T.inline
        def tma_load(stage, k_offset, m_st, n_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        # === Outer loop: iterate over tiles ===
        while tile_scheduler.valid():
            # Get current tile position from scheduler
            m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
            n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

            # === Inner loop: same pipeline as Step 5 ===
            phase_tma: T.int32 = 0
            phase_mma: T.int32 = 0

            # Prefetch first PIPE_DEPTH stages
            if tid == 0:
                for s in range(min(PIPE_DEPTH, K_TILES)):
                    tma_load(s, s * BLK_K, m_st, n_st)

            # Main K-loop
            for k in range(K_TILES):
                stage = k % PIPE_DEPTH
                T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)
                if tid == 0:
                    mma(stage, accum=(k != 0))
                T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                phase_mma ^= 1
                next_k = k + PIPE_DEPTH
                if next_k < K_TILES:
                    if tid == 0:
                        tma_load(stage, next_k * BLK_K, m_st, n_st)
                if stage == PIPE_DEPTH - 1:
                    phase_tma ^= 1

            # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
            Dreg = T.alloc_local((BLK_N,), acc_type)
            Dreg_f16 = T.alloc_local((BLK_N,), d_type)
            Dreg_wg = Dreg.view(128, BLK_N,
                                layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
            Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
            T.ptx.tcgen05.wait.ld()
            T.cuda.cta_sync()
            Tx.cast(Dreg_f16[:], Dreg[:])
            Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.warpgroup_sync(10)
            if tid == 0:
                Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                              Dsmem[:, :], dispatch="tma")
                T.ptx.cp_async.bulk.commit_group()
                T.ptx.cp_async.bulk.wait_group(0)
            T.cuda.warpgroup_sync(10)

            T.cuda.cta_sync()
            tile_scheduler.next_tile()  # Move to next tile

        # Deallocate TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

## Exercises

1. In Step 4, `arrive.expect_tx` uses `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` bytes. What does the mbarrier wait for if this byte count is too small or too large?
2. In Step 5, why does each SMEM stage need its own TMA barrier instead of sharing one `tma_bar` for both stages?
3. In Step 6, a 4096 x 4096 output with `BLK_M=BLK_N=128` has how many output tiles? With `SM_COUNT=148`, how many tiles does each persistent CTA process on average?
