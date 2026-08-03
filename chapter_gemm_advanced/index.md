(chap_gemm_advanced)=
# Scaling GEMM with Warp Specialization and Clusters

:::{admonition} Overview
:class: overview

- Step 7 assigns TMA load, MMA, and writeback to separate warp roles, then connects data readiness and buffer reuse with four barriers.
- Step 8 uses cooperative MMA across two CTAs to compute a larger output tile, including cross-CTA operand access and barrier handoffs.
- Step 9 adds a second MMA consumer so that two groups of A blocks share the same staged B tile. Under this chapter's benchmark conditions, the final kernel matches the cuBLAS reference.
:::

The pipelined GEMM from the previous chapter ({ref}`chap_gemm_async`) already uses TMA, a software pipeline, and persistent scheduling, but the kernel still has only one warpgroup. It issues the TMA loads, waits for the A and B tiles, issues the MMA, and finally writes the result back. Although separate hardware units execute these operations, their control and synchronization remain concentrated in one warpgroup.

The SMEM ring and prefetch structure are already in place, but one warpgroup still controls every stage. While it waits for data, issues an MMA, or performs writeback, it cannot independently advance another part of the pipeline. Sustained overlap between data movement, matrix computation, and writeback requires assigning those jobs to separate warps.

This chapter broadens cooperation in three steps. Step 7 assigns TMA, MMA, and writeback to different warp roles. Step 8 lets two CTAs cooperate on a larger output tile. Step 9 adds a second MMA consumer. The GEMM itself does not change; the focus is how warps and CTAs divide the work and use barriers to hand off data and resources.


(chap_warp_specialization)=
## Step 7: Warp Specialization and Pipelining

In the single-warpgroup kernel, every thread follows the same load, compute, and writeback path. The Tensor Cores have no work while data is being loaded, and the TMA engine may sit idle during computation. Warp specialization assigns these jobs to different warps and uses a software pipeline to pass data between them, allowing several stages to run concurrently.

> **What this step changes: Scope**
> - Scope: one warpgroup walking load → MMA → writeback in order becomes three concurrent roles (TMA producer, MMA consumer, writeback) connected by full/empty barriers.
> - Layout: unchanged, same SMEM stages and TMEM accumulator as Step 6.
> - Dispatch: unchanged, TMA loads, `tcgen05` MMA.

The multi-stage SMEM pipeline and persistent `ClusterPersistentScheduler2D` carry over from Steps 5 and 6. Step 7 changes how that work is divided among warps and how the roles synchronize with one another.

### From Sequential to Concurrent

The figure below compares scheduling before and after warp specialization. The upper half uses the serial Step 4 timeline to summarize the unspecialized execution in Steps 4 through 6; the lower half shows the concurrent schedule in Step 7.

![Warp Specialization Timeline](../img/warp_specialization_timeline.png)

In the upper half, the same threads control both load and MMA, so one path can easily be idle while the other runs. Steps 5 and 6 add double buffering and persistent scheduling, but do not yet split load and compute into independent producer and consumer roles. In the lower half, the TMA producer prefetches the next tile while the MMA consumer computes the current tile, and writeback proceeds independently. Producer warp 3 can issue the next load while consumer warp 0 is still executing the current MMA.

The figure labels the full and empty states of the SMEM pipeline as `smem_pipe.full` and `smem_pipe.empty`. In the implementation below, these correspond to `tma2mma` and `mma2tma`.

Two barriers hand the SMEM buffer between load and MMA:

- **`tma2mma`** (TMA → MMA): signals that the loaded SMEM data is ready for MMA to consume.
- **`mma2tma`** (MMA → TMA): signals that MMA has finished reading a buffer, so TMA can reuse it for the next load.

The `mma2tma` arrows skip one stage because of the ring-buffer reuse order. With `PIPE_DEPTH=2`, TMA Load k=0 fills stage 0 and TMA Load k=1 fills stage 1. Once MMA Compute k=0 finishes reading stage 0, the next operation that needs that slot is TMA Load k=2, not k=1. The signal from MMA Compute k=0 therefore releases stage 0 for TMA Load k=2.

### Warp Roles

With `WG_NUMBER=2`, the kernel uses two warpgroups and assigns load, compute, and writeback as follows:

| Actor | Location | Job |
|-------|----------|-----|
| **TMA Producer** | Warpgroup 1, warp 3 | Continuously loads A and B tiles via TMA |
| **MMA Consumer** | Warpgroup 1, warp 0 | Runs MMA as soon as data is ready |
| **Writeback** | Warpgroup 0 (all warps) | Reads TMEM results, writes to GMEM |

### Four Barriers

The three concurrent roles communicate through four barriers. The forward path, TMA → MMA → Writeback, reports that data is ready. The reverse path, Writeback → MMA → TMA, releases a buffer for reuse. Barrier names follow `source2destination`; for example, `tma2mma` carries the notification from TMA to MMA.

| Barrier | Type | Direction | Meaning |
|---------|------|-----------|---------|
| **tma2mma** | `TMABar` | TMA -> MMA | "SMEM data is ready" |
| **mma2tma** | `TCGen05Bar` | MMA -> TMA | "SMEM buffer can be reused" |
| **mma2ld** | `TCGen05Bar` | MMA -> Writeback | "TMEM results are ready" |
| **ld2mma** | `MBarrier` | Writeback -> MMA | "TMEM is free for next tile" |

The barrier type depends on how its producer reports completion. **TMA loads** use `TMABar`, an mbarrier with byte counting that the TMA hardware updates after the transfer completes. **TMA stores** are tracked by the issuing thread through an async group: `cp_async.bulk.commit_group()` submits the group, and `wait_group(0)` waits for the write to finish. **MMA operations** use `TCGen05Bar`; `tcgen05.commit()` updates the barrier when the MMA completes.

In these calls, `cta_mask=0` selects the non-multicast form used by the single-CTA kernel. Step 8 forms a cluster and uses `cta_mask=3` to multicast the notification to both CTAs.

### PipelineState

The four barriers indicate when a buffer is available. `PipelineState` records which stage a role is using and which phase of that stage it should wait for. Maintaining both values by hand is prone to off-by-one errors that can deadlock the kernel, so `PipelineState` keeps them together:

```python
tma_ps = PipelineState(PIPE_DEPTH, phase=1)   # Producer starts ready (phase=1)
# tma_ps.stage = current stage index
# tma_ps.phase = current phase (0 or 1)
tma_ps.advance()                          # Advance to next stage
```

The initial `phase` determines whether a role's first `wait` passes or blocks. The two ends of the pipeline need opposite initial states:
- `phase=1` (producer): the first `wait(phase=1)` sees the barrier at phase 0 and passes immediately. The buffers start empty, so the producer can begin filling them.

- `phase=0` (consumer): the first `wait(phase=0)` sees the barrier at phase 0 and blocks. No data is available until the producer finishes the first load.

If both ends use the same initial phase, the kernel may deadlock or continue before the data is ready.

### Using a Named Barrier Within One Warpgroup

In Step 7, all 128 threads in Warpgroup 0 perform writeback. They first write their register values to `Dsmem`, wait until the complete tile is present, and then allow one thread to issue the TMA store. This requires synchronization within Warpgroup 0, but `cta_sync()` cannot be used inside the branch: the other warpgroup is executing the producer and MMA consumer paths and will never reach that synchronization point. Waiting for the entire CTA would deadlock.

`warpgroup_sync(10)` lowers to:

```text
bar.sync 10, 128
```

Here, `10` is the named barrier ID and `128` is the number of thread arrivals required. The instruction does not identify the warpgroup automatically. It synchronizes Warpgroup 0 here because only those 128 threads execute this code, and all of them use the same barrier ID. The first `warpgroup_sync(10)` ensures that `Dsmem` is complete. The second ensures that the selected thread has waited for the TMA store before the remaining threads continue to the next tile.

Each CTA has 16 named barrier slots, numbered 0 through 15. Threads participating in one synchronization must use the same ID, while independent synchronizations need different IDs. Step 7 has only one writeback warpgroup and uses ID 10. Step 9 has two writeback warpgroups and calls `warpgroup_sync(wg_id + 10)`, assigning IDs 10 and 11 so their arrivals are not counted together.

### Epilogue (Writeback)

In Step 7, `BLK_N=128`, so the writeback warpgroup can read the entire TMEM tile into registers in one pass and issue one TMA store. The sequence is:

1. Wait for MMA with `mma2ld.wait(phase)`, then execute `T.ptx.tcgen05.fence.after_thread_sync()` to order the subsequent `tcgen05.ld` after the cross-thread completion notification.
2. Read TMEM into registers. Each thread receives 128 fp32 values; the warpgroup issues `Tx.copy_async(reg_wg, tmem[:, :BLK_N])` and waits for the load with `T.ptx.tcgen05.wait.ld()`.
3. Have all 128 writeback threads execute `ld2mma.arrive(0, cta_id=0, pred=True)`, indicating that the TMEM region can be used by the next tile. `cta_id=0` selects the current CTA's local barrier, and `pred=True` makes every writeback thread report an arrival. Step 8 uses `cta_mask` to notify both CTAs in a cluster.
4. Convert fp32 to fp16 in registers.
5. Write the registers to `Dsmem`, then execute `fence.proxy_async("shared::cta")` and `warpgroup_sync(10)`.
6. Use TMA to store `Dsmem` to GMEM, with `cp_async.bulk.commit_group()` and `wait_group(0)` tracking completion.

The mbarrier wait and `tcgen05.wait.ld()` wait for different operations. The former confirms that MMA has completed, `fence.after_thread_sync()` establishes cross-thread `tcgen05` ordering, and the latter confirms that the asynchronous TMEM load has populated its destination registers.

### Complete Kernel

The complete Step 7 kernel combines the role assignment, four barriers, `PipelineState`, and writeback path described above. It retains the persistent scheduler from Step 6 and uses `PIPE_DEPTH=2`, the minimum depth needed to overlap load and compute. A deeper pipeline can hide more memory latency, but also consumes more SMEM.

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.pipeline import TMABar, TCGen05Bar, MBarrier, PipelineState
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D

SM_COUNT = 148  # Number of SMs on NVIDIA B200 GPU
F16_SIZE = 2

def hgemm_v7(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    WG_NUMBER = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, 1)
        ld2mma  = MBarrier(pool, 1)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)

        # --- Barrier init ---
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128)   # all 128 Warpgroup 0 threads arrive
        pool.commit()

        # --- TMEM alloc + fence ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // BLK_M, num_n_tiles=N // BLK_N,
            l2_group_size=8, num_clusters=SM_COUNT)
        tile_scheduler.init(bx)
        m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
        n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

        # =============================================
        # Warpgroup 1: TMA Producer (warp 3) + MMA Consumer (warp 0)
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                # === TMA Producer ===
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=1,
                                  mbar=tma2mma.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=1,
                                  mbar=tma2mma.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            tma2mma.arrive(tma_ps.stage,
                                           (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                # === MMA Consumer ===
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        # Wait for TMEM to be free from previous tile's writeback
                        ld2mma.wait(ld_ps.stage, ld_ps.phase)
                        ld_ps.advance()

                        for k in range(K_TILES):
                            tma2mma.wait(mma_ps.stage, mma_ps.phase)
                            Tx.gemm_async(
                                tmem[:, :BLK_N],
                                Asmem[mma_ps.stage, :, :],
                                Bsmem[mma_ps.stage, :, :],
                                accum=(k != 0), dispatch="tcgen05", cta_group=1)
                            mma2tma.arrive(mma_ps.stage, cta_group=1, cta_mask=0)
                            mma_ps.advance()

                        # Signal results ready for writeback
                        mma2ld.arrive(0, cta_group=1, cta_mask=0)
                        tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0: Writeback
        # =============================================
        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((BLK_N,), d_type)

            while tile_scheduler.valid():
                # Wait for MMA results
                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                # Read TMEM -> registers (warpgroup scope)
                reg = T.alloc_local((BLK_N,), acc_type)
                reg_wg = reg.view(128, BLK_N,
                    layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
                Tx.wg.copy_async(reg_wg[:], tmem[:, :BLK_N])
                T.ptx.tcgen05.wait.ld()

                # Signal TMEM free (all 128 threads arrive)
                ld2mma.arrive(0, cta_id=0, pred=True)

                # Cast fp32 -> fp16
                Tx.cast(reg_f16[:], reg[:])

                # Write to Dsmem + TMA store
                Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                T.ptx.fence.proxy_async("shared::cta")
                T.cuda.warpgroup_sync(10)
                if warp_id == 0:
                    if lane_id == 0:
                        Tx.copy_async(D[m_st:m_st+BLK_M, n_st:n_st+BLK_N],
                                      Dsmem[:, :], dispatch="tma")
                        T.ptx.cp_async.bulk.commit_group()
                        T.ptx.cp_async.bulk.wait_group(0)
                T.cuda.warpgroup_sync(10)

                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

### Checking Barrier Handoffs

Step 7 connects three roles with four barriers. If the kernel waits indefinitely, inspect each barrier in turn: which role waits, which role arrives, and whether the initialized arrival count matches the number of notifications. If the kernel finishes but produces the wrong result, check whether a consumer reads data before the required wait and fence have completed.

Also verify when each storage region becomes reusable. TMA may overwrite an `Asmem` or `Bsmem` stage only after `mma2tma` completes. MMA may overwrite the TMEM accumulator only after `ld2mma` completes. A later writeback may reuse `Dsmem` only after the previous TMA store finishes. Following these handoffs narrows a deadlock or wrong result to one data transition instead of the entire pipeline.

**Trace the barriers**: follow one K tile through `tma2mma`, `mma2tma`, `mma2ld`, and `ld2mma`. For each barrier, identify who waits, who arrives, which data becomes safe to read, and which buffer becomes reusable.

### SMEM Cost of Pipeline Depth

`PIPE_DEPTH=2` provides two A/B stage pairs: while MMA reads one pair, TMA can fill the other. Increasing the depth lets the producer prepare more tiles in advance, but every additional stage allocates another `Asmem` and `Bsmem` pair.

With `BLK_M=BLK_N=128`, `BLK_K=64`, and fp16 operands, each stage uses:

```text
(128×64 + 128×64) × 2 bytes = 32 KB
```

The `Dsmem` writeback buffer requires another 32 KB. `PIPE_DEPTH=4` therefore uses roughly `4×32+32=160 KB`, while `PIPE_DEPTH=6` uses about `6×32+32=224 KB`, excluding the small amount of metadata used by barriers and other state. A B200 SM provides 228 KB of shared memory, so depth 6 nearly exhausts the available capacity. A deeper pipeline is not necessarily faster and may make the current tile shape impossible to launch.

(chap_cta_cluster)=
## Step 8: Two-CTA Cluster

Step 7 coordinates several roles within one CTA. Step 8 extends that cooperation to a cluster of two CTAs.

> **What this step changes: Scope + Layout + Dispatch**
> - Scope: the cooperating scope now spans two CTAs in a cluster, not one.
> - Layout: A and B slices reside in the SMEM of both CTAs, while the accumulator spans their two TMEM spaces.
> - Dispatch: `Tx.gemm_async` uses `cta_group=2` to issue a two-CTA cooperative MMA, and `cta_mask=3` sends completion notifications to both CTAs.

### How A and B Are Split Across the CTAs

The figure below divides one $256\times256$ output tile between two CTAs. Start with the `Asmem` blocks: CTA 0 loads rows 0–127 of A, while CTA 1 loads rows 128–255. These slices determine the output rows owned by each CTA, so the two orange regions are `D[0:128, 0:256]` and `D[128:256, 0:256]`.

Now consider the two `Bsmem` blocks. B is stored with shape `N×K`, so the stored-B rows loaded by CTA 0 and CTA 1 become the first and second groups of 128 output columns after `B.T`. Each CTA must compute all 256 columns for its own 128 output rows. The cooperative MMA therefore follows the cross-CTA reads in the center of the figure to access the other CTA's `Bsmem` as well. Each A slice is multiplied by both B slices.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/cta_cluster.html" title="A 2-CTA cluster: cooperative MMA via cross-CTA SMEM read" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Click either `Asmem` or `Bsmem`, or the cross-CTA reads in the center, to see how each part participates in the cooperative MMA.*

The tile scheduler returns `(m_idx, n_idx)` for one $256\times256$ cluster output tile. Let its upper-left corner be `m_base = m_idx * 256` and `n_base = n_idx * 256`. The two CTAs divide the work as follows:

| CTA | A slice loaded | Stored-B slice loaded | D region written |
|-----|----------------|-----------------------|------------------|
| CTA 0 | `A[m_base:m_base+128, :]` | `B[n_base:n_base+128, :]` | `D[m_base:m_base+128, n_base:n_base+256]` |
| CTA 1 | `A[m_base+128:m_base+256, :]` | `B[n_base+128:n_base+256, :]` | `D[m_base+128:m_base+256, n_base:n_base+256]` |

In Step 7, one CTA loads one $128\times K$ slice of A and one $128\times K$ slice of B to compute a $128\times128$ output tile. The cluster now loads two slices from each operand, doubling the staged operand data, while the $256\times256$ output tile contains four times as many elements. Because both A slices are multiplied by both B slices, each staged operand participates in roughly twice as much computation.

### Tile Address Calculation

The scheduler now operates on $256\times256$ cluster tiles, so the numbers of tiles along M and N are `M // 256` and `N // 256`.

`cbx` is the CTA's position in the cluster and is either 0 or 1. Starting from `m_base` and `n_base`, each CTA uses `cbx` to select the A and B slices it loads:

```python
cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
m_st = m_base + cbx * BLK_M
n_st = n_base + cbx * BLK_N
```

CTA 0 starts at `m_base` and `n_base`; CTA 1 advances each address by 128 rows. `m_st` is also the first output row written by that CTA. `n_st` only selects the stored-B rows that the CTA contributes to the cooperative MMA.

Both B slices participate in the MMA, so each CTA receives all 256 output columns for its own 128 rows. The epilogue writes those columns in two 128-column chunks, selected by `no=0,1`:

```python
n_st_epi = n_base + no * BLK_N
```

There is no `cbx` in this expression: `cbx` selects the B slice loaded by the CTA, not the output columns it writes.

### Data Handoffs Within the Cluster

The two CTAs have separate SMEM spaces and barriers. A cooperative MMA must wait until both sets of A and B slices are ready, then notify both CTAs when the relevant results or buffers can be used. This requires three handoffs.

**TMA → MMA.** This implementation uses CTA 0's `tma2mma` barrier to track the TMA loads issued by both CTAs. Both sides refer to it through the same remote view:

```python
tma2mma_cta0 = tma2mma.remote_view(0)
```

The TMA loads from both CTAs report completion to this barrier. The selected producer thread in CTA 0 registers the combined byte count for both CTAs. Each CTA loads one `BLK_M×BLK_K` A slice and one `BLK_N×BLK_K` B slice, so the count is:

```python
CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE
```

The MMA consumer in CTA 0 can proceed only after all of these bytes have been transferred.

**MMA → TMA and writeback.** A cooperative MMA is issued once. The `if cbx == 0:` guard keeps only the MMA path in CTA 0, where one selected thread issues the operation with `cta_group=2`:

```python
if cbx == 0:
    Tx.gemm_async(..., cta_group=2)
```

The hardware reads SMEM from both CTAs and updates the TMEM accumulator on both sides. After each asynchronous MMA in the K-loop, the same thread registers a completion notification through `mma2tma`. After the entire K-loop has been issued, it registers the final accumulator notification through `mma2ld`:

```python
for k in range(K_TILES):
    Tx.gemm_async(..., cta_group=2)
    mma2tma.arrive(mma_ps.stage, cta_group=2, cta_mask=3)

mma2ld.arrive(0, cta_group=2, cta_mask=3)
```

`cta_mask=3` is binary `11`, so both CTA 0 and CTA 1 receive the notification. Once an MMA finishes, `mma2tma` allows the TMA producers on both sides to reuse the SMEM stage it consumed. Once the K-loop finishes, `mma2ld` tells the writeback warpgroup in each CTA that the TMEM accumulator is ready.

**Writeback → the next output tile.** After the writeback warpgroups have finished using TMEM, 128 threads in each CTA arrive on CTA 0's `ld2mma` barrier. Its expected arrival count is therefore `128 * CTA_GROUP`, or 256. Only after all arrivals have been received may the next output tile reuse that TMEM region.

TMEM allocation and deallocation also use `cta_group=2`. Before deallocation, `cluster_sync()` ensures that both CTAs have finished accessing TMEM. The writeback path divides the 256 output columns into two 128-column chunks. It loads and stores one half at a time, so each thread does not need to keep 256 fp32 values live at once.


### Complete Kernel

The complete kernel combines the tile partition, cooperative MMA, and cross-CTA barrier protocol:

```python
def hgemm_v8(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_M, MMA_N = 256, 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    WG_NUMBER = 2
    F16_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, 128))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, 1)
        ld2mma  = MBarrier(pool, 1)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, 128), d_type, layout=D_layout)

        # --- Barrier init ---
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128 * CTA_GROUP)  # both CTAs' writeback threads
        pool.commit()

        # --- TMEM alloc (cooperative) ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler (cluster tiles) ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // 256, num_n_tiles=N // 256,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
        tile_scheduler.init(bx // CTA_GROUP)
        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
        n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

        # --- Cross-CTA barrier view ---
        tma2mma_cta0 = tma2mma.remote_view(0)

        # =============================================
        # Warpgroup 1: TMA Producer (warp 3) + MMA Consumer (warp 0)
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_ps.stage,
                                    CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(ld_ps.stage, ld_ps.phase)
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.gemm_async(
                                    tmem[:, :MMA_N],
                                    Asmem[mma_ps.stage, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_ps.advance()

                            mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)
                            tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0: Writeback (256 columns in 2 x 128-column chunks)
        # =============================================
        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((128,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                for no in T.unroll(2):  # 2 chunks of 128 columns = 256 total
                    reg = T.alloc_local((128,), acc_type)
                    reg_wg = reg.view(128, 128,
                        layout=TileLayout(S[(128, 128) : (1@tid_in_wg, 1)]))
                    Tx.wg.copy_async(reg_wg[:], tmem[:, no * 128:(no + 1) * 128])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(10)
                    if warp_id == 0:
                        if lane_id == 0:
                            n_st_epi = T.meta_var(n_idx * 256 + no * 128)
                            Tx.copy_async(D[m_st:m_st+BLK_M, n_st_epi:n_st_epi+128],
                                          Dsmem[:, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(10)

                ld2mma.arrive(0, cta_id=0, pred=True)
                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

(chap_multi_consumer)=
## Step 9: Multi-Consumer Warp Specialization

In Step 8, one MMA consumer uses the A and B slices from both CTAs to compute a $256\times256$ cluster output tile. Step 9 keeps the same B slices, loads one additional A slice in each CTA, and adds a second MMA consumer. Consumer 0 computes the first 256 output rows, while consumer 1 computes the next 256 rows; both cover the same 256 columns. The cluster output therefore grows along M from $256\times256$ to $512\times256$, without increasing the B data loaded per stage.

> **What this step changes: Scope + Layout**
> - Scope: CTA 0 now has two consumer warps that issue MMA operations, selected by `warp_id`.
> - Layout: A gains a consumer axis, and TMEM is divided into two accumulator ranges; both consumers reuse the same staged B tile.
> - Dispatch: the kernel still uses `tcgen05` with `cta_group=2`, but issues a separate cooperative MMA for each consumer.

### Multi-Consumer Structure

Adding the second consumer requires two MMA issue warps and two writeback warpgroups, one for each accumulator. A single thread selected by `elect_sync()` still issues the instruction within each MMA warp. With `NUM_CONSUMER=2` and `WG_NUMBER=3`, the roles are assigned as follows:

| Warpgroup | Warp | Role |
|-----------|------|------|
| **WG 2** | warp 0 | MMA consumer 0: a selected thread in CTA 0 issues `Asmem[..., 0] × B` and writes TMEM cols `[0:256]` |
| **WG 2** | warp 1 | MMA consumer 1: a selected thread in CTA 0 issues `Asmem[..., 1] × B` and writes TMEM cols `[256:512]` |
| **WG 2** | warp 2 | Unused in this version |
| **WG 2** | warp 3 | TMA producer: each CTA loads its local two A blocks and one B block |
| **WG 0** | all warps | Each CTA writes its local output rows for consumer 0, reading TMEM `[0:256]` |
| **WG 1** | all warps | Each CTA writes its local output rows for consumer 1, reading TMEM `[256:512]` |

The consumers need different A blocks because they compute different output rows. They use the same B slices because both sets of results cover the same output columns. One set of staged B slices can therefore participate in two cooperative MMAs, roughly halving the B-load cost relative to the amount of computation.

### Adding the Second MMA Consumer

Adding the second consumer requires coordinated changes to the layouts, barriers, and tile scheduler.

**Extend the operand and accumulator layouts.** `Asmem` gains a dimension of length `NUM_CONSUMER`, allowing each stage to hold two A blocks:

```python
Asmem = pool.alloc(
    (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ...
)
```

The TMA producer in each CTA loads `Asmem[stage, 0]`, `Asmem[stage, 1]`, and its local `Bsmem[stage]`. The consumers share the B slices, so no second B block is needed. The total byte count registered for the TMA loads from both CTAs is:

```python
CTA_GROUP * (
    NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K
) * F16_SIZE
```

The two MMA warps use `warp_id` to select their A block and write to TMEM cols `[0:256]` and `[256:512]`, respectively.

**Update the barriers.** Both consumers must finish reading the current SMEM stage before the TMA producer can overwrite it, so the expected arrival count for `mma2tma` becomes `NUM_CONSUMER`. The `mma2ld` and `ld2mma` objects each contain two slots. For `mma2ld`, consumer 0 multicasts completion through slot 0 to WG 0 in both CTAs, while consumer 1 uses slot 1 for WG 1. The matching `ld2mma` slot then collects the writeback arrivals from both CTAs before that consumer may reuse its TMEM range. The MMA side selects a slot with `warp_id`; the writeback side selects the same slot with `wg_id`.

**Adjust the scheduler and writeback.** The cluster output tile is now $512\times256$, so the scheduler uses:

```python
num_m_tiles = M // (NUM_CONSUMER * CTA_GROUP * BLK_M)  # M // 512
num_n_tiles = N // (CTA_GROUP * BLK_N)                 # N // 256
```

`m_st` points to the A rows for consumer 0. Consumer `c` starts at `m_st + c * CTA_GROUP * BLK_M`, so consumer 1 advances by another 256 rows. During writeback, each consumer divides its 256 columns into four `EPI_N=64` chunks. Each thread converts and stores 64 columns at a time before moving to the next chunk.


### Complete Kernel

The complete kernel retains the two-CTA cluster from Step 8 and adds a second A tile, MMA consumer, and writeback warpgroup. Both consumers share the same B tile and update separate TMEM accumulator ranges.

```python
def hgemm_v9(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    NUM_CONSUMER = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N = BLK_N * CTA_GROUP   # 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    WG_NUMBER = 3
    F16_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (NUM_CONSUMER, BLK_M, EPI_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, NUM_CONSUMER)   # depth=2, one slot per consumer
        ld2mma  = MBarrier(pool, NUM_CONSUMER)     # depth=2, one slot per consumer
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((NUM_CONSUMER, BLK_M, EPI_N), d_type, layout=D_layout)

        # --- Barrier init ---
        tma2mma.init(1)
        mma2tma.init(NUM_CONSUMER)  # each stage expects 2 arrivals
        mma2ld.init(1)              # each slot gets 1 arrival
        ld2mma.init(128 * CTA_GROUP)  # both CTAs' writeback threads
        pool.commit()

        # --- TMEM alloc (cooperative) ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler (512x256 cluster tiles) ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // 256 // NUM_CONSUMER, num_n_tiles=N // 256,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
        tile_scheduler.init(bx // CTA_GROUP)
        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
        n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

        tma2mma_cta0 = tma2mma.remote_view(0)

        # =============================================
        # Warpgroup 2: TMA Producer (warp 3) + 2 MMA Consumers (warp 0, 1)
        # =============================================
        if wg_id == 2:
            if warp_id == 3:
                # === TMA Producer: loads 2 A blocks + 1 B block per stage ===
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    m_st_c1 = T.meta_var(m_st + CTA_GROUP * BLK_M)
                    Tx.copy_async(Asmem[tma_ps.stage, 0, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Asmem[tma_ps.stage, 1, :, :],
                                  A[m_st_c1:m_st_c1+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_ps.stage,
                                    CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id < NUM_CONSUMER:
                # === MMA Consumer: warp_id selects A block and TMEM range ===
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(warp_id, ld_ps.phase)
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.gemm_async(
                                    tmem[:, warp_id * MMA_N:warp_id * MMA_N + MMA_N],
                                    Asmem[mma_ps.stage, warp_id, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_ps.advance()

                            mma2ld.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)
                            tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0/1: Writeback (each reads its consumer's TMEM range)
        # =============================================
        elif wg_id < NUM_CONSUMER:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((EPI_N,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(wg_id, wb_ps.phase)  # wait for THIS consumer
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                # Read TMEM in EPI_N=64 column chunks (4 iterations for 256 cols)
                for i in T.unroll(MMA_N // EPI_N):
                    reg = T.alloc_local((EPI_N,), acc_type)
                    reg_wg = reg.view(128, EPI_N,
                        layout=TileLayout(S[(128, EPI_N) : (1@tid_in_wg, 1)]))
                    col_st = T.meta_var(wg_id * MMA_N + i * EPI_N)
                    col_end = T.meta_var(wg_id * MMA_N + i * EPI_N + EPI_N)
                    Tx.wg.copy_async(reg_wg[:], tmem[:, col_st:col_end])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(Dsmem[wg_id, warp_id * 32 + lane_id, :], reg_f16[:])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if warp_id == 0:
                        if lane_id == 0:
                            m_st_epi = T.meta_var(
                                (m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M)
                            n_st_epi = T.meta_var(n_idx * MMA_N + i * EPI_N)
                            Tx.copy_async(
                                D[m_st_epi:m_st_epi+BLK_M, n_st_epi:n_st_epi+EPI_N],
                                Dsmem[wg_id, :, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(wg_id + 10)

                ld2mma.arrive(wg_id, cta_id=0, pred=True)
                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

## End-to-End Results

The table below follows the progression from the naive baseline to the warp-specialized cluster kernel, with cuBLAS included as a reference. The measurements use an NVIDIA B200 with `M=N=K=4096`, fp16 inputs, locked clocks, and 1,000 timed iterations per measured version:

| Step | Technique | Time | Speedup |
|------|-----------|------|---------|
| 1 | Sync load + MMA | 70 ms | 1× |
| 2 | K-loop accumulation | — | — |
| 3 | Spatial tiling | 53.6 ms | ~1.3× |
| 4 | TMA async load | 0.49 ms | ~142× |
| 5 | Software pipeline | — | — |
| 6 | Persistent kernel | — | — |
| 7 | Warp specialization | 0.23 ms | ~309× |
| 8 | Two-CTA cluster | 0.104 ms | ~676× |
| 9 | Multi-consumer | 0.094 ms | ~744× |
| --- | cuBLAS (reference) | 0.094 ms | ~744× |

Every row with a measured time uses the same `M=N=K=4096` problem, so those rows can be compared directly. Times are rounded for display, while the speedups are calculated from the underlying measurements. The 70 ms in Step 1 comes from a full-matrix baseline with the same sequential data path; it is not a run of the single-tile `hgemm_v1` from {ref}`chap_gemm_basics`. The introductory chapter uses smaller problems to explain Steps 1 through 3, while the Step 1 and Step 3 rows here measure the corresponding full-matrix implementations.

Step 2 still computes only one output tile, so it is not directly comparable with the full-matrix results. Steps 5 and 6 are intermediate versions between the TMA-load kernel and the warp-specialized kernel; their mechanisms are retained in Step 7. The table therefore shows only the endpoints of that interval. Steps 2, 5, and 6 use dashes and have no standalone speedup.

These numbers come from one B200 reference run. They are intended to compare the versions in this tutorial under the same conditions, rather than to represent peak performance for other problem sizes or environments.

The measured versions support four comparisons:

1. **Step 1 → Step 4**: runtime falls from 70 ms to 0.49 ms, a cumulative speedup of roughly 142×. This interval also adds the K-loop, spatial tiling, multi-CTA parallelism, and TMA, so the entire gain cannot be attributed to TMA alone.
2. **Step 4 → Step 7**: software pipelining, persistent scheduling, and warp specialization reduce runtime from 0.49 ms to 0.23 ms, or about 2.2×.
3. **Step 7 → Step 8**: the two-CTA cooperative MMA increases reuse of the staged operands, reducing runtime from 0.23 ms to 0.104 ms, another gain of about 2.2×.
4. **Step 8 → Step 9**: the second MMA consumer reuses the same staged B slices, reducing runtime from 0.104 ms to 0.094 ms, an improvement of about 10%.

The figure compares the measured versions with the cuBLAS reference:

![GEMM Optimization Journey](../img/gemm_perf.png)

Across these nine versions, the optimization has two recurring goals: keep the Tensor Core from waiting for data, and perform more computation with each tile brought on chip.

Steps 1 through 3 begin with one $128\times128$ output tile, add the K-loop, and tile the M and N dimensions to cover a full matrix. Steps 4 through 7 then improve the supply of data: TMA moves the tiles, double buffering prepares the next K tile, the persistent scheduler keeps CTAs working, and warp specialization allows load, MMA, and writeback to advance concurrently. By Step 7, the Tensor Core no longer has to wait for an entire load or writeback path to finish before computation can continue.

The final two steps improve reuse. The two-CTA cluster computes a larger output tile, allowing each A and B tile to participate in more multiply-adds. The second MMA consumer then lets two A blocks share the same B tile. Each transfer from GMEM consequently supports more on-chip computation.

In this B200 run, combining TMA, software pipelining, persistent scheduling, warp specialization, and cluster-level reuse reduces runtime from 70 ms to 0.094 ms, matching cuBLAS under the same test conditions. The result comes from coordinating several optimizations across data movement, execution overlap, and on-chip reuse rather than from any single mechanism.


## Exercises

1. What happens if you set the initial `phase` to `0` for both the TMA and MMA `PipelineState` in Step 7? Draw the deadlock scenario.
2. With `cta_group=2` in Step 8, the TMA arrive byte count is `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`. Why multiply by `CTA_GROUP` when each CTA loads its own data?
3. In Step 9, each consumer handles different M rows but the same B tile. Why is sharing B (not A) the right choice?
