(chap_flash_attention)=
# Flash Attention 4

:::{admonition} Overview
:class: overview

- FlashAttention processes `K` and `V` in blocks and maintains row-wise state with online softmax, avoiding a full score-matrix write to GMEM.
- FA4 reorganizes the pipeline for Blackwell: separate roles execute QKᵀ MMA, softmax, PV MMA, and output correction, while TMEM carries `S`, `P`, and `O` between them.
- Conditional rescaling avoids many TMEM round trips for `O`, while hardware `exp2` and an FMA-based polynomial approximation share the exponential work.
:::

Attention is a core operation in Transformer models and one of the main performance and memory bottlenecks for long sequences. This chapter studies Flash Attention 4 (FA4), an attention forward kernel optimized for Blackwell GPUs. Given query `Q`, key `K`, and value `V`, it computes:

$$O = \text{softmax}(QK^{\top} / \sqrt{d})V$$

Here, `QKᵀ` gives the attention scores between queries and keys, and $d$ is the dimension of each attention head. Dividing by $\sqrt{d}$ keeps dot-product magnitudes under control; row-wise softmax converts the scores into attention weights, and multiplying by `V` produces `O`. A direct implementation materializes the full score matrix, creating substantial memory traffic as the sequence length grows.

FlashAttention divides the computation into blocks and keeps only the current tiles and per-row softmax state on chip, avoiding the full score matrix while preserving the result of standard attention. Successive versions differ mainly in how this algorithm maps to the GPU. FlashAttention-2 improved work partitioning across thread blocks and warps. FlashAttention-3 used TMA, WGMMA, and warp specialization on Hopper to interleave data movement, the two MMAs, and softmax. FA4 targets Blackwell and reorganizes the pipeline around `tcgen05` and TMEM.

The [FA4 paper](https://arxiv.org/abs/2603.05451) notes that, from Hopper to Blackwell, Tensor Core matrix-multiply throughput grew faster than exponential throughput, general-purpose floating-point throughput, and on-chip data movement. As the QKᵀ and PV MMAs become shorter, softmax exponentials, `O` rescaling, and TMEM/SMEM traffic account for a larger share of execution time. FA4 therefore cannot be obtained by simply replacing FA3's WGMMA instructions with `tcgen05.mma`; it must also reorganize where intermediate results live, which warpgroups execute each stage, and how those stages overlap.

The preceding GEMM chapters introduced these Blackwell hardware paths: TMA moves tiles, `tcgen05` executes MMA, and TMEM holds accumulators. FA4 connects them into a different computation chain: a QKᵀ MMA computes the score tile `S = QKᵀ`, CUDA cores turn `S` into the unnormalized attention-weight tile `P`, and a PV MMA uses `P` and `V` to update the output accumulator `O`. Following the terminology in the FA4 paper, this chapter calls these operations the QKᵀ MMA and the PV MMA. Whenever softmax changes its exponent reference, the existing `O` in TMEM must first be converted to the new scale.

This chapter is organized around three questions: how TMEM connects the two MMAs with softmax, how conditional rescaling reduces the number of `O` rescaling operations, and how multiple floating-point execution paths share exponential evaluation. We first derive the mathematical dependencies, then examine the TMEM layouts of `S`, `P`, and `O`, the division of work among warpgroups, and the barriers that hand off data and storage resources.

## Algorithm Structure

The matrix formula above describes the complete attention computation. For self-attention with sequence length $L$, each head has an $L\times L$ score matrix `S`, which requires $4L^2$ bytes in fp32. The full matrix cannot remain on chip. Writing it to GMEM and reading it back for softmax and the second matrix multiplication would introduce intermediate traffic that grows quadratically with the sequence length. FlashAttention instead processes one query block at a time and streams `K` and `V` in blocks, avoiding the full `S` matrix in GMEM.

For one head of a length-$L$ self-attention operation, `Q`, `K`, and `V` each have shape $L\times d$. Let $i$ index query positions and $j$ index key/value positions; denote the corresponding rows by:

$$q_i,k_j,v_j\in\mathbb{R}^d$$

The dot product of $q_i$ and $k_j$ gives the scalar score at position $(i,j)$:

$$s_{ij}=q_i\cdot k_j$$

Fixing query vector $q_i$ and taking its dot product with every key vector $k_j$ produces the scores $s_{ij}$ for that query. These scores form row $i$ of the score matrix $S=QK^\top$. Let $m_i^{\max}$ denote the exact largest score in that row:

$$m_i^{\max}=\max_j s_{ij}$$

Basic stable softmax uses $m_i^{\max}$ as its exponent reference. Subtracting it before exponentiation makes the largest exponent input in the row zero and avoids excessively large values. The same shift applies to both the numerator and denominator, so the normalized softmax result is unchanged. The unnormalized attention weight at each position is:

$$p_{ij}=\exp\left(\frac{s_{ij}-m_i^{\max}}{\sqrt d}\right)$$

Summing the $p_{ij}$ values in the row gives the unnormalized weight sum $\ell_i$. Using the same $p_{ij}$ values to weight the value vectors gives an output vector $o_i$ that has not yet been divided by $\ell_i$:

$$\ell_i=\sum_j p_{ij}$$

$$o_i=\sum_j p_{ij}v_j$$

The final output is:

$$O_i=\frac{o_i}{\ell_i}$$

FlashAttention processes K/V in blocks. Once a block's scores have been consumed, they can be discarded. For each row, the kernel retains an exponent reference $r_i$, the running denominator $\ell_i$, and the running weighted sum $o_i$. Basic online softmax updates $r_i$ to the largest score seen so far, whereas FA4 may temporarily keep an older value. Both $\ell_i$ and $o_i$ are accumulated relative to the current $r_i$. If a later block adopts a larger reference, the old state must first be converted to the new scale before the current block's contribution can be added.

Basic online softmax performs this conversion whenever it encounters a larger row maximum. FA4 first checks the gap between the old and candidate references. When the gap is small enough, it retains the old reference and avoids immediately rescaling the accumulated output. To understand this optimization, we first derive the scale conversion caused by changing the reference.

The implementation uses base-2 exponentials, so define:

$$\alpha=\frac{\log_2(e)}{\sqrt d}$$

The natural exponential can then be written as:

$$\exp\left(\frac{s-m}{\sqrt d}\right)=2^{(s-m)\alpha}$$

The code calls $\alpha$ `scale_log2`. Let $r_{\mathrm{old}}$ be the reference used by the running state and $m_{\mathrm{block}}$ be the row maximum of the current block. With $c$ denoting the candidate, the candidate reference is:

$$r_c=\max(r_{\mathrm{old}},m_{\mathrm{block}})$$

Define their signed gap in the base-2 exponent domain as $\delta$, corresponding to the code variable `delta`:

$$\delta=(r_{\mathrm{old}}-r_c)\alpha\le 0$$

$\delta$ is the old reference minus the candidate reference, measured in base-2 exponent units. Thus $-\delta$ is the amount by which the candidate exceeds the old reference. Because $r_c\ge r_{\mathrm{old}}$, $\delta$ cannot be positive.

The [FA4 paper](https://arxiv.org/abs/2603.05451) typically sets the threshold to $\tau=\log_2(256)=8$. When $-\delta=8$, retaining the old reference lets the largest unnormalized weight in the current block reach $2^8=256$; switching to the candidate reference would instead multiply the old state by $2^\delta=1/256$. The threshold therefore permits at most a 256-fold scale gap before rescaling: `delta >= -8` retains the old reference, whereas `delta < -8` changes the reference. Using this threshold to delay rescaling reduces the data movement and multiplications performed by the correction path; the value 8 balances fewer rescaling operations against bounded exponent growth.

If this iteration adopts the candidate reference $r_c$, every exponential accumulated under the old reference must be multiplied by the same factor:

$$e^{(s-r_c)/\sqrt d}
=e^{(s-r_{\mathrm{old}})/\sqrt d}
\cdot e^{(r_{\mathrm{old}}-r_c)/\sqrt d}$$

Writing this conversion factor as $a_{\mathrm{scale}}$ gives:

$$a_{\mathrm{scale}}
=e^{(r_{\mathrm{old}}-r_c)/\sqrt d}
=2^\delta$$

After switching to the candidate reference $r_c$, the accumulated denominator $\ell_i$ and weighted sum $o_i$ remain on the old scale. The kernel first multiplies both by $a_{\mathrm{scale}}=2^\delta$ to convert them to the new scale, then adds the current block's contributions. In the pseudocode below, $\ell_i$ and $o_i$ become `row_sum` and `O`, while `acc_scale = exp2(delta)` computes the conversion factor.

The three values retained across K/V blocks map to the pseudocode as follows:

- `row_max`: the exponent reference $r_i$ subtracted from every score in the row. Basic online softmax uses the largest score seen so far; FA4 may retain the old reference while the threshold permits it. Despite its name, `row_max` therefore need not equal the exact maximum $m_i^{\max}$ at every iteration.
- `row_sum`: the sum of $p_{ij}$ over all key positions processed so far, namely $\ell_i$.
- `O`: the weighted sum $o_i$ formed from the same $p_{ij}$ values. It is divided by `row_sum` only after all blocks have been processed.

This produces three cases:

- The first K/V block has no previous state. It adopts `candidate_max` and sets `acc_scale = 1`.
- When `delta >= -8`, the kernel keeps the old reference, computes the current block against that same reference, and sets `acc_scale = 1` because the old state needs no conversion.
- When `delta < -8`, the gap exceeds the threshold. The kernel adopts `candidate_max` and sets `acc_scale = exp2(delta)` to convert the old `row_sum` and `O` to the new scale.

The following pseudocode gives the core algorithmic loop for one query block. It temporarily ignores warpgroup roles and pipeline overlap; the real kernel performs the same steps while interleaving them across roles:

```text
scale_log2 = log2(e) / sqrt(d)
rescale_threshold = 8

row_max = -inf
row_sum = 0
O = 0
first_block = true

for each (K_block, V_block):
    S = Q_block @ K_block.T

    if causal:
        S[masked positions] = -inf

    candidate_max = max(row_max, rowmax(S))

    if first_block:
        new_ref = candidate_max
        acc_scale = 1
    else:
        delta = (row_max - candidate_max) * scale_log2  # delta <= 0
        if delta >= -rescale_threshold:                 # gap stays within threshold
            new_ref = row_max
            acc_scale = 1
        else:                                           # gap exceeds threshold
            new_ref = candidate_max
            acc_scale = exp2(delta)

    row_max_safe = 0 if new_ref == -inf else new_ref
    P = exp2((S - row_max_safe) * scale_log2)
    row_sum = row_sum * acc_scale + rowsum(P)
    O = O * acc_scale[:, None] + P @ V_block

    row_max = new_ref
    first_block = false

for each row:
    O[row, :] = O[row, :] / row_sum[row] if row_sum[row] != 0 else 0
store O
```

`new_ref` is the exponent reference selected for this iteration. If the old reference is retained, `acc_scale=1` and the running state is unchanged. If the candidate reference is adopted, the kernel first uses `acc_scale` to convert the old `row_sum` and `O`. It then computes the current block's `P` against `new_ref`. Only after every K/V block has been processed does the kernel compute the final `O / row_sum`. *Rescaling and Writeback* explains how WG2 performs or skips the conversion of `O`.

If a row has not encountered any valid score up to and including the current block, both its old reference and the current block maximum are `-inf`, so `new_ref` is also `-inf`. Evaluating `S - new_ref` directly would then produce `-inf - (-inf)`. In this case, `row_max_safe` uses zero so that the masked scores have zero exponentials and `P`, `row_sum`, and `O` remain zero. If an earlier block already contributed valid scores, a later fully masked block contributes only zeros and does not clear the accumulated `row_sum` or `O`.

Rewriting the natural exponential in base-2 form is only an algebraic transformation; by itself, it does not remove the throughput bottleneck in the exponential path. If every element still uses the hardware `exp2` path, those units can continue to limit softmax throughput.

FA4 therefore divides exponential evaluation between two execution paths. In the [paper](https://arxiv.org/abs/2603.05451), some elements use hardware `exp2`, while others use a cubic polynomial evaluated with FP32 FMA instructions. In the current TIRx implementation, `ex2_emulation_2` provides the latter path. Hardware exponential units and FMA units can then work concurrently, reducing dependence on a single execution path. This changes how the exponential is evaluated, not the online-softmax recurrence above.

When this algorithm is mapped to the kernel, each K/V block produces or updates three kinds of tiles. Their storage locations determine the layouts and barriers that follow:

- `S` is the score tile. The QKᵀ MMA writes it to TMEM.
- `P` is the unnormalized attention-weight tile. Softmax reads `S` from TMEM into registers, computes `P = exp2((S - row_max_safe) * scale_log2)`, and writes `P` back to TMEM.
- `O` is the output accumulator tile. The PV MMA reads `P` from TMEM and `V` from SMEM, then accumulates into `O` in TMEM.

When the exponent reference changes, the old `O` is read from TMEM, rescaled in registers, and written back to TMEM before the next PV MMA accumulates into it.

## Tile Primitive Data Flow

With the roles of `S`, `P`, and `O` established, we can expand one K/V block into its concrete data paths:

```text
Q, K:  GMEM --TMA load--> SMEM --QKᵀ MMA--> S in TMEM
S:     TMEM --tcgen05.ld--> registers --softmax--> P in registers
P:     registers --TMEM store--> P in TMEM
V:     GMEM --TMA load--> V in SMEM
P, V:  P in TMEM + V in SMEM --PV MMA--> O in TMEM

when needed: O in TMEM --tcgen05.ld--> registers --rescale/TMEM store--> O in TMEM
at the end: O in TMEM --tcgen05.ld--> registers --normalize/cast--> O in SMEM --TMA store--> O in GMEM
```

The QKᵀ MMA reads only Q and K and produces `S`. Softmax then loads `S` from TMEM, computes `P` in registers, and stores `P` back to TMEM. The PV MMA combines that `P` with V from SMEM to update `O` in TMEM. Later K/V blocks may first rescale the existing `O`; after the final block, the epilogue normalizes and stores the result.

The table below maps these paths to the concrete TIRx primitives and hardware instructions:

| Stage | Tile movement or compute | TIRx primitive | Hardware path |
|-------|--------------------------|----------------|---------------|
| Load Q/K/V | GMEM tiles -> SMEM tiles | `Tx.copy_async(..., dispatch="tma")` | TMA load |
| QKᵀ MMA | Q in SMEM and K in SMEM -> score tile `S` in TMEM | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` |
| Softmax read | `S` in TMEM -> warpgroup register tile | `Tx.wg.copy_async(reg, tmem)` | `tcgen05.ld` |
| Softmax write | unnormalized weight tile `P` in registers -> fp16 TMEM view | `Tx.copy_async(tmem_as_f16, reg)` | TMEM store, followed by `tcgen05.wait.st()` |
| PV MMA | `P` in TMEM and V in SMEM -> output accumulator `O` in TMEM | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` with a TMEM operand |
| Correction | `O` in TMEM -> registers -> `O` in TMEM | TMEM readback, register multiply, TMEM store | `tcgen05.ld` / TMEM store |
| Epilogue | final `O` in TMEM -> registers -> SMEM -> GMEM | TMEM readback, `Tx.copy`, TMA store | `tcgen05.ld` + TMA store |

Compared with GEMM, FA4 inserts softmax between two MMAs: `S` must be read from TMEM into registers, and `P` must then be written back to TMEM. A change in the exponent reference adds another TMEM -> registers -> TMEM pass to rescale `O`. The layouts and barriers introduced later ensure that these accesses occur in the required order.

## Warp Roles and Scope

With the data path established, the next step is to assign each stage to a set of threads. A CTA contains four warpgroups, each made up of four warps and 128 threads, for 512 threads in total. We abbreviate warpgroup 0 through 3 as WG0 through WG3.

The kernel keeps two Q tiles in flight. Each tile uses a reusable slot that includes a Q buffer in SMEM, the corresponding `S`, `P`, and `O` regions in TMEM, and the barriers that protect those values. The code calls these slots Q stages and numbers them stage 0 and stage 1. WG0 runs softmax for stage 0, WG1 runs softmax for stage 1, WG3 issues TMA and MMA work for both stages, and WG2 handles their rescaling and epilogues.

The four warpgroups divide the work as follows:

| Owner | Role | What it does |
|-------|------|--------------|
| WG3, warp 1 | TMA load | Loads Q, K, and V tiles from GMEM to SMEM |
| WG3, warp 0 | MMA | Issues both QKᵀ MMA and PV MMA |
| WG3, warp 2 | TMA store | Stores final O tiles from SMEM to GMEM |
| WG0 | Softmax for Q stage 0 | Reads S from TMEM, computes P, writes P to TMEM |
| WG1 | Softmax for Q stage 1 | Same work for the second Q pipeline stage |
| WG2 | Correction and epilogue | Rescales O in TMEM, normalizes, stages output |

The code selects each thread's role with two thread coordinates:

```python
wg_id = T.warpgroup_id([4])
warp_id = T.warp_id_in_wg([4])
```

Both `wg_id` and `warp_id` range from 0 through 3. The former selects the thread's warpgroup, and the latter selects a warp within that warpgroup. The kernel branches on these values to enter the corresponding role.

Within WG3, warp 1, warp 0, and warp 2 submit TMA loads, MMAs, and TMA stores, respectively. One elected lane in each warp issues each instruction; the TMA engine or Tensor Core performs the operation. WG0 and WG1 execute softmax across their full warpgroups, while WG2 handles rescaling and the epilogue. The FA4 code calls the rescaling of `O` correction.

### Redistributing Registers Across Roles

Warp specialization partitions more than just work. It also lets the kernel concentrate register capacity in the roles that need it. WG3 mostly issues TMA and MMA instructions and does not retain large intermediate tiles. WG0 and WG1, in contrast, need every thread to hold an entire row of 128 fp32 scores together with softmax temporaries. Reserving that worst-case register budget for all 512 threads in the CTA would exceed the available register capacity.

The kernel therefore uses `setmaxnreg` to adjust the per-thread register limit for each role dynamically:

```python
if wg_id == 3:
    Tx.ptx.setmaxnreg(False, 48)       # WG3 releases excess registers
elif wg_id < 2:
    Tx.ptx.setmaxnreg(True, 200)       # WG0/WG1 acquire registers for softmax

with WarpgroupRole(wg_id, 2, regs=64): # WG2 performs correction / epilogue
    ...
```

In this configuration, the per-thread register limits are 200 for WG0 and WG1, 64 for WG2, and 48 for WG3. Across the four 128-thread warpgroups, these budgets add up to:

```text
128 × (200 + 200 + 64 + 48) = 65,536 32-bit registers
```

This redistribution gives the softmax threads enough registers to retain a full score row without reserving the same large allocation for the instruction-issuing threads in WG3.

### Differences Between the Paper and the Current TIRx Kernel

This chapter follows the default path in `flash_attention4.py`. It uses the overall FA4 pipeline from the paper, but two implementation choices differ.

First, the paper staggers the exponential-heavy regions of WG0 and WG1 so that the two softmax warpgroups do not compete for the exponential units at the same time. The current implementation retains `bar_s0_s1_sequence` and the corresponding synchronization branches, but sets `USE_S0_S1_BARRIER=False` by default. The default path described here therefore does not enable that ordering constraint.

Second, the paper uses otherwise idle TMEM to communicate correction statistics. The current TIRx implementation instead writes per-row `acc_scale` values and the final `row_sum` values to the SMEM buffer `sScale`, then hands them from the softmax warpgroups to WG2 through `softmax_corr.full/empty`. The mailbox described later is this TIRx-specific SMEM path, not the paper's TMEM communication path.

## Conventions for Reading the Code

The excerpts in this chapter come from [`flash_attention4.py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py), so they refer to shapes, stage indices, and phase variables defined elsewhere in the kernel. The table below collects the names that recur later but are not self-explanatory:

| Name | Meaning |
|------|---------|
| `q_stage`, `i_q` | Current Q pipeline stage, 0 or 1; inside the WG0/WG1 softmax branches, `wg_id` is the same stage index |
| `MMA_N` | Base width of the score tile and TMEM regions, currently 128 columns |
| `MMA_K`, `K_SPLIT` | Each PV MMA inner-K step consumes 16 positions; `K_SPLIT = 6 * MMA_K = 96` divides the 128 positions into segments of 96 and 32 |
| `should_accumulate` | Whether the current PV MMA initializes `O` or accumulates into an existing `O` |
| `phase_tmem` | Phase parity expected by barriers associated with the current `P` and `O` iteration |
| `should_rescale` | Per-row flag indicating whether the old `O` must be rescaled before the next PV MMA |
| `rescale_threshold` | Threshold for delaying an exponent-reference update, currently 8.0 |
| `scale_log2` | Softmax scale for base-2 exponentiation, `log2(e)/sqrt(d)` |
| `acc_scale` | Per-row scale passed from softmax to WG2 to adjust the old `row_sum` and `O` |

### Barrier Roles and Completion Conditions

The FA4 pipeline maintains several independent handoff states. The Q and K/V SMEM stages are handed off between TMA and MMA. The S, P, and O TMEM slots are handed off among the Tensor Core, softmax, and correction. Softmax and WG2 also reuse a mailbox, while the epilogue and TMA store reuse `O_smem`. Different roles complete these events at different times, and each event protects a different storage location, so the kernel tracks them separately.

When storage is reused cyclically, the handoff usually runs in both directions. `full` or `ready` means that the producer has written the data and the consumer may read it. `empty` means that the consumer has finished and the producer may overwrite the storage. The barriers below record these data-ready and resource-return events.

The initialization count is not always a thread count. A regular `MBarrier` counts explicit arrivals; its count is 128 only when every thread in a 128-thread warpgroup executes one `arrive`. A `TMABar` waits for one producer arrival and for the registered transfer byte count to drain to zero. A `TCGen05Bar` waits for one Tensor Core completion notification attached by `tcgen05.commit`.

In the current implementation, `q_load.full` and `kv_load.full` use `TMABar`; `q_load.empty`, `kv_load.empty`, `s_ready`, and `o_ready` use `TCGen05Bar`; all remaining barriers use regular `MBarrier`. The table below gives the completion condition for one phase of each barrier slot. The Q pipeline has two slots, the K/V pipeline has three, and the other staged barriers in the table have two slots each.

For a `TCGen05Bar`, the table describes the barrier's logical contract: which data it protects and which role may proceed after completion. An actual `tcgen05.commit` makes the barrier track the relevant asynchronous `tcgen05` operations issued earlier by the same issuing thread; it is not necessarily limited to the single MMA named in the table. Read the QKᵀ/PV MMA labels as the last result or last use relevant to that handoff. The hardware completion dependency may be more conservative.

| Barrier | Threads participating in the notification | Completion condition for one phase | What becomes safe |
|---------|------|------------------------------------|-------------------|
| `q_load.full` | one elected TMA-load thread | the thread reports one arrival; TMA then completes `CTA_GROUP * BLK_M * HEAD_DIM * 2` bytes of Q traffic | the QKᵀ MMA may read the Q SMEM tile |
| `q_load.empty` | one elected MMA thread | the thread submits a completion notification; the Tensor Core updates the barrier after the QKᵀ MMAs that still read this Q stage finish | TMA may overwrite the stage with the next query tile |
| `kv_load.full` | one elected TMA-load thread | the thread reports one arrival; TMA then completes `CTA_GROUP * BLK_N * HEAD_DIM * 2` bytes of K or V traffic | the QKᵀ or PV MMA may read the current K/V SMEM tile |
| `kv_load.empty` | one elected MMA thread | the thread submits a completion notification; the Tensor Core updates the barrier after both MMAs that read this stage finish | TMA may reuse the K/V stage |
| `s_ready` | one elected MMA thread | the Tensor Core reports one notification when the QKᵀ MMA completes | softmax may read the S TMEM tile |
| `p_o_rescale` | 128 softmax threads + 128 WG2 threads | the two groups report 256 arrivals in total | the first PV MMA may read `P[:, 0:96]` and initialize or accumulate into O |
| `p_ready_2` | the 128 threads in the softmax warpgroup | the warpgroup reports 128 arrivals | the second PV MMA may read `P[:, 96:128]` |
| `o_ready` | one elected MMA thread | the Tensor Core reports one notification when the final PV MMA segment completes | the epilogue may read the final O accumulator |
| `softmax_corr.full` | the 128 threads in the softmax warpgroup | the warpgroup reports 128 arrivals | WG2 may read `acc_scale` or the final `row_sum` from the mailbox |
| `softmax_corr.empty` | the 128 threads in WG2 | WG2 reports 128 arrivals | softmax may advance and reuse the mailbox |
| `corr_epi.full` | the 128 threads in WG2 | WG2 reports 128 arrivals | the TMA-store warp may read the completed `O_smem` tile |
| `corr_epi.empty` | the 32 threads in the TMA-store warp | after waiting for the TMA store, the warp reports 32 arrivals | the epilogue may reuse the `O_smem` stage |

Every count in the table applies to one slot in its current phase. Multiple slots keep independent barrier state for different pipeline stages; they do not multiply the expected arrival count. The later sections revisit each barrier at its corresponding wait and arrive sites.

## QKᵀ MMA and PV MMA

For a fixed Q stage, the kernel processes the streamed K/V blocks one at a time. Each block passes through three steps:

```text
Q, K -> QKᵀ MMA -> S
S    -> softmax   -> P
P, V -> PV MMA -> O
```

The QKᵀ MMA first produces the current block's attention scores `S`. Softmax converts `S` into the unnormalized weights `P`, and the PV MMA then computes `P @ V`. The first K/V block initializes `O`; later blocks accumulate into the same `O` tile. Once all blocks have been processed, the epilogue divides `O` by `row_sum` to produce the final output.

The following sections examine these three steps in order. For each tile operation, we identify which threads execute it, where its operands and result are laid out, which hardware path dispatch selects, and which barrier hands the result to the next role.

The code uses `S_region`, `P_region`, and `O_region` to name the parts of one TMEM allocation that hold the three tile types. Both `q_stage` and `i_q` identify the current Q stage and take the value 0 or 1. Using the same stage index with all three regions selects the `S`, `P`, and `O` tiles for the same Q tile. For now, treat them as named TMEM regions; [TMEM Layout and Reuse](#tmem-layout-and-reuse) explains their physical column ranges.

### QKᵀ MMA

For the current Q stage and K block, the QKᵀ MMA computes:

$$S = Q_{\text{block}}K_{\text{block}}^{\top}$$

Both `Q_block` and `K_block` have shape `128 x HEAD_DIM`. Transposing `K_block` makes each Q row take a dot product with all 128 K rows, producing a `128 x 128` score tile: rows correspond to queries, and columns correspond to keys in the current K block. The result is written to `S_region[q_stage]`; `MMA_N=128` is the width of this score tile.

```python
Tx.warp.gemm_async(
    S_region[q_stage],
    Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
    K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
if T.ptx.elect_sync():
    s_ready.arrive(q_stage)
```

> **Tile primitive: QKᵀ MMA**
> - Scope: WG3 warp 0 executes the warp-scoped tile operation; one elected lane commits its completion notification.
> - Layout: Q, K in SMEM → `S` in TMEM (`S_region[q_stage]`).
> - Dispatch: `tcgen05`.
> - Handoff: `s_ready` (→ softmax).

`s_ready` is a `TCGen05Bar` that tracks Tensor Core completion. Here, `s_ready.arrive(q_stage)` emits a `tcgen05.commit` that associates the previously issued QKᵀ MMA with the barrier for this stage. Only one elected lane issues the commit. The hardware reports completion only after the Tensor Core has finished writing `S`, so the softmax warpgroup waits for `s_ready` before reading `S_region[q_stage]`.

### Softmax Between MMAs

Softmax sits between the two MMAs and turns the score tile `S` into the unnormalized attention-weight tile `P`. The same four-part analysis applies:

> **Tile primitive: Softmax**
> - Scope: WG0 (Q stage 0) / WG1 (Q stage 1), full warpgroup.
> - Layout: `S` in TMEM → registers → `P` in fp16 TMEM (`P_region[wg_id]`).
> - Dispatch: `tcgen05.ld` reads `S`, row-wise softmax runs in registers, and `tcgen05.st` writes `P` back.
> - Handoff: waits on `s_ready`; reports the first 96 columns through `p_o_rescale`, then reports the final 32 through `p_ready_2`.

Each score tile has 128 rows, and each softmax warpgroup has 128 threads, so the kernel assigns logical row `r` to thread `r`. The `wg_local_layout` encodes this mapping: each thread ultimately processes one row of 128 scores.

Each thread keeps the complete row in a 128-value fp32 register buffer named `s_chunk_buf`. The 200-register limit assigned to WG0 and WG1 above primarily makes room for this buffer and the remaining softmax temporaries. After WG0 or WG1 waits on `s_ready`, it fills that buffer with four 32-column `tcgen05.ld` operations rather than one full-row load:

```python
for chunk_idx in Tx.unroll(BLK_N // SOFTMAX_LD_CHUNK):
    Tx.copy_async(
        s_chunk[
            :, chunk_idx * SOFTMAX_LD_CHUNK : (chunk_idx + 1) * SOFTMAX_LD_CHUNK
        ],
        S_region[
            wg_id,
            chunk_idx * SOFTMAX_LD_CHUNK : (chunk_idx + 1) * SOFTMAX_LD_CHUNK,
        ],
    )
```

Here `SOFTMAX_LD_CHUNK=32`. The TMEM load is chunked, not the softmax computation. The current implementation loads the row in four 32-value fragments, keeping the register tuple for each tile operation small. After all four loads, all 128 scores are live in each thread's registers. This is the load granularity chosen by the current kernel; softmax itself still processes the complete row. Each thread then:

1. finds the maximum of the 128 current scores and combines it with the saved `row_max` to choose the exponent reference and `acc_scale`,
2. computes the row's $p_{ij}$ values and converts the fp32 results to fp16 to form `P`,
3. sums those $p_{ij}$ values to update `row_sum`.

The following excerpt omits profiling and the optional WG0/WG1 ordering barrier while retaining the main computation. It first selects the new reference and uses the threshold to decide whether the old `O` needs rescaling:

```python
row_max_old = row_max[0]
with Tx.thread():
    if is_first:
        Tx.max(tile_max, s_chunk_buf)
    else:
        tile_max[0] = row_max_old
        Tx.max(tile_max, s_chunk_buf, accum=True)

row_max_new = tile_max[0]
row_max_safe = Tx.if_then_else(tile_max[0] == -float("inf"), 0.0, tile_max[0])
if is_first:
    acc_scale = Tx.float32(1.0)
else:
    acc_scale_ = (row_max_old - row_max_safe) * scale_log2
    if acc_scale_ >= -rescale_threshold:
        row_max_new = row_max_old
        row_max_safe = row_max_old
        acc_scale = Tx.float32(1.0)
    else:
        acc_scale = Tx.ptx.exp2(acc_scale_)
row_max[0] = row_max_new
```

It then converts the scores into arguments for base-2 exponentiation, computes the fp32 weights, and casts them to the fp16 `P` consumed by the PV MMA. The implementation selects between hardware `exp2` and `ex2_emulation_2`:

```python
Tx.fma(s_chunk, s_chunk, scale_log2, -row_max_safe * scale_log2)
for frag_idx in Tx.unroll(4):
    with Tx.thread():
        s_chunk_local = s_chunk_buf.local(BLK_N)
        for i in Tx.unroll(BLK_N // 4 // 2):
            idx = Tx.meta_var(frag_idx * BLK_N // 4 + 2 * i)
            if i * 2 % 16 < 16 - 4 or frag_idx >= 4 - 1 or apply_mask:
                s_chunk_local[idx] = Tx.ptx.exp2(s_chunk_local[idx])
                s_chunk_local[idx + 1] = Tx.ptx.exp2(s_chunk_local[idx + 1])
            else:
                ex2_emulation_2(
                    s_chunk_local,
                    idx,
                    s_chunk_local[idx],
                    s_chunk_local[idx + 1],
                )
    Tx.cast(
        p_chunk[:, frag_idx * BLK_N // 4 : (frag_idx + 1) * BLK_N // 4],
        s_chunk[:, frag_idx * BLK_N // 4 : (frag_idx + 1) * BLK_N // 4],
    )
```

Softmax then writes `P` back to TMEM as four 32-column chunks. The code stores the first three chunks, waits for those TMEM stores to finish, and reports that the first 96 columns are ready:

```python
for i in Tx.unroll(3):
    Tx.copy_async(
        P_region[wg_id, i * BLK_N // 4 : (i + 1) * BLK_N // 4],
        p_chunk[:, i * BLK_N // 4 : (i + 1) * BLK_N // 4],
    )
T.ptx.tcgen05.wait.st()
p_o_rescale.arrive(wg_id)

Tx.copy_async(P_region[wg_id, 3 * BLK_N // 4 : BLK_N],
              p_chunk[:, 3 * BLK_N // 4 : BLK_N])
T.ptx.tcgen05.wait.st()
p_ready_2.arrive(wg_id)
```

The fp32 `P` values remain in `s_chunk_buf`. After WG2 consumes `acc_scale` and returns the mailbox slot, the softmax warpgroup uses those values to update the denominator:

```python
softmax_corr.empty.wait(wg_id, phase_q)
with Tx.thread():
    if is_first:
        Tx.sum(row_sum, s_chunk_buf)
    else:
        row_sum[0] = row_sum[0] * acc_scale
        Tx.sum(row_sum, s_chunk_buf, accum=True)
```

The first PV MMA reads `P[:, 0:96]` and updates `O`, so it must wait for two independent conditions: softmax has stored that portion of `P`, and WG2 has made `O` ready for initialization or accumulation. `p_o_rescale` joins those two completion signals. The final 32 columns use a separate `p_ready_2` handoff, so the first MMA does not need to wait for the final TMEM store.

Why write `P` back to TMEM when it was just computed in registers? In this kernel, the PV MMA requires its `P` operand in an MMA-readable TMEM layout; it cannot consume values scattered across the softmax threads' private registers. `P_region` is an fp16 view of the same physical TMEM allocation. Writing `P` there turns the per-thread softmax results into the matrix operand expected by the next MMA.

### PV MMA

Once the current block's `P` and V are ready, the PV MMA updates `O` as follows:

```text
first K/V block: O = P_block @ V_block
later K/V blocks: O = O + P_block @ V_block
```

`P` has shape `128 x 128`, and the V block has shape `128 x d`, so `P @ V` produces a `128 x d` output tile. The first K/V block has no previous result; with `should_accumulate=false`, its product initializes `O`. Later blocks use `should_accumulate=true`. Before those MMAs are issued, WG2 must either rescale the old `O` or confirm that this iteration needs no rescaling.

The operands come from different memory spaces: `P` is in TMEM, V is in SMEM, and the fp32 accumulator `O` is also in TMEM. The kernel further divides the 128 reduction positions into segments of 96 and 32. The two MMA segments are:

```python
# First segment: the first 96 columns of P and matching rows of V.
Tx.warp.gemm_async(
    O_region[i_q],
    P_region[i_q, 0:K_SPLIT],
    V_smem[kv_stage, 0:K_SPLIT, 0:HEAD_DIM],
    transB=True,
    accum=should_accumulate,
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)

p_ready_2.wait(i_q, phase_tmem)
Tx.warp.gemm_async(
    O_region[i_q],
    P_region[i_q, K_SPLIT:BLK_N],
    V_smem[kv_stage, K_SPLIT:BLK_N, 0:HEAD_DIM],
    transB=True,
    accum=True,
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
```

> **Tile primitive: PV MMA**
> - Scope: WG3 warp 0 executes the warp-scoped tile operation.
> - Layout: `P` in TMEM + V in SMEM → `O` in TMEM (`O_region[i_q]`).
> - Dispatch: `tcgen05` with a TMEM operand.
> - Handoff: the first segment waits on `kv_load.full` and `p_o_rescale`; the second also waits on `p_ready_2`. After the final K/V block, `o_ready` hands the result to the epilogue.

`kv_load.full` confirms that V is in SMEM. `p_o_rescale` confirms both that the first 96 columns of `P` are in TMEM and that `O` is ready for initialization or further accumulation. After issuing the first MMA segment, the kernel waits on `p_ready_2` for the final 32 columns, then issues the second segment with `accum=true`. The second segment always accumulates: even for the first K/V block, `O` already contains the partial sum produced by the first segment.

Here, inner K is the reduction dimension of `P(128×128) @ V(128×d)`: the 128 positions in the current K/V block. Each `MMA_K=16` step consumes 16 positions. The kernel groups the first six steps into a 96-position MMA segment and handles the remaining 32 positions in a second segment:

1. Softmax writes `P` in four 32-column chunks.
2. As soon as the first three chunks are ready, the PV MMA starts on the first 96 columns of `P` and the matching rows of `V`.
3. The final 32 columns wait for `p_ready_2`.
4. A second MMA consumes that final chunk and finishes the tile.

The split reduces the time the Tensor Core spends waiting for `P` writeback. If all 128 reduction positions were handed off as one unit, the PV MMA could not begin until all four `P` chunks were in TMEM. Instead, it starts on the first 96 columns while the softmax warpgroup performs the final 32-column TMEM store and completion handoff.

## TMEM Layout and Reuse

FA4 allocates 128 rows by 512 physical TMEM columns for each CTA, with one 32-bit cell at every row-column coordinate. Each of the two Q stages needs a 128-column fp32 score tile `S` and a 128-column fp32 output accumulator `O`. Those tiles alone fill the allocation:

```text
2 stages × (128 columns for S + 128 columns for O) = 512 columns
```

The source first creates two buffers over this allocation. `move_base_to(0)` rewinds the allocation cursor, so `tmem_as_f16` starts at the same physical TMEM column as `tmem`:

```python
tmem_pool = T.TMEMPool(
    pool, total_cols=N_COLS_TMEM, cta_group=CTA_GROUP, tmem_addr=tmem_addr
)
tmem = tmem_pool.alloc((128, N_COLS_TMEM), "float32")
tmem_pool.move_base_to(0)
tmem_as_f16 = tmem_pool.alloc((128, N_COLS_TMEM * 2), "float16")
tmem_pool.commit()
```

The two buffers contain the same number of bits per row:

```text
tmem:         512 × 32 bits = 16384 bits
tmem_as_f16: 1024 × 16 bits = 16384 bits
```

`tmem_as_f16` is therefore another indexing scheme for the same TMEM row, not a second allocation. Hardware still divides each row into 512 cells, each 32 bits wide; we call the cell index the physical column. Through the fp16 buffer, each cell appears as two independently indexed 16-bit element slots:

```text
physical column p (32 bits)
┌────────────────┬────────────────┐
│ fp16 slot 2p   │ fp16 slot 2p+1 │
└────────────────┴────────────────┘
```

Thus `tmem[:, p]` addresses the entire cell as one fp32 value, while `tmem_as_f16[:, 2p]` and `tmem_as_f16[:, 2p+1]` address its two fp16 values.

The source then defines two pipeline stages for `S`, `P`, and `O`:

```python
S_region = T.TMEMStages(
    tmem, col_start=0, width=MMA_N,
    stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N,
)
O_region = T.TMEMStages(
    tmem, col_start=MMA_N * SMEM_PIPE_DEPTH_Q, width=MMA_N,
    stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N,
)
P_region = T.TMEMStages(
    tmem_as_f16, col_start=MMA_N, width=BLK_N,
    stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N * 2,
)
```

Here `MMA_N=BLK_N=128`, and the Q pipeline has two stages. `S_region` and `O_region` index the fp32 buffer, so their indices are also physical column numbers. `P_region` indexes the fp16 buffer, so dividing an index by two gives its physical column.

For `P0`, let `n` denote the logical column within the tile:

```text
P_region[0, n]
    -> tmem_as_f16[:, 128 + n]       # col_start = 128
    -> physical column 64 + n // 2
```

`P0[:, 0]` and `P0[:, 1]` therefore occupy the two 16-bit halves of physical column 64. `P0[:, 2]` and `P0[:, 3]` occupy physical column 65. The 128 fp16 values fill 64 physical columns, `[64, 128)`.

For stage 1, the fp16 start is `128 + 1 × 256 = 384`:

```text
P_region[1, n]
    -> tmem_as_f16[:, 384 + n]
    -> physical column 192 + n // 2
```

`P1` therefore occupies physical columns `[192, 256)`. The figure and table summarize the final placement of every region:

![S, P, and O slots share one TMEM allocation](../img/tmem_layout_v3.png)

| Region | Data stored in each row | Physical columns occupied |
|---|---:|---:|
| `S0` | 128 fp32 scores | `[0, 128)` |
| `P0` | 128 fp16 weights | `[64, 128)`, reusing the second half of `S0` |
| `S1` | 128 fp32 scores | `[128, 256)` |
| `P1` | 128 fp16 weights | `[192, 256)`, reusing the second half of `S1` |
| `O0` | 128 fp32 accumulator values | `[256, 384)` |
| `O1` | 128 fp32 accumulator values | `[384, 512)` |

There is no separate region reserved for `P`. The overlap is temporal reuse; `S` and `P` do not coexist in those bits. For stage 0, the QKᵀ MMA first writes the complete `S0` tile to physical columns `[0, 128)`. After softmax has loaded all of `S0` into registers, it packs the 128 fp16 `P0` values two per column and writes them to `[64, 128)`. That store overwrites the final 64 fp32 scores, which are no longer needed.

This reuse requires three operations to occur in order. Softmax must read the complete `S` tile into registers before `P` overwrites the second half of `S`. The PV MMA must wait until the corresponding `P` chunks have been stored. The next QKᵀ MMA must not overwrite the region again until the current `P` has been consumed.

Ordinary source-level program order alone does not establish these conditions. The `tcgen05.commit` for the QKᵀ MMA reports completion through `s_ready`, which releases softmax; softmax uses the scores only after its TMEM-to-register loads have completed. When writing `P`, `tcgen05.wait::st` first waits for the asynchronous TMEM stores, after which the softmax threads arrive on `p_o_rescale` or `p_ready_2`; the PV MMA waits on the matching barrier before reading. Finally, WG3 warp 0 issues the PV MMA and the following QKᵀ MMA as a fixed `tcgen05` sequence from the same issuing thread, and lowering must preserve the required `tcgen05` dependencies between them. Together, these completion and ordering mechanisms prevent the aliased TMEM region from being read or overwritten too early.

Once these regions are defined, the compute code can index `S_region[...]`, `P_region[...]`, and `O_region[...]` by stage without computing raw TMEM column numbers.

## Key Barrier Protocols

The summary table above identifies every barrier's notifier, completion condition, and the operation it releases. This section expands only the two protocols that are easiest to confuse: the conditions that the QKᵀ and PV MMAs wait for, and the full/empty handshake through which softmax and WG2 reuse an SMEM exchange slot for per-row state.

### What Each MMA Waits For

The next figure shows the readiness gates for the QKᵀ MMA and for each of the two PV MMA segments: which operands and accumulator state must be ready before each segment can be issued.

![The QKᵀ MMA waits for Q and K; the two PV MMA segments wait for the corresponding P and V ranges and for O to be ready](../img/flash_attention_main_handoff.png)

The upper path is the QKᵀ MMA. `q_load.full` proves that the current Q stage is in SMEM, while `kv_load.full` proves that the current K stage is in SMEM. The QKᵀ MMA can produce `S` only after both conditions hold.

The lower half separates the PV MMA into the two segments issued by the code. The first segment covers inner-K positions `0:96`. `kv_load.full` proves that the complete `V` tile is in SMEM, while `p_o_rescale` combines two conditions: `P[:, 0:96]` is in TMEM, and the `O` slot may be initialized or accumulated into. The first K/V block initializes `O` directly; later blocks must first complete the required rescale or confirm that the current round does not need one.

After issuing the first segment, the same MMA warp waits on `p_ready_2`, then issues the second segment with `P[:, 96:128]` and `V[96:128, :]`, using `accum=True` to update the same `O` tile. It does not wait on `kv_load.full` again because that barrier already proved that the complete `V` tile was ready. `p_ready_2` gates only the second segment, so it does not delay the first.

The expected arrival count of `p_o_rescale` is 256. The softmax warpgroup contributes 128 arrivals after storing the first 96 columns of `P`, and WG2 contributes another 128 after making `O` ready. For the first K/V block, no old `O` exists, so WG2 contributes its half in advance. On later blocks, it arrives after completing the rescale or determining that no rescale is needed. All 256 arrivals must occur before the first PV MMA segment can begin. The expected count of `p_ready_2` is 128; the softmax warpgroup contributes those arrivals after storing the final 32 columns, releasing only the second segment.

### Passing Per-Row State from Softmax to WG2

The softmax warpgroup sends WG2 two kinds of per-row values. During the K/V loop, `acc_scale[row]` tells WG2 how much to rescale that row of the old `O` tile in TMEM. After all K/V blocks have been processed, the final `row_sum[row]` lets WG2 compute `O[row, :] / row_sum[row]`. The kernel reserves one reusable exchange slot per Q stage in the `sScale` SMEM buffer; below, we call this slot a mailbox. After softmax writes the slot, `softmax_corr.full` notifies WG2. After WG2 reads it, `softmax_corr.empty` returns the slot. The figure below shows this full/empty protocol for one mailbox slot:

![Softmax and WG2 reuse one SMEM mailbox through full and empty barriers](../img/flash_attention_softmax_correction.png)

Read `softmax_corr.full` and `softmax_corr.empty` as a producer-consumer pair:

1. Softmax waits for `softmax_corr.empty` before reusing the scale/sum slot.
2. Softmax writes `acc_scale` or final `row_sum` into that slot.
3. Softmax arrives on `softmax_corr.full`.
4. WG2 waits on `softmax_corr.full`, then reads the slot.
5. WG2 arrives on `softmax_corr.empty`.
6. The softmax warpgroup may reuse the slot in the next phase.

The first K/V block has no old `O`, so it does not need an `acc_scale`. Softmax and WG2 still complete one full/empty handoff so that both barriers advance to the same next phase; otherwise, the following iteration could wait on different phases. Later iterations use the same mailbox to carry `acc_scale`, and the final handoff carries `row_sum`.

The kernel interleaves correction for two Q stages. After processing stage `i_q`, WG2 calls `softmax_corr.empty.arrive(1 - i_q)` to release the other softmax stage, keeping WG0 and WG1 in their fixed alternating order. During the epilogue, after reading the final `row_sum`, WG2 returns the slot for the same `i_q`. The figure therefore describes one mailbox slot; the stage index in code also reflects this two-stage interleaving.

`softmax_corr.empty` and `p_o_rescale` serve different purposes. The former advances the softmax mailbox protocol. The latter proves to the PV MMA that both `P` and `O` satisfy the first MMA segment's input conditions.

Most barriers that FA4 adds beyond GEMM surround softmax. Register computation, the TMEM rewrite of `P`, and the optional rescale of `O` now sit between the QKᵀ and PV MMAs, so each boundary needs an explicit readiness or reuse signal.

## Pipeline Timeline

The handoff diagram tells us what must be ready before each role can consume a tile, but it does not show which roles execute at the same time. A barrier may complete before the consumer reaches it, or it may force the consumer to wait, so dependencies and execution timing need separate views.

There is no single pipeline depth here, because different tile streams move at different rates. The kernel therefore maintains a separate set of circular stages for each:

- Q pipeline depth 2: one CTA advances two query tiles, with WG0 and WG1 running softmax for stages 0 and 1.
- KV pipeline depth 3: K and V blocks move in reverse order through three reusable SMEM stages, feeding both query tiles.
- TMEM pipeline depth 2: the two query tiles use separate S/P/O slots, which enter their next phase after the corresponding handoffs complete.

The figure below uses a timeline to show which roles can be active at roughly the same time once these pipelines are in flight. It separates initialization, the steady-state K/V loop, and the final drain:

![Overlapping TMA loads, QKᵀ and PV MMAs, softmax, correction, and TMA stores in the FA4 pipeline](../img/flash_attention_pipeline_v2.png)

Use this figure to see which roles can overlap. Use the earlier barrier-flow figure to check the exact waits and arrivals between producers and consumers. The two figures therefore separate execution overlap from correctness dependencies.

Each row matches one of the code's role branches:

- WG3 warp 1 issues TMA loads.
- WG3 warp 0 issues both QKᵀ MMA and PV MMA.
- WG0 and WG1 run softmax for the two Q stages.
- WG2 releases both `O` slots before the first iteration, rescales `O` as needed on later iterations, and finally normalizes the output.
- WG3 warp 2 issues the TMA store.

Reading the figure from left to right shows one representative pass through the pipeline. Here $n$ is the number of K/V blocks needed by these two query tiles. The kernel starts at the last valid block and visits `n-1`, `n-2`, and so on. The load warp begins with `Q0`, `K[n-1]`, `Q1`, and `V[n-1]`, then continues with lower-index K/V blocks. The MMA warp produces `S0` and `S1`, and WG0/WG1 turn them into `P0` and `P1`.

The MMA warp does *not* run all the QKᵀ MMAs followed by all the PV MMAs. Once both Q stages are primed, it interleaves the two kinds: a PV MMA for the current `V` block, then a QKᵀ MMA for the next `K` block, and so on:

```text
score Q0*K[n-1]
score Q1*K[n-1]
value P0*V[n-1]
score Q0*K[n-2]
value P1*V[n-1]
score Q1*K[n-2]
value P0*V[n-2]
...
```

This interleaving is why the QKᵀ MMA, softmax, correction, and PV MMA rows overlap in the figure instead of running serially, one stage after another.

The `pre-release O0/O1` event at the left of the timeline occurs before the main loop. TMEM contains no old `O` yet, so WG2 immediately contributes arrivals to both `p_o_rescale` slots and lets the first PV MMAs initialize `O0` and `O1` with `accum=false`. In the steady-state loop, WG2 rescales an old `O` as needed after the corresponding softmax produces `acc_scale`, then releases the next PV MMA. The ellipsis carries this interleaving through `V[0]`. Only after the final two PV MMAs finish does WG2 normalize `O0` and `O1`; WG3 warp 2 then issues the two TMA stores in order.

Q tiles, K/V blocks, and TMEM slots advance at different rates. The kernel uses `PipelineState` to track the stage index and phase of the K/V circular pipeline, and separate local phase variables for the Q and TMEM slots. Each path can therefore wait on its own barrier and reuse storage independently after the corresponding consumer is finished.

## Rescaling and Writeback

The *Algorithm Structure* section derived the correction rule. When `delta >= -8`, softmax retains the old reference, `acc_scale = 1`, and the `O` tile in TMEM needs no update. When `delta < -8`, softmax adopts the new reference, and the old `O` must be multiplied by `acc_scale = exp2(delta)` before accumulation continues.

`row_sum` remains in the softmax warpgroup's registers and can be multiplied by `acc_scale` as part of its normal update. `O`, however, resides in TMEM and requires a separate data path through WG2. Softmax writes the per-row `acc_scale` values to the SMEM mailbox; WG2 waits on `softmax_corr.full`, reads the current `O` from TMEM, multiplies it by the scale, and writes it back:

```python
RESCALE_TILE = T.meta_var(16)
o_row = T.wg_reg_tile(RESCALE_TILE)
Tx.copy_async(o_row, O_region[i_q, d_start : d_start + RESCALE_TILE])
Tx.mul(o_row, o_row, acc_scale)
Tx.copy_async(O_region[i_q, d_start : d_start + RESCALE_TILE], o_row)
T.ptx.tcgen05.wait.st()
```

Each warp in WG2 handles 32 rows and decides independently whether its rows need correction. Every lane forms a per-row `should_rescale` flag from `acc_scale`, and `any_sync` combines those 32 flags within the current warp. If all 32 scales are 1, that warp skips the TMEM → registers → TMEM data path. If any row needs correction, the warp processes its 32-row stripe; rows whose scale is 1 are simply multiplied by 1. The other WG2 warps make the same decision for their own rows.

The control flow reduces to:

```python
should_rescale = T.Select(acc_scale < T.float32(1.0), 1, 0)
any_needs_rescale = T.ptx.any_sync(0xFFFFFFFF, should_rescale)

if any_needs_rescale != 0:
    # This warp: TMEM -> registers -> multiply -> TMEM
    ...

# The correction loop returns the other Q stage in its alternating protocol.
p_o_rescale.arrive(i_q)
softmax_corr.empty.arrive(1 - i_q)
```

Skipping the data path does not skip the synchronization protocol. Every warp still contributes the arrivals required by `p_o_rescale` and `softmax_corr.empty`, allowing the PV MMA to proceed and returning the softmax mailbox for reuse.

Conditional rescaling therefore acts as a two-level filter. The threshold test first makes `acc_scale = 1` for many rows; `any_sync` then checks whether all 32 rows owned by the current warp can skip the correction data path. Even when it skips the TMEM load, multiply, and store, the warp still performs the barrier arrivals required to advance the pipeline.

When correction is required, each warp applies the following TMEM -> registers -> TMEM tile operation to its own stripe of `O` rows:

> **Tile primitive: Correction (rescale)**
> - Scope: WG2; each warp independently checks and processes its own 32 rows.
> - Layout: `O` in TMEM → registers → `O` in TMEM (`O_region[i_q]`).
> - Dispatch: `tcgen05.ld` to read, TMEM store to write; register multiply between them.
> - Handoff: waits `softmax_corr.full`; arrives `p_o_rescale` (→ PV MMA) and `softmax_corr.empty` (→ softmax).

Tracing the synchronization from end to end:

1. Softmax writes the scale value to SMEM.
2. WG2 waits on `softmax_corr.full`.
3. Each WG2 warp checks its 32 rows and updates `O` in TMEM only when needed.
4. WG2 completes the required arrivals on `p_o_rescale` and `softmax_corr.empty`, whether or not the data path ran.
5. WG3's PV MMA can now consume `P` and accumulate into the rescaled `O` tile.

Once the K/V loop ends, WG2 switches from correction to epilogue. It waits for the final `row_sum`, `o_ready`, and a reusable `O_smem` stage. It then reads the final `O` from TMEM, multiplies by `1 / row_sum`, casts to fp16, and writes `O_smem`. `corr_epi.full` hands that tile to WG3, whose TMA store warp writes it to GMEM.

Extending this kernel to a training-time forward pass would normally require writing the log-sum-exp (LSE) for reuse by the backward pass; otherwise, backward must recompute it. The current implementation writes only the output `O`.

Let $r_i$ denote the exponent reference ultimately stored in `row_max`. The source selects this reference from the unscaled $QK^T$ scores and applies `scale_log2` only when evaluating the exponential. Delayed rescaling means that $r_i$ need not equal the exact row maximum, but every accumulated weight is represented relative to the same $r_i$:

$$
\mathrm{row\_sum}_i
= \sum_{j\in\mathrm{valid}}
  \exp\left(\frac{s_{ij}-r_i}{\sqrt d}\right)
= \sum_{j\in\mathrm{valid}}
  2^{(s_{ij}-r_i)\,\mathrm{scale\_log2}}.
$$

Adding the reference back gives the natural-log LSE of the scaled logits:

$$\mathrm{LSE}_i = \log(\mathrm{row\_sum}_i) + r_i / \sqrt{d}$$

The derivation requires only that `row_sum` and $r_i$ use the same reference; $r_i$ does not have to be the exact maximum. The formula applies to valid rows with `row_sum > 0`; a row with no valid key has LSE $-\infty$. This implementation does not write LSE.

## Causal Masking

Causal attention allows each query to use only keys at or before its own position. When Q and K have the same sequence length, the valid region lies on and below the main diagonal of the score matrix. For unequal lengths, the current implementation uses a bottom-right-aligned causal mask: query position `i` may access at most key position `i + SEQ_LEN_KV - SEQ_LEN_Q`, clipped to `SEQ_LEN_KV - 1`. The kernel handles this at both levels: it skips blocks that are entirely invalid and masks invalid columns in blocks that cross the boundary.

At the block level, `get_n_block_max(...)` returns the exclusive upper bound of the K/V blocks needed by the current Q task. The loop visits blocks `0` through `n_block_max - 1` and never loads higher-numbered blocks that contain no valid score.

Blocks that straddle the causal boundary contain both valid and invalid columns. They still run the QKᵀ MMA, but softmax masks the invalid columns before exponentiation. For each row, it derives a column limit from the query position and block offset, keeps columns at or below that limit, and sets later columns to `-inf` in registers. Those columns do not affect the row maximum, and their $p_{ij}$ values become zero.

Rather than compare coordinates separately for every element, `mask_r2p(...)` converts the column limit into a set of bit masks. It handles at most 24 elements per mask and uses bit tests to form predicates, which lower to an efficient register-to-predicate path. Blocks that lie fully inside the causal boundary keep every column and need no mask at all.

Seen from the tile-primitive view, causal mode does not change the data path. It only trims the K/V trip count and inserts a masking step into the register-resident softmax, between the QKᵀ MMA and the `P` writeback.

## GQA Support

Grouped Query Attention lets several query heads share a single K/V head, reducing K/V storage and memory traffic. With `num_qo_heads` query heads and `num_kv_heads` K/V heads, each K/V head serves `GQA_RATIO = num_qo_heads // num_kv_heads` query heads. The kernel processes that group against one scheduled `kv_head_idx` at a time:

```python
GQA_RATIO = num_qo_heads // num_kv_heads
SEQ_Q_PER_TILE = BLK_M // GQA_RATIO
```

The key is to reinterpret the 128 Q-tile rows. For `GQA_RATIO=4`, they represent 32 sequence positions times four query heads. For a row within the tile:

```text
seq_offset    = row // GQA_RATIO
q_head_offset = row % GQA_RATIO
q_head        = kv_head_idx * GQA_RATIO + q_head_offset
```

The Q load expresses this packing with a 3D view. The source is the natural `Q[batch, seq, qo_head, dim]` layout, while the destination is the same SMEM tile that the QKᵀ MMA will later read as a flat `128 x HEAD_DIM` operand. The view tells the TMA copy how to interpret the source and destination coordinates; it does not require a separate rearrangement pass:

```python
Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH_Q, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
Tx.copy_async(
    Q_smem_3d[i_q, :, :, :],
    Q[batch_idx,
      m_start : m_start + SEQ_Q_PER_TILE,
      kv_head_idx * GQA_RATIO : (kv_head_idx + 1) * GQA_RATIO,
      :],
    **tma_copy_q,
)
```

K and V are not replicated for each query head. Instead, all `GQA_RATIO` query heads packed into the Q rows reuse the single K/V tile for `kv_head_idx`. The output side mirrors the input, with a matching 3D view storing the packed rows back to `O[batch, seq, qo_head, dim]` after the epilogue.

GQA does not change the QKᵀ MMA, softmax, or PV MMA tile shapes: the compute path still sees a plain `128 x HEAD_DIM` Q operand. The Q load and O store use 3D views to translate between packed rows and `(sequence, query head)` coordinates. The scheduler's query-tile stride and the causal mask's row position also use `SEQ_Q_PER_TILE` and `GQA_RATIO` to interpret those packed rows.

## Tile Scheduling

The scheduler maps each CTA to a `(batch, kv_head, m_block)` attention task. One `m_block` contains the two Q stages introduced earlier, so each task advances two query tiles together. Causal masking makes task costs uneven, so causal and non-causal modes use different scheduling strategies:

- Non-causal mode uses `FlashAttentionLinearScheduler`. Every task visits the same number of K/V blocks, so the kernel launches a fixed set of persistent CTAs. After completing one task, each CTA advances its linear task index by `num_ctas` and processes the next assignment.
- Causal mode uses `FlashAttentionLPTScheduler`. A Q block near the beginning may visit only one K/V block, while a later Q block may visit all of them. The scheduler first reverses the `m_block` order so that later, heavier blocks are scheduled first, reducing load imbalance near the end of the launch. It also groups the flattened `batch × kv_head` index by `L2_SWIZZLE`: before advancing to the next `m_block`, it visits the batch/KV-head tasks in the current group. This keeps a bounded group of K/V working sets active in L2 as the scheduler advances through `m_block`. The current implementation launches one CTA per causal task.

The scheduling constants in the current source are tuned for the B200 configuration used in this book; they are not universal Blackwell parameters. `max_ctas=148` caps the non-causal persistent worker count at 148, while `SM_NUMBER=148` also participates in profiler-buffer indexing. `L2_SIZE=50 MiB` is the usable cache budget assumed when computing `L2_SWIZZLE`, not the GPU's full reported L2 capacity. A Blackwell GPU with a different SM count or cache configuration should retune these values or derive them from the target configuration.

Both schedulers expose the same loop interface:

```python
while scheduler.valid():
    m_block_idx = scheduler.m_block_idx
    batch_idx = scheduler.batch_idx
    kv_head_idx = scheduler.head_idx
    # process one Q block against its K/V block range
    scheduler.next_tile()
```

The difference lies in `next_tile()`: non-causal mode advances a persistent CTA to another task, while a causal CTA owns only its current task and therefore exits the loop. Both modes run the same local primitives inside the loop: TMA load, QKᵀ MMA, softmax, PV MMA, correction, and TMA store.

## Compile and Verify

The preceding sections used excerpts from the complete kernel. To run FA4, import [`flash_attention4.py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py) from `tirx-kernels`, compile it, and compare its output with a PyTorch reference. Unlike the GEMM examples, this kernel is constructed with `get_flash_attention4_kernel` and accepts an additional `profiler_buffer` argument for its built-in profiler.

The current `flash_attention4.py` is specialized for fixed tile shapes rather than serving as a general attention interface. Its inputs must satisfy these constraints:

- `NUM_QO_HEADS` must be divisible by `NUM_KV_HEADS`, producing an integral `GQA_RATIO`.
- `GQA_RATIO` must divide `BLK_M=128`, so the 128 packed Q rows map evenly back to sequence positions.
- `HEAD_DIM` must currently be 128; the TMEM regions, PV MMA, and epilogue are organized around that width.
- On the non-causal path, `SEQ_LEN_KV` must be divisible by `BLK_N=128`. The code rounds the K/V block count up but does not apply a tail mask to a final partial non-causal block. The built-in causal and non-causal test configurations both use multiples of 128.

The example checks these requirements before compiling:

```python
import torch
import torch.nn.functional as F
import tvm
from tirx_kernels.attention.flash_attention4 import (
    get_flash_attention4_kernel, PROFILER_BUFFER_SIZE)

B, S, Hq, Hkv, D = 1, 1024, 32, 8, 128   # GQA: 32 query heads share 8 KV heads
assert Hq % Hkv == 0
assert 128 % (Hq // Hkv) == 0
assert D == 128
assert S % 128 == 0

Q = torch.randn(B, S, Hq, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
O = torch.empty(B, S, Hq, D, dtype=torch.float16, device="cuda")
prof = torch.zeros(PROFILER_BUFFER_SIZE, dtype=torch.uint64, device="cuda")

kernel = get_flash_attention4_kernel(B, S, S, Hq, Hkv, D, is_causal=False)
target = tvm.target.Target("cuda")
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
ex.mod(Q, K, V, O, prof)
torch.cuda.synchronize()

# torch reference; enable_gqa lets the 32 query heads share the 8 KV heads
qt, kt, vt = (x.transpose(1, 2).float() for x in (Q, K, V))
ref = F.scaled_dot_product_attention(qt, kt, vt, enable_gqa=True).transpose(1, 2).half()
torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)
print(f"FA4: B={B} S={S} Hq={Hq} Hkv={Hkv} D={D}, non-causal -> PASS")
```

**Expected output**: `... -> PASS`. The kernel accumulates online softmax in fp32, but several finite-precision effects still cause it to differ from the PyTorch float32 reference above: fp16 storage and rounding of inputs and operands, the finite precision of both hardware `exp2` and the cubic polynomial approximation, blockwise accumulation with a different summation order, and the final fp16 cast of `O`.

The `rtol`/`atol` values match the source kernel's own test and cover these effects together. A larger error usually points back to the softmax handoffs: a missing `s_ready`, `p_o_rescale`, or `p_ready_2` wait, or a `row_max` / `row_sum` update that did not reach the correction path.

FA4 reuses the TMA, `tcgen05`, TMEM, and barrier machinery developed for the GEMM kernels, but its dependency chain is longer: the QKᵀ MMA produces `S`, softmax transforms `S` into `P`, and the PV MMA uses `P` and `V` to update `O`. Because `S`, `P`, and `O` pass between different warpgroups and reuse the same TMEM allocation, the kernel must overlap these stages while ensuring that each tile is read or overwritten only after the corresponding handoff completes.

## Exercises

1. Consider one query row with `scale_log2=1`, `rescale_threshold=8`, `row_max=2`, `row_sum=3`, and `O=[4,6]`. Let the next block have `S=[5,4]` and `V=[[1,0],[0,1]]`. Compute `candidate_max`, `delta`, `new_ref`, `acc_scale`, `P`, and the updated `row_sum` and `O`. Repeat with `S=[11,10]`, and explain why only the second case rescales the old state.
2. Trace these four paths separately: Q/K in SMEM → S in TMEM, S in TMEM → P in TMEM, P in TMEM + V in SMEM → O in TMEM, and O in TMEM → O in GMEM. For each path, identify the executing role, source and destination storage, tile primitive, and hardware path. Which paths do not exist in the preceding GEMM kernel?
3. A column $c$ in the fp16 view maps to physical 32-bit column $\lfloor c/2\rfloor$. Use this relation to derive the physical column ranges of `S0`, `S1`, `P0`, `P1`, `O0`, and `O1`. Which regions overlap, and which waits or barriers prevent an overlapping region from being read or overwritten too early?
4. Trace one K/V block through `s_ready`, `p_o_rescale`, `p_ready_2`, and `o_ready`. For each barrier, identify who waits, who contributes arrivals, and which tile becomes safe to consume. Why does `p_o_rescale` expect 256 arrivals, and what overlap is gained by handing `P` to the PV MMA as 96 columns followed by 32 columns?
5. The driver warpgroup reduces its register ceiling to 48 registers per thread, the two softmax warpgroups raise theirs to 200, and WG2 uses 64. Compute the total register budget for the four 128-thread warpgroups, then compare it with assigning 200 registers to every thread in the CTA. Why do the softmax roles need the largest allocation, and how does reducing WG3's ceiling make that allocation possible?
6. The kernel already rewrites the natural exponential as base-2 `exp2`. Why can the hardware exponential path still bottleneck softmax? Explain how splitting the elements between hardware `exp2` and the FMA-based cubic approximation changes execution-unit utilization, and identify which online-softmax equations remain unchanged.
7. Let `SEQ_LEN_Q=6` and `SEQ_LEN_KV=8` with a bottom-right-aligned causal mask. What is the largest key index visible to query positions 0 and 5? With `BLK_N=4`, classify the K/V blocks for each query as fully valid, partially valid, or skipped. How does this difference affect causal task cost and scheduling order?
8. Let `num_qo_heads=32`, `num_kv_heads=8`, and `BLK_M=128`. Compute `GQA_RATIO` and `SEQ_Q_PER_TILE`. For `kv_head_idx=3`, map packed rows 0, 5, and 127 to `(sequence offset, query head)`, and explain why all 128 rows can share one K/V tile.
