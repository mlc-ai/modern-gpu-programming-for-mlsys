(chap_layout_generations)=
# Data Layout Through GPU Generations

Building on the layout notation from {ref}`chap_data_layout` (`S[...]`, named axes, swizzle), this
chapter is organized by generation — **Ampere, Hopper, Blackwell** — because what changed across
them is exactly *how operands reach the tensor core*. Each generation's memory and compute engines
demand a *specific* operand layout, and getting it wrong is silently slow (or wrong); the Blackwell
TMEM specifics are in {ref}`chap_tmem`.

## Two Constraints That Never Went Away

Two layout rules hold on every generation, tensor cores or not:

- **Global-memory coalescing.** The 32 lanes of a warp should address one contiguous, aligned
  segment so the memory system serves them in as few transactions as possible.
- **Shared-memory bank conflicts.** SMEM is divided into 32 banks; if several lanes in a warp
  address different rows of the *same* bank, those accesses serialize. The fix is **swizzle** —
  permuting the address mapping so a warp's lanes spread across distinct banks.

Everything below is a third demand layered on top: the layout the *tensor core* requires for its
operands.

## Ampere — Register Fragment over warp/lane

On Ampere-class GPUs (`sm_80`) the tensor-core instruction is the warp-level
`mma.sync.aligned.m16n8k*`, and it reads its operands from **registers**: A, B, and the C/D
accumulator are all per-thread register fragments spread across the warp's 32 lanes. The data path
is therefore a shuffle through registers:

```text
SMEM --ldmatrix--> registers --mma.sync--> registers --stmatrix--> SMEM
```

### What the Tensor Core expects: an m8n8 register fragment

The register fragment is built from **8×8 ("m8n8") sub-tiles** — the unit `ldmatrix` moves and the
tensor core reads. For `mma.m16n8k16` (fp16/bf16 in, fp32 accumulate), the 32 lanes are carved
**8 along M × 4 along N**, and each lane owns a few registers:

- **C/D accumulator (M×N = 16×8):** lane `l` holds rows `m ∈ {l/4, l/4 + 8}` and columns
  `n ∈ {2·(l%4), 2·(l%4)+1}` — four fp32 values per lane (two 8-row halves × two adjacent columns).
  Four consecutive lanes cover one row's eight columns.
- **A operand (M×K = 16×16):** same M carve as C/D; K runs across `l%4` and the registers — four
  b32 registers per lane, each packing two fp16 along K.
- **B operand (K×N = 16×8):** K matches A; N is the 8-lane group — two b32 registers per lane.

This is the concrete `m16n8k16` C/D fragment behind the named-axes demo in {ref}`chap_data_layout`
(`S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]`): each lane holds two adjacent columns per 8×8, with
four consecutive lanes covering one row.

### `ldmatrix`: SMEM → the fragment

![ldmatrix loads an 8x8 SMEM tile into the warp register fragment; stmatrix is the reverse](../img/ldstmatrix.svg)

`ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16` loads one, two, or four 8×8 16-bit
matrices from SMEM into that fragment in a single warp-collective instruction:

- **Addresses come from lanes.** Each source row's base address is supplied by one lane: matrix
  `m`, row `r` is addressed by lane `m·8 + r`. So `.x1` uses lanes 0–7, `.x2` lanes 0–15, and `.x4`
  lanes 0–31 as the row-address suppliers.
- **The result is distributed** so lane `l` ends up holding row `l/4`, columns `2·(l%4)` and
  `2·(l%4)+1` (one b32 packing the two adjacent fp16) — exactly the fragment the MMA reads.
- **`.trans`** transposes each 8×8 as it loads (the two halves map *down a column* instead of
  across a row), which is how you feed an operand stored the opposite way from what the MMA wants.

That is why the instruction exists: a plain per-lane `ld.shared` loop cannot cheaply produce the
MMA's scattered fragment, but one `ldmatrix` does the whole SMEM→register shuffle the tensor core
demands.

### `stmatrix`: the fragment → SMEM

`stmatrix` is the reverse — register→SMEM, with the same lane/address mapping. It is the epilogue
counterpart: after the MMA, the accumulator is scattered across lanes in the C/D layout above, and
`stmatrix` writes it back into a SMEM tile, from which a coalesced `st.global` (or, later, a TMA
store) reaches GMEM. The Ampere story is this shuffle in both directions: the register fragment is
fixed by hardware, and ldmatrix/stmatrix are the bridges between it and SMEM.

### Swizzle: the same conflict, already on Ampere

Swizzle is not a Hopper invention — Ampere kernels already needed it, because the SMEM tile is
*written* one way (coalesced from GMEM, along a **row**) and *read* another way (by `ldmatrix`,
along a **column**). With a plain row-major tile the row write hits 8 distinct banks (conflict-free)
but the column read hits one bank 8 times (an 8-way conflict); col-major is the mirror image, and no
unpermuted layout satisfies both:

![Row write hits 8 distinct banks (conflict-free); column read hits one bank 8 times (conflict)](../img/swizzle_conflict.svg)

The fix is the XOR **swizzle** from {ref}`chap_data_layout` — store `(r, c)` at column `c ⊕ r`, which
makes the row write *and* the column read conflict-free at once. Hopper later folds the same
permutation into the TMA and MMA descriptors (next section); on Ampere it lived in hand-written
index math.

## Hopper — `wgmma`, SMEM Descriptors, and Swizzle Formats

### What the Tensor Core expects: a SMEM matrix descriptor

Hopper (`sm_90`) removes the per-lane register shuffle on the *input* side. `wgmma` reads its A and
B operands **directly from SMEM** — no `ldmatrix`. But the Tensor Core does not read arbitrary
SMEM: it reads through a 64-bit **matrix descriptor** that fixes the one format the operand may be
stored in. A GEMM logically has to find where `A[m, k]` lives; the descriptor is what turns
`(m, k)` into a SMEM address. It has four parts:

| Field | Meaning |
|---|---|
| **start_address** | base of the tile in SMEM, 16-byte-aligned (stored as `addr ≫ 4`) |
| **swizzle** | the swizzle format — sets the **atom shape** (8 × 128/64/32/16 B) and the XOR pattern inside it |
| **ldo** — leading byte offset | stride to the next atom along the **major** dim |
| **sdo** — stride byte offset | stride to the next atom along the **other** dim |

Given those, A(M×K) is laid out as a 2-D grid of **atoms**. The swizzle format sets each atom's
shape — 8 × 128 B for `SWIZZLE_128B` (8 × 64 / 32 / 16 B for the smaller modes) — and how its bytes
are XOR-permuted inside (the Ampere section's swizzle), so the `wgmma` read is bank-conflict-free.
**ldo** and **sdo** are the byte strides *between* atoms, and which axis each walks depends on the
operand's major-ness: **ldo** strides along the **major** dimension, **sdo** along the **other**.
For a K-major tile (A stored K-contiguous) that puts `ldo` along K and `sdo` down M; an MN-major
tile swaps them. To resolve `A[m, k]` the hardware combines the two strides with the swizzle inside
the atom:

![A SMEM matrix descriptor (start_address, ldo, sdo, swizzle) tiles A(M×K) into 8×N B swizzle atoms, with ldo/sdo the strides between atoms](../img/smem_descriptor.svg)

So the kernel's job is to write A into SMEM in exactly this atom-tiled, swizzled format — the TMA
load does that — and hand `wgmma` a descriptor whose `ldo` / `sdo` / `swizzle` match (in the kernels
these are literal constants, e.g. `ldo = 1`, `sdo = 64`, `swizzle = 64B`). The swizzle is itself a
first-class format on Hopper — `SWIZZLE_NONE / 32B / 64B / 128B` — and the *same* format is named in
both the TMA descriptor that fills the tile and the `wgmma` descriptor that reads it, so the load and
the MMA agree by construction. (On Ampere that
same permutation lived in hand-written index math.)

The element arrangement inside one atom for each format (`SWIZZLE_128B` = 8 × 128 B, `SWIZZLE_64B`,
`SWIZZLE_32B`) is the swizzle-atom demo in {ref}`chap_data_layout`.

The **output**, though, is unchanged: `wgmma`'s accumulator `D` is still a per-thread **register**
fragment in the same m8n8 layout as Ampere (above). A Hopper GEMM reads operands the new way but
writes its accumulator and runs its epilogue exactly as before — moving the accumulator out of
registers waits for Blackwell's TMEM.

## Blackwell — `tcgen05` and TMEM

### What the Tensor Core expects: SMEM operands and a TMEM accumulator

Blackwell (`sm_100`) keeps Hopper's SMEM matrix descriptor for the A/B operands (an A operand may
also be read from TMEM), but moves the **accumulator** into TMEM: it never visits a register
fragment during the compute phase the way an Ampere `mma` accumulator does — it stays in TMEM until
the epilogue reads it out. How the (M, N) accumulator and the A/B operands split across one or two
CTAs (`cta_group::1` vs `cta_group::2`) is covered in {ref}`chap_tensor_cores`. The layout that is
genuinely new here is the **scale factors** of a block-scaled MMA.

### Scale-factor layout in TMEM

A block-scaled MMA (mxfp8, nvfp4) carries two operands beyond A and B — `SFA (M, SFK)` and
`SFB (N, SFK)`, where `SFK = K / block` — and, unlike A and B, **the scale factors live in TMEM**,
not SMEM. They take the SMEM→TMEM detour: a TMA load brings them into SMEM, then `tcgen05.cp` copies them into TMEM before the MMA.

The TMEM layout itself (the PTX *tcgen05 MMA scale-factor A layout*) is the
lane-replication example from {ref}`chap_data_layout`: a 128-row scale vector packs into 32 lanes
(row → lane `r % 32`, `r // 32` along TMEM columns at stride `epc = 4`) and is broadcast `warpx4`
to all 128 reading lanes (`R[4 : 32@TLane]`).

What is new here is the byte packing: how many distinct scales a `uint32` column holds depends on
the **scale_vec** mode, matching the PTX *scale-factor A* 1x/2x/4x layouts:

![scale_vec byte packing: 1X (fp8) broadcasts one scale across 4 bytes; 2X (mxfp4) packs two scales each duplicated; 4X (nvfp4) packs four K-block scales](../img/sf_scale_vec.svg)

Under `cta_group::2` the scale factors split the way their data does — **SFA follows A** (each CTA
holds the M-half matching its A rows) and **SFB is multicast** to both CTAs ({ref}`chap_tensor_cores`).

So the **m8n8 register fragment** recurs across generations: it is what `ldmatrix` builds on
Ampere, what `wgmma` outputs on Hopper, and what `tcgen05.ld` reads TMEM into on Blackwell
({ref}`chap_tmem`) — one register layout across all three.

## The Throughline

The trend across generations is that more of the layout is **described to hardware** instead of
open-coded with shuffle instructions:

| Generation | Operands read from | Layout described by |
|---|---|---|
| Ampere (`sm_80`) | registers | `ldmatrix`/`stmatrix` + hand-staged (swizzled) SMEM |
| Hopper (`sm_90`) | SMEM | `wgmma` matrix descriptor + TMA box & swizzle |
| Blackwell (`sm_100`) | SMEM / TMEM | `tcgen05` matrix descriptor + TMEM accumulator & scale-factor layouts |

The descriptors do not remove the work — they relocate it. The kernel still has to place bytes in
the exact format the engine reads: a TMA load, the matrix descriptor it feeds, and the MMA that
consumes it must all agree on the same swizzle, or the tensor core reads scrambled data.
