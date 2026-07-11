(chap_background)=
# GPU Execution Model

:::{admonition} Overview
:class: overview

- A GPU kernel's execution is shaped first by its thread hierarchy: thread, warp, warpgroup, CTA, cluster, and grid each correspond to a different scale of cooperation. Many Blackwell operations have their own natural scope: a TMA copy is launched by a single thread, a full TMEM accumulator is read back by four warps working on separate 32-lane windows, and a 2-CTA cooperative MMA spans two CTAs.
- Data does not live in a single place. GMEM, SMEM, TMEM, and registers offer different tradeoffs in capacity, latency, and access scope; clusters also use DSMEM so one CTA can access another CTA's shared memory. A core task of a high-performance kernel is to move data efficiently between these spaces.
- Compute and data movement are handled by different hardware engines. CUDA cores handle address calculation, control flow, and scalar logic; Tensor Cores perform the main matrix computation; TMA moves data asynchronously. We end the chapter with a GEMM data pipeline that shows how overlap keeps multiple engines busy at the same time.
:::

To write high-performance GPU kernels, we first need to understand how threads are organized, where
data lives, and how different hardware engines work together. This chapter follows those three
questions. We begin with the GPU thread hierarchy, then cover the memory spaces used to store and
move data, and finally introduce the engines responsible for computation and data movement. A GEMM
pipeline at the end ties these ideas together and shows how computation can overlap with data
movement.

We begin with the Blackwell SM architecture. The figure below shows the main hardware units used in
this chapter.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/sm_architecture.html" title="Blackwell SM architecture" loading="lazy"
        style="width:100%; height:620px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Click a component to inspect the warps, warpgroups, shared memory, Tensor Memory, Tensor Cores, and
TMA engine inside a Blackwell SM.*

## The Execution Hierarchy

A GPU does not manage thousands of threads as one flat collection. Instead, it organizes them into
several levels, each with a different cooperation granularity. The figure below shows the thread
hierarchy on Blackwell.

```{raw} html
<iframe src="../demo/thread_hierarchy.html" title="Blackwell thread hierarchy" loading="lazy"
        style="width:100%; height:520px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Click a component to step through the thread, warp, warpgroup, CTA, cluster, and grid levels.*

- **Thread**: the scalar unit of execution. Each thread has its own program counter and its own
  registers, and it is identified by a lane ID within its warp.
- **Warp**: 32 threads that execute in SIMT (*single instruction, multiple threads*). The lanes of
  a warp issue the same instruction together, yet each keeps its own registers and can be masked off
  on its own, which is what lets the lanes of a single warp follow different branches.
- **Warpgroup**: four consecutive warps, or 128 threads. Hopper introduced the warpgroup as the
  unit that issues warpgroup-level MMA (`wgmma`). On Blackwell, its four warps can also cover the
  four 32-lane windows of Tensor Memory.
- **CTA** (*Cooperative Thread Array*, what CUDA also calls a thread block): the basic unit the
  hardware schedules. A CTA runs on a single SM and owns a private shared-memory allocation inside
  it. Several CTAs can be resident on the same SM at once, and when they are, they divide up that
  SM's shared-memory capacity between them.
- **Cluster**: a group of cooperating CTAs that may live on different SMs. The CTAs in a cluster
  can synchronize with one another and can read and write each other's shared memory, a capability
  known as distributed shared memory.

Blackwell's key operations are not all issued by the same group of threads. A single thread launches
a TMA copy, which the hardware then executes. Each warp issues warp-level TMEM loads for its own
32-lane window. One designated thread commits a `tcgen05` MMA, while a 2-CTA cooperative MMA spans
two CTAs.

We call the set of threads involved in an operation its **scope**. Analyzing a kernel requires
considering the operation's scope together with its data layout and dispatch mechanism.

## Memory Spaces

The thread hierarchy tells us how computation is organized. We next need to determine where the data
lives. A GPU provides several memory spaces with different tradeoffs in capacity, latency, and access
scope. A kernel must move data efficiently among them.

| Memory | Ownership | Role | Notes |
|--------|-----------|------|-------|
| **Global (GMEM)** | Device-wide | Persistent tensor storage | Large HBM, shared by all SMs |
| **Shared (SMEM)** | Per-CTA (one SM) | Tile staging | Low-latency scratchpad; up to 228 KB/SM on B200 |
| **Tensor Memory (TMEM)** | Per-CTA | MMA accumulator storage | Introduced with Blackwell; used by `tcgen05` |
| **Register File (RF)** | Per-thread | Scalars and per-thread tile fragments | Fast; holds epilogue/temp values |

**Tensor Memory (TMEM)** is an on-chip storage space introduced with Blackwell. On earlier
architectures, MMA accumulators usually lived in registers. As MMA tiles grew, those accumulators
consumed a large share of the register file. Blackwell's `tcgen05` writes accumulators to TMEM
instead, reducing this register pressure.

TMEM can be viewed as a two-dimensional scratchpad used by a CTA. It contains 128 rows,
corresponding to 128 TMEM lanes, and up to 512 columns, each 32 bits wide. Logically, this space
belongs to the CTA; physically, it remains on the SM.

Programs manage TMEM explicitly. A kernel must allocate and free it, and the epilogue must explicitly
read MMA accumulators from TMEM back into registers. To read a full 128-lane accumulator, the four
warps in a warpgroup each load their own 32-lane TMEM window.

### Distributed Shared Memory Across a Cluster

A cluster can contain CTAs running on different SMs. Each CTA still owns its own shared memory, but
**distributed shared memory (DSMEM)** allows other CTAs in the same cluster to access that data.

This capability avoids unnecessary round trips through GMEM. One CTA can directly access another
CTA's SMEM without requiring the owner to write the data back to GMEM for the peer to reload. When
an asynchronous operation moves such data, a completion barrier notifies later computation after the
transfer finishes.

The figure below shows the DSMEM access path in a 2-CTA cluster. Each CTA retains its own SMEM but can
read the other CTA's SMEM.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/cta_cluster.html" title="A 2-CTA cluster sharing distributed shared memory" loading="lazy"
        style="width:100%; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Click a component to see how the two CTAs access each other's shared memory through DSMEM.*

In the 2-CTA GEMM shown above, each CTA stores its own slices of A and B and reads the peer CTA's B
slice through DSMEM. Here, sharing does not merge the two SMEM allocations. It means only that CTAs
in the same cluster can access one another's data across SMs.

The two CTAs can also form `cta_group=2` and execute a cooperative MMA that produces a larger output
tile.

## Compute: CUDA Cores and Tensor Cores

The thread hierarchy determines how computation is organized, and the memory spaces determine where
data lives. The arithmetic itself is carried out by compute units inside the SM. An SM mainly has two
kinds of compute units: CUDA cores and Tensor Cores.

- **CUDA cores** are general-purpose SIMT ALUs. They run the scalar and vector instructions that
  handle index arithmetic, elementwise math, reductions, and control flow.
- **Tensor Cores** are fixed-function units that perform a dense matrix multiply-accumulate at *tile*
  granularity, computing $D = AB + C$ in a single instruction.

Tensor Cores provide much higher arithmetic throughput than CUDA cores, often by a factor of 10 or
more in FLOP/s. Dense linear algebra workloads such as GEMM, convolution, and attention can approach
peak performance only by making effective use of Tensor Cores. A high-performance kernel must also
prepare data in time so that the Tensor Cores do not sit idle waiting on data or dependencies.

Different GPU generations change not only Tensor Core throughput, but also the programming interface
and accumulator placement. Hopper introduced asynchronous warpgroup MMA (`wgmma.mma_async`).
Blackwell's fifth-generation Tensor Core, `tcgen05`, stores accumulators in Tensor Memory rather than
registers. Later chapters discuss these differences in detail.

Clusters also introduce two important GEMM uses. **2-CTA cooperative MMA** allows two CTAs to each
provide part of the SMEM operands and jointly issue a larger Tensor Core MMA tile. **TMA multicast**
allows one GMEM load to deliver the same tile to multiple CTAs, avoiding redundant global memory
traffic from each CTA loading the same data separately. Both rely on the cluster and DSMEM mechanism
introduced earlier.

## The GEMM Data Pipeline

The previous sections introduced the thread hierarchy, memory spaces, data-movement mechanism, and
compute units separately. We now connect them with a GEMM pipeline and look at how these hardware
structures work together. The figure below shows the main units involved in a three-stage GEMM tile
pipeline.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/pipeline_arch.html" title="Blackwell GEMM data pipeline" loading="lazy"
        style="width:100%; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Click a stage to inspect the data path from load through MMA to epilogue on Blackwell.*

A single GEMM tile usually flows through three stages.

1. **Load:** A TMA copy moves an A or B operand tile from GMEM to SMEM. One thread issues the copy and
   records the expected number of arriving bytes. As data reaches SMEM, the TMA engine updates the
   progress count. The completion barrier fires only after all expected bytes have arrived.
2. **Compute:** A `tcgen05` MMA reads operand tiles from SMEM and accumulates the product into a TMEM
   tile. One designated thread commits the MMA; when computation completes, the hardware signals the
   corresponding barrier.
3. **Epilogue:** A warpgroup reads the TMEM accumulator back into registers, converts the result to
   the output dtype, and writes it to GMEM. This step often stages through SMEM and may use a TMA
   store for the final writeback.

These stages have data dependencies, but they do not need to run entirely in sequence. A naive kernel
executes load, wait, compute, wait, and store one after another, leaving the hardware units idle in
turn.

A high-performance kernel organizes the stages as a pipeline. While a Tensor Core computes tile `k`,
the TMA engine can move tile `k+1`, and the epilogue can process the output of tile `k-1`. The barrier
and phase model coordinates safe handoffs between these asynchronous stages. The GEMM optimizations
later in the book build on this mechanism.
