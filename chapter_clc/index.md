(chap_clc)=
# Advanced Scheduling: Cluster Launch Control

:::{admonition} Overview
:class: overview

- A persistent kernel lets resident CTAs or CTA clusters compute multiple output tiles, reducing launch and repeated setup overhead.
- Cluster Launch Control (CLC) lets a running CTA or cluster cancel a launch that has not started and take over its grid coordinate.
- A CLC request completes asynchronously and returns its result through shared memory and an `mbarrier`. Workers that finish early can claim pending work instead of remaining idle.
:::

The previous chapters focused on how one tile is computed: how A and B reach SMEM, how the Tensor Core executes MMA, and how asynchronous operations hand data off through an `mbarrier`. Once a matrix has been divided into many output tiles, the kernel faces another question: in what order should those tiles be assigned to CTAs or CTA clusters?

Suppose the output matrix contains 100 tiles. The most direct launch uses 100 CTAs: CTA 0 computes tile 0, CTA 1 computes tile 1, and so on. The GPU usually cannot run all 100 CTAs at once. It starts a subset, then launches later CTAs as running CTAs finish and release resources, until every tile has been processed.

A conventional persistent kernel takes a different approach. It launches a fixed pool of long-lived CTAs or clusters and lets each worker compute several tiles in a loop. This can reduce CTA launch and repeated setup overhead, but it introduces a scheduling question: after a worker finishes its current tile, where does the next tile come from?

Blackwell's Cluster Launch Control (CLC) answers that question. A CLC kernel still launches a grid that covers the complete output space, but a running worker can cancel a CTA or cluster launch that has not yet started and inherit its coordinate. The launch grid retains a coordinate for every task, while resident workers dynamically claim work according to when they become available.

## The limits of a static persistent scheduler

In this chapter, a *worker* means a CTA or cluster that is already running and can repeatedly claim tasks.

The simplest scheduler decides every worker's tiles in advance. With 12 tiles and four workers, a grid-stride assignment might be:

```text
worker 0: tile 0, 4, 8
worker 1: tile 1, 5, 9
worker 2: tile 2, 6, 10
worker 3: tile 3, 7, 11
```

This static assignment works well when all four workers can run together and the tiles cost roughly the same amount. In practice, a kernel cannot always know how many SMs will be available at launch time. Another kernel may already occupy part of the GPU. If worker 3 is delayed, workers 0, 1, and 2 may finish their three tiles before worker 3 even begins tiles `3, 7, 11`. Most workers assigned to this kernel have then exited, leaving one worker to execute a long launch tail.

Tiles can also have unequal costs. Boundary handling, masks, sparse computation, or work fused around GEMM may make some tiles slower than others. A static scheduler commits to an assignment before execution begins and cannot redistribute work in response to actual completion times.

CLC organizes the same 12 tiles differently. Its launch grid contains 12 CTAs with `blockIdx.x` values from 0 through 11, and CTA `i` is assigned tile `i`. Suppose resources initially permit only three CTAs to run and the scheduler starts CTAs 0, 1, and 2. CTAs 3 through 11 then remain pending in the launch queue.

Suppose CTA 0 finishes tile 0. Instead of exiting, it asks the hardware for work whose launch has not begun. If the scheduler selects CTA 3, it cancels CTA 3's launch and returns the `blockIdx` coordinate that CTA 3 would have used. CTA 0 decodes that coordinate, computes tile 3, and then asks for another task.

The canceled CTA never started, so no register state or execution state has to move. Hardware gives CTA 0 only coordinate 3, which the kernel already uses as the task identifier for tile 3. This reassignment of a pending launch coordinate to a running worker is CLC work stealing.

Each coordinate is therefore processed exactly once. Its CTA either launches normally, or its launch is canceled first and the coordinate is handed to an existing worker. As long as the launch queue contains a cancelable coordinate, a worker that becomes free can continue computing instead of waiting for a predetermined CTA to start.

The example above schedules individual CTAs. When a kernel uses thread block clusters, CLC cancels one pending cluster launch at a time and hands its coordinate to a running cluster. Introduced with Hopper, a thread block cluster is a group of co-scheduled CTAs that can synchronize at cluster scope and access distributed shared memory within the cluster. Blackwell CLC dynamically schedules CTA or cluster coordinates; it is separate from the cluster execution model itself.

## One CLC request

Assume that a worker already owns its current tile. To request another task, one thread submits:

```text
clusterlaunchcontrol.try_cancel.async
```

`try_cancel` asks the grid scheduler to cancel a CTA or cluster that has not begun executing. Hardware encodes the response in a 16-byte record and writes it to shared memory. A kernel normally selects one thread to submit the request. If several threads issue the instruction, they create several cancellation requests and must provide separate response locations while accounting for every request in the barrier's arrival count and tx-count.

The request is asynchronous. The worker may continue computing its current tile after issuing the instruction, but it cannot immediately read the shared-memory response. CLC uses an `mbarrier` to report when all 16 bytes have been written. As with a TMA load, the issuing thread contributes one arrival and registers 16 bytes with the tx-count. The worker may query the response only after the corresponding barrier phase completes.

It first checks whether the cancellation succeeded:

```text
clusterlaunchcontrol.query_cancel.is_canceled
```

This query returns a predicate. A `true` result means that the scheduler canceled a pending launch. The worker can then execute:

```text
clusterlaunchcontrol.query_cancel.get_first_ctaid
```

to obtain the `(x, y, z)` coordinate of the canceled CTA, or of the first CTA in the canceled cluster. The kernel converts that coordinate into the corresponding output tile.

A `false` result means that this request did not obtain a coordinate to take over. The most common reason is that the launch queue contains no pending work, although the scheduler may also be preparing to schedule a higher-priority kernel. Once a worker observes failure, it must leave the work-request loop. PTX defines a subsequent cancellation request as undefined behavior.

CLC does not use a sentinel number to represent "no task." The coordinate returned by `get_first_ctaid` is valid only when `is_canceled` returns `true`; querying the coordinate after a failed request is also undefined behavior.

If the scheduling unit is a cluster with several CTAs, one request takes over the entire cluster. `get_first_ctaid` returns the coordinate of the cluster's first CTA. Each CTA combines that coordinate with its local block index to recover its own grid coordinate.

## Overlapping the request with the current tile

A worker begins with the tile identified by its own `blockIdx`. Every loop iteration then follows five steps:

1. Submit `try_cancel` early to request a possible next task.
2. Compute the current tile while the request is in flight.
3. After the tile finishes, wait on the CLC request's `mbarrier`.
4. Query whether cancellation succeeded.
5. Continue with the returned coordinate on success, or exit on failure.

The pseudocode below omits barrier initialization, phase updates, and the async-proxy fences required for the response buffer. It shows only the ordering between task acquisition and computation:

```text
tile = decode(blockIdx)

while true:
    async_try_cancel(result, barrier)
    compute(tile)
    wait(barrier)

    if not is_canceled(result):
        break

    tile = decode(get_first_ctaid(result))
```

Why request the next task before computing the current tile? The grid scheduler needs time to process the request. If the worker waits until its current tile is finished, that latency falls directly between two tiles and leaves the worker idle.

Issuing the request first allows scheduling to overlap the current tile's computation. By the time the tile finishes, the coordinate for the next task is often already in shared memory. TMA hides data-transfer latency behind computation; CLC uses the same asynchronous pipeline idea to hide scheduling latency.

CLC writes its response to shared memory through the async proxy, while an ordinary thread queries it through the generic proxy. The `mbarrier` wait confirms that the asynchronous response write has completed. Real code must also execute the PTX-required proxy fences before submitting a new request and after reading the response. These fences order accesses to the response buffer across the async and generic proxies, preventing a later asynchronous write from racing with data that is still being read. The kernel must also manage the barrier phase and the required CTA- or cluster-wide thread synchronization.

## When to use CLC

A static scheduler and CLC can share exactly the same tile computation. They differ only in how the next coordinate is obtained:

```text
Static scheduling: derive the next coordinate from the worker ID and iteration
CLC scheduling:    receive a pending CTA or cluster coordinate from hardware
```

Static scheduling has almost no task-acquisition overhead. When SM availability is stable and tile costs are similar, a static formula is often sufficient.

CLC is more useful when available resources or tile costs are difficult to predict. If other kernels occupy some SMs, or if tile execution times vary, workers that finish early can claim pending coordinates and shorten the period when only a few workers remain active at the launch tail.

In TIRx, CLC can be wrapped in a dynamic tile scheduler. The GEMM mainloop and epilogue receive only the current tile coordinate; they do not need to know whether it came from a static formula or a CLC response. The computation stays unchanged while the scheduler that supplies the next coordinate is replaced.
