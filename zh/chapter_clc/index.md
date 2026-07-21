(chap_clc)=
# 进阶调度：Cluster Launch Control

:::{admonition} 概览
:class: overview

- Persistent kernel 让已经驻留的 CTA 或 CTA cluster 连续计算多个 output tiles，从而减少 CTA 启动和公共准备工作的开销。
- Cluster Launch Control（CLC）允许正在运行的 CTA 或 cluster 取消一个尚未开始的 launch，并接管它的 grid coordinate。
- CLC 请求异步完成，结果通过 shared memory 和 `mbarrier` 交给 worker。先完成当前 tile 的 worker 可以继续领取 pending work，减少工作量不均匀造成的空闲时间。
:::

前面几章关注的是一块 tile 如何完成计算：A、B 如何搬入 SMEM，Tensor Core 如何执行 MMA，以及异步操作之间如何通过 `mbarrier` 交接数据。当一个矩阵被划分成许多 output tiles 后，kernel 还需要解决另一个问题：这些 tiles 应该按照什么顺序分配给 CTAs 或 CTA clusters？

假设输出矩阵被划分成 100 个 tiles。最直接的做法是启动 100 个 CTAs，让第 0 个 CTA 计算 tile 0，第 1 个 CTA 计算 tile 1，以此类推。GPU 通常无法同时运行全部 100 个 CTAs，因此会先运行其中一部分；某个 CTA 结束并释放资源后，硬件再启动后续 CTA，直到所有 tiles 都处理完毕。

传统的 fixed-number persistent kernel 使用另一种方式：它只启动一组长期运行的 CTAs 或 clusters，让每个 worker 在循环中连续计算多个 tiles。这样可以减少 CTA 启动和重复准备工作的开销，但也带来了新的调度问题：一个 worker 完成当前 tile 后，下一块 tile 从哪里来？

本章介绍 Blackwell 提供的 Cluster Launch Control（CLC）。CLC kernel 的 launch grid 仍然覆盖全部 output tiles，但运行中的 worker 可以取消尚未开始的 CTA 或 cluster launch，并接管它的 coordinate。这样既保留了完整 grid 的任务编号，又能让已经驻留的 workers 根据实际完成情况动态领取工作。

## 静态 persistent scheduler 的局限

下面把这种已经开始运行、可以反复取任务的 CTA 或 cluster 统称为 worker。

最简单的调度方法是提前算好每个 worker 要处理哪些 tiles。例如，现在有 12 个 tiles 和 4 个 workers，使用 grid-stride 分配后，结果可能是：

```text
worker 0: tile 0, 4, 8
worker 1: tile 1, 5, 9
worker 2: tile 2, 6, 10
worker 3: tile 3, 7, 11
```

如果四个 workers 能够同时运行，而且每块 tile 的计算量接近，这种静态分配没有问题。但 kernel 实际能够使用多少个 SM，并不总能在启动前准确知道。例如，其他 kernel 可能正在占用一部分 SM。假设上面的 worker 3 迟迟无法启动，那么 workers 0、1、2 完成各自的三块 tile 后，worker 3 才开始处理剩下的 `3、7、11`。这时，这个 kernel 的大部分 workers 都已经退出，只剩一个 worker 继续执行，形成很长的 launch tail。

不同 tiles 的计算量也可能不一样。边界处理、mask、稀疏计算或融合在 GEMM 周围的其他操作，都可能让某些 tiles 更慢。静态 scheduler 在工作真正开始前就已经确定了分工，无法根据实际完成时间重新分配任务。

CLC 会换一种方式组织这 12 个 tiles。它的 launch grid 包含 12 个 CTAs，它们的 `blockIdx.x` 分别是 0 到 11，并约定 `blockIdx.x = i` 的 CTA 负责 tile `i`。假设当前只能容纳三个 CTAs，并且 scheduler 首先启动了 CTA 0、1、2，那么 CTA 3 到 CTA 11 会暂时留在 launch queue 中，等待资源空闲。

假设 CTA 0 已经算完 tile 0。它可以先不退出，而是向硬件请求一份尚未开始的工作。如果硬件选择了 CTA 3，就会取消 CTA 3 的启动，并把 CTA 3 原本应当使用的 `blockIdx` coordinate 返回给 CTA 0。CTA 0 根据这个 coordinate 找到 tile 3，接着完成 tile 3，然后再次请求下一份工作。

被取消的 CTA 3 从未开始执行，因此也没有需要迁移的寄存器或执行状态。硬件交给 CTA 0 的只有 coordinate 3；这个 coordinate 在 kernel 中正好充当 tile 3 的任务编号。已经运行的 CTA 通过这种方式接管 pending launch 的工作，就是 CLC 所说的 work stealing。

所以，每个 coordinate 都只会被处理一次：它可能正常启动自己的 CTA，也可能在启动前被取消，再交给一个已经运行的 worker。只要 launch queue 中还有可以取消的 coordinate，空闲下来的 worker 就能继续计算，而不必等待某个预先指定的 CTA 启动。

上面的例子把一个 CTA 当作调度单位。如果 kernel 使用 thread block cluster，CLC 每次会取消一个尚未启动的 cluster launch，并把它的 coordinate 交给一个已经运行的 cluster。Thread block cluster 是从 Hopper 开始提供的执行层级，一组 CTAs 会被共同调度，可以进行 cluster 范围的同步，也可以访问 cluster 内其他 CTA 的 distributed shared memory；Blackwell 的 CLC 则负责动态调度这些 CTA 或 cluster coordinates。

## 一次 CLC 请求

假设一个 worker 已经拿到当前 tile。为了取得下一块工作，它先提交：

```text
clusterlaunchcontrol.try_cancel.async
```

`try_cancel` 会让 grid scheduler 尝试取消一个尚未启动的 CTA 或 cluster。硬件把结果编码为一条 16-byte 记录，并写入 shared memory。通常只选择一个 thread 提交请求；如果多个 threads 同时执行，就会产生多个取消请求，还必须分别准备结果位置，并在 barrier 的 arrival count 和 tx-count 中计入每个请求。

这个请求是异步的。指令发出后，worker 可以继续计算当前 tile；此时不能立即读取 shared memory 中的结果。CLC 会通过 `mbarrier` 报告这 16 bytes 何时写入完成。具体做法与上一章的 TMA load 相同：发起请求的 thread 报告一次 arrival，并把 16 bytes 登记到 tx-count；worker 等到对应的 barrier phase 完成后，才能查询结果。

请求完成后，先检查取消是否成功：

```text
clusterlaunchcontrol.query_cancel.is_canceled
```

这条查询返回一个 predicate。结果为 true，说明 scheduler 已经取消了一个 pending launch；这时再执行：

```text
clusterlaunchcontrol.query_cancel.get_first_ctaid
```

即可取得被取消 CTA 的 `(x, y, z)` coordinate；如果取消的是 cluster，则返回其中第一个 CTA 的 coordinate。kernel 再把这个 coordinate 换算成对应的 output tile。

结果为 false，说明这次请求没有取得可接管的 coordinate。最常见的原因是 launch queue 中已经没有 pending work，也可能是 scheduler 准备调度更高优先级的 kernel。worker 观察到失败后应结束取任务循环；按照 PTX 规定，此后再次提交取消请求属于未定义行为。

CLC 没有使用某个特殊数字表示“没有任务”。只有 `is_canceled` 返回 true 时，`get_first_ctaid` 的结果才有效；对失败的请求读取 coordinate 同样属于未定义行为。

如果调度单位是一个包含多个 CTAs 的 cluster，取消请求会一次接管整个 cluster。`get_first_ctaid` 返回 cluster 中第一个 CTA 的 coordinate，其余 CTAs 再加上各自的 local block index，得到自己的 grid coordinate。

## 把请求与当前计算重叠

worker 先用自己的 `blockIdx` 确定第一个 tile。此后，循环中的每一轮都执行以下步骤：

1. 提交 `try_cancel`，提前请求下一块可能的工作；
2. 在请求执行期间计算当前 tile；
3. 当前 tile 完成后，等待 CLC 对应的 `mbarrier`；
4. 查询取消是否成功；
5. 成功时使用返回的 coordinate 进入下一轮，失败时退出。

下面的伪代码省略 barrier 初始化、phase 更新和 response buffer 所需的 async-proxy fence，只展示取任务与计算的顺序：

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

为什么要在计算当前 tile 之前请求下一块工作？因为 grid scheduler 处理请求需要时间。如果等当前 tile 算完才提交请求，这段延迟会直接落在两块 tile 之间，worker 只能停下来等待。

提前提交后，scheduler 处理请求和当前 tile 的计算可以同时进行。等当前 tile 完成时，下一块工作的 coordinate 往往已经写入 shared memory。TMA 用计算覆盖数据搬运延迟，CLC 则用当前 tile 的计算覆盖调度请求延迟，两者采用的是同一种异步流水思路。

CLC 通过 async proxy 把 response 写入 shared memory，普通 thread 则通过 generic proxy 查询这份结果。`mbarrier` wait 用来确认异步 response 已经写完；实际代码在提交新请求前和读完 response 后，还必须按照 PTX 要求执行相应的 proxy fence，建立 async proxy 与 generic proxy 之间的访问顺序，防止下一轮异步写入与尚未结束的读取发生冲突。此外，kernel 还需要正确处理 barrier phase，以及 CTA 或 cluster 范围的 thread synchronization。

## 什么时候使用 CLC

静态 scheduler 与 CLC 可以共用同一个 tile 计算主体。它们只在“下一块 tile 从哪里来”这个问题上有所不同：

```text
静态调度：根据 worker ID 和迭代次数算出下一个 coordinate
CLC 调度：由硬件返回一个尚未启动的 CTA 或 cluster coordinate
```

静态调度几乎没有取任务开销。当可用 SM 数量稳定、各个 tiles 的成本接近时，静态公式通常已经足够。

CLC 更适合运行资源或 tile 成本难以提前确定的情况。如果部分 SM 被其他 kernel 占用，或者不同 tiles 的运行时间差异较大，先完成任务的 workers 可以继续接管 pending coordinates，从而减少只有少数 workers 留在 launch tail 中工作的时间。

在 TIRx 中，可以把 CLC 封装成动态 tile scheduler。GEMM mainloop 和 epilogue 只接收当前 tile coordinate，不需要知道它是由静态公式算出，还是由 CLC 返回。这样，kernel 的计算代码保持不变，只替换负责提供下一个 coordinate 的 scheduler。
