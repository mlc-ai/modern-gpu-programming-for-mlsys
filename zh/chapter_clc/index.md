(chap_clc)=
# 进阶：Cluster Launch Control

:::{admonition} 概览
:class: overview

- Persistent kernel 会让固定数量的 CTA 或 CTA cluster 驻留在 GPU 上（通常规模接近每个 SM 一个 active work owner，但不依赖严格的 1:1 映射），并让它们循环处理多个 output tile，而不是每个 tile 启动一个 CTA。
- Cluster Launch Control 是 Blackwell 的硬件机制，它让一个驻留中的 cluster 可以在运行时请求另一个 tile。它是一条硬件 work-stealing 路径，围绕两条 PTX 指令构建：一条指令请求 work，另一条指令读回请求是否成功。
- 主要收益是改善 tail 行为。当 tile 开销不均匀，或者 tile 数量不能被可用 SM 均匀整除时，提前完成的 CTA 可以继续拉取更多 work，而不是空等。
:::

Persistent GEMM 不会把 CUDA grid 当作固定的 “one CTA per output tile” launch。相反，它启动较小数量的长生命周期 CTA 或 CTA cluster。每个 CTA/cluster 计算一个 tile，然后前进到另一个 tile，再继续计算，直到输出空间完成。这正是 {ref}`chap_gemm_advanced` 中逐步构建出的执行模式。

一旦 kernel 变成 persistent，主要调度问题就很简单：当一个 CTA 或 cluster 完成当前 tile 后，下一个 tile 从哪里来？

最简单的答案是静态公式。例如，kernel 可以根据 CTA id 计算 tile coordinate，然后按 grid stride 前进。这很容易实现，并且当所有 tile 开销大致相同、tile 数量在 GPU 上分布均匀时效果很好。但这个 schedule 在 work 真正运行之前就已经决定好了。如果少数 tile 更慢，或者最后几个 tile 分配不均匀，有些 SM 会提前完成自己的份额，而另一些 SM 仍在处理 tail。

Cluster Launch Control，简称 CLC，会改变这个调度模型。Persistent cluster 不再预先决定完整分配，而是可以向硬件 grid scheduler 请求另一个尚未 launch 的 cluster 的 work。如果请求成功，当前 cluster 会接管那个 cluster coordinate，并计算对应 tile。如果请求失败，就说明没有更多 work 可以 steal，loop 退出。

这和 thread block cluster 本身不是一回事。Thread block cluster（一起 launch 的 CTA，带有 cluster-level synchronization，并能访问 distributed shared memory）是在 Hopper 引入的（{ref}`chap_background`）。CLC 是 Blackwell 新增的机制，它让这些 cluster coordinate 上的调度变成动态的。Cluster 已经是 launch 的单位；CLC 让一个已经运行的 cluster 可以取消一个 pending launch，并继承它的 coordinate。

## 两条指令

Cluster Launch Control 通过两条 PTX 指令暴露。第一条指令向 grid scheduler 发送一个异步请求。第二条指令读取响应。

请求指令是 `clusterlaunchcontrol.try_cancel.async`。

一次 `try_cancel` 会请求 scheduler 取消一个 pending cluster 的 launch，并把那个 cluster 的 coordinate 返回给调用方。响应会作为一个 16-byte record 写入 shared memory。由于请求是异步的，指令不会等待响应到达。相反，完成事件通过 `mbarrier` 报告，使用的还是 TMA 中同样的 barrier-and-phase 模型。

这个细节很重要，因为它意味着 CLC 不会引入新的等待模型。Kernel 发起请求，把它关联到一个 barrier，随后在读取响应之前等待这个 barrier。响应到达通过带 byte-count completion 的 barrier 发出信号，整体风格和其他异步硬件操作一致（见 {ref}`chap_async_barriers`）。

Barrier 触发后，kernel 使用 query 指令。

第一个 query 是 `clusterlaunchcontrol.query_cancel.is_canceled`。它返回一个 predicate，告诉 kernel cancellation 是否成功。True predicate 表示 scheduler 找到了一个 pending cluster launch，取消了它，并返回了它的 coordinate。False predicate 表示没有 pending work 可以拿了。

只有当 `is_canceled` 为 true 时，kernel 才应该读取 coordinate。读取使用 `clusterlaunchcontrol.query_cancel.get_first_ctaid`，它会提取被取消 cluster 的第一个 CTA id。这个 CTA id 是一个 coordinate vector，通常读作 `(x, y, z)`，kernel 会把它解码成下一步要计算的 output tile。

这个协议中没有数值形式的 sentinel tile id。Kernel 根据 predicate 分支。如果 predicate 为 true，coordinate 有效。如果 predicate 为 false，work-stealing loop 完成。

从底层看，这个形状直接来自 CLC 的实际行为。硬件不是从软件队列里分配一个抽象 task。它取消的是一个尚未发生的 cluster launch。因此，成功响应包含一个真实的 cluster coordinate。失败响应只是意味着 launch queue 已经耗尽。

## Work-Stealing Loop

有了这两条指令，persistent scheduler 就变成一个短循环。

在循环中的任意时刻，cluster 都负责计算一个 tile。在开始这个 tile 之前，它会为可能的下一个 tile 发送一个 `try_cancel` 请求。请求异步运行。当 scheduler 处理这个请求时，cluster 计算当前 tile。

当前 tile 完成后，cluster 等待与 `try_cancel` 响应关联的 `mbarrier`。响应就绪后，它调用 `query_cancel.is_canceled`。如果 predicate 为 true，它调用 `query_cancel.get_first_ctaid`，解码返回的 coordinate，并把它作为下一个 tile。如果 predicate 为 false，就说明没有 work 了，cluster 退出。

代码形状是：

1. 为可能的下一个 tile 发起 `try_cancel`；
2. 在请求飞行期间计算当前 tile；
3. 等待响应 barrier；
4. 查询 cancellation 是否成功；
5. 要么用返回的 coordinate 继续，要么退出。

请求放置的位置让这个循环有价值。Cluster 不是等当前 tile 完成之后才请求更多 work。它先请求，再计算。这样 scheduler 请求就与有用计算重叠。当前 tile 完成时，下一个 tile 的答案往往已经可用。

这和 persistent kernel 在其他位置使用异步 copy 与 tensor-core barrier 的原因相同。Kernel 会避免把长延迟操作直接放在关键路径上。CLC 把同样的思想应用到 tile scheduling：提前请求下一份 work，计算当前 work，然后在需要时消费调度结果。

## 与 Persistent GEMM 的关系

{ref}`chap_gemm_advanced` 中的 persistent GEMM 在主要讲解里使用静态 scheduler。静态 scheduler 更容易解释，因为下一个 tile 可以直接从 loop state 计算出来。例如，`ClusterPersistentScheduler2D` 这样的 scheduler 可以用 output tile 空间上的 grid-stride pattern 分配 tile。

CLC 是这种静态分配的动态替代品。外层 loop 保持不变：每个驻留 cluster 反复计算一个 output tile，然后前进到另一个。变化的是下一个 tile 从哪里来。使用静态 scheduler 时，下一个 tile 由公式计算。使用 CLC 时，下一个 tile 由硬件 work stealing 返回。

这个差异在 launch 尾部最明显。在静态 schedule 中，剩余 work 可能分布不均。有些 SM 可能已经没有分配到的 tile，而另一些 SM 仍然还有几个 tile 要做。使用 CLC 时，提前完成的 cluster 会请求另一个 pending cluster coordinate。只要 launch queue 中还有 work，提前完成者就能继续拉取更多 tile。

当 tile 开销不均匀时，这也很重要。有些 GEMM tile 可能因为边界、masking、sparsity、grouped scheduling，或者主矩阵乘法周围的 fused work，而走不同路径。静态 schedule 假设 tile 分配在任何成本被观察到之前就足够好。CLC 不需要这个假设。它只在 cluster 变得可用之后才分配更多 work。

因此，在 TIRx 中，CLC 可以暴露成一个动态 tile scheduler。编程模型不需要改变单个 tile 的计算。Tile body 仍然是静态 scheduler 使用的同一个 persistent GEMM body。Scheduler 从“用公式计算我的下一个 tile coordinate”变成“向硬件请求下一个可用 cluster coordinate”。结果仍然是同一个 persistent loop，只是 work distribution 从固定的 launch-time schedule 变成硬件驱动。
