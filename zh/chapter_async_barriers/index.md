(chap_async_barriers)=
# 异步协作：mbarrier

:::{admonition} 概览
:class: overview

- TMA 和 Tensor Core 都是异步的，所以“发起工作”和“工作完成”不是一回事；消费者需要显式的完成信号。
- `mbarrier` 就是这个信号：producer arrive，consumer wait，并且它会跟踪 arrival count 以及 TMA 使用的 byte count。
- 每个 barrier 都带有一个 *phase*，每完成一轮就翻转一次；consumer 等待正确的 phase，才能安全地越过这个同步点。
:::

TMA（{ref}`chap_tma`）和 Tensor Core（{ref}`chap_tensor_cores`）操作都是异步的。当 kernel 发起 TMA load 或 `tcgen05` MMA 时，发起线程不会等待操作完成。指令只是被提交给硬件引擎；真正的数据搬运或矩阵运算会继续与程序的其他部分并行执行。

这很有用，因为它允许内存搬运和计算重叠。但这也意味着，单靠程序顺序无法证明数据已经准备好。后面的指令可能会在前面的异步操作完成之前运行。如果 TMA 仍在写 shared-memory tile 时 MMA 已经开始读取它，MMA 就会读到不完整的数据。如果 epilogue 在 Tensor Core 写完 accumulator 之前读取 TMEM，它会读到错误值。如果 kernel 等错了条件，它甚至可能永远无法继续前进。

因此，kernel 在每一个异步交接点都需要显式完成信号。`mbarrier` 就是这个信号。Producer 在工作完成时 arrive 到 barrier，consumer 在使用产物之前 wait 这个 barrier。同一个机制可以用于 TMA 到 MMA 的交接、MMA 到 epilogue 的交接，以及 pipeline stage 中的 buffer 复用。

Barrier 不只是一次性的 flag。它带有一个 phase bit，并且每当 barrier 完成一轮 arrival，这个 phase bit 都会改变。Phase 让同一个 barrier 可以在很多 loop iteration 中复用，而不会把一次 iteration 的完成误认为另一次 iteration 的完成。

## mbarrier

`mbarrier` 是 memory barrier 的缩写，是存放在 shared memory 中的硬件同步对象。概念上，它包含两部分状态：arrival counter 和 phase bit。Counter 告诉 barrier 当前轮还缺多少次 arrival；phase bit 告诉 kernel 当前 barrier 处于哪一轮。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo_zh/mbarrier_mechanism.html" title="mbarrier data structure and APIs" loading="lazy"
        style="width:100%; min-width:1320px; height:620px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互图：`mbarrier` 的状态视图，展示 arrival counter、phase bit，以及 `init`、`arrive` 和 `wait` 操作；点击字段可以聚焦查看。*

Barrier 从初始化开始。在 `init` 期间，kernel 设置这个 barrier 期望收到多少次 arrival。Barrier 从 phase 0 开始，并把 counter 设置为期望的 arrival count。从这之后，barrier 会等待所有需要的 producer 或资源使用者报告自己已经完成。

一次 arrival 会减少 barrier 仍在等待的工作量。Kernel 的不同部分可以用不同方式 arrive 到 barrier，这些差异很重要。

对于 TMA load，常见的 arrival 路径是 tx-count arrival。像 `mbarrier.arrive.expect_tx(bytes)` 这样的操作会做两件事。第一，它算作发起线程在 barrier 上的一次 arrival。第二，它记录 TMA 引擎预计要传输的 byte 数。Barrier 不会因为发起线程已经 arrive 就完成。它还会等待 TMA 引擎在传输完成时把 byte count 清零。只有两个条件都满足时 phase 才会翻转：普通 arrival count 到达 0，并且 pending tx byte count 也到达 0。

因此，不应该把 `expect_tx` 理解成“多一次普通 arrival”。它是在为异步 copy 设置一个 byte 预算。硬件随后通过 complete-tx 更新来记录真实 copy 的完成情况。只有 arrival 和 byte transfer 都完成之后，barrier 才完成。

对于 Tensor Core 工作，arrival 路径不同。`tcgen05` MMA 不会因为 MMA 被发起就自动推进一个 barrier。Kernel 必须显式把 barrier arrival 绑定到 commit 路径上，例如通过 `tcgen05.commit.mbarrier::arrive` 操作。当被 commit 的 group 完成时，Tensor Core 侧会执行 barrier arrival。如果 kernel 忘了这个 commit arrival，等待 barrier 的 consumer 就会永远等待下去。

普通线程也可以直接 arrive 到 barrier。当普通线程代码是 producer，或者一组线程要宣布它已经使用完某个资源时，就会用到这种方式。例如，consumer 读完一个 shared-memory buffer 后，可以 arrive 到一个 barrier，用来告诉 producer 这个 buffer 可以复用了。

等待是同一个协议的 consumer 侧。Consumer 会等待 barrier 完成当前 iteration 所期望的 phase。只有这样，读取数据或复用这个 barrier 保护的资源才是安全的。

关键点是：异步硬件不仅会跑在程序前面，它还会通过 barrier 把完成事件报告回来。TMA 可以发出 shared-memory tile 已经就绪的信号。Tensor Core 工作可以发出 TMEM 结果已经就绪的信号。普通线程可以发出某个 buffer 不再被使用的信号。Barrier 把这些情况统一成 producer-consumer 形状：producer arrive，consumer wait。

## Phase Tracking

Barrier 通常不会只分配给一次使用。一个 pipelined K-loop 可能会执行同一个交接数百次；如果每次 iteration 都分配一个新的 shared-memory barrier，并不现实。相反，kernel 会保留一小组固定 barrier，并在 loop 前进时复用它们。

Phase bit 让这种复用变得安全。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo_zh/phase_tracking.html" title="mbarrier phase tracking" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互图：多个 pipeline iteration 复用同一个 barrier，展示每一轮完成后 phase bit 如何翻转。*

每当 barrier 完成当前轮的全部 arrival，它都会翻转 phase：phase 0 变成 phase 1，phase 1 变成 phase 0，如此循环。Wait 操作会检查 consumer 期望的 phase。这个期望 phase 由 kernel 保存在寄存器中。某个 stage 成功等待一轮之后，kernel 会在下一轮使用这个 barrier 前翻转自己的本地 phase 值。

这可以防止 kernel 把旧完成误认为新完成。假设一个 barrier 被用于某次 TMA load，并且已经完成。如果下一次 loop iteration 复用同一个 barrier 但没有跟踪 phase，那么 consumer 可能看到前一次完成，并错误地认为新的 load 已经就绪。Phase bit 把这两轮区分开。Iteration 0 等待一个 phase，iteration 1 等待相反的 phase，iteration 2 再等待第一个 phase，这个模式持续下去。

在真实 pipeline 中，记录通常按 stage 进行。Kernel 有固定数量的 shared-memory stage、匹配数量的 barrier，以及一小组保存在寄存器中的 phase 值。Loop 前进时，每个逻辑 iteration 映射到一个物理 stage，而 phase 值告诉 wait 操作它正在等待这个物理 barrier 的哪一轮。

这就是为什么后面的 GEMM 代码不需要每个 K tile 一个 barrier（{ref}`chap_gemm_async`）。它需要每个可复用 stage 一个 barrier，再加上 phase tracking。Stage index 选择 shared-memory buffer 和 barrier。Phase 值把当前对这个 stage 的使用与上一次使用区分开。

**Try with your agent**：给它一个 two-stage pipeline，并让它 trace 四次 iteration。对每次 iteration，列出 stage index、本地 phase 值、barrier 何时翻转，以及如果 stage 复用前没有翻转 phase 会出什么问题。

## 同步规则

理解 barrier 和 phase 机制之后，tensor-core kernel 中的同步模式就相当机械了。每当一条路径产生数据，或者释放另一条路径将要消费的资源时，这个交接都必须显式表达。

常见情况有三类。

第一类是线程代码为异步引擎生产数据。如果线程写 shared memory，而后续 TMA store 或 MMA 指令要读这个 shared memory，那么 kernel 必须先让线程写入对引擎可见。这需要合适的 thread-level synchronization 或 fence。具体指令取决于交接 scope，但原因始终相同：engine 不能在 producing threads 写完之前观察这个 shared-memory buffer。

第二类是 TMA 为 MMA 生产数据。TMA load 会异步填充一个 shared-memory tile。MMA 路径不能因为 TMA 指令已经发起就推断 tile 已经就绪。TMA 操作必须关联一个 `mbarrier`，MMA 路径必须在读取 tile 之前等待这个 barrier。

第三类是 MMA 为 epilogue 生产数据。`tcgen05` MMA 会异步地把结果写入 TMEM。Epilogue 不能在 Tensor Core 完成相关工作之前安全读取 accumulator。因此，MMA commit 路径会 arrive 到一个 completion barrier，而 epilogue 在读取 TMEM 之前等待这个 barrier。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo_zh/mbarrier_tma_timeline.html" title="mbarrier signalling TMA completion" loading="lazy"
        style="width:100%; min-width:1320px; height:700px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互图：TMA load 通过 `mbarrier` 发出完成信号。MMA 路径在读取 shared-memory tile 之前等待 barrier。Tensor Core 到 epilogue 的交接也是同一个形状，只是 arrival 来自 Tensor Core commit 路径，而不是 TMA。*

同一个思想也适用于资源复用。Barrier 不只是 data-ready 信号，也可以是“资源已经空闲”的信号。一个 shared-memory stage 不能在旧 tile 的所有 consumer 都完成之前被覆盖。一个 TMEM 区域不能在上一个使用者读写完成之前被复用。在这些情况下，arrival 表示“我已经用完这个资源”，wait 表示“现在可以安全地为下一个 stage 复用这个资源”。

这也是阅读 pipelined GEMM kernel 中同步逻辑的正确方式。Wait 和 arrive 并不是作为防御性编程到处散落。每一个都标记了一次具体的 ownership transfer：tile 变得可读、accumulator 变得可读，或者 buffer 变得可复用。一旦识别出这些交接，控制流就会容易跟踪得多。
