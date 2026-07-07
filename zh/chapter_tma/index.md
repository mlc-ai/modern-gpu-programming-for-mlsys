(chap_tma)=
# 异步数据搬运：TMA

:::{admonition} 概览
:class: overview

- TMA 是在 global memory 和 shared memory 之间异步拷贝 tile 的硬件引擎。一个线程发起 copy，硬件引擎移动 byte。
- 一次 TMA copy 由 tensor-map descriptor 描述。Descriptor 告诉引擎 global tensor shape、stride、tile coordinate，以及 shared-memory swizzle mode。
- 在 load 路径上，TMA 可以在写 shared memory 时对 tile 应用 swizzle，使 tile 直接落到 Tensor Core 期望的布局中。
- TMA load 通过带 byte-count tracking 的 `mbarrier` 完成。TMA store 使用 commit group 和 wait group。
:::

只有当数据已经准备好供 Tensor Core 消费时，Tensor Core 才有用。在 GEMM 或 attention kernel 中，一旦 pipeline 被填满，数学部分可能是 compute-bound（{ref}`chap_performance`），但只有下一块 operand tile 及时到达，pipeline 才能一直保持填满。

移动 tile 的旧方式是让线程自己 copy。每个线程计算地址，从 global memory 发起 load，再把值 store 到 shared memory。这样当然可行，但它把 warp 指令花在地址计算和 copy bookkeeping 上，而不是计算上。它也让 copy 路径暴露在同一批本应喂给 Tensor Core 的 warp 的指令流中。

Tensor Memory Accelerator，简称 TMA，会把这项工作移动到硬件 copy engine 中。一个线程发起一次 tile copy，copy engine 随后在 global memory 和 shared memory 之间异步移动一个矩形 tile。当引擎移动 byte 时，CTA 的其他部分可以继续执行别的工作。

TMA 还会处理一部分布局问题。Tensor Core 不只是需要 shared memory 中有正确的值。它还需要这些值位于正确的 shared-memory layout 中。在 load 路径上，TMA 可以在写 tile 时应用 shared-memory swizzle。这样 tile 会直接落到后续 MMA 期望的布局中。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo_zh/tma_intro.html" title="TMA: the Tensor Memory Accelerator" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互图：TMA 把 tile 从 global memory 拷贝到 shared memory。切换 swizzle mode，并悬停在 source cell 上查看它在 shared memory 中落到哪里。*

## 一个线程发起，硬件搬运 Tile

一次 TMA copy 从一个 issuing thread 开始。这个线程不会遍历 tile 中的所有元素。它会把 copy 的描述交给硬件，然后由 TMA engine 执行整个 transfer。

主要输入是 tensor-map descriptor。Descriptor 描述 global tensor，以及应该如何从中读取一个 tile。它会记录 tensor shape、stride、element size、tile shape 和 swizzle mode 等信息。Issuing thread 还会提供 tile 应该落到的 shared-memory address。

指令发出之后，copy 会异步运行。Issuing thread 可以继续执行。CTA 中的其他线程也可以继续执行。Transfer 现在由 TMA engine 负责，而不是由普通 load/store 指令组成的 loop 负责。

这让 kernel 有两种不同方式表达同一个逻辑操作：“copy 这个 tile”。

一条路径是 thread copy。线程协作从 global memory load，并 store 到 shared memory。这样 kernel 可以直接控制每一次访问，但会消耗线程指令和寄存器来做地址计算。

另一条路径是 TMA copy。一个线程发起 transfer，硬件 copy engine 执行矩形 copy。对于大的规则 tile，尤其是 Tensor Core kernel 使用的 operand tile，这是自然路径。

这两条路径有不同的同步规则和不同的性能行为。选择其中哪一条是一个 dispatch decision。Layout 告诉 kernel 想要什么内存排列。Scope 告诉它哪些线程或 CTA 参与。Dispatch 决定这次 copy 是由普通 thread copy 实现，还是由 TMA 实现。

## Swizzled Layout

移动 tile 本身还不够。Tile 还必须以 Tensor Core 能高效读取的布局放入 shared memory。

这就是 TMA swizzling 的用途。当 TMA 把 tile 写入 shared memory 时，它可以重排 shared-memory address pattern。Global memory tile 仍然是一个逻辑矩形，但 shared memory 中的 destination layout 可以是 swizzled 的。

Swizzle mode 是 TMA descriptor 的一部分。Descriptor 设置好之后，issuing thread 不需要手工应用 swizzle。Engine 会在 byte 落入 shared memory 时应用它。

重要要求是一致性。TMA descriptor、shared-memory tile layout，以及后续 MMA 指令，都必须描述同一个布局（{ref}`chap_data_layout`）。如果 TMA 用一种 swizzle 写入 tile，而 MMA 以为它是另一种 swizzle，硬件仍然会忠实执行收到的指令。只是这些 byte 对计算来说会排列错误。

这正是布局记号不再只是 bookkeeping 的地方。DSL 使用的布局必须匹配 TMA descriptor 和 Tensor Core 指令使用的硬件布局。例如，如果 kernel 说某个 operand tile 存储在 128-byte swizzled layout 中，TMA descriptor 就必须使用匹配的 swizzle mode，MMA dispatch 也必须期望同样的 shared-memory arrangement。上面的 demo 可以在 no swizzle 和 128-byte swizzle 之间切换；悬停在 source element 上可以看到 swizzle 应用后它落在哪里。

理解 swizzle 的一个有用方式是：TMA 并没有改变逻辑 tile。它改变的是逻辑元素在 shared memory 中的物理落点。后续 MMA 仍然消费同一个逻辑 A 或 B tile。Swizzle 只决定这个 tile 如何排列在 shared memory bank 上。

## 用 3D TMA 表达 Tiling 和 Swizzling

普通 TMA copy 会移动一个扁平 2D tile，但 Tensor Core 期望的 shared-memory layout 通常会被 *tiled* 成 swizzle atom（来自 {ref}`chap_data_layout` 的 8 x 128-byte atom）。TMA 通过额外的 descriptor dimension 处理这一点。**3D TMA** 把 shared-memory box 描述成 `(group, row, col)`，其中 group 维度沿 atom 前进，内部两个维度在一个 atom 内寻址。一次 3D copy 随后既会 atom by atom 地布置 tile（tiling），又会在每个 atom 内应用 swizzle，因此数据到达时就已经在 MMA 期望的布局中，不需要单独的 tiling 或 swizzling pass。

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../demo_zh/tma_3d.html" title="Tiling and swizzling with 3D TMA" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互图：一次 3D TMA copy，以 (group, row, col) 寻址，并 tiled 到 swizzled shared memory 中。*

选择 swizzle *format* 与这种 tiling 绑定在一起。更宽的 swizzle 会把一个 column 分散到更多 bank，所以只要能适用，128-byte swizzle 就是默认选择。但一个 N-byte atom 要求 tile 的 contiguous dimension 能填满它。因此，一个由于 shape constraint 而变小的 tile 不能使用 128-byte swizzle，必须降到 64-byte 或 32-byte：经验法则是选择 tile 能填满的最大 swizzle（{ref}`chap_data_layout`）。下面的 demo 直接展示这个约束：16 x 16 tile 上的 128-byte swizzle 只有在 tile 被切成匹配 atom 的 16 x 8 group 之后，才会变成 conflict-free。

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../demo_zh/tiling_constraint.html" title="Swizzle imposes a tiling constraint" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
<script>
(function () {
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    document.querySelectorAll('iframe.demo-tma3d').forEach(function (f) {
      if (e.source === f.contentWindow) f.style.height = d.height + 'px';
    });
  });
})();
</script>
```
*交互图：16 x 16 tile 上的 128-byte swizzle；切成 16 x 8 group 后变成 conflict-free。*

## 完成通知：Load

Copy 是异步的，所以发起它还不够。Consumer 不能因为 TMA 指令已经发出就读取 shared-memory tile。只有在引擎写完 byte 之后，读取 tile 才是安全的。

对于 TMA load，完成信号是 `mbarrier`（{ref}`chap_async_barriers`）。

常见顺序是：

1. 为 pipeline stage 初始化或复用一个 `mbarrier`；
2. 告诉 barrier 这次 TMA transfer 预计会写入多少 byte；
3. 发起 TMA load；
4. 让 TMA engine 在 byte 到达时更新 barrier；
5. consumer 在读取 shared-memory tile 之前等待这个 barrier phase。

Byte count 通过如下操作设置：

```text
mbarrier.arrive.expect_tx(bytes)
```

它做两件事。第一，记录期望 transfer size。第二，它也执行 issuing thread 在 barrier 上的 arrival。Barrier 不会因为这个调用发生了就完成。它仍然等待 TMA engine 报告期望 byte 已经到达。

Transfer 进行时，引擎会对 barrier 执行 complete-tx 更新。只有两个条件都满足时，barrier phase 才会翻转：arrival count 已经满足，并且 pending byte count 到达 0。

Consumer 随后等待这个 barrier。对期望 phase 的 wait 完成后，shared-memory tile 就准备好了。此时 MMA 路径可以安全读取它。

![TMA load synchronization flow](../../img/tma_sync_flow.png)

这与其他异步 producer-consumer 交接使用的是同一个 barrier 模型。Producer 是 TMA engine。Consumer 是 MMA 路径，或者任何其他读取 shared-memory tile 的代码。Barrier 是它们之间的显式交接。

## 完成通知：Store

TMA store 沿相反方向移动数据，从 shared memory 到 global memory。它们也是异步的，但完成机制不同。

TMA load 通常喂给同一个 kernel 内部的 consumer。MMA 路径需要知道 shared-memory tile 何时就绪。因此 load 路径使用 `mbarrier`。

TMA store 通常把最终数据写出到 global memory。通常没有立即在 kernel 内等待存储结果的 consumer。Kernel 主要需要知道的是，什么时候可以复用 shared-memory buffer，或者结束 store 序列。

为此，TMA store 使用 commit group 和 wait group。Kernel 发起一次或多次 store，commit 这个 group，随后等待这个 group drain。Wait 完成后，从 kernel 视角看，这个 group 中的 store 已经完成，store 使用的 shared-memory 区域可以安全复用。

规则很简单：

```text
TMA load:  wait through an mbarrier with byte-count tracking
TMA store: wait through a commit group and wait group
```

这两个机制在不同交接点服务于同一个目的。Load 需要让 shared-memory tile 对后续 consumer 可见。Store 需要确保 outgoing transfer 在 kernel 复用 source storage 或依赖 store 已 drain 之前完成。

## 为什么 TMA 对 Pipelining 很重要

TMA 最有用的场景是作为 pipeline 的一部分。Kernel 可以在 Tensor Core 计算当前 tile 时，发起未来 tile 的 load。Load 在后台运行。Compute 在前台运行。当未来 tile 变成当前 tile 时，barrier 把两者连接起来。

典型 GEMM loop 会反复使用这个结构。Shared memory 的一个 stage 保存当前被 MMA 消费的 tile。另一个 stage 正在被 TMA 填充。Loop 前进时，这些角色轮换。在 MMA 读取某个 stage 之前，它会等待该 stage 的 load barrier。在 TMA 覆盖某个 stage 之前，kernel 会确保上一个 consumer 已经使用完它。

这就是为什么 TMA 和 `mbarrier` 通常一起出现在 Blackwell 和 Hopper 风格的 kernel 中。TMA 给 kernel 一个异步 copy engine。Barrier 给 kernel 一个精确方式来知道 copy 出来的 byte 何时准备好。
