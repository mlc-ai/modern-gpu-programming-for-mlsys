(chap_tensor_cores)=
# Tensor Core：`tcgen05`

:::{admonition} 概览
:class: overview

- `tcgen05` 是 Blackwell 的 Tensor Core 指令族。它的 MMA 指令会协作执行 tile matrix-multiply-accumulate 工作，并由一个被选中的线程 commit 指令。
- Accumulator 位于 TMEM，而不是寄存器。Epilogue 随后用 `tcgen05.ld` 把它带回寄存器。
- `cta_group::1` 和 `cta_group::2` 控制一个 CTA 还是两个 CTA 协作执行 MMA。这个选择也会改变 M 维度映射到 TMEM 的方式。
- Block-scaled MMA mode，例如 `mxfp8` 和 `nvfp4`，会添加 scale-factor operand。数据 operand 位于 SMEM，而 scale factor 会通过 TMEM staged。
:::

Dense linear algebra 是现代 GPU 花费最多有效工作的地方。普通 CUDA core 矩阵乘法无法接近芯片标称峰值（{ref}`chap_background`）。快速 GEMM 和 attention kernel 通过给 Tensor Core 喂入正确的 tile shape、layout 和 synchronization，才能达到这个峰值。

基本操作从 Volta 以来在精神上没有改变。Tensor Core 消费矩阵 tile，相乘，并把结果累加起来。每一代变化的是操作如何发起、操作数如何布局，以及 accumulator 位于哪里。

Blackwell 对最后一部分做了很大改变。`tcgen05` 的 accumulator 不再作为长生命周期 register fragment 保存。它会写入 Tensor Memory，也就是 TMEM（{ref}`chap_tmem`）。这一个变化会影响整个 kernel。MMA 写入 TMEM。完成状态被异步跟踪。Epilogue 随后从 TMEM 中加载 accumulator，并把它重新变成用于转换和 store 的 register fragment。

本章聚焦 compute instruction 本身。TMA（{ref}`chap_tma`）负责把 operand 移入 SMEM。TMEM 负责保存 accumulator 和某些 scale-factor operand。`tcgen05.mma` 是位于这两类数据移动之间的 Tensor Core 操作。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo_zh/tcgen05_intro.html" title="tcgen05 and Tensor Memory" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互图：`tcgen05` accumulator 行为。切换 A 或 B 的 transpose，选择输出宽度 `N`，并逐步执行 `K` iteration，观察 partial sum 如何在 TMEM 中累加。*

## `tcgen05` MMA

`tcgen05` MMA 是 Blackwell Tensor Core 的 matrix-multiply-accumulate 指令。它是一条协作指令。工作由一个 warpgroup 执行，并且在某些 mode 中可以涉及同一 cluster 中的两个 CTA。指令不是由每个线程独立发起的。一个被选中的线程会代表参与的 group commit 这个操作。

把 MMA 拆成三个问题来看会更清楚。

第一个问题是谁参与协作。普通 mode 使用一个 CTA，写作 `cta_group::1`。更大的 mode 使用 cluster 中的两个 CTA，写作 `cta_group::2`。在两种情况下，这条指令都表示一个 tile 上的一次 Tensor Core 操作，而不是某个线程执行的 scalar operation。

第二个问题是 operand 和 result 位于哪里。数据 operand 通常位于 SMEM。某些变体也可以从 TMEM 读取 A operand。Accumulator 写入 TMEM。Operand layout 必须匹配 Tensor Core 的期望，包括数据 operand 使用的 swizzled shared-memory layout（{ref}`chap_data_layout`）。

第三个问题是如何观察完成。`tcgen05.mma` 是异步的。发起 MMA 并不表示 multiply-accumulate 已完成。指令在操作 commit 后返回，而 Tensor Core 会继续运行。Kernel 使用 commit group 和 `mbarrier` 来知道结果何时就绪（{ref}`chap_async_barriers`）。

这种异步行为让 overlap 成为可能。快速 kernel 不会发起 MMA 后立即停下来等待它完成。它可以发起 MMA，开始准备后续 tile，并且只在真正需要结果时等待。代价是每个交接都必须显式表达。如果 epilogue 在 MMA completion barrier 触发前读取 TMEM，那就是读得太早。

## Accumulator 位于 TMEM

在 Ampere 和 Hopper 上，accumulator 以寄存器形式暴露给程序。MMA 产生一个 per-lane register fragment，epilogue 直接消费这个 fragment。这很简单，但它把 accumulator 大小绑定到了每个线程的寄存器预算上。

Blackwell 切断了这条绑定。`tcgen05.mma` 会把 accumulator 写入 TMEM，这是 Blackwell 上作用域属于 CTA 的内存空间。Accumulator 可以在 compute phase 中保存在 TMEM 中，epilogue 随后用 `tcgen05.ld` 把它加载回寄存器。

这改变了 kernel 的形状。Register fragment 在边界上仍然重要。Epilogue 仍然需要寄存器，以便转换、执行 elementwise 工作并 store 结果。但长生命周期 accumulator state 不再是寄存器分配问题，而是 TMEM 分配和布局问题（{ref}`chap_tmem`）。

这就是为什么必须把 `tcgen05` 和 TMEM 放在一起理解。MMA 指令决定计算哪个 tile。TMEM 决定 accumulator 落在哪里。Epilogue 必须使用匹配的 load path，把 accumulator 恢复到它期望的 register layout。

## `cta_group::1` 和 `cta_group::2`

`tcgen05` MMA 可以在 `cta_group::1` 或 `cta_group::2` mode 中运行。

在 `cta_group::1` 中，一个 CTA 拥有这次 MMA。它的 operand 位于该 CTA 的 SMEM 中，accumulator 写入该 CTA 的 TMEM。

在 `cta_group::2` 中，cluster 中的两个 CTA 协作执行一个 MMA tile。每个 CTA 都有自己的 SMEM 和自己的 TMEM。Accumulator 并不是存储在跨两个 CTA 的单个物理 TMEM 区域中。它会在两个 CTA 之间切分，每个 CTA 保存自己的部分。偶数 CTA 负责发起指令，并为这对 CTA commit completion barrier。

这个选择很重要，因为它会改变逻辑 accumulator tile `C(M, N)` 到 TMEM 的映射。TMEM 有 128 个硬件 Lane 行和最多 512 个硬件 Col 列。在 TIRx 布局记号中，这些轴写作 `TLane` 和 `TCol`。MMA mode 决定 C 的行和列如何放到这些 TMEM 轴上。

有四种有用情况需要记住。

下面的图沿用 demo 中的颜色约定：紫色表示 SMEM operand，橙色表示 TMEM accumulator state，绿色表示 Tensor Core MMA 路径。CTA identity 通过标签和位置表示，而不是改变这些硬件颜色。

### `cta_group::1`，`M = 128`

这是最简单的情况。一个 CTA 计算一个 128-row tile。TMEM 也有 128 个 Lane 行。因此映射是直接的：accumulator 的 row `m` 映射到 Lane `m`，N 维度映射到 TMEM column。

结果填满 128 个 Lane 行乘以 N 个 Col 列。这是基线图。CTA 在 SMEM 中拥有 A 和 B，并在自己的 TMEM 中拥有完整 accumulator tile。

![cta_group::1, M=128: row m maps directly to TMEM Lane m](../../img/mma_cg1_m128.svg)

### `cta_group::1`，`M = 64`

当 `M = 64` 时，accumulator 只有 64 行，但 TMEM 仍然有 128 个 Lane 行。硬件并不是简单地把 row 0 到 63 打包到 lane 0 到 63。相反，它会把这些行分散到四段 16-row run 中。

Rows 0 到 15 放到 lanes 0 到 15。Rows 16 到 31 放到 lanes 32 到 47。Rows 32 到 47 放到 lanes 64 到 79。Rows 48 到 63 放到 lanes 96 到 111。

这会在 lanes 16 到 31、48 到 63、80 到 95、112 到 127 留出空隙。这些空隙是有意的。使用不同 lane alignment 时，另一个独立的 `M = 64` MMA 可以占用互补 lane。这样两个较小的 M tile 可以共享 128-lane TMEM 结构，而不会互相覆盖。

N 维度仍然映射到 TMEM column。不寻常的地方只有 M row 在 Lane 上的 placement。

![cta_group::1, M=64: four 16-row runs at a Lane stride of 32, leaving space for another aligned M=64 tile](../../img/mma_cg1_m64.svg)

### `cta_group::2`，`M = 256`

当 M 维度大到单个 CTA 自然容纳不下时，MMA 可以使用 `cta_group::2`。对于 `M = 256`，切分是直接的。CTA 0 持有 rows 0 到 127。CTA 1 持有 rows 128 到 255。

每个 CTA 使用自己的 TMEM Lane rows 0 到 127，以及完整 N columns。物理上，这是两个独立的 128-row TMEM 区域，每个 CTA 一个。逻辑上，它们组成一个 256 by N 的 accumulator tile。

每个 CTA 也提供 A 中对应自己 M row 的部分。B 会按照 mode 的要求对两个 CTA 可用。偶数 CTA 负责发起 MMA，并为这对 CTA commit completion barrier。

这是 {ref}`chap_gemm_advanced` 中 two-CTA cluster GEMM 使用的 mode。

![cta_group::2, M=256: M split contiguously across two CTAs, 128 rows per CTA](../../img/mma_cg2_m256.svg)

### `cta_group::2`，`M = 128`

`cta_group::2`, `M = 128` mode 仍然使用两个 CTA，但 M 维度更短。因为总共只有 128 行，每个 CTA 得到 64 个 M row。

剩余 lane capacity 被用来打包 N 维度。在每个 CTA 内部，N 的一半占用 lanes 0 到 63，另一半占用 lanes 64 到 127。这样即使每个 CTA 只拥有 64 个 M row，也能使用全部 128 个 Lane 行。

因此这个 split 有两部分。M 在 CTA pair 之间切分，每个 CTA 64 行。N 随后在每个 CTA 内部切分到 TMEM Lane 行的 lower half 和 upper half。

![cta_group::2, M=128: 64 M rows per CTA, with the two halves of N stacked across the lower and upper Lane halves](../../img/mma_cg2_m128.svg)

在这些 mode 中，原则相同。`tcgen05.mma` 计算一个逻辑 accumulator tile，但这个 tile 必须放入物理的 128 Lane by up to 512 Col TMEM 空间中。Mode 和 M shape 决定这种 placement。Kernel 的其他部分稍后读取 accumulator 时，必须使用同一映射。

对于这里的 kernel，accumulator 通常在 TMEM 中使用 f32。这是常见的高精度路径。它不是唯一可能的 accumulator type。`.kind::f16` 路径可以用 f16 accumulation。

## Operand Placement

对于 dense MMA mode，A 和 B 在 MMA 运行前准备在 SMEM 中。TMA 负责把 global memory tile 移入 SMEM。Kernel 把这些 SMEM tile 安排成 Tensor Core 期望的布局，包括任何必要的 swizzle。

Accumulator C 写入 TMEM。这是相对早期世代的主要区别。Epilogue 不会直接收到 MMA 指令输出的 accumulator。它必须用 `tcgen05.ld` 显式从 TMEM 加载。

在 `cta_group::1` 中，一个 CTA 提供 operand 并拥有 accumulator。在 `cta_group::2` 中，每个 CTA 从自己的 SMEM 中提供自己那一侧的 operand，并拥有 accumulator 中自己的 TMEM 部分。当 A 按 M 切分时，每个 CTA 保留自己 M slice 对应的 A row。B 根据 mode 被共享，因为两个 M slice 都要乘同一个 N by K tile。

阅读 kernel 时，这种分离很重要。SMEM placement 回答 Tensor Core 如何读取 A 和 B。TMEM placement 回答 accumulator 去哪里。这两个 layout 由 MMA mode 关联起来，但它们不是同一个内存空间，不能互换对待。

## Block-Scaled MMA

Dense mode 直接从 SMEM 读取数据 operand，并累加到 TMEM。Block-scaled MMA 添加了两个额外 operand：A 和 B 的 scale-factor tensor。

这用于 `mxfp8` 和 `nvfp4` 等非常低精度格式。低精度格式效率高，但动态范围小。单个 global scale 通常太粗。如果 scale 按最大值选择，小值会丢精度。如果 scale 按小值选择，大值可能溢出或被截断。

Block scaling 通过给较小的 K block 分配 scale factor 来解决这个问题。一组连续 K element 共享一个 scale。MMA 在概念上先用 scale 对每个 block dequantize，再把乘积累加到 accumulator type 中。

对于 A 和 B，这引入两个 scale-factor tensor：

```text
SFA(M, SFK)
SFB(N, SFK)
```

其中 `SFK = K / B`，`B` 是沿 K 的 block size。

具体 block size 取决于格式。重要的是，scale axis 以更粗粒度跟随 K。每个 scale factor 描述一块 K value，而不是单个元素，也不是整张矩阵。

数学形状是：

```text
acc += (Aq * scale_a) * (Bq * scale_b)
```

其中 `Aq` 和 `Bq` 是量化的低精度值，scale 在 accumulation 前恢复它们的近似 magnitude。

Scale dtype 也很重要。使用 `e8m0` scale 时，每个 scale 实际上是 2 的幂。使用 `e4m3` scale 时，例如 `nvfp4` 中的 scale，它是一个小浮点值，可以表示两个 2 的幂之间的数值。

## Scale Factor 放在哪里

Block-scaled `tcgen05.mma` 与 dense MMA 有一个重要 placement 规则不同：scale factor 从 TMEM 读取。

数据 operand A 和 B 仍然 staged 在 SMEM 中。Scale factor SFA 和 SFB 通过 TMEM staged。由于 TMA 加载到 SMEM，scale factor 通常需要额外一步。Kernel 先把它们加载到 SMEM，再用 `tcgen05.cp` 从 SMEM copy 到 TMEM。只有 scale factor 位于 TMEM 后，block-scaled MMA 才能读取它们。

这给 scale factor 一条不同于数据 operand 的移动路径：

```text
A, B:     global memory to SMEM, then MMA reads SMEM
SFA, SFB: global memory to SMEM, then tcgen05.cp copies SMEM to TMEM, then MMA reads TMEM
```

Scale factor 的 TMEM layout 是紧凑的。一个 128-row scale vector 可以打包到 32 个 Lane row 中，lane 位置基于 `r % 32` 映射，`r / 32` 沿 column 方向前进。随后数据可以 broadcast 到读取完整 128 Lane 空间的四个 warp 上（{ref}`chap_layout_generations`）。

这很好地说明为什么 TMEM layout 必须显式。Accumulator layout 和 scale-factor layout 都在 TMEM 中，但它们不是同一个 layout。Accumulator 使用 MMA output mapping。Scale factor 使用 block-scaled MMA 期望的紧凑布局。

## `cta_group::2` 中的 Scale Factor

在 two-CTA 情况下，scale factor 跟随它们缩放的数据。

SFA 缩放 A。由于 A 在 CTA pair 之间按 M 切分，SFA 也按 M 切分。每个 CTA 持有与自己 A row 对应的 SFA row。

SFB 缩放 B。由于两个 CTA 都要乘同一个 B tile，SFB 必须对两个 CTA 可见。实践中，这意味着 SFB 会 multicast 到 CTA pair。

这是 block-scaled cluster GEMM 中常见 loading pattern 的来源。SFA 按 CTA 加载，使用该 CTA 自己 M slice 的 mask。SFB broadcast 给这对 CTA，因为两个 CTA 都需要同一组 N-side scale factor。

![Block-scaled MMA placement: A and B packed in SMEM; SFA, SFB, and C in TMEM, with SFA split by M across CTAs and SFB multicast across the CTA pair](../../img/mma_block_scaled.svg)

## 保持 MMA Contract 匹配

一个 Blackwell GEMM tile 会经过多条专用路径。

TMA 把 A 和 B 从 global memory 带入 SMEM。对于 block-scaled mode，它也会把 scale factor 带入 SMEM。需要时，`tcgen05.cp` 把这些 scale factor 移到 TMEM。`tcgen05.mma` 读取 operand，在 Tensor Core 上异步运行，并累加到 TMEM。Completion barrier 告诉 kernel accumulator 何时就绪。Epilogue 随后用 `tcgen05.ld` 从 TMEM 把 accumulator 加载回寄存器，并 store 最终输出。

在这些路径之间，kernel 必须保持三份 contract 匹配：SMEM operand layout、TMEM accumulator 或 scale-factor layout，以及让下一个 consumer 安全运行的异步完成信号。
