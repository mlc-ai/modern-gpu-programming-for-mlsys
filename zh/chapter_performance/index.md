(chap_performance)=
# Kernel 性能从何而来

:::{admonition} 概览
:class: overview

- roofline 模型用内存带宽和计算吞吐给出 kernel 的性能上限，而算术强度决定当前受哪个上限限制。
- 低算术强度通常意味着 memory-bound，也就是性能主要受内存带宽限制。优化重点是减少 HBM 流量、提高复用、融合操作，并尽量接近内存带宽上限。
- 高算术强度通常意味着 compute-bound，也就是性能主要受计算吞吐限制。优化重点是让 Tensor Cores 保持忙碌，并通过重叠执行（overlap）减少计算路径上的空闲时间。

:::

一个 kernel 快不快，不能只看它的绝对吞吐数字，而要看它离硬件性能上限有多近。像 330 TFLOP/s 这样的数字单独看起来可能很大，但如果对应的 GPU 在 dense fp16 或 bf16 Tensor Core 计算上能够持续达到约 2 PFLOP/s，那么这个数字的意义就完全不同了。没有性能上限作为参照，我们就很难判断一个 kernel 是已经接近硬件极限，还是仍然让芯片的大部分算力处于空闲状态。

本章用 roofline 模型建立这个参照，并以 NVIDIA B200 为例进行分析。沿用 {ref}`chap_background` 中的约定，我们采用两个便于计算的近似值：dense fp16 或 bf16 Tensor Core 吞吐约为 2 PFLOP/s，HBM3e 带宽约为 8 TB/s。实际性能还会受到设备配置、时钟频率、功耗限制和测量环境等因素影响，因此这里的数字不是官方规格中的精确上限。


## Roofline 模型

从性能分析的角度看，一个 kernel 的执行时间主要来自两部分：数据搬运和计算。roofline 模型给出了一个简单的判断方式：kernel 的性能上限由计算吞吐上限和内存带宽上限共同决定，更具体地说，是由二者中更低的那个上限决定。

这里的计算吞吐上限，也就是“峰值计算吞吐”，指的是硬件在当前 kernel 所使用的计算路径上能够提供的最大 FLOP/s。对于 B200 上的 dense FP16/BF16 Tensor Core GEMM，这个上限通常来自 Tensor Core 吞吐；对于 scalar 或 elementwise kernel，这个上限则可能来自 CUDA Core、特殊函数单元，或者其他指令执行单元的吞吐。

内存带宽对应的性能上限可以用 HBM 带宽乘以算术强度来估算。如果一个 kernel 每搬运一个 byte 只做少量计算，它的性能通常会被 HBM 带宽限制；如果每个 byte 对应很多次计算，那么它更有机会进入 compute-bound 区域，性能上限也更可能由计算吞吐决定。


以 FLOP/s 为单位，基本的 roofline 性能上界是：

$$
\text{可达到的性能}
\le \min(\text{峰值计算吞吐}, \text{内存带宽} \times \text{算术强度})
$$

算术强度定义为：

$$
\text{算术强度}
= \frac{\text{计算量}}{\text{搬运的数据量}}
$$

这里的计算量指算法需要完成的数学运算量，通常用 FLOPs 表示，而不是 kernel 执行的指令总数。按照常用约定，一次浮点加法或乘法计为 1 FLOP，一次 fused multiply-add（融合乘加）`a * b + c` 计为 2 FLOPs。

对于 GEMM `C = A @ B`，如果 `A` 的形状是 `M × K`，`B` 的形状是 `K × N`，计算量通常记为：

$$
2 \times M \times N \times K
$$

这里的“搬运的数据量”也必须对应到具体的内存层级。对于 HBM roofline 模型，它指的是 kernel 对 HBM 产生的读写字节数；对于 L2 roofline 模型，它指的是经过 L2 的读写字节数；对于 SMEM roofline 模型，它指的是 shared memory 中的读写字节数。本章默认讨论的是 HBM roofline 模型。


在 roofline 图中，横轴表示**算术强度**，单位是 **FLOP/byte**；纵轴表示 kernel 能达到的性能。由内存带宽给出的性能上限是一条斜线：

$$
\text{性能} = \text{带宽} \times \text{算术强度}
$$
由计算吞吐给出的性能上限是一条水平线：

$$
\text{性能} = \text{峰值计算吞吐}
$$

这条水平线和前面的内存带宽上限线相交的位置称为**拐点（ridge point）**，也就是从 memory-bound（受内存限制）过渡到 compute-bound（受计算限制）的分界点：

$$
\text{拐点} = \frac{\text{峰值计算吞吐}}{\text{带宽}}
$$

用本章的 B200 近似数值代入，拐点的单位是 FLOP/byte：

$$
\text{拐点}
\approx \frac{2000}{8}
\approx 250
$$
在这个粗略模型中，一个 kernel 每从 HBM 搬运 1 byte 数据，需要完成大约 250 FLOPs，才有机会接近 Tensor Core 的计算吞吐上限。算术强度低于这个值时，kernel 就是 **memory-bound**：计算单元会因为 HBM 无法及时提供数据而等待。

Roofline 模型的价值在于判断当前是哪类资源限制了性能。对于 memory-bound kernel，减少少量计算指令通常没有帮助；对于 compute-bound kernel，少量访存优化也不会改变主要瓶颈。因此，优化前应先判断 kernel 位于拐点的哪一侧。


![B200 roofline 示例：图中展示内存上限、计算上限和拐点](../../img/roofline.png)

## 常见算子的算术强度

算术强度首先取决于算法的数据复用方式。因此，在编写 kernel 之前，通常就能对它的主要瓶颈作出初步判断。


### Elementwise 和 Reduction

像 GELU 这样的 elementwise kernel，以及 RMSNorm 这样的 reduction kernel，通常需要读写大量 tensor 数据，但每个元素上的计算量并不高。因此，它们的算术强度通常较低，在 roofline 图上位于拐点左侧，性能主要受内存带宽限制。


### GEMM

GEMM 则正好相反。它的算术强度会随着问题规模增大而提高，因为每个被加载进来的 tile 都可以被复用，用来执行许多次乘加操作。

对于一个方阵 fp16 matmul，假设 M = N = K，理想情况下的算术强度（AI）大约为：

$$
\mathrm{AI} \approx \frac{2N^3}{3 \cdot 2N^2}
= \frac{N}{3}
$$

单位是 FLOP/byte。这个估计假设 A 和 B 各只从 HBM 读取一次，C 只写入一次，并且 `beta = 0`，不需要读取原来的 C。它还假设片上复用是完美的，没有额外的 metadata、padding 或冗余访存。这里的 metadata 包括低精度格式使用的 scale 等辅助数据。真实 kernel 的数据搬运量通常更大，但这个估计仍可用于判断大致趋势。

### Attention

Attention 介于这两个极端之间。它的算术强度取决于序列长度、head dimension、tiling 方式、mask 方式，以及是否会实际生成中间张量。

标准 attention 的一个主要性能问题来自 score matrix，也就是由 `QK^T` 得到的 attention 分数矩阵。如果 kernel 先把它写入 HBM，随后又读回来，就会产生大量中间数据搬运。Flash Attention（包括 Flash Attention 4）把相关 tile 保留在片上，避免这次 HBM 往返，从而提高算术强度。

因此，attention 优化包含两个层面：算法上减少 HBM 流量，提高算术强度；实现上调整调度，让剩余的数据搬运与计算尽可能重叠。


## 优化 Memory-Bound Kernel

确定一个 kernel 是 memory-bound 后，优化有两个方向：一是减少 HBM 搬运量，提高算术强度；二是在搬运量无法继续减少时，让实际带宽尽可能接近硬件上限。

算子融合通常是最直接的方法。低算术强度的一个常见来源是：kernel 把中间张量写入 HBM，而下一个操作又立刻把它读回来。把 producer（产生中间结果的操作）和 consumer（使用中间结果的操作）融合在一起后，这个中间结果就可以保留在寄存器或片上存储中，例如 SMEM 或 TMEM，从而避免这次 HBM 往返。

- 将 GEMM 与 elementwise epilogue 融合；
- 将 normalization 融合进相邻算子；
- 计算 attention 时不生成完整的 score matrix。

另一个办法是通过 blocking 提高复用。Blocking 把大问题切成较小的 tile，使加载到片上的数据可以被多次使用。对 GEMM 来说，A 的一个元素会参与同一行多个 C 元素的计算，B 的一个元素也会参与同一列多个 C 元素的计算。如果每次使用都重新从 HBM 读取，数据搬运量会很大。

将 A/B tile 留在片上复用后，同样的 `2MNK` 次计算只需要更少的 HBM 访问，因此算术强度更高。其他存在 tile 复用的 workload 也可以采用同样的思路。

下面用一个简化模型说明这种复用。假设一个 CTA 计算 `B_M × B_N` 的 C tile；每个 K-stage 加载 `B_M × B_K` 的 A tile 和 `B_K × B_N` 的 B tile；每个元素占 `s` bytes。只统计这个 stage 的 A/B global memory 访问，不考虑 C 的读写，算术强度近似为：

$$
\mathrm{AI}
\approx
\frac{2 \times B_M \times B_N \times B_K}
{s \times (B_M \times B_K + B_K \times B_N)}
=
\frac{2 \times B_M \times B_N}
{s \times (B_M + B_N)}
$$

如果 `B_M = B_N = B`，就变成：

$$
\mathrm{AI} \approx \frac{B}{s}
$$

举个带数字的例子。这里整个 GEMM 的矩阵尺寸 `M`、`N`、`K` 不变，总计算量 `2MNK` 也不变；我们只是在比较同一个 GEMM 下不同 CTA tile 大小带来的片上复用差异。假设是 FP16/BF16，所以 `s = 2 bytes`，并且每个 K-stage 取 `B_K = 64`。如果 CTA tile 是 `16 × 16`，这一 stage 的计算量是：

$$
2 \times 16 \times 16 \times 64 = 32768
$$

A/B 从 global memory 读取的数据量是：

$$
2 \times (16 \times 64 + 64 \times 16) = 4096
$$

所以这一 stage 的 A/B global-memory 算术强度是 `32768 / 4096 = 8 FLOP/byte`。如果把 CTA tile 改成 `64 × 64`，仍然使用 `B_K = 64`，计算量变成：

$$
2 \times 64 \times 64 \times 64 = 524288
$$

A/B 读取的数据量变成：

$$
2 \times (64 \times 64 + 64 \times 64) = 16384
$$

于是算术强度变成 `524288 / 16384 = 32 FLOP/byte`。tile 变大后，每个从 global memory 读入的 A/B 元素会在片上参与更多次计算，因此相同计算量所需的数据搬运更少。

实际 kernel 还会受到 C 的读写、L2 reuse、occupancy、register pressure、SMEM bandwidth 和 Tensor Core issue 等因素影响。这里的模型虽然简化，但已经说明了 blocking 提高算术强度的原因：同一个 A/B 元素从 global memory 读入后，可以在 tile 内参与更多次乘加。

除了增加片上复用，还可以缩小数据类型。从 fp32 换成 fp16、fp8 或 fp4，可以减少数据搬运量，并提高每 byte 对应的有效计算量。如果低精度格式需要额外的 metadata、scale factor 或类型转换，实际收益会低于按 dtype 大小估算的结果。Scale factor 是低精度数据使用的缩放系数，block-scaled fp8 和 fp4 就需要这类辅助数据。即便如此，使用更小的 dtype 通常仍是提高算术强度的直接方法。

如果算术强度已经很难提高，优化目标就应转向有效带宽。纯 copy、简单的 elementwise 操作或大 tensor 上的 single-pass reduction，通常缺少可融合的中间结果，也没有足够的数据复用。这时应尽量做到：

- 每个 byte 只搬运一次，避免冗余读取；
- 使用 coalesced 或 vectorized 访存；
- 对规则的大块 tile 使用 TMA；
- 保持足够多的 memory requests 并发执行，避免内存管线空闲。

一旦一个 memory-bound kernel 已经达到内存上限，进一步优化计算部分就不会有帮助。唯一能让它更快的方法，是改变算法，让它搬运更少的数据。

## 优化阶梯

Roofline 模型可以判断一个 kernel 的性能上限，但不会告诉我们怎样实现才能接近这个上限。

一个大规模 fp16 GEMM 在理论上可能是 compute-bound 的。但这只说明 HBM 层的内存上限不是主要瓶颈，并不意味着任意一种实现都能达到 Tensor Core 的计算上限。要缩小这中间的差距，需要正确的指令、layout、staging、同步和调度。后续 GEMM 章节会在 B200 上通过一系列步骤展示这一点：每一步都保持相同的基本算法，但改变 tile 的计算方式或调度方式。

在 GEMM 的优化阶梯中，第一个明显的实测性能跃升，是从 thread-copy tiled 路径切换到 TMA-backed 路径。前者由 CTA 中的普通线程执行 GMEM 到 SMEM 的拷贝；后者把这种规则的 tile 搬运交给 TMA 硬件引擎，让 kernel 可以通过硬件管理的大块拷贝来持续为 Tensor Cores 提供数据。

在第一次跃升之后，后续优化都围绕一个问题展开：如何减少数据搬运、Tensor Core 计算和 epilogue 之间的等待。Software pipelining 和 warp specialization 会重新安排这些阶段，使不同硬件单元能够重叠工作。下一节具体说明这种调度方式。

有些结构调整本身不一定立刻提速。例如，warp specialization 可能暂时增加资源开销，但它为后续更复杂的重叠执行提供了必要的结构。

下图给出 GEMM 章节将要展开的优化路线。每个点对应一种实现结构，后文会逐步解释 TMA、warp specialization、CTA cluster 和 multi-consumer execution 分别改变了什么。

![B200 上的 GEMM 优化路线：从同步 tiled baseline，到 TMA、warp specialization、CTA cluster 和 multi-consumer execution 的实测点](../../img/gemm_perf.png)

## 通过重叠执行减少硬件空闲

一旦 GEMM 已经是 compute-bound 的，并且已经使用了 Tensor Cores，剩下的性能差距通常来自于硬件的空闲时间。

一个简单的 kernel 可能会这样执行：

```text
load tile k
compute tile k
store tile k
load tile k + 1
compute tile k + 1
store tile k + 1
```

这种调度会让各硬件单元轮流空闲。load 运行时，Tensor Core 可能在等待；Tensor Core 计算时，copy engine 可能处于空闲；store 写回时，两者都可能在等待。

相比之下，一个 pipelined kernel 会尽量让相互独立的阶段同时运行：

```text
load tile k + 1
compute tile k
store tile k - 1
```

在 Blackwell 上，这三个阶段分别主要由 TMA、`tcgen05.mma` 和 epilogue/store 路径完成，`mbarrier` 负责它们之间的数据交接。

重叠执行并不会消除依赖关系：tile `k` 的 MMA 仍然必须等待它加载完成，epilogue 也必须等待 MMA 完成。可以提前执行的是与当前计算没有直接依赖的工作，例如加载 tile `k+1`，或者写回 tile `k-1`。

## SM 占用率与资源压力

除了重叠执行（overlap），GPU 还可以通过提高 SM 占用率（occupancy）来隐藏延迟。

SM 占用率描述一个 SM 上能够同时驻留多少工作。当某个 warp 因等待而暂停时，scheduler 可以转而执行另一个已经就绪的 warp，从而隐藏延迟。

SM 占用率受 registers、shared memory、warp slots 和 CTA slots 的限制。Warp slots 和 CTA slots 分别规定一个 SM 能同时容纳的 warp 数和 CTA 数。如果每个 thread 使用很多 registers，或者每个 CTA 使用大量 shared memory，一个 SM 能驻留的 CTAs 或 warps 就会减少，occupancy 也会下降。

许多现代 Tensor Core kernel 会主动消耗更多资源，即使这会降低 occupancy。多 stage 的 shared memory pipeline 会占用 SMEM；较大的 register fragments 会占用 registers；TMEM allocation 会占用 Tensor Memory 容量；warp specialization 也可能把整组 warp 固定分配给 producer 或 consumer 角色。

这是有意做出的取舍。这些 kernel 不依靠大量 warp 同时驻留来隐藏延迟，而是在较少的驻留 CTA 内显式重叠不同阶段。只要 pipeline 能让 TMA、Tensor Core 和 store 路径持续运行，低 occupancy 的 kernel 仍然可以获得很高的性能。

两种方式各有适用场景。内存访问不规则、难以显式构造流水线的 kernel，通常更依赖高 occupancy；采用深度 staging 和 warp specialization 的 kernel，则可能用较低的 occupancy 换取更充分的阶段重叠。评价一个 kernel 时，不能只看 occupancy，还要看关键硬件单元是否被持续利用。


## 用 Roofline 指导优化

实际分析一个 kernel 时，可以按下面的顺序进行：

1. 估算算术强度，也就是每搬运一个 byte 完成多少次计算。
2. 将算术强度与 roofline 拐点比较，判断性能更可能受内存带宽还是计算吞吐限制。
3. 检查实际实现离对应上限还有多远，并优化真正处于瓶颈的资源。

对于 memory-bound kernel，重点是减少数据搬运并提高有效带宽；对于 compute-bound kernel，重点是减少计算单元的等待时间。Roofline 模型不能直接给出最终实现，但可以避免在不构成瓶颈的部分反复调参。
