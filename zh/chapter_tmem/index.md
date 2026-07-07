(chap_tmem)=
# 特殊内存：TMEM

:::{admonition} 概览
:class: overview

- TMEM 是 Blackwell 上 `tcgen05` 使用的专用内存空间。它是每个 SM 上的二维暂存区，包含 128 个 Lane 行和最多 512 个 Col 列。
- `tcgen05.mma` 会把 accumulator 写入 TMEM。Block-scaled MMA 也会用 TMEM 保存 scale factor。
- TMEM 通过 Lane 和 Col 寻址。在 TIRx 的布局记号里，这两个硬件轴写作 `TLane` 和 `TCol`。
- TMEM 不像寄存器那样自动分配。Kernel 必须以 32 列为单位显式分配和释放 TMEM。
- 普通 shared-memory load/store 不能访问 TMEM。TMEM、寄存器和 shared memory 之间的数据搬运需要通过专用的异步 `tcgen05` 指令完成。
:::

在 Hopper 以及更早的 GPU 上，Tensor Core（{ref}`chap_tensor_cores`）的 accumulator 位于寄存器中。这个模型很容易理解：MMA 指令产生一个寄存器 fragment，kernel 在计算阶段让这个 fragment 保持存活，epilogue 随后读取它、做类型转换，并把结果存出去。

问题在于寄存器压力。寄存器是固定的 per-thread 资源。随着 MMA tile 变大，accumulator fragment 也会变大。到一定程度之后，accumulator 会挤占线程需要保存的其他值。更大的 tile 有利于 Tensor Core 吞吐，但把整个 accumulator 都放在寄存器里，会让这些大 tile 更难使用。

Blackwell 改变了这段数据路径。`tcgen05` 的 accumulator 不必在整个计算阶段都留在寄存器里。相反，`tcgen05.mma` 会把 accumulator 写入 Tensor Memory，也就是 TMEM。TMEM 是早期 NVIDIA GPU 没有的内存空间。它是 SM 上的一个二维暂存区，形状是 128 个 Lane 行乘以最多 512 个 Col 列，并且作用域属于使用它的 CTA。

这个额外的内存空间让 Blackwell 可以支持更大的 Tensor Core tile，而不必把完整 accumulator 压到每个线程的寄存器里。但 TMEM 并不像寄存器那样自动存在。编译器不会把它当作普通寄存器存储直接分配给程序。Kernel 必须分配 TMEM，用正确的布局寻址，通过正确的指令搬入搬出，并在 CTA 完成后释放它。

## 二维地址空间

TMEM 不是一个扁平的 byte array。它是二维地址空间。硬件把两个坐标称为 Lane 和 Col。TMEM 有 128 个 Lane 行，最多 512 个 Col 列。每个 Col 是一个 32-bit 列。

这个形状很重要，因为 `tcgen05.mma` 会按照这个二维结构把 accumulator 写入 TMEM。一个 TMEM 位置由 Lane 坐标和 Col 坐标描述，而不是由一个类似 shared memory byte offset 的单一地址描述。

当 kernel 在 TIRx 中声明 TMEM buffer 时，会给这个 buffer 一个覆盖这两个硬件坐标的布局。在布局记号（{ref}`chap_data_layout`）里，我们把 TMEM Lane 轴写作 `TLane`，把 TMEM Col 轴写作 `TCol`。这些名字不是要替代官方硬件术语，而是 DSL 中的布局轴名，用来显式表达 TMEM 的两个维度。

例如，一个 accumulator tile 可以写作：

```text
S[(128, N) : (1@TLane, 1@TCol)]
```

这表示 tile 沿硬件 Lane 维度有 128 行，沿硬件 Col 维度有 `N` 列。在布局记号中，这两个维度分别是 `TLane` 和 `TCol`。这个布局是直接映射：相邻行沿 `TLane` 移动，相邻列沿 `TCol` 移动。下图展示了这个网格，其中硬件 Lane 沿 128 行向下，硬件 Col 沿列方向展开。

![TMEM as a 2D grid: TLane rows × TCol columns](../../img/tmem_grid.png)

核心要点是：TMEM 是 tile 布局问题的一部分。它不是 Tensor Core 背后的隐藏存储。Kernel 必须命名这块内存，从中分配列，并使用与 `tcgen05` 指令读写方式匹配的布局。

## 分配

Kernel 使用 TMEM 之前，必须先预留空间。这一点不同于寄存器。寄存器由编译器分配，而 TMEM 由 kernel 显式分配。

分配按 CTA 进行。CTA 中的一个 warp 会请求一段 TMEM 列。请求以 32 列为单位，列数会按照硬件分配规则向上取整。分配完成后，CTA 会得到一个 base TMEM address。后续 `tcgen05` 指令用这个 base address 访问预留区域。

把 TMEM 理解成一种有预算的 CTA 资源很有用，它类似 shared memory。CTA 拥有它分配到的 TMEM 列。Kernel 决定 accumulator、scale factor 或临时 staging 需要多少列。CTA 完成后，必须释放这段分配。

因此，TMEM 也是 kernel 资源规划的一部分。更大的 accumulator tile 可能提高 Tensor Core 吞吐，但会消耗更多 TMEM 列。Block-scaled MMA 可能还需要额外 TMEM 空间来保存 scale factor。Kernel 必须让这些用途都落在可用的 TMEM 预算内，就像必须让 shared-memory buffer 落在 SMEM 预算内一样。

## 读写 TMEM

普通 `ld.shared` 和 `st.shared` 指令不能访问 TMEM。TMEM 是单独的地址空间，因此数据必须通过专用 `tcgen05` 指令移动。

主要有三条路径。

第一条路径是 `tcgen05.ld`，它把数据从 TMEM 加载到寄存器中。这是 MMA 阶段之后 epilogue 使用的路径。Accumulator 已经在 TMEM 中产生，但 epilogue 通常需要寄存器 fragment，才能做类型转换、执行 elementwise 操作并写出最终结果。

在 DSL 层面，TMEM load 分布在一个 warpgroup 上。它会 lowering 成四个 warp-level 的 `tcgen05.ld` 操作，每个 warp 一个。每个 warp 处理 128 个 TMEM Lane 行中的 32 行，因此四个 warp 合起来覆盖完整的 Lane 维度。在布局记号里，这个完整维度就是 `TLane` 轴。

指令本身来自一组 load shape，例如 `.16x64b`、`.16x128b`、`.16x256b`、`.32x32b` 和 `.16x32bx2`，并带有从 `.x1` 到 `.x128` 的 repeat factor。选择的 shape 决定读取多少个 TMEM 列，以及每个线程会收到多少个寄存器。

重要结果是寄存器 fragment 的布局。对于常见的 epilogue 路径，lane `l` 会收到来自 TMEM 行 `l / 4` 以及两个列位置的值。这会产生与早期世代从 MMA 直接暴露出的 per-lane accumulator fragment 同类的布局（{ref}`chap_layout_generations`）。这种连续性很重要：虽然 Blackwell 的 accumulator 在计算阶段位于 TMEM 中，但 epilogue 仍然可以复用 Ampere `mma` 或 Hopper `wgmma` 中已经使用过的寄存器级 cast 和 store 结构。

![tcgen05.ld / st move the TMEM accumulator to and from registers in the m8n8 fragment (lane l → row l/4, two columns)](../../img/tcgen05_ldst.svg)

第二条路径是 `tcgen05.st`，它把数据从寄存器写回 TMEM。这是 `tcgen05.ld` 的反方向。当线程已经持有一个寄存器 fragment，并且需要把它放入 TMEM 时会使用这条路径。例如，某些操作数或中间值可能会先经过寄存器 staging，再写入 TMEM 供后续 `tcgen05` 操作使用。

第三条路径是 `tcgen05.cp`，它把数据从 shared memory 拷贝到 TMEM。这是一条 bulk copy 路径，常用于 block-scaled MMA 的 scale factor。在这种情况下，TMA 或普通线程代码会先把 scale 数据准备到 shared memory 中，然后 `tcgen05.cp` 把它移动到 Tensor Core 期望的 TMEM 布局里。

这三条路径都是异步的。`tcgen05.ld`、`tcgen05.st` 或 `tcgen05.cp` 指令可能在数据搬运完成之前就返回。因此，kernel 在消费结果或复用存储之前，必须使用正确的完成机制（{ref}`chap_async_barriers`）。

等待路径取决于具体指令。`tcgen05.ld` 通过 `tcgen05.wait::ld` 完成。`tcgen05.st` 通过 `tcgen05.wait::st` 完成。`tcgen05.cp` 和 `tcgen05.mma` 一样，通过 commit group 和 `mbarrier` 完成。如果数据需要从一组线程交给另一组线程，kernel 还可能需要 fence，确保接收方线程按预期顺序看到已完成的写入。

TMEM 位于 Blackwell Tensor Core 数据路径的中间。TMA 把操作数 staged 到 shared memory。`tcgen05.mma` 读取操作数并累加到 TMEM。对于 block-scaled MMA，scale factor 也可以 staged 到 TMEM。计算阶段结束后，`tcgen05.ld` 把 accumulator 带回寄存器，epilogue 再转换并存出最终输出。
