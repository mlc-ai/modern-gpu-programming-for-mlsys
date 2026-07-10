(chap_layout_generations)=
# 不同 GPU 架构代际的 Tensor Core 操作数布局

:::{admonition} 概览
:class: overview

- 从 Ampere、Hopper 到 Blackwell，Tensor Core 执行的高层操作始终相同：`D = A B + C`。
- 不同代际之间的变化主要在于：操作数如何送入 Tensor Core、支持哪些 tile shape 和 data type，以及累加器存放在哪里。
- Ampere 使用 warp-level register fragments。Shared memory 中的 tile 通过 `ldmatrix` 加载到 fragment 中，累加器也保存在 registers 中。
- Hopper 允许 `wgmma` 通过 matrix descriptor 直接从 shared memory 读取操作数。Descriptor 会指定 Tensor Core 所要求的 shared-memory swizzle format。
- Blackwell 保留了从 shared memory 读取操作数的路径，但把累加器移到 TMEM。Block-scaled MMA 的 scale factors 也会通过 TMEM 暂存。
- 两个访存约束贯穿所有架构代际：global memory coalescing 和 shared memory bank conflict。
:::

从抽象层面看，Tensor Core 所执行的核心运算自 Volta 以来一直没有改变：读取 A、B 和 C，计算矩阵乘加，并得到结果 D：

$$
D = AB + C
$$

但这项运算背后的硬件接口却在不断演进。一个在某代 GPU 上表现出色的 kernel，迁移到下一代架构后未必仍然高效。更重要的是，如果操作数没有按照硬件要求的 layout 排列，即使程序所表达的数学公式仍然是 $D = AB + C$，Tensor Core 也可能读取到错误的数据，从而产生错误结果。因为 Tensor Core 直接处理的并不是抽象意义上的矩阵，而是按照特定硬件布局组织的数据。

本章将沿着三个 GPU 架构代际，介绍 Tensor Core 对操作数布局和数据供给方式的要求。Ampere 使用由一个 warp 持有的 register fragments 提供操作数和累加器；Hopper 改为从 shared memory 中读取输入操作数，并通过 descriptor 描述它们的布局；Blackwell 延续了 shared memory 操作数的设计，同时将累加器迁移到 TMEM 中。

三个架构执行的仍然是同一种矩阵乘加运算，但操作数存放在哪里、如何送入 Tensor Core，以及累加结果存放在哪里，都在持续发生变化。

我们将使用“{ref}`数据布局 <chap_data_layout>`”一章中介绍的 layout 记号，准确描述这些硬件接口所要求的布局。

## 两项始终存在的访存约束

在讨论 Tensor Core 对 layout 的要求之前，GPU kernel 的 layout 设计已经受到两项基本访存约束：global memory coalescing 和 shared memory bank conflict。

第一项约束是 global memory coalescing。当一个 warp 的 32 个 lanes 发起 global memory load 时，硬件会将它们访问的地址合并为尽可能少的、连续且对齐的 memory transactions。如果这些地址彼此分散，一次 warp load 就可能被拆成多次 transactions。即使搬运的数据量相同，也会消耗更多内存带宽，并带来更高的访问延迟。

第二项约束是 shared memory bank conflict。上一章已经介绍了 bank conflict 及 swizzling；这里需要记住的是，Tensor Core operands 的布局不仅要符合指令要求，还要让数据写入和读取 shared memory 时避免严重的 bank conflict。

无论是否使用 Tensor Core，kernel 都会受到这两项访存约束。Tensor Core kernel 还需要满足第三项要求：操作数必须按照具体 Tensor Core 指令所规定的 layout 排列。本章接下来将介绍这项要求如何随 Ampere、Hopper 和 Blackwell 三代架构不断演进。

## Ampere：分布在 warp 各 lane 中的寄存器 fragment

在 Ampere 架构上，Tensor Core 主要通过 warp-level 的 `mma.sync.aligned.m16n8k*` 系列指令完成矩阵乘加。与后续架构不同，`mma.sync` 的输入操作数和累加结果都保存在寄存器中。

A、B 操作数以及 C/D 累加器会按照 `mma.sync` 规定的布局，分散存放在一个 warp 的 32 个 threads 各自的寄存器中。每个 thread 只持有矩阵 tile 的一部分；它在 warp 中的 lane ID 决定了自己持有哪些元素。32 个 threads 中的这些寄存器合在一起，才表示一个完整的操作数或累加器 tile。这种分布在整个 warp 寄存器中的矩阵表示，称为 register fragment。

Shared memory 在这里主要充当中间暂存区。执行 MMA 之前，A 和 B 的 tile 必须先从 shared memory 加载到寄存器中，并按照 `mma.sync` 规定的 fragment layout 分配到各个 lanes。

数据路径可以概括为：

```text
SMEM --ldmatrix--> registers
registers --mma.sync--> registers
registers --常规 store--> SMEM
```

Ampere 上的大部分 layout 问题都围绕这条数据路径展开。这里的 `ldmatrix` 是一条由整个 warp 协同执行的 load 指令，专门负责把 shared memory 中的矩阵块加载到各个 thread 的寄存器中。在执行 `ldmatrix` 之前，A、B tile 必须先按照适合该指令读取的布局写入 shared memory。随后，`ldmatrix` 根据固定的 lane 和 register 映射，将数据组织成 `mma.sync` 所需的 register fragment。后文会进一步介绍它的具体加载形式和地址映射规则。

## Ampere Tensor Core 的操作数布局要求

在 Ampere 上，`mma.sync` 使用的 register fragment 可以进一步拆分为若干 `8×8` 子块。`ldmatrix` 以这样的 `8×8` 矩阵块为基本单位，从 shared memory 中加载数据，并将其分布到各个 lane 的寄存器中，形成 `mma.sync` 所要求的 operand fragment。

以输入类型为 `fp16` 或 `bf16`、累加类型为 `fp32` 的 `mma.m16n8k16` 为例。该指令计算一个 `16×8` 的输出 tile，其 C/D 累加器会按照固定模式分布在一个 warp 的 32 个 lanes 中。

对于 C/D 累加器，lane `l` 对应两行：

```text
l // 4
l // 4 + 8
```

以及相邻的两列：

```text
2 * (l % 4)
2 * (l % 4) + 1
```

因此，每个 lane 持有四个 `fp32` 累加值。令

```text
g = l // 4
t = l % 4
```

则这四个值对应的 `(m, n)` 坐标为：

```text
(g,     2t)
(g,     2t + 1)
(g + 8, 2t)
(g + 8, 2t + 1)
```

连续四个 lanes 共同覆盖第 `g` 行和第 `g+8` 行中的全部八列。

对于 A operand，每个 lane 持有 8 个 `fp16` 或 `bf16` 元素。相邻两个元素被打包到一个 32-bit register 中，因此总共占用 4 个 registers。它们对应的 `(m, k)` 坐标为：

```text
(a0, a1): (g,     2t + {0, 1})
(a2, a3): (g + 8, 2t + {0, 1})
(a4, a5): (g,     2t + {8, 9})
(a6, a7): (g + 8, 2t + {8, 9})
```

对于 B operand，每个 lane 持有 4 个元素，并将相邻两个元素打包到一个 32-bit register 中，因此共占用 2 个 registers。它们对应的 `(k, n)` 坐标为：

```text
(b0, b1): (2t + {0, 1}, g)
(b2, b3): (2t + {8, 9}, g)
```

从这些映射可以看出，`g` 决定 A 的 M 坐标和 B 的 N 坐标，而 `t` 与 lane 内不同的 register slot 一起覆盖 K 维。

具体的坐标映射会随 instruction shape 和 data type 而变化，但基本原则始终相同：Tensor Core 指令要求操作数按照特定的 per-lane register fragment layout 排列。如果元素没有被放入正确 lane 的正确 register slot 中，指令仍然会正常执行，但会把错误的元素组合起来进行乘加。

使用本书的 layout 记号，一个 `8×8` fragment 可以写成带有命名 lane axis 的形式，例如：

```text
S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]
```

前两个分量共同描述 fragment 中不同位置如何分布到各个 lanes，最后一个 `reg` 分量则描述同一 lane 内不同的 register slot。

## `ldmatrix`：从 shared memory 到寄存器 fragment

`ldmatrix` 负责将数据从 shared memory 加载到 Tensor Core 所使用的 register fragment 中。它是一条由整个 warp 协同执行的 load 指令，一次可以读取一个或多个 `8×8` 的 16-bit 矩阵块，并按照 `mma.sync` 所要求的方式，将数据分布到各个 lane 的寄存器中。

这条指令有以下几种形式：

```text
ldmatrix.sync.aligned.m8n8.x1.shared.b16
ldmatrix.sync.aligned.m8n8.x2.shared.b16
ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

此外，这些形式还可以带有可选的 `.trans` qualifier。

`.x1`、`.x2` 和 `.x4` 分别加载一个、两个或四个 `8×8` 矩阵块。每一行的起始地址由不同的 lane 提供：对于第 `m` 个矩阵块的第 `r` 行，其起始地址由 lane

```text
m * 8 + r
```

提供。因此，`.x1` 使用 lanes 0–7 提供八行的地址，`.x2` 使用 lanes 0–15，`.x4` 则使用全部 32 个 lanes。

加载得到的数据会被直接写入各个 lane 的寄存器，并共同构成 MMA 指令所需的 register fragment。对于基本的 `8×8` 情况，每个 lane 都会获得 fragment 中分配给自己的那部分元素。

如果改用逐 lane 的 `ld.shared` 指令完成同样的加载，kernel 就需要手动实现这套跨 lane 的数据分布和重排。`ldmatrix` 则通过一条 warp-collective 指令，直接完成从 shared memory 到 register fragment 的加载。

带有 `.trans` 的形式会在加载过程中转置每个 `8×8` 矩阵块。当操作数在 shared memory 中的存放方向与 MMA 指令所要求的方向不一致时，可以使用这一形式。

![`ldmatrix` 将一个 8×8 shared memory tile 加载到 warp register fragment；Ampere 使用普通 stores 完成反向写回，而专用的 `stmatrix` 指令要到 Hopper 才出现](../../img/ldstmatrix.svg)

## Ampere Fragment 的写回

`mma.sync` 完成后，累加器仍然是一个 register fragment，epilogue 必须将这个 fragment 写出。

Ampere 没有与 `ldmatrix` 反向对应的专用指令。Kernel 使用普通的 per-thread stores 将累加器写入 shared memory 或 global memory；写入前有时还需要进行 warp shuffles 或局部重排，才能得到合适的 layout。

这种模型相对直接，但也把大量 layout 处理暴露给了 kernel。输入侧由 `ldmatrix` 构造 fragment，计算指令读取并写入 register fragments，输出侧则通过这些 fragments 上的普通 stores 完成。

## Ampere 上的 Swizzle

Ampere kernel 已经需要对 shared memory 使用 swizzle，因为同一个 shared memory tile 往往以一种访问模式写入，再以另一种访问模式读取。

假设一个 tile 沿行方向从 global memory 写入。Row-major layout 可以让写入保持 coalesced，并且对 bank 访问友好。但随后，`ldmatrix` 可能以一种近似沿列方向、或跨越多个 `8×8` subtiles 的模式读取这个 tile。对于普通 row-major layout，这些读取可能集中到同一个 shared memory bank。

以一个简单的 `(8, 64)` float16 tile 为例，一行占用：

```text
64 * 2 bytes = 128 bytes
```

Shared memory 的每个 bank 对应 4 bytes，32 个 banks 的地址映射每 128 bytes 重复一次。这个 tile 的 row stride 恰好也是 128 bytes，因此沿固定列向下移动时，每一行都会映射到相同的 bank。八行可能全部集中到一个 bank 上，形成 8-way conflict。

改成普通的 column-major layout 也不能解决全部问题，它通常只是把 conflict 转移到另一种访问上：列方向读取变得更好，但行方向写入变得更差。

XOR swizzle 通过让物理列位置依赖于行坐标来解决这个问题。一个简单形式是：

```text
physical_col = logical_col xor row
```

逻辑 tile 保持不变，但 shared memory 中的物理位置被重新排列，使行方向写入和 Tensor Core 的读取模式都能避免 bank conflict。

在 Ampere 上，这种 swizzle 通常通过手写的 shared memory index 计算实现。后续架构则会把它放进硬件 engine 使用的 descriptor format 中。

![在普通 row-major tile 中，行方向写入会分散到不同 banks，而列方向读取会集中到同一个 bank；XOR swizzle 在不破坏 coalesced row write 的前提下，将列方向读取分散到不同 banks](../../img/swizzle_conflict.svg)

## Hopper：`wgmma`、Shared Memory Descriptor 与 Swizzle Format

Hopper 改变了 Tensor Core 输入侧的数据路径。对于来自 shared memory 的 operands，不再要求先通过 `ldmatrix` 将它们全部加载到 registers 中；Hopper 的 `wgmma` 可以直接从 shared memory 读取。

B operand 通过 shared memory matrix descriptor 读取。A operand 既可以通过 shared memory descriptor 读取，也可以来自 registers，分别对应 `.ss` 和 `.rs` 两种形式。

对于来自 SMEM 的 operands，这样就不再需要显式执行 `ldmatrix`。但 layout 要求并没有消失：Tensor Core 仍然要求 operand 按照精确的 shared memory format 存放。变化在于，这个 format 现在通过 matrix descriptor 告诉硬件。

## Hopper Tensor Core 所需的操作数布局

Hopper shared memory matrix descriptor 是一份对 shared memory matrix tile 的紧凑描述，它告诉 `wgmma` 如何将逻辑 operand coordinates 转换成 shared memory addresses。

Descriptor 包含以下字段：

```text
start address
leading dimension offset
stride dimension offset
swizzle mode
base offset
```

这些字段的具体含义取决于 operand major mode。对于 K-major tile，其中一个 stride 沿 K 维移动，另一个沿 M 维移动；对于 MN-major tile，两者的作用会交换。

Swizzle mode 是 shared memory descriptor format 的一部分，例如：

```text
SWIZZLE_NONE
SWIZZLE_32B
SWIZZLE_64B
SWIZZLE_128B
```

Swizzle mode 决定两件事：descriptor 使用的 atom shape，以及 atom 内部应用的 XOR permutation。例如，128-byte swizzle mode 会把 operand 看成由 `8 rows × 128 bytes` atoms 组成的网格，并在每个 atom 内应用 swizzle。

Kernel 仍然必须把字节放在正确位置。Shared memory tile 通常由 TMA 填充，而 TMA descriptor 必须使用与随后 `wgmma` descriptor 指定的相同 swizzle format。如果 TMA 写入的是 128-byte swizzled tile，`wgmma` descriptor 也必须按 128-byte swizzled tile 读取。只要 descriptor 与实际数据不一致，Tensor Core 就会读到被打乱的 operands。

这就是 Hopper 相比 Ampere 的主要变化。Swizzle 不再只体现在手写的 shared memory index 计算中，而是由 descriptor 直接编码。负责写入 tile 的 TMA load 和负责读取 tile 的 `wgmma` 指令，可以在各自的 descriptor 中指定相同的 swizzle format。

![Hopper shared memory matrix descriptor 将 operand coordinates 映射到经过 swizzle 的 shared memory atoms：descriptor strides 选择 atom，swizzle 决定 atom 内的 byte position](../../img/smem_descriptor.svg)

## Hopper 输出仍然使用寄存器

Hopper 改变了输入路径，但累加器仍然位于 registers 中。

`wgmma` 指令会把累加器写入 per-thread register fragment。具体 fragment size 和 register count 取决于 instruction shape，例如 `m64nNk16` 中的 N 会决定 accumulator registers 的数量。但基本思路与 Ampere 相同：epilogue 处理的是 register fragment。

因此，Hopper 的输入和输出采用两套不同的 layout 机制：输入 operands 通过 shared memory descriptor 提供，swizzle 也编码在 descriptor 中；输出 accumulator 则仍然分布在各个 threads 的 registers 中。

Blackwell 会进一步改变输出侧。

## Blackwell：`tcgen05` 和 TMEM

Blackwell 为 data operands 保留了 shared memory descriptor 这一思路。A 和 B 仍然会按照 Tensor Core 要求的 layout 准备在 shared memory 中；某些 mode 也允许从 TMEM 读取 A operand。

最大的变化是累加器。`tcgen05.mma` 不再把累加器作为长期存活的 register fragment，而是将其写入 Tensor Memory，也就是 TMEM。在计算阶段，累加器会一直留在 TMEM 中；随后，epilogue 使用 `tcgen05.ld` 将它加载回 registers。

这样，输出 layout 问题就从 registers 转移到了 TMEM。Kernel 必须分配 TMEM、选择正确的 TMEM layout、等待 MMA 完成，再通过匹配的 `tcgen05.ld` 路径取回 accumulator fragment，供 epilogue 使用。

与前代差异最大的 layout，是 block-scaled MMA 使用的 scale-factor layout。

## TMEM 中的 Scale-Factor Layout

`mxfp8`、`nvfp4` 等 block-scaled MMA mode 会增加 scale-factor operands。除 A 和 B 外，MMA 还会读取：

```text
SFA(M, SFK)
SFB(N, SFK)
```

其中，`SFK` 表示 K 维上的 scale blocks 数量。

Data operands A 和 B 位于 shared memory，而 scale factors 位于 TMEM，因此二者的数据搬运路径不同。

TMA 可以从 global memory 加载到 shared memory，但不能直接加载到 TMEM。因此，scale factors 通常需要经过两步搬运：

```text
global memory --TMA--> shared memory
shared memory --tcgen05.cp--> TMEM
```

完成这次复制后，scale factors 才会进入 `tcgen05.mma` 所要求的 memory space。

TMEM scale-factor layout 使用 TMEM 的硬件坐标 Lane 和 Col。在 TIRx layout 记号中，这两个 axes 分别写作 `TLane` 和 `TCol`。

一个 128-row scale vector 会先被压缩到一组 32 lanes 中，再复制到 TMEM 的四个 32-lane windows。在 layout 记号中，核心模式是：

```text
S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
```

其中，shard 放置基础的 32-row group：

```text
TLane = r
TCol  = s
```

replica term 会在 lane offsets 0、32、64 和 96 处添加副本：

```text
TLane = r + 32 * q, q ∈ {0, 1, 2, 3}
TCol  = s
```

这对应向四个 warp 广播的模式。同一组紧凑排列的 scale factors 会出现在完整的 128-lane TMEM space 中。

32-bit `TCol` cell 内部还会进行 byte packing。具体打包方式取决于 `scale_vec` mode：

```text
1X：一个 scale value 广播到整个 32-bit cell
2X：打包两个 scale values，并分别复制一次
4X：打包四个 K-block scale values
```

![`scale_vec` byte packing：1X 将一个 scale 广播到 4-byte cell；2X 打包两个 scale，并分别复制一次；4X 打包四个 K-block scales](../../img/sf_scale_vec.svg)

## 一个反复出现的 Fragment

虽然周围的数据搬运路径不断变化，但有一种结构反复出现：m8n8-style register fragment。

在 Ampere 上，`ldmatrix` 构造这一 fragment，供 `mma.sync` 读取。

在 Hopper 上，`wgmma` 将累加器写成 register fragment，供 epilogue 使用。

在 Blackwell 上，累加器在计算阶段位于 TMEM，但 `tcgen05.ld` 会在 epilogue 处理和存储结果前，将其重新加载为 register fragment。

因此，fragment 并没有消失，只是作用发生了变化。早期架构会在整个计算阶段将累加器保存在 fragment 中；Blackwell 则主要在 TMEM 与 epilogue 的边界上使用它。

## 主线

在 Ampere 上，kernel 会显式构造 Tensor Core register fragments。Shared memory swizzle 主要由 kernel 通过 index 计算负责。

在 Hopper 上，Tensor Core 可以通过 matrix descriptor 直接从 shared memory 读取 operands。Swizzle 成为 TMA 和 `wgmma` 共同使用的命名 descriptor format。

在 Blackwell 上，输入侧仍然使用 shared memory operands，但累加器被移到 TMEM。Block-scaled MMA 还增加了必须暂存到 TMEM 中的 scale-factor operands。

Descriptor 并不会消除 layout 工作，而是将 layout contract 显式化。Kernel 仍然必须保证数据搬运路径、memory layout 和 Tensor Core 指令彼此一致：写入 swizzled SMEM tile 的 TMA descriptor、读取该 tile 的 MMA descriptor，以及附着在 buffer 上的 layout，都必须描述相同的物理排列。

只要其中任何一项不一致，硬件仍然会执行，但它读取的字节可能是错误的，访问也可能很慢。因此，layout 并不是 Tensor Core kernel 外围的装饰，而是指令接口的一部分。
