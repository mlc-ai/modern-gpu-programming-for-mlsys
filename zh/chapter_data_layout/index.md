(chap_data_layout)=
# 数据布局及其记号

:::{admonition} 概览
:class: overview

- **数据布局**将张量的逻辑索引映射到物理位置。它会影响访存能否合并、是否产生 bank conflict，以及 tile 是否符合特定硬件单元要求的格式。
- 本章使用 `S[(shape) : (strides)]` 统一描述布局，并通过命名轴（`@laneid`、`@TLane` 等）描述数据在硬件资源上的分布，通过复制维度 `R[...]` 表示广播和复制。
- Swizzle 是一种基于 XOR 的地址重映射，可以在特定的元素宽度和访问模式下避免 shared memory bank conflict。
:::

同一组数字，如果以不同的物理排列方式写入内存，在同一块 GPU 上的运行速度可能相差一个数量级。

原因在于，张量的逻辑索引并不说明它的字节在物理上实际存放在哪里。硬件对这种位置关系非常敏感：它决定 32 个 lane 的 load 能否合并成一次 transaction，还是分散成 32 次；决定这些地址会落到不同的 memory bank，还是撞到同一个 bank 并被串行化；甚至还决定一个 tile 的字节排列是否符合 Tensor Core 能够读取的格式。

机器学习程序通常用逻辑 shape 来描述张量。**数据布局**补上了缺失的物理部分：它说明带有逻辑索引 `(i, j, …)` 的元素实际存放在哪里，可以是在 memory 中、register 中，也可以是在其他硬件存储空间中。

本章会介绍现代 GPU 编程中常见的主要布局。为了避免讨论变得过于复杂，我们会先引入一套**表示法**，用它统一描述机器学习系统中会遇到的各种布局形式。最后，我们会介绍 **swizzling**：它让同一个 tile 的按行访问和按列访问都能保持高效。

## Shape-Stride 模型

在介绍 GPU 特有的布局之前，我们先从最基本的 Shape-Stride 模型开始。后面讨论的许多布局，都可以看作这个模型的扩展。最基本的 Shape-Stride 模型由两部分组成：**shape** 描述张量在每个维度上的大小，**strides** 描述逻辑索引在某个维度上增加 1 时，物理位置要前进多少个元素。我们把这对信息写成 `S[(shape) : (strides)]`；要找一个逻辑索引对应的位置时，就把索引和 strides 做点积。比如，一个 row-major 的 `4×4` 矩阵可以写成：

```text
S[(4, 4) : (4, 1)]

addr(i, j) = i·4 + j·1
```

这就是经典的 Shape-Stride 模型；熟悉 CUTLASS/CuTe 的读者可以把 `S[...]` 看作 CuTe layout 记号的一个 row-major 简化版。

PyTorch 和 NumPy 中的 tensor 已经在使用这个模型：一块扁平的 storage buffer，加上描述如何解释这块 storage 的 `shape` 和 `strides`。

```python
import torch

t = torch.arange(12).reshape(3, 4)
t.shape        # torch.Size([3, 4])
t.stride()     # (4, 1)        ← 正是 S[(3, 4) : (4, 1)]
```

`t` 的底层 storage 仍然是一维的：

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

这里的 `t` 使用 `S[(3, 4) : (4, 1)]`：每行占 4 个连续元素，列方向的相邻元素在 storage 中也相邻。很多 view 类操作只需要更换 `shape` 和 `strides`，而不需要重新排列这些元素。以二维 tensor 的转置为例，PyTorch 中的 `permute(1, 0)` 等价于 `t.T`：

```python
tt = t.permute(1, 0)               # 或 t.T
tt.shape                           # torch.Size([4, 3])
tt.stride()                        # (1, 4)        ← strides 交换了，但没有搬数据
tt.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
                                   # True，底层仍然是同一块 storage
```

转置后的 view 使用 `S[(4, 3) : (1, 4)]`，因此 `tt[i, j]` 的地址偏移为 `i·1 + j·4`，正好对应原来 `t[j, i]` 的位置。对 contiguous tensor 做 `view`，或者在布局兼容时做 `reshape`，机制相同。NumPy 也采用这一模型，只是它的 `.strides` 以字节而不是元素为单位。

GPU layout 同样描述逻辑坐标到物理位置的映射；这个位置既可以是 memory 地址，也可以是后面命名轴表示的 lane、register 或其他硬件资源。不过，改变 GPU layout 并不一定能像 tensor view 那样零拷贝。只有新的布局与现有的字节排列和 ownership 兼容时，才可以只改变解释方式；如果元素改由其他 thread、lane 或 register 持有，或者 shared memory 的 swizzle 发生变化，就需要通过 loads、stores、shuffles 或 transpose 实际搬运或重排数据。

## Tile Layout

到目前为止，我们描述的是整个 tensor 的布局。但 GPU kernel 很少一次处理完整矩阵；它们通常处理更小的 tiles，而这些 tiles 会由不同硬件单元加载、重排并参与计算。这里并不需要引入新的概念：tiling 仍然可以看作一种布局，只是把原来的索引拆成更多维度。

这里采用一种常见的 tiling 展开方式，把每个原始维度依次拆成 outer 坐标和 inner 坐标，然后再进入下一个原始维度：

```text
(outer_dim0, inner_dim0, outer_dim1, inner_dim1, ...)
```

因此，把一个 8×8 矩阵切成 2×4 的 tiles 后，逻辑坐标 `(row, col)` 会变成 `(tile_row, row_in_tile, tile_col, col_in_tile)`。对应的 4-D layout 使用下面这组 strides，让每个 tile 保持 contiguous：

```text
S[(4, 2, 2, 4) : (16, 4, 8, 1)]
```

逻辑索引 `(i, j)` 会先拆成 `(i//2, i%2, j//4, j%4)`，再带入 strides 计算地址。

下图展示了这一索引分解和地址计算过程。

```{raw} html
<iframe src="../demo_zh/tiled_layout.html?v=tile-order-20260709" title="Tile layout: interactive address computation" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击一个 cell，查看它的 tiled index 和地址。*

## 命名轴

到目前为止，`S[...]` 里的每个 stride 都表示线性 memory 中的偏移，我们也一直把 address 当成这个空间里的位置。但在 GPU 上，数据并不一定只位于一个空间中：除了 memory，一个 tile 还可能分布在 warp lanes、thread registers，或者 TMEM lanes 和 columns 中。为了统一描述这些情况，我们把前面的记号推广到带命名轴的形式。基本思想是让每个 stride 系数都带一个 axis tag，用来说明这个坐标沿着哪个空间移动：`@m` 表示普通 memory，`@laneid` 表示 warp lanes，`@reg` 表示 registers，`@warpid` 表示 warps，`@TLane` 和 `@TCol` 分别表示 TMEM 的 lane 坐标（可以把它看作行）和 column 坐标（列）。有了这些 tag，同一个布局不仅能描述数据放在 memory 的哪里，也能描述数据如何分布到负责处理它的硬件资源上。

当 memory tag 被显式写出来后，一个 row-major 的 8×16 memory tile 就是：

```text
S[(8, 16) : (16@m, 1@m)]
```

当布局描述的是分散到 threads 上的数据，而不是线性 memory 中的数据时，这些 tag 就开始发挥作用了。例如，考虑下面这个布局：

```text
S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]
```

它描述了一个分布在 32 个 warp lanes 上的逻辑 `8×8` tile。逻辑坐标 `(row, col)` 首先被拆成 `(row, col//2, col%2)`，因此三个维度的 shape 分别是 `(8, 4, 2)`。带 tag 的 strides 进一步说明了这三个坐标如何映射到硬件资源：

```text
laneid = row·4 + (col//2)·1
reg    = col%2
```

也就是说，每个 lane 持有同一行中相邻的两个元素，它们分别放在该 lane 的 register 0 和 register 1 中。例如，逻辑坐标 `(5, 3)` 会被拆成 `(5, 1, 1)`，因此它位于 lane 21 的 register 1。这里的 `laneid` 指的是一个 warp 内的 warp lane index，也就是 `thread_index % warp_size`。这正是后续 layout generation 章节会介绍的 tensor-core register fragment。

下图展示了这个 `8×8` tile 在 warp lanes 和 registers 上的分布。

```{raw} html
<iframe src="../demo_zh/thread_register.html" title="Thread + register layout via named axes" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*一个使用 `@laneid` 和 `@reg` 的布局；点击一个 cell，查看它由哪个 lane/register 持有。*

## 分布式布局

前面我们用命名轴描述单个 GPU 内的 lanes 和 registers；同样的记号也可以扩展到多张 GPU。GPU mesh 是把多张 GPU 按一个或多个逻辑轴组织起来的设备网格；例如，`2×2` GPU mesh 包含四张 GPU，每张 GPU 都有一个由 `@gpuid_x` 和 `@gpuid_y` 组成的坐标。因此，`@gpuid_x` 和 `@gpuid_y` 这样的轴可以表示数据落在 GPU mesh 的哪个坐标上。借助这些轴，同一套记号也可以描述分布式训练和推理中常见的 sharding 模式。

不过，仅靠这些轴还不能表达复制，也就是同一份数据被放到多个位置。因此，我们引入记号 `R[n : stride]`，其中 `R` 表示一个复制维度。例如：

```text
R[2 : 1@gpuid_x]
```

表示沿着 `@gpuid_x` 轴复制 2 份。

把 sharding 和 replication 结合起来，一个表达式就可以同时描述张量在 2×2 GPU mesh 上的分片方式，以及沿某个轴的复制方式：

```text
S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)] + R[2 : 1@gpuid_x]
```

下图展示了一个小型 GPU mesh 上同时包含 sharding 和 replication 的布局，并支持在 fully-sharded、shard + replica 和 shard + offset 三种模式之间切换。

```{raw} html
<iframe src="../demo_zh/tile_distributed.html" title="Distributed layout across a GPU mesh" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*一个分布在 2×2 GPU mesh 上的布局；点击一个 cell，查看哪些 device 持有它。*

## TMEM 中 Scale Factors 的跨 Warp 广播

前面为了描述 GPU mesh，我们引入了 replication dimension `R[...]`。这个记号不仅适用于多设备场景，也可以描述单个 kernel 内部的数据广播。Blackwell 的 block-scaled MMA 就是一个例子。

Block-scaled MMA 面向低精度输入。它沿 K 维将 A、B 划分为若干 scale blocks，并为每个 block 配置一个 scale factor，用来恢复该块的数值尺度。设每个 scale block 包含 `K_blk` 个 K 元素，那么元素 `k` 所属 block 的索引为：

```text
sfk = k // K_blk
```

从数学上看，block-scaled MMA 等价于按对应的 scale factor 对 A、B 的元素进行缩放后完成矩阵乘加：

```text
A_real[m, k] = A_low[m, k] · SFA[m, k // K_blk]
B_real[k, n] = B_low[k, n] · SFB[n, k // K_blk]
D = C + A_real × B_real
```

`SFA[m, sfk]` 是 A 的第 `m` 行、第 `sfk` 个 K-scale block 对应的 scale factor；`SFB[n, sfk]` 是 B 的第 `n` 列、第 `sfk` 个 K-scale block 对应的 scale factor。`SF_K` 表示 K 维上 scale blocks 的数量。图中的示例取 `M = 128`、`SF_K = 4`，因此 A-side scale-factor tensor 的逻辑形状为 `128×4`。

为了说明这个布局，先固定一个 `sfk`，只考察 `SFA[m, sfk]` 在 `m = 0…127` 上的 128 个元素。这些元素并不会分别占用 128 条 TMEM lanes，而是先被紧凑地打包：

```text
TLane = m % 32
TCol  = m // 32
```

因此，`m = 0…31`、`32…63`、`64…95` 和 `96…127` 分别使用同样的 32 条 TMEM lanes，但位于四个不同的 columns。也就是说，128 个逻辑元素先被打包成一个 `32 lanes × 4 columns` 的基础布局。

接下来才是 replication。在实际的 SMEM-to-TMEM copy 中，这对应 `tcgen05.cp` 的 `.warpx4` multicast：为了让读取它的 warpgroup 中四个 warps 都能在自己的 32-lane TMEM window 中找到同一份 scale，硬件把这个 32-lane 基础布局沿 `TLane` 轴复制四份：

```text
S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]
```

这意味着对基础 lane `l`，同一个值会出现在 lanes `l`、`l+32`、`l+64` 和 `l+96` 中，而 column 保持不变。`R[4 : 32@TLane]` 不产生新的逻辑数据；它只是说明同一个值在四个 warp 的 TMEM window 中各出现一次。

扩展回完整的 `SFA[m, sfk]` 后，`m // 32` 决定落在哪一组四个 TMEM columns，`sfk` 决定组内的具体 column，因此 `TCol = (m // 32)·4 + sfk`。下图同时展示了这一打包过程和沿 `TLane` 轴的四份复制。

```{raw} html
<iframe src="../demo_zh/sf_tmem.html?v=warpx4-20260710" title="Scale factors in TMEM: packing and .warpx4 multicast" loading="lazy"
        style="width:100%; height:560px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击一个 scale factor，查看它的 TMEM 坐标以及分布在四个 warp window 中的副本。*

## Swizzle Layout

本章最后要介绍的是 swizzle layout，它主要用于解决 shared memory 中的 bank conflict。

GPU 的 shared memory 由多个 memory bank 组成，可以把每个 bank 理解为一条能够独立服务访问的存储通道。当不同 lane 的访问分布到不同 bank 上时，这些访问可以并行完成；但如果多个 lane 同时访问同一个 bank 中的不同地址，硬件就必须将它们分批处理，从而产生 bank conflict。

在 tensor 程序中，同一个 tile 往往会被沿不同方向访问。处理矩阵时，我们既可能连续读取一行，也可能取出一列。但简单布局通常只能让其中一种访问方式高效。以 row-major tile 为例，同一行的相邻元素地址连续，通常会分散到不同 bank；而同一列的相邻元素之间隔着一个 row stride。如果这个 stride 与 bank 的映射周期重合，多个 lane 的访问就可能集中到同一个 bank，产生 bank conflict。Column-major layout 的情况则恰好相反。

Swizzling 通过改变元素的物理地址排列来缓解这一问题，同时保持 tile 的逻辑形状不变。常见做法是将行索引的一部分与列索引做 XOR，使目标访问模式下的元素更均匀地分布到不同 bank 上。

在下面的 `8×8` 例子中，可以把逻辑坐标 `(row, logical_col)` 映射为：

```text
mapped_col   = logical_col XOR row
physical_addr = row·8 + mapped_col
```

`XOR` 是按位异或。例如，当读取逻辑列 `logical_col = 0` 时，第 `0…7` 行会分别得到 `mapped_col = 0 XOR row = 0…7`。这样，同一逻辑列的元素在各行中会落到不同的物理列，从而分散到不同 bank。

```{raw} html
<iframe src="../demo_zh/swizzle_8x8.html" title="8x8 XOR swizzle" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击一个 column index，比较普通 row-major 和 XOR swizzle 的 bank 映射：前者需要 8 个 cycle，后者只需要 1 个。*

上图用 8 个 bank 说明了 XOR 的基本思想。实际硬件使用更大的重复单元：我们把连续的 16 B 数据称为一个 sector，并用一个色块表示。对于 `SWIZZLE_128B`，atom 的每一行包含 8 个 sector，共 128 B；在常见的 4-byte bank 粒度下，这一行覆盖 32 个 bank slot。swizzle 根据行坐标对这 8 个 sector 的位置做 XOR 重排。

一个 `SWIZZLE_128B` atom 包含 8 行，因此大小为 `8 × 128 B = 1024 B`。这里的 `128 B` 指 atom 每一行在连续维度上的宽度，而不是 atom 的总大小。atom 是地址重排的最小重复块，更大的 tile 由多个 atom 平铺而成。

```{raw} html
<iframe src="../demo_zh/swizzle_128B.html" title="SWIZZLE_128B layout" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*一个 `SWIZZLE_128B` atom；每个 cell 表示 16 B sector。逐步查看 read cycles，观察 XOR 如何把一列访问分散到不同 bank。*

其他 swizzle mode 使用相同的层级，只是 atom 的每行宽度不同：`SWIZZLE_64B` 和 `SWIZZLE_32B` 的 atom 分别为 `8 × 64 B` 和 `8 × 32 B`。

下图可以直接比较这些 atom，其中还包括 16 B interleaved mode（无 XOR swizzle）。

```{raw} html
<iframe src="../demo_zh/swizzle_atom_general.html?v=interleaved-note-20260709" title="Swizzle atom layout per format (128B/64B/32B)" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*选择一种 swizzle 格式和数据类型，查看对应 atom 的形状（8 × N B）；将鼠标悬停在任意单元格上，即可查看该元素在 atom 内被重新映射到的位置。*

应该选择哪一种 swizzle mode？一个实用的原则是：**在 tile 尺寸允许的情况下，优先选择每行宽度最大的 atom。** 每行宽度为 `N` bytes 的 atom 要求 tile 的连续维度至少达到 `N` bytes，最好还能被 `N` 整除。因此，一行至少包含 128 bytes，也就是 64 个 `float16` 元素时，通常优先使用 `SWIZZLE_128B`；若连续维度不足 128 bytes，则选择能够容纳的 `SWIZZLE_64B` 或 `SWIZZLE_32B`。

对于图中使用 `fp16` 的访问方式，`SWIZZLE_128B` 可以让连续的行读取和跨 8 行的列读取都避免 bank conflict。不过，这一保证只适用于与硬件 descriptor 匹配的元素宽度、swizzle mode 和访问模式；元素宽度、对齐方式或访问模式改变后，仍可能产生冲突。

实际编程时，不需要手工计算 swizzle 后的地址。**swizzle 不属于 affine layout 本身，而是叠加在 affine address mapping 之上的独立、非仿射变换。** `S[...]` 先把逻辑元素映射到线性的 memory 地址 `@m`，swizzle 再重排这个地址。

所有访问同一个 tile 的操作必须使用一致的 swizzle mode，具体的地址变换由组合后的 layout 统一处理。不同硬件 engine 对 swizzle mode 的要求会随 GPU 架构代际变化，下一章会进一步介绍这些约束。
