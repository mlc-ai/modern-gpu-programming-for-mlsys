(chap_data_layout)=
# 数据布局及其记号

:::{admonition} 概览
:class: overview

- **数据布局**将张量的逻辑索引映射到物理位置。它会影响访存能否合并、是否产生 bank conflict，以及 tile 是否符合特定硬件单元要求的格式。
- 本章使用 `S[(shape) : (strides)]` 统一描述布局，通过命名轴（`@laneid`、`@TLane` 等）描述物理坐标，通过复制维度 `R[...]` 表示广播和复制，并用固定 offset 移动整个布局。
- Swizzle 是一种基于 XOR 的地址重映射，可以在特定的元素宽度和访问模式下避免 shared memory bank conflict。
:::

同一组数字，如果以不同的物理排列方式写入内存，在同一块 GPU 上的运行速度可能相差一个数量级。

原因在于，张量的逻辑索引并不说明它的字节在物理上实际存放在哪里。硬件对这种位置关系非常敏感：它决定 32 个 lane 的 load 能否合并成一次 transaction，还是分散成 32 次；决定这些地址会落到不同的 memory bank，还是撞到同一个 bank 并被串行化；甚至还决定一个 tile 的字节排列是否符合 Tensor Core 能够读取的格式。

机器学习程序通常用逻辑 shape 来描述张量。**数据布局**补上了缺失的物理部分：它说明带有逻辑索引 `(i, j, …)` 的元素实际存放在哪里，可以是在 memory 中、register 中，也可以是在其他硬件存储空间中。

本章会介绍现代 GPU 编程中常见的主要布局。我们先建立一套统一的**表示法**，再用它描述不同物理空间中的数据布局。最后介绍 **swizzling**：它通过重排地址映射，在匹配的访问模式下兼顾同一个 tile 的按行和按列访问。

## Shape-Stride 模型

在介绍 GPU 特有的布局之前，我们先从最基本的 Shape-Stride 模型开始。后面讨论的许多布局，都可以看作这个模型的扩展。最基本的 Shape-Stride 模型由两部分组成：**shape** 描述张量在每个维度上的大小，**strides** 描述逻辑索引在某个维度上增加 1 时，物理位置要前进多少个元素。我们把这对信息写成 `S[(shape) : (strides)]`；要找一个逻辑索引对应的位置时，就把索引和 strides 做点积。比如，一个 row-major 的 `4×4` 矩阵可以写成：

```text
S[(4, 4) : (4, 1)]

addr(i, j) = i·4 + j·1
```

这就是经典的 Shape-Stride 模型。PyTorch 和 NumPy 中的 tensor 已经在使用这个模型：一块扁平的 storage buffer，加上描述如何解释这块 storage 的 `shape` 和 `strides`。

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

GPU kernel 很少一次处理完整矩阵，通常会将矩阵划分成更小的 tiles。例如，可以把一个 `8×8` 矩阵划分成 `2×4` 的 tiles，并让这些 tiles 按 row-major 顺序存储、每个 tile 内部也保持 row-major contiguous。

要准确描述这种排列，需要同时表示 tile 在矩阵中的位置和元素在 tile 内的位置。这里先给出本章使用的完整 layout 定义，再把它应用到这个例子。

### Layout 函数

设 $A$ 是命名轴的集合。不同轴上的整数坐标组成轴空间：

$$
\mathbb{Z}A=
\left\{
\sum_i z_i\mathbin{@}a_i
\;\middle|\;
z_i\in\mathbb{Z},\ a_i\in A
\right\}.
$$

一个 iter 定义为三元组

$$
I=(e_I,s_I,a_I),
\qquad
e_I>0,\quad s_I\ne0,\quad a_I\in A,
$$

其中 $e_I$、$s_I$ 和 $a_I$ 分别是它的 extent、stride 和 axis。它对应的映射为

$$
f_I:[0,e_I)\longrightarrow\mathbb{Z}A,
\qquad
f_I(x)=(x s_I)\mathbin{@}a_I.
$$

一个完整的 layout 定义为：

$$
\begin{aligned}
L&=(D,R,O),\\
D&=(I_0,\ldots,I_{n_D-1}), & n_D&\ge1,\\
R&=(J_0,\ldots,J_{n_R-1}), & n_R&\ge0,\\
O&\in\mathbb{Z}A.
\end{aligned}
$$

其中，$D$ 是由 sharded iters 组成的有序 tuple，$R$ 是由 replicated iters 组成的 multiset，$O$ 是固定 offset。

令

$$
E_D=\prod_k e_{I_k}.
$$

对于扁平逻辑索引 $x\in[0,E_D)$，标准的 lexicographic unflatten

$$
\iota:[0,E_D)\longrightarrow
\prod_k[0,e_{I_k})
$$

按照各个 sharded iter 的 extent 将 $x$ 展开为坐标 tuple。基础物理位置为

$$
f_D(x)=
\sum_{k=0}^{n_D-1}
\bigl(\iota(x)_k\,s_{I_k}\bigr)\mathbin{@}a_{I_k}.
$$

对于 replicated iters，令

$$
E_R=\prod_t e_{J_t},
\qquad
\text{并约定当 }R=\varnothing\text{ 时 }E_R=1.
$$

对于 replica coordinate tuple

$$
r\in\prod_t[0,e_{J_t}),
$$

对应的副本偏移为

$$
f_R(r)=
\sum_{t=0}^{n_R-1}
\bigl(r_t\,s_{J_t}\bigr)\mathbin{@}a_{J_t}.
$$

因此，完整 layout 是下面这个集合值映射：

$$
f_L(x)=
\left\{
f_D(x)+f_R(r)+O
\;\middle|\;
r\in\prod_t[0,e_{J_t})
\right\}.
$$

如果 $R=\varnothing$，则 $f_L(x)=\{f_D(x)+O\}$；否则，一个逻辑元素可以对应多个物理位置。

### 用 Layout 函数表示 Tiling

回到前面的 `8×8` 矩阵。这个例子没有 replication 或 offset，因此只需要考虑 $f_D$。对于逻辑坐标 `(i, j)`，首先按原矩阵的 shape 得到扁平逻辑索引：

```text
x = i·8 + j
```

将矩阵划分成 `2×4` 的 tiles 后，会得到 4 个 tile rows、每个 tile 内的 2 个 rows、2 个 tile columns，以及每个 tile 内的 4 个 columns。因此，选择下面这组 sharded iter extents：

```text
(4, 2, 2, 4)
```

$\iota$ 按照这些 extents 对 `x` 做 unflatten：

```text
(c0, c1, c2, c3) = unflatten(x; 4, 2, 2, 4)

c0 = x // 16
c1 = (x // 8) % 2
c2 = (x // 4) % 2
c3 = x % 4
```

代入 `x = i·8 + j` 后：

```text
c0 = i // 2    = tile_row
c1 = i % 2     = row_in_tile
c2 = j // 4    = tile_col
c3 = j % 4     = col_in_tile
```

因此，tile 坐标分解就是 layout 函数对扁平逻辑索引 `x` 做 unflatten 的结果，不需要额外定义一个 tiling 函数。

Sharded iter extents 确定逻辑坐标如何分解，strides 则确定这些坐标如何映射到物理位置。对于前面指定的 tile-major 排列，对应的 shard mapping 是：

```text
f_D(x) = c0·16 + c1·4 + c2·8 + c3·1
```

所以这个 layout 可以写成：

```text
S[(4, 2, 2, 4) : (16, 4, 8, 1)]
```

对于原始矩阵坐标 `(i, j)`，最终地址为：

```text
f_D(i·8 + j)
    = (i//2)·16 + (i%2)·4 + (j//4)·8 + (j%4)·1
```

这里的 shard iter 顺序是 `(tile_row, row_in_tile, tile_col, col_in_tile)`，物理嵌套顺序则是 `tile_row → tile_col → row_in_tile → col_in_tile`，所以 strides 不一定从左到右递减。

下图展示了这一索引分解和地址计算过程。

```{raw} html
<iframe src="../demo_zh/tiled_layout.html?v=tile-order-20260709" title="Tile layout: interactive address computation" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击一个 cell，查看它的 tiled index 和地址。*

## 命名轴：从线性地址到物理坐标

到目前为止，`S[...]` 中的 strides 都是普通整数，映射函数返回的是线性 memory 中的一个地址。GPU 上的物理位置却不一定能用一个整数表示：TMEM 使用 lane 和 column 两个坐标，register fragment 需要同时说明 lane 和 register，分布式布局还可能需要 GPU mesh 坐标。

命名轴把 Shape-Stride 模型推广到了这些情况。每个 stride 系数都可以带一个 axis tag，用来说明这一项沿哪个物理轴移动；映射函数的结果也从一个整数，推广为一组带名称的物理坐标。

### 多维物理空间

`@m` 表示普通的线性 memory。显式写出这个 tag 后，一个 row-major 的 `8×16` memory tile 是：

```text
S[(8, 16) : (16@m, 1@m)]

(row, col) = unflatten(x; 8, 16)
f_D(x) = (row·16 + col)@m
```

TMEM 则是一个二维物理空间，`@TLane` 和 `@TCol` 分别表示 TMEM 的 lane 坐标（可以看作行）和 column 坐标（列）。例如：

```text
S[(128, 256) : (1@TLane, 1@TCol)]

(row, col) = unflatten(x; 128, 256)
f_D(x) = row@TLane + col@TCol
```

### 数据在 Lane 和 Register 中的分布

命名轴不只表示 memory 或 device 坐标，也可以表示数据由哪些执行资源持有。`@laneid` 表示一个 warp 内的 lanes，`@reg` 表示每个 thread 的 registers，`@warpid` 表示 warps。例如，考虑下面这个布局：

```text
S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]
```

它描述了一个分布在 32 个 warp lanes 上的逻辑 `8×8` tile。对于逻辑坐标 `(row, col)`，扁平索引是 `x = row·8 + col`。layout 按照 shard extents `(8, 4, 2)` 对 `x` 做 unflatten：

```text
(c0, c1, c2) = unflatten(x; 8, 4, 2)
             = (row, col//2, col%2)

f_D(x) = (c0·4 + c1·1)@laneid + c2·1@reg
```

因此：

```text
laneid = row·4 + col//2
reg    = col%2
```

也就是说，每个 lane 持有同一行中相邻的两个元素，它们分别放在该 lane 的 register 0 和 register 1 中。例如，逻辑坐标 `(5, 3)` 对应 `x = 43`，unflatten 后得到 `(5, 1, 1)`，因此它位于 lane 21 的 register 1。这里的 `laneid` 指的是一个 warp 内的 warp lane index，也就是 `thread_index % warp_size`。这类布局可以用于描述后续 layout generation 章节中的 tensor-core register fragments。

下图展示了这个 `8×8` tile 在 warp lanes 和 registers 上的分布。

```{raw} html
<iframe src="../demo_zh/thread_register.html" title="Thread + register layout via named axes" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*一个使用 `@laneid` 和 `@reg` 的布局；点击一个 cell，查看它由哪个 lane/register 持有。*

## Replication 与 Offset

### TMEM 中 Scale Factors 的跨 Warp 广播

先看一个发生在单个 kernel 内部的例子。Blackwell block-scaled MMA 会把 scale factors 存放在 TMEM 中，并通过 `.warpx4` broadcast 让读取它的四个 warps 都能访问同一份数据。

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
TLane  = m % 32
Mgroup = m // 32
TCol   = Mgroup·4 + sfk
```

因此，`m = 0…31`、`32…63`、`64…95` 和 `96…127` 分别使用同样的 32 条 TMEM lanes。对于固定的 `sfk`，四个 `Mgroup` 对应的 TCol 分别是 `sfk`、`4+sfk`、`8+sfk` 和 `12+sfk`。这样，128 个逻辑元素先被打包到 32 条 lanes 和 4 个 TCol 位置中。

随后，为了让读取它的 warpgroup 中四个 warps 都能在自己的 32-lane TMEM window 中找到同一份 scale，硬件通过 `.warpx4` broadcast，把这个 32-lane 基础布局沿 `TLane` 轴复制四份。对于基础 lane `l`，同一个值会出现在 lanes `l`、`l+32`、`l+64` 和 `l+96` 中，而 column 保持不变。

这个例子的关键特征是：一个逻辑 scale factor 同时对应四个物理位置。前面的 shard mapping $f_D(x)$ 只能为逻辑元素 $x$ 给出一个基础位置，因此还需要一种方式表示这些与 $x$ 无关的额外副本。

### 用 Replication 捕获多个物理位置

Tile Layout 小节中的完整 layout 定义已经包含 replicated iters $R$。与由逻辑索引 $x$ 决定的 $D$ 不同，$R$ 使用独立的 replica coordinates 枚举额外偏移，因此 $f_L(x)$ 可以为同一个逻辑元素返回多个物理位置。记号 `R[shape : strides]` 用来书写这些 replicated iters；固定 offset $O$ 则只平移所有结果，不会增加位置的数量。

回到前面的 TMEM 例子，沿 `TLane` 轴的四份复制可以写成：

```text
S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]
```

这里的 `R[4 : 32@TLane]` 让 replica coordinate 依次取 `0`、`1`、`2` 和 `3`，产生 `0`、`32`、`64` 和 `96` 四个 `TLane` 偏移。它不产生新的逻辑数据，只说明同一个值在四个 warp 的 TMEM window 中各出现一次。

下图同时展示了这套打包映射，以及随后沿 `TLane` 轴的四份复制。

```{raw} html
<iframe src="../demo_zh/sf_tmem.html?v=warpx4-broadcast-20260710" title="Scale factors in TMEM: packing and .warpx4 broadcast" loading="lazy"
        style="width:100%; height:560px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击一个 scale factor，查看它的 TMEM 坐标以及分布在四个 warp window 中的副本。*

### GPU Mesh 中的 Replication 与 Offset

同一个 replication 结构也可以描述多设备布局。GPU mesh 是把多张 GPU 按一个或多个逻辑轴组织起来的设备网格；例如，一个 `2×2` GPU mesh 包含四张 GPU，每张 GPU 都有一个由 `@gpuid_x` 和 `@gpuid_y` 组成的坐标。

先定义一个沿 `@gpuid_y` 分片的基础布局：

```text
base = S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)]
```

在这个基础上加入 replication：

```text
base + R[2 : 1@gpuid_x]

一个逻辑元素 → {gpuid_x = 0, gpuid_x = 1}
```

`R[2 : 1@gpuid_x]` 沿 `@gpuid_x` 轴生成两个相距 1 的位置。相比之下，加入固定 offset：

```text
base + 1@gpuid_x

一个逻辑元素 → gpuid_x = 1
```

只会把基础位置沿 `@gpuid_x` 平移 1，不会产生副本。下图将它们与 fully-sharded 布局放在一起比较，并支持在 fully-sharded、shard + replica 和 shard + offset 三种模式之间切换。

```{raw} html
<iframe src="../demo_zh/tile_distributed.html?v=offset-labels-20260710" title="Distributed layout across a GPU mesh" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*一个分布在 `2×2` GPU mesh 上的布局；点击一个 cell，查看哪些 device 持有它。*

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
