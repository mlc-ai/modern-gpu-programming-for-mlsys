(chap_data_layout)=
# 数据布局及其记号

:::{admonition} 概览
:class: overview

- *数据布局*把 tensor 的逻辑索引映射到物理位置，并决定 coalescing、bank conflict，以及某个硬件引擎是否能读这个 tile。
- 本书用一套统一记号描述布局：`S[(shape) : (strides)]`，并配合命名轴（`@laneid`、`@TLane` 等）以及用于 broadcast 或复制数据的 replication term `R[...]`。
- Swizzle 是一种基于 XOR 的地址重映射，用来消除 shared memory bank conflict。
:::

同一组数值，如果以不同的物理排列写入内存，在同一块 GPU 上的运行速度可能相差一个数量级。

原因在于，tensor 的逻辑索引本身并不说明它的 byte 实际放在哪里。硬件对这个位置非常敏感：它决定了 32 个 lane 的 load 是合并成一次 transaction，还是分散成 32 次；决定了这些地址是落在不同 memory bank 上，还是撞到同一个 bank 后被串行化；甚至还决定了一个 tile 是否符合 Tensor Core 能够读取的 byte 排列。

机器学习程序通常用逻辑 shape 来描述 tensor。**数据布局**补上缺失的物理部分：它说明带有逻辑索引 `(i, j, …)` 的元素位于哪里，不论这个位置是在内存中、寄存器中，还是其他硬件存储中。

本章介绍现代 GPU 编程中会遇到的主要布局。为了让讨论可控，我们会发展一套紧凑的**记号**，用它描述机器学习系统中不同场景下的布局。最后，我们会讨论 **swizzling**：一种让同一个 tile 的 row-wise 和 column-wise 访问都高效的机制。

## Shape-Stride 模型

在进入 GPU-specific 的布局之前，值得先从最简单的布局开始，因为本章后面的内容都建立在它之上。布局的核心只有两件事：一个 **shape**，以及一组与之匹配的 **stride**。我们把这一对写作 `S[(shape) : (strides)]`。要找到某个逻辑索引的位置，只需要把这个索引与 strides 做点积。例如，一个 row-major 的 4×4 矩阵可以写成：

```text
S[(4, 4) : (4, 1)]        addr(i, j) = i·4 + j·1
```

这只是经典 shape/stride 模型的紧凑写法（可以看作 CuTe 记号的 row-major 简化版），后面所有内容都从它扩展而来。

事实上，你几乎肯定已经用过这个模型。任何写过 PyTorch 或 NumPy 的人都用过，因为这些库中的 tensor 本质上就是一个 shape，加上 flat storage buffer 上的一组 stride：

```python
import torch
t = torch.arange(12).reshape(3, 4)
t.shape        # torch.Size([3, 4])
t.stride()     # (4, 1)        ← exactly S[(3, 4) : (4, 1)]
```

一旦从这个角度看 tensor，很多 “reshape” 操作为什么完全不移动数据就很清楚了。它们只是重写 strides，并返回同一块 storage 上的一个 **view**。最清楚的例子是 transpose 或 permute：

```python
tt = t.permute(1, 0)               # or t.T
tt.shape                           # torch.Size([4, 3])
tt.stride()                        # (1, 4)        ← strides swapped, no data moved
tt.data_ptr() == t.data_ptr()      # True, same bytes
```

这里 `t.permute(1, 0)` 是同一块内存上的 `S[(4, 3) : (1, 4)]`：transpose 只是 stride 改变，没有移动任何 byte。对 contiguous tensor 上的 `reshape` 或 `view` 也是同样道理：旧 storage 上的新 shape 和新 stride。（NumPy 的行为也相同；唯一差别是它的 `.strides` 以 byte 而不是 element 为单位。）

GPU 上的布局也是这样工作的。本章剩下的内容本质上是一系列围绕同一思想的变体：一个 tile 的映射，不论是映射到内存，还是通过稍后介绍的命名轴映射到 lane 和寄存器，都是固定 buffer 上的 stride 规则。因此，重排 tile 通常是改变 *layout*，而不是 copy。不过，我们也要小心这个推理的边界。零拷贝的说法只对单个线性地址空间上的逻辑 view 清晰成立；在 GPU 上，它只在新的 view 与已有 byte 排列和 ownership 安排兼容时成立。一旦改变哪个线程或寄存器拥有某个元素，或者改变 SMEM swizzle，通常就需要真正的数据搬运：load、store、shuffle、`ldmatrix`、transpose。

## Tile Layout

到目前为止，我们描述的是整个 tensor 的布局。但 GPU kernel 很少一次性操作整张矩阵；它们会处理较小的 tile，这些 tile 会被加载、变换，并由不同硬件部分参与计算。好消息是，tiling 不需要新概念。它仍然只是布局，只不过多了几个维度。把一个 8×8 矩阵切成 2×4 tile，我们会得到一个 4-D 布局，坐标是 `(tile_row, row_in_tile, tile_col, col_in_tile)`，stride 的选择让每个 tile 保持连续：

```text
S[(4, 2, 2, 4) : (16, 4, 8, 1)]
```

逻辑 `(i, j)` 会先变成 `(i//2, j//4, i%2, j%4)`，再经过 strides 计算地址。值得注意的是，这套记号并没有引入任何特殊的 “tile” 概念，就表达了 tiling：它仍然是同一个 shape-stride 模型，只是把索引拆成了 outer 和 inner 坐标。

下面的交互图展示了一个逻辑矩阵索引如何分解成 tile 坐标，再映射到物理地址。

```{raw} html
<iframe src="../demo_zh/tiled_layout.html" title="Tile layout: interactive address computation" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互图：点击一个 cell，查看它的 tiled index 和 address。*

## 命名轴

到目前为止，`S[...]` 中的每个 stride 都表示线性内存中的 offset，我们也把 address 当作这个空间中的位置。但在 GPU 上，数据可以位于不止一个地方：除了内存，一个 tile 还可能分布在 warp lane 上、线程寄存器上，或者 TMEM lane 与 column 上。为了统一描述这些情况，我们给记号扩展出**命名轴**。核心想法是让每个 stride 系数携带一个 axis tag，说明它穿过的是哪个空间：`@m` 表示普通内存，`@laneid` 表示 warp lane，`@reg` 表示寄存器，`@warpid` 表示 warp，`@TLane` / `@TCol` 表示 TMEM 坐标。有了这些 tag，单个布局不仅能描述数据在内存中的位置，也能描述它如何分布在实际操作这些数据的硬件资源上。

把 memory tag 显式写出来之后，内存中的一个 row-major 8×16 tile 就是：

```text
S[(8, 16) : (16@m, 1@m)]
```

当布局描述的数据是*分布在线程之间*，而不是放在线性内存中时，这些 tag 就开始发挥作用。例如 `S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]` 并不指向线性内存，而是把行列映射到 lane ID 和每个 lane 的寄存器上。这里 `laneid` 表示 warp 内的 lane index，也就是 `thread_index % warp_size`。这正是你会在 {ref}`chap_layout_generations` 中遇到的 tensor-core register fragment。

下面的交互图展示了一个布局如何把 tensor element 分布到 warp lane 和 per-lane register 上，而不是放进线性内存中。

```{raw} html
<iframe src="../demo_zh/thread_register.html" title="Thread + register layout via named axes" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互图：一个覆盖 `@laneid` 和 `@reg` 的布局；点击 cell 可以看到哪个 lane/register 持有它。*

## 分布式布局

命名轴最有用的地方在于，它们让我们可以统一描述系统中很多层级上的 placement，包括*跨设备*的 placement。刚才我们用它们描述单个 GPU 内部的 lane 和 register，但同样的思想可以继续向外延伸：`@gpuid_x` 和 `@gpuid_y` 这样的轴可以说明数据位于 GPU mesh 中的哪里。有了这些轴，记号就能捕捉分布式训练和推理中出现的 sharding pattern。不过，这些轴还没有描述*复制*，也就是被拷贝到多个位置的数据。因此我们加入 `R[n : stride]` 记号，其中 `R` 标记 replicated dimension。例如，`R[2 : 1@gpuid_x]` 表示沿 `@gpuid_x` 轴复制。把两者合在一起，单个表达式就能同时描述 tensor 在 2×2 GPU mesh 上的 sharding，以及沿一个轴的 replication：

```text
S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)] + R[2 : 1@gpuid_x]
```

下面的 demo 展示了一个小 GPU mesh 上 partition-and-replication 的组合模式。点击任意 cell 可以看到哪个 device 持有它，并观察 `@gpuid_x` replication 如何把同一个副本放到配对 device 上；按钮可以在 fully-sharded、shard + replica 和 shard + offset 布局之间切换。

```{raw} html
<iframe src="../demo_zh/tile_distributed.html" title="Distributed layout across a GPU mesh" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互图：分布在 2×2 GPU mesh 上的布局；点击 cell 可以看到哪些 device 持有它。*

### Kernel 内复制模式：TMEM 中的 Scale Factor

我们刚才为 GPU mesh 引入的 replication dimension `R[...]`，不只适用于多设备。事实证明，同一个结构也可以描述单个 kernel 内部发生的事情：硬件*跨 lane broadcast* 的数据。Blackwell 的 block-scaled MMA（{ref}`chap_layout_generations`）就是很好的例子。它的 scale factor 位于 TMEM 中，其中一个 128-row scale vector 只存储在 **32 个 TMEM lane** 中，逻辑行 `r` 映射到 TMEM lane `r % 32`，而 `r // 32` 沿 column 方向前进。随后这 32 个存储的 TMEM lane 会沿 TMEM `TLane` 轴**复制**，从 32 个 lane 扩展到 128 个 lane，这样读取 warpgroup 中的四个 warp 都能在自己的 32-lane TMEM window 中找到一份副本。这是一个 `warpx4` broadcast，我们用 replication dimension 写出它。读取本身由这些 warp 的线程执行：

```text
S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]
```

这会在 32 个 TMEM lane 的 stride 上产生四份 replica：TMEM lane `l`、`l+32`、`l+64` 和 `l+96` 都保存同一个 scale。和前面一样，replication dimension 不携带新数据；它只是说明“同一个值位于四个 TMEM-lane 位置”，就像刚才 `@gpuid_x` 在 GPU mesh 上 broadcast 一行一样。

下面的交互 demo 同时展示两步：先紧凑打包到 32 个 TMEM lane 中，再通过 `warpx4` broadcast 扩展到 128 个读取 lane。

```{raw} html
<iframe src="../demo_zh/sf_tmem.html" title="Scale factors in TMEM: packing and warpx4 replication" loading="lazy"
        style="width:100%; min-width:1040px; height:560px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互图：点击一个 scale factor `SFA[m, sf]`；它会被打包到 TMEM lane `m mod 32`、column `(m // 32)·4 + sf`，随后沿 `TLane` 轴做 `warpx4` broadcast，产生四个 lane copy（`l`、`l+32`、`l+64`、`l+96`），每个 warp 的 32-lane window 各一份。*

每个 column 内的 byte packing（`scale_vec` 的 1X/2X/4X 模式）以及 `cta_group::2` split 会在 {ref}`chap_layout_generations` 中介绍。

已经熟悉 CuTe 的读者可以把本章记号理解为 CuTe 的 row-major 变体，并额外扩展了显式硬件命名轴以及专用 replication 结构。

## Swizzle Layout

本章最后一种布局是为了解决一个具体硬件问题。GPU 上的 shared memory 被组织成多个 memory bank。当不同 lane 落在不同 bank 上时，访问最快。反过来，如果多个 lane 访问同一个 bank 内的不同地址，硬件只能把它们串行化，我们就要付出 **bank conflict** 的代价。

在 tensor 程序中，这很难避免，因为内存访问并不总是纯线性顺序。处理矩阵时，我们经常需要从同一个 tile 中读取 row slice 和 column slice，这会产生真实的张力：对 row-wise 访问高效的布局，通常会让 column-wise 访问产生 bank conflict；而偏向 column 的布局又会伤害 row 访问。**Swizzling** 就是为了解开这个张力而设计的技术。

Swizzle 的想法是重排地址映射，通常是把 column index 与 row 做 XOR，使得 *row* 和 *column* 两类访问最终都分散到不同 bank 上。它提供的 conflict-free 保证是有条件的：只对匹配的 element width、swizzle mode 和 access pattern 成立，也就是对应 engine descriptor 所期望的模式；它并不保证任意 element width 或 alignment 都无冲突。

第一个交互 demo 让这个过程具体化。点击一个 column index，观察每个元素落到哪个 bank：左侧普通 row-major tile 中，一个 column 会把 8 个元素全部汇入同一个 bank，因此读取会串行化成 8 个 cycle；右侧 XOR-swizzled layout 中，同一个 column 被分散到 8 个不同 bank，可以在 1 个 cycle 中读完。

```{raw} html
<iframe src="../demo_zh/swizzle_8x8.html" title="8x8 XOR swizzle" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互图：一个 8×8 tile，普通 row-major 按列读取会产生 bank conflict，XOR swizzle 后变成 conflict-free。*

这个小的 8×8 例子抓住了核心思想，但真实 GPU 内存的 bank 数远多于这个玩具图。为了让 swizzling 在完整尺度上工作，我们不会把整个 tile 当作一个单块对象处理。相反，我们会把内存切成小 segment，并在每个 segment 内应用 swizzle pattern。实践中最常见的是 `SWIZZLE_128B`，围绕 128-byte segment 组织，使同样的 row/column 重映射技巧自然适配 32-bank memory system。

下面的交互 demo 展示一个具体硬件 swizzle：`SWIZZLE_128B`。在我们推广到其他格式之前，可以先看到这个逐 segment 重复的 pattern。

```{raw} html
<iframe src="../demo_zh/swizzle_128B.html" title="SWIZZLE_128B layout" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互图：128-byte segment 内的 `SWIZZLE_128B` pattern；逐 cycle 查看读取过程，可以看到 `physical_sector = logical_sector XOR row` 如何把每个 column 分散到不同 bank。*

同一个思想也可以扩展到 128-byte 之外。为了简化可视化，我们接下来会用单个彩色 block 表示一个 segment，而不是画出每个 bank。一般来说，硬件会定义一个小的重复 **atom**，在 atom 内应用 permutation；不同 swizzle mode 选择不同的 atom 大小。`SWIZZLE_128B` 使用 8 × 128 B atom，`SWIZZLE_64B` 使用 8 × 64 B atom，`SWIZZLE_32B` 使用 8 × 32 B atom；整个 tile 再由所选 atom 平铺而成。

最后一个交互 demo 可以在这些格式之间切换（也包括一种 16 B interleaved mode），选择数据类型，并把鼠标悬停在任意 cell 上直接查看一个 atom 内部的 element 排列。对于判断某条 load/store 指令期望哪种 swizzle，这正是合适的细节层级。

```{raw} html
<iframe src="../demo_zh/swizzle_atom_general.html" title="Swizzle atom layout per format (128B/64B/32B)" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互图：选择 swizzle format 和数据类型，查看它的 atom shape（8 × N B）；悬停在 cell 上可以看到其中的元素如何被 permutation。*

应该选择哪种 mode？经验法则是优先选择 tile 能填满的*最大* atom。一个 N-byte atom 要求 tile 的 contiguous dimension 至少有 N byte，并且是 N 的倍数，因此 `SWIZZLE_128B` 只有在一行至少跨越 128 byte，也就是 64 个 `float16` 元素时才适用。只要能适用，它就是默认选择，因为它的 8 × 128 B atom 覆盖一整条 128-byte bank line，可以一次把一个 column 分散到全部 32 个 bank，在 fp16 中同时给 8 行和 8 列提供 conflict-free 访问。但如果问题 shape 迫使 contiguous dimension 变小，tile 无法填满 128 B atom，就要降到 `SWIZZLE_64B` 或 `SWIZZLE_32B`，选择 row 仍然可以覆盖的最大 atom。

你不需要手工推导这些 permutation address。这里值得精确说明 swizzle 与 `S[...]` 记号的关系：swizzle *不是* affine map 的一部分。它是组合在 affine map 之上的一个单独的非仿射层。`S[...]` 布局把元素放到线性内存（`@m`）地址上，swizzle 再重排这个地址。在 TIRx layout API 中，这写作 `ComposeLayout(swizzle, tile)`（{ref}`chap_tirx_layout_api`）。你的工作只是为所有会触碰这个 tile 的 op 选择一致的 mode，然后让组合布局处理剩下的事。

硬件填充的也是同一个组合布局，这正是 swizzling 与 tiling 汇合的地方。TMA descriptor 是多维的，因此一个三维 box 可以同时描述 tile 的 atom tiling 以及每个 atom 内部的 swizzle；一次 TMA load 随后会 atom by atom 地布置 tile，并在写 shared memory 时应用 swizzle（{ref}`chap_tma`），不需要单独的 swizzling pass。每个 engine 需要*哪一种* swizzle 是 generation-specific 的，这就是下一章的主题。
