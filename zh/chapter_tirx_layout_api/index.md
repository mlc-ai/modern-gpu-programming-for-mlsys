(chap_tirx_layout_api)=
# TIRx Layout API

:::{admonition} 概览
:class: overview

- TIRx layout API 会把 {ref}`chap_data_layout` 中的布局记号变成编译器对象。主要对象是 `TileLayout`、`SwizzleLayout` 和 `ComposeLayout`。
- `TileLayout` 描述命名硬件轴上的 affine placement。它由 shard spec `S[...]`、replica spec `R[...]` 和可选 offset 构成。
- 一个布局会把一个逻辑坐标映射到一个或多个物理坐标。`layout.apply()` 会计算这个映射。
- `SwizzleLayout` 描述用于避免 bank conflict 的 XOR-based shared-memory swizzle。`ComposeLayout` 会把一个 swizzle 叠加到 tile layout 上。
- `tmem_datapath_layout`、`tcgen05_atom_layout` 和 `wg_local_layout` 这类现成 constructor 覆盖了 kernel 中反复出现的硬件布局。
:::

{ref}`chap_data_layout` 介绍了本书一直使用的记号：tile shape、命名轴上的一组 stride，以及可选的 replication term，用于表示被复制而不是被切分的值。本章会把这套记号变成编译器使用的 API。

目标是让页面上的记号和 kernel 里的代码几乎长得一样。当你写出这样的布局时：

```python
S[(128, 256) : (1@TLane, 1@TCol)]
```

你写的不只是解释文字。你正在构造一个可以挂到 buffer 上的 `TileLayout` 对象。之后，每个触碰这个 buffer 的 tile operation 都能从 layout 中读取它的 placement。Placement 只写一次、检查一次，并由编译器复用。

布局会在从 pool 分配时，或声明 buffer 时附着上去：

```python
pool.alloc(shape, dtype, layout=layout)

T.decl_buffer(shape, dtype, scope=scope, layout=layout)
```

从那时起，buffer 就携带了自己的物理 placement。Tile operation 不需要反复说明每个元素位于哪里。

这些 layout object 位于同一个模块中：

```python
from tvm.tirx.layout import (
    TileLayout,
    SwizzleLayout,
    ComposeLayout,
    S,
    R,
    laneid,
    warpid,
    tid_in_wg,
    TLane,
    TCol,
    m,
    tcgen05_atom_layout,
    tmem_datapath_layout,
)
```

这个 API 背后有一个核心思想：布局不一定把逻辑索引映射到单个物理地址。它会把逻辑索引映射到命名轴上的一组物理坐标。通常这组坐标只有一个元素。当存在 replication 时，同一个逻辑元素会有多个物理 placement。

这就是为什么 layout model 有三部分：shard、replica 和 offset。Shard 放置元素。Replica 把它复制到额外坐标。Offset 移动整个 placement。

## 通过例子看布局

下面的例子展示 API 的基本形状。

TMEM 中的 accumulator 可以写成覆盖 TMEM 轴的直接 placement：

```python
acc = TileLayout(S[(128, 256) : (1@TLane, 1@TCol)])
```

这里逻辑行映射到 `TLane`，逻辑列映射到 `TCol`。在 {ref}`chap_tmem` 中，硬件坐标称为 Lane 和 Col。在 TIRx 布局记号中，这些硬件轴写作 `TLane` 和 `TCol`。

Block-scaled MMA 的 scale-factor layout 会使用 replication：

```python
scale_factor_layout = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
)
```

Shard 会把一个 32-row group 放入 TMEM。Replica 以 32 个 lane 为 stride，把这个 group 重复四次，使这个 32-row group 在完整 128-lane TMEM 空间中都可见。

Tensor-core register fragment 可以分布在 lane 和 warp 上：

```python
frag = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
)
```

同一个物理轴可以出现多次。在这个例子中，两个不同 iter 都会贡献到 `laneid`。没有显式 axis 的 stride 使用默认 memory axis `m`。

在真实 kernel 中，常见硬件布局通常来自 constructor：

```python
acc = tmem_datapath_layout("D", 128, 256)

ld = tcgen05_atom_layout("32x32b", (128, 64), "float32")
```

这些 constructor 返回普通 `TileLayout` 对象。它们只是方便写法，不是单独机制。你可以检查返回的 layout，把它和其他 layout 组合，或者在 shape 不寻常时手写底层 `S[...]` 和 `R[...]` 形式。

## 交互 Demo

在进入机制之前，先有一个可以操作的具体对象会更容易理解。下面的 demo 允许你选择 preset layout、编辑 logical shape 和 `S` 或 `R` term、选择 dtype 和 swizzle mode，并点击某个 element 查看哪个物理坐标或哪些物理坐标拥有它。

```{raw} html
<p>
  <a class="reference external" href="../_static/tirx-layout-demo/index.html"
     target="_blank" rel="noopener"
     style="display:inline-block; padding:10px 18px; background:#3b82f6;
     color:#fff !important; font-weight:700; border-radius:8px;
     text-decoration:none;">▶ 全屏打开 demo ↗</a>
</p>
<iframe id="tirx-layout-demo-frame" src="../_static/tirx-layout-demo/index.html?notitle"
        style="width:100%; height:1040px; border:1px solid #dfe1e6;
        border-radius:10px; margin:10px 0 6px; display:block;"
        title="TIRx interactive layout demo" loading="lazy"></iframe>
<script>
// The demo (viz-base.js) posts its content height; size the iframe to fit so
// there is no inner scrollbar. This demo is responsive (fills the width), so
// only the height follows content.
(function () {
  var f = document.getElementById('tirx-layout-demo-frame');
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    if (f && e.source === f.contentWindow) f.style.height = d.height + 'px';
  });
})();
</script>
```

这个 demo 很有用，因为 API 的大部分内容只是把 demo 中展示的过程精确定义出来。一个 logical element 进入 layout。Layout 会把它 flatten，按 iter 拆分，在命名轴上累加坐标，然后在需要时应用 replication。

## TileLayout

`TileLayout` 是主要的 affine layout object。它通常用正文中同样的记号写出：

```python
TileLayout(S[shape : strides])
```

`S` term 是 shard spec。可以这样读：取一个具有这个 shape 的逻辑 tile，并用这些命名轴 stride 放置它。

当某个值需要出现在多个地方时，shard spec 会扩展出 replica spec：

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride])
```

还可以加入可选 offset：

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride] + offset)
```

在表面之下，这些部分由 iter 表示。一个 iter 是三元组：

```text
(extent, stride, axis)
```

它描述命名轴上的一次 strided walk。Extent 表示这个 iter 有多少个位置。Stride 表示每一步移动多远。Axis 表示被改变的是哪个硬件坐标。

一个 layout 有三部分。

### Shard

Shard，也就是 `D`，是由 `S[...]` 构建的部分。它把逻辑索引切分到一个或多个 iter 上，并产生 base physical coordinate。

例如：

```python
S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
```

有四个 shard iter。它们的 extent 是 `8`、`2`、`4` 和 `2`。它们的 stride 会把数据放到 `laneid`、`warpid`、再次 `laneid`，以及默认 memory axis `m` 上。

这推广了普通 shape-and-stride 规则。差别在于，这里的 stride 附着在命名硬件轴上，而不是单个 flat address 上。

### Replica

Replica，也就是 `R`，描述同一个逻辑元素的额外物理副本。Replica iter 与逻辑索引无关。它们枚举硬件空间中的额外 offset。

例如：

```python
R[2 : 4@warpid]
```

会在 `warpid` 轴上创建两个相距四个 warp 的副本。

Replication 不是为了方便写法的技巧。它描述真实硬件行为。有些数据会 broadcast 到多个 warp、lane 或 memory region。Logical-to-physical mapping 很自然地支持这一点，因为一个逻辑元素可以映射到一组物理坐标。

### Offset

Offset，也就是 `O`，是加到每个结果上的固定坐标。

例如：

```python
5@warpid
```

会把整个 placement 在 `warpid` 轴上移动 5。

Offset 用于把 tile 放到选定 base coordinate、为独占使用预留区域，或者描述在同一资源中位于另一个 tile 之后开始的 tile。

### 把三部分合在一起

Layout 会按顺序应用这三部分。

首先，shard 计算 base coordinate。然后，replica 把这个 coordinate fan out 成零个或多个额外副本。最后，offset 移动每个 coordinate。

对于逻辑坐标 `x`，结果是：

```text
L(x) = { D(x) + r + O | r in R }
```

如果没有 replica，`R` 只包含 zero offset，因此结果是 singleton set。如果有 replica，结果中会有每个 replica position 对应的一个 coordinate。

在 TIRx 语法中，一个完整 layout 可以写作：

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

从左到右读，shard 放置逻辑 tile，replica 在四个 warp ID 之外创建第二份副本，offset 把整个 placement 移动到从 `warpid = 5` 开始。

如果 iter 已经被构造成对象，同一个 layout 也可以直接构造：

```python
TileLayout.from_iters(shard, replica, offset)
```

大多数用户代码使用 `S[...]` 和 `R[...]` 记号，因为它更接近数学形式。

## 命名轴

Layout 中的轴不是匿名维度。每个轴都命名一个真实硬件坐标，或者一个编译器级 placement 坐标。

例子包括：

```text
bx, by, bz
cbx, cby, cbz
tx
warpid
laneid
wgid
tid_in_wg
wid_in_wg
m
P, F
Bank
TLane, TCol
```

`bx`、`by`、`bz` 这样的 grid axis 把 work 放到 CTA 上。`cbx`、`cby`、`cbz` 这样的 cluster axis 把 work 放到 CTA cluster 内部。`tx`、`warpid`、`laneid`、`tid_in_wg` 和 `wid_in_wg` 这样的 thread axis 描述 CTA 或 warpgroup 内的 ownership。`m` 是默认 linear memory axis。`P` 和 `F` 用于二维 scratchpad-style placement。`Bank` 命名 shared memory bank。`TLane` 和 `TCol` 是 TIRx 布局中对 TMEM Lane 和 Col 坐标的命名。

Axis name 是 layout 的一部分。这一点很重要，因为两个整数值相同的坐标可能表示不同硬件事物。`1@tx` 不是 `1@tid_in_wg`。`1@laneid` 不是 `1@TLane`。Layout 会让这些含义保持显式。

## Forward Mapping

求值一个 layout，意味着取一个逻辑坐标并计算它物理上落在哪里。API 方法是：

```python
layout.apply(*coord)
```

对于没有 replication 的 layout，结果是一个 coordinate dictionary。带有 replication 时，结果是一组 coordinate dictionary。Coordinate dictionary 把 axis name 映射到整数位置，例如：

```python
{"laneid": 7, "warpid": 2, "m": 1}
```

求值规则有四步。

第一步，按 row-major 顺序 flatten 逻辑坐标。对于逻辑 shape：

```text
(S0, S1, ..., Sr-1)
```

中的逻辑坐标：

```text
x = (x0, x1, ..., xr-1)
```

flat index 是：

```text
flat = x0 * S1 * S2 * ... * Sr-1
     + x1 * S2 * ... * Sr-1
     + ...
     + xr-2 * Sr-1
     + xr-1
```

第二步，把这个 flat index 按 shard extents 拆分。如果 shard extents 是：

```text
(e0, e1, ..., en-1)
```

那么拆分会产生 components：

```text
c0, c1, ..., cn-1
```

使用的是同样的 row-major 顺序，只不过作用在 shard extents 上。

第三步，用每个 component 的 stride 把它累加到对应 axis 上。如果 shard iter `k` 的 extent 是 `ek`、stride 是 `sk`、axis 是 `ak`，那么 component `ck` 的贡献是：

```text
ck * sk @ ak
```

对同一个 axis 的所有贡献会相加。然后加入 offset。

第四步，应用 replica iter。每个 replica iter 都贡献一个与逻辑坐标无关的额外 offset。如果有多个 replica iter，layout 会枚举所有组合。

这个规则有一个有用结果：layout 不需要 hard-code 输入 shape。它需要的是逻辑 tile 的元素总数等于 shard extents 的乘积。只要这个条件成立，flattening 和 splitting 就定义了映射。

## Case Study：Tensor Core Register Tile

考虑一个逻辑 `(8, 16)` tile，它分布在两个 warp 上，每个 warp 有 32 个 lane。每个 lane 拥有一个小 register fragment。Register slot 用默认 memory axis `m` 表示。

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

取 `(8, 16)` tile 中的一个逻辑元素 `(i, j)`。

Row-major flat index 是：

```text
flat = 16 * i + j
```

按 shard extents `(8, 2, 4, 2)` 拆分得到：

```text
c0 = i
c1 = floor(j / 8)
c2 = floor(j / 2) mod 4
c3 = j mod 2
```

Shard 贡献是：

```text
laneid = 4 * c0 + c2
warpid = c1
m      = c3
```

加上 offset `5@warpid` 后变成：

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5
m      = j mod 2
```

Replica term：

```python
R[2 : 4@warpid]
```

会向 `warpid` 添加 `0` 或 `4`。因此完整映射是：

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5 + 4 * r, where r in {0, 1}
m      = j mod 2
```

Shard 把 tile 放到 warps 5 和 6 上。Replica 随后把它复制到 warps 9 和 10。于是同一个逻辑元素会出现在两个 warp 位置。

这个例子说明了为什么模型使用一组物理坐标。Replication 很难自然表示成从物理坐标到逻辑坐标的函数。它更自然地表示成从一个逻辑坐标到多个物理坐标的函数。

## Case Study：Blackwell Tensor Memory

同一个 layout model 也适用于内存 placement。Axis 不一定是 thread axis，也可以是 memory axis。

TMEM 通过硬件 Lane 和 Col 坐标寻址。在 TIRx 布局记号中，这些轴写作 `TLane` 和 `TCol`。

考虑这个 layout：

```python
layout = TileLayout(
    S[(2, 128, 112) : (112@TCol, 1@TLane, 1@TCol)]
)
```

如果逻辑 tile shape 是 `(2, 128, 112)`，split component 就是逻辑坐标本身。对于元素 `(a, l, c)`，映射是：

```text
TLane = l
TCol  = 112 * a + c
```

Extent-128 iter 以 stride `1@TLane` 填充 128 个 TMEM Lane row。Extent-2 iter 以 stride `112@TCol`，extent-112 iter 以 stride `1@TCol`，两者共同覆盖 224 列：

```text
TCol in [0, 224)
```

224-column span 是有意选择的。TMEM layout 不一定是 2 的幂。Block-scaled FP8 GEMM 可能选择 224-column accumulator，因为完整 256-column tile 可能无法为两个 accumulator stage 加 scale factor 留出足够 TMEM 容量。Layout API 可以直接表达这种 shape。

## Scale Factor Layout

上面的 accumulator layout 是纯 placement。每个逻辑 accumulator element 映射到一个 TMEM coordinate。Block-scaled MMA 的 scale factor 不同，因为同一个物理 group 可能需要在多个 warp window 中可见。这正是 replication 有用的地方。

一个紧凑 scale-factor layout 可以写作：

```python
scale = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)]
    + R[4 : 32@TLane]
)
```

Shard 把一个 32-row scale-factor group 放入 TMEM：

```text
TLane = r
TCol  = s
```

对于逻辑 scale coordinate `(r, s)`。

Replica term 会创建四份副本，间隔 32 个 lane：

```text
TLane = r + 32 * q, where q in {0, 1, 2, 3}
TCol  = s
```

所以 32-row group 会在 TMEM lanes 0 到 31、32 到 63、64 到 95 和 96 到 127 处可见。这就是 `warpx4` broadcast pattern（{ref}`chap_layout_generations`）。四个 warp-sized TMEM lane window 中的每一个，都能看到同一个 scale-factor group。

在完整 block-scaled MMA layout 中，这个 atom 会与 M row 和 K scale-factor group 上的 outer iter 组合在一起。多个 scale factor 也可能被打包到同一个 32-bit `TCol` cell 中，具体取决于 scale-factor dtype。例如，fp8 scale factor 可以把四个值打包到一个 32-bit column cell 中。可选的 stride-zero reuse 和 pipeline-depth iter 可以进一步描述多个 MMA 之间的 scale reuse 以及 double buffering。

重要的是，同一个 `TileLayout` model 描述了两种情况。Accumulator 是 TMEM 中的单一 placement。Scale factor 是同一 TMEM 地址空间中的 replicated placement。

## 现成 Layout

大多数 kernel 不会手写每一个硬件布局。TIRx 为常见布局提供了 constructor。

```python
tmem_datapath_layout(datapath, rows, cols)
```

返回 `tcgen05.mma` 写出的 TMEM accumulator layout。`datapath` 参数选择 row placement pattern。例如，`"D"` 对应 `M = 128` identity-style placement，而 `"F"` 对应 `M = 64` scattered placement。

```python
tcgen05_atom_layout(instr_shape, tensor_shape, dtype)
```

返回由 `tcgen05.ld` 或 `tcgen05.st` atom 移动的 register tile layout。Instruction shape 的例子包括 `.32x32b`、`.16x64b`、`.16x128b` 以及相关形式。在 DSL 层面，这是一个 warpgroup-distributed tile。Lowering 时，它会变成四条 warp-collective `tcgen05.ld` 或 `tcgen05.st` 指令，每个 warp 一条，并且每个 warp 处理自己的 32 个 TMEM lane。

```python
wg_local_layout(cols, rows=128)
```

返回一个 warpgroup-local register tile，通常在 `tid_in_wg` 上每个线程对应一行。

这些 helper 的作用是避免手写常见硬件映射。它们不会隐藏模型。每个 helper 都返回一个普通 `TileLayout`，由上面描述的同一组 `S` 和 `R` 片段构成。

## SwizzleLayout 和 ComposeLayout

`TileLayout` 是 affine 的。它可以在命名轴上表达 stride、replication 和 offset。这足以描述很多 placement，包括 thread fragment、TMEM tile 和紧凑 scale-factor layout。

Shared memory swizzle 需要另一种东西。用于避免 bank conflict 的 swizzle 不是 affine stride pattern。它是对线性 shared-memory address 的 XOR-based permutation。

因此，TIRx 把 swizzling 保持为一个单独的 layout object：

```python
SwizzleLayout(...)
```

并把它与 tile layout 组合：

```python
ComposeLayout(swizzle, tile)
```

Tile layout 先产生线性 memory address。Swizzle 随后重排这个 address。把两层分开，比强行把 XOR permutation 塞进 affine layout model 更干净。

## 为什么需要 Swizzle

Shared memory 分成 32 个 bank，每个 bank word 保存 4 byte。当一次访问中的多个 lane 触碰同一 bank 中的不同地址时，访问会因为 bank conflict 被串行化。

普通 row-major tile 会结构性地产生这种 conflict。考虑一个 row-major layout 的 `(8, 64)` float16 tile：

```python
TileLayout(S[(8, 64) : (64@m, 1@m)])
```

逻辑元素 `(i, j)` 的线性 element address 是：

```text
m = 64 * i + j
```

每行有 64 个 float16 值，也就是 128 byte。这刚好是一整条 shared memory bank line。如果一个 warp 沿固定 `j` 读取 column，每向下一行都会前进完整 128-byte line。Bank index 重复，因此 column read 会跨行塌缩到同一个 bank 上。

Swizzle 通过让低地址位依赖更高的 row bit 来改变这一点。原本会反复落到同一个 bank 的 column 会被分散到不同 bank 上。

## Swizzle Transform

`SwizzleLayout` 由三个整数参数控制：

```text
per_element = M
swizzle_len = B
atom_len    = S
```

输入是一个线性 element address `m`。

`m` 的低 `M` 位保持不变。这会保留一个小的连续 element group。更高的 bit 会被右移到一个临时值中：

```text
x = m >> M
```

然后，`x` 中位于 `[S, S + B)` 的 bit group 会 XOR 到 `x` 中位于 `[0, B)` 的 bit group 上。最后把保持不变的低 `M` bit 放回去，形成 swizzled address。

等价地：

```text
mask = (1 << B) - 1

low  = m & ((1 << M) - 1)
x    = m >> M
x2   = x ^ ((x >> S) & mask)

addr = (x2 << M) | low
```

要让 layout well formed，`S` 必须至少为 `B`。

这个 transform 的目的不是改变 tile 中有哪些逻辑元素。它改变的是这些元素落在 shared memory 中的位置。MMA 仍然读取同一个逻辑 tile。Swizzle 让物理 bank pattern 更好。

## 选择 Swizzle 参数

正常使用时，swizzle 参数由 dtype 和 shared-memory swizzle mode 共同决定。常见 mode 是 32-byte、64-byte 和 128-byte swizzle。

`per_element` 参数的选择要保证一个小的 vector-sized group 保持连续。对于 float16，一个 16-byte vector 包含 8 个元素，因此：

```text
M = log2(8) = 3
```

使用 128-byte swizzle 时，layout 使用：

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

这会保持 16-byte vector group 完整，同时仍然充分重排更大的 shared-memory address pattern，以打破 column bank conflict。

大多数代码不应该手工推导这些参数。Dtype 和 descriptor mode 通常会决定它们。对程序员来说，重要的是 TIRx layout 中的 swizzle、TMA descriptor 和 MMA 期望三者保持匹配。

一个 swizzled shared memory allocation 因此会写成：

```python
tile = TileLayout(S[(8, 64) : (64@m, 1@m)])
swizzle = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

layout = ComposeLayout(swizzle, tile)
```

组合后的 layout 会附着到 shared memory buffer 上。

## 元素的 Bank 和 Line

要判断 swizzle 是否有帮助，可以把 swizzled element address 转回 shared memory bank。

设 `addr` 为 swizzled element address，`b` 为 element size in bytes。Byte address 是：

```text
byte = addr * b
```

Bank 是：

```text
bank = floor(byte / 4) mod 32
```

128-byte bank line 是：

```text
line = floor(byte / 128)
```

对于 float16，`b = 2`，因此 bank 公式变成：

```text
bank = floor(addr / 2) mod 32
```

这就是下面 worked example 使用的公式。

## Worked Example：`(8, 64)` float16 Tile 上的 128B Swizzle

回到 row-major float16 tile：

```text
m = 64 * i + j
```

使用：

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

Transform 变成：

```text
x    = m >> 3
addr = ((x ^ ((x >> 3) & 7)) << 3) | (m & 7)
```

因为：

```text
m = 64 * i + j
```

我们可以写：

```text
q = floor(j / 8)
r = j mod 8
```

swizzled address 是：

```text
addr = 64 * i + 8 * (q xor i) + r
```

现在看 column `j = 0`。此时 `q = 0` 且 `r = 0`，因此：

```text
addr = 72 * i
```

对于 float16，bank 是：

```text
bank = floor(addr / 2) mod 32
```

所以八行映射到：

```text
i = 0: bank 0
i = 1: bank 4
i = 2: bank 8
i = 3: bank 12
i = 4: bank 16
i = 5: bank 20
i = 6: bank 24
i = 7: bank 28
```

这个 column 现在触碰八个不同 bank。Conflict 消失了。

如果没有 swizzling，同一个 column 的地址是：

```text
m = 64 * i
```

因此：

```text
bank = floor(64 * i / 2) mod 32 = 0
```

每一行都落在 bank 0 上，所以访问会被串行化。Swizzle 只改变物理 placement，但这已经足以把 column access 变成 conflict-free。

这个保证依赖于按设计方式使用 swizzle。Dtype、swizzle width 和 access shape 必须匹配 TMA 和 MMA descriptor mode。128-byte float16 swizzle 围绕相关的 16-byte row chunk 和 Tensor Core access pattern 设计。它并不承诺任意 shared memory access 都会变成 conflict free。本章开头的 demo 可以看到这一点：选择 dtype 和 swizzle mode，观察无 swizzle 时一个 column 如何塌缩到一个 bank 上，再观察匹配 swizzle 应用后它如何分散到 bank view 中。

## 设计理由

Layout API 遵循三个设计选择。

第一，它支持通用 shape。硬件 tile 并不总是 2 的幂。Global tensor、shared memory stage、TMEM accumulator 和 scale-factor buffer 的 shape 常常来自容量限制或算法选择。Layout model 把这些 shape 当作正常情况处理。

第二，映射方向是从逻辑坐标到物理坐标。这个方向很重要，因为 replication 很常见。一个逻辑元素可能位于多个物理位置。Logical-to-physical map 可以直接把它表示成一组坐标。

第三，硬件轴是显式的。Layout 不使用匿名维度，再依赖上下文在事后解释它们。`tx`、`tid_in_wg`、`laneid`、`warpid`、`TLane` 和 `TCol` 之间的差异直接写在 layout 里。

Legality 和 feasibility 检查不只由 layout object 自己负责。Layout 可以说明数据放在哪里。更高层的 tile primitive 会决定某个操作能否合法且高效地使用这个 placement。这种分离让 layout API 保持小而清晰，同时仍然给编译器足够信息去 dispatch 真实硬件操作。
