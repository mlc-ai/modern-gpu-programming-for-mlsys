(chap_tirx_layout_api)=
# TIRx Layout API

:::{admonition} 概览
:class: overview

- `TileLayout` 使用 `S[...]`、`R[...]` 和 offset 描述逻辑 tile 在命名轴上的放置方式。
- `TileLayout.apply()` 计算逻辑元素的基础物理坐标；replica 信息保存在 `layout.replica` 中，由使用该 layout 的 tile 操作处理。
- `SwizzleLayout` 描述 shared memory 中基于 XOR 的地址重排；需要同时表达普通 tile layout 时，可以使用 `ComposeLayout` 组合两者。
:::

{ref}`chap_data_layout` 介绍了 tile shape、带命名轴的 strides、replication dimension 和固定 offset。本章接着说明如何在 TIRx 程序中构造、使用和查询这些 layout。

例如，下面的记号描述一个位于 TMEM 中的 `128×256` tile：

```python
S[(128, 256) : (1@TLane, 1@TCol)]
```

在 TIRx 程序中，可以直接用这个记号构造 `TileLayout`，再将它绑定到 buffer：

```python
layout = TileLayout(S[(128, 256) : (1@TLane, 1@TCol)])

pool.alloc(shape, dtype, layout=layout)

T.decl_buffer(shape, dtype, scope=scope, layout=layout)
```

Buffer 由此记录自己的物理布局。后续 tile 操作可以直接读取这项信息，不必在每次访问时重新说明元素位于哪些 lanes、registers 或线性存储位置。

本章使用的 layout 对象和命名轴都位于 `tvm.tirx.layout`：

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
    wg_local_layout,
)
```

Layout 的结果不一定是一个线性地址，也可以是 `laneid`、`warpid`、`TLane` 和 `TCol` 等硬件坐标。前面的 TMEM layout 将逻辑 row 映射到 `TLane`，将逻辑 column 映射到 `TCol`。再看一个同时使用 lane 和 warp 轴的 register fragment：

```python
frag = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
)
```

同一个物理轴可以出现多次。这里，第一个和第三个 iter 都会对 `laneid` 产生贡献。最后一个 stride 没有显式 axis tag，因此使用默认轴 `m`。

{ref}`chap_data_layout` 为了区分 lane 内的 fragment slots，使用过 `@reg` 这个记号。当前 TIRx API 没有单独注册 `reg` axis；当 layout 绑定到 register-backed local buffer 时，默认轴 `m` 表示该 thread 的局部线性位置。Buffer scope 决定数据实际位于 registers 中，所以这里的 `m` 并不表示数据存放在 global memory 或 shared memory。

对于 `m` 和 `TCol` 这类存储轴，stride 以 buffer element 为单位。对于 32-bit TMEM buffer，沿 `TCol` 前进一个元素就对应一个 32-bit hardware Col；对于 8-bit 或 16-bit buffer，多个相邻元素会打包进同一个 hardware Col。后面的 scale-factor 示例会具体说明这一点。

## 交互图

下面的交互图提供几种常用 layout preset。可以修改逻辑 shape、`S[...]` 或 `R[...]`，选择 data type 和 swizzle mode，再点击一个逻辑元素，查看它对应的物理坐标。

```{raw} html
<p>
  <a class="reference external" href="../_static/tirx-layout-demo/index.html?lang=zh"
     target="_blank" rel="noopener"
     style="display:inline-block; padding:10px 18px; background:#3b82f6;
     color:#fff !important; font-weight:700; border-radius:8px;
     text-decoration:none;">在新窗口中打开交互图</a>
</p>
<iframe id="tirx-layout-demo-frame" src="../_static/tirx-layout-demo/index.html?notitle&lang=zh"
        style="width:100%; height:1040px; border:1px solid #dfe1e6;
        border-radius:10px; margin:10px 0 6px; display:block; box-sizing:content-box;"
        title="TIRx layout 交互图" loading="lazy"></iframe>
<script>
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

这张图也展示了 `TileLayout` 的基本计算过程：先将逻辑坐标 flatten，再按照各个 iters 的 extent 将扁平索引拆开；每个分量根据自己的 stride 和 axis 生成基础坐标，随后加入 offset。交互图还会枚举 replica，从而显示同一个元素的所有副本。

## TileLayout

`TileLayout` 是 TIRx 中主要的仿射布局对象，通常写成：

```python
TileLayout(S[shape : strides])
```

`S[...]` 是 shard spec。它给出一组 iter extents 和 strides，决定逻辑 tile 如何映射到命名轴上的基础物理位置。

同一个值需要出现在多个物理位置时，可以加入 replica spec：

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride])
```

还可以再加入固定 offset：

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride] + offset)
```

在 API 内部，每个 iter 都是一个三元组：

```text
(extent, stride, axis)
```

`extent` 表示这个 iter 包含多少个位置，`stride` 表示每前进一步会移动多少，`axis` 则说明移动发生在哪个物理轴上。

### Shard

Shard 由 `S[...]` 构造。它将逻辑索引拆分到一个或多个 iters 上，并产生基础物理坐标。前面的 `frag` 包含四个 shard iters，extents 分别为 `8`、`2`、`4` 和 `2`；相应的 strides 将坐标映射到 `laneid`、`warpid`、再次映射到 `laneid`，以及默认线性轴 `m`。

这仍然是普通 Shape-Stride 规则，只是每个 stride 现在属于一个明确的命名轴，而不再全部累加到同一个线性地址。

### Replica

Replica 由 `R[...]` 构造，用于描述同一个逻辑元素的额外物理副本。Replica iters 不依赖逻辑索引，而是枚举物理空间中的额外 offsets。

例如：

```python
R[2 : 4@warpid]
```

表示沿 `warpid` 轴放置 2 份副本，两份之间相隔 4 个 warps。

GPU 硬件经常需要将同一份数据广播到多个 warps、lanes 或存储区域。Replica 直接将这种行为表示为“一项逻辑数据对应多个物理坐标”。

### Offset

Offset 会加到所有映射结果上。在下面的集合公式中将它记为 `O`。

例如：

```python
5@warpid
```

表示整个 layout 沿 `warpid` 轴移动 5 个位置。

Offset 可以用来指定 tile 的起始坐标，也可以把多个 tiles 放入同一硬件资源中的不同区域。

### 组合三个部分

对于逻辑坐标 `x`，把 shard 产生的基础坐标记为 `D(x)`。`TileLayout` 再加上固定 offset，并根据 replica iters 枚举额外位置：

```text
L(x) = { D(x) + r + O | r in R }
```

这里的 `r` 表示 replica iters 枚举出的一个 offset。没有 replica 时，可以把 `R` 看作只包含零 offset，因此集合中只有一个坐标；存在 replica 时，集合中会包含每个副本的位置。当前 `layout.apply()` 只计算 `D(x)+O` 这一基础坐标，不会枚举 `R`；replica iters 保存在 `layout.replica` 中，由使用该 layout 的 tile 操作处理。

完整的 TIRx 写法如下：

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

从左到右读取：`S[...]` 放置逻辑 tile，`R[...]` 在相隔 4 个 warps 的位置增加第二份副本，`5@warpid` 再将所有位置整体移动 5。

如果已经直接构造了 shard、replica 和 offset 对象，也可以使用：

```python
TileLayout.from_iters(shard, replica, offset)
```

普通 kernel 代码更常使用 `S[...]` 和 `R[...]`，因为它们能够直接显示 layout 的 shape、strides 和 axes。

## 命名轴

Layout 中的 axis 不是匿名维度，每个名字都表示一种硬件坐标或编译器布局坐标。下面列出本章会遇到的几组轴：

| Axis | 含义 |
|---|---|
| `bx`、`by`、`bz` | CTA 在 grid 中的坐标 |
| `cbx`、`cby`、`cbz` | CTA 在 cluster 中的坐标 |
| `tx` | thread 在 CTA 中的坐标 |
| `warpid`、`laneid` | warp ID，以及 thread 在 warp 中的 lane ID |
| `wgid`、`tid_in_wg`、`wid_in_wg` | warpgroup ID，以及 thread 或 warp 在 warpgroup 中的位置 |
| `m` | 默认的线性物理轴；实际对应哪种存储由 buffer scope 决定 |
| `TLane`、`TCol` | TMEM 的 Lane 方向和 Col 方向 |

Axis 名称本身就是 layout 的一部分。即使整数值相同，位于不同 axis 上也表示不同的硬件位置。例如，`1@tx` 与 `1@tid_in_wg` 的含义不同，`1@laneid` 也不是 `1@TLane`。`TCol` stride 仍以 buffer element 为单位；只有在 element width 为 32 bits 时，它才与 hardware Col 一一对应。

## 正向映射

`apply()` 从逻辑坐标出发，计算 shard 和 offset 对应的基础物理坐标。它支持三种输入形式：

```python
layout.apply(linear_coord)
layout.apply(*shard_coord)
layout.apply(*logical_coord, shape=input_shape)
```

第三种形式最能完整展示计算过程。设逻辑坐标为：

```text
x = (x0, x1, ..., xr-1)
```

相应的逻辑 shape 为：

```text
(S0, S1, ..., Sr-1)
```

首先，按照 row-major 顺序将逻辑坐标 flatten：

```text
flat = x0 * S1 * S2 * ... * Sr-1
     + x1 * S2 * ... * Sr-1
     + ...
     + xr-2 * Sr-1
     + xr-1
```

随后，按照 shard extents：

```text
(e0, e1, ..., en-1)
```

将 `flat` 拆成分量：

```text
(c0, c1, ..., cn-1)
```

如果第 `k` 个 shard iter 的 stride 为 `sk`、axis 为 `ak`，那么 `ck` 对物理坐标的贡献就是：

```text
ck * sk @ ak
```

属于同一 axis 的贡献会相加，最后再加入固定 offset。`apply()` 返回的 coordinate dictionary 就是这一步得到的结果。

另外两种形式从中间开始计算。`layout.apply(linear_coord)` 直接接收已经 flatten 的索引；`layout.apply(*shard_coord)` 直接接收各个 shard iter 的坐标，因此不再执行 flatten 和拆分。使用 `shape=input_shape` 时，逻辑 shape 的维数和分解方式不必与 shard extents 相同，只要 flatten 后的索引没有超出 shard 表示的逻辑范围即可。

`apply()` 不枚举 replica。完整 layout 中的 replica iters 会在基础坐标之外增加额外位置，但这些信息保存在 `layout.replica` 中，由实际使用该 layout 的 tile 操作处理。

下面用前面由 shard、replica 和 offset 组成的 `layout` 具体计算一次。将输入坐标解释为 `(8, 16)` tile 中的 `(1, 3)`：

```python
layout.apply(1, 3, shape=[8, 16])

# {"laneid": 5, "warpid": 5, "m": 1}
```

这个结果可以分三步得到。首先，`(1, 3)` 按 `(8, 16)` 的 row-major 顺序变成扁平索引 `19`。随后，`19` 按 shard extents `(8, 2, 4, 2)` 拆成：

```text
(c0, c1, c2, c3) = (1, 0, 1, 1)
```

各分量乘以自己的 stride 后，基础坐标为 `laneid=5`、`warpid=0`、`m=1`；最后加入 `5@warpid`，得到返回值中的 `warpid=5`。

`apply()` 不枚举 replica，所以返回值中只有这个基础位置。当前 layout 的 `R[2 : 4@warpid]` 还会让使用该 layout 的 tile 操作处理 `warpid=5` 和 `warpid=9` 两个物理位置。

`apply()` 支持三种输入形式：

```python
layout.apply(linear_coord)
layout.apply(*shard_coord)
layout.apply(*logical_coord, shape=input_shape)
```

第一种形式直接使用已经扁平化的索引；第二种形式为每个 shard iter 直接提供一个坐标；第三种形式先按 `input_shape` flatten，再按照 shard extents 拆分。

对于整个 `(8, 16)` tile，基础映射为：

```text
laneid = 4 * i + (floor(j / 2) mod 4)
warpid = floor(j / 8) + 5
m      = j mod 2
```

Replica 再给 `warpid` 加上 `0` 或 `4`。因此，shard 和 offset 将 tile 放在 warps 5 和 6，replica 又在 warps 9 和 10 中放置一份副本。

## 示例：Blackwell Tensor Memory

命名轴也可以表示存储坐标。TMEM 使用硬件 Lane 和 Col 坐标，在 TIRx layout 中分别写作 `TLane` 和 `TCol`。

考虑下面的 layout：

```python
layout = TileLayout(
    S[(2, 128, 112) : (112@TCol, 1@TLane, 1@TCol)]
)
```

逻辑 tile shape 与 shard extents 都是 `(2, 128, 112)`，因此拆分后的三个分量就是逻辑坐标本身。对于元素 `(a, l, c)`：

```text
TLane = l
TCol  = 112 * a + c
```

Extent 为 128、stride 为 `1@TLane` 的 iter 填满 128 条 TMEM Lane rows；另外两个 `TCol` iters 共同覆盖 224 个 `TCol` 位置：

```text
TCol in [0, 224)
```

TMEM layout 不要求各维大小为 2 的幂。这里可以直接使用 extent 为 112 的 column iter；两个这样的区域共覆盖 224 个 `TCol` 位置，不需要把 extent 补齐到 128。实际 kernel 也可能主动选择这样的大小：例如，block-scaled FP8 GEMM 可以为两个 accumulator stages 和 scale factors 共同分配 TMEM，而不是让单个 accumulator tile 占满 256 个 columns。

## Scale Factor 布局

前面的 accumulator layout 是一对一映射，每个逻辑 accumulator 元素只对应一个 TMEM 坐标。Block-scaled MMA 需要让同一组逻辑 scale factors 对多个 warp windows 可见，因此会使用 replication。先看完整 scale-factor layout 中反复出现的一个 `32×sf_per_mma` atom：

```python
scale = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)]
    + R[4 : 32@TLane]
)
```

对于逻辑 scale coordinate `(r, s)`，shard 先给出：

```text
TLane = r
TCol  = s
```

对于 8-bit scale-factor buffer，这里的 `TCol` 坐标同样以 buffer element 为单位。四个连续的元素位置会打包进一个 32-bit hardware Col；具体的 hardware Col 编号和 byte 位置分别是 `s//4` 和 `s%4`。

Replica 再沿 `TLane` 轴以 32 为 stride 创建 4 份副本：

```text
TLane = r + 32 * q, where q in {0, 1, 2, 3}
TCol  = s
```

因此，这个 32-row group 会同时出现在 lanes `0-31`、`32-63`、`64-95` 和 `96-127`。每个 warp 对应的 32-lane TMEM window 都能访问同一组 scale factors。完整布局还会加入 M 方向和 K-scale-block 方向的外层 iters；这里的 atom 只负责描述一次 MMA 所读取的局部模式。具体硬件数据路径可参考{ref}`chap_layout_generations`。

Accumulator 和 scale factors 使用的仍是同一个 `TileLayout` 模型：前者通常是 TMEM 中的一组基础坐标，后者则在相同的 `TLane`、`TCol` 空间中加入 replication。

## 常用 Layout 构造函数

实际 kernel 通常不需要手写每一种硬件 layout。TIRx 为常见模式提供了构造函数。

```python
tmem_datapath_layout(datapath, rows, cols)
```

返回 `tcgen05.mma` 写入的 TMEM accumulator layout。`datapath` 选择行映射方式。例如，`"D"` 对应 `M=128` 时按 row 直接映射的布局，`"F"` 对应 `M=64` 时分散到多个 Lane 区域的布局。

```python
tcgen05_atom_layout(instr_shape, tensor_shape, dtype)
```

返回与 `tcgen05.ld` 或 `tcgen05.st` data-movement shape 对应的 register tile layout。`instr_shape` 可以是 `"32x32b"`、`"16x64b"`、`"16x128b"` 等字符串；`tensor_shape` 和 `dtype` 共同决定相应的 repeat factor。

这个对象描述一个分布到 warpgroup 各 thread registers 中的 tile。当它用于 TMEM 与 local fragment 之间的 `Tx.wg.copy_async` 时，lowering 可以据此选择匹配的 warp-collective `tcgen05.ld` 或 `tcgen05.st`；每个 warp 处理自己的 32-lane TMEM partition。

```python
wg_local_layout(cols, rows=128)
```

返回 warpgroup-local register tile。它把逻辑 row 映射到 `tid_in_wg`，把同一 row 中的 columns 映射到该 thread 的局部 `m` 轴；使用默认的 `rows=128` 时，每个 thread 持有一行。

这些构造函数返回的仍是 `TileLayout`，使用的也是相同的 iter 和命名轴模型；构造函数只是替 kernel 生成了常见的硬件映射。

## SwizzleLayout 与 ComposeLayout

`TileLayout` 是仿射布局，可以描述命名轴上的 strides、replication 和 offsets，适合表示 register fragments、TMEM tiles 和 scale-factor layouts。

Shared memory swizzle 不属于仿射变换。它通过 XOR 重排线性 shared-memory address，以改变元素落到各个 banks 的方式。因此，TIRx 使用单独的对象表示它：

```python
SwizzleLayout(...)
```

如果只需要描述 swizzle，可以直接把 `SwizzleLayout` 绑定到 buffer。需要在仿射 tile mapping 之上叠加 swizzle 时，再使用 `ComposeLayout`：

```python
ComposeLayout(swizzle, tile)
```

这里的 `tile` 必须只产生默认 `m` 轴上的线性地址。计算时，tile layout 先得到这个地址，swizzle 再对它进行重排。这样可以让仿射 Shape-Stride 映射与非仿射 XOR 变换各自保持清晰。

## 为什么需要 Swizzle

{ref}`chap_data_layout` 已经介绍了 shared memory bank conflict 和 XOR swizzle。这里关注它们在 API 中如何表示。

考虑一个 row-major 的 `(8, 64)` float16 tile：

```python
TileLayout(S[(8, 64) : (64@m, 1@m)])
```

逻辑元素 `(i, j)` 的线性 element address 为：

```text
m = 64 * i + j
```

每行包含 64 个 float16，也就是 128 bytes。固定 `j` 并沿 column 读取时，每换一行都会前进 128 bytes，多个访问可能重复落到同一组 banks。

Swizzle 让 address 的低位同时依赖较高的 row bits，使原本落到同一个 bank 的 column access 分散到多个 banks。

## Swizzle 变换

`SwizzleLayout` 由三个整数参数控制：

```text
per_element = M
swizzle_len = B
atom_len    = S
```

这三个参数都是 bit counts，而不是 bytes。下面的公式以线性 element address `m` 为输入。

`M` 表示需要原样保留的低位数量，`B` 表示参与 XOR 的 bit-field 宽度，`S` 表示两个 bit fields 之间的距离。首先保留 `m` 的低 `M` bits，使一小组相邻元素继续保持 contiguous；其余高位右移得到：

```text
x = m >> M
```

随后，把 `x` 中 `[S, S+B)` 位置的 bits XOR 到 `[0, B)`，最后再放回此前保留的低 `M` bits：

```text
mask = (1 << B) - 1

low  = m & ((1 << M) - 1)
x    = m >> M
x2   = x ^ ((x >> S) & mask)

addr = (x2 << M) | low
```

一个合法的 swizzle 要求 `S >= B`。

这个变换不会改变 tile 中包含哪些逻辑元素，只会改变它们在 shared memory 中的物理地址。后续 MMA 读取的仍是同一个逻辑 tile，但相应的 bank access pattern 已经发生变化。

## 选择 Swizzle 参数

实际使用中，swizzle 参数通常由 data type 和 shared-memory swizzle mode 决定。常见模式包括 32-byte、64-byte 和 128-byte swizzle。

`per_element` 决定保留多少个 element-address 低位，使一个 vector group 内的元素继续保持连续。对于 float16，一个 16-byte vector 包含 8 个元素，因此：

```text
M = log2(8) = 3
```

128-byte swizzle 使用：

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

这里的 128 bytes 指 swizzle atom 中每一行的宽度；完整 atom 包含 8 行。上述参数既能保持每个 16-byte vector group 连续，又能重排更高的 address bits，打散 column access 的 bank pattern。

普通 kernel 不需要反复手工推导这些参数。Data type 和 descriptor mode 通常已经决定具体配置；kernel 需要保证 TIRx layout、TMA descriptor 和 MMA 使用同一种 shared-memory 排列。

一个 swizzled shared-memory allocation 可以写成：

```python
tile = TileLayout(S[(8, 64) : (64@m, 1@m)])
swizzle = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

layout = ComposeLayout(swizzle, tile)
```

最终绑定到 shared-memory buffer 上的是组合后的 `layout`。下面用这个 layout 检查一次实际地址映射。

## 示例：对 `(8, 64)` float16 Tile 应用 128B Swizzle

继续使用上面的 128-byte swizzle。前面的 row-major tile 使用线性 element address：

```text
m = 64 * i + j
```

令：

```text
q = floor(j / 8)
r = j mod 8
```

代入前面的 swizzle 公式，物理 element address 为：

```text
addr = 64 * i + 8 * (q xor i) + r
```

以 column `j=0` 为例，此时 `q=0`、`r=0`：

```text
addr = 72 * i
```

Shared memory 包含 32 个 banks，每个 bank word 为 4 bytes。对于 float16：

```text
bank = floor(addr / 2) mod 32
```

因此，8 行分别映射到：

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

这次 column access 使用了 8 个不同的 banks。不使用 swizzle 时，同一列的地址为 `m=64*i`，因此：

```text
bank = floor(64 * i / 2) mod 32 = 0
```

8 行都会落到 bank 0。Swizzle 没有改变逻辑 tile，只改变了元素的物理地址，使这个访问模式不再集中到同一个 bank。

上面的推导只说明这种 float16 按列访问会被分散到 8 个 banks。其他访问是否无冲突，还取决于 data type、每次访问的宽度以及硬件指令采用的 access shape。在本章开头的交互图中切换 data type 和 swizzle mode，可以直接比较不同组合的地址映射。

对于构造函数已经覆盖的 `tcgen05` 布局，可以直接使用 `tmem_datapath_layout`、`tcgen05_atom_layout` 等 helper。其他仿射布局仍然使用 `S[...]`、`R[...]` 和 offset 表示。查询 `TileLayout` 时，`apply()` 只计算逻辑元素的基础物理坐标，不会枚举 replica。Shared-memory swizzle 由 `SwizzleLayout` 表示；需要将它叠加到只产生线性 `m` 地址的 tile layout 上时，再使用 `ComposeLayout(swizzle, tile)`。
