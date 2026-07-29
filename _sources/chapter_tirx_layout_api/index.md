(chap_tirx_layout_api)=
# TIRx Layout API

:::{admonition} Overview
:class: overview

- `TileLayout` uses `S[...]`, `R[...]`, and an offset to describe how a logical tile is placed over named axes.
- `TileLayout.apply()` computes the base physical coordinate of a logical element. Replica information remains in `layout.replica` and is handled by the tile operation that uses the layout.
- `SwizzleLayout` describes XOR-based address permutations in shared memory. Use `ComposeLayout` when a swizzle needs to be combined with an ordinary tile layout.
:::

{ref}`chap_data_layout` introduced tile shapes, strides over named axes, replication dimensions, and fixed offsets. This chapter explains how to construct, attach, and inspect those layouts in TIRx programs.

For example, the following notation describes a `128x256` tile in TMEM:

```python
S[(128, 256) : (1@TLane, 1@TCol)]
```

In a TIRx program, the same notation constructs a `TileLayout` that can be attached to a buffer:

```python
layout = TileLayout(S[(128, 256) : (1@TLane, 1@TCol)])

pool.alloc(shape, dtype, layout=layout)

T.decl_buffer(shape, dtype, scope=scope, layout=layout)
```

The buffer now carries its physical layout. Tile operations can use that information directly instead of restating which lanes, registers, or linear storage locations hold its elements.

The layout objects and named axes used in this chapter live in `tvm.tirx.layout`:

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

A layout does not have to produce a single linear address. Its result may instead contain hardware coordinates such as `laneid`, `warpid`, `TLane`, and `TCol`. The TMEM layout above maps the logical row to `TLane` and the logical column to `TCol`. Here is a register fragment that uses both lane and warp axes:

```python
frag = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
)
```

The same physical axis may appear more than once. Here, the first and third iters both contribute to `laneid`. The final stride has no explicit axis tag, so it uses the default axis `m`.

{ref}`chap_data_layout` used the notation `@reg` to distinguish lane-local fragment slots. The current TIRx API does not register a separate `reg` axis. When a layout is attached to a register-backed local buffer, the default axis `m` denotes that thread's local linear position. The buffer scope determines that the data resides in registers; `m` does not imply global or shared memory in this context.

For storage axes such as `m` and `TCol`, strides are measured in buffer elements. In a 32-bit TMEM buffer, advancing one element along `TCol` advances one 32-bit hardware Col. In an 8-bit or 16-bit buffer, several adjacent elements are packed into one hardware Col. The scale-factor example later in this chapter makes this distinction concrete.

## Interactive Demo

The interactive demo below provides several common layout presets. You can edit the logical shape, `S[...]`, or `R[...]`, select a data type and swizzle mode, and click a logical element to inspect its physical coordinates.

```{raw} html
<p>
  <a class="reference external" href="../_static/tirx-layout-demo/index.html"
     target="_blank" rel="noopener"
     style="display:inline-block; padding:10px 18px; background:#3b82f6;
     color:#fff !important; font-weight:700; border-radius:8px;
     text-decoration:none;">Open the interactive demo in a new window</a>
</p>
<iframe id="tirx-layout-demo-frame" src="../_static/tirx-layout-demo/index.html?notitle"
        style="width:100%; height:1040px; border:1px solid #dfe1e6;
        border-radius:10px; margin:10px 0 6px; display:block; box-sizing:content-box;"
        title="TIRx interactive layout demo" loading="lazy"></iframe>
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

The demo also shows the basic `TileLayout` evaluation process. It first flattens a logical coordinate, then splits the flat index according to the extent of each iter. Each component contributes a base coordinate according to its stride and axis, after which the offset is added. The demo also enumerates replicas so that every physical copy of an element is visible.

## TileLayout

`TileLayout` is the primary affine layout object in TIRx. It is usually written as:

```python
TileLayout(S[shape : strides])
```

`S[...]` is the shard spec. It supplies a sequence of iter extents and strides that map the logical tile to a base position on the named axes.

When the same value must appear at several physical locations, add a replica spec:

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride])
```

A fixed offset may be added as well:

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride] + offset)
```

Inside the API, each iter is a triple:

```text
(extent, stride, axis)
```

The `extent` gives the number of positions in the iter, the `stride` gives the distance moved by one step, and the `axis` identifies the physical axis along which that movement occurs.

### Shard

The shard is constructed by `S[...]`. It splits the logical index across one or more iters and produces the base physical coordinate. The `frag` layout above has four shard iters with extents `8`, `2`, `4`, and `2`. Their strides map the components to `laneid`, `warpid`, `laneid` again, and the default linear axis `m`.

This is still the ordinary shape-and-stride rule, except that every stride belongs to an explicit named axis instead of contributing to a single linear address.

### Replica

The replica is constructed by `R[...]` and describes additional physical copies of the same logical element. Replica iters do not depend on the logical index; they enumerate additional offsets in physical space.

For example:

```python
R[2 : 4@warpid]
```

places two copies along the `warpid` axis, separated by four warps.

GPU hardware often needs to broadcast the same data across warps, lanes, or storage regions. A replica expresses that behavior directly as one logical element with several physical coordinates.

### Offset

The offset is added to every mapped coordinate. We denote it by `O` in the set expression below.

For example:

```python
5@warpid
```

shifts the entire layout by five positions along the `warpid` axis.

An offset can select a tile's starting coordinate or place several tiles in different regions of the same hardware resource.

### Putting the Three Pieces Together

For a logical coordinate `x`, let `D(x)` be the base coordinate generated by the shard. `TileLayout` adds the fixed offset and uses the replica iters to enumerate additional positions:

```text
L(x) = { D(x) + r + O | r in R }
```

Here, `r` is one offset generated by the replica iters. With no replica, `R` can be treated as containing only the zero offset, so the set contains one coordinate. With replication, the set contains one coordinate for each copy. The current `layout.apply()` method computes only the base coordinate `D(x) + O`; it does not enumerate `R`. Replica iters remain in `layout.replica` and are handled by the tile operation that uses the layout.

The complete TIRx form is:

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

Read it from left to right: `S[...]` places the logical tile, `R[...]` adds a second copy four warps away, and `5@warpid` shifts every position by five.

If the shard, replica, and offset objects have already been constructed directly, the same layout can be created with:

```python
TileLayout.from_iters(shard, replica, offset)
```

Kernel code usually uses `S[...]` and `R[...]` because the notation exposes the layout's shape, strides, and axes directly.

## Named Axes

Axes in a layout are not anonymous dimensions. Each name identifies a hardware coordinate or a compiler-defined layout coordinate. The axes used in this chapter are summarized below:

| Axis | Meaning |
|---|---|
| `bx`, `by`, `bz` | CTA coordinates in the grid |
| `cbx`, `cby`, `cbz` | CTA coordinates within a cluster |
| `tx` | Thread coordinate within a CTA |
| `warpid`, `laneid` | Warp ID and the thread's lane ID within its warp |
| `wgid`, `tid_in_wg`, `wid_in_wg` | Warpgroup ID and the thread or warp position within a warpgroup |
| `m` | Default linear physical axis; the buffer scope determines the backing storage |
| `TLane`, `TCol` | The Lane and Col directions in TMEM |

The axis name is part of the layout. Equal integer values on different axes identify different hardware positions. For example, `1@tx` differs from `1@tid_in_wg`, and `1@laneid` differs from `1@TLane`. A `TCol` stride is still measured in buffer elements and corresponds one-to-one with hardware Col only when the element width is 32 bits.

## Forward Mapping

`apply()` starts from a logical coordinate and computes the base physical coordinate contributed by the shard and offset. It supports three input forms:

```python
layout.apply(linear_coord)
layout.apply(*shard_coord)
layout.apply(*logical_coord, shape=input_shape)
```

The third form makes the full evaluation process easiest to see. Let the logical coordinate be:

```text
x = (x0, x1, ..., xr-1)
```

inside the logical shape:

```text
(S0, S1, ..., Sr-1)
```

First flatten the logical coordinate in row-major order:

```text
flat = x0 * S1 * S2 * ... * Sr-1
     + x1 * S2 * ... * Sr-1
     + ...
     + xr-2 * Sr-1
     + xr-1
```

Next, split `flat` according to the shard extents:

```text
(e0, e1, ..., en-1)
```

to obtain:

```text
(c0, c1, ..., cn-1)
```

If shard iter `k` has stride `sk` and axis `ak`, component `ck` contributes:

```text
ck * sk @ ak
```

Contributions to the same axis are added, followed by the fixed offset. The resulting coordinate dictionary is what `apply()` returns.

The other two forms skip one or both of these steps. `layout.apply(linear_coord)` accepts an index that is already flat. `layout.apply(*shard_coord)` accepts one coordinate for each shard iter, so neither flattening nor splitting is needed. With `shape=input_shape`, the logical shape may have a different rank and decomposition from the shard extents, provided that its flat index stays within the logical range represented by the shard.

`apply()` does not enumerate replicas. Replica iters add positions beyond the base coordinate, but they remain in `layout.replica` for the tile operation that consumes the layout.

Now evaluate the layout assembled earlier from a shard, replica, and offset. Interpret `(1, 3)` as a coordinate in an `(8, 16)` input tile:

```python
layout.apply(1, 3, shape=[8, 16])

# {"laneid": 5, "warpid": 5, "m": 1}
```

The result follows in three steps. First, `(1, 3)` becomes flat index `19` in the row-major `(8, 16)` shape. Splitting `19` according to the shard extents `(8, 2, 4, 2)` gives:

```text
(c0, c1, c2, c3) = (1, 0, 1, 1)
```

After multiplying each component by its stride, the base coordinate is `laneid=5`, `warpid=0`, and `m=1`. Adding `5@warpid` produces the returned coordinate `warpid=5`.

Because `apply()` does not enumerate replicas, it returns only this base position. The layout's `R[2 : 4@warpid]` tells the tile operation to handle both `warpid=5` and `warpid=9`.

Across the complete `(8, 16)` tile, the base mapping is:

```text
laneid = 4 * i + (floor(j / 2) mod 4)
warpid = floor(j / 8) + 5
m      = j mod 2
```

The replica adds either `0` or `4` to `warpid`. The shard and offset therefore place the tile on warps 5 and 6, while the replica adds a copy on warps 9 and 10.

## Example: Blackwell Tensor Memory

Named axes can also describe storage coordinates. TMEM uses hardware Lane and Col coordinates, written as `TLane` and `TCol` in TIRx layouts.

Consider:

```python
layout = TileLayout(
    S[(2, 128, 112) : (112@TCol, 1@TLane, 1@TCol)]
)
```

The logical tile shape and shard extents are both `(2, 128, 112)`, so the three split components are the logical coordinates themselves. For element `(a, l, c)`:

```text
TLane = l
TCol  = 112 * a + c
```

The extent-128 iter with stride `1@TLane` fills all 128 TMEM Lane rows. The other two iters together span 224 `TCol` positions:

```text
TCol in [0, 224)
```

TMEM layout dimensions need not be powers of two. The column iter can use an extent of 112 directly; two such regions cover 224 `TCol` positions without padding the extent to 128. A real kernel may choose this shape deliberately. For example, a block-scaled FP8 GEMM can allocate TMEM for two accumulator stages and scale factors instead of allowing one accumulator tile to occupy all 256 columns.

## Scale-Factor Layouts

The accumulator layout above is one-to-one: each logical accumulator element has one TMEM coordinate. Block-scaled MMA needs the same group of logical scale factors to be visible from several warp windows, so it uses replication. Consider the `32xsf_per_mma` atom that recurs in the complete scale-factor layout:

```python
scale = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)]
    + R[4 : 32@TLane]
)
```

For logical scale coordinate `(r, s)`, the shard first produces:

```text
TLane = r
TCol  = s
```

For an 8-bit scale-factor buffer, the `TCol` coordinate is still measured in buffer elements. Four adjacent element positions are packed into one 32-bit hardware Col. Their hardware Col and byte position are therefore `s//4` and `s%4`, respectively.

The replica then creates four copies along `TLane` at a stride of 32:

```text
TLane = r + 32 * q, where q in {0, 1, 2, 3}
TCol  = s
```

The 32-row group consequently appears in lanes `0-31`, `32-63`, `64-95`, and `96-127`. Each warp's 32-lane TMEM window can access the same scale factors. The complete layout adds outer iters for the M and K-scale-block dimensions; this atom describes only the local pattern read by one MMA. See {ref}`chap_layout_generations` for the corresponding hardware data path.

Accumulators and scale factors therefore use the same `TileLayout` model. An accumulator layout normally maps each element to one TMEM coordinate, while a scale-factor layout adds replication in the same `TLane`/`TCol` space.

## Common Layout Constructors

Kernels rarely need to spell out every hardware layout by hand. TIRx provides constructors for common patterns.

```python
tmem_datapath_layout(datapath, rows, cols)
```

This returns the TMEM accumulator layout written by `tcgen05.mma`. The `datapath` argument selects the row mapping. For example, `"D"` is the direct row mapping used for `M=128`, while `"F"` is the mapping that scatters `M=64` across several Lane regions.

```python
tcgen05_atom_layout(instr_shape, tensor_shape, dtype)
```

This returns the register tile layout associated with a `tcgen05.ld` or `tcgen05.st` data-movement shape. `instr_shape` may be `"32x32b"`, `"16x64b"`, `"16x128b"`, or another supported string. Together, `tensor_shape` and `dtype` determine the repeat factor.

The returned object describes a tile distributed across the registers of a warpgroup's threads. When used by `Tx.wg.copy_async` between TMEM and a local fragment, lowering can select the matching warp-collective `tcgen05.ld` or `tcgen05.st`; each warp handles its own 32-lane TMEM partition.

```python
wg_local_layout(cols, rows=128)
```

This returns a warpgroup-local register tile. It maps logical rows to `tid_in_wg` and columns within a row to that thread's local `m` axis. With the default `rows=128`, each thread owns one row.

All three constructors return ordinary `TileLayout` objects built from the same iters and named axes. They are convenience wrappers for hardware mappings that recur across kernels.

## SwizzleLayout and ComposeLayout

`TileLayout` is affine. It can express strides, replication, and offsets over named axes, making it suitable for register fragments, TMEM tiles, and scale-factor layouts.

A shared-memory swizzle is not affine. It uses XOR to permute a linear shared-memory address and change which banks receive the elements. TIRx therefore represents it with a separate object:

```python
SwizzleLayout(...)
```

When a buffer needs only the swizzle, `SwizzleLayout` can be attached directly. When the swizzle must be applied on top of an affine tile mapping, use `ComposeLayout`:

```python
ComposeLayout(swizzle, tile)
```

Here, `tile` must produce a linear address on the default `m` axis only. During evaluation, the tile layout produces that address first, and the swizzle then permutes it. This keeps the affine shape-and-stride mapping separate from the non-affine XOR transform.

## Why Swizzle

{ref}`chap_data_layout` introduced shared-memory bank conflicts and XOR swizzling. Here we focus on how the API represents them.

Consider a row-major `(8, 64)` float16 tile:

```python
TileLayout(S[(8, 64) : (64@m, 1@m)])
```

Logical element `(i, j)` has linear element address:

```text
m = 64 * i + j
```

Each row contains 64 float16 values, or 128 bytes. Reading a fixed column `j` advances by 128 bytes from one row to the next, so several accesses may repeatedly land in the same set of banks.

A swizzle makes low address bits depend on higher row bits, scattering a column access that would otherwise repeatedly hit the same bank.

## The Swizzle Transform

`SwizzleLayout` is controlled by three integer parameters:

```text
per_element = M
swizzle_len = B
atom_len    = S
```

All three parameters are bit counts, not byte counts. The formulas below take a linear element address `m` as input.

`M` is the number of low bits left unchanged, `B` is the width of the bit field participating in the XOR, and `S` is the distance between the two bit fields. First preserve the low `M` bits of `m` so that a small group of adjacent elements remains contiguous, then shift the remaining high bits down:

```text
x = m >> M
```

Next, XOR bits `[S, S+B)` of `x` into bits `[0, B)`, then restore the preserved low `M` bits:

```text
mask = (1 << B) - 1

low  = m & ((1 << M) - 1)
x    = m >> M
x2   = x ^ ((x >> S) & mask)

addr = (x2 << M) | low
```

A valid swizzle requires `S >= B`.

The transform does not change which logical elements belong to the tile. It changes only their physical addresses in shared memory. A subsequent MMA still reads the same logical tile, but its bank access pattern is different.

## Choosing Swizzle Parameters

In practice, the data type and shared-memory swizzle mode usually determine these parameters. Common modes include 32-byte, 64-byte, and 128-byte swizzles.

`per_element` preserves enough low element-address bits to keep one vector group contiguous. For float16, a 16-byte vector contains eight elements:

```text
M = log2(8) = 3
```

A 128-byte swizzle uses:

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

Here, 128 bytes is the width of one row in the swizzle atom; the complete atom contains eight rows. These parameters preserve each contiguous 16-byte vector group while permuting higher address bits to spread column accesses across banks.

Kernel code generally should not derive these parameters by hand. The data type and descriptor mode usually select the configuration. The important requirement is that the TIRx layout, TMA descriptor, and MMA all agree on the shared-memory arrangement.

A swizzled shared-memory allocation can be written as:

```python
tile = TileLayout(S[(8, 64) : (64@m, 1@m)])
swizzle = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

layout = ComposeLayout(swizzle, tile)
```

The composed `layout` is attached to the shared-memory buffer. We can now use it to inspect a concrete address mapping.

## Example: Applying a 128B Swizzle to an `(8, 64)` float16 Tile

Continue with the 128-byte swizzle above. The row-major tile starts with linear element address:

```text
m = 64 * i + j
```

Define:

```text
q = floor(j / 8)
r = j mod 8
```

Substituting into the swizzle transform gives:

```text
addr = 64 * i + 8 * (q xor i) + r
```

For column `j=0`, both `q` and `r` are zero:

```text
addr = 72 * i
```

Shared memory has 32 banks, each with a 4-byte bank word. For float16:

```text
bank = floor(addr / 2) mod 32
```

The eight rows therefore map to:

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

This column access uses eight distinct banks. Without swizzling, the same column has address `m=64*i`, so:

```text
bank = floor(64 * i / 2) mod 32 = 0
```

All eight rows land in bank 0. The swizzle leaves the logical tile unchanged but rearranges its physical addresses so that this access pattern no longer concentrates on one bank.

This derivation only shows that this float16 column access is spread over eight banks. Whether another access is conflict-free still depends on the data type, access width, and hardware instruction's access shape. Change the data type and swizzle mode in the demo at the beginning of the chapter to compare their address mappings directly.

For `tcgen05` layouts covered by existing constructors, use helpers such as `tmem_datapath_layout` and `tcgen05_atom_layout`. Other affine layouts still use `S[...]`, `R[...]`, and an offset. When inspecting a `TileLayout`, remember that `apply()` computes only the base physical coordinate and does not enumerate replicas. Shared-memory swizzles are represented by `SwizzleLayout`; use `ComposeLayout(swizzle, tile)` when applying one to a tile layout that produces a linear `m` address.
