(chap_data_layouts)=
# Data Layouts

```{note}
**Outline stub** — to be drafted (P1). Will absorb the layout taxonomy currently in
{ref}`chap_layouts` plus the data-layout and swizzle slides. See `OUTLINE.md`, Ch7.
```

**Goal:** treat layouts as a first-class concept; understand bank conflicts and swizzle.

## Logical → Physical Mapping
`TileLayout` / `S[...]` notation; named axes (`@m`, `@laneid`, `@reg`, `tid_in_wg`).

## Tiled, Thread, and Distributed Layouts
Per-thread register views; cluster-distributed layouts.

## Memory Banks, Bank Conflicts, and Swizzle
`SWIZZLE_128B`, swizzle atoms, and the tiling constraint; tie-in to `tma_shared_layout`.

*Feeds from:* data-layout slides (Tile Layout, Named Axes, Distributed Axes, Memory Banks, Simple Swizzle, SWIZZLE_128B), modern-gpu-gemm (Bank Sector View, Swizzle Atoms, Tiling Constraint, Why 128B), and the `axe_layout` demos.
