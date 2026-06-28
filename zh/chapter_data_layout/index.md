(chap_data_layout)=
# 数据布局及其记号

<!--
翻译模板

英文源文件：chapter_data_layout/index.md
建议：保留布局表达式、轴名、代码符号和 demo iframe。
-->

:::{admonition} 概览
:class: overview

- TODO：翻译本章 overview 第一条。
- TODO：翻译本章 overview 第二条。
- TODO：翻译本章 overview 第三条。
:::

TODO：翻译导言部分。

## Shape-Stride 模型

TODO：翻译 “The Shape-Stride Model” 小节。

## Tile Layout

TODO：翻译 “Tile Layout” 小节。

```{raw} html
<iframe src="../demo_zh/tiled_layout.html" title="Tile layout: interactive address computation" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击图中组件查看细节：TODO：翻译 tile layout 图注。*

## 命名轴

TODO：翻译 “Named Axes” 小节。

```{raw} html
<iframe src="../demo_zh/thread_register.html" title="Thread + register layout via named axes" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击图中组件查看细节：TODO：翻译 named axes 图注。*

## 分布式布局

TODO：翻译 “Distributed Layout” 小节。

```{raw} html
<iframe src="../demo_zh/tile_distributed.html" title="Distributed layout across a GPU mesh" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击图中组件查看细节：TODO：翻译 distributed layout 图注。*

### Kernel 内复制模式：TMEM 中的 Scale Factor

TODO：翻译 “Intra-Kernel Replication Pattern: Scale Factors in TMEM” 小节。

```{raw} html
<iframe src="../demo_zh/sf_tmem.html" title="Scale factors in TMEM: packing and warpx4 replication" loading="lazy"
        style="width:100%; height:560px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击图中组件查看细节：TODO：翻译 scale factors in TMEM 图注。*

## Swizzle Layout

TODO：翻译 “Swizzle Layout” 小节。

```{raw} html
<iframe src="../demo_zh/swizzle_8x8.html" title="8x8 XOR swizzle" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击图中组件查看细节：TODO：翻译 8x8 XOR swizzle 图注。*

```{raw} html
<iframe src="../demo_zh/swizzle_128B.html" title="SWIZZLE_128B layout" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击图中组件查看细节：TODO：翻译 SWIZZLE_128B 图注。*

```{raw} html
<iframe src="../demo_zh/swizzle_atom_general.html" title="Swizzle atom layout per format (128B/64B/32B)" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击图中组件查看细节：TODO：翻译 swizzle atom 图注。*
