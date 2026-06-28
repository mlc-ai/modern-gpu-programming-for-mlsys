(chap_tma)=
# 异步数据搬运：TMA

<!--
翻译模板

英文源文件：chapter_tma/index.md
建议：保留 TMA 指令名、barrier 语义、demo iframe 和图。
-->

:::{admonition} 概览
:class: overview

- TODO：翻译本章 overview 第一条。
- TODO：翻译本章 overview 第二条。
- TODO：翻译本章 overview 第三条。
:::

TODO：翻译导言部分。

```{raw} html
<iframe src="../demo_zh/tma_intro.html" title="TMA: the Tensor Memory Accelerator" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击图中组件查看细节：TODO：翻译 TMA intro 图注。*

## 一个线程发起，硬件搬运 Tile

TODO：翻译 “One Thread Issues, Hardware Moves the Tile” 小节。

## Swizzled Layout

TODO：翻译 “Swizzled Layouts” 小节。

## 用 3D TMA 表达 Tiling 和 Swizzling

TODO：翻译 “3D TMA for Tiling and Swizzling” 小节。

```{raw} html
<iframe class="demo-tma3d" src="../demo_zh/tma_3d.html" title="Tiling and swizzling with 3D TMA" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击图中组件查看细节：TODO：翻译 3D TMA 图注。*

```{raw} html
<iframe class="demo-tma3d" src="../demo_zh/tiling_constraint.html" title="Swizzle imposes a tiling constraint" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击图中组件查看细节：TODO：翻译 tiling constraint 图注。*

## 完成通知：Load

TODO：翻译 “Completion: Loads” 小节。

![TODO：翻译 TMA load synchronization flow 图注](../../img/tma_sync_flow.png)

## 完成通知：Store

TODO：翻译 “Completion: Stores” 小节。

## 为什么 TMA 对 Pipelining 很重要

TODO：翻译 “Why TMA Matters for Pipelining” 小节。
