(chap_tensor_cores)=
# Tensor Core：`tcgen05`

<!--
翻译模板

英文源文件：chapter_tensor_cores/index.md
建议：保留 `tcgen05`、`cta_group`、TMEM、MMA shape 和图。
-->

:::{admonition} 概览
:class: overview

- TODO：翻译本章 overview 第一条。
- TODO：翻译本章 overview 第二条。
- TODO：翻译本章 overview 第三条。
:::

TODO：翻译导言部分。

```{raw} html
<iframe src="../demo_zh/tcgen05_intro.html" title="tcgen05 and Tensor Memory" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*点击图中组件查看细节：TODO：翻译 tcgen05 and Tensor Memory 图注。*

## `tcgen05` MMA

TODO：翻译 “The tcgen05 MMA” 小节。

## Accumulator 位于 TMEM

TODO：翻译 “The Accumulator Lives in TMEM” 小节。

## `cta_group::1` 和 `cta_group::2`

TODO：翻译 “cta_group::1 and cta_group::2” 小节。

### `cta_group::1`，`M = 128`

TODO：翻译本小节。

![TODO：翻译 cta_group::1 M=128 图注](../../img/mma_cg1_m128.svg)

### `cta_group::1`，`M = 64`

TODO：翻译本小节。

![TODO：翻译 cta_group::1 M=64 图注](../../img/mma_cg1_m64.svg)

### `cta_group::2`，`M = 256`

TODO：翻译本小节。

![TODO：翻译 cta_group::2 M=256 图注](../../img/mma_cg2_m256.svg)

### `cta_group::2`，`M = 128`

TODO：翻译本小节。

![TODO：翻译 cta_group::2 M=128 图注](../../img/mma_cg2_m128.svg)

## Operand Placement

TODO：翻译 “Operand Placement” 小节。

## Block-Scaled MMA

TODO：翻译 “Block-Scaled MMA” 小节。

## Scale Factor 放在哪里

TODO：翻译 “Where the Scale Factors Live” 小节。

## `cta_group::2` 中的 Scale Factor

TODO：翻译 “Scale Factors in cta_group::2” 小节。

![TODO：翻译 block-scaled MMA placement 图注](../../img/mma_block_scaled.svg)

## 保持 MMA Contract 匹配

TODO：翻译 “Keeping the MMA Contracts Matched” 小节。
