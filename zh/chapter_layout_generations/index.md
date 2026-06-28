(chap_layout_generations)=
# 跨 GPU 世代的 Tensor Core 操作数布局

<!--
翻译模板

英文源文件：chapter_layout_generations/index.md
建议：保留指令名、layout 名称、图和表格；只翻译解释性文字。
-->

:::{admonition} 概览
:class: overview

- TODO：翻译本章 overview 第一条。
- TODO：翻译本章 overview 第二条。
- TODO：翻译本章 overview 第三条。
:::

TODO：翻译导言部分。

## 两个始终存在的约束

TODO：翻译 “Two Constraints That Never Went Away” 小节。

## Ampere：Warp Lane 上的寄存器 Fragment

TODO：翻译 “Ampere: Register Fragments over Warp Lanes” 小节。

## Ampere Tensor Core 期望的输入

TODO：翻译 “What the Ampere Tensor Core Expects” 小节。

## `ldmatrix`：从共享内存到寄存器 Fragment

TODO：翻译 “ldmatrix: Shared Memory to Register Fragment” 小节。

![TODO：翻译 ldmatrix 图注](../../img/ldstmatrix.svg)

## 写回 Ampere Fragment

TODO：翻译 “Writing the Ampere Fragment Back” 小节。

## Ampere 上的 Swizzle

TODO：翻译 “Swizzle on Ampere” 小节。

![TODO：翻译 swizzle conflict 图注](../../img/swizzle_conflict.svg)

## Hopper：`wgmma`、共享内存 Descriptor 和 Swizzle Format

TODO：翻译 “Hopper: wgmma, Shared Memory Descriptors, and Swizzle Formats” 小节。

## Hopper Tensor Core 期望的输入

TODO：翻译 “What the Hopper Tensor Core Expects” 小节。

![TODO：翻译 Hopper shared memory descriptor 图注](../../img/smem_descriptor.svg)

## Hopper 输出仍然使用寄存器

TODO：翻译 “Hopper Output Still Uses Registers” 小节。

## Blackwell：`tcgen05` 和 TMEM

TODO：翻译 “Blackwell: tcgen05 and TMEM” 小节。

## TMEM 中的 Scale Factor Layout

TODO：翻译 “Scale Factor Layout in TMEM” 小节。

![TODO：翻译 scale_vec byte packing 图注](../../img/sf_scale_vec.svg)

## 一个反复出现的 Fragment

TODO：翻译 “A Recurring Fragment” 小节。

## 主线

TODO：翻译 “The Throughline” 小节。
