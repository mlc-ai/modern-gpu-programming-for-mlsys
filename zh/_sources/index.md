# 面向机器学习系统的现代 GPU 编程

机器学习系统支撑着现代 AI 的许多核心计算任务。随着模型规模扩大、部署场景变得更加复杂，端到端性能越来越依赖少数关键 GPU kernel 的实现质量。Attention、LLM prefill 和 decode、低精度 block-scaled GEMM、融合 MoE 层以及其他大型融合 kernel，都会直接影响训练和服务的速度。

要让这些 kernel 真正跑得快，不能只罗列优化技巧。近年来的 GPU 架构引入了更丰富的内存空间、新的数据搬运机制和越来越专用化的执行单元。要充分利用这些硬件能力，既要理解 GPU 如何执行程序，也要掌握一个基础 kernel 如何逐步演变成高性能实现。本书将围绕这两个方面展开。

本书按照从硬件、编程模型到完整 kernel 的顺序展开。我们会先介绍 GPU 的组织方式和执行模型，再学习本书使用的编程模型，最后逐步构建高性能 kernel。本书主要面向 NVIDIA Blackwell，并以 General Matrix-Matrix Multiplication（GEMM）和 FlashAttention 为贯穿全书的示例。在构建这些 kernel 的过程中，还会系统介绍数据布局、异步数据搬运和异步协作等关键主题。

本书内容源自卡内基梅隆大学的 [Machine Learning Systems](https://mlsyscourse.org/) 课程系列。书中的示例使用 TIRx Python DSL，让读者能够在真实 kernel 中学习、运行和验证这些概念。TIRx 会明确表示与硬件执行有关的选择，因此可以结合可运行的代码分析控制流、内存访问和同步逻辑。

本书是开源项目，欢迎通过 [GitHub 仓库](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys) 贡献代码、勘误和示例。


## 本书结构

- **第一部分：理解 GPU。** 这一部分介绍 GPU 的整体架构组织、编写高性能 kernel 的通用方法，以及数据布局、异步内存操作和协作等关键概念，并建立后续章节所依赖的硬件理解。
- **第二部分：TIRx 概览。** 这一部分介绍 TIRx 的核心组成部分，为理解后续章节中的代码示例做准备。
- **第三部分：GEMM：从 Tiled 到 SOTA。** 这一部分完整讲解如何优化一个 tiled GEMM，并逐步加入 TMA pipelining、persistent scheduling、warp specialization 和 2-CTA cluster。
- **第四部分：Flash Attention 4。** 这一部分基于第三部分的技术构建完整的 attention kernel：两个 MMA，中间插入 softmax，并包含 online-softmax rescaling、causal mask 和 GQA。
- **参考资料。** TIRx 语言参考、编译器内部机制，以及异步 kernel 调试指南。

```{toctree}
:caption: 第一部分：理解 GPU
:maxdepth: 1

chapter_background/index
chapter_performance/index
chapter_data_layout/index
chapter_layout_generations/index
chapter_tma/index
chapter_tensor_cores/index
chapter_tmem/index
chapter_async_barriers/index
chapter_clc/index
```

```{toctree}
:caption: 第二部分：TIRx 概览
:maxdepth: 1

chapter_intro_tirx/index
chapter_tirx_layout_api/index
```

```{toctree}
:caption: 第三部分：GEMM：从 Tiled 到 SOTA
:maxdepth: 2

chapter_gemm_basics/index
chapter_gemm_async/index
chapter_gemm_advanced/index
```

```{toctree}
:caption: 第四部分：Flash Attention 4
:maxdepth: 2

chapter_flash_attention/index
```

```{toctree}
:caption: 参考资料
:maxdepth: 1

appendix/index
tirx_guide/language_reference/index
appendix/debugging_warp_specialized
tirx_guide/arch/index
```
