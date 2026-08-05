# Modern GPU Programming For MLSys

Machine learning systems power many of today's AI workloads. As models grow and deployment
settings become more complex, end-to-end performance increasingly depends on a small number of
critical GPU kernels. Attention, LLM prefill and decode, low-precision block-scaled GEMM, fused MoE
layers, and other large fused kernels directly affect both training and serving speed.

Making these kernels fast requires more than a list of optimization tricks. Recent GPU
architectures introduce richer memory spaces, new data-movement mechanisms, and increasingly
specialized execution units. Using them effectively requires both a clear understanding of how the
hardware executes a program and practical knowledge of how a basic kernel evolves into a
high-performance implementation. This book develops both.

The book proceeds from hardware to programming model to complete kernels. It first introduces GPU
organization and execution, then presents the programming model used throughout the book, and
finally builds high-performance kernels step by step. The main target is NVIDIA Blackwell, and the
running examples are General Matrix-Matrix Multiplication (GEMM) and FlashAttention. Along the way,
the book develops the key ideas behind GPU optimization: data layout, asynchronous data movement,
and asynchronous coordination.

The material grows out of the [Machine Learning Systems](https://mlsyscourse.org/) course series at
Carnegie Mellon University. The examples use the **TIRx** Python DSL so that the ideas can be
studied, run, and verified in real kernels. TIRx keeps hardware-level choices explicit, making it
possible to reason about control flow, memory access, and synchronization while working with
runnable code.

This book is open source. Contributions, corrections, and examples are welcome through the
[GitHub repository](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys).

## How This Book Is Organized

- **Part I, Understanding the GPU.** This part introduces the overall organization of the GPU,
  general techniques for writing fast kernels, and key concepts such as data layout, asynchronous
  memory operations, and coordination. It builds the hardware intuition that the rest of the book
  relies on.
- **Part II, TIRx Overview.** This part introduces the key elements of TIRx, which serve as the
  foundation for the code examples throughout the book.
- **Part III, GEMM: Tiled to SOTA.** A complete guide to optimizing a tiled GEMM, built up through
  TMA pipelining, persistent scheduling, warp specialization, and 2-CTA clusters.
- **Part IV, Flash Attention 4.** A complete attention kernel built from the Part III techniques:
  two MMAs with softmax between them, online-softmax rescaling, causal masking, and GQA.
- **Reference.** TIRx language reference, compiler internals, and a guide to debugging asynchronous
  kernels.

```{toctree}
:caption: Part I, Understanding the GPU
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
:caption: Part II, TIRx Overview
:maxdepth: 1

chapter_intro_tirx/index
chapter_tirx_layout_api/index
```

```{toctree}
:caption: "Part III, GEMM: Tiled to SOTA"
:maxdepth: 2

chapter_gemm_basics/index
chapter_gemm_async/index
chapter_gemm_advanced/index
```

```{toctree}
:caption: Part IV, Flash Attention 4
:maxdepth: 2

chapter_flash_attention/index
```

```{toctree}
:caption: Reference
:maxdepth: 1

appendix/index
tirx_guide/language_reference/index
appendix/debugging_warp_specialized
tirx_guide/arch/index
```
