# TIRX: Blackwell Tile-Primitive Programming

TIRX (Tensor IR neXt) is a Python DSL for writing high-performance GPU kernels at the IR level. This tutorial targets Blackwell because it exposes the newest TIRX tile primitives: TMA tile copies, `tcgen05` MMA, TMEM layouts, mbarriers, and CTA clusters.

TIRX sits between high-level kernel DSLs and raw CUDA/PTX. High-level DSLs are easier to write and often the right tool, but they can hide the exact hardware contract behind scheduling or library choices. Raw CUDA/PTX gives maximum control, but the program becomes descriptor setup, barrier state, layout bookkeeping, and instruction-level plumbing. TIRX is for the middle ground: the kernel still names Blackwell concepts directly, while the compiler can see scope, layout, and dispatch as structured IR instead of scattered intrinsic arguments.

A tile primitive is a structured operation on tile values. Its lowering is controlled by three things:

- **Scope** — which group of threads issues or cooperates on the operation.
- **Layout** — how the operand tiles map to GMEM, SMEM, TMEM, or registers.
- **Dispatch** — which hardware path is intended when there is a choice, such as TMA or `tcgen05`.

For asynchronous primitives, barriers and waits describe the handoff between tile operations.

## Reading Path

- **Part I: TIRX and Blackwell**
  1. **Blackwell Background** — Thread groups, memory spaces, TMEM, TMA, `tcgen05`, async barriers, and CTA clusters.

- **Part II: Tile Primitive Programming**
  2. **Tile Primitive Mental Model** — Why tile primitives matter, then execution scope, tensor layout, and dispatch.

- **Part III: GEMM Deep Dive**
  3. **GEMM with Tile Primitives** — Single-tile GEMM, K-loop accumulation, and spatial tiling across CTAs.
  4. **Asynchronous GEMM Pipeline** — TMA async loads, software pipelining, and persistent tile scheduling.
  5. **Warp-Specialized and Clustered GEMM** — Warp specialization, 2-CTA cooperative MMA, and multi-consumer execution.

- **Part IV: Advanced Kernel**
  6. **Flash Attention on Blackwell**

- **Appendix**
  - Setup, practice kernels, debugging notes, and references.


```toc
:maxdepth: 2
:numbered:

chapter_background/index
chapter_layouts/index
chapter_gemm_basics/index
chapter_gemm_async/index
chapter_gemm_advanced/index
chapter_flash_attention/index
appendix/index
```
