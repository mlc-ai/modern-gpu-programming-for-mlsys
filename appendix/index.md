# Appendix
:label:`chap_appendix`

The main tutorial path ends with Writing TIRX Kernels with Agents. The appendix is not another reading path; use it when you need setup help, extra practice, debugging checklists, or API details.

## How to Use This Appendix

| Need | Use |
|------|-----|
| Install TVM/TIRX or run a tiny kernel | **Environment Setup** |
| Practice basic thread indexing, elementwise code, and reductions | **Practice Kernel: Fused GELU Gate** and **Practice Kernel: RMSNorm Reduction** |
| Inspect generated CUDA for scope guards, barriers, and lowered instructions | **TIRX Language and Compile Pipeline** |
| Read the complete Flash Attention 4 TIRX source | **Flash Attention 4 TIRX Source** |
| Look up the exact spelling of a TIRX API used in the tutorial | **TIRX API Lookup** |
| Understand the parser, buffer model, scopes, metaprogramming, or compile pipeline | **TIRX Language and Compile Pipeline** |

Most readers only need **Environment Setup** and **TIRX API Lookup** at first. The practice kernels are optional.

```toc
:maxdepth: 1

../chapter_setup/index
../chapter_fused_gelu/index
../chapter_rmsnorm/index
../chapter_fa4_source/index
../chapter_api_reference/index
../chapter_tirx_primer/index
```
