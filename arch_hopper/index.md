(part_hopper)=
# Hopper — The Asynchronous Model

Hopper (the H100 generation, `sm_90`) introduced the asynchronous machinery that defines modern
GPU kernels and that Blackwell builds directly on. Two pieces matter most for this book:

- the **Tensor Memory Accelerator (TMA)** — hardware-driven tile movement issued by a single
  thread, and
- **mbarrier**-based coordination — the counter/phase primitive that hands data safely between
  asynchronous engines.

Together they turn a kernel from a sequence of blocking thread loops into a set of overlapping
asynchronous transfers and computations. Everything here carries forward to Blackwell; the next
section covers what Blackwell *adds* on top.

```{toctree}
:maxdepth: 1

../chapter_tma/index
../chapter_async_barriers/index
```
