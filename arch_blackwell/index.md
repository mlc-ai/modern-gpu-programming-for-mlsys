(part_blackwell)=
# Blackwell — Tensor Cores and Cluster Control

Blackwell (the B200 generation, `sm_100`) keeps Hopper's asynchronous model and adds a new compute
path: the **`tcgen05` Tensor Core** and its dedicated **Tensor Memory (TMEM)**. The headline change
is *where the accumulator lives* — large MMA accumulators now sit in TMEM instead of registers,
which reshapes the epilogue of every tensor-core kernel. This is the compute path the GEMM
(Part III) and Flash Attention (Part IV) kernels target.

Blackwell also extends Hopper's cluster model with **Cluster Launch Control (CLC)** — a hardware
mechanism that lets persistent clusters steal not-yet-launched work from the grid scheduler, for
dynamic load balancing.

```{toctree}
:maxdepth: 1

../chapter_tensor_cores/index
../chapter_clc/index
```
