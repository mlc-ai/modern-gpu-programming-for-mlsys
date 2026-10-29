# Modern GPU Programming For MLSys

This book teaches modern GPU kernel programming as a progression: **understand the
GPU hardware → learn to program it → write state-of-the-art kernels.** It treats
the Blackwell-class GPU — its memory hierarchy and Tensor Memory, its tensor-core and
asynchronous data-movement engines, warpgroups and clusters — as the real subject. The
vehicle is **TIRx** (Tensor IR next), a Python DSL for writing GPU kernels at the IR level.

📖 **Read it online: <https://mlc.ai/modern-gpu-programming-for-mlsys/>**

**Chinese version: <https://mlc.ai/modern-gpu-programming-for-mlsys/zh/>**

🤝 **Contribute:** Corrections, examples, and improvements are welcome through the
[GitHub repository](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys).

## What's inside

- **Part I — Understanding the GPU.** Execution and memory model, the performance model
  (roofline, overlap), a deep dive into data layout, the memory and compute engines (TMA,
  Tensor Memory, Tensor Cores), asynchronous coordination, and advanced scheduling (CLC).
- **Part II — Programming a GPU with TIRx.** An introduction to TIRx through one runnable
  single-MMA GEMM — scope, layout, and dispatch, and how compilation works — plus the tensor
  layout model (`TileLayout`, named axes, swizzle).
- **Part III — GEMM: Tiled to SOTA.** A tiled GEMM built up through TMA pipelining,
  persistent scheduling, warp specialization, and 2-CTA clusters.
- **Part IV — Flash Attention 4.** A complete attention kernel built from the Part III techniques:
  two MMAs with softmax between them, online-softmax rescaling, causal masking, and GQA.
- **Appendices.** TIRx language reference, [reproducible GPU benchmarking and
  profiling](appendix/benchmarking_gpu_kernels.md), compiler internals, and asynchronous-kernel
  debugging.

## Build the book locally

The book is a [Sphinx](https://www.sphinx-doc.org/) site (Markdown/MyST + reStructuredText):

```bash
pip install -r requirements-docs.txt
sphinx-build -b html . _build/html
```

### Preview

```bash
python -m http.server -d _build/html 8000
```

Open <http://localhost:8000>. On a remote machine the server runs there, so forward the
port — `ssh -L 8000:localhost:8000 user@your-server` — then open the URL locally. (VS Code
Remote SSH auto-forwards it.)

## Running the kernels (requires a Blackwell GPU)

The kernels in this book target Blackwell (`sm_100a`), so running them needs a Blackwell GPU
(such as a B200), the TIRx compiler, and a CUDA build of PyTorch.

**1. Install the TIRx compiler.** The Python module is `tvm_tirx`:

```bash
python -m pip install apache-tvm-tirx
python -c "import tvm_tirx; print(tvm_tirx.__version__)"
python scripts/check_installation.py
```

The minimal wheel includes native LLVM and its ORC JIT for CPU execution and GPU host
launchers. To generate ARM or WebAssembly objects from an x86 host, install the LLVM
backend extra and check both targets:

```bash
python -m pip install 'apache-tvm-tirx[all-targets]'
python scripts/check_installation.py --cross
```

In the tested `0.1.0a3` release, `[all]` adds process-pool dependencies; `[all-targets]`
installs the additional LLVM targets. Cross checks generate objects without executing
foreign-architecture code.

**2. Install PyTorch** with a CUDA build matching your GPU — see <https://pytorch.org>.
The examples are checked with PyTorch `2.14.0+cu132` on an NVIDIA B200.

**3. Install the pinned example environment:**

```bash
python -m pip install -r requirements-examples.txt
```

All kernels use the TIRx TVMScript parser and the compiler's native CUDA tile API.
The FA4 source is included in `examples/flash_attention.py`, migrated from the
TVMScript reference originally used by the chapter.

**4. Run the complete examples from both language editions:**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_examples.py --large
```

Run these commands from the repository root. Each program runs in a fresh Python process. GEMM checks refill inputs and poison the
output before each comparison with the FP32 PyTorch reference. Single-tile and multi-tile
K loops are covered; Step 6 uses sizes satisfying its pipeline-phase constraint.
`--large` adds `4096x4096x4096` GEMMs for steps 3–9. Use `--suite kernels`,
`--suite reference`, or `--suite benchmarks` to select a group, and `--language en`
or `--language zh` to select an edition. Generated programs, logs, environment metadata
and results go to local `validation/examples/`. Profiling tools have additional dependencies
listed in the benchmarking appendix. Add `--scheduler-stress` to check partial L2 groups
and repeated persistent scheduling at `5120x5120x256`.

TIRx parses kernel source via Python source inspection, so examples should live in a file
or notebook cell rather than inside `python -c`.

## Deployment

Every push to `main` is built and published automatically by GitHub Actions
(`.github/workflows/build_deploy.yaml`) to <https://mlc.ai/modern-gpu-programming-for-mlsys/>.
