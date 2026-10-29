(chap_tirx_primer)=
# Introduction to TIRx

:::{admonition} Overview
:class: overview

- TIRx is a Python DSL for writing GPU kernels. It exposes hardware concepts such as threads, SMEM, TMEM, barriers, and Tensor Cores through a structured intermediate representation.
- Three pieces of information determine a tile operation in TIRx: which threads execute it (scope), how its data is laid out (layout), and which hardware path implements it (dispatch).
- This chapter starts with a runnable single-tile GEMM, then explains how to write, compile, and verify a TIRx kernel and how scope, layout, and dispatch work together to determine its behavior.
:::

:::{admonition} Running the examples
:class: note

The examples in this chapter require a Blackwell GPU (`sm_100a`, such as a B200), the TIRx compiler, and a CUDA-enabled build of PyTorch. TIRx is available as the `tvm_tirx` module in the `apache-tvm-tirx` wheel. Compiling CUDA through NVRTC also requires `cuda-bindings`, so install both packages:

```bash
pip install apache-tvm-tirx==0.1.0a5 cuda-bindings
```

After installation, verify that TVM and TIRx import correctly:

```bash
python -c "import tvm_tirx; print(tvm_tirx.__version__)"
```

The runnable examples in later chapters use the same environment.
:::

Part I introduced the execution model of modern GPUs, data layouts, and hardware mechanisms such as TMA, Tensor Cores, TMEM, and asynchronous synchronization. The next step is to organize those mechanisms into kernels that can actually run.

The same work can be done directly in CUDA or PTX, but low-level programs tend to scatter several important decisions across intrinsic arguments, address calculations, and coding conventions: which threads execute an operation, where an operand tile lives, and which hardware instruction ultimately implements the operation. All of that information is present, but it is difficult for a compiler to inspect and transform as a whole.

TIRx (Tensor IR next) is a Python DSL that makes these three decisions explicit in structured IR:

- **Scope**: which threads execute an operation;
- **Layout**: how a logical tile maps to memory, lanes, or registers;
- **Dispatch**: which hardware implementation executes a tile operation.

TIRx still works directly with hardware concepts such as threads, SMEM, TMEM, barriers, and `tcgen05.mma`. The difference is that these choices are represented explicitly in the IR, where the compiler can check them and lower them into machine-level code.

Rather than begin with a list of language constructs, this chapter starts with a complete kernel. We first run a minimal single-tile GEMM, then return to its scope, layout, and dispatch choices, and finally inspect how it is compiled.

## A First TIRx Kernel

The kernel below computes:

```text
D = A B^T
```

Both `A` and `B` have shape `128x64`, and the output `D` has shape `128x128`. The example computes a single `128x128` output tile, so the grid contains only one CTA. Its data path is:

```text
A/B: GMEM -> SMEM -> tcgen05.mma
D:   tcgen05.mma -> TMEM -> registers -> GMEM
```

The matrix multiplication is expressed as one `Tx.apply` tile operation. This single operation describes the full `128x128x64` tile GEMM. Because each underlying `tcgen05.mma` advances by 16 elements along K, the compiler emits four MMA instructions to cover the full K dimension. The operand views provide the repeat axes for the explicitly selected instruction.

When reading the code, keep its four stages in mind:

1. Allocate SMEM and TMEM.
2. Copy A and B from GMEM to SMEM.
3. Issue the MMA through `Tx.apply`.
4. Load the result from TMEM into registers, then write it to GMEM.

Focus on the thread-driven tensor copies and the two `Tx.apply` calls for MMA and TMEM readback. The remaining low-level calls allocate and release TMEM, initialize the barrier, and establish synchronization. For now, treat them as the supporting steps for those four stages.

First import the modules used by the kernel:

```python
import tvm_tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.backend.cuda.script import tile as Tx
from tvm_tirx.backend.cuda.layout import mma_shared_layout, SwizzleMode
from tvm_tirx.tirx.layout import TileLayout, S
from tvm_tirx.backend.cuda.axis import TLane, TCol, tid_in_wg
from tvm_tirx.backend.cuda.lang import SMEMPool
```

The function `hgemm_v1(M, N, K)` constructs and returns a TIRx `Function`.

`@T.function` defines the GPU function, and `T.device_entry(launch=T.cuda.LaunchConfig(...))` specifies the device launch. `T.cuda.block_idx` returns the CTA coordinates in the grid, `T.cuda.warpgroup_id` returns the warpgroup index within the CTA, `T.cuda.warp_in_warpgroup` returns the warp index within the warpgroup, and `T.cuda.lane_id` returns the thread's lane ID within its warp. Together, these APIs expose the thread hierarchy used to define tile coordinates and execution guards.

```python
def hgemm_v1(M, N, K):
    a_type = "float16"
    b_type = "float16"
    d_type = "float16"
    acc_type = "float32"

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    A_layout = mma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = mma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.function
    def kernel(
        A: T.Tensor((M, K), a_type),
        B: T.Tensor((N, K), b_type),
        D: T.Tensor((M, N), d_type),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=[M // BLK_M, N // BLK_N], block=128))
        # This chapter calls the builder with M=BLK_M and N=BLK_N,
        # so the grid shape is 1x1 and both m_st and n_st are zero.
        bx, by = (T.cuda.block_idx("x"), T.cuda.block_idx("y"))
        wg_id = T.cuda.warpgroup_id()
        warp_id = T.cuda.warp_in_warpgroup()
        lane_id = T.cuda.lane_id()

        # --- Allocate SMEM ---
        pool = SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout, align=1024)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout, align=1024)
        pool.commit()

        # --- Warp 0 initializes the barrier and TMEM ---
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init.shared.b64(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                tmem_addr.ptr_to([0]), 512)

        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()

        tmem = T.cuda.decl_tmem(tmem_addr[0], ty_args=[T.Tensor((128, 512), "float32", scope="tmem", layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))])

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_mma: T.int32 = 0

        # --- Load: all threads synchronously copy A and B from GMEM to SMEM ---
        for copy_col in T.serial(BLK_K):
            Asmem.sub[:, :][warp_id * 32 + lane_id, copy_col] = A.sub[m_st:m_st + BLK_M, :][warp_id * 32 + lane_id, copy_col]
        for copy_col in T.serial(BLK_K):
            Bsmem.sub[:, :][warp_id * 32 + lane_id, copy_col] = B.sub[n_st:n_st + BLK_N, :][warp_id * 32 + lane_id, copy_col]
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        # --- Compute: one elected thread issues the MMA ---
        if warp_id == 0:
            if T.cuda.elect_sync():
                Tx.apply(
                    T.ptx["tcgen05.mma.cta_group::1.kind::f16"],
                    tmem.sub[:, :BLK_N].rearrange("(mr m) (nr n) -> mr nr m n", m=128, n=128),
                    Asmem.sub[:, :].rearrange("(mr m) (kr k) -> mr kr m k", m=128, k=16),
                    Bsmem.sub[:, :].rearrange("(nr n) (kr k) -> nr kr n k", n=128, k=16),
                    instruction_shape=(128, 128, 16), accum=False)
                T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                    mma_bar.ptr_to([0]))

        T.cuda.mbarrier_wait(mma_bar.ptr_to([0]), phase_mma)

        # --- Writeback: TMEM -> registers -> GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        T.ptx.tcgen05.fence__after_thread_sync()
        Tx.apply(
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x128.b32"],
            Dreg_wg.sub[:, :].view(4, 32, 128),
            tmem.sub[:, :BLK_N].view(4, 32, 128), scope="warpgroup")
        T.ptx.tcgen05.wait__ld.sync.aligned()
        for cast_col in T.unroll(128):
            Dreg_f16.sub[:][cast_col] = T.float16(Dreg.sub[:][cast_col])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        for store_col in T.unroll(128):
            D.sub[m_thr, n_st : n_st + BLK_N][store_col] = Dreg_f16.sub[:][store_col]

        # --- Release TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], 512)

    return kernel
```

The later GEMM chapters use this version as their starting point, then add a K loop, more output tiles, TMA, and warp specialization.

## Compile and Verify the Result

We can now compile the kernel and compare it with the same matrix multiplication computed by PyTorch. The target can simply be `"cuda"`; TVM detects the current device architecture, such as `sm_100a`. `tvm_tirx.tirx.compile` uses the TIRx lowering pipeline by default.

The compiled `ex.mod(...)` accepts PyTorch tensors directly, so no manual conversion is needed:

```python
import torch

target = tvm_tirx.target.Target("cuda")
device = torch.device("cuda")

M, N, K = 128, 128, 64
kernel = hgemm_v1(M, N, K)
with target:
    ex = tvm_tirx.tirx.compile(tvm_tirx.IRModule({"main": kernel}), target=target)

torch.cuda.empty_cache()
torch.cuda.synchronize()
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")
```

If the program prints `PASS`, the compiled kernel agrees with the PyTorch reference within the selected tolerance.

## Scope, Layout, and Dispatch

Now return to the kernel itself. Every tile operation in TIRx answers three questions: who executes it, where its data lives, and which hardware implementation it uses. These are the three design elements of scope, layout, and dispatch.

The interactive figure below extracts the key lines from the kernel. Click `Scope`, `Layout`, or `Dispatch` to highlight the lines controlled by that information.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tirx_dispatch.html?v=intro-tirx-wheel-20260811" title="Scope, Layout, and Dispatch in TIRx" loading="lazy"
        style="width:100%; min-width:960px; height:900px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```

**Scope determines which threads execute an operation.** All 128 CTA threads participate in the GMEM-to-SMEM copies. The MMA `Tx.apply` is guarded by both `warp_id == 0` and `elect_sync()`, leaving one elected thread to issue it. `T.cuda.mbarrier_wait` blocks until MMA completes; the TMEM-load `Tx.apply` then distributes the accumulator across the registers of all 128 warpgroup threads.

**Layout determines how a tile maps to physical locations.** `A_layout` and `B_layout` place A and B in SMEM using a 128-byte swizzle. The `TileLayout` of `tmem` maps the accumulator onto `TLane` and `TCol`. The `Dreg_wg` view then uses `tid_in_wg` to assign one result row to each thread. For an MMA or copy to work correctly, every operation that produces or consumes the tile must agree on the physical location of each logical element.

**Dispatch determines which hardware instruction executes a tile operation.** The first argument of `Tx.apply` explicitly names the PTX instruction: `tcgen05.mma` for GEMM and `tcgen05.ld` for TMEM readback. The operand views supply its layout and repeat axes. In this version, ordinary threads perform the GMEM-to-SMEM copies; later versions use TMA for those copies.

The compiler combines scope, layout, and dispatch to generate concrete thread-level control flow, address calculations, and hardware instructions.

## How TIRx Is Compiled

We already compiled the kernel with these two lines:

```python
target = tvm_tirx.target.Target("cuda")
ex = tvm_tirx.tirx.compile(tvm_tirx.IRModule({"main": kernel}), target=target)
```

The `Function` is first placed in an `IRModule` and passed to `tvm_tirx.tirx.compile`, which uses the TIRx lowering pipeline by default. `TileDispatch` validates and expands the PTX instructions named by `Tx.apply` over their operand views. `LowerTensorLayouts` then turns logical tensor coordinates into physical accesses.

Later passes flatten tensors, split host and device code, and generate device code, producing an `Executable` that can be invoked directly.

To inspect what the compiler produced before and after lowering, print the TIRx `Function` and the final CUDA C source:

```python
kernel.show()
print(kernel.script())

print(ex.mod.imports[0].inspect_source())
```

Comparing these two levels shows which low-level instructions a tile operation generates and how its layout and thread scope become concrete address calculations and control flow.

## Where to Go Next

The next chapter, {ref}`chap_tirx_layout_api`, introduces `TileLayout`, named axes, and swizzle in more detail. The GEMM chapters then extend this kernel with K-loop accumulation, spatial tiling, TMA, and warp specialization. The language reference separately covers data types, buffers, control flow, thread synchronization, and the rest of the TIRx syntax.
