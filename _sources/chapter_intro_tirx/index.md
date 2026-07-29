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

The examples in this chapter require a Blackwell GPU (`sm_100a`, such as a B200), the TIRx compiler, and a CUDA-enabled build of PyTorch. TIRx is available as the `tvm.tirx` module in the Apache TVM wheel. Compiling CUDA through NVRTC also requires `cuda-bindings`, so install both packages:

```bash
pip install apache-tvm cuda-bindings
```

After installation, verify that TVM and TIRx import correctly:

```bash
python -c "import tvm, tvm.tirx; print(tvm.__version__)"
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

The matrix multiplication is expressed as one `Tx.gemm_async` tile operation. This single operation describes the full `128x128x64` tile GEMM. Because each underlying `tcgen05.mma` advances by 16 elements along K, the compiler emits four MMA instructions to cover the full K dimension. It derives their exact sequence from the shape, layout, and dispatch information.

When reading the code, keep its four stages in mind:

1. Allocate SMEM and TMEM.
2. Copy A and B from GMEM to SMEM.
3. Issue the MMA through `Tx.gemm_async`.
4. Load the result from TMEM into registers, then write it to GMEM.

The three tile operations to focus on are `Tx.cta.copy`, `Tx.gemm_async`, and `Tx.wg.copy_async`. The remaining low-level calls allocate and release TMEM, initialize the barrier, and establish synchronization. For now, treat them as the supporting steps for those four stages.

First import the modules used by the kernel:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

The function `hgemm_v1(M, N, K)` constructs and returns a TIRx `PrimFunc`.

`@T.prim_func` defines the GPU function, and `T.device_entry()` marks the entry to device code. `T.cta_id` returns the CTA coordinates in the grid, `T.warpgroup_id` returns the warpgroup index within the CTA, `T.warp_id_in_wg` returns the warp index within the warpgroup, and `T.lane_id` returns the thread's lane ID within its warp. Together, these APIs expose the thread hierarchy used to define tile coordinates and execution guards.

```python
def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # This chapter calls the builder with M=BLK_M and N=BLK_N,
        # so the grid shape is 1x1 and both m_st and n_st are zero.
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocate SMEM ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        # --- Warp 0 initializes the barrier and TMEM ---
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_mma: T.int32 = 0

        # --- Load: all threads synchronously copy A and B from GMEM to SMEM ---
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()

        # --- Compute: one elected thread issues the MMA ---
        if warp_id == 0:
            if T.ptx.elect_sync():
                Tx.gemm_async(
                    tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                    accum=False, dispatch="tcgen05", cta_group=1
                )
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)

        # --- Writeback: TMEM -> registers -> GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

        # --- Release TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

The later GEMM chapters use this version as their starting point, then add a K loop, more output tiles, TMA, and warp specialization.

## Compile and Verify the Result

We can now compile the kernel and compare it with the same matrix multiplication computed by PyTorch. The target can simply be `"cuda"`; TVM detects the current device architecture, such as `sm_100a`. The argument `tir_pipeline="tirx"` selects the TIRx lowering pipeline.

The compiled `ex.mod(...)` accepts PyTorch tensors directly, so no manual conversion is needed:

```python
import torch

target = tvm.target.Target("cuda")
device = torch.device("cuda")

M, N, K = 128, 128, 64
kernel = hgemm_v1(M, N, K)
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

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
<iframe src="../demo/tirx_dispatch.html?v=intro-tirx-wheel-20260723" title="Scope, Layout, and Dispatch in TIRx" loading="lazy"
        style="width:100%; min-width:960px; height:900px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```

**Scope determines which threads execute an operation.** `Tx.cta.copy(...)` is executed cooperatively by the entire CTA, so all 128 threads in this kernel participate in the GMEM-to-SMEM copy. `Tx.gemm_async(...)` is guarded by both `warp_id == 0` and `elect_sync()`, leaving one elected thread to issue it. The subsequent `mbarrier.try_wait` blocks until the MMA completes; `Tx.wg.copy_async(...)` then cooperatively distributes the TMEM accumulator across the registers of all 128 threads in the warpgroup.

**Layout determines how a tile maps to physical locations.** `A_layout` and `B_layout` place A and B in SMEM using a 128-byte swizzle. The `TileLayout` of `tmem` maps the accumulator onto `TLane` and `TCol`. The `Dreg_wg` view then uses `tid_in_wg` to assign one result row to each thread. For an MMA or copy to work correctly, every operation that produces or consumes the tile must agree on the physical location of each logical element.

**Dispatch determines which hardware implementation executes a tile operation.** `Tx.gemm_async` denotes an asynchronous tile GEMM, while `dispatch="tcgen05"` specifically selects Blackwell's `tcgen05.mma` path. In this version, ordinary threads perform the GMEM-to-SMEM copies; later versions dispatch those copies to TMA instead.

The compiler combines scope, layout, and dispatch to generate concrete thread-level control flow, address calculations, and hardware instructions.

## How TIRx Is Compiled

We already compiled the kernel with these two lines:

```python
target = tvm.target.Target("cuda")
ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
```

The `PrimFunc` is first placed in an `IRModule` and then passed to `tvm.compile`. Setting `tir_pipeline="tirx"` starts the TIRx lowering pipeline. Its central pass, `LowerTIRx`, uses the scope, layout, and dispatch of each tile primitive to select a concrete implementation and lower operations such as `Tx.gemm_async` and `Tx.cta.copy` into lower-level TIR.

Later passes flatten buffers, split host and device code, and generate device code, producing an `Executable` that can be invoked directly.

To inspect what the compiler produced before and after lowering, print the TIRx `PrimFunc` and the final CUDA C source:

```python
kernel.show()
print(kernel.script())

print(ex.mod.imports[0].inspect_source())
```

Comparing these two levels shows which low-level instructions a tile operation generates and how its layout and thread scope become concrete address calculations and control flow.

## Where to Go Next

The next chapter, {ref}`chap_tirx_layout_api`, introduces `TileLayout`, named axes, and swizzle in more detail. The GEMM chapters then extend this kernel with K-loop accumulation, spatial tiling, TMA, and warp specialization. The language reference separately covers data types, buffers, control flow, thread synchronization, and the rest of the TIRx syntax.
