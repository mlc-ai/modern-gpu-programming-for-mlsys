(chap_tirx_primer)=
# Introduction to TIRx

:::{admonition} Running the examples
:class: note

Install TIRx and CUDA compilation bindings:

```bash
python -m pip install apache-tvm-tirx cuda-bindings
python -c "import tvm_tirx; print(tvm_tirx.__version__)"
```

The examples are tested with `apache-tvm-tirx==0.1.0a3`, a CUDA-enabled PyTorch build,
and a Blackwell GPU (`sm_100a`, such as a B200). The minimal compiler wheel includes
native LLVM for the CPU launcher. ARM/Wasm cross-compilation uses the separate
`apache-tvm-tirx[all-targets]` extra. Follow the repository README to install the
example dependencies and run `scripts/run_examples.py`.
:::

TIRx represents GPU work as typed tensor programs. Three choices remain visible:
which threads execute an operation (**scope**), where each operand element lives
(**layout**), and which hardware instruction executes (**instruction**).
`T.cuda.tile.apply` repeats a caller-selected PTX instruction over explicit operand
views. It validates those views; it does not select a different opcode or infer a
replacement layout. Scalar tensor loads and stores remain available for ordinary
thread-driven work.

## A First TIRx Kernel

The kernel computes `D = A @ B.T`. A and B are `128x64` FP16 tensors; D is
`128x128`. Its data path is `GMEM -> SMEM -> tcgen05.mma -> TMEM -> registers -> GMEM`.
All 128 threads load one row each. One elected thread issues MMA, then the warpgroup
reads the completed accumulator and writes FP16 output.

Import layout axes and memory helpers from the CUDA backend:

```python
import tvm_tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.backend.cuda.layout import mma_shared_layout, SwizzleMode
from tvm_tirx.tirx.layout import TileLayout, S
from tvm_tirx.backend.cuda.axis import TLane, TCol, tid_in_wg
from tvm_tirx.backend.cuda.lang import SMEMPool
```

`@T.function` defines a TIRx function. `T.Tensor` annotates tensor parameters.
`T.device_entry(launch=T.cuda.LaunchConfig(...))` specifies the grid and block;
`T.cuda.block_idx`, `warpgroup_id`, `warp_in_warpgroup` and `lane_id` read hardware
coordinates. Allocation is separate from the views used by each instruction.

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
                T.cuda.tile.apply(
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
        T.cuda.tile.apply(
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

The MMA views have axes `(M_repeat,N_repeat,M,N)` for D,
`(M_repeat,K_repeat,M,K)` for A and `(N_repeat,K_repeat,N,K)` for B. The explicit
`instruction_shape=(128,128,16)` and four K repeats cover K=64. `rearrange` prepares
these views over the existing storage; it does not move data.

The synchronous SMEM writes are published with `fence.proxy.async` before the CTA
barrier. `tcgen05.commit` associates the asynchronous MMA with an mbarrier;
`T.cuda.mbarrier_wait` waits for completion. TMEM loads use the explicitly named
`tcgen05.ld` instruction, followed by its load wait.

## Compile and Verify the Result

`tvm_tirx.tirx.compile` uses the default lowering pipeline and returns an
`Executable`. Its module accepts PyTorch tensors directly:

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

`PASS` confirms agreement with the FP32 PyTorch reference at the specified tolerance.
Store this code in a file or notebook cell so the script parser can inspect it.

## Scope, Layout, and Dispatch

`scope="warpgroup"` states which threads participate in the TMEM load. Its register
view assigns one output row to each `tid_in_wg`. The TMEM layout uses `TLane` and
`TCol`, while SMEM uses an MMA-compatible swizzle. The exact PTX spelling chooses
the instruction and modifiers; changing a physical data path requires changing
that spelling and providing matching operand views.

Select Scope, Layout or Dispatch to highlight the key statements from the kernel.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tirx_dispatch.html?v=tirx-a3-20261026" title="TIRx Scope, Layout, and Dispatch" loading="lazy"
        style="width:100%; min-width:960px; height:900px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```

## How TIRx Is Compiled

```python
target = tvm_tirx.target.Target("cuda")
ex = tvm_tirx.tirx.compile(tvm_tirx.IRModule({"main": kernel}), target=target)
```

The pipeline binds the target, dispatches explicit tile instructions, lowers tensor
layouts, flattens accesses and splits host/device code. Inspect the authored IR and
final CUDA source:

```python
kernel.show()
print(kernel.script())

print(ex.mod.imports[0].inspect_source())
```

The compiler appendix describes the passes in more detail. The next chapter,
{ref}`chap_tirx_layout_api`, explains layouts and named axes; the GEMM chapters add
K-loop accumulation, tiling, TMA and warp specialization.
