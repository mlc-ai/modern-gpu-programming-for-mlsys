(chap_tirx_primer)=
# TIRx 入门

:::{admonition} 运行示例
:class: note

安装 TIRx 和 CUDA 编译 bindings：

```bash
python -m pip install apache-tvm-tirx cuda-bindings
python -c "import tvm_tirx; print(tvm_tirx.__version__)"
```

示例使用 `apache-tvm-tirx==0.1.0a3`、支持 CUDA 的 PyTorch 和 Blackwell GPU
（`sm_100a`，例如 B200）验证。最小 wheel 已包含 CPU launcher 所需的 native
LLVM；ARM/Wasm 交叉编译使用 `apache-tvm-tirx[all-targets]`。
仓库 README 列出示例依赖和 `scripts/run_examples.py` 的运行方法。
:::

TIRx 用带类型的 tensor program 表达 GPU 工作，明确保留三个选择：哪些线程执行
操作（scope）、operand 元素放在哪里（layout）、使用哪条硬件指令（instruction）。
`T.cuda.tile.apply` 将调用者指定的 PTX 指令应用到显式准备的 operand views，
检查这些 views 的约定，不会自动更换 opcode 或推断替代 layout。
普通线程的数据访问仍可使用标量 tensor load/store。

## 第一个 TIRx Kernel

Kernel 计算 `D = A @ B.T`。A、B 为 `128x64` FP16，D 为 `128x128`。
数据经过 `GMEM -> SMEM -> tcgen05.mma -> TMEM -> registers -> GMEM`。
128 个线程各加载一行；选出的单个线程发出 MMA；warpgroup 等待结果后读出 TMEM
并写回 FP16 输出。

CUDA layout axes 和内存 helper 从 backend 导入：

```python
import tvm_tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.backend.cuda.layout import mma_shared_layout, SwizzleMode
from tvm_tirx.tirx.layout import TileLayout, S
from tvm_tirx.backend.cuda.axis import TLane, TCol, tid_in_wg
from tvm_tirx.backend.cuda.lang import SMEMPool
```

`@T.function` 定义 TIRx function，`T.Tensor` 标注 tensor 参数。
`T.device_entry(launch=T.cuda.LaunchConfig(...))` 指定 grid 和 block；
`T.cuda.block_idx`、`warpgroup_id`、`warp_in_warpgroup`、`lane_id` 读取硬件坐标。
存储分配与指令使用的 view 分开描述。

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
        # 本章调用时 M=BLK_M、N=BLK_N，
        # 所以 grid shape 为 1x1，m_st 和 n_st 都是 0。
        bx, by = (T.cuda.block_idx("x"), T.cuda.block_idx("y"))
        wg_id = T.cuda.warpgroup_id()
        warp_id = T.cuda.warp_in_warpgroup()
        lane_id = T.cuda.lane_id()

        # --- 申请 SMEM ---
        pool = SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout, align=1024)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout, align=1024)
        pool.commit()

        # --- 由 warp 0 初始化 barrier 和 TMEM ---
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

        # --- Load：所有 threads 同步地将 A、B 从 GMEM 搬入 SMEM ---
        for copy_col in T.serial(BLK_K):
            Asmem.sub[:, :][warp_id * 32 + lane_id, copy_col] = A.sub[m_st:m_st + BLK_M, :][warp_id * 32 + lane_id, copy_col]
        for copy_col in T.serial(BLK_K):
            Bsmem.sub[:, :][warp_id * 32 + lane_id, copy_col] = B.sub[n_st:n_st + BLK_N, :][warp_id * 32 + lane_id, copy_col]
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        # --- Compute：由一个被选中的 thread 发出 MMA ---
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

        # --- Writeback：TMEM -> registers -> GMEM ---
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

        # --- 释放 TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], 512)

    return kernel
```

MMA 的 D view 使用 `(M_repeat,N_repeat,M,N)`，A 使用
`(M_repeat,K_repeat,M,K)`，B 使用 `(N_repeat,K_repeat,N,K)`。
`instruction_shape=(128,128,16)` 配合四个 K repeats 覆盖 K=64。
`rearrange` 只准备已有存储的 view，不搬动数据。

线程写入 SMEM 后，先通过 `fence.proxy.async` 发布给异步代理，再同步 CTA。
`tcgen05.commit` 将异步 MMA 的完成关联到 mbarrier，`T.cuda.mbarrier_wait`
等待完成。TMEM 读取明确指定 `tcgen05.ld`，随后等待 load 完成。

## 编译与验证

`tvm_tirx.tirx.compile` 使用默认 lowering pipeline 并返回 `Executable`。
其 module 直接接受 PyTorch tensor：

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

输出 `PASS` 表示在指定容差下与 FP32 PyTorch reference 一致。
代码应放在文件或 notebook cell 中，供 script parser 检查 source。

## Scope、Layout 与 Dispatch

`scope="warpgroup"` 指定哪些线程参与 TMEM load。Register view 将每行映射到
一个 `tid_in_wg`。TMEM 使用 `TLane`、`TCol`；SMEM 使用 MMA-compatible swizzle。
完整 PTX 名称决定具体指令和修饰符，切换数据路径时也要提供对应的 operand views。

点击 Scope、Layout 或 Dispatch，高亮 kernel 中对应的关键语句。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo_zh/tirx_dispatch.html?v=tirx-a3-20261026" title="TIRx Scope, Layout, and Dispatch" loading="lazy"
        style="width:100%; min-width:960px; height:900px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```

## TIRx 如何编译

```python
target = tvm_tirx.target.Target("cuda")
ex = tvm_tirx.tirx.compile(tvm_tirx.IRModule({"main": kernel}), target=target)
```

Pipeline 先绑定 target，再 dispatch 显式 tile 指令、降低 tensor layout、展开访问，
最后拆分 host/device 并生成代码。可以查看原始 IR 和最终 CUDA source：

```python
kernel.show()
print(kernel.script())

print(ex.mod.imports[0].inspect_source())
```

编译器附录介绍完整 pass 流程。下一章 {ref}`chap_tirx_layout_api` 解释 layout 和
named axes；后续 GEMM 章节加入 K-loop、空间 tiling、TMA 和 warp specialization。
