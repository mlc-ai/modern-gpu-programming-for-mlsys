(chap_tirx_primer)=
# TIRx 入门

:::{admonition} 概览
:class: overview

- TIRx 是用于编写 GPU kernel 的 Python DSL。程序可以直接使用 threads、SMEM、TMEM、barriers 和 Tensor Core 等硬件概念，同时保留结构化的中间表示。
- TIRx 中的 tile 操作主要由三项信息决定：哪些 threads 执行操作（scope）、数据采用什么布局（layout），以及操作通过哪条硬件路径实现（dispatch）。
- 本章从一个可以运行的单 tile GEMM 出发，依次介绍 TIRx kernel 的写法、编译与验证方法，以及 scope、layout 和 dispatch 如何共同决定 kernel 的行为。
:::

:::{admonition} 运行环境
:class: note

本章示例需要 Blackwell GPU（`sm_100a`，例如 B200）、TIRx 编译器和支持 CUDA 的 PyTorch。TIRx 位于 Apache TVM wheel 的 `tvm.tirx` 模块中；通过 NVRTC 编译 CUDA 代码时还需要 `cuda-bindings`，可以一起安装：

```bash
pip install apache-tvm cuda-bindings
```

安装后可以运行下面的命令，确认 TVM 和 TIRx 能够正常导入：

```bash
python -c "import tvm, tvm.tirx; print(tvm.__version__)"
```

后续章节中的可运行示例也使用同一套环境。
:::

第一部分介绍了现代 GPU 的执行模型、数据布局，以及 TMA、Tensor Core、TMEM 和异步同步等硬件机制。接下来需要把这些机制组织成真正可以运行的 kernel。

直接使用 CUDA 或 PTX 当然可以完成这项工作，但一段底层程序往往会把几个关键决定分散在 intrinsic 参数、地址计算和代码约定中：某项操作由哪些 threads 执行，operand tile 存放在哪里，以及最终使用哪条硬件指令。这些信息虽然都写在程序里，却很难作为一个整体被编译器检查和变换。

TIRx（Tensor IR next）是一种 Python DSL，它将这三类决定显式写入结构化 IR：

- **Scope**：哪些 threads 执行一项操作；
- **Layout**：逻辑 tile 如何映射到 memory、lanes 或 registers；
- **Dispatch**：一项 tile 操作通过哪种硬件实现。

TIRx 仍然直接使用 threads、SMEM、TMEM、barriers 和 `tcgen05.mma` 等硬件概念。区别在于，这些信息现在具有明确的 IR 结构，编译器可以据此检查程序并生成底层代码。

与其先单独罗列语法，本章会从一个完整的 kernel 开始。先运行一个最小的单 tile GEMM，再回头分析其中的 scope、layout 和 dispatch，最后查看它如何被编译。

## 第一个 TIRx Kernel

下面的 kernel 计算：

```text
D = A × B^T
```

其中 `A` 和 `B` 的 shape 都是 `128×64`，输出 `D` 的 shape 是 `128×128`。这个示例只处理一个 `128×128` output tile，因此 grid 也只有一个 CTA。Kernel 的数据路径可以概括为：

```text
A/B: GMEM -> SMEM -> tcgen05.mma
D:   tcgen05.mma -> TMEM -> registers -> GMEM
```

矩阵乘法在 TIRx 中写成一次 `Tx.gemm_async` tile 操作。这项操作描述完整的 `128×128×64` tile GEMM；当前 `tcgen05.mma` 每次处理 16 个 K 元素，因此编译器会沿 K 维生成 4 次 MMA。具体指令序列由编译器根据 shape、layout 和 dispatch 决定。

阅读下面的代码时，可以先抓住四个阶段：

1. 申请 SMEM 和 TMEM；
2. 将 A、B 从 GMEM 搬入 SMEM；
3. 通过 `Tx.gemm_async` 发起 MMA；
4. 将结果从 TMEM 读回 registers，再写入 GMEM。

其中最关键的三项 tile 操作是 `Tx.cta.copy`、`Tx.gemm_async` 和 `Tx.wg.copy_async`。其余 PTX 调用用于申请和释放 TMEM、初始化 barrier 并建立同步；本章先把它们看作完成这几个阶段所需的底层步骤。

先导入这个 kernel 使用的模块：

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

函数 `hgemm_v1(M, N, K)` 会构造并返回一个 TIRx `PrimFunc`：

`@T.prim_func` 用来定义这个 GPU 函数，`T.device_entry()` 标记 device 代码的入口。`T.cta_id` 取得 CTA 在 grid 中的坐标，`T.warpgroup_id` 取得 warpgroup 在 CTA 中的编号，`T.warp_id_in_wg` 取得 warp 在 warpgroup 中的编号，`T.lane_id` 则取得 thread 在 warp 中的 lane ID。后面的 scope 和条件判断会使用这些值。

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
        # 本章调用时 M=BLK_M、N=BLK_N，
        # 所以 grid shape 为 1x1，m_st 和 n_st 都是 0。
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- 申请 SMEM ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        # --- 由 warp 0 初始化 barrier 和 TMEM ---
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

        # --- Load：所有 threads 同步地将 A、B 从 GMEM 搬入 SMEM ---
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()

        # --- Compute：由一个被选中的 thread 发出 MMA ---
        if warp_id == 0:
            if T.ptx.elect_sync():
                Tx.gemm_async(
                    tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                    accum=False, dispatch="tcgen05", cta_group=1
                )
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)

        # --- Writeback：TMEM -> registers -> GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

        # --- 释放 TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

后面的 GEMM 章节会以这个版本为起点，逐步加入 K-loop、更多 output tiles、TMA 和 warp specialization。

## 编译并验证结果

下面先编译 kernel，再用 PyTorch 计算同样的矩阵乘法作为参考结果。目标写成 `"cuda"` 即可；TVM 会从当前设备检测具体架构，例如 `sm_100a`。参数 `tir_pipeline="tirx"` 用来选择 TIRx lowering pipeline。

编译后的 `ex.mod(...)` 可以直接接收 PyTorch tensors，不需要手工转换数据：

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

如果最后输出 `PASS`，说明编译后的 kernel 与 PyTorch 参考实现的结果在允许误差内一致。

## Scope、Layout 与 Dispatch

现在回头看这段 kernel。TIRx 中的每项 tile 操作都需要回答三个问题：由谁执行、数据放在哪里，以及使用哪种硬件实现。对应的三个设计要素就是 scope、layout 和 dispatch。

下面的交互图摘出了 kernel 中的关键代码。点击 `Scope`、`Layout` 或 `Dispatch`，可以高亮受该项信息控制的代码行。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo_zh/tirx_dispatch.html?v=intro-tirx-wheel-20260723" title="TIRx 中的 Scope、Layout 与 Dispatch" loading="lazy"
        style="width:100%; min-width:960px; height:900px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```

**Scope 决定哪些 threads 执行操作。** `Tx.cta.copy(...)` 由整个 CTA 协作执行，在这个 kernel 中共有 128 个 threads 参与 GMEM 到 SMEM 的 copy。`Tx.gemm_async(...)` 位于 `warp_id == 0` 和 `elect_sync()` 两层条件内，最终只由一个被选中的 thread 发出。MMA 完成后，`mbarrier.try_wait` 等待计算完成；随后，`Tx.wg.copy_async(...)` 由整个 warpgroup 协作，将 TMEM accumulator 分配到 128 个 threads 的 registers 中。

**Layout 决定 tile 如何映射到物理位置。** `A_layout` 和 `B_layout` 指定 A、B 在 SMEM 中采用 128-byte swizzle；`tmem` 的 `TileLayout` 将 accumulator 映射到 `TLane` 和 `TCol`；`Dreg_wg` 的 view 再通过 `tid_in_wg` 指定每个 thread 读取哪一行结果。MMA 和 copy 的两端必须采用彼此匹配的 layout，硬件才能把同一批元素解释为正确的矩阵 tile。

**Dispatch 决定 tile 操作使用哪种硬件实现。** `Tx.gemm_async` 表示一个异步 tile GEMM，`dispatch="tcgen05"` 进一步要求编译器选择 Blackwell 的 `tcgen05.mma` 路径。这个版本中的 GMEM 到 SMEM copy 由普通 threads 完成；后面的版本会将同一类 tile copy 改为 TMA。

编译器会结合 scope、layout 和 dispatch，生成具体的 thread-level 控制流、地址计算和硬件指令。

## TIRx 如何编译

前面已经用下面两行代码编译了 kernel：

```python
target = tvm.target.Target("cuda")
ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
```

`PrimFunc` 首先被放入一个 `IRModule`，随后交给 `tvm.compile`。`tir_pipeline="tirx"` 会启动 TIRx lowering pipeline。其中的核心 pass `LowerTIRx` 根据每项 tile primitive 的 scope、layout 和 dispatch 选择具体实现，将 `Tx.gemm_async`、`Tx.cta.copy` 等高层 tile 操作展开成更底层的 TIR。

后续 passes 再完成 buffer flatten、host/device 拆分和设备代码生成，最终得到可以直接调用的 `Executable`。

如果想查看编译器在 lowering 前后生成了什么，可以分别检查 TIRx `PrimFunc` 和最终的 CUDA C 代码：

```python
kernel.show()
print(kernel.script())

print(ex.mod.imports[0].inspect_source())
```

对照这两层代码，可以看到一个 tile 操作最终生成了哪些底层指令，也可以检查 layout 和 thread scope 如何变成具体的地址计算与控制流。

## 接下来

下一章将进一步介绍 `TileLayout`、命名轴和 swizzle。之后的 GEMM 章节会继续扩展这个 kernel，加入 K-loop accumulation、空间 tiling、TMA 和 warp specialization。语言参考将另行介绍数据类型、buffers、控制流和线程同步等语法。
