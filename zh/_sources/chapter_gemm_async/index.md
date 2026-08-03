(chap_gemm_async)=
# 使用 TMA 为 GEMM 建立 Pipeline

:::{admonition} 本章概览
:class: overview

- 第 4 步使用 TMA 搬运 GMEM 与 SMEM 之间的 tiles：load 通过 mbarrier 等待，store 通过 async group 等待。
- 第 5 步将 A、B buffers 改成双缓冲 SMEM ring，加入预取、stage 复用和 phase 管理，为重叠 TMA load 与 MMA 建立基础。
- 第 6 步使用 tile scheduler 构建 persistent kernel，让固定数量的 CTAs 连续处理多个 output tiles，并改善 tile 的 L2 locality。
:::

上一章的 kernel 按照固定顺序处理每个 K tile：threads 先把 A、B 搬入 shared memory，等待所有写入完成，再发起 MMA 并等待计算结束；之后才开始加载下一块。这个执行顺序容易理解，也能得到正确结果，但数据搬运和 Tensor Core 计算无法重叠。

本章将在前面三步完成的 kernel 上继续优化。第 4 步用 TMA 代替 threads 搬运 A、B tiles；第 5 步为 shared memory 准备两个 stages，使 TMA 加载下一块 K tile 时，Tensor Core 可以计算当前 tile；第 6 步再加入 tile scheduler，让已经驻留的 CTAs 连续处理多个 output tiles。经过这三步，kernel 将从串行执行的 tiled GEMM 逐步变成 pipelined persistent GEMM。

(chap_tma_async)=
## 第 4 步：TMA Async Load

第 1 至第 3 步使用 `Tx.cta.copy` 搬运 A、B tiles：CTA 中的 threads 分别计算地址，再执行相应的 load 和 store。第 4 步改用 TMA，只由一个 thread 发起操作，后续的地址生成和 tile 搬运交给 TMA engine 完成。从这里开始，示例统一使用完整的 `M=N=K=4096` 规模。

> **这一步改变 Dispatch**
> - Scope：不变，仍为一个 warpgroup。
> - Layout：不变，仍使用相同的 SMEM/TMEM/register tiles。
> - Dispatch：GMEM → SMEM load 从 CTA 协作执行的 `Tx.cta.copy` 改为 TMA engine。

### 发起 TMA Load

先对比第 3 步和第 4 步的写法。

**修改前（第 3 步）**：128 个 threads 共同参与 copy，随后由 `cta_sync` 保证 shared-memory writes 可见：
```python
Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])   # all 128 threads
Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])
T.cuda.cta_sync()
```

**修改后（第 4 步）**：一个 thread 发起 TMA load，mbarrier 跟踪硬件传输何时完成：
```python
tid = warp_id * 32 + lane_id                 # 0..127 within the warpgroup
if tid == 0:  # exactly one thread starts TMA
    Tx.copy_async(Asmem, A[...], dispatch="tma")
    Tx.copy_async(Bsmem, B[...], dispatch="tma")
    T.ptx.mbarrier.arrive.expect_tx(tma_bar, byte_count)  # bytes expected from TMA
T.ptx.mbarrier.try_wait(tma_bar, phase)                  # wait before MMA reads SMEM
```

`tid` 将 warp ID 和 lane ID 合并为 warpgroup 内的 thread ID，因此 `tid == 0` 只会选中一个 thread。若四个 warps 都直接执行 `elect_sync()`，每个 warp 都会选出一个 active lane，共有四个 threads 发起 TMA。也可以先限制 `warp_id == 0` 再使用 `elect_sync()`；这里使用 `tid == 0`，写法更直接。

第 4 步仍然在每次 TMA load 后立即等待，因此 load 和 compute 还没有重叠。此时的变化只是将地址生成和 tile 搬运从 CTA threads 转交给 TMA engine，从而减少 threads 执行的搬运指令。第 5 步会加入第二个 SMEM stage，真正让 TMA load 与 MMA 同时进行。

### 等待 TMA Load 和 Store 完成

TMA load 发出后，数据传输仍会在 TMA engine 中继续执行。`cta_sync()` 只能同步 CTA 中的 threads，不能判断异步传输是否已经完成。因此，MMA 在读取 SMEM tile 前，需要通过 mbarrier 等待 TMA load 完成。

下图把这次交接画成一条从上到下推进的时间线。四条竖线依次表示发起 copy 的 thread、TMA engine、mbarrier 和使用数据的 MMA。图中用一个简化的例子说明协议：A、B tiles 各占 `2048 bytes`，两次 TMA load 共传输 `4096 bytes`。

![TMA Async Load 的同步流程](../../img/tma_sync_flow_zh.svg)

图的第 1、2 步发生在发起 copy 的 thread 上。它先为 A、B 各发出一次 `copy_async`，再执行 `arrive.expect_tx(4096)`。这条指令既向 mbarrier 报告该 thread 的一次 arrival，也登记接下来需要等待的 `4096 bytes` 异步传输。此时 pending arrival count 已经归零，但 pending bytes 仍为 4096，barrier 还不能完成。

第 3 步由 TMA engine 完成。随着 A、B 被写入 SMEM，硬件通过 `complete_tx` 扣减 pending bytes。两次传输全部结束后，pending bytes 也变为 0。第 4 步中，consumer 的 `try_wait(phase)` 此时才能通过。到了第 5 步，MMA 才开始读取已经准备好的 A、B tiles。

本节 kernel 使用相同的同步过程，只是 tile 更大。A、B tiles 都包含 `128×64` 个 fp16 元素，各占 `16384 bytes`，因此 `arrive.expect_tx` 登记的总字节数是 `32768`。

TMA store 使用另一套完成机制。Threads 将结果写入 `Dsmem` 并执行 CTA 同步后，`tid == 0` 的 thread 发起从 `Dsmem` 到 GMEM 的异步 copy。Kernel 随后通过 `cp_async.bulk.commit_group()` 提交这次 store，并使用 `cp_async.bulk.wait_group(0)` 等待完成。在 wait 返回前，`Dsmem` 不能被覆盖或复用。

### 完整 Kernel

完整 kernel 在第 3 步结构中加入 TMA load 和 store，其余部分保持不变。Imports 与前面相同：

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

这个版本封装为 `hgemm_v4(M, N, K)`。Wrapper 将依赖 shape 的 constants 和 layouts 与使用它们的 kernel 放在一起。

```python
def hgemm_v4(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    F16_SIZE = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation (now includes Dsmem for TMA store) ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((1,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # --- Barrier + TMEM init ---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
        if warp_id == 0:
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
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0

        # --- Inline helpers ---
        @T.inline
        def tma_load(k_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([0])
            })
            Tx.copy_async(Asmem[:, :],
                          A[m_st : m_st + BLK_M, k_st : k_st + BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[:, :],
                          B[n_st : n_st + BLK_N, k_st : k_st + BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([0]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE
            )

        @T.inline
        def mma(accum):
            Tx.gemm_async(
                tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                accum=accum, dispatch="tcgen05", cta_group=1
            )
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        # --- K-loop with TMA async ---
        tid = T.meta_var(warp_id * 32 + lane_id)
        for k in range(K_TILES):
            k_st = T.meta_var(k * BLK_K)

            # Single thread issues TMA load
            if tid == 0:
                tma_load(k_st)

            # Wait for TMA to finish; the mbarrier release carries SMEM
            # visibility to the subsequent MMA, so no extra fence is needed.
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([0]), phase_tma)

            # Single thread issues MMA
            if tid == 0:
                mma(accum=k != 0)

            # Wait for MMA to finish
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_tma ^= 1
            phase_mma ^= 1

        # --- TMA Store Writeback ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        # Read TMEM -> registers (async; wait.ld then cta_sync to ensure read completes)
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        # Cast fp32 -> fp16
        Tx.cast(Dreg_f16[:], Dreg[:])
        # Write registers -> Dsmem, flush, then sync
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        # TMA store: Dsmem -> GMEM. One selected thread starts the store and drains the
        # store group before Dsmem is reused.
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)

        # --- Deallocate TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

### Kernel 中的 TMA 配置

这个 kernel 的大部分结构来自第 3 步。真正决定 TMA 语义的是下面五处配置：

- **TMA config**：`{"dispatch": "tma", "cta_group": 1, "mbar": tma_bar.ptr_to([0])}` 指定 `Tx.copy_async` 使用 TMA，并通过 `tma_bar` 报告 load 完成。

- **Byte count**：`(BLK_M * BLK_K + BLK_N * BLK_K) * 2` 是两块 fp16 operand tiles 的总 byte 数；`arrive.expect_tx(...)` 将该数值登记到 mbarrier。

- **mbarrier initialization**：`init(tma_bar.ptr_to([0]), 1)` 初始化 TMA load 使用的 completion barrier。

- **`@T.inline`**：`tma_load(...)` 和 `mma(...)` 是 helper functions，在编译时展开到 kernel body 中，并可使用外围 kernel 的变量。

- **TMA store synchronization**：epilogue 先将 fp16 rows 写入 `Dsmem`。`fence.proxy_async` 和 `warpgroup_sync` 使这些由 threads 写入的 SMEM values 对 TMA store path 可见；随后通过 `commit_group()` 和 `wait_group(0)` 等待 SMEM → GMEM 传输完成。

至此，数据搬运路径已经正确，但执行顺序仍然是串行的：每次 load 完成后才会启动对应的 MMA，因此两个 engines 仍在轮流工作。下一步保持 TMA load/store path 不变，先为预取建立可以循环复用的 SMEM stages。

(chap_software_pipeline)=
## 第 5 步：Software Pipeline（`PIPE_DEPTH=2`）

第 4 步无法重叠 load 与 compute，原因在于 SMEM 中只有一对 operand tiles。下一次 load 没有独立位置可以写入；如果提前开始，就会覆盖当前 MMA 仍在读取的数据。第 5 步通过 shared memory 双缓冲解决这个存储冲突。当前单 warpgroup loop 仍会等待每次 MMA，再发起下一次 TMA load，但现在已经有独立 stages 可用于预取和循环复用。

> **这一步改变 Layout**
> - Scope：不变，仍为一个 warpgroup。
> - Layout：单个 SMEM tile pair 改为包含 `PIPE_DEPTH` 个 stages 的 ring buffer。
> - Dispatch：不变，仍使用 TMA load 和 `tcgen05` MMA。本步加入 prefetch 和 stage 复用；完整的 load/compute 重叠会在第 7 步实现。

### Pipeline 执行过程

当 `PIPE_DEPTH=2` 时，kernel 分配两个 SMEM stages，使 load path 和 MMA path 可以使用不同 slots。这是重叠数据搬运与计算的前提，但当前单 warpgroup kernel 仍会等待 MMA 完成，再发起下一次 TMA load。下图画出这组双缓冲最终要支持的目标调度；第 7 步将 TMA 和 MMA 分配给不同角色后，才会真正按这条时间线并发执行。

![*`PIPE_DEPTH=2` 的目标调度*](../../img/pipe_depth2.png)

Pipeline 启动时，两次 TMA load 先填满两个 stages。之后，loop 等待当前 stage、执行 MMA，再把 `k + PIPE_DEPTH` 对应的 tile 加载到刚刚释放的位置。这样既建立了 ring buffer，也完成了最初两块数据的预取。

代码与第 4 步有四处不同：

1. `Asmem` 和 `Bsmem` 增加前导 `PIPE_DEPTH` 维度，每个 stage 拥有独立 SMEM storage。
2. `tma_bar` 变为数组，每个 stage 对应一个 mbarrier。
3. 进入 main K-loop 前，kernel 预取最初两个 stages。
4. K-loop 使用 `stage = k % PIPE_DEPTH`：等待当前 stage，对其执行 MMA，再复用它加载 `k + PIPE_DEPTH`。

### Pipeline 机制

**1. Prefetch**：main loop 开始前，先加载最初 `PIPE_DEPTH` 个 stages，使第一个 iteration 进入时已经有数据可用：
```python
for s in range(min(PIPE_DEPTH, K_TILES)):
    tma_load(s, s * BLK_K)
```

**2. Main loop**：对每个 K tile，先等待对应 stage 准备完成，再执行 MMA；该 stage 释放后，立即用它加载前方 `PIPE_DEPTH` 距离处的 tile：
```python
stage = k % PIPE_DEPTH
wait(tma_bar[stage], phase_tma)
mma(stage, accum)
wait(mma_bar[0], phase_mma)
phase_mma ^= 1
tma_load(stage, next_k * BLK_K)
```

**3. Phase 管理**：前面的异步同步章节已经说明，同一个 mbarrier 每完成一轮，phase 就会翻转。这里的两个 phase 变量更新频率不同，是因为它们分别跟踪一个 MMA accumulator 和多个 SMEM stages。

所有 K iterations 都通过 `mma_bar.ptr_to([0])` 跟踪同一个 TMEM accumulator，因此 `phase_mma` 每轮都要翻转。TMA 则为每个 SMEM stage 分配一个 barrier；只有 ring buffer 再次使用同一个 stage 时，对应的 barrier 才会进入下一轮。因此，`phase_tma` 只在 stage index 绕回 0 时翻转：
```python
if stage == PIPE_DEPTH - 1:
    phase_tma ^= 1
```

**Pipeline 推演**：取 `PIPE_DEPTH=2`、`K_TILES=5`，追踪 main loop。对每个 `k`，列出 `stage`、传给 waits 的 `phase_tma` 和 `phase_mma`，以及是否发起新的 prefetch。`phase_tma` 在哪里翻转？为什么最后两个 iterations 不会再 prefetch？

### 完整 Kernel

完整 kernel 保留第 4 步的 TMA load/store path，并加入 staged buffers 和 phase logic。Imports 不变：

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

这个版本封装为 `hgemm_v5(M, N, K)`。`PIPE_DEPTH=2` 指定两个 pipeline stages，也就是双缓冲：

```python
PIPE_DEPTH = 2

def hgemm_v5(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    # Double-buffered layouts: first dimension is pipeline stage
    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        # Double-buffered TMA barriers (one per stage), single MMA barrier
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # Initialize barriers: PIPE_DEPTH for TMA, 1 for MMA
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                for s in range(PIPE_DEPTH):
                    T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0

        @T.inline
        def tma_load(stage, k_offset):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        tid = T.meta_var(warp_id * 32 + lane_id)

        # === Prefetch: load first PIPE_DEPTH stages ===
        if tid == 0:
            for s in range(min(PIPE_DEPTH, K_TILES)):
                tma_load(s, s * BLK_K)

        # === Main loop ===
        for k in range(K_TILES):
            stage = k % PIPE_DEPTH

            # Wait for TMA to finish loading this stage
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)

            # MMA on this stage's data
            if tid == 0:
                mma(stage, accum=(k != 0))

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

            # Issue next prefetch load (k + PIPE_DEPTH)
            next_k = k + PIPE_DEPTH
            if next_k < K_TILES:
                if tid == 0:
                    tma_load(stage, next_k * BLK_K)

            # TMA phase flips when stage wraps around
            if stage == PIPE_DEPTH - 1:
                phase_tma ^= 1

        # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.cast(Dreg_f16[:], Dreg[:])
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)

        # Deallocate TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

(chap_persistent_kernel)=
## 第 6 步：Persistent Kernel 与 Tile Scheduler

前面的几步都在优化单个 output tile 内部的执行。第 6 步把关注点移到 tiles 之间的调度。

第 5 步为每个 $128\times128$ output tile 启动一个 CTA。对于 $4096\times4096$ 的输出，一共需要 1024 个 CTAs。每个 CTA 都要单独完成初始化，计算完一个 tile 后便退出。

Persistent kernel 则只启动固定数量的 CTAs，让每个 CTA 依次处理多个 tiles。这样做有两个好处：初始化开销可以分摊到多个 tiles 上；tile 的分配也转移到了 kernel 内部，scheduler 可以按有利于复用 operands 的顺序安排工作。

> **这一步改变 Scope**
> - Scope：固定数量的 persistent CTAs，每个 CTA 通过 scheduler 循环处理多个 output tiles。
> - Layout：不变，每个 tile 仍使用相同的 SMEM、TMEM 和 register 数据路径。
> - Dispatch：不变。

### Persistent Scheduling

Persistent kernel 使用一个较小的一维 grid。本例设置 `SM_COUNT=148`，因此启动 148 个 persistent CTAs。每个 CTA 从 scheduler 获取一个 output tile，完成后再获取下一个，直到所有 tiles 都处理完毕。`SM_COUNT` 控制同时工作的 CTAs 数量，并不表示这些 CTAs 会与 SM 固定绑定。

由于一个 CTA 会连续处理多个 tiles，它只需申请一次 TMEM、初始化一次 barriers，并创建一次 scheduler state。这些资源可以一直保留到该 CTA 完成全部任务。

Scheduler 还会调整 tiles 的处理顺序。`l2_group_size=8` 表示把 M 方向上连续 8 行 output tiles 分为一组。组内先固定一个 N tile column，依次处理这 8 行，再移动到下一个 N tile column。这样，读取同一个 B tile 的任务会相邻出现，同一组 A tiles 也会在较短时间内再次被访问。各 CTA 仍然独立搬运数据，但这种顺序更有利于 L2 cache 命中。

```python
bx = T.cta_id([SM_COUNT])  # 1D persistent grid

tile_scheduler = ClusterPersistentScheduler2D(
    "ts",
    num_m_tiles=M // BLK_M,
    num_n_tiles=N // BLK_N,
    l2_group_size=8,
    num_clusters=SM_COUNT
)
tile_scheduler.init(bx)
```

CTA 开始处理下一块 output tile 时，还会继续使用同一组 TMA 和 MMA barriers，因此本地记录的 phase parity 必须与 barrier 的当前状态一致。

当前参数下，每个 output tile 包含 64 次 K iterations。`mma_bar` 使用 64 次，两个 TMA stage barriers 各使用 32 次。由于这些次数都是偶数，处理完一个 tile 后，各 barrier 都回到初始 parity，下一块 tile 可以重新从 0 开始：

```python
while tile_scheduler.valid():
    phase_tma: T.int32 = 0
    phase_mma: T.int32 = 0
    ...
```

如果修改 `K`、`BLK_K` 或 `PIPE_DEPTH`，使某个 barrier 的使用次数变为奇数，就不能直接将对应的 phase parity 重置为 0。当前 wrapper 使用 assertion 限定了支持的参数组合。

### 完整 Kernel

从结构上看，这个 kernel 只是在第 5 步的 pipeline 外增加了一层 tile loop。新增的依赖只有 scheduler：

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D
```

Grid dimension 由 `(M//BLK_M, N//BLK_N)` 改为 `SM_COUNT`，每个 CTA 要处理的 tile 则由 `ClusterPersistentScheduler2D` 分配：

```python
SM_COUNT = 148  # Number of SMs on NVIDIA B200 GPU
PIPE_DEPTH = 2

def hgemm_v6(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    assert K % BLK_K == 0, "K must be divisible by BLK_K"
    K_TILES = K // BLK_K
    assert K_TILES % (2 * PIPE_DEPTH) == 0, (
        "K_TILES must be divisible by 2 * PIPE_DEPTH"
    )

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 1D grid: one CTA per SM (not a 2D grid anymore!)
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation (same as Step 5) ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # --- Barrier + TMEM init (same as Step 5) ---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            for s in range(PIPE_DEPTH):
                T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        # Tile scheduler: assigns tiles to CTAs in L2-friendly order
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts",
            num_m_tiles=M // BLK_M,
            num_n_tiles=N // BLK_N,
            l2_group_size=8,
            num_clusters=SM_COUNT
        )
        tile_scheduler.init(bx)

        tid = T.meta_var(warp_id * 32 + lane_id)

        @T.inline
        def tma_load(stage, k_offset, m_st, n_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        # === Outer loop: iterate over tiles ===
        while tile_scheduler.valid():
            # Get current tile position from scheduler
            m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
            n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

            # === Inner loop: same pipeline as Step 5 ===
            phase_tma: T.int32 = 0
            phase_mma: T.int32 = 0

            # Prefetch first PIPE_DEPTH stages
            if tid == 0:
                for s in range(min(PIPE_DEPTH, K_TILES)):
                    tma_load(s, s * BLK_K, m_st, n_st)

            # Main K-loop
            for k in range(K_TILES):
                stage = k % PIPE_DEPTH
                T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)
                if tid == 0:
                    mma(stage, accum=(k != 0))
                T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                phase_mma ^= 1
                next_k = k + PIPE_DEPTH
                if next_k < K_TILES:
                    if tid == 0:
                        tma_load(stage, next_k * BLK_K, m_st, n_st)
                if stage == PIPE_DEPTH - 1:
                    phase_tma ^= 1

            # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
            Dreg = T.alloc_local((BLK_N,), acc_type)
            Dreg_f16 = T.alloc_local((BLK_N,), d_type)
            Dreg_wg = Dreg.view(128, BLK_N,
                                layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
            Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
            T.ptx.tcgen05.wait.ld()
            T.cuda.cta_sync()
            Tx.cast(Dreg_f16[:], Dreg[:])
            Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.warpgroup_sync(10)
            if tid == 0:
                Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                              Dsmem[:, :], dispatch="tma")
                T.ptx.cp_async.bulk.commit_group()
                T.ptx.cp_async.bulk.wait_group(0)
            T.cuda.warpgroup_sync(10)

            T.cuda.cta_sync()
            tile_scheduler.next_tile()  # Move to next tile

        # Deallocate TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

## 练习

1. 第 4 步中的 `arrive.expect_tx` 使用 `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` bytes。如果这个 byte count 过小或过大，mbarrier 的等待会发生什么？
2. 第 5 步中，为什么每个 SMEM stage 都需要自己的 TMA barrier，而不能让两个 stages 共用一个 `tma_bar`？
3. 第 6 步中，`BLK_M=BLK_N=128` 时，一个 $4096\times4096$ 输出包含多少个 output tiles？若 `SM_COUNT=148`，每个 persistent CTA 平均处理多少个 tiles？
