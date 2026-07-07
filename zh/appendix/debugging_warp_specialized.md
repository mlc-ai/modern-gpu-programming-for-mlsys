(chap_warp_spec_debug)=
# 调试 Warp-Specialized Kernel

{ref}`chap_gemm_advanced` 中的 GEMM Steps 7-9 会重叠 TMA load、`tcgen05` MMA 以及 TMEM/SMEM writeback。同样的调试方法也适用于 Flash Attention handoff：识别角色，识别每个角色拥有的 storage，然后用这个模型检查生成的 CUDA。

不要一开始就重写 kernel。先确认运行环境有效，再检查生成的 CUDA。排除环境和编译期问题后，这类 kernel 的运行时失败通常都会归结为某个 broken handoff：未初始化的 barrier、错误 arrival count、藏在 role guard 里的 collective、过期 barrier phase，或者 producer 还没让写入可见就复用了 storage。

## 调试 Kernel 之前

先排除运行时上下文问题：

```bash
python -c "import tvm, tvm.tirx; print(tvm.__file__, tvm.__version__)"
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

这些 kernel 目标是 Blackwell（`sm_100a`）。如果 Python import 了过期 TVM checkout，或者 GPU 不是 Blackwell 级别，先修复这些问题再改 kernel。然后在看性能之前，先运行 kernel 最小的 correctness check，例如 `run_correctness()`。

## 调试流程

1. 用仍会失败的最小 shape 复现问题。如果 failure 是 illegal memory access，下一次运行前重启 Python。
2. 如果 compilation fails，先检查安装的 API、target、`dispatch=` 和 buffer scope，再阅读 runtime synchronization 代码。
3. 保存 `inspect_source("cuda")` 输出。在重新读 Python 之前，先搜索 role guard、`mbarrier_init`、`tcgen05`、`cp.async.bulk.tensor` 和 `cta_sync()`。
4. 为失败的 kernel path 写出 roles / storage / handoff / lifetime 表。
5. 用这张表检查生成的 CUDA：barrier init 是否位于 role branch 前；TMA producer、MMA issuer、writeback group 是否符合预期；warpgroup-only branch 内是否没有 CTA-wide collective。
6. 把运行归类为 deadlock、crash、wrong result，或 correct-but-slow，然后使用下面对应小节。
7. 一次只改一个 handoff：init count、arrive/wait phase、role guard、fence、TMA store drain、TMEM alloc/dealloc 或 tile-scheduler advance。
8. 重新测性能前先重新跑 correctness。

## 需要记录什么

对任何异步 kernel，改代码前先做一个小 worksheet：

| 项目 | 需要写下什么 |
|---|---|
| Roles | 发起每个 async operation 的精确线程、warp、warpgroup 或 CTA。 |
| Storage | 每个 tile 在每一步的 live 位置：GMEM、SMEM、TMEM 或 registers。 |
| Handoff | Producer、consumer、signal object、arrival count、phase，以及让数据可见的 fence 或 drain。 |
| Lifetime | 每个 storage slot 最早何时可以被复用、读回或释放。 |

然后用 worksheet 检查生成的 CUDA：

- Role guard 与 roles 表匹配。
- Barrier init 出现在 guarded role branch 之前。
- Collective operation 没有被 lane、warp 或 warpgroup guard 意外缩窄。
- Arrive/wait phase 与 handoff 表匹配。
- TMA store drain、TMEM dealloc 和 SMEM reuse 只在 lifetime 表允许之后发生。

同一张 worksheet 可以用于 TMA->MMA->writeback GEMM pipeline，也可以用于 Flash Attention 中的 score/softmax/value/correction handoff。

## 如果编译失败

先修复 compile-time failure，再调试 runtime synchronization：

| 症状 | 可能区域 | 首先检查 |
|---|---|---|
| Unknown TIRx API 或 attribute error | 安装的 wheel 与教程代码不匹配 | 打印 `tvm.__file__` 和 `tvm.__version__`；用 {ref}`chap_language_reference` 对比 API 名称。 |
| Unsupported `dispatch=` | 选中的 target 或 primitive 不支持该路径 | 检查 `dispatch` 参数和 target capability；本教程中的 `tcgen05` 路径需要 Blackwell。 |
| Buffer scope mismatch | Buffer 正在通过错误硬件路径使用 | 检查 worksheet 中的 storage 行：TMEM 必须通过 `tcgen05` 访问，TMA operand 必须使用兼容的 GMEM/SMEM layout。 |
| 编译成功但生成 CUDA 缺少预期路径 | Dispatch 没有按预期 lower | 改算法前先检查生成 CUDA 中是否有 `tcgen05` 和 `cp.async.bulk.tensor`。 |

## 检查生成代码

对任何已编译 kernel，保存 CUDA，方便搜索和 diff：

```python
from pathlib import Path

cuda_source = ex.mod.imports[0].inspect_source("cuda")
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/my_kernel.cu").write_text(cuda_source, encoding="utf-8")
print(cuda_source)
```

生成代码中 TIRx construct 到 CUDA 的映射如下：

| TIRx | Generated CUDA |
|------|---------------|
| `wg_id == 0` | `(warp_id_in_cta >> 2) == 0` |
| `wg_id == 1` | `(warp_id_in_cta >> 2) == 1` |
| `warp_id == 0` | `(warp_id_in_cta & 3) == 0` |
| `warp_id == 3` | `(warp_id_in_cta & 3) == 3` |
| `lane_id == 0` | `(((int)threadIdx.x) % 32) == 0` |
| `.init()` internal guard | `((int)threadIdx.x) < 1`（仅 CTA thread 0） |
| `elect_sync()` | `tvm_builtin_elect_one_sync_op()` |

读完整 kernel 前先扫描这些字符串：

| Generated CUDA | 检查 |
|---|---|
| `if (threadIdx.x < 1)` | 单个 CTA-thread guard，常用于 barrier initialization |
| `mbarrier_init` | Barrier initialization 存在，并出现在 role branch 之前 |
| `tcgen05` | Tensor Core 路径已生成 |
| `cp.async.bulk.tensor` | Copy lowered 到 TMA |
| `cta_sync();` | CTA-wide barrier；它不能位于 `wg_id` branch 内 |

## Step 7 参考骨架

正确编译的 Step 7 kernel 有如下 top-level 形状。下面的 guard 用 role name 写出以便阅读；在生成 CUDA 中，请搜索上表中的对应表达式。

```c
// (1) Barrier inits: top level, CTA thread 0 only
if (threadIdx.x < 1) {
  mbarrier_init(tma2mma[0..1], 1);
  mbarrier_init(mma2tma[0..1], 1);
  mbarrier_init(mma2ld, 1);
  mbarrier_init(ld2mma, 128);   // arrived by all 128 WG0 threads
}

// (2) TMEM alloc: WG0 warp 0, all lanes of the issuing warp
if (wg_id == 0 && warp_id == 0) tcgen05_alloc(..., 512);

// (3) Fences + cta_sync, then phase init: producer=1, consumer=0

// (4) Warp-specialized loop
if (wg_id == 1 && warp_id == 3 && elect_sync) { /* TMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 1 && warp_id == 0 && elect_sync) { /* MMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 0)                                { /* WB   */ while(valid){ ... next_tile(); } }

// (5) Cleanup: issuing warp, no lane guard
cta_sync();
if (warp_id == 0) { tcgen05_relinquish_alloc_permit(); tcgen05_dealloc(..., 512); }
```

改算法前先检查这些点：

- Barrier init 位于 top level，而不是 `wg_id` guard 内。
- `tcgen05_alloc` 和 `tcgen05_dealloc` 有 warp guard，但没有 lane guard；issuing warp 的所有 lane 都参与。
- TMA 和 MMA loop 都迭代 `K_TILES` 次。
- Phase init 是 producer=`1`，consumer=`0`。

## 症状映射

从症状开始，但把它当作线索，而不是最终诊断：

| 线索 | 可能区域 | 首先检查 |
|---|---|---|
| Kernel hang，随后 runtime 报 unspecified launch failure | Deadlock | Barrier init 位置、arrival count、`cta_sync()` 位置和 `next_tile()` 参与情况 |
| Illegal memory access、XID，或后续无关 CUDA 调用也失败 | Crash / poisoned context | 重启 Python，然后检查 pointer range、storage lifetime 和 collective participation |
| 错误 row 以 128-row 或 tile-sized stripe 出现 | Sync race 或 tile-index mismatch | Producer/consumer phase、scheduler advance，以及哪个 warpgroup 拥有每个 row stripe |
| `NaN` 或明显 invalid values | Descriptor、operand setup 或未初始化 accumulation | SMEM/TMEM descriptor setup、swizzle/layout 和 accumulator initialization |
| 有限但带 pattern 的错误值 | Stale 或部分可见数据 | 缺少 fence、缺少 TMA store drain，或 storage 在 lifetime 表允许前被复用 |
| 输出正确但没有预期 speedup | Dispatch 或 resource 问题 | 生成 CUDA 路径、pipeline depth、occupancy 和 register spill |

## 何时重启 Python

CUDA error 不一定会自动清理状态。发生 illegal memory access、XID 或 “CUDA context poisoned” 错误后，后续无关调用如 `torch.randn` 可能继续失败。测试下一个修复前重启 Python process，否则你可能在调试上一次 crash，而不是当前代码。

## Deadlock

按顺序检查这些点：

- **Arrival count 与 init count 不匹配。** 常见情况：`MBarrier.init(128)`，但 `arrive` 被 `if warp_id == 0: if lane_id == 0:` guard 住，于是只有 1 个线程 arrive，wait 永远不返回。

  | Barrier | init(count) | Who arrives | Arrivals |
  |---|---|---|---|
  | `TMABar` (tma->mma) | 1 | TMA engine 通过 `arrive(stage, bytes)` | 1 |
  | `TCGen05Bar` (mma->tma, mma->ld) | 1 | MMA warp 通过 `tcgen05.commit` | 1 |
  | `MBarrier` (ld->mma) | 128 | 所有 WG0 线程通过 `arrive` | 128 |

- **Barrier init 嵌在 `wg_id` guard 内。** `.init()` 会 lower 成 `if threadIdx.x < 1:`，也就是 CTA thread 0。CTA thread 0 位于 WG0，所以 `if wg_id == 1:` 会阻止所有线程执行 init。Init 必须位于 top level；用 `inspect_source()` 中的 `grep mbarrier_init` 验证。

- **`cta_sync()` 位于 warpgroup branch 内。** `cta_sync` 是 `__syncthreads()`，要求所有 CTA 线程到达。放在 `if wg_id == 0:` 内时，WG1 永远不会到达。单个 warpgroup barrier 请使用 `T.cuda.warpgroup_sync(10)`。

- **`tile_scheduler.next_tile()` 被某些 consumer-warpgroup thread 跳过。** Scheduler 跟踪 per-thread state；跳过它的线程可能永远循环。

- **TMA 和 MMA 的 K-tile count 不一致。** 如果 MMA 做 `K_TILES - 1` 而不是 `K_TILES`，barrier phase 会 drift，第二个 outer tile 可能 deadlock。

- **`PipelineState` 初始 phase 错误。** Producer 从 `phase=1` 开始，使第一次 wait 通过；consumer 从 `phase=0` 开始，使第一次 wait 阻塞。如果两者从同一个 phase 开始，第一次 handoff 就可能立即 deadlock。

## Crash 和 Context Poisoning

常见原因：

- **`pool.commit()` 之后又 `pool.alloc`。** Barrier wrapper 内部会调用 `alloc`。正确顺序是：`tmem_addr -> barrier wrappers -> move_base_to(1024) -> Asmem / Bsmem / Dsmem -> commit()`。
- **`tcgen05.alloc` 或 `tcgen05.dealloc` 带 lane guard。** Issuing warp 必须所有 lane 都参与。`if lane_id == 0:` 只运行一个线程，属于 undefined behavior。
- **`tcgen05.dealloc` 前缺少 `cta_sync()`。** TMEM 在 writeback 仍在读取时被释放。
- **GMEM 或 SMEM 越界访问。** 缩小到一个 tile，检查 scheduler 的 `m_idx` / `n_idx`，并检查当前 shape 是否是 kernel 的 tile 或 cluster tile 的倍数。

## 错误结果

猜测前先按 pattern 分类错误输出。整条 row stripe 通常指向 producer/consumer phase、tile-index 或 role-ownership mismatch。`NaN` 输出通常指向 descriptor setup、operand setup 或未初始化 accumulation。有限但带 pattern 的错误值通常意味着 consumer 读取了旧 tile、部分写入的 tile，或者 store 还没 drain 的数据。

- **`tcgen05.commit` 在 `elect_sync` 外。** 所有 32 个线程都会创建 commit group；其中 31 个空 group 会立即 signal mbarrier。TMA 可能在 MMA 读取 SMEM 前覆盖它。
- **TMA store 前缺少 `fence.proxy_async("shared::cta")`。** TMA engine 可能看不到线程对 SMEM 的写入。
- **TMA store 后缺少 `cp_async.bulk.commit_group()` 加 `wait_group(0)`。** 下一 tile 可能在 store drain 之前复用 Dsmem。
- **Persistent kernel 在 1024x1024 等小尺寸上间歇失败。** 大尺寸可能用更长 K-loop 掩盖 race。重新检查 tile 之间的 phase reset 和 TMA-store commit/wait。
- **`fence.after_thread_sync()` 通常不是修复。** MMA-completion mbarrier 已经携带 release-acquire 语义。Steps 8 和 9 只在 writeback edge 上保守添加它，也就是 `mma2ld.wait` 之后、第一个 `tcgen05.ld` 之前；不要在 TMA-to-MMA edge 上例行添加。

## 正确但很慢

如果输出正确但性能远低于预期，使用同样的 inspection loop：

| 线索 | 可能区域 | 首先检查 |
|---|---|---|
| 生成 CUDA 没有 `cp.async.bulk.tensor` | Copy 没有 lower 到 TMA | 检查 `dispatch="tma"`、target capability 和 operand layout |
| 生成 CUDA 没有 `tcgen05` path | MMA 没有 lower 到 Blackwell Tensor Core 指令 | 检查 `dispatch="tcgen05"`、target capability 和 operand layout |
| TMA 和 MMA 没有重叠 | Pipeline 太浅或 phase 串行化了 producer/consumer | 检查生成 CUDA 中 wait/arrive/advance 的顺序 |
| 小 shape correctness 好，但大 shape 很慢 | Register spill、occupancy 或 staging-buffer pressure | 检查编译器 resource report；减小 tile size、chunk writeback 或降低 pipeline depth |

## 提交高质量 Issue

如果 failure 在上述检查后仍然存在，请先 reduce，再到 [Apache TVM GitHub 仓库](https://github.com/apache/tvm/issues)提交 issue。请包含：

- `tvm.__file__` / `tvm.__version__` 输出和 GPU capability；
- 能复现 failure 的最小 shape；
- failure 是 compile-time、deadlock、crash、wrong result，还是 correct-but-slow；
- 最小 kernel 或 notebook cell，以及对应 correctness check；
- 保存的 `inspect_source("cuda")` 输出，或能显示可疑 guard、barrier、dispatch path 的最小摘录。
