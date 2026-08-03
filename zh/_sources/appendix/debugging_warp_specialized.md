(chap_warp_spec_debug)=
# 调试 Warp-Specialized Kernel

第三部分 GEMM 的第 7-9 步让 TMA load、`tcgen05` MMA 和 TMEM/SMEM writeback
重叠执行。Flash Attention 也使用类似的数据交接方式。调试这类 kernel 时，
可以先确定各个角色及其使用的存储空间，再检查生成的 CUDA 是否符合这个模型。

不要一开始就重写 kernel。先确认运行环境正确，并用最小测试稳定复现问题，
然后再检查生成的 CUDA。排除环境和编译问题后，这类 kernel 的运行时故障通常
来自某次数据交接：barrier 没有初始化、arrival count 错误、collective 的参与
范围被角色分支缩小、wait 使用了旧的 barrier phase，或者 producer 的写入尚未
可见，存储空间就被提前复用。

## 调试前先检查环境

先排除运行环境的问题：

```bash
python -c "import tvm, tvm.tirx; print(tvm.__file__, tvm.__version__)"
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

这些 kernel 面向 Blackwell（`sm_100a`）。如果 Python 导入了旧的 TVM
checkout，或者当前 GPU 不是 Blackwell 架构，应先修正环境，再修改 kernel。
随后先运行最小的正确性测试，例如 `run_correctness()`；正确性通过后再看性能。

## 调试步骤

1. 用仍能复现问题的最小 shape 运行。如果发生 illegal memory access，下一次运行前先重启 Python。
2. 如果编译失败，先检查已安装的 API、target、`dispatch=` 和 buffer scope，再检查运行时同步代码。
3. 保存 `inspect_source("cuda")` 的输出。先搜索 role guard、`mbarrier_init`、`tcgen05`、`cp.async.bulk.tensor` 和 `cta_sync()`，再回头阅读 Python。
4. 针对出错的 kernel 路径，写出 roles、storage、handoff 和 lifetime 表。
5. 根据这张表检查生成的 CUDA：barrier 初始化是否位于角色分支之前，TMA producer、MMA issuer 和 writeback group 是否符合预期，以及要求整个 CTA 参与的 collective 是否误放进了只由一个 warpgroup 执行的分支。
6. 将问题归类为 deadlock、crash、wrong result 或 correct-but-slow，再查看下方对应的小节。
7. 每次只修改一处交接：init count、arrive/wait phase、role guard、fence、TMA store 的完成等待、TMEM alloc/dealloc 或 tile scheduler 的推进。
8. 每次修改后先重新验证正确性，再测量性能。

## 先写清楚数据如何交接

调试异步 kernel 前，可以先填写一张简单的表：

| 项目 | 需要记录的内容 |
|---|---|
| Roles | 哪些 threads、warps、warpgroups 或 CTAs 发出每项异步操作。 |
| Storage | 每一步中 tile 所在的位置：GMEM、SMEM、TMEM 或 registers。 |
| Handoff | Producer、consumer、同步对象、arrival count、phase，以及保证数据可见或传输完成的 fence 和 wait。 |
| Lifetime | 每个存储位置最早何时可以复用、读回或释放。 |

再根据这张表检查生成的 CUDA：

- Role guard 与 roles 表一致。
- Barrier 初始化出现在各个角色分支之前。
- Collective 没有被 lane、warp 或 warpgroup guard 意外缩小参与范围。
- Arrive/wait phase 与 handoff 表一致。
- 完成相应的 wait，并且 lifetime 表允许之后，才释放 TMEM 或复用 SMEM。

这张表既适用于 GEMM 的 TMA -> MMA -> writeback pipeline，也适用于
Flash Attention 中 score、softmax、value 和 correction 之间的交接。

## 编译失败

先解决编译问题，再调试运行时同步：

| 现象 | 可能的位置 | 首先检查 |
|---|---|---|
| TIRx API 未知或发生 attribute error | 安装的 wheel 与教程代码不匹配 | 输出 `tvm.__file__` 和 `tvm.__version__`，并对照 {ref}`chap_language_reference` 检查 API 名称。 |
| 不支持指定的 `dispatch=` | 当前 target 或 primitive 不支持这条路径 | 检查 `dispatch` 参数和 target capability；本教程中的 `tcgen05` 路径需要 Blackwell。 |
| Buffer scope 不匹配 | Buffer 通过错误的硬件路径使用 | 检查表中的 storage：TMEM 必须通过 `tcgen05` 访问，TMA 搬运的 buffer 必须使用兼容的 GMEM/SMEM layout。 |
| 编译成功，但生成的 CUDA 中没有预期路径 | Dispatch 没有生成预期的硬件指令 | 修改算法前，先在生成的 CUDA 中搜索 `tcgen05` 和 `cp.async.bulk.tensor`。 |

## 检查生成的代码

对于任意已编译的 kernel，都可以保存 CUDA 源码，便于搜索和比较：

```python
from pathlib import Path

cuda_source = ex.mod.imports[0].inspect_source("cuda")
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/my_kernel.cu").write_text(cuda_source, encoding="utf-8")
print(cuda_source)
```

常见的 TIRx 结构与生成 CUDA 的对应关系如下：

| TIRx | 生成的 CUDA |
|------|---------------|
| `wg_id == 0` | `(warp_id_in_cta >> 2) == 0` |
| `wg_id == 1` | `(warp_id_in_cta >> 2) == 1` |
| `warp_id == 0` | `(warp_id_in_cta & 3) == 0` |
| `warp_id == 3` | `(warp_id_in_cta & 3) == 3` |
| `lane_id == 0` | `(((int)threadIdx.x) % 32) == 0` |
| `.init()` 内部的 guard | `((int)threadIdx.x) < 1`（只允许 CTA thread 0） |
| `elect_sync()` | `tvm_builtin_elect_one_sync_op()` |

阅读完整 kernel 前，先搜索下面这些字符串：

| 生成的 CUDA | 检查内容 |
|---|---|
| `if (threadIdx.x < 1)` | 单个 CTA thread 的 guard，通常用于初始化 barrier |
| `mbarrier_init` | 是否生成了 barrier 初始化，并且位于角色分支之前 |
| `tcgen05` | 是否生成了 Tensor Core 路径 |
| `cp.async.bulk.tensor` | Copy 是否生成了 TMA 路径 |
| `cta_sync();` | CTA-wide barrier；不能位于 `wg_id` 分支内部 |

## 第 7 步的参考结构

正确编译的第 7 步 kernel 顶层结构如下。为了便于阅读，这里用角色名称写
guard；在生成的 CUDA 中，应搜索上表对应的表达式。

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

修改算法前，先检查：

- Barrier 初始化位于顶层，而不是 `wg_id` guard 内。
- `tcgen05_alloc` 和 `tcgen05_dealloc` 有 warp guard，但没有 lane guard；发出指令的 warp 中所有 lanes 都参与。
- TMA 和 MMA loop 都迭代 `K_TILES` 次。
- Producer 的初始 phase 为 `1`，consumer 的初始 phase 为 `0`。

## 根据现象定位问题

现象只能作为线索，不应直接当作最终诊断：

| 线索 | 可能的位置 | 首先检查 |
|---|---|---|
| Kernel 卡住，随后 runtime 报告 unspecified launch failure | Deadlock | Barrier 初始化的位置、arrival count、`cta_sync()` 的位置和 `next_tile()` 的参与范围 |
| Illegal memory access、XID，或之后无关的 CUDA 调用也失败 | Crash / poisoned context | 重启 Python，再检查 pointer 范围、storage lifetime 和 collective 的参与范围 |
| 错误结果以连续 128 行或一个 tile 为单位呈条纹状出现 | 同步竞争或 tile index 不匹配 | Producer/consumer phase、scheduler 推进方式，以及每段 rows 属于哪个 warpgroup |
| 出现 `NaN` 或明显无效的数值 | Descriptor、operand 设置或 accumulator 未初始化 | SMEM/TMEM descriptor、swizzle/layout 和 accumulator 初始化 |
| 数值有限，但错误呈固定模式 | 读取了旧数据或只完成一部分的数据 | 是否缺少 fence 或没有等待 TMA store 完成，storage 是否在 lifetime 允许前被复用 |
| 结果正确，但没有预期加速 | Dispatch 或资源问题 | 生成的 CUDA 路径、pipeline depth、occupancy 和 register spill |

## 何时需要重启 Python

CUDA 错误不一定会自动恢复。发生 illegal memory access、XID 或
“CUDA context poisoned” 后，后续无关调用（例如 `torch.randn`）也可能持续
失败。测试下一项修改前先重启 Python 进程，否则你看到的可能仍是上一次
crash 留下的状态。

## Deadlock

按下面的顺序检查：

- **Arrival count 与 init count 不匹配。** 常见情况是 `MBarrier.init(128)`，但 `arrive` 位于 `if warp_id == 0: if lane_id == 0:` 中，最终只有一个 thread 执行 arrival，wait 永远不会返回。

  | Barrier | init(count) | 完成状态如何报告 | Arrivals |
  |---|---|---|---|
  | `TMABar`（tma->mma） | 1 | 选出的 producer thread 执行 `arrive(stage, bytes)`；TMA engine 完成传输后再扣减 tx-count | 1 |
  | `TCGen05Bar`（mma->tma、mma->ld） | 1 | 选出的 MMA thread 执行 `tcgen05.commit`；MMA 完成后由硬件报告 arrival | 1 |
  | `MBarrier`（ld->mma） | 128 | WG0 的所有 threads 通过 `arrive` | 128 |

- **Barrier 初始化位于 `wg_id` guard 内。** `.init()` 会 lower 成 `if threadIdx.x < 1:`，也就是只由 CTA thread 0 执行。CTA thread 0 位于 WG0，因此把 `.init()` 放在 `if wg_id == 1:` 内时，没有 thread 会真正执行初始化。Barrier 初始化必须位于顶层；可以在 `inspect_source()` 中搜索 `mbarrier_init` 验证。

- **`cta_sync()` 位于 warpgroup 分支中。** `cta_sync` 对应 `__syncthreads()`，要求 CTA 的所有 threads 参与。放在 `if wg_id == 0:` 中时，WG1 永远无法到达。只同步一个 warpgroup 时，应使用 `T.cuda.warpgroup_sync(10)`。

- **Consumer warpgroup 中的部分 threads 跳过了 `tile_scheduler.next_tile()`。** Scheduler 保存 per-thread 状态；跳过调用的 threads 可能永远留在 loop 中。

- **TMA 与 MMA 的 K-tile 数量不一致。** 如果 MMA 执行 `K_TILES - 1` 次而不是 `K_TILES` 次，barrier phases 会逐渐错位，并在第二个 outer tile 上 deadlock。

- **`PipelineState` 的初始 phase 错误。** Producer 从 `phase=1` 开始，使第一次 wait 直接通过；consumer 从 `phase=0` 开始，使第一次 wait 阻塞。二者若从相同 phase 开始，第一次交接就可能立即 deadlock。

## Crash 与 Context Poisoning

常见原因包括：

- **在 `pool.commit()` 后调用 `pool.alloc`。** Barrier wrapper 内部会调用 `alloc`。正确顺序是：`tmem_addr -> barrier wrappers -> move_base_to(1024) -> Asmem / Bsmem / Dsmem -> commit()`。
- **用 lane guard 包围 `tcgen05.alloc` 或 `tcgen05.dealloc`。** 发出指令的 warp 必须由全部 lanes 参与。`if lane_id == 0:` 只执行一个 thread，属于未定义行为。
- **`tcgen05.dealloc` 前缺少 `cta_sync()`。** Writeback 仍在读取时，TMEM 就被释放。
- **GMEM 或 SMEM 越界访问。** 将问题缩小到一个 tile，检查 scheduler 的 `m_idx` / `n_idx`，并确认当前 shape 是 kernel tile 或 cluster tile 的整数倍。

## 结果错误

先根据错误模式分类，再推测原因。错误结果集中出现在连续的整行区域时，通常表示
producer/consumer phase、tile index 或角色 ownership 不匹配。`NaN` 往往来自
descriptor、operand 设置或未初始化的 accumulation。数值有限但错误呈固定模式，
通常表示 consumer 读到了旧 tile、只写完一部分的 tile，或尚未完成的 TMA store。

- **`tcgen05.commit` 位于 `elect_sync` 外。** 32 个 threads 都会创建 commit group；其中 31 个空 group 会立即通知 mbarrier，使 TMA 在 MMA 读取前覆盖 SMEM。
- **TMA store 前缺少 `fence.proxy_async("shared::cta")`。** TMA engine 可能看不到 threads 对 SMEM 的写入。
- **TMA store 后缺少 `cp_async.bulk.commit_group()` 和 `wait_group(0)`。** Store 尚未完成，下一 tile 就复用了 Dsmem。
- **Persistent kernel 在 `1024×1024` 等较小 shape 上偶发失败。** 更大的 shape 和更长的 K-loop 可能掩盖竞争。重新检查 tiles 之间的 phase reset 和 TMA store commit/wait。
- **等待 MMA 完成后直接读取 TMEM。** `mma2ld.wait` 只能确认 MMA 已经完成；writeback thread 在随后发出 `tcgen05.ld` 前，还需要执行 `T.ptx.tcgen05.fence.after_thread_sync()`，把这次 TMEM load 排在跨 thread 的完成通知之后。第 7-9 步都将它放在 `mma2ld.wait` 之后。这个 fence 只负责 `tcgen05` 指令之间的顺序；等待 TMA load 和让普通 thread 的 SMEM 写入对 TMA engine 可见，分别使用各自的 mbarrier 和 proxy fence 协议。

## 结果正确但性能较差

如果结果正确，但性能远低于预期，可以继续使用同一套检查流程：

| 线索 | 可能的位置 | 首先检查 |
|---|---|---|
| 生成的 CUDA 中没有 `cp.async.bulk.tensor` | Copy 没有生成 TMA 路径 | 检查 `dispatch="tma"`、target capability 和 operand layout |
| 生成的 CUDA 中没有 `tcgen05` | MMA 没有生成 Blackwell Tensor Core 指令 | 检查 `dispatch="tcgen05"`、target capability 和 operand layout |
| TMA 与 MMA 没有重叠 | Pipeline 太浅，或者 phase 使 producer/consumer 串行执行 | 检查生成 CUDA 中 wait、arrive 和 advance 的顺序 |
| 小 shape 正确，但大 shape 性能差 | Register spill、occupancy 或 staging buffer 压力 | 检查 compiler resource report；减小 tile、分块 writeback，或降低 pipeline depth |

## 提交有效的问题报告

如果完成上述检查后问题仍然存在，请先缩小复现范围，再到
[Apache TVM GitHub 仓库](https://github.com/apache/tvm/issues)提交 issue。
需要包含：

- `tvm.__file__`、`tvm.__version__` 和 GPU capability；
- 能复现问题的最小 shape；
- 问题属于编译失败、deadlock、crash、wrong result 还是 correct-but-slow；
- 最小 kernel 或 notebook cell，以及对应的正确性测试；
- 保存的 `inspect_source("cuda")` 输出，或者能展示可疑 guard、barrier 或 dispatch 路径的最小片段。
