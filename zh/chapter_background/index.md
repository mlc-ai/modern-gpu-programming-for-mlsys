(chap_background)=
# GPU 执行模型

:::{admonition} 概览
:class: overview

- GPU kernel 的执行首先由线程层级决定：thread、warp、warpgroup、CTA、cluster 和 grid 分别对应不同的协作粒度。Blackwell 上的很多操作都有自己的天然 scope，例如 TMA copy 由单个 thread 发起，完整的 TMEM accumulator 由四个 warp 分别读取各自的 32-lane window，而 2-CTA cooperative MMA 会跨越两个 CTA。
- 数据不会只停留在一个地方。GMEM、SMEM、TMEM 和寄存器分别服务于不同的容量、延迟和访问范围；cluster 还通过 DSMEM 让一个 CTA 可以访问另一个 CTA 的 shared memory。高性能 kernel 的核心任务之一，就是安排数据在这些空间之间高效移动。
- 计算和数据搬运由不同硬件引擎共同完成。CUDA Core 处理地址计算、控制流和标量逻辑，Tensor Core 执行主要矩阵计算，TMA 负责异步搬运数据。最后我们会用一条 GEMM 数据流水线把这些部件串起来，说明高性能 kernel 如何通过 overlap 让多个引擎同时保持忙碌。


:::

想要写出高性能 GPU kernel，首先要理解一次 kernel 执行时，线程如何被组织，数据放在哪里，以及不同硬件引擎如何协同工作。本章会围绕这三件事展开：先介绍 GPU 的线程层级，再介绍保存和搬运数据的内存空间，最后介绍承担计算和数据搬运的硬件引擎。随后，我们会用一条 GEMM 流水线把这些概念串起来，说明数据如何在内存空间之间移动，计算又如何与数据搬运重叠。后续章节中的许多优化，都会建立在这几个基本机制之上。

我们先从 Blackwell SM 的架构开始。下图展示了本章会用到的几个主要硬件单元。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo_zh/sm_architecture.html" title="Blackwell SM architecture" loading="lazy"
        style="width:100%; height:620px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*点击图中组件，可以查看 Blackwell SM 中的 warp、warpgroup、shared memory、Tensor Memory、Tensor Core 和 TMA 引擎。*

## 执行层级

GPU 不会把成千上万个线程当作一个扁平集合来管理，而是把它们组织成多个层级。每一层对应一种协作粒度。下图展示了 Blackwell 上的线程层次结构。

```{raw} html
<iframe src="../demo_zh/thread_hierarchy.html" title="Blackwell thread hierarchy" loading="lazy"
        style="width:100%; height:520px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*点击图中组件，可以逐层查看 thread、warp、warpgroup、CTA、cluster 和 grid。*

* **Thread**：最小的执行单位。每个 thread 都有自己的程序计数器和寄存器，并通过它所在 warp 内的 lane ID 来标识。
* **Warp**：以 SIMT（*single instruction, multiple threads*）方式执行的一组 32 个 thread。一个 warp 中的 lane 会一起发出同一条指令，但每个 lane 保留自己的寄存器，也可以被单独 mask 掉。遇到分支时，warp 会通过不同的 mask 分别执行各条路径。
* **Warpgroup**：四个连续的 warp，也就是 128 个 thread。Hopper 架构引入了 warpgroup，作为发起 warpgroup-level MMA（`wgmma`）的单位；在 Blackwell 上，它的四个 warp 还可以分别覆盖 Tensor Memory 的四个 32-lane window。
* **CTA**（*Cooperative Thread Array*，也就是 CUDA 中的 thread block）：硬件调度的基本单位。一个 CTA 运行在单个 SM 上，并拥有该 SM 中一块属于自己的 shared memory 分配。多个 CTA 可以同时驻留在同一个 SM 上，此时它们共享这个 SM 的 shared memory 总容量。
* **Cluster**：一组可以相互协作的 CTA。一个 cluster 中的 CTA 可以位于不同 SM 上；它们可以彼此同步，也可以通过 distributed shared memory（DSMEM）访问对方的 shared memory。

Blackwell 上的关键操作并不都由同一组线程发起。TMA copy 由单个 thread 发起，再交给硬件执行；每个 warp 通过 warp-level TMEM load 读取自己的 32-lane window；`tcgen05` MMA 由一个选定的 thread 提交；2-CTA cooperative MMA 则跨越两个 CTA。

后文把一项操作涉及的线程范围称为它的 **scope**。分析 kernel 时，需要同时考虑操作的 scope、数据 layout 和 dispatch 方式。


## 内存空间

线程层级说明了计算如何组织，接下来还需要确定数据存放在哪里。GPU 提供多种内存空间，它们在容量、延迟和访问范围之间做出不同取舍。kernel 必须在这些空间之间高效地移动数据。

| 内存 | 作用范围 | 用途 | 说明 |
|------|----------|------|------|
| **Global（GMEM）** | 整个 device | 存放输入/输出 tensor | 大容量 HBM，由所有 SM 共享 |
| **Shared（SMEM）** | 每个 CTA（一个 SM 内） | Tile 暂存 | 低延迟的临时存储区；B200 上最高可达 228 KB/SM |
| **Tensor Memory（TMEM）** | 每个 CTA | MMA accumulator 存储 | Blackwell 新增；供 `tcgen05` 使用 |
| **Register File（RF）** | 每个 thread | 标量和每个 thread 的 tile fragment | 很快；保存 epilogue/临时值 |

**Tensor Memory（TMEM）** 是 Blackwell 新增的片上存储空间。在此前的架构中，MMA accumulator 通常保存在寄存器中；随着 MMA tile 增大，这些 accumulator 会占用大量寄存器。Blackwell 的 `tcgen05` 将 accumulator 写入 TMEM，从而降低这部分寄存器压力。

可以把 TMEM 看作一个由 CTA 使用的二维临时存储区。这个空间包含 128 行（对应 128 条 TMEM lanes）和最多 512 列，每列宽 32 bit。它在逻辑上归 CTA 使用，物理上仍位于 SM 内。

TMEM 需要由程序显式管理。Kernel 必须分配和释放 TMEM；MMA 完成后，epilogue 还需要显式地将 accumulator 从 TMEM 读回寄存器。读取完整的 128-lane accumulator 时，warpgroup 中的四个 warp 会分别加载自己的 32-lane TMEM window。

### 跨 Cluster 的分布式共享内存

一个 cluster 可以包含位于不同 SM 上的多个 CTA。每个 CTA 仍然拥有自己的 shared memory，但 distributed shared memory（DSMEM）允许同一 cluster 内的其他 CTA 访问其中的数据。

这种能力可以避免不必要的 GMEM 往返。一个 CTA 可以直接访问另一个 CTA 的 SMEM，而不需要让对方先写回 GMEM、再重新读取。使用异步操作搬运这些数据时，completion barrier 会在搬运完成后通知后续计算继续执行。

下图展示了一个 2-CTA cluster 中的 DSMEM 访问路径。每个 CTA 仍然拥有自己的 SMEM，但可以读取另一个 CTA 的 SMEM。


```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo_zh/cta_cluster.html" title="A 2-CTA cluster sharing distributed shared memory" loading="lazy"
        style="width:100%; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*点击图中组件，可以查看两个 CTA 如何通过 DSMEM 访问彼此的 shared memory。*

在图中的 2-CTA GEMM 中，每个 CTA 都保存自己的 A 和 B 分片，同时通过 DSMEM 读取另一个 CTA 的 B 分片。这里的“共享”并不表示两个 CTA 的 SMEM 被合并成一块；它只表示 cluster 内的 CTA 可以跨 SM 访问对方的数据。

在此基础上，两个 CTA 还可以组成 `cta_group=2`，共同执行 cooperative MMA，生成一个更大的输出 tile。

## 计算核心：CUDA Core 和 Tensor Core

线程层级决定计算如何组织，内存空间决定数据放在哪里；真正执行算术运算的，是 SM 内的计算单元。一个 SM 中主要有两类计算单元：CUDA Core 和 Tensor Core。

* **CUDA Core** 是通用的 SIMT ALU，负责执行标量和向量指令。kernel 中的地址计算、elementwise 运算、reduction 和控制流，大多由 CUDA Core 完成。
* **Tensor Core** 是面向矩阵计算的固定功能单元，以 *tile* 为粒度执行 dense matrix multiply-accumulate，在一条指令中计算 $D = AB + C$。

Tensor Core 的算术吞吐量远高于 CUDA Core，通常可以达到后者 10 倍或更多的 FLOP/s。GEMM、convolution 和 attention 这类 dense linear algebra 计算，只有充分利用 Tensor Core，才可能接近峰值性能。高性能 kernel 还要及时准备好数据，避免 Tensor Core 因等待数据或依赖而空转。

不同 GPU 架构不仅改变 Tensor Core 的吞吐量，也改变它们的编程方式和 accumulator 的存放位置。Hopper 引入了异步 warpgroup MMA（`wgmma.mma_async`）；Blackwell 的第五代 Tensor Core，也就是 `tcgen05`，则把 accumulator 放入 Tensor Memory，而不是寄存器中。后续章节会专门讨论这一点。

Cluster 在 GEMM 中还会带来两个重要用法。**2-CTA cooperative MMA** 允许两个 CTA 各自提供一部分 SMEM operand，共同发起一个更大的 Tensor Core MMA tile。**TMA multicast** 允许一次 GMEM load 把同一个 tile 送到多个 CTA，避免每个 CTA 分别读取同一份数据造成冗余 global memory traffic。二者都依赖前面介绍的 cluster 和 DSMEM 机制。


## GEMM 数据流水线

前面几节分别介绍了线程层级、内存空间、数据搬运机制和计算单元。现在我们用一条 GEMM 流水线把它们串起来，看这些硬件结构如何协同工作。下图展示了一条三阶段 GEMM tile 流水线中涉及的主要单元。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo_zh/pipeline_arch.html" title="Blackwell GEMM data pipeline" loading="lazy"
        style="width:100%; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```

*点击图中的阶段，可以查看 Blackwell 上从 load、MMA 到 epilogue 的数据路径。*

单个 GEMM tile 通常会经过三个阶段。

1. **Load：** TMA copy 把 A 或 B 的 operand tile 从 GMEM 搬到 SMEM。一个 thread 发起这次 copy，并记录预计到达的字节数。随着数据写入 SMEM，TMA 引擎会更新进度；所有预期字节到达后，completion barrier 才会被触发。
2. **Compute：** `tcgen05` MMA 从 SMEM 读取 operand tile，并把乘积累加到 TMEM tile 中。一个选定的 thread 提交这次 MMA；计算完成后，硬件会向对应的 barrier 发出完成信号。
3. **Epilogue：** warpgroup 把 TMEM accumulator 读回寄存器，将结果转换成输出 dtype，再写回 GMEM。这一步通常会先经过 SMEM staging，也可能使用 TMA store 完成最终写回。

这三个阶段存在数据依赖，但不必完全串行执行。朴素的 kernel 会依次执行 load、wait、compute、wait 和 store，导致各个硬件单元轮流等待。

高性能 kernel 会把它们组织成流水线：Tensor Core 计算第 `k` 个 tile 时，TMA 引擎可以搬运第 `k+1` 个 tile，epilogue 则处理第 `k-1` 个 tile。Barrier 和 phase 模型负责这些异步阶段之间的安全交接，后面的 GEMM 优化也建立在这一机制之上。
