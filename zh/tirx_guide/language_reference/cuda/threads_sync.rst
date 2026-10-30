..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

CUDA C++/PTX Intrinsics
=======================

如果现有 tile primitive 无法表达需要的操作，可以用两种方式直接访问硬件：
调用 backend intrinsic，也就是 ``tvm_tirx.backend.cuda`` 提供的 ``T.cuda.*`` /
``T.ptx.*`` namespaces；或者直接内联 CUDA 源码。

调用 Backend Intrinsic
----------------------

``T.cuda.*`` 和 ``T.ptx.*`` 直接暴露 CUDA backend 的 device intrinsics，
包括同步、mbarrier、reduction，以及 PTX data movement 和 MMA 指令：

.. code-block:: python

    T.cuda.cta_sync()                    # block barrier (__syncthreads)
    T.cuda.warp_sync()                   # __syncwarp
    T.cuda.warpgroup_sync(8)             # warpgroup barrier

    bar = T.alloc_shared((1,), "uint64")
    T.ptx.mbarrier.init.shared__cta.b64(bar.data, 1)     # mbarrier for async completion
    T.cuda.mbarrier_wait(bar.data, phase)

下面是一个完整示例，通过 ``T.gpu_warp_shuffle_xor`` 完成 warp all-reduce：

.. code-block:: python

    @T.function
    def warp_reduce(A: T.Tensor((32,), "float32", align=16)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=[1], block=32))
        cta_id = T.cuda.block_idx("x"); warp_id = T.cuda.warp_id(); lane_id = T.cuda.lane_id()
        v = T.alloc_local((1,), "float32"); i = T.alloc_local((1,), "int32")
        v[0] = T.float32(31 - lane_id)
        i[0] = 16
        while i[0] >= 1:
            v[0] += T.gpu_warp_shuffle_xor(0xFFFFFFFF, v[0], i[0], 32, 32)
            i[0] = i[0] // 2
        A[lane_id] = v[0]

Shuffle 会直接转换为 ``__shfl_xor_sync``：

.. code-block:: c++

    v_ptr[0] = v_ptr[0] + __shfl_xor_sync(0xFFFFFFFF, v_ptr[0], i_ptr[0], 32);

``T.ptx.*`` / ``T.cuda.*`` 还包含 ``cp.async``（LDGSTS）、
``cp.async.bulk.tensor``（TMA）、``ldmatrix`` / ``stmatrix``、``tcgen05.*``
（Blackwell MMA）、``atomic_add`` 和 ``fence`` 等指令类别。完整列表请参阅
``tvm_tirx.backend.cuda`` backend API reference。

同步语义
--------

GEMM 和 Flash Attention kernel 中经常出现下面四种同步机制。它们控制异步
engine 和并行 thread groups；使用错误通常会让结果在没有报错的情况下损坏，
或者造成 deadlock。

**Mbarrier phase。** Mbarrier 使用一个内部 phase bit 追踪不同轮次的
arrival。``T.cuda.mbarrier_wait(bar, phase)`` 会一直等待，直到 barrier
内部 phase 与调用者提供的 ``phase`` 不同。循环复用 barrier 时，每次 wait
之后都必须翻转本地 phase tracker（``phase ^= 1``）。否则，后续 wait 可能
误把上一轮的完成状态当作当前轮，导致 consumer 在本轮 producer 或异步操作
真正完成前访问数据。第三部分的 GEMM 章节会给出完整的 phase tracking 表。

**Election。** ``T.cuda.elect_sync()`` 从一个 warp 的 active lanes 中选择
**一个 lane**；它不一定选择 lane 0，也不是每个 CTA 选择一个 thread。如果
只允许一个 thread 发出指令，还需要配合 warp-level guard。第三部分的 GEMM
kernel 使用 ``if warp_id == 0:`` 加 ``if T.cuda.elect_sync():`` 发出
``Tx.apply`` 和 ``tcgen05.commit``。

**Named warpgroup barrier。** ``T.cuda.cta_sync()`` 对应
``__syncthreads()``，要求 CTA 的所有 threads 到达。Warpgroups 进入不同
角色分支后，不能把 ``cta_sync()`` 放进其中一个 warpgroup 的分支，否则其他
warpgroups 无法到达，kernel 会 deadlock。硬件提供 16 个 named barriers
（ID 0 到 15）；``T.cuda.warpgroup_sync(10)`` 只同步一个 warpgroup 的
threads。可能同时处于 active 状态的独立同步必须使用不同 ID，例如
``warpgroup_sync(wg_id + 10)``；前一次同步完成后，ID 可以再次使用。第三部分
的 warp-specialized GEMM 会展示完整用法。

**Fence。** Fence 保证 producer 的写入先于 consumer（通常是异步 engine）
读取：

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Fence
     - 保证的顺序
   * - ``T.ptx.fence.proxy.async_.shared__cta()``
     - Thread 写入 shared memory 后，async proxy（TMA store / MMA）才能读取
   * - ``T.ptx.fence.mbarrier_init.release.cluster()``
     - 初始化 mbarrier 后，后续 arrival 或 wait 才能使用它
   * - ``T.ptx.tcgen05.fence__after_thread_sync()``
     - ``tcgen05`` writeback edge 上的保守 ordering fence；Step 7–9 使用它，TMA-to-MMA 路径不需要

内联 CUDA 源码
---------------

如果某项操作没有对应 intrinsic，可以通过
``T.cuda.func_call(name, *args, SOURCE, ty=...)`` 将源码
字符串中的 ``__device__`` function 注入生成代码：

.. code-block:: python

    SRC = r"""
    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    """

    @T.function
    def k(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=[1], block=[256])); bx = T.cuda.block_idx("x"); tx = T.cuda.thread_idx("x")
        B[tx] = T.cuda.func_call("my_relu", A[tx], SRC, ty="float32")

源码会原样输出，并接入对应的调用位置：

.. code-block:: c++

    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    // ...
    B_ptr[tx] = my_relu(A_ptr[tx]);
