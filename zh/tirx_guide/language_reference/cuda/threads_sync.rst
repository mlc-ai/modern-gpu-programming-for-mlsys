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

CUDA C++/PTX intrinsics
==============================================

``T.cuda`` 提供 CUDA operations/helpers，``T.ptx`` 通过完整修饰符指定单条指令。Tile API 使用 ``T.cuda.tile.apply`` 配合明确的指令和 operand views；``T.cuda.tile.ewise`` 将标量 ``@T.function`` 应用于 tile。Collective operation 的参与线程必须一致到达，mbarrier 复用时要翻转 phase。下面的完整示例分别演示 warp reduction 和内联 CUDA。

同步语义
--------

``T.cuda.cta_sync()`` 同步 CTA。``T.cuda.warpgroup_sync(id)`` 用调用者管理的 named barrier 同步完整的 128-thread warpgroup。所有参与者必须一致到达 collective。``T.cuda.elect_sync()`` 在每个 warp 内选择一个 active lane。``T.cuda.mbarrier_wait(bar, phase)`` 等待 barrier 离开指定 phase；复用前需更新 parity。由一个 thread 初始化 barrier，再用 fence 和同步发布初始化。

Warp reduction
----------------------------

下面的完整 all-reduce 向每个输出元素写入 496。

.. code-block:: python

    @T.function
    def warp_reduce(A: T.Tensor((32,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=32))
        lane = T.cuda.lane_id()
        value: T.float32 = T.float32(31 - lane)
        offset: T.int32 = 16
        while offset >= 1:
            value += T.gpu_warp_shuffle_xor(0xFFFFFFFF, value, offset, 32, 32)
            offset //= 2
        A[lane] = value

Inline CUDA
----------------------

最后一个位置参数是 CUDA source；``ty`` 指定返回类型。

.. code-block:: python

    SRC = r"""
    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    """

    @T.function
    def k(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=256))
        tx = T.cuda.thread_idx("x")
        B[tx] = T.cuda.func_call("my_relu", A[tx], SRC, ty="float32")
