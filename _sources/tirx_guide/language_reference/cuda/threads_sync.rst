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
=======================

``T.cuda`` provides CUDA operations and helpers. ``T.ptx`` names individual instructions with exact modifiers. ``T.cuda.tile.apply`` repeats an explicitly selected instruction over caller-prepared views; ``T.cuda.tile.ewise`` applies a scalar ``@T.function``.

Synchronization semantics
-------------------------

``T.cuda.cta_sync()`` synchronizes the CTA. ``T.cuda.warpgroup_sync(id)`` synchronizes a complete 128-thread warpgroup with a caller-owned named barrier. Reach collectives uniformly. ``T.cuda.elect_sync()`` selects one active lane per warp. ``T.cuda.mbarrier_wait(bar, phase)`` waits until the barrier leaves the specified phase; update parity before reuse. Initialize barriers from one thread, then fence/synchronize initialization.

Warp reduction
--------------

This complete all-reduce writes 496 to every output element.

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
-----------

The final positional argument is the CUDA source; ``ty`` specifies the return type.

.. code-block:: python

    SRC = r"""
    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    """

    @T.function
    def k(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=256))
        tx = T.cuda.thread_idx("x")
        B[tx] = T.cuda.func_call("my_relu", A[tx], SRC, ty="float32")
