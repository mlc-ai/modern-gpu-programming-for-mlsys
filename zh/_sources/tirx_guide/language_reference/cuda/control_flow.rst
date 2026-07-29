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

控制流
======

TIRx 的控制流包括 ``if``、多种 loop 和 ``while``，它们会生成对应的 CUDA
控制流。

if
--

Python 的 ``if`` / ``else`` 会变成 CUDA 的 ``if`` / ``else``。可以根据
thread 或 lane 条件限制一段工作，也可以通过 ``T.ptx.elect_sync()`` 从一个
warp 中选出一个 thread 发出指令：

.. code-block:: python

    if tx < 128:
        A[tx] = A[tx] * T.float32(2.0)
    else:
        A[tx] = A[tx] + T.float32(1.0)

    if T.ptx.elect_sync():
        ...                              # one elected lane (e.g. to issue TMA/MMA)

.. code-block:: c++

    if (((int)threadIdx.x) < 128) {
      A_ptr[tx] = A_ptr[tx] * 2.0f;
    } else {
      A_ptr[tx] = A_ptr[tx] + 1.0f;
    }

如果只需要在表达式中选择值，不需要控制流分支，可以使用
``T.if_then_else(cond, a, b)``。它会 lowering 为三元表达式，不会产生
control-flow divergence：

.. code-block:: c++

    O_ptr[tx] = (A_ptr[tx] > 0.0f) ? A_ptr[tx] : 0.0f;

Uniform 与 Divergent 控制流
---------------------------

``if tx < 128`` 这样的 per-thread guard 可以用于普通工作，但
**collective** 必须由其同步范围内的所有 threads 一致到达。

例如，``T.cuda.cta_sync()`` 对应 ``__syncthreads()``，要求 thread block 中
的所有 threads 参与。它不能放在 thread-divergent 或 warpgroup-divergent
分支中：如果放进 ``if wg_id == 0:``，其他 warpgroups 无法到达，kernel
就会 deadlock。只需要同步一个 warpgroup 时，应使用 warpgroup-scoped
``T.cuda.warpgroup_sync(id)``，详见第三部分的 warp-specialized GEMM 和
:doc:`threads_sync`。

初始化 barrier 时也要注意参与范围。``mbarrier`` 的 ``.init()`` 会 lowering
为 single-thread guard（``if (threadIdx.x < 1)``）。如果再把它放进另一个
divergent branch，barrier 可能没有被初始化，进而导致 unspecified launch
failure。

loop
----

TIRx 提供四种 loop；普通 Python ``range`` 会变成 ``T.serial``：

- ``T.serial(n)``：顺序 loop，ptxas 仍可能将它展开。
- ``T.unroll(n)``：完全展开为连续 statements。
- ``T.vectorized(n)``：vectorized loop。
- ``T.grid(*extents)``：多层嵌套 loop。

Loop 中可以使用 ``break`` 和 ``continue``。

.. code-block:: python

    for i, j in T.grid(8, 8):
        B[i, j] = T.max(A[i, j], T.float32(0.0))

.. code-block:: c++

    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        B_ptr[i * 8 + j] = max(A_ptr[i * 8 + j], 0.0f);

``T.unroll(4)`` 则不会生成 loop，而是直接展开为四条连续 statements。

while
-----

``while`` 会一直执行，直到条件变为 false。计数器需要使用 mutable scalar，
详见 :doc:`buffers`：

.. code-block:: python

    i: T.int32 = 0
    while i < 64:
        A[i] = A[i] + T.float32(1.0)
        i += 1

它会 lowering 为带有提前退出 ``break`` 的 ``while (1)``。其中计数器使用
一个只有一个元素的 register buffer：

.. code-block:: c++

    int i_ptr[1];
    i_ptr[0] = 0;
    while (1) {
      if (!(i_ptr[0] < 64)) { break; }
      A_ptr[i_ptr[0]] = A_ptr[i_ptr[0]] + 1.0f;
      i_ptr[0] = i_ptr[0] + 1;
    }
