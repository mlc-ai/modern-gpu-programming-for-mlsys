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

Compilation and lowering
========================

``tvm_tirx.tirx.compile`` accepts a ``tirx.Function`` or ``IRModule`` and returns an ``Executable``. The default pipeline handles CUDA entries and their CPU launchers.

The default pipeline
--------------------

Target/backend binding precedes ``TileDispatch``, ``LowerTensorLayouts``, simplification, ``FlattenTensor``, index narrowing, vectorization and unrolling, optional common-subexpression elimination, memory verification and thread-allreduce lowering. Finalization splits host/device code, lowers IKET and device helpers, legalizes low-precision compute/storage and adapts the packed ABI. Host and device functions then receive separate finalization and code generation.

The exact order is in `compilation_pipeline.py <https://github.com/apache/tvm-tirx/blob/a821c08826b4721d47122ee9b1dced7f3b88e738/python/tvm_tirx/tirx/compilation_pipeline.py>`_. Dispatch checks the requested opcode and operand layout; it does not fall back to an older instruction-selection algorithm.

A complete device entry
-----------------------


.. code-block:: python

    import tvm_tirx
    from tvm_tirx.script import tirx as T

    @T.function
    def scale(A: T.Tensor((1024,), "float32"), B: T.Tensor((1024,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(4,), block=256))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        B[bx * 256 + tx] = A[bx * 256 + tx] * T.float32(2)

Inspect an intermediate module
------------------------------


.. code-block:: python

    from tvm_tirx.tirx import transform as TT

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"})
    mod = tvm_tirx.IRModule({"main": scale})
    mod = TT.BindTarget(target)(mod)
    mod = TT.TileDispatch()(mod)
    print(mod.script())

Compile and inspect CUDA
------------------------


.. code-block:: python

    exe = tvm_tirx.tirx.compile(tvm_tirx.IRModule({"main": scale}), target=target)
    print(exe.mod.imports[0].inspect_source())
