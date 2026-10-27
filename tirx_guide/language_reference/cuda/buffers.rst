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

Buffers and memory
==================

Kernel parameters use ``T.Tensor(shape, dtype)``. ``TensorType`` carries shape, dtype, storage scope, alignment and layout. Ordinary pointer-backed tensors and CUDA Tensor Memory use different declaration operations.

Declaring tensors
-----------------

``alloc_tensor`` allocates storage; ``decl_tensor`` aliases a typed pointer. Put allocation statements inside the kernel; imports belong outside it.

.. code-block:: python

    # Inside @T.function:
    tile = T.alloc_tensor(ty_args=[T.Tensor((128, 64), "float16", scope="shared", align=1024)])
    registers = T.alloc_local((4,), "float32")
    alias = T.decl_tensor(tile.data, ty_args=[T.Tensor((128, 64), "float16", scope="shared")])

Shared memory
-------------


.. code-block:: python

    @T.function
    def smem_demo(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=128))
        tx = T.cuda.thread_idx("x")
        sm = T.alloc_shared((128,), "float32")
        sm[tx] = A[tx]
        T.cuda.cta_sync()
        B[tx] = sm[tx] * T.float32(2)

SMEMPool
--------

Import the pool from the CUDA backend. Commit after all allocations; ``align`` is in bytes and layout strides are in dtype elements.

.. code-block:: python

    from tvm_tirx.backend.cuda.lang import SMEMPool

    # Inside @T.function, after device_entry:
    pool = SMEMPool()
    operand = pool.alloc_tcgen05_mma_AB((128, 64), "float16")
    barrier = pool.alloc((1,), "uint64", align=8)
    pool.commit()

Registers and mutable scalars
-----------------------------

``T.alloc_local`` creates per-thread arrays. Type-annotated assignments such as ``total: T.float32 = 0.0`` create mutable local cells; ``T.local_scalar`` is the explicit scalar form. Register arrays may spill under pressure or dynamic indexing.

Tensor memory
-------------

Use the actual address returned by ``tcgen05.alloc`` with ``T.cuda.decl_tmem``. Legal allocation sizes are 32, 64, 128, 256 and 512 columns. One complete warp participates; ``cta_group::2`` requires one warp in each CTA.

.. code-block:: python

    from tvm_tirx.backend.cuda.axis import TLane, TCol
    from tvm_tirx.tirx.layout import TileLayout, S

    # Inside a device entry; allocation and deallocation are warp-collective.
    addr = T.alloc_shared((1,), "uint32")
    if T.cuda.warp_id() == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(addr.ptr_to([0]), 512)
    T.cuda.cta_sync()
    acc = T.cuda.decl_tmem(addr[0], ty_args=[T.Tensor(
        (128, 512), "float32", scope="tmem",
        layout=TileLayout(S[(128, 512):(1@TLane, 1@TCol)]),
    )])
    # Issue and wait for all MMA / TMEM transfers before cleanup.
    T.cuda.cta_sync()
    if T.cuda.warp_id() == 0:
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(addr[0], 512)

Tensor APIs
-----------

``sub`` returns a tensor view; slicing without ``sub`` returns a region for tile instructions. ``view``, ``permute`` and ``rearrange`` change metadata without moving values. ``view("uint32")`` reinterprets packed storage when its shape and alignment permit it.

.. code-block:: python

    # For a 256-element tensor A:
    rows = A.view(64, 4)
    transposed = rows.permute(1, 0)
    window = rows.sub[0:16, :]
    pointer = rows.ptr_to([2, 3])
