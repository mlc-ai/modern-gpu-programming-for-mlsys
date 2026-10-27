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
====================================

Kernel 参数使用 ``T.Tensor(shape, dtype)``，其 ``TensorType`` 保存 shape、dtype、scope、alignment 和 layout。普通 pointer 上的 tensor 与 CUDA Tensor Memory 使用不同的声明操作。下面按 allocation、shared memory、寄存器、TMEM 和 view 列出当前 API；代码内的语句放在 kernel 中，import 放在外部。

声明 Tensor
------------------

``alloc_tensor`` 分配存储；``decl_tensor`` 在带类型的 pointer 上声明别名。分配语句放在 kernel 内，import 放在外部。

.. code-block:: python

    # Inside @T.function:
    tile = T.alloc_tensor(ty_args=[T.Tensor((128, 64), "float16", scope="shared", align=1024)])
    registers = T.alloc_local((4,), "float32")
    alias = T.decl_tensor(tile.data, ty_args=[T.Tensor((128, 64), "float16", scope="shared")])

Shared Memory
--------------------------


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
----------------

从 CUDA backend 导入 pool。完成所有分配后调用 ``commit``；``align`` 以字节为单位，layout stride 以 dtype 元素为单位。

.. code-block:: python

    from tvm_tirx.backend.cuda.lang import SMEMPool

    # Inside @T.function, after device_entry:
    pool = SMEMPool()
    operand = pool.alloc_tcgen05_mma_AB((128, 64), "float16")
    barrier = pool.alloc((1,), "uint64", align=8)
    pool.commit()

寄存器与可变标量
----------------

``T.alloc_local`` 创建每个 thread 私有的数组。``total: T.float32 = 0.0`` 这样的带类型赋值创建可变局部单元；也可显式调用 ``T.local_scalar``。寄存器压力或动态索引可能导致数组 spill。

Tensor Memory
--------------------------

把 ``tcgen05.alloc`` 返回的实际地址交给 ``T.cuda.decl_tmem``。合法分配大小为 32、64、128、256、512 列。分配由完整 warp 共同执行；``cta_group::2`` 要求两个 CTA 各参与一个 warp。

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
----------------------

``sub`` 返回 tensor view；不带 ``sub`` 的切片返回供 tile 指令使用的 region。``view``、``permute``、``rearrange`` 修改元数据，不搬运数值。shape 和 alignment 满足条件时，``view("uint32")`` 可以重新解释打包存储。

.. code-block:: python

    # For a 256-element tensor A:
    rows = A.view(64, 4)
    transposed = rows.permute(1, 0)
    window = rows.sub[0:16, :]
    pointer = rows.ptr_to([2, 3])
