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

Data types and expressions
==========================

Expressions expose their type through ``.ty``. Scalars and vectors have ``PrimType`` and its element ``dtype``; tensor values have ``TensorType`` and pointers have ``PtrType``. Dtype strings include ``float32``, ``float16``, ``bfloat16``, ``int32``, ``uint8`` and ``bool``. ``T.float32(x)`` explicitly converts a scalar.


.. code-block:: python

    @T.function
    def dtypes(A: T.Tensor((256,), "float32"), O: T.Tensor((256,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=64))
        tx = T.cuda.thread_idx("x")
        f16 = T.alloc_local((1,), "float16")
        bf16 = T.alloc_local((1,), "bfloat16")
        i32 = T.alloc_local((1,), "int32")
        for j in T.vectorized(4):
            O[tx * 4 + j] = A[tx * 4 + j]

Typed pointers
--------------

A tensor exposes its typed ``data`` pointer. Reinterpret an address with ``T.reinterpret(value, ty=PtrType(...))``, then attach a tensor access contract with ``T.decl_tensor``. Pointer bindings are immutable. ``ptr_to`` indices are in elements; ``tvm_tirx.tirx.ptr_byte_offset`` offsets are in bytes.

.. code-block:: python

    from tvm_tirx.ir import PtrType

    # Inside a cluster kernel with an existing shared mbarrier:
    mapped = T.alloc_local((1,), "uint64")
    T.ptx.mapa.u64(mapped[0], mbar.ptr_to([0]), T.uint32(0))
    ptr = T.reinterpret(mapped[0], ty=PtrType("uint64", "shared"))
    remote = T.decl_tensor(ptr, ty_args=[T.Tensor((1,), "uint64", scope="shared")])
