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

Every TIRx expression carries a high-level **type**. Scalar and vector types
also contain a low-level **dtype**.

Expression dtypes
-----------------

A scalar or vector ``Expr`` exposes its ``PrimType`` through ``.ty`` and its
element dtype through ``.ty.dtype`` — ``float32``, ``float16``, ``bfloat16``,
``int32``, ``uint8``, ``bool``, the low-precision ``float8_e4m3fn`` /
``float4_e2m1fn`` …, and vector forms such as ``float32x4``. (Pointer
expressions instead have a ``PtrType``.) Each dtype prints to the matching
CUDA type. Allocating local and shared buffers across several dtypes, plus a
vectorized ``float32x4`` load/store:

.. code-block:: python

    @T.function
    def dtypes(A: T.Tensor((256,), "float32"), O: T.Tensor((256,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=[1], block=[64])); bx = T.cuda.block_idx("x"); tx = T.cuda.thread_idx("x")
        f16  = T.alloc_local((1,), "float16")        # register scalars ...
        bf16 = T.alloc_local((1,), "bfloat16")
        i32  = T.alloc_local((1,), "int32")
        u8   = T.alloc_local((1,), "uint8")
        b1   = T.alloc_local((1,), "bool")
        sm   = T.alloc_shared((64,), "float16")      # ... and a shared tile
        v    = T.alloc_local((1,), "float32x4")      # a vector-dtype register (float4)
        A4 = T.decl_tensor(A.data, ty_args=[T.Tensor((64,), "float32x4")])
        O4 = T.decl_tensor(O.data, ty_args=[T.Tensor((64,), "float32x4")])
        v[0] = A4[tx]                               # vectorized load
        O4[tx] = v[0]                               # vectorized store
        # ... (use f16/bf16/i32/u8/b1/sm) ...

lowers to (generated CUDA, elided):

.. code-block:: c++

    half          f16_ptr[1];               // float16
    nv_bfloat16   bf16_ptr[1];              // bfloat16
    int           i32_ptr[1];               // int32
    uchar         u8_ptr[1];                // uint8
    bool          b1_ptr[1];                // bool
    __shared__ alignas(64) half sm_ptr[64]; // shared float16
    float4        v_ptr[1];                 // float32x4  (vector)
    v_ptr[0]                  = *(float4*)(A_ptr + tx * 4);   // vectorized load
    *(float4*)(O_ptr + tx * 4) = v_ptr[0];                   // vectorized store

A buffer's dtype can itself be a **vector type**: ``T.alloc_local((1,), "float32x4")``
declares a ``float4`` register directly (you index it as ``v[0]``), and a
``float32x4`` tensor views move it as one 16-byte access. A vector
dtype can also be used by ordinary tensors and scalars.

The resulting dtype → CUDA mapping is:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - dtype → CUDA
     - dtype → CUDA
     - dtype → CUDA
   * - ``float32`` → ``float``
     - ``float16`` → ``half``
     - ``bfloat16`` → ``nv_bfloat16``
   * - ``int32`` → ``int``
     - ``uint8`` → ``uchar``
     - ``bool`` → ``bool``
   * - ``float32x4`` → ``float4``
     - ``PtrType`` → ``T*``
     - (vector dtypes → CUDA vector types)

dtype vs type
-------------

The ``dtype`` is *low-level* — it says "what bits". An expression's ``.ty`` is
its high-level **type**: ``PrimType(dtype)`` for a scalar or vector, or
``PtrType(PrimType(dtype), scope)`` for a pointer. Most expressions have a
``PrimType``; the distinction matters mainly for **pointers**.

Pointers (``handle``)
---------------------

A buffer value is a ``Var`` whose type is ``TensorType``. Its ``data`` property
projects an **immutable**, pointer-typed ``Expr``; the projection itself need not
be a ``Var``. That shapes how you obtain and reuse one:

- ``T.alloc_tensor(ty_args=[T.Tensor(...)])`` allocates storage **and** defines its ``data`` pointer.
- ``T.decl_tensor(ptr, ty_args=[T.Tensor(...)])`` declares a buffer over an existing compatible
  pointer expression ``ptr``.
- To back a buffer with a pointer **expression** — e.g. ``T.ptx.mapa.u64``
  (PTX ``mapa``) giving another cluster CTA's shared address — convert the raw
  ``uint64`` address returned by ``mapa.u64`` to a pointer with the
  intended element type and storage scope. The typed pointer expression can be
  passed directly to ``T.decl_tensor``; assigning it to an unannotated name first creates
  an immutable ``Bind``:

  .. code-block:: python

      from tvm_tirx.ir import PtrType

      mapped = T.alloc_local((1,), "uint64")
      T.ptx.mapa.u64(mapped[0], mbar.ptr_to([0]), T.uint32(0))
      ptr = T.reinterpret(mapped[0], ty=PtrType("uint64", "shared"))
      remote_mbar = T.decl_tensor(ptr, ty_args=[T.Tensor(
          (1,), "uint64", scope="shared", layout="default")])

  Pointer bindings cannot be reassigned; use a new name for a different
  pointer value.
