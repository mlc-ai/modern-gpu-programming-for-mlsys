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
expressions instead have a ``PointerType``.) Each dtype prints to the matching
CUDA type. Allocating local and shared buffers across several dtypes, plus a
vectorized ``float32x4`` load/store:

.. code-block:: python

    @T.prim_func
    def dtypes(A_ptr: T.handle, O_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        O = T.match_buffer(O_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([64])
        f16  = T.alloc_local((1,), "float16")        # register scalars ...
        bf16 = T.alloc_local((1,), "bfloat16")
        i32  = T.alloc_local((1,), "int32")
        u8   = T.alloc_local((1,), "uint8")
        b1   = T.alloc_local((1,), "bool")
        sm   = T.alloc_shared((64,), "float16")      # ... and a shared tile
        v    = T.alloc_local((1,), "float32x4")      # a vector-dtype register (float4)
        v[0] = A.vload([tx * 4], dtype="float32x4")  # vectorized load
        O.vstore([tx * 4], v[0])                     # vectorized store
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
``float32x4`` ``vload`` / ``vstore`` then moves it as one 16-byte access. The vector
dtype is not tied to ``vload`` — any buffer or scalar can carry it.

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
     - ``PointerType`` → ``T*``
     - (vector dtypes → CUDA vector types)

dtype vs type
-------------

The ``dtype`` is *low-level* — it says "what bits". An expression's ``.ty`` is
its high-level **type**: ``PrimType(dtype)`` for a scalar or vector, or
``PointerType(PrimType(dtype), scope)`` for a pointer. Most expressions have a
``PrimType``; the distinction matters mainly for **pointers**.

Pointers (``handle``)
---------------------

A buffer value is a ``Var`` whose type is ``BufferType``. Its ``data`` property
projects an **immutable**, pointer-typed ``Expr``; the projection itself need not
be a ``Var``. That shapes how you obtain and reuse one:

- ``T.alloc_buffer(...)`` allocates storage **and** defines its ``data`` pointer.
- ``T.decl_buffer(..., data=ptr)`` declares a buffer over an existing compatible
  pointer expression ``ptr``.
- To back a buffer with a pointer **expression** — e.g. ``T.ptx.map_shared_rank``
  (PTX ``mapa``) giving another cluster CTA's shared address — convert the raw
  ``uint64`` address returned by ``map_shared_rank`` to a pointer with the
  intended element type and storage scope. The typed pointer expression can be
  passed directly as ``data``; assigning it to an unannotated name first creates
  an immutable ``Bind``:

  .. code-block:: python

      from tvm.ir import PointerType, PrimType

      ptr = T.reinterpret(
          PointerType(PrimType("uint64"), "shared"),
          T.ptx.map_shared_rank(mbar.ptr_to([0]), 0),
      )
      remote_mbar = T.decl_buffer([1], "uint64", data=ptr, scope="shared")

  Pointer bindings cannot be reassigned; use a new name for a different
  pointer value.
