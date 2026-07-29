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

数据类型与表达式
================

每个 TIRx 表达式都有底层的 **dtype** 和高层的 **type**。

表达式的 dtype
--------------

``PrimExpr`` 的 ``.dtype`` 表示 scalar 或 vector 的元素类型，例如
``float32``、``float16``、``bfloat16``、``int32``、``uint8``、``bool``、
低精度 ``float8_e4m3fn`` / ``float4_e2m1fn``、表示 pointer 的 ``handle``，
以及 ``float32x4`` 这样的 vector 类型。生成 CUDA 时，每种 dtype 都会变成
对应的 CUDA 类型。下面同时分配几种 dtype 的 local 和 shared buffers，并
执行一次 ``float32x4`` vector load/store：

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

生成的 CUDA 如下，省略了无关代码：

.. code-block:: c++

    half          f16_ptr[1];               // float16
    nv_bfloat16   bf16_ptr[1];              // bfloat16
    int           i32_ptr[1];               // int32
    uchar         u8_ptr[1];                // uint8
    signed char   b1_ptr[1];                // bool
    __shared__ alignas(64) half sm_ptr[64]; // shared float16
    float4        v_ptr[1];                 // float32x4  (vector)
    v_ptr[0]                  = *(float4*)(A_ptr + tx * 4);   // vectorized load
    *(float4*)(O_ptr + tx * 4) = v_ptr[0];                   // vectorized store

Buffer 本身也可以使用 **vector dtype**。
``T.alloc_local((1,), "float32x4")`` 会直接声明一个 ``float4`` register，
通过 ``v[0]`` 访问；``float32x4`` 的 ``vload`` / ``vstore`` 则用一次
16-byte access 搬运它。Vector dtype 并不只用于 ``vload``，普通 buffer
和 scalar 也可以使用。

常见 dtype 与 CUDA 类型的对应关系如下：

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
     - ``bool`` → ``signed char``
   * - ``float32x4`` → ``float4``
     - ``handle`` → ``T*`` （pointer）
     - vector dtypes → CUDA vector types

dtype 与 type
-------------

``dtype`` 是底层表示，描述一个值由哪些 bits 组成。一个值另外还有高层
**type**：scalar 使用 ``PrimType(dtype)``，pointer 使用
``PointerType(PrimType(dtype), scope)``。大多数表达式都是 scalar
（``PrimType``）；type system 主要在处理 **pointer** 时发挥作用。

Pointer（``handle``）
---------------------

Buffer 的 ``data`` 是一个 pointer，也是 pointer type 的 ``Var``。它是
**immutable** 的，不能被重新赋值。获得 pointer 的方式因此分为三种：

- ``T.alloc_buffer(...)`` 分配 storage，同时定义其 ``data`` pointer。
- ``T.decl_buffer(..., data=ptr)`` 在已有 pointer ``Var`` ``ptr`` 上声明 buffer。
- 如果要用 pointer **表达式** 支撑 buffer，例如用
  ``T.ptx.map_shared_rank``（PTX ``mapa``）取得另一个 cluster CTA 的
  shared address，必须先通过 ``PointerType`` 的 ``T.let`` 将表达式绑定为
  pointer ``Var``。``data`` 必须是 ``Var``，不能直接使用表达式：

  .. code-block:: python

      from tvm.ir.type import PointerType, PrimType

      ptr: T.let[T.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = \
          T.reinterpret("handle", T.ptx.map_shared_rank(mbar.ptr_to([0]), 0))
      remote_mbar = T.decl_buffer([1], "uint64", data=ptr, scope="shared")
