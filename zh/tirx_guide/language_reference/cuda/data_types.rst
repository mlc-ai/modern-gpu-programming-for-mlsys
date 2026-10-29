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

每个 TIRx 表达式都有高层的 **type**；scalar 和 vector type 还包含底层的
**dtype**。

表达式的 dtype
--------------

Scalar 或 vector ``Expr`` 通过 ``.ty`` 暴露其 ``PrimType``，通过
``.ty.dtype`` 暴露元素 dtype，例如 ``float32``、``float16``、
``bfloat16``、``int32``、``uint8``、``bool``、低精度
``float8_e4m3fn`` / ``float4_e2m1fn``，以及 ``float32x4`` 这样的 vector
类型。（Pointer 表达式的 type 则是 ``PointerType``。）生成 CUDA 时，每种
dtype 都会变成对应的 CUDA 类型。下面同时分配几种 dtype 的 local 和
shared buffers，并执行一次 ``float32x4`` vector load/store：

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
    bool          b1_ptr[1];                // bool
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
     - ``bool`` → ``bool``
   * - ``float32x4`` → ``float4``
     - ``PointerType`` → ``T*``
     - vector dtypes → CUDA vector types

dtype 与 type
-------------

``dtype`` 是底层表示，描述一个值由哪些 bits 组成。表达式的 ``.ty`` 是
其高层 **type**：scalar 或 vector 使用 ``PrimType(dtype)``，pointer 使用
``PointerType(PrimType(dtype), scope)``。大多数表达式的 type 是
``PrimType``；这一差别主要在处理 **pointer** 时发挥作用。

Pointer（``handle``）
---------------------

Buffer value 是一个 type 为 ``BufferType`` 的 ``Var``。其 ``data`` property
投影出一个 **immutable**、pointer-typed ``Expr``；这个投影本身不一定是
``Var``。获得和复用 pointer 的方式因此分为三种：

- ``T.alloc_buffer(...)`` 分配 storage，同时定义其 ``data`` pointer。
- ``T.decl_buffer(..., data=ptr)`` 在已有、类型兼容的 pointer 表达式
  ``ptr`` 上声明 buffer。
- 如果要用 pointer **表达式** 支撑 buffer，例如用
  ``T.ptx.map_shared_rank``（PTX ``mapa``）取得另一个 cluster CTA 的
  shared address，需要将 ``map_shared_rank`` 返回的原始 ``uint64`` 地址
  转换为具有正确元素类型和 storage scope 的 pointer。Typed pointer
  表达式可以直接作为 ``data`` 传入；先把它赋值给未标注类型的名字会创建
  immutable ``Bind``：

  .. code-block:: python

      from tvm.ir import PointerType, PrimType

      ptr = T.reinterpret(
          PointerType(PrimType("uint64"), "shared"),
          T.ptx.map_shared_rank(mbar.ptr_to([0]), 0),
      )
      remote_mbar = T.decl_buffer([1], "uint64", data=ptr, scope="shared")

  Pointer binding 不能重新赋值；如需不同的 pointer value，请使用新的名字。
