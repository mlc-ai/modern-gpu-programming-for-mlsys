Data types and expressions
==========================

每个 TIRx expression 都带有一个低层 **dtype** 和一个高层 **type**。

Expression dtype
----------------

``PrimExpr`` 的 ``.dtype`` 是其 scalar（或 vector）element type，例如 ``float32``、``float16``、``bfloat16``、``int32``、``uint8``、``bool``、低精度 ``float8_e4m3fn`` / ``float4_e2m1fn``、``handle``（pointer），以及 ``float32x4`` 这样的 vector form。每种 dtype 都会打印成匹配的 CUDA type。下面示例跨多个 dtype 分配 local 和 shared buffer，并执行 vectorized ``float32x4`` load/store：

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

它会 lower 成（生成 CUDA，省略部分）：

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

Buffer 的 dtype 本身也可以是 **vector type**：``T.alloc_local((1,), "float32x4")`` 会直接声明一个 ``float4`` register（用 ``v[0]`` 索引），随后 ``float32x4`` 的 ``vload`` / ``vstore`` 会以一次 16-byte access 移动它。Vector dtype 不绑定到 ``vload``；任何 buffer 或 scalar 都可以携带它。

dtype -> CUDA 映射如下：

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
     - ``handle`` → ``T*`` (pointer)
     - (vector dtypes → CUDA vector types)

dtype vs type
-------------

``dtype`` 是 *low-level* 的，说明“哪些 bit”。另外，值还有一个高层 **type**：scalar 使用 ``PrimType(dtype)``，pointer 使用 ``PointerType(PrimType(dtype), scope)``。大多数 expression 是 scalar（``PrimType``）；type system 主要在 **pointer** 上重要。

Pointer（``handle``）
--------------------------

Buffer 的 ``data``，也就是 pointer，是 pointer type 的 ``Var``，并且是 immutable（pointer 不会被重新赋值）。这决定了你如何取得它：

- ``T.alloc_buffer(...)`` 分配 storage，并定义它的 ``data`` pointer。
- ``T.decl_buffer(..., data=ptr)`` 在已有 pointer ``Var`` ``ptr`` 上声明 buffer。
- 如果要让 buffer 以 pointer **expression** 为 backing，例如 ``T.ptx.map_shared_rank``（PTX ``mapa``）给出的另一个 cluster CTA 的 shared address，必须先把这个 expression 绑定成 pointer ``Var``（``data`` 必须是 ``Var``，不能是 expression），使用 ``PointerType`` 的 ``T.let``：

  .. code-block:: python

      from tvm.ir.type import PointerType, PrimType

      ptr: T.let[T.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = \
          T.reinterpret("handle", T.ptx.map_shared_rank(mbar.ptr_to([0]), 0))
      remote_mbar = T.decl_buffer([1], "uint64", data=ptr, scope="shared")
