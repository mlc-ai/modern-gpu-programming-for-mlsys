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

.. _chap_arch:

TIRx Compiler Internals: Compilation and Lowering Pipeline
===========================================================

``tvm.compile(mod, target, tir_pipeline="tirx")`` turns an authored TIRx module
into host launcher code and device code. The work does not happen in one step:
TIRx-specific constructs are lowered first, general-purpose TIRx normalization
and legalization passes then process the result, and the module is finally split
and prepared for code generation.

The exact sequence is defined in `compilation_pipeline.py
<https://github.com/apache/tvm/blob/v0.26.0/python/tvm/tirx/compilation_pipeline.py>`_.
This page explains where that sequence sits in ``tvm.compile``, what each stage
changes, and where the host and device paths diverge.

The overall compilation path
----------------------------

``tvm.compile`` first binds the target, runs the **tirx pipeline** (the module-level
passes below), then applies **finalization** passes separately to the host and
device functions, and finally hands each device function to the CUDA code
generator:

.. code-block:: text

    authored TIRx  ──BindTarget──▶  tirx_pipeline  ──▶  host func  ──host finalize──▶  C/LLVM
                                          │
                                          └──────────▶  device func ──device finalize──▶  CUDA

Pass order inside ``tirx_pipeline``
-----------------------------------

The pipeline is organized into the 19 steps below. Common-subexpression
elimination is optional, while vectorization and unrolling behavior can be
controlled through ``PassContext``:

.. list-table::
   :header-rows: 1
   :widths: 6 24 24 46

   * - #
     - Stage
     - Pass
     - What it does
   * - 1
     - TIRx lowering
     - ``LowerTIRx``
     - the core lowering — see `Inside LowerTIRx`_ below
   * - 2
     - TIR normalization
     - ``UnifyThreadBinding``
     - merges equivalent thread-axis bindings so each ``threadIdx`` / ``blockIdx``
       axis is declared once
   * - 3
     - TIR normalization
     - ``StmtSimplify``
     - statement-level arithmetic simplification (the arith analyzer)
   * - 4
     - TIR normalization
     - ``LowerTIRxOpaque``
     - lowers remaining opaque constructs, including thread-binding loops,
       unit loops, and pragma annotations
   * - 5
     - TIR normalization
     - ``FlattenBuffer``
     - flattens the remaining multi-dimensional TIR ``BufferLoad`` /
       ``BufferStore`` accesses to 1-D
   * - 6
     - Compute legalization
     - ``BF16ComputeLegalize``
     - rewrites ``bfloat16`` compute to a legal (f32-up-cast) form
   * - 7
     - TIR normalization
     - ``NarrowDataType(32)``
     - narrows index/loop scalar ``Expr`` types to 32-bit where provably safe
   * - 8
     - Loop lowering
     - ``VectorizeLoop``
     - lowers ``T.vectorized`` loops to vector operations; when
       ``tir.disable_vectorize`` is set, it instead scalarizes those loops
   * - 9
     - Loop lowering
     - ``UnrollLoop``
     - unrolls loops marked ``T.unroll``; ordinary constant loops are
       auto-unrolled only when the corresponding config or pragma enables it
   * - 10
     - TIR normalization
     - ``StmtSimplify``
     - simplify again, now that vectorize/unroll exposed constants
   * - 11
     - TIR normalization
     - ``CommonSubexprElim``
     - hoists repeated subexpressions into temporaries (skipped if
       ``tir.disable_cse_tir``)
   * - 12
     - Compute legalization
     - ``FP8ComputeLegalize``
     - rewrites ``float8`` compute to a legal form
   * - 13
     - Validation and ABI
     - ``VerifyMemory``
     - checks no host-side code directly dereferences device memory (a safety gate)
   * - 14
     - Validation and ABI
     - ``AnnotateEntryFunc``
     - marks the sole function, or the sole externally visible PrimFunc in a
       multi-function module, as the entry point
   * - 15
     - Validation and ABI
     - ``SplitHostDevice``
     - identifies device regions, splits host and device PrimFuncs, and lowers
       host-to-device calls to the kernel-launch ABI
   * - 16
     - Validation and ABI
     - ``LowerIket``
     - removes frontend-only NVIDIA IKET annotations for normal builds, or emits
       IKET metadata and placeholders when the IRModule is explicitly IKET-enabled
   * - 17
     - Validation and ABI
     - ``MakePackedAPI``
     - rewrites the host function to the packed-func ABI (the launcher TVM calls)
   * - 18
     - Storage legalization
     - ``FP8StorageLegalize``
     - legalizes ``float8`` storage to ``uint8`` containers
   * - 19
     - Storage legalization
     - ``BF16StorageLegalize``
     - legalizes ``bfloat16`` storage to ``uint16`` containers

Host and device finalization
----------------------------

The 19 listed steps form ``tirx_pipeline``. After that module-level pipeline,
``tvm.compile`` runs a different finalization sequence for each function kind:

- **host**: ``LowerTVMBuiltin`` (lower ``tvm_*`` builtins), ``LowerIntrin``
  (target-specific intrinsics)
- **device**: ``LowerWarpMemory`` (warp-scoped buffers → shuffles), ``StmtSimplify``,
  ``LowerIntrin``

Inside ``LowerTIRx``
--------------------

In a normal build, ``LowerTIRx`` is itself a two-pass sequence defined in
`lower_tirx.cc
<https://github.com/apache/tvm/blob/v0.26.0/src/tirx/transform/lower_tirx.cc>`_:

.. code-block:: text

    LowerTIRx = Sequential([ TilePrimitiveDispatch, LowerTIRxCleanup ])

- **``TilePrimitiveDispatch``** selects a backend variant for every
  ``TilePrimitiveCall`` (``copy``, ``gemm``, ``reduction``, …), replaces the
  call with the selected implementation, and resolves execution-scope IDs such
  as ``T.cta_id`` and ``T.thread_id`` into kernel launch parameters and bindings.
- **``LowerTIRxCleanup``** runs the ``LayoutApplier``: it resolves every
  ``TileLayout``-typed buffer access into concrete physical address arithmetic
  (``addr = data + elem_offset + layout.apply(*coord, shape=shape)``), replaces
  layout-aware buffer parameters with physical views, and removes explicit
  buffer offsets.

After ``LowerTIRx``, tile primitives and ``TileLayout`` indirection are gone,
and execution-scope IDs have been resolved. Some opaque TIRx constructs still
remain; the later ``LowerTIRxOpaque`` pass converts those before
``tirx.transform.FlattenBuffer`` flattens ordinary TIR buffer accesses.

End-to-end IR evolution
-----------------------

Take a one-line scale kernel:

.. code-block:: python

    @T.prim_func
    def scale(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        B = T.match_buffer(B_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
        B[tx] = A[tx] * T.float32(2.0)

This simple 1-D kernel has no nontrivial ``TileLayout``; it chiefly shows how
``LowerTIRx`` turns scope IDs into real thread axes. The core body looks like the
following excerpt. Buffer declarations and an unused warp-ID binding are omitted;
``A_1`` and ``B_1`` are the materialized physical views:

.. code-block:: python

    # match_buffer / decl_buffer declarations omitted
    with T.launch_thread("blockIdx.x", 1) as blockIdx_x:
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        bx: T.let = blockIdx_x
        tx: T.let = threadIdx_x
        B_1[threadIdx_x] = A_1[threadIdx_x] * T.float32(2.0)

``SplitHostDevice`` then turns the single function into a host launcher and a
device kernel. ``MakePackedAPI`` later rewrites the host launcher to TVM's
packed-function ABI:

.. code-block:: text

    @I.ir_module
    class Module:
        def main(...):          # host: packed-API launcher (computes the grid/block, launches)
            ...
        def scale_kernel(...):  # device: the __global__ body, run on the GPU

The CUDA backend then renders ``scale_kernel`` to the ``__global__`` function
(``B_ptr[threadIdx.x] = A_ptr[threadIdx.x] * 2.0f``).

Inspecting intermediate IR and generated code
----------------------------------------------

You can run any prefix of the pipeline by hand to inspect a stage — this is how the
IR snippets across these docs were produced:

.. code-block:: python

    from tvm.tirx import transform as TT

    target = tvm.target.Target("cuda")
    mod = TT.BindTarget(target.with_host("llvm"))(tvm.IRModule({"main": scale}))
    mod = TT.LowerTIRx()(mod)         # tile primitives dispatched, layouts applied
    print(mod.script())               # inspect the lowered TIRx IR

Or compile the whole module and read the generated CUDA:

.. code-block:: python

    exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")
    print(exe.mod.imports[0].inspect_source())
