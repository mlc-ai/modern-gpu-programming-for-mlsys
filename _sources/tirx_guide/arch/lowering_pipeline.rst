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

TIRx Lowering Pipeline
======================

``tvm.compile(mod, target, tir_pipeline="tirx")`` takes a TIRx module and
eventually produces two pieces of code: a CPU-side launcher that prepares the
arguments and launches a GPU kernel, and the GPU kernel that performs the
computation. The compiler reaches that result through an ordered series of
passes. Each pass performs a particular transformation, validation, or
annotation on the IR.

The exact pass order is defined in Apache TVM's `python/tvm/tirx/compilation_pipeline.py
<https://github.com/apache/tvm/blob/v0.26.0/python/tvm/tirx/compilation_pipeline.py>`_.

The overall compilation path
----------------------------

The ``target`` identifies the hardware and code-generation backend. The example
below uses CUDA for the device and LLVM for the host. ``tvm.compile`` first
attaches that target information to the module and then runs the module-level
**tirx pipeline**. Once the pipeline has separated the CPU-side host function
from the GPU-side device function, each follows a target-specific finalization
path before code generation:

.. code-block:: text

    authored TIRx
          │ BindTarget
          ▼
    tirx_pipeline
    (SplitHostDevice creates the two paths)
          ├── host PrimFunc   ──host finalization──▶ C/LLVM
          └── device PrimFunc ─device finalization─▶ CUDA

A ``PrimFunc`` is TIR's representation of a function. The host PrimFunc above is
the CPU-side launcher, while the device PrimFunc is the GPU kernel.
``Finalization`` refers to the last target-specific transformations performed
before code generation.

Pass order inside ``tirx_pipeline``
-----------------------------------

The table lists the 19 pipeline steps in execution order. An ABI is the calling
convention between functions; the ABI passes below adapt ordinary TIR functions
to forms that the runtime can invoke. ``PassContext`` holds compiler options:
common-subexpression elimination can be disabled, and it also controls aspects
of vectorization and unrolling.

.. list-table::
   :header-rows: 1
   :widths: 6 24 24 46

   * - #
     - Category
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
     - simplifies arithmetic expressions in the IR
   * - 4
     - TIR normalization
     - ``LowerTIRxOpaque``
     - converts thread-binding loops, eliminates unannotated unit loops, and
       normalizes loop pragmas
   * - 5
     - TIR normalization
     - ``FlattenBuffer``
     - flattens the remaining multi-dimensional TIR ``BufferLoad`` /
       ``BufferStore`` accesses to 1-D
   * - 6
     - Compute legalization
     - ``BF16ComputeLegalize``
     - when the target lacks native ``bfloat16`` compute, promotes operations to
       ``float32`` and rewrites them into a legal form
   * - 7
     - TIR normalization
     - ``NarrowDataType(32)``
     - narrows index expressions and loop variables to 32 bits where provably safe
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
     - simplifies again after vectorization and unrolling expose more constants
   * - 11
     - TIR normalization
     - ``CommonSubexprElim``
     - hoists repeated subexpressions into temporaries (skipped if
       ``tir.disable_cse_tir``)
   * - 12
     - Compute legalization
     - ``FP8ComputeLegalize``
     - when the target lacks native ``float8`` compute, promotes operations to a
       supported type (``float32`` by default)
   * - 13
     - Validation and ABI
     - ``VerifyMemory``
     - ensures that host-side code does not directly dereference device memory
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
     - removes NVIDIA IKET annotations in normal builds, or lowers them for
       tracing when IKET is enabled
   * - 17
     - Validation and ABI
     - ``MakePackedAPI``
     - rewrites the host function to the packed-function ABI used by the TVM runtime
   * - 18
     - Storage legalization
     - ``FP8StorageLegalize``
     - when the target lacks native ``float8`` storage, uses ``uint8`` containers
   * - 19
     - Storage legalization
     - ``BF16StorageLegalize``
     - when the target lacks native ``bfloat16`` storage, uses ``uint16`` containers

Host and device finalization
----------------------------

The 19 listed steps form ``tirx_pipeline``. After that module-level pipeline,
``tvm.compile`` runs a different finalization sequence for each function kind:

- **host**: ``LowerTVMBuiltin`` (lowers ``tvm_*`` builtins), ``LowerIntrin``
  (lowers target-specific intrinsics)
- **device**: ``LowerWarpMemory`` (lowers warp-scoped buffers to shuffles),
  ``StmtSimplify``, ``LowerIntrin``

Inside ``LowerTIRx``
--------------------

``LowerTIRx`` has two main jobs: choosing concrete implementations for tile-level
operations, and turning logical data layouts into physical memory indices. Its
core transformation is the following two-pass sequence, defined in Apache TVM's
`src/tirx/transform/lower_tirx.cc
<https://github.com/apache/tvm/blob/v0.26.0/src/tirx/transform/lower_tirx.cc>`_:

.. code-block:: text

    LowerTIRx = Sequential([ TilePrimitiveDispatch, LowerTIRxCleanup ])

- **``TilePrimitiveDispatch``** chooses concrete implementations for tile
  operations. TIRx represents operations such as ``copy``, ``gemm``, and
  ``reduction`` as ``TilePrimitiveCall`` nodes; this pass selects a backend
  implementation for each one. It also turns abstract execution-scope
  identifiers such as ``T.cta_id`` and ``T.thread_id`` into kernel-launch
  parameters and thread bindings.
- **``LowerTIRxCleanup``** maps logical coordinates to physical indices. It
  applies supported logical layouts to buffer accesses so later passes can work
  directly with concrete index expressions.

After ``LowerTIRx``, tile operations have been replaced by their selected
implementations, logical layouts have become physical indices, and abstract
identifiers such as ``T.cta_id`` and ``T.thread_id`` have become thread
bindings. Thread-binding loops and TIRx-specific loop annotations may still
remain; ``LowerTIRxOpaque`` normalizes those structures before
``tirx.transform.FlattenBuffer`` flattens ordinary TIR buffer accesses.

Compiling a Simple Kernel to CUDA
---------------------------------

The following scale kernel illustrates two transformations: how ``T.cta_id``
and ``T.thread_id`` become concrete thread identifiers, and how one TIRx
function is split into a CPU-side launcher and a GPU kernel. The kernel processes
1,024 elements using 4 CUDA thread blocks (CTAs), with 256 threads per CTA.

**1. TIRx source uses abstract thread identifiers.**

.. code-block:: python

    import tvm
    from tvm.script import tirx as T

    @T.prim_func
    def scale(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (1024,), "float32")
        B = T.match_buffer(B_ptr, (1024,), "float32")
        T.device_entry()
        bx = T.cta_id([4])
        tx = T.thread_id([256])
        B[bx * 256 + tx] = A[bx * 256 + tx] * T.float32(2.0)

``T.device_entry()`` marks the entry into GPU code. ``LowerTIRx`` uses the
marker to establish the corresponding thread bindings; the later
``SplitHostDevice`` pass extracts the resulting device region into a separate
kernel. ``T.cta_id([4])`` specifies 4 CTAs along x, while
``T.thread_id([256])`` specifies 256 threads per CTA. At this point, ``bx`` and
``tx`` are still abstract TIRx identifiers.

**2. ``LowerTIRx`` lowers the abstract identifiers to TIR thread bindings.** It
binds ``bx`` to ``blockIdx.x`` and ``tx`` to ``threadIdx.x``. Omitting buffer
declarations, the core computation is equivalent to:

.. code-block:: python

    with T.launch_thread("blockIdx.x", 4) as bx:
        tx = T.launch_thread("threadIdx.x", 256)
        B[bx * 256 + tx] = A[bx * 256 + tx] * T.float32(2.0)

This is still TIR, not CUDA source code. The excerpt retains only the important
mapping; the next section shows how to print the complete compiler output.

**3. Later passes split host/device code and generate CUDA.** The compiler starts
with one TIRx function. After ``LowerTIRx`` establishes thread bindings and a
device region, ``SplitHostDevice`` produces two TIR functions (PrimFuncs):

.. code-block:: text

    host launcher (generated from scale)
      `-- launch scale_kernel with gridDim.x = 4 and blockDim.x = 256

    device scale_kernel
      `-- each GPU thread multiplies one input element by 2

The host function retains the kernel-launch logic, while the device function
retains the elementwise computation. ``MakePackedAPI`` then adapts the host
function to the uniform calling convention used by the TVM runtime. The device
function proceeds to the CUDA backend, which generates code equivalent to:

.. code-block:: cuda

    __global__ void scale_kernel(float* A, float* B) {
        int i = blockIdx.x * 256 + threadIdx.x;
        B[i] = A[i] * 2.0f;
    }

In short, TIRx describes the thread organization and computation,
``LowerTIRx`` turns abstract identifiers into TIR thread bindings,
``SplitHostDevice`` separates CPU-side launch logic from GPU-side computation,
and the CUDA backend finally emits CUDA source code.

No bounds check is needed here because ``4 * 256`` is exactly 1,024. For a
general length ``N``, choose the CTA count with ceiling division and guard the
kernel body with ``i < N``.

Inspecting intermediate IR and generated code
----------------------------------------------

To inspect an intermediate IR, run only the first few passes and stop before the
rest of the pipeline. The following code first places ``scale`` in an
``IRModule`` under the global name ``main``. The CUDA target selects the GPU
backend, while ``with_host("llvm")`` selects LLVM for the CPU-side launcher.
``BindTarget`` attaches both choices to the module, after which we run only
``LowerTIRx``:

.. code-block:: python

    from tvm.tirx import transform as TT

    target = tvm.target.Target("cuda").with_host("llvm")
    mod = tvm.IRModule({"main": scale})
    mod = TT.BindTarget(target)(mod)
    mod = TT.LowerTIRx()(mod)         # run LowerTIRx to lower abstract thread IDs
    print(mod.script())               # inspect the IR after LowerTIRx

The output should contain thread bindings for ``blockIdx.x`` and ``threadIdx.x``;
the original ``T.cta_id`` and ``T.thread_id`` calls should be gone.

To inspect the final CUDA, run the complete pipeline. The host module in this
example imports exactly one device module, so ``imports[0]`` is the generated
CUDA module, and ``inspect_source()`` returns its source code:

.. code-block:: python

    exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")
    cuda_mod = exe.mod.imports[0]
    print(cuda_mod.inspect_source())

The generated code should contain ``blockIdx.x``, ``threadIdx.x``, and the
elementwise multiplication that doubles each input value.
