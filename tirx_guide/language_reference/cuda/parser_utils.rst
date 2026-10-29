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

Parser utilities
================

A few helpers act at **parse time** (when TVMScript is turned into TIRx), letting
you inline Python-computed values, factor out reusable fragments, and bundle
parser-side state.

``T.meta_var`` — preserve Python metadata
-----------------------------------------

``T.meta_var(x)`` returns ``x`` unchanged. It is useful for retaining a
parser-side handle or a tuple of Python metadata. It does not bypass the
ordinary assignment rules: scalar expressions assigned inside TVMScript create
native value bindings. Compute shape parameters, instruction widths, and TMA
box sizes in the Python factory before ``@T.function``.

.. code-block:: python

    widths = T.meta_var((32, 64))  # the tuple remains Python metadata
    for j in T.unroll(widths[0]):
        acc[0] = acc[0] + A[tx, j]

A plain ``range(n)`` creates a serial TIRx loop; ``T.unroll(n)`` marks it for
unrolling during lowering. ``meta_var`` does not make a runtime value recompute
when its inputs change. For a persistent scheduler, compute tile coordinates
inside each task loop, after the current cursor is updated.

``@T.inline`` — inline functions
--------------------------------

``@T.inline`` defines a function whose body is **inlined at each call site** during
parsing — no call appears in the generated code. It follows Python's lexical (LEGB)
scoping with late binding, so a parameter shadows an enclosing variable:

.. code-block:: python

    @T.inline
    def add_into(acc, x):
        acc[0] = acc[0] + x

    add_into(acc, A[tx, j])       # inlined -> acc[0] = acc[0] + A[tx, j]

``@T.meta_class`` — parser-side state objects
---------------------------------------------

``@T.meta_class`` marks a plain Python class whose **instances are parser meta
values**: their fields can hold buffers and scalars, so you can bundle related
allocations and state into one object and use it in the kernel body.

Define ``State`` outside ``@T.function``. Instantiate it and use its fields
inside the kernel, where ``smem`` has already been allocated.

.. code-block:: python

    @T.meta_class
    class State:
        def __init__(self, smem):
            self.acc = T.bind_(T.alloc_local([1], "float32"))
            self.buf = T.bind_(T.decl_tensor(smem, ty_args=[
                T.Tensor((64,), "float16", scope="shared.dyn")]))

    s = State(smem.ptr_to([0]))
    s.acc[0] = T.float32(0.0)     # use its fields like ordinary buffers
    # ... s.buf[i] ...

This is handy for grouping a kernel's pipeline state (barriers, accumulators,
scratch views) instead of threading many separate locals through the body.

``T.constexpr``
---------------

A parameter annotated with ``T.constexpr`` is fixed by ``@T.jit``'s
``.specialize(...)``. Inside a function body, ``if T.constexpr(condition)``
selects a Python branch during parsing. Use it for known specialization flags,
such as ``is_causal`` in FA4. An ordinary ``if`` constructs IR control flow and
parses both branches.
