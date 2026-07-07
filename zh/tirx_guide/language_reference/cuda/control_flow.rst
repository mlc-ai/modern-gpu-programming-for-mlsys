Control flow
============

Control flow 包括 ``if``、loop family 和 ``while``；它们都会映射到直观的 CUDA。

if
--

Python ``if`` / ``else`` 会变成 CUDA ``if`` / ``else``。可以用 thread/lane 比较来 guard work，或者用 ``T.ptx.elect_sync()`` 选出一个 issuing thread：

.. code-block:: python

    if tx < 128:
        A[tx] = A[tx] * T.float32(2.0)
    else:
        A[tx] = A[tx] + T.float32(1.0)

    if T.ptx.elect_sync():
        ...                              # one elected lane (e.g. to issue TMA/MMA)

.. code-block:: c++

    if (((int)threadIdx.x) < 128) {
      A_ptr[tx] = A_ptr[tx] * 2.0f;
    } else {
      A_ptr[tx] = A_ptr[tx] + 1.0f;
    }

如果需要 expression-level choice（没有 branch），使用 ``T.if_then_else(cond, a, b)``。它会 lower 成 ternary，因此不引入 control-flow divergence：

.. code-block:: c++

    O_ptr[tx] = (A_ptr[tx] > 0.0f) ? A_ptr[tx] : 0.0f;

Uniform vs. divergent control flow
----------------------------------

``if tx < 128`` 这类 per-thread guard 对普通 work 没问题，但 **collective** operation 必须被它同步的所有线程 *uniformly* 到达。

例如，``T.cuda.cta_sync()`` 映射到 ``__syncthreads()`` ，需要 thread block 中所有线程到达。它绝不能放在 thread-divergent 或 warpgroup-divergent branch 内：如果放在 ``if wg_id == 0:`` 内，其他 warpgroup 永远不会到达，kernel 会 deadlock。当只有一个 warpgroup 需要同步时，使用 warpgroup-scoped ``T.cuda.warpgroup_sync(id)`` （见 :ref:`chap_gemm_advanced` 和 :doc:`threads_sync`）。

同样的注意事项适用于 barrier setup。``mbarrier`` 的 ``.init()`` 会 lower 成 single-thread guard（``if (threadIdx.x < 1)``）。如果把它嵌到另一个 divergent branch 中，barrier 可能保持未初始化，导致 unspecified launch failure。

loop
----

Loop 有四种形式；普通 Python ``range`` 会变成 ``T.serial``：

- ``T.serial(n)``：顺序 loop（ptxas 仍可能 unroll 它）。
- ``T.unroll(n)``：完全 unrolled（展开成 straight-line statement）。
- ``T.vectorized(n)``：vectorized loop。
- ``T.grid(*extents)``：嵌套 loop nest。

``break`` / ``continue`` 可以在 loop 内使用。

.. code-block:: python

    for i, j in T.grid(8, 8):
        B[i, j] = T.max(A[i, j], T.float32(0.0))

.. code-block:: c++

    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        B_ptr[i * 8 + j] = max(A_ptr[i * 8 + j], 0.0f);

``T.unroll(4)`` 则展开成四条 straight-line statement，没有 loop。

while
-----

``while`` loop 会运行到 condition 为 false。请使用 mutable scalar counter（见 :doc:`buffers`）：

.. code-block:: python

    i: T.int32 = 0
    while i < 64:
        A[i] = A[i] + T.float32(1.0)
        i += 1

它会 lower 成带 early-exit ``break`` 的 ``while (1)`` （counter 是一个 one-element register buffer）：

.. code-block:: c++

    int i_ptr[1];
    i_ptr[0] = 0;
    while (1) {
      if (!(i_ptr[0] < 64)) { break; }
      A_ptr[i_ptr[0]] = A_ptr[i_ptr[0]] + 1.0f;
      i_ptr[0] = i_ptr[0] + 1;
    }
