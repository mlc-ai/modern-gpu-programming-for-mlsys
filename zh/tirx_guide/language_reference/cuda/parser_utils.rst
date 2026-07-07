Parser utilities
================

少数 helper 会在 **parse time** 起作用，也就是 TVMScript 被转换成 TIRx 的阶段。它们允许你 inline Python 计算出来的值，抽出可复用片段，并打包 parser-side state。

``T.meta_var`` — inline Python 值
---------------------------------

``T.meta_var(x)`` 告诉 parser 把 ``x``，一个在 **Python** 中计算出来的值，当作 compile-time *meta* value，并直接 inline 到 IR 中，而不是把它解析成 script variable。它可以避免一次性的 local，也驱动 metaprogramming：对 meta value 的普通 Python ``for`` 会在 parser 中展开。

.. code-block:: python

    n = T.meta_var(4)              # n is a Python int, inlined
    for j in range(n):            # unrolled at parse time
        acc[0] = acc[0] + A[tx, j]

``@T.inline`` — inline function
-------------------------------

``@T.inline`` 定义一个函数，其 body 会在 parsing 阶段 **inline 到每个 call site**，生成代码中不会出现调用。它遵循 Python 的 lexical（LEGB）scope，并使用 late binding，因此参数会 shadow 外层变量：

.. code-block:: python

    @T.inline
    def add_into(acc, x):
        acc[0] = acc[0] + x

    add_into(acc, A[tx, j])       # inlined -> acc[0] = acc[0] + A[tx, j]

``@T.meta_class`` — parser-side state object
--------------------------------------------

``@T.meta_class`` 标记一个普通 Python class，其 **instance 是 parser meta value**：字段可以持有 buffer 和 scalar，因此你可以把相关 allocation 和 state 打包成一个对象，并在 kernel body 中使用它。

.. code-block:: python

    @T.meta_class
    class State:
        def __init__(self, smem):
            self.acc = T.alloc_local([1], "float32")
            self.buf = T.decl_buffer([64], "float16", smem, scope="shared.dyn")

    s = State(smem.data)
    s.acc[0] = T.float32(0.0)     # use its fields like ordinary buffers
    # ... s.buf[i] ...

这很适合把 kernel 的 pipeline state（barrier、accumulator、scratch view）分组，而不是把许多独立 local 在线程中传来传去。

``T.constexpr``
---------------

``T.constexpr`` 标记 compile-time kernel parameter，它会由 ``@T.jit`` 的 ``.specialize(...)`` 固化。细节见 :ref:`chap_tirx_primer`。
