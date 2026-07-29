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

Parser 工具
===========

下面几项工具在 TVMScript 转换为 TIRx 的 **parse time** 生效。它们可以将
Python 计算得到的值直接写入 IR、提取可复用的代码片段，以及组织 parser
阶段使用的状态。

``T.meta_var``：内联 Python 值
--------------------------------

``T.meta_var(x)`` 告诉 parser，``x`` 是由 **Python** 计算得到的编译期 meta
value，应直接内联到 IR，而不是作为 script variable 解析。它既可以省去没有
实际用途的临时变量，也可以用于 metaprogramming：使用 meta value 作为范围的
普通 Python ``for`` 会在 parse time 展开。

.. code-block:: python

    n = T.meta_var(4)              # n is a Python int, inlined
    for j in range(n):            # unrolled at parse time
        acc[0] = acc[0] + A[tx, j]

``@T.inline``：内联函数
-----------------------

``@T.inline`` 定义的函数会在 parse time 内联到每个调用位置，生成的代码中
不会保留函数调用。它遵循 Python 的 lexical（LEGB）scope 和 late binding；
函数参数会覆盖外层的同名变量：

.. code-block:: python

    @T.inline
    def add_into(acc, x):
        acc[0] = acc[0] + x

    add_into(acc, A[tx, j])       # inlined -> acc[0] = acc[0] + A[tx, j]

``@T.meta_class``：组织 Parser 状态
------------------------------------

``@T.meta_class`` 修饰普通的 Python class，使它的 instances 成为 parser meta
values。对象字段可以保存 buffers 和 scalars，因此可以把相关的 allocation
和状态放进同一个对象，再在 kernel body 中使用。

.. code-block:: python

    @T.meta_class
    class State:
        def __init__(self, smem):
            self.acc = T.alloc_local([1], "float32")
            self.buf = T.decl_buffer([64], "float16", smem, scope="shared.dyn")

    s = State(smem.data)
    s.acc[0] = T.float32(0.0)     # use its fields like ordinary buffers
    # ... s.buf[i] ...

这种写法适合组织 kernel 的 pipeline state，例如 barriers、accumulators 和
临时 views，避免在函数中传递大量分散的局部变量。

``T.constexpr``
---------------

``T.constexpr`` 声明编译期 kernel parameter，其值由 ``@T.jit`` 的
``.specialize(...)`` 固定。具体用法见 :ref:`chap_tirx_primer`。
