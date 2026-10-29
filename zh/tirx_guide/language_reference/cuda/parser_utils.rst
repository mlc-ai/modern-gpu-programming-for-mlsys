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

``T.meta_var``：保留 Python metadata
------------------------------------

``T.meta_var(x)`` 原样返回 ``x``，可以保留 parser 侧的对象句柄或 Python
metadata tuple。它不会绕过普通赋值规则：在 TVMScript 内赋值标量表达式仍会
创建原生的值绑定。Shape 参数、指令宽度和 TMA box 大小应在
``@T.function`` 之前的 Python factory 中计算。

.. code-block:: python

    widths = T.meta_var((32, 64))  # tuple remains Python metadata
    for j in T.unroll(widths[0]):
        acc[0] = acc[0] + A[tx, j]

普通 ``range(n)`` 创建 serial TIRx loop；``T.unroll(n)`` 标记该 loop 在
lowering 时展开。``meta_var`` 不会让运行时值在输入变化后自动重新求值。
使用 persistent scheduler 时，应在每轮任务循环内，根据更新后的 cursor
重新计算 tile 坐标。

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

``State`` 定义在 ``@T.function`` 外部；创建 ``s`` 和使用其字段的语句则放在
kernel 内部，并先分配 ``smem``。

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

这种写法适合组织 kernel 的 pipeline state，例如 barriers、accumulators 和
临时 views，避免在函数中传递大量分散的局部变量。

``T.constexpr``
---------------

以 ``T.constexpr`` 标注的参数由 ``@T.jit`` 的 ``.specialize(...)``
固定。在函数体内，``if T.constexpr(condition)`` 会在解析时选择 Python
分支，适用于 FA4 中的 ``is_causal`` 等已知 specialization flag。
普通 ``if`` 会构造 IR 控制流，并解析两个分支。
