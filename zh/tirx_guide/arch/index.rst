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

编译器内部机制
==============

本节先说明 TIRx IR 怎样组织函数体、buffer、layout 和执行层级，再沿编译流水线观察这些信息怎样变成 CPU 端的启动函数和 GPU 端的 device code。

.. toctree::
   :maxdepth: 1

   ir_representation
   lowering_pipeline
   tile_primitive_layout_lowering
