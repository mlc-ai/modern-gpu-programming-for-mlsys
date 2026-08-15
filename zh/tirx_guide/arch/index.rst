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

本节介绍 TIRx 编译器如何将编写好的 module 转换成 CPU 端的启动函数和 GPU 端的 device code，并沿着编译流水线说明 TIRx 高层结构、host/device 拆分以及 CUDA 代码生成之间的关系。

.. toctree::
   :maxdepth: 1

   lowering_pipeline
