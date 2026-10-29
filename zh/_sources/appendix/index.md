(chap_appendix)=
# 概览

本书的主线内容位于第一至第四部分。附录收录了阅读过程中可能需要查询的补充内容：

| 需要查询的内容 | 对应页面 |
|---|---|
| TIRx 语言特性的准确写法和语义 | **{ref}`chap_language_reference`** |
| 可复现地测量、比较和分析 GPU kernel 性能 | **{ref}`chap_benchmarking`** |
| 编译器内部机制与 lowering 流程 | **{ref}`chap_arch`** |
| 排查异步 GEMM 或 Flash Attention kernel 的卡死、崩溃、错误结果和性能下降 | **{ref}`chap_warp_spec_debug`** |

完整的 `tvm.tirx` Python API 请参阅
[TVM 官方文档](https://tvm.apache.org/docs/)。

第二部分介绍了 TIRx 编程模型（{ref}`chap_tirx_primer`）和 tensor layout 模型
（{ref}`chap_tirx_layout_api`）。
