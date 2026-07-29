(chap_appendix)=
# 参考资料

本书的主线内容位于第一至第四部分。阅读过程中需要查询具体细节时，可以使用下面的参考资料：

| 需要查询的内容 | 对应页面 |
|---|---|
| TIRx 语言特性的准确写法和语义 | **{ref}`chap_language_reference`** |
| 编译器内部机制与 lowering pipeline | **{ref}`chap_arch`** |
| 排查异步 GEMM 或 Flash Attention kernel 的卡死、崩溃、错误结果和性能下降 | **{ref}`chap_warp_spec_debug`** |

完整的 `tvm.tirx` Python API 请参阅
[TVM 官方文档](https://tvm.apache.org/docs/)。

TIRx 的基本用法见第二部分的 {ref}`chap_tirx_primer`，tensor layout 模型见
{ref}`chap_tirx_layout_api`。
