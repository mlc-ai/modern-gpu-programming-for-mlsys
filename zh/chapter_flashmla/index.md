(chap_flashmla)=
# FlashMLA

:::{admonition} 概览
:class: overview

- 从普通 MHA 的 KV cache 出发，理解 MLA 为什么只需为每个 token 保存一份共享的压缩状态，以及各 head 特有的 K/V 变换被移到了哪里。
- 理解 FlashMLA sparse attention 的输入：外部 indexer 先选出 KV rows，kernel 再对这些 rows 完成 QK、softmax 和 PV；随后用一个可执行 reference 明确其数值语义和边界条件。
- 以一个具体实现为例，理解 sparse attention 如何映射到 Blackwell，并在 B200 上完成编译与数值验证。
:::

语言模型生成文本时，通常一次只预测一个新 token。为了完成这一步，attention 会用当前位置的 query 与所有历史 tokens 的 key 和 value 计算相关性，再根据 attention weights 汇总历史信息。如果不保存历史 K/V，模型就必须在每一步重新计算整个 prefix 的 K/V，造成大量重复计算。因此，系统会把已经得到的 K/V 保存在 **KV cache** 中，供后续 attention 直接读取。生成模型通常采用 causal attention，位置 $s$ 的 K/V 只依赖位置 $s$ 及其之前的 tokens，后来追加的 token 不会改变它们。Prefill 会并行计算 prompt 中的 tokens 并填充 cache；decode 随后逐步读取已有 cache，再追加新 token 的 K/V。

普通 multi-head attention（MHA）中，每个 attention head 都有自己的 K 和 V，因此 KV cache 必须为每个已经处理过的 token 保存各个 head 的 K/V。上下文越长，cache 占用的显存越多；生成每个新 token 时，attention 还要读取不断增长的历史 K/V，使它同时面临显存容量和内存带宽的压力。**Multi-head Latent Attention（MLA）** 通过改变 cache 表示来减少这些开销：它把每个 token 中与 K/V 相关的信息压缩成一份由所有 heads 共享的状态。计算 attention 时，每个 head 仍使用自己的变换，因此能够保持各自的 attention 行为。

FlashMLA 是 DeepSeek 为 MLA 开发的高性能 GPU kernel library。本章分析其中一条运行在 Blackwell 上、用于 prefill 的 sparse-attention forward kernel。模型做 prefill，也就是并行处理 prompt 中的 tokens 时，会先由外部 indexer 为每个 query token 选出需要关注的历史位置，再由 kernel 读取这些位置对应的 KV rows，依次完成 QK、softmax 和 PV。与 dense attention 相比，它只减少了参与计算的历史位置，attention 的主流程没有改变。

在本章分析的 kernel 中，一个 query token 对应 128 个 query heads，而每个被选中的历史 token 在 KV cache 中只有一条供这些 heads 共享的 row。但 128 个 heads 读取同一条 row，为什么仍能得到不同的结果？共享的是 cache 中压缩后的状态，各 head 特有的变换仍保留在 query 和 output 路径中。要看清这些变换怎样移动，先从普通 MHA 的 KV cache 算起。

## 普通 MHA 的 KV cache 开销

普通 MHA 有 $n_h$ 个 heads，每个 head 的宽度为 $d_h$，并拥有自己的 query、key 和 value projections。下面固定其中一个 head，把它的投影矩阵记为 $W^Q$、$W^K$ 和 $W^V$。在某一层中，把送入 attention projections 的当前 token 向量记为 $h_t$，把第 $s$ 个历史 token 的输入向量记为 $h_s$，则

$$
q=W^Q h_t,\qquad
k_s=W^K h_s,\qquad
v_s=W^V h_s.
$$

这个 head 的 attention 为

$$
p_s=\operatorname{softmax}_s
\left(\frac{q^{\mathsf T}k_s}{\sqrt{d_h}}\right),
\qquad
o=\sum_s p_s v_s.
$$

自回归生成时，当前的 query 用完即可丢弃，历史 $k_s$ 和 $v_s$ 则会被后续 query 反复读取。因此，每层需要为每个 token 缓存 $2n_hd_h$ 个元素：每个 head 各一份 K 和 V。

总容量会随 batch size、context length 和层数线性增长。例如，对于一个 32 层的模型，若 batch size 为 1、context length 为 4096，共有 32 个 head 且每个 head 为 128 维，那么使用 BF16 时，KV cache 就需要约 2 GB。生成每个新 token 还要重新读取这段不断增长的历史，所以 KV cache 同时带来显存容量和内存带宽压力。

降低这项开销的一个直接办法，是让多个 query heads 共享更多状态。Multi-query attention（MQA）让所有 query heads 共享一组 K/V，grouped-query attention（GQA）则在组内共享。MLA 采用另一种共享方式：各 head 仍使用自己的 projection，cache 中却只保存一份共享的低维状态。

### 从共享状态到低维 latent cache

先看不含位置信息的 content 部分，并把当前 query 的 content 记为 $q^C$。普通 MHA 会先生成 $k_s^C=W^Kh_s$，再计算 QK 的 content score。矩阵结合律允许我们把这一步改写为

$$
\mathrm{QK}_s
=(q^{C})^{\mathsf T}k_s^C
=(q^{C})^{\mathsf T}W^Kh_s
=\left((W^K)^{\mathsf T}q^{C}\right)^{\mathsf T}h_s.
$$

PV 同样会先生成 $v_s=W^Vh_s$，再根据 attention weights $p_s$ 做加权求和。线性变换对求和的分配性给出

$$
\mathrm{PV}
=\sum_s p_s v_s
=\sum_s p_s W^Vh_s
=W^V\left(\sum_s p_s h_s\right).
$$

第一式把 key projection 移到当前 query，第二式把 value projection 移到加权求和之后。数值结果保持不变，cache 却只需长期保存一份 $h_s$。这可以看成一个尚未压缩的 latent cache 思想实验。

$h_s$ 的宽度仍然是 $d_{model}$，直接缓存它虽然实现了共享，QK 打分和 PV 加权聚合却仍要在这个较宽的空间中计算。实际的 MLA 分成两步。

第一步，所有 heads 共用矩阵 $D$，把 $h_s$ 压缩成 $d_c$ 维的 $c_s$。就 content 部分而言，KV cache 只保存这个 $c_s$：

$$
c_s=Dh_s,\qquad c_s\in\mathbb{R}^{d_c}.
$$

第二步，每个 head 用自己的矩阵 $U_K$ 和 $U_V$，从 $c_s$ 得到该 head 的 content key 和 value：

$$
k_s^C=U_Kc_s,\qquad
v_s=U_Vc_s.
$$

$D$、$U_K$ 和 $U_V$ 会在模型训练时一起学习。由于 $d_c$ 远小于 $d_{model}$，每个 token 需要缓存的 content 状态也随之缩小。

### RoPE 与独立的 positional channel

前面的 content QK 没有加入位置信息，因此移到 query 侧的变换与历史位置 $s$ 无关，一次计算后便可用于所有 cached rows。RoPE 根据 token 位置旋转 Q/K；记 $R_t$ 和 $R_s$ 分别为位置 $t$ 和 $s$ 的旋转。如果直接对 content query 和 content key 使用 RoPE，则

$$
\left(R_tq^{C}\right)^{\mathsf T}
\left(R_sU_Kc_s\right)
=
\left(U_K^{\mathsf T}R_s^{\mathsf T}R_tq^{C}\right)^{\mathsf T}c_s.
$$

当前 query 的位置 $t$ 固定，历史位置 $s$ 却会随每条 cached row 改变，因此括号中的 transformed query 也随 $s$ 改变，无法只计算一次再复用于所有历史位置。

MLA 把 content 和 position 拆成两条支路。Content 支路不做 RoPE，继续计算 $(q^C)^{\mathsf T}U_Kc_s$；positional channel 则是额外拼接在 Q/K 上的一小段坐标，不是新的 attention head。记 $q_t^R$ 和 $k_s^R$ 分别为在位置 $t$ 和 $s$ 做过 RoPE 后的 positional query 和 key。$q_t^R$ 属于当前 head，$k_s^R$ 由所有 heads 共享并缓存。

两条支路分别产生

$$
\mathrm{content}_s=(q^{C})^{\mathsf T}U_Kc_s,
\qquad
\mathrm{position}_s=(q_t^{R})^{\mathsf T}k_s^R,
$$

最终相加得到

$$
\mathrm{score}_s=\mathrm{content}_s+\mathrm{position}_s.
$$

每个历史 token 的 cache 因而是 $[c_s;k_s^R]$：$c_s$ 保存共享的 content，$k_s^R$ 保存共享的位置信息。Content 支路仍可把 $U_K$ 移到 query 侧，positional channel 则保持显式。

### 各 head 的 up-projection 与 weight absorption

前面的未压缩思想实验已经展示了怎样移动 projections。现在把同样的重排应用到 $U_K$ 和 $U_V$。MLA 的 core attention 可以用两种代数上等价的模式执行。这里的 “MQA mode” 指 MLA core attention 的一种执行方式，MLA 的模型结构保持不变。

| MLA 执行模式 | 提交给 kernel 的 core-attention K/V | Up-projection 发生的位置 |
| --- | --- | --- |
| MHA mode | 每个 head 的 $[U_Kc;k^R]$ 和 $U_Vc$ | Core attention 之前 |
| MQA mode | 共享的 $[c;k^R]$ 和 latent value $c$ | 吸收到 query 与 output 路径 |

```{figure} ../../img/flashmla_mla_modes_zh.svg
:width: 100%
:alt: MLA 的 MHA 与 MQA 执行模式，以及两条 weight-absorption 路径

MLA 的 MHA mode 在 core attention 之前展开 latent KV；MQA mode 则把 key up-projection 移到 query 路径，把 value up-projection 移到 output 路径。两种模式都显式保留共享的 RoPE key。
```

两种模式计算相同的结果，却具有不同的执行成本。MHA mode 先展开 per-head K/V，core attention 的 feature width 较小；如果许多 query rows 会复用这些展开结果，这项成本可以被摊薄。Absorbed MQA mode 让 core attention 直接处理较宽的 latent representation，同时省去为历史 rows materialize per-head K/V 的工作。最佳 mode 由 sequence stage、稀疏性、shape、数据搬运和硬件 schedule 共同决定。

Weight absorption 利用矩阵乘法的结合律改变求值分组，省去展开 K/V 的 materialization；权重和矩阵顺序都保持不变。Key 一侧可以重新结合为：

$$
(q^{C})^{\mathsf T}U_Kc_s
=\left(U_K^{\mathsf T}q^{C}\right)^{\mathsf T}c_s
=(q^A)^{\mathsf T}c_s,
\qquad q^A=U_K^{\mathsf T}q^C.
$$

吸收权重后的 query $q^A$ 可以直接与 cached latent 做 dot product。

用两个坐标算一次就能看清这件事。取 $q=(1,2)^{\mathsf T}$、$c=(3,4)^{\mathsf T}$，以及 $U=\begin{bmatrix}1&2\\0&1\end{bmatrix}$。先展开 key 得到 $q^{\mathsf T}(Uc)=19$；先变换 query 则得到 $(U^{\mathsf T}q)^{\mathsf T}c=19$。前者对每个 cached row 计算 $Uc$，后者对当前 query 只计算一次 $U^{\mathsf T}q$。

Value 一侧则利用线性关系：

$$
\sum_s p_s U_Vc_s
=U_V\left(\sum_s p_s c_s\right).
$$

因此，$U_V$ 可以与模型的 output projection 合并，attention kernel 直接使用共享 latent representation，省去展开 per-head K/V 的中间结果。

下图把普通 MHA cache、MLA shared cache 和两条 weight-absorption 路径放在一起：

```{figure} ../../img/flashmla_cache_story_zh.svg
:width: 100%
:alt: 普通 MHA 为每个 head 分别缓存 key 和 value；MLA 只保存一份共享压缩状态，并把各 head 特有的计算放在 attention 两侧

*普通 MHA 为每个 head 分别保存一份 key/value。MLA 为每个 token 只保存一份共享的压缩内容状态和位置信息；各 head 特有的 query 与 output 变换分别在 attention 前后完成。*
```

现在可以比较不同机制实际缓存的内容。下表只计算每个 token、每层保存的标量元素，不计数据类型和 allocator metadata：

| 机制 | Cache 元素数 | 缓存的内容 |
| --- | ---: | --- |
| MHA | $2n_hd_h$ | 每个 head 的完整 K 和 V |
| GQA | $2n_{kv}d_h$ | 每个 KV head 的完整 K 和 V |
| MQA | $2d_h$ | 所有 query heads 共用一组完整 K/V |
| MLA | $d_c+d_h^R$ | 所有 heads 共用 $c_s$ 和 $k_s^R$ |

这里 $n_{kv}$ 是 GQA 的 KV-head 数，$d_h^R$ 是 $k_s^R$ 的宽度。MLA 一行没有 $2d_c$，因为同一个 $c_s$ 同时提供生成 content K 和 V 所需的信息，只需缓存一次。本章使用 $d_c=512$、$d_h^R=64$，所以 $[c_s;k_s^R]$ 一共有 $512+64=576$ 个标量元素；后文 kernel 中的 `d_qk=576` 就是这条 cached row 的宽度。

本章研究的 DeepSeek Sparse Attention（DSA）sparse-prefill 路径采用这种 MQA mode：每个被选中的 latent KV entry 由所有 query heads 共享。为验证 weight absorption 保持数值结果不变，CPU 程序会同时计算两条代数路径。

### 两种执行模式的数值验证

这个 CPU 程序同时构造两种执行方式。MHA 路径显式展开 K/V，MQA 路径吸收相同的矩阵，并且两边都加入共享的 RoPE score term。使用 Float64 可以让等价性检查足够敏感，从而发现 index 转置或 contraction 写错之类的问题。

```python
import math
import torch

torch.manual_seed(0)
Q, K, H = 3, 5, 4
D_CONTENT, D_LATENT, D_VALUE, D_ROPE, D_MODEL = 7, 6, 8, 3, 11

# 每个 head 的 query、每个 key token 一份共享 latent KV，以及共享 RoPE key。
q_content = torch.randn(Q, H, D_CONTENT, dtype=torch.float64)
q_rope = torch.randn(Q, H, D_ROPE, dtype=torch.float64)
c_kv = torch.randn(K, D_LATENT, dtype=torch.float64)
k_rope = torch.randn(K, D_ROPE, dtype=torch.float64)

W_UK = torch.randn(H, D_CONTENT, D_LATENT, dtype=torch.float64)
W_UV = torch.randn(H, D_VALUE, D_LATENT, dtype=torch.float64)
W_O = torch.randn(D_MODEL, H, D_VALUE, dtype=torch.float64)

# MHA mode：为每个 head 显式展开 key 和 value。
k_content = torch.einsum("hdc,kc->khd", W_UK, c_kv)
v_content = torch.einsum("hvc,kc->khv", W_UV, c_kv)
scale = 1.0 / math.sqrt(D_CONTENT + D_ROPE)
scores_mha = (
    torch.einsum("qhd,khd->qhk", q_content, k_content)
    + torch.einsum("qhr,kr->qhk", q_rope, k_rope)
) * scale
prob = torch.softmax(scores_mha, dim=-1)
head_out_mha = torch.einsum("qhk,khv->qhv", prob, v_content)
model_out_mha = torch.einsum("mhv,qhv->qm", W_O, head_out_mha)

# MQA mode：把 W_UK 移到 Q，attention 直接读取 c_kv，再把 W_UV 移到 output。
q_absorbed = torch.einsum("qhd,hdc->qhc", q_content, W_UK)
scores_mqa = (
    torch.einsum("qhc,kc->qhk", q_absorbed, c_kv)
    + torch.einsum("qhr,kr->qhk", q_rope, k_rope)
) * scale
latent_out = torch.einsum("qhk,kc->qhc", torch.softmax(scores_mqa, -1), c_kv)
W_O_absorbed = torch.einsum("mhv,hvc->mhc", W_O, W_UV)
model_out_mqa = torch.einsum("mhc,qhc->qm", W_O_absorbed, latent_out)

torch.testing.assert_close(scores_mha, scores_mqa, rtol=1e-12, atol=1e-12)
torch.testing.assert_close(model_out_mha, model_out_mqa, rtol=1e-12, atol=1e-12)
print("weight absorption: exact up to float64 roundoff")
```

这里的 scale 由模型语义中的 QK head dimension 决定，因此 absorbed path 沿用原有 scale。吸收权重后的 dot product 虽然有 $D_{latent}$ 个 coordinates，$1/\sqrt{D_{latent}}$ 对应的却是另一套 scale 语义。

:::{admonition} Query 侧低秩分解的作用范围
:class: note

Query projection 也可以通过独立的低秩 latent 分解。记 down-projection 和 up-projection 为 $D_Q$ 和 $U_Q$，则

$$
c^Q=D_Qh,\qquad q^C=U_Qc^Q.
$$

这项分解主要降低训练时的 activation memory，KV cache 大小仍由 KV-side representation 决定。它发生在 core attention 之前：attention 实现接收到的 `q` 已经是 projection 后的 query。因此，分析 KV 路径时可以直接把 $q^C$ 作为输入。
:::

MLA 的完整结构、训练方式和原始符号可参见 [DeepSeek-V2 论文](https://arxiv.org/abs/2405.04434)。这里保留了理解后续 FlashMLA kernel 所需的 KV compression、decoupled RoPE 和 weight absorption。

Weight absorption 解释了一条共享 KV row 怎样服务多个 query heads。Sparse prefill 还涉及另一项独立边界：token selection 与 sparse core attention 的职责划分。

## Sparse-prefill operator 的 token 选择边界

Dense attention 会访问每个合法的 KV token。DSA 先用轻量的 *lightning indexer* 为候选 token 打分，并为每个 query 选出一个 top-$k$ 集合；sparse core attention 随后只读取这些 latent KV entries。若原始 context length 为 $L$，prefill 的 core-attention 计算量会从 $O(L^2)$ 变成 $O(Lk)$，不过 indexer 自身也有开销。

```{figure} ../../img/flashmla_sparse_story_zh.svg
:width: 100%
:alt: Lightning indexer 先选择 token，sparse-prefill operator 再执行 attention

Token 选择与 sparse core attention 是两个相互独立的算子。Indexer 输出 row addresses；sparse-prefill operator 根据这些地址 gather 对应的 rows，再完成 QK--softmax--PV 计算。
```

本章研究的 sparse-prefill operator 直接接收外部 indexer 生成的 `indices` tensor，并据此执行以下语义：

1. gather 指定的 KV rows；
2. 将越界位置和被 length mask 的位置标为 invalid；
3. 对剩余 rows 计算 attention；如果 caller 给出了重复 indices，重复项也会参与计算；
4. 返回 output、maximum logit 和 log-sum-exp。

Causal legality 编码在 index list 中：caller 通过只写入允许访问的 keys 来实现 causal attention。因此，稀疏选择与 causal constraint 是两项独立约束。

如果提供 `topk_length`，FlashMLA sparse-prefill contract 要求每个 query 的值都满足 `0 <= topk_length[q] <= topk`。这是 caller 应保证的前置条件。

这个 prefill 接口以不含 batch 维度的 flattened queries 为输入。每个 query token 提供一份 selected-token list，它的 `h_q` 个 query heads 共用这份 list；serving system 在调用前负责 flatten batch 或完成等价的 batch mapping。

CPU reference 会把这些规则固定为可独立执行的数值语义，为分析 FlashMLA 实现提供基准。

## 可执行的 sparse-attention reference

CPU reference 使用以下通用 contract 和 shape 符号：

| 符号 | 含义 |
| --- | --- |
| `s_q` | query rows 的数量 |
| `s_kv` | 可寻址 KV rows 的数量 |
| `h_q` | 每个 query row 的 query-head 数 |
| `h_kv` | 每个 KV row 的 KV-head 数；本接口要求为 1 |
| `d_qk` | QK 使用的 query/key 宽度 |
| `d_v` | value 与 output 的宽度，且 `d_v <= d_qk` |
| `topk` | 每个 query row 提供的 index slots 数量 |

对应的 tensors 为 `q[s_q,h_q,d_qk]`、`kv[s_kv,h_kv,d_qk]`、`indices[s_q,1,topk]` 和 `out[s_q,h_q,d_v]`。可选 sink 的 shape 是 `[h_q]`，可选 `topk_length` 的 shape 是 `[s_q]`，返回的 `max_logits` 和 `lse` 都是 `[s_q,h_q]`。Regular head-128 specialization 会进一步固定 `h_q=128` 和 `d_v=512`；`h_kv=1` 已经是通用 sparse-prefill contract 的一部分。

在吸收权重后的 MQA contract 中，`kv[:, 0, :]` 同时提供 K 和 V：全部 `d_qk` 个坐标参与 QK，前 `d_v` 个坐标则作为 latent value。可执行 reference 将据此定义“应该算什么”。

还有两个边界语义会直接影响代码。第一，**attention sink** 可以看成额外加入一个 logit，但它对应的 value vector 为 0。固定一个 query 和 head，令 $x_j$ 表示第 $j$ 个 KV row 的普通 logit，$v_j$ 表示对应的 value，$a$ 表示 sink logit，$m$ 表示普通 logits 的最大值。Sink 只进入 output 的 denominator：

$$
O=\frac{\sum_j e^{x_j-m}v_j}
{\sum_j e^{x_j-m}+e^{a-m}}.
$$

Sink 不参与返回的 `max_logits` 或 `lse`。第二，如果一个 query 的 selected rows 全部无效，reference 约定 `output=0`、`max_logits=-inf`、`lse=+inf`。显式写出这项约定，可以避免在 softmax 中计算 `(-inf)-(-inf)`。

CPU 程序按四步实现这个 contract：

1. 将 `indices` clamp 成可安全读取的地址，再 gather 对应 KV rows；
2. 合并地址边界与 `topk_length`，得到 validity mask；
3. 计算 QK、mask、softmax 与 PV，并把 sink 加入最终 denominator；
4. 返回 output、maximum logit 和不含 sink 的 log-sum-exp。

第一步为 gather 提供安全地址，第二步的 mask 才决定 row 是否参与计算。越界地址对应的 V row 还要在 PV 前清零，因为按照 IEEE arithmetic，softmax weight 为 0 时，`0 * NaN` 仍会产生 NaN。完整代码如下：

```python
import math
import torch

def sparse_prefill_reference(
    q, kv, indices, sm_scale, d_v, *, attn_sink=None, topk_length=None
):
    """Reference for q[SQ,H,D], kv[SKV,1,D], indices[SQ,1,TOPK]."""
    s_q, h_q, d_qk = q.shape
    s_kv, h_kv, kv_width = kv.shape
    assert h_kv == 1 and kv_width == d_qk and d_v <= d_qk
    assert indices.shape[:2] == (s_q, 1)

    idx = indices[:, 0].to(torch.long)                 # [SQ, TOPK]
    topk = idx.shape[1]
    assert topk > 0
    in_range = (idx >= 0) & (idx < s_kv)
    safe_idx = idx.clamp(0, s_kv - 1)
    focused_kv = kv[:, 0].float()[safe_idx]            # [SQ, TOPK, D]

    # OOB index 只把 clamp 后的边界 row 当作安全的 gather 地址。PV 前清零这些
    # V rows，避免 sentinel row 的 NaN 经 0 * NaN 泄漏。刻意保留 topk_length
    # 之后、但地址合法的 rows，以匹配 kernel 已说明的异常 NaN 行为。
    focused_v = torch.where(
        in_range[:, :, None],
        focused_kv[:, :, :d_v],
        torch.zeros_like(focused_kv[:, :, :d_v]),
    )

    position = torch.arange(topk, device=q.device)[None, :]
    if topk_length is not None:
        assert topk_length.shape == (s_q,)
        assert bool(((0 <= topk_length) & (topk_length <= topk)).all())
    length = (
        topk_length.to(torch.long)[:, None]
        if topk_length is not None
        else torch.full((s_q, 1), topk, device=q.device)
    )
    valid = in_range & (position < length)

    logits = torch.einsum("qhd,qkd->qhk", q.float(), focused_kv) * sm_scale
    logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
    max_logits = logits.amax(dim=-1)
    have_valid = valid.any(dim=-1)[:, None]

    # 避免所有 selected indices 都 invalid 时出现 (-inf)-(-inf)。
    softmax_origin = torch.where(have_valid, max_logits, torch.zeros_like(max_logits))
    weight = torch.exp(logits - softmax_origin[:, :, None])
    weight = torch.where(valid[:, None, :], weight, torch.zeros_like(weight))
    denominator = weight.sum(dim=-1)
    numerator = torch.einsum("qhk,qkv->qhv", weight, focused_v)

    if attn_sink is None:
        sink_term = torch.zeros_like(denominator)
    else:
        sink_term = torch.exp(attn_sink.float()[None, :] - softmax_origin)
    out = numerator / (denominator + sink_term).clamp_min(torch.finfo(torch.float32).tiny)[
        :, :, None
    ]
    out = torch.where(have_valid[:, :, None], out, torch.zeros_like(out))

    # FlashMLA 报告的 LSE 不包含 attention sink。全 invalid 时约定为
    # max_logits=-inf、lse=+inf、output=0。
    lse = torch.where(
        have_valid,
        max_logits + torch.log(denominator),
        torch.full_like(max_logits, torch.inf),
    )
    return out, max_logits, lse


torch.manual_seed(1)
q = torch.randn(2, 3, 6)
kv = torch.randn(9, 1, 6)
indices = torch.tensor([[[0, 3, -1, 12]], [[8, 1, 4, 2]]], dtype=torch.int32)
topk_length = torch.tensor([3, 2], dtype=torch.int32)
attn_sink = torch.randn(3)
out, max_logits, lse = sparse_prefill_reference(
    q, kv, indices, 1 / math.sqrt(6), 4,
    attn_sink=attn_sink,
    topk_length=topk_length,
)
assert out.shape == (2, 3, 4)
assert max_logits.shape == lse.shape == (2, 3)
assert torch.isfinite(out).all()

# OOB sentinel 不能继承其安全地址所指 row 中的 NaN。
nan_q = torch.ones(1, 1, 2)
nan_kv = torch.tensor([[[torch.nan, torch.nan]], [[2.0, 3.0]]])
nan_indices = torch.tensor([[[-1, 1]]], dtype=torch.int32)
nan_out, _, _ = sparse_prefill_reference(nan_q, nan_kv, nan_indices, 1.0, 2)
torch.testing.assert_close(nan_out, torch.tensor([[[2.0, 3.0]]]))
print(out.shape, max_logits.shape, lse.shape)
```

这段 reference 明确了 FlashMLA sparse-prefill operator 对合法输入和返回值的数值 contract。实现可以自由选择 tile 切分、storage reuse 和 pipeline overlap，同时需要在自身支持的 QK scale 范围内复现这套数值语义。

## Sparse prefill 在 FlashMLA 算子族中的位置

[FlashMLA 官方仓库](https://github.com/deepseek-ai/FlashMLA)按 selection 和 sequence stage 覆盖四类组合：

| Selection | Sequence stage | 代表性用途 |
| --- | --- | --- |
| dense | prefill | MHA forward 与 backward |
| dense | decode | 为新生成的 queries 读取 MLA KV cache |
| token-sparse | prefill | 对 selected-token list 执行 DSA core attention |
| token-sparse | decode | 对 selected FP8 KV cache 执行 DSA inference |

FlashMLA 是由表中四类算子组成的 library。本章先聚焦其中的 sparse-prefill operator，再深入它的 regular head-128 specialization；这条路径把算子语义直接连接到核心实现任务——将不规则 row addresses 整理成规则的 tensor-core tiles。

FlashMLA sparse-prefill public call 在概念上是：

```text
out, max_logits, lse = flash_mla_sparse_fwd(
    q, kv, indices, sm_scale,
    d_v=512,
    attn_sink=attn_sink,       # optional [h_q], float32
    topk_length=topk_length,   # optional [s_q], int32
)
```

这些参数对应前面 reference 明确的语义：`h_q` 是 query-head 数，`s_q` 是 query-row 数，`sm_scale` 用来缩放 QK scores，`d_v` 决定 value 和 output 的宽度，`attn_sink` 可以为每个 query head 加入一个 value 为 0 的额外 logit，`topk_length` 则限定每个 query 中有效 indices 的前缀长度。这个调用返回归一化后的 output、经过 scale 的最大 logit，以及不包含 sink 的 log-sum-exp。

这段 public call 定义 FlashMLA sparse-prefill API，前面的 reference 将相关数值语义写成可执行形式。TIRx 是建立在 TVM 0.26 TIR 之上的 Python DSL 扩展；其中同名的 `flash_mla_sparse_fwd` 负责 registry 和 shape dispatch，按输入 shape 选择三个 SM100 phase-1 specializations 之一。它覆盖的是 public API 的 dispatch 子集。

Caller 负责满足这个入口的两项前置条件。第一，每个 `topk_length[q]` 都必须位于 `[0, topk]`；TIRx prefill specializations 直接使用该值，不做 clip 或逐项验证，因此大于 `topk` 的值会使实现读取到 `indices` storage 之外。第二，只因 `topk_length` 被 mask、地址本身仍合法的 row 会保留原数据，其中的 NaN 也可能进入优化路径。前面的 reference 复现了这项行为；越界地址则按 reference 的规则安全处理。

:::{admonition} TIRx 入口中的 `sm_scale`
:class: warning

FlashMLA public call 在 runtime 接收 `sm_scale`；TIRx 入口则将三个 prefill specializations 的 `sm_scale` specialize 为 `1 / sqrt(d_qk)`，launch ABI 不含 scale argument。通过 `**kwargs` wrappers 传入的 `sm_scale=...` 会被静默忽略。因此，本章的 B200 示例覆盖 TIRx 入口所支持的 QK scale。需要其他 semantic QK scale 时，应将正确数值暴露为参数，或在编译时将其 specialize。

Weight absorption 保持该 semantic QK scale 不变。
:::

这三层边界将本章的实现目标限定为 TIRx dispatch 选中的一条具体 specialization。

## Blackwell 上的 regular head-128 案例

Regular head-128 案例自然衔接前面的 FlashAttention 章节：它沿用 QK--softmax--PV 主链，再加入 irregular gather、吸收权重后的 latent KV 和两个线程块之间的协作分工。

### Regular head-128 的 shape 与 dispatch 条件

Regular head-128 specialization 将共享 latent KV 的算法描述具体化为以下输入输出 shapes：

| Tensor | Shape | Type | 含义 |
| --- | --- | --- | --- |
| `q` | `[s_q, 128, d_qk]` | BF16 | 吸收权重后的 queries |
| `kv` | `[s_kv, 1, d_qk]` | BF16 | 共享 latent/positional KV rows |
| `indices` | `[s_q, 1, topk]` | int32 | 直接 KV row indices |
| `attn_sink` | `[128]` | FP32 | 可选的 per-head sink logits |
| `topk_length` | `[s_q]` | int32 | 可选的有效 prefix length；每项位于 `[0, topk]` |
| `out` | `[s_q, 128, 512]` | BF16 | sparse-attention 结果 |
| `max_logits` | `[s_q, 128]` | FP32 | 最大 scaled logit |
| `lse` | `[s_q, 128]` | FP32 | 不含 sink 的 natural-log sum-exp |

这里的 128 正是开篇问题中的 query-head 数，而 `kv` 中的 1 表示所有 heads 共享同一条 KV row。常见的 $d_{qk}=576$ 由 512 个 latent-content coordinates 和 64 个 RoPE coordinates 组成，`d_v=512` 则对应 latent value 宽度。后续 shape 讨论专指本章的 absorbed MQA specialization。

给前面定性的 mode 成本一个数值锚点：对同一 MLA layer，等价的 MHA representation 具有 $128+64=192$ 的 QK feature width 和 128 的 value/output feature width；这里的 absorbed MQA representation 则为 $512+64=576$ 和 512。若只粗略统计每个 query--key pair 在 QK 点积与 value 累加中涉及的乘加坐标数，两者是 $192+128=320$ 对 $576+512=1088$，后者约为前者的 3.4 倍。这一坐标数只作为算术宽度锚点；实际 runtime 还取决于数据移动、复用和 schedule。[DeepSeek-V3.2 report](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf) Appendix A 也说明，同一模型会根据阶段和算法采用不同 mode：DeepSeek-V3.1-Terminus 在 training 和 prefill 时使用 MHA mode，在 decode 时使用 MQA mode；DSA sparse prefill 则使用 MQA mode。

本节使用一组具体 shape：`s_q=1`、`s_kv=8192`、`h_q=128`、`h_kv=1`、`d_qk=576`、`d_v=512`、`topk=2048`。它表示一个 query row、128 个 query heads，以及由这些 heads 共享的 2048 个 selected-index slots。这些 slots 可能包含重复或越界地址。没有更短的 `topk_length` 时，物理调度会按 128 个 slots 一组访问 $N=16$ 个 tiles；若给出 `topk_length`，实际访问的 tile 数是 `max(ceil(topk_length / 128), 1)`。

每个 selected-index tile 在 kernel 中经历六步算术过程。为与源码对应，$L$ 表示原始 QK logits，$W$ 表示 BF16 未归一化指数权重，`mi` 是 online-softmax 的指数参考值，`li` 是相对于该参考值累积的 denominator，$\widetilde O$ 是尚未除以 denominator 的累计 output：

```text
对每个包含 128 个 selected-index slots 的 tile：
    1. gather 该 tile 的 K rows
    2. gather 该 tile 的 V rows，并构造 validity mask
    3. QK：128 个 query heads × 128 个 selected-index slots → logits L
    4. 更新 mask 后的 online softmax，得到 W、参考值 mi 和分母 li
    5. 必要时 rescale running state，再累加 O~ += W @ V
所有 selected-index tiles 完成后：
    6. 用 li + sink 归一化 O~，写回 out、max_logits 和 lse
```

在这组具体 shape 中，前五步重复 16 轮，最后执行第六步。TIRx 实现为这六步分配不同的硬件角色。

### 完整源码导航

正文不会复制整份 device function，而是按照 QK、softmax、PV、数据搬运和同步关系展示短摘录。完整源码可以按下面的顺序阅读：

| 阅读目标 | 源码入口 |
| --- | --- |
| 统一入口与 shape dispatch | [`flash_mla_sparse_fwd.py` lines 66--125](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/flash_mla_sparse_fwd.py#L66-L125) |
| Config、测试数据、PyTorch reference 与 launch ABI | [`sparse_prefill_head128_phase1.py` lines 66--244](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L66-L244) |
| 完整 regular head-128 device kernel | [`sparse_prefill_head128_phase1.py` lines 247--865](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L247-L865) |
| CTA-pair TMA、tcgen05 MMA 与 validity mask helpers | [`_tma.py` lines 10--60](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/_tma.py#L10-L60)、[`_gemm.py` lines 8--29](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/_gemm.py#L8-L29)、[`_mask.py` lines 10--29](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/_mask.py#L10-L29) |
| Specialize、编译、运行与数值检查 | [`sparse_prefill_head128_phase1.py` lines 868--905](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L868-L905) |

推荐先看 dispatch 和 tensor ABI，再沿 `_kernel` 中的 WG0、WG1、WG2、WG3 分支阅读；遇到 TMA、MMA 或 mask 调用时再跳到相应 helper，最后用 `run_test` 对照输入、输出和 reference。正文中的代码块保留源码变量名和切片，并在相关段落链接到完整上下文。

TIRx regular head-128 实现位于 [`sparse_prefill_head128_phase1.py`](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py)。代码里的 `T` 是 TIR script namespace，`Tx` 是 GPU kernel helpers。`phase1` 沿用了对应 CUDA 实现的文件命名；在这条 regular prefill 路径中，它会由一个 kernel 直接生成完整的 `(out, max_logits, lse)`。

实现使用三个执行层级。一个 **CTA**（cooperative thread array）就是一个 CUDA thread block（线程块）；相邻两个 CTA 组成一个 **cluster**，可以共同发起 CTA-group tensor-core operation。一个 **warpgroup** 由 4 个 warps、共 128 个 threads 组成，并在这条 kernel 中承担一个专门角色。这里的一个 cluster 负责一个 query row。

在数学推导中，$p$ 表示归一化后的 softmax probability；而在源码里，`tmem_p` 和 register 变量 `p` 保存的是前面记作 $L$ 的原始 QK logits。源码中的 `s_frag` 和 `s_smem_gemm` 保存前面记作 $W$ 的未归一化指数权重。只有在 epilogue 中用 `li`（以及可选的 sink term）除 accumulated output 后，最终 output 才完成归一化。

较短的 TIRx 代码块用于展示 regular head-128 kernel 的局部上下文；可独立执行的 blocks 会显式说明，完整编译与数值验证集中在验证一节。

以下常量给出一次 tile 的尺寸、线程数和同步槽位：

```python
B_H = 128
B_TOPK = 128
D_V = 512
NUM_BUFS = 2
NUM_THREADS = 512
D_TQ = 384
```

这些名字分别对应 kernel 的主要结构：`B_H=128` 是每个 logical tile 的 query-head 数，`B_TOPK=128` 是每次处理的 selected-index slots 数，`D_V=512` 是 output feature 宽度，`NUM_THREADS=512` 表示每个 CTA 有四个 warpgroups，`D_TQ=384` 是移入专用片上存储的 Q suffix 宽度。`NUM_BUFS=2` 为同步状态和 packed-validity mask 提供两个槽位。

这个 specialization 接受 512 或 576 的 `d_qk`，要求 `h_kv=1`、`d_v=512`，并要求 regular path 的 `topk` 是 128 的正整数倍。对于 128 heads，统一 front door 还会多做一次选择：`d_qk=512` 且 `topk<=1280` 时进入 small-top-k specialization；其他支持的 head-128 shapes 进入本章的 regular specialization。Head-64 shapes 使用 head-64 specialization。

`topk > 0` 是调用者必须满足的前置条件。统一 front door 会拒绝 `topk<=0`；各 specialization 的 `_cfg().validate()` 只检查整除性，因此直接 import 某个 specialization 时，caller 还需单独验证 `topk` 为正数。

Dispatch 可以在 launch GPU kernel 之前独立检查。这个代码块可直接执行，但需要按验证一节安装 `tirx-kernels`：

```python
from tirx_kernels.flashmla.flash_mla_sparse_fwd import (
    dispatch_reason,
    select_kernel,
)

shape = dict(h_q=128, h_kv=1, d_qk=576, d_v=512, topk=2048)
assert select_kernel(**shape) == "sparse_flashmla_prefill_head128_phase1"
print(dispatch_reason(**shape))
# sm100 h_q=128 dispatches to regular head128 phase1
```

Dispatch 记录在 [`flash_mla_sparse_fwd.py`](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/flash_mla_sparse_fwd.py#L66-L120) 中；它负责选择 specialization，device schedule 则定义被选实现的执行细节。

### 两个 CTA 的 query-row 分工

难点出现在第 3、5 步的切分轴不同：QK 要形成所有 head 与 selected token 的两两组合，PV 又要沿 token 维归约，产生 512 个 value coordinates。Regular head-128 实现让两个 CTA 通过 `cta_group=2` 共同形成这块 logical tile，并在 QK 与 PV 之间旋转切分轴。

这个关系先用所有权图（ownership map）表示：

```{figure} ../../img/flashmla_cta_ownership_zh.svg
:width: 100%
:alt: 两个 CTA 对 query heads、selected K rows 和 V feature columns 的 ownership

对于每个 query row，CTA pair 会在 QK 与 PV 之间改变 logical partition。每个 CTA 最终写出 64 个完整的 output heads，每个 head 含 512 个 coordinates。
```

这个 CTA pair 会用三种不同方式切分三个轴：

| 一个 128-token top-k tile 中的资源 | CTA 0 | CTA 1 |
| --- | --- | --- |
| Query/output head ownership | heads 0--63 | heads 64--127 |
| K-row gather ownership | selected tokens 0--63 | selected tokens 64--127 |
| V-feature gather ownership | value columns 0--255 | value columns 256--511 |

这个 2-CTA tensor-core operation 共同覆盖一块完整的 logical tile：QK 形成 128 个 heads 与 128 个 selected tokens 的两两组合，PV 再沿 token 维归约，得到 512 个 value coordinates。Collective `cta_group=2` MMA、配对的片上布局和跨 CTA 同步负责协调这两种切分。

源码中的 launch topology 直接实现了这张 ownership map。Launch grid 包含 `2 * s_q` 个 CTA，并将相邻 CTA 两两组成 cluster：

```python
block_idx = T.cta_id([2 * s_q])
T.cta_id_in_cluster([2])
cta_idx: T.let = block_idx % 2
s_q_idx: T.let = block_idx // 2
thread_idx = T.thread_id([512])
T.warpgroup_id([4])
```

因此，一个 cluster 负责一个 query row，每个 CTA 含 4 个 warpgroups。这个划分还可以从后面的数据索引直接看出：Q 按 `cta_idx` chunk，K producer 选择每个 top-k block 的 `cta_idx` 半块，V producer 则从 `cta_idx * 256` 开始。

## Tile 的数据驻留与生命周期

下面的数据驻留图使用三种硬件存储：**global memory（GMEM）** 保存 kernel 的输入输出；**shared memory（SMEM）** 是 CTA 内线程共同访问的片上存储；**tensor memory（TMEM）** 是 Blackwell tensor cores 附近用于 operands 与 accumulators 的专用片上存储。

```{figure} ../../img/flashmla_dataflow_zh.svg
:width: 100%
:alt: QK、softmax、PV 与 epilogue 期间，global memory、shared memory、tensor memory 和 WG0 registers 中的数据驻留与生命周期复用

Q 被拆成 SMEM prefix 和 TMEM suffix。Gather 后的 K/V 进入 SMEM；原始 QK logits 与 output 在 TMEM 中累积；未归一化 softmax weights 再经过 SMEM 交给 PV。
```

图中的 registers 归当前线程所有。**Tensor Memory Accelerator（TMA）** 负责在 GMEM 与 SMEM 之间异步搬运规则 tile，也能根据地址列表执行 gather；`tcgen05` tensor-core operation 则从 SMEM/TMEM 读取 operands，并把大型 accumulators 留在 TMEM。

QK 的 operand 来源用两个简写表示：**SS** 表示 Q、K 都从 SMEM 读取；**TS** 表示 Q 从 TMEM 读取、K 仍从 SMEM 读取。这条 kernel 将 Q 的 384-column suffix 搬到 TMEM，只把 prefix 留在 SMEM，所以 QK 要先做 SS prefix，再做 TS suffix。Softmax 产生的 BF16 未归一化权重则要写入 SMEM，供 PV GEMM 使用。

对一个 CTA 而言，重要的 logical views 如下：

| Storage | Logical tile | Lifetime 与用途 |
| --- | --- | --- |
| SMEM `q_full` | `64 x d_qk` BF16 | Q prologue；其 prefix 留给 SS QK |
| TMEM `q_tmem` | `64 x 384` BF16 | Q 的 suffix，供 TS QK 使用 |
| SMEM `k_smem` | `64 x d_qk` BF16 | 128-row K tile 中由本 CTA gather 的一半 |
| TMEM `tmem_p` | `64 x 128` FP32 logical view | 交给 softmax 的原始 QK logits $L$ |
| SMEM `s_smem_gemm` | `64 x 128` BF16 | 交给 PV 的未归一化指数权重 $W$ |
| SMEM `v_smem_gemm` | `128 x 256` BF16 logical view | `v_smem` 的 rearranged view：所有 tile rows、本 CTA 的 V columns |
| TMEM `o_tmem` | `64 x 512` FP32 logical view | running unnormalized output |
| SMEM `o_smem` | `64 x 512` BF16 | TMA store 前的 epilogue staging |

这些都是逻辑视图（logical views）；CTA-group TMEM layout 和 rearrangement 决定 MMA 与 load/store instructions 的实际 lane mapping。源码中的 `SMEMPool` 是共享内存分配器，用来从同一片动态 SMEM 中切出带对齐和生命周期约束的 views。源码还另行分配一个 512-column CTA-group TMEM pool，再从中切出 O、raw-logit 和 Q views。

SMEM 采用激进的 alias 与复用策略。只要 lifetime 允许，`q_full`、gather 后的 K/V region 和 output epilogue 就会复用类似 union 的 base。最后 384 个 Q columns 移到 TMEM 后，只有 $d_{sq}=d_{qk}-384$ 的 prefix 仍需保持 live，供 QK 第一部分使用。`d_qk=512` 时 $d_{sq}=128$；`d_qk=576` 时则为 192。具体 allocation plan 见 [`sparse_prefill_head128_phase1.py` lines 302--365](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L302-L365)。

这些存储区域能否安全地原位复用，取决于 **completion barrier**。这个小型硬件状态对象记录异步 producer 何时完成；phase bit 区分同一 barrier slot 的前后两次使用，使释放后的 K、V、$L$ 和 $W$ segments 能够安全地被原位覆盖。

## Warpgroup 的角色分工

数据放置确定后，还需要为每块 tile 指定 producer、consumer 和 storage 归还者。四个 warpgroups 由此组成 role-specialized pipeline：

| Warpgroup | Warps | 职责 |
| --- | --- | --- |
| WG0 | 0--3 | 从 TMEM 加载原始 logits $L$，mask、online softmax、写权重 $W$、rescale O、执行 epilogue |
| WG1 | 4--7 | 加载 index fragments，并为 K 发起 gather4 TMA |
| WG2 | 8--11 | 加载 index fragments，并为 V 发起 gather4 TMA |
| WG3 | 12--15 | CTA 0 的 warp 12 发起 CTA-group QK/PV MMA；每个 CTA 的 warp 13 构造 validity mask |

WG3 的有效工作集中在 warp 12 的 MMA issue 与 warp 13 的 validity mask。这种不对称分工与各项操作所需的并行度相匹配：一个 elected lane 为 CTA pair 发起 MMA，warp 13 负责 validity packing，WG0 则使用较多 lanes 完成 exponentiation、row reduction 和 epilogue conversion。

:::{admonition} Register budget 也跟着角色分配
:class: note

WG0 将上限提高到 144 registers，WG3 提高到 168；producer groups 则降到 96。TIRx API 用 `T.ptx.setmaxnreg(True, ...)` 表示提高，用 `T.ptx.setmaxnreg(False, ...)` 表示降低。这组配额为各 warpgroups 的角色分工提供相应的 register budget。
:::

### 从不规则 rows 到规则 tiles

稀疏的 row addresses 破坏了 dense attention 所用的 contiguous 2-D copy pattern。WG1 和 WG2 使用显式 TMA `gather4`：一次 issue 提供恰好 4 个 row coordinates，让一个 warp 可以把不连续的 KV rows 搬进规则的 SMEM tile。共享 helper 固定了 CTA-pair policy。

Gather 使用最基本的 barrier producer--consumer handshake：producer 完成数据写入后，通过 ready/completion barrier 通知 consumer；consumer 等待后读取数据，使用完这段 storage 后再通过 done/free barrier 把复用权还给 producer。Phase ring 将这套 handshake 扩展到连续的循环迭代。

这段上下文摘录展示一次 `gather4` issue 读取的 addresses，以及接收其 completion 的 barrier。三个关键名字是：`cur_buf` 表示当前 tile 在两槽 barrier ring 中使用的槽位；`bar` 是 producer 与 consumer 共享的 completion barrier；`leader_mbar(...)` 取得 CTA pair 中负责汇总 TMA completion 的 leader 地址。Index names 和 slices 与链接源码一致：

```python
_kv_gather_tma = partial(
    tma_config,
    dispatch="tma_explicit",
    cta_group=2,
    cta_mask=T.uint16(1),
    cache_hint=T.uint64(0x14F0000000000000),
)

for row_group in T.unroll(WG1_ROWS_PER_WARP):
    for col_atom in T.unroll(col_count):
        col = T.meta_var((col_start + col_atom) * 64)
        Tx.copy_async(
            k_gather_tile[
                row_group * 4 : row_group * 4 + 4,
                col_atom * 64 : col_atom * 64 + 64,
            ],
            kv_tma[0:1, col : col + 64],
            **_kv_gather_tma(
                mbar=leader_mbar(bar.ptr_to([cur_buf])),
                gather4=[indices_int4[row_group, lane] for lane in range(4)],
            ),
        )
```

Gather 负责搬运 rows，validity mask 负责决定这些 rows 是否参与计算。Warp 13 的每个 active lane 加载 8 个 indices，并调用 `pack_valid_mask8`。以下两个条件同时满足时，bit $i$ 才为 1：

$$
0\leq\text{index}_i<s_{kv}
\quad\text{and}\quad
\text{absolute_topk_position}_i<\text{topk_length}.
$$

WG0 等待 packed mask，再把对应的原始 logit $L$ 替换成 negative infinity，之后才求 maximum 或 exponential。Packed mask 决定每个数值有限的 KV row 是否参与 attention；gather 只负责按地址搬运数据。Mask 必须发生在 online-softmax state 更新之前。

源码清楚分开了这些 roles：[K producer 位于 lines 608--676](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L608-L676)， [V producer 位于 lines 679--729](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L679-L729)， [validity packing 位于 lines 841--865](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L841-L865)。

经过 gather 与 mask，不规则的 addresses 已经变成规则的 K/V SMEM tiles。QK 为什么还要把一次 dot product 拆成 SMEM--SMEM 和 TMEM--SMEM 两部分？

## QK 的 SMEM--SMEM 与 TMEM--SMEM 分解

前面的驻留图已经定义了 SS 与 TS。拆分让 Q 的大块 suffix 尽早离开 SMEM，从而给 K/V 和 epilogue 让出可复用空间，同时让 tensor core 完成完整 dot product。QK 在 $d_{sq}=d_{qk}-384$ 处分成两部分；源码摘录展示这两个 partial products 如何写入同一个 accumulator：

```python
if d_sq > 0:
    sq_smem = q_full.sub[:, :d_sq]
    Tx.gemm_async(
        tmem_p[:, :],
        sq_smem[:, :d_sq],
        k_smem[:, :d_sq],
        **_mma_config(accum=mma_p_accumulate, smem_desc=mma_smem_desc),
    )
    mma_p_accumulate = T.uint32(1)

Tx.gemm_async(
    tmem_p[:, :],
    q_tmem[:, :D_TQ],
    k_smem[:, d_sq : d_sq + D_TQ],
    **_mma_config(accum=mma_p_accumulate, smem_desc=mma_smem_desc),
)
```

第一步是 SS：Q 和 K operands 都由 SMEM 描述。第二步是 TS：Q 的 384-column suffix 来自 TMEM，K 仍在 SMEM。两步写入同一个 FP32 raw-logit accumulator（源码中的 `tmem_p`）；第一步清零，第二步累加。这样拆分 Q 后，SMEM 中只需保留较小的 Q prefix，使 union allocation 成为可能，较大的 suffix 则继续走 TS path。

Softmax 之后，PV 是 SS GEMM：BF16 $W$ 和 V 都在 SMEM，FP32 O accumulator 则留在 TMEM。Kernel 将 V rows 和 output columns 各分成两半，四种组合共同更新全部 512 个 value coordinates。

## Online softmax 中的按需 O 重缩放

当 `topk` 大于 `B_TOPK=128` 时，一个 query row 会连续处理多个 selected-token tiles。每个 tile 都只看到一部分 scores，不能各自独立做完整 softmax；kernel 必须把前面 tiles 的状态带到下一轮。前面的具体例子在没有更短 `topk_length` 时有 $N=16$ 个 tiles，因此会递推合并 16 轮状态。

为了直接使用硬件 `exp2`，先把当前 tile 中的每个原始 QK dot product $x_j$ 转换到以 2 为底的指数单位：

$$
r_j=x_j\cdot\text{semantic\_QK\_scale}\cdot\log_2(e).
$$

对于连续到来的 score tiles，online softmax 会保存指数参考值 $m$、denominator $\ell$ 和未归一化 output $\widetilde O$。合并下一块 tile 时：

$$
m'=\max(m,\max_j r_j),\qquad
\alpha=2^{m-m'},
$$

$$
\ell'=\alpha\ell+\sum_j2^{r_j-m'},\qquad
\widetilde O'=\alpha\widetilde O+\sum_j2^{r_j-m'}v_j.
$$

在 TIRx regular head-128 specialization 中，编译期 `sm_scale_div_log2` 就是 `(1 / sqrt(d_qk)) * log2(e)`。这与原来的 softmax 相同，只是改写成直接映射到快速 base-2 exponential instructions 的形式。

对应到源码，`cur_pi_max` 是当前 128-token tile 的 base-2 最大值，`mi` 是递推中实际采用的指数参考值，`li` 是相对于 `mi` 累积的 denominator。`real_mi` 则单独保存迄今为止真实的最大值，用于最终报告 `max_logits`。区分 `mi` 与 `real_mi`，使 lazy rescaling 能够保持输出统计量不变。

每次 row maximum 增加时都重缩放完整的 512-coordinate O tile，代价会很高。Head-128 kernel 因此使用 lazy threshold：

```python
should_scale_o: T.bool = (
    T.ptx.any_sync(T.uint32(0xFFFFFFFF), cur_pi_max - mi > 6.0) != 0
)

if not should_scale_o:
    scale_for_old = 1.0
    new_max = mi
else:
    new_max = T.max(cur_pi_max, mi)
    scale_for_old = T.ptx.exp2(mi - new_max)
```

如果当前 tile 的最大值比保存的指数参考值至多高 6 个 base-2 units，kernel 会继续保留旧参考值。新的 exponential 此时最大可能达到 $2^6=64$，已累积的 O 保持原 scale。一旦差值超过 6，kernel 就更新参考值，同时 rescale $\ell$ 和已经存在的 O。Warp-wide `any_sync` 让参与计算的 rows 使用一致的决策。

`real_mi` 始终维护迄今为止真实的最大值，所以 `max_logits` 保持原有语义。结束时，两个 64-token half 对每个 logical row 的贡献会被合并，kernel 输出：

$$
\mathrm{lse}=m\ln 2+\ln\ell.
$$

可选的 attention sink 将最终 output scale 改成：

```python
output_scale: T.float32 = T.cuda.fdividef(
    T.float32(1.0), li + T.ptx.exp2(attn_sink_log2 - mi)
)
```

这项 sink 修正只作用于最终 output scale，报告的 LSE 保持不变。对于全 invalid row，特殊分支会输出 0，同时令 `max_logits=-inf`、`lse=+inf`，与前面的 reference 一致。

## Pipeline 各阶段的安全重叠

这个 schedule 的主线是数据所有权在各角色之间的交接。每个 tile 依次经历 `K ready → QK done → L consumed → W ready → PV done → V/O reusable`：QK 完成后 softmax 才能消费 logits，softmax 产生 weights 后 PV 才能开始。这些依赖仍然留下了重叠空间；K 或 V segment 一旦释放，负责 gather 的 warpgroups 就可以继续推进。

源码用 **memory barrier（mbarrier）** 表示这些交接。Producer 在数据或异步操作完成时贡献 arrival，consumer 等待相应 phase；consumer 用完后，再通过 done/free barrier 把覆盖这段 storage 的权利交还 producer。Ready barrier 授予读取权，done/free barrier 授予下一轮覆盖权。

```{figure} ../../img/flashmla_pipeline_stages_zh.svg
:width: 100%
:alt: 相差一个 tile 的 pipeline 填充、稳态与排空，以及 QK、softmax、PV 的重叠

填充阶段发起 QK(0)；排空阶段发起 PV($N-1$)，随后进入最终 epilogue。稳态中，softmax($k-1$) 可与 QK($k$) 重叠；唯一的 MMA issuer 串行发出 QK($k$) 与 PV($k-1$)；QK($k$) 完成后，softmax($k$) 可与仍在异步执行的 PV($k-1$) 重叠。Tensor-core issue stream 因而只有一条，重叠来自异步执行和其他 warpgroups 的并行工作；前面的具体例子中 $N=16$。
```

这张宏观时间线由初始化和一组具体 barrier edges 实现。Kernel 先由 warp 0 初始化 mbarriers，执行 cluster sync，launch Q prologue，分配 CTA-group TMEM，再进入 specialized loops。

CTA-group gather 的 TMA completion 会被路由到指定的 leader barrier，使 CTA pair 发出的操作共同满足同一 expected byte count。

主要 barrier edges 对应以下 storage ownership transfers：

| Barrier | Producer 到 consumer | 保护的 storage |
| --- | --- | --- |
| `bar_k_part0_ready` | WG1 到 WG3 | SS QK 使用的 K prefix |
| `bar_qk_part_done` | WG3 到 WG1 | SS QK 完成后允许覆盖 K prefix |
| `bar_k_part1_ready` | WG1 到 WG3 | TS QK 使用的 K suffix |
| `bar_qk_done` | WG3 到 WG0 和 WG1 | 原始 logits $L$ ready；QK 完成后 K suffix 可复用 |
| `bar_p_free` | WG0 到 WG3 | 下一次覆盖前，TMEM raw-logit tile 已被消费 |
| `bar_k_valid_ready/free` | warp 13 到/从 WG0 | packed validity mask |
| `bar_so_ready` | WG0 到 WG3 | PV 所需的 BF16 权重 $W$ ready |
| `bar_v_part0_ready` / `bar_sv_part_done` | WG2 到/从 WG3 | 第一半 V |
| `bar_v_part1_ready` / `bar_sv_done` | WG2 到 WG3，再由 WG3 到 WG2 和 WG0 | 第二半 V；复用 V、rescale O 或执行 epilogue 前，PV/O 已完成 |

Barrier 槽位按 tile 编号循环使用：

```python
cur_buf = k % 2
cur_phase = (k // 2) & 1
```

这样，复用的 barrier slot 可以区分本轮到达与两轮之前的旧到达。

`bar_qk_part_done` 允许 producer 在 K suffix 可以复用之前先替换 K prefix。两条 `bar_sv_*` edge 对 V 做同样的事。

```{figure} ../../img/flashmla_pipeline_zh.svg
:width: 100%
:alt: Sparse-prefill 详细 pipeline，展示 QK 与 PV 的串行发起、K/V 分段复用、mask-slot ring 和 WG0 交接

在稳态中，QK($k$) 与 PV($k-1$) 交错发起，其他 roles 同时执行分段 K/V gather 和 softmax。Barrier phase 保护单份 tile 的 in-place storage reuse。
```

同一片 SMEM 在不同 memory proxy 之间交接时，正确可见性同时依赖 completion signaling 和 proxy ordering。线程执行的普通 SMEM load/store 属于 **generic proxy**，TMA 与 tcgen05 的异步访问属于 **async proxy**。

因此，`T.ptx.tcgen05.fence.*` 约束 TMEM access 与线程可见操作的先后关系；`T.ptx.fence.proxy_async("shared::cta")` 则在 SMEM 的 generic 与 async proxy 之间建立顺序。它既用于普通 store 写入 $W$ 或 epilogue tile 后、tcgen05/TMA 执行异步读取之前，也用于异步 SMEM read 完成后、普通代码覆写共用 storage 之前。

Mbarrier 传达 completion 并移交 storage 使用权，proxy fence 约束不同 proxy 的 memory effects；两者共同完成一次安全交接。

## Regular head-128 的编译与数值验证

完整验证覆盖三层：regular head-128 实现能够用 TVM 0.26 编译，生成的 kernel 能够在 B200 上实际 launch，并且它的 output、maximum logits 和 LSE 与前面的 reference 一致。

这个 specialization 面向 compute capability 10，其 TMA/tcgen05 形式要求 SM100 class GPU。环境应使用 B200、CUDA 12.9 或更高版本，以及官方 Apache TVM 0.26.0 package。

首先通过 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/)安装支持 B200 的 CUDA-enabled PyTorch build。`tirx-kernels` 会 import PyTorch，因此需要单独安装这个未声明的 package dependency。

随后安装 TVM 和 kernel repository，再运行示例：

```bash
python -m pip install "apache-tvm==0.26.0" cuda-bindings
git clone https://github.com/mlc-ai/tirx-kernels.git
cd tirx-kernels
git checkout 5be39749e7dfd2c4bdae9b4d396f8ec35af07126
pip install -e .
```

只用一个 query row 的 smoke test 仍会覆盖完整的 128-head、top-k-2048 kernel，同时保持 reference 时间较短：

```python
from tirx_kernels.flashmla.sparse_prefill_head128_phase1 import run_test

run_test(
    label="tutorial_smoke",
    s_q=1,
    s_kv=8192,
    topk=2048,
    d_qk=576,
    h_q=128,
    h_kv=1,
    d_v=512,
    have_attn_sink=True,
    have_topk_length=False,
    seed=0,
)
print("compile, launch, and randomized reference check passed")
```

`run_test` 同时覆盖这三层验证：它会分配随机 BF16 Q/KV 和随机 indices，编译并运行生成的 kernel，逐个 query row 求 FP32 PyTorch oracle，再用明确的 tolerance 检查 output、maximum logits 和 LSE。因此，它可以作为默认的 end-to-end gate，覆盖单纯 PTX 编译无法发现的 head partition 错误或遗漏 validity bit。

`tirx-kernels` CLI 可以运行完整的 registered configuration：

```bash
python -m tirx_kernels.test \
  --kernel sparse_flashmla_prefill_head128_phase1 \
  --config bench_regular_dqk576_hq128_s4096_kv8192_topk2048
```

Negative tests 同样重要。设置 `inject_invalid_indices=True` 可以覆盖负数和过大的 row ID，设置 `have_topk_length=True` 可以覆盖 position predicate；全 invalid row 则用于确认约定的 0/-infinity/+infinity 行为。对 head-64 和 small-top-k shapes 调用统一 dispatch entry，可以把覆盖范围扩展到其他 prefill specializations。完整 FlashMLA API 还包含 runtime `sm_scale` 等 TIRx 入口未覆盖的 call semantics，需要单独验证。

这些正向与异常输入测试分别验证 operator semantics 与 specialization schedule contract，二者可以归纳为五条不变量。

## Operator 与 specialization 的不变量

1. **缓存不变量（cache invariant）。** 一条 `h_kv=1` 的 latent KV row 可以服务全部 `h_q` 个 query heads，是因为 key up-projection 已吸收到各 head 的 query 路径，而 value up-projection 被移到 core attention 之后。RoPE channel 仍保持显式，QK scale 仍是模型语义规定的 scale。

2. **稀疏契约不变量（sparse-contract invariant）。** Token selection 发生在 sparse-prefill operator 之前；`indices` 提供 rows，重复项仍按重复项计算，因果合法性由 caller 保证，并且每个 `topk_length` 都必须位于 `[0, topk]`。优化路径与 reference path 必须保持相同的 attention sink 和全 invalid 约定。

3. **所有权不变量（ownership invariant）。** 在 regular head-128 specialization 中，一个 2-CTA cluster 负责一个 query row。这对 CTA 沿不同的轴切分 Q/output heads、selected K rows 和 V features；CTA 0 的 warp 12 是 CTA-group MMA 的唯一 issuer。

4. **驻留不变量（residency invariant）。** 在这个 specialization 中，$L$ 表示 FP32 原始 logits，$W$ 表示 BF16 未归一化指数权重。K、V、$L$ 和 $W$ 的大型 workspace 都是原位复用的单份 tile，而 `NUM_BUFS=2` 驱动的是 barrier/phase ring。在数据缓冲区（data buffers）中，只有小型 packed-validity mask 才有两个 physical slots。

5. **交接不变量（handoff invariant）。** 在这个 specialization 中，ready/done barriers 转移每个可复用 segment 的 ownership，proxy fences 则对 generic 与 asynchronous SMEM access 排序。唯一 issuer 保证 QK($k$) 先于 PV($k-1$) 发射，而 threshold 为 6 的优化保证上报 maximum 的语义不变，并保持 online LSE 的递推关系。

前两条不变量定义 FlashMLA sparse-prefill operator semantics，后三条定义 regular head-128 specialization 的 schedule contract。其他由 dispatch 选中的 specializations 可以采用各自的 tile sizes、register budgets、ownership 和 barrier topology，同时保持相同的 operator 结果。

## 练习与扩展验证

Regular head-128 specialization 是 dispatch space 中的一个点。源码树还包含 head-64 phase-1 specialization，以及 head-128 `d_qk=512` small-top-k specialization。它们采用不同 schedule，dispatch 会据此按 tile economics 选择 specialization。

1. **复现 weight absorption。** 给可执行的 absorption proof 加入 causal mask。确认 MHA mode 与 MQA mode 仍然一致，再故意把 absorbed path 的 scale 改为 $1/\sqrt{D_{latent}}$，测量产生的误差。

2. **压力测试 sparse validity。** 在 `sparse_prefill_reference` 中加入全 invalid query、重复 indices、`topk_length=0`，以及正负无穷的 sink values。分别写出预期的 `(out, max_logits, lse)`。

3. **追踪 ownership。** 对一个 top-k tile，给 Q、K、$L$、$W$、V 和 O 的每个 dimension 标注 `(CTA, local row, local column)`，找出一个 logical row 在哪些位置需要另一个 CTA 拥有的信息。

4. **审计 residency。** 从 `SMEMPool` allocation 开始画出每个 alias interval。验证为什么 $d_{sq}$ Q prefix 必须保持 live，而 384-column suffix 可以移到 TMEM，并找出结束每个 reuse hazard 的 barrier。

5. **测量 threshold。** 对 random 和 adversarial logits 加入 instrumentation，统计 `should_scale_o` 取 true 的频率。比较 threshold 6、always-rebase 和 never-rebase 三种策略的 numerical error 与 TMEM O traffic。

6. **比较 dispatch。** 对 head count 64/128、`d_qk` 512/576，以及 1280 附近的 top-k values 调用 `select_kernel`。运行前先预测选择的 specialization，再检查哪些 constraint 属于 front door，哪些由单个 specialization 强制执行。

7. **阅读 generated program。** 编译 smoke shape，在生成的 PTX 中找到 `tcgen05` MMA、TMA gather、mbarrier 和 proxy-fence instructions，再把它们映射回对应 TIRx line。最后重新运行 numerical check：source、generated code 和实际观察值是三类互补的证据。

FlashMLA 展示了一种通用的高性能 irregular-operator 设计模式：indexer 生成 sparse addresses，TMA 把这些 addresses gather 成 dense tiles，tensor cores 消费 tiles，显式 barrier 则保护激进的 storage reuse。把 algorithm、dispatch contract、ownership map 和 memory protocol 放在一起分析，便能将高速 kernel 还原成一套可以解释和验证的程序。
