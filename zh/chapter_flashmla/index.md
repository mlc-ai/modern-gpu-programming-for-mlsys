(chap_flashmla)=
# FlashMLA

:::{admonition} 概览
:class: overview

- 从普通 MHA 的 KV cache 出发，理解 MLA 为什么只需为每个 token 保存一份共享的压缩状态，以及各 head 特有的 K/V 变换被移到了哪里。
- 理解 FlashMLA sparse-prefill operator 的输入：外部 indexer 先选出 KV rows，FlashMLA 再对这些 rows 完成 attention；随后用一个可执行 reference 明确其数值语义和边界条件。
- 以一个具体实现为例，理解 sparse attention 如何映射到 Blackwell，并在 B200 上完成编译与数值验证。
:::

在介绍 MLA 之前，先回顾普通 attention 为什么需要 **KV cache**。计算一个新的 query 时，attention 需要读取更早 tokens 的 key 和 value。如果每次都重新计算这些 K/V，就会产生大量重复工作，因此系统会把已经得到的 K/V 缓存起来，供后续 query 直接读取。

这些状态能够安全复用，是因为生成模型通常采用因果（causal）attention：位置 $s$ 的 K/V 只依赖位置 $s$ 及其之前的 tokens，后来追加的 token 不会改变它们。Prefill 会并行计算 prompt 中的 tokens，并填充 cache；decode 随后逐步读取已有 cache，再追加新 token 的 K/V。KV cache 因而是在用存储容量和读取带宽换掉对整个 prefix 的重复计算。

普通 multi-head attention（MHA）中，每个 attention head 都有自己的 K 和 V。因此，KV cache 必须为每个已经处理过的 token 保存各个 head 的 K/V。上下文越长，需要保存和读取的数据就越多。接下来要解决的问题，就是怎样缩小这部分 cache。

**Multi-head Latent Attention（MLA）** 的核心做法是换一种 cache 表示：不再保存各个 head 展开后的 K/V，而只为每个 token 保存一份由所有 heads 共享的压缩状态。这并不是把所有 heads 合并成一个 head。每个 head 仍有自己的变换，只是 cache 不再需要长期保存这些变换产生的 K/V。至于一份共享状态怎样保留各 head 的差异，后面会用公式逐步推导。

MLA 说明的是 cache 中保存什么，以及 attention 应该怎样计算；FlashMLA 解决的则是怎样在 GPU 上高效完成这些计算。**FlashMLA** 是 DeepSeek 为 MLA 开发的高性能 GPU kernel library，包含适用于不同 attention 阶段和 cache 格式的多类算子。

本章关注 prefill 阶段的 sparse attention。与访问全部历史位置的 dense attention 不同，这里的外部 indexer 会先选出需要关注的 KV rows，FlashMLA 只对这些 rows 计算 attention。我们会先明确统一的算子语义，等算法讲清以后再选择具体实现。

这就引出了 MLA 的核心问题：多个 query heads 读取同一份缓存状态，为什么仍能得到不同的结果？答案的关键是：共享的只是压缩后的源状态。各 head 特有的变换并没有消失，而是移到了核心 attention 计算的两侧。下面这张图用 128 个 heads 把问题具体化，随后的推导仍保留一般的 head 数。

```{figure} ../../img/flashmla_cache_story_zh.svg
:width: 100%
:alt: 普通 MHA 为每个 head 分别缓存 key 和 value；MLA 只保存一份共享压缩状态，并把各 head 特有的计算放在 attention 两侧

普通 MHA 为每个 head 分别保存一份 key/value。MLA 为每个 token 只保存一份共享的压缩内容状态和位置信息；各 head 特有的 query 与 output 变换分别在 attention 前后完成。
```

图中的变换为什么成立？我们先从普通 MHA 的 KV cache 算起。

## 先从普通 MHA 的 KV cache 算起

上面的直观描述还没有回答 cache 到底有多大。设 $h_t\in\mathbb{R}^{d_{model}}$ 是 token $t$ 的 hidden state，$n_h$ 是 head 数量，$d_h$ 是每个 head 的宽度；$i$ 表示某个 head，$s$ 表示一个已经缓存的 key-token 位置。普通 MHA 会为每个 head $i$ 分别生成 query、key 和 value：

$$
q_{t,i}=W_i^Q h_t,\qquad
k_{t,i}=W_i^K h_t,\qquad
v_{t,i}=W_i^V h_t.
$$

对于位置 $t$ 上的 query，head $i$ 计算：

$$
p_{t,s,i}=\operatorname{softmax}_s
\left(\frac{q_{t,i}^{\mathsf T}k_{s,i}}{\sqrt{d_h}}\right),
\qquad
o_{t,i}=\sum_s p_{t,s,i}v_{s,i}.
$$

Output projection 会混合各个 head 的结果。但在 autoregressive generation 期间，所有更早位置的 $k_{s,i}$ 和 $v_{s,i}$ 都必须继续可用。因此，每层的 cache 需要为每个 token 保存 $2n_h d_h$ 个元素，也就是为每个 head 分别保存一份 K slice 和一份 V slice。

把 batch、序列长度、层数和数据类型都计入后，cache 的实际容量会很快放大。若 batch size 为 $B$，每条序列长度均为 $S$，层数为 $N_{layer}$，每个元素占 $b_{elem}$ bytes，那么普通 MHA 的 KV cache 大小为

$$
M_{MHA}=2N_{layer}BSn_hd_hb_{elem}\quad\text{bytes}.
$$

对于变长 batch，只需将 $BS$ 换成 $\sum_{j=1}^{B}S_j$，其中 $S_j$ 是第 $j$ 条序列的长度。例如，32 层、batch size 为 1、context length 为 4096、32 个 heads、每个 head 128 维，并使用 BF16 时，仅 KV cache 就需要 2 GiB。这个估算未计入 allocator metadata、模型权重和其他中间状态。

若固定 $d_{model}=n_h d_h$，仅改变 head 数并不一定改变这项总宽度。结构上的代价在于 cache 仍需 materialize 各 head 的独立状态。

在 dense decode 中，每个新 query 都要读取不断增长的历史 K/V，因此这些读取常会成为瓶颈。缩小 cache 不仅能容纳更长的 context 或更大的 batch，也会减少每个 decode step 的 HBM 读取量。

Multi-query attention（MQA）让所有 query heads 共享一个 K/V head，以此减小 cache；grouped-query attention 则在组内共享。它们都是有用的模型架构，但 MLA 采用了另一条路径：保留具有表达能力的 per-head projection，同时只 cache 一个共享的低维数据源，需要时可以从中恢复各 head 的表示。

### 如果先缓存一份共享状态，会发生什么？

暂时忽略 RoPE，也不做低秩压缩。令 $h_s$ 表示位置 $s$ 在某一层送入 attention projections 的 hidden state。它不是词表中的 token embedding。上标 $C$ 表示不含位置信息的 content channel；普通 MHA 的 content key 和 value 可以写成

$$
k_{s,i}^{C}=W_i^Kh_s,\qquad v_{s,i}^{C}=W_i^Vh_s.
$$

如果直接缓存一份共享的 $h_s$，矩阵结合律和线性变换对求和的分配性允许我们改变计算顺序：

$$
(q_{t,i}^{C})^{\mathsf T}W_i^Kh_s
=\left((W_i^K)^{\mathsf T}q_{t,i}^{C}\right)^{\mathsf T}h_s,
\qquad
\sum_s p_{t,s,i}W_i^Vh_s
=W_i^V\left(\sum_s p_{t,s,i}h_s\right).
$$

左边先为每个 key、每个 head 展开 K/V；右边则把 key projection 移到当前 query，把 value projection 移到加权求和之后。两边的数值完全相同，但右边只需长期保存一份 $h_s$。这可以看成一个尚未压缩的 latent cache 思想实验。

问题是，$h_s$ 的宽度仍然是 $d_{model}$。缓存虽然已经在 heads 之间共享，QK（query--key 点积）和 PV（probability--value 加权求和）却都要在这个较宽的空间中计算，因此这个方案通常并不实用。MLA 再加入一个训练得到的低秩 bottleneck，把共享状态压到 $d_c$ 维。原始 [DeepSeek-V2 MLA 推导](https://arxiv.org/abs/2405.04434)定义

$$
c_t^{KV}=W^{DKV}h_t,
$$

再分别做 up-projection：

$$
k_{t,i}^{C}=W_i^{UK}c_t^{KV},\qquad
v_{t,i}^{C}=W_i^{UV}c_t^{KV}.
$$

这里 $c^{KV}\in\mathbb{R}^{d_c}$ 是共享的 latent state。可以把前面的思想实验理解为 $d_c=d_{model}$、$W^{DKV}=I$、$c^{KV}=h$，并令 $W_i^{UK}=W_i^K$、$W_i^{UV}=W_i^V$；真正的 MLA 则学习一个窄得多的 $c^{KV}$。

这个低秩步骤不是把任意已经训练好的 MHA 无损压成 $d_c$ 维。它是模型训练时施加的 joint low-rank 结构。对 MLA 的 effective content K/V projections 来说，

$$
W_i^K=W_i^{UK}W^{DKV},\qquad
W_i^V=W_i^{UV}W^{DKV}.
$$

也就是说，所有 heads 的 content K/V projections 共享右侧因子 $W^{DKV}$。模型会在这个约束下学习怎样把需要的信息保存在 $c^{KV}$ 中。

于是思路分成两步：先利用结合律和线性性，把 per-head projections 移到 query 与 output 路径，让一份共享状态直接参与 core attention；再用学习到的 bottleneck 缩小这份状态。Cache 不需要保存展开后的 $k^C$ 和 $v^C$，只需为每个 token 保存一份 $c^{KV}$。

### RoPE 为什么需要单独的 positional channel？

Rotary positional embedding（RoPE）会旋转 Q/K 中负责表示位置的子空间。记 $R_u$ 为位置 $u$ 对应的 RoPE rotation。这让上面的简单图景变得复杂。假设直接对 content query 和 key 使用 RoPE，它们的 dot product 会变成

$$
\left(R_tq_{t,i}^{C}\right)^{\mathsf T}
\left(R_sW_i^{UK}c_s^{KV}\right)
=
(q_{t,i}^{C})^{\mathsf T}R_t^{\mathsf T}R_sW_i^{UK}c_s^{KV}.
$$

其中 $R_t^{\mathsf T}R_s$ 随 query 与 key 的相对位置变化。矩阵乘法当然仍可结合，但 $W_i^{UK}$ 无法再被预先合并成一个可供当前 query 对所有 key 位置共用的 projection。为此，MLA 使用 decoupled positional channel：每个 query head 有自己的 $q_{t,i}^{R}$，而所有 heads 共享一个 cached $k_t^R$。真正以 MHA 形式计算的 query 和 key 为：

$$
q_{t,i}=[q_{t,i}^{C};q_{t,i}^{R}],\qquad
k_{s,i}=[k_{s,i}^{C};k_s^R].
$$

因此，未乘 scale 的 score 可以拆成

$$
\operatorname{score}_{t,s,i}
=(q_{t,i}^{C})^{\mathsf T}W_i^{UK}c_s^{KV}
+(q_{t,i}^{R})^{\mathsf T}k_s^R.
$$

此时 cache 保存的是 $[c_s^{KV};k_s^R]$，仍然由所有 heads 共享。Positional channel 保持显式，weight absorption 只作用于 content channel。

把几种机制放在一起，就能看出 MLA 与普通 MQA 的区别。下表只比较每个 token、每层需要缓存的元素数，不比较模型能力，也不计数据类型和 allocator metadata：

| 机制 | 每个 token、每层的 cache 元素数 | 缓存的内容 |
| --- | ---: | --- |
| MHA | $2n_hd_h$ | $n_h$ 组展开后的 K/V |
| GQA | $2n_{kv}d_h$ | $n_{kv}$ 组展开后的 K/V，$1<n_{kv}<n_h$ |
| MQA | $2d_h$ | 1 组展开后的 K/V |
| MLA | $d_c+d_h^R$ | 共享的 latent content 与 RoPE key |

这里 $n_{kv}$ 是 GQA 的 KV-head 数，$d_c$ 是 $c^{KV}$ 的宽度，$d_h^R$ 是共享 RoPE key 的宽度。对于后文常见的配置，$d_c=512$、$d_h^R=64$，所以每个 cached row 有 $512+64=576$ 个 coordinates。若只与 $d_h=128$ 的普通 MQA 比较，576 是 $2d_h=256$ 的 2.25 倍；这只是该组维度下的缓存元素数对照，并不表示 MLA 等同于某种 GQA，也不推出模型能力高低。

### 各 head 的 up-projection 去了哪里？

前面的未压缩思想实验已经展示了怎样移动 projections。现在把同样的重排应用到 MLA 实际的 $W_i^{UK}$ 和 $W_i^{UV}$。MLA 的 core attention 可以用两种代数上等价的模式执行。这里的 “MQA mode” 是一种执行方式，不能据此把所有 MLA 模型都理解成普通 MQA 模型。

| MLA 执行模式 | 提交给 kernel 的 core-attention K/V | Up-projection 发生的位置 |
| --- | --- | --- |
| MHA mode | 每个 head 的 $[W_i^{UK}c^{KV};k^R]$ 和 $W_i^{UV}c^{KV}$ | Core attention 之前 |
| MQA mode | 共享的 $[c^{KV};k^R]$ 和共享 latent value $c^{KV}$ | 吸收到 query 与 output 路径 |

```{figure} ../../img/flashmla_mla_modes_zh.svg
:width: 100%
:alt: MLA 的 MHA 与 MQA 执行模式，以及两条 weight-absorption 路径

MLA 的 MHA mode 在 core attention 之前展开 latent KV；MQA mode 则把 key up-projection 移到 query 路径，把 value up-projection 移到 output 路径。两种模式都显式保留共享的 RoPE key。
```

两种模式计算相同的结果，却不一定具有相同的执行成本。MHA mode 先展开 per-head K/V，core attention 的 feature width 较小；如果许多 query rows 会复用这些展开结果，这项成本可以被摊薄。Absorbed MQA mode 让 core attention 直接处理较宽的 latent representation，但不必为历史 rows materialize per-head K/V。具体选择取决于 sequence stage、稀疏性、shape、数据搬运和硬件 schedule，不能简化成“prefill 一定用 MHA”或“decode 一定用 MQA”。

图中已经给出答案，下面两个恒等式会证明它。这里的“吸收”不是删除权重，也不是交换矩阵顺序，而是利用矩阵乘法的结合律改变求值分组，使展开后的 K/V 无需 materialize。Key 一侧先重新结合矩阵乘法：

$$
(q_{t,i}^{C})^{\mathsf T}W_i^{UK}c_s^{KV}
=\left((W_i^{UK})^{\mathsf T}q_{t,i}^{C}\right)^{\mathsf T}c_s^{KV}.
$$

定义吸收权重后的 query $q_{t,i}^{A}=(W_i^{UK})^{\mathsf T}q_{t,i}^{C}$，content score 就可以直接与 cached latent 做 dot product。

用两个坐标算一次就能看清这件事。取 $q=(1,2)^{\mathsf T}$、$c=(3,4)^{\mathsf T}$，以及 $W=\begin{bmatrix}1&2\\0&1\end{bmatrix}$。先展开 key 得到 $q^{\mathsf T}(Wc)=19$；先变换 query 则得到 $(W^{\mathsf T}q)^{\mathsf T}c=19$。前者对每个 cached row 计算 $Wc$，后者对当前 query 只计算一次 $W^{\mathsf T}q$。

Value 一侧则利用线性关系：

$$
\sum_s p_{t,s,i}W_i^{UV}c_s^{KV}
=W_i^{UV}\left(\sum_s p_{t,s,i}c_s^{KV}\right).
$$

因此，$W_i^{UV}$ 可以与模型的 output projection 合并。Attention kernel 不必真正 materialize 展开的 per-head K 或 per-head V。

本章研究的 DeepSeek Sparse Attention（DSA）sparse-prefill 路径采用的正是这个 MQA mode：每个被选中的 latent KV entry 由所有 query heads 共享。因此，下面先用一个小程序验证两种计算确实等价。具体模型为什么选择不同 mode，以及这些宽度如何映射到完整 tensor contract 与 dispatch，留到进入 kernel contract 时再说明。

### 一个小程序能否证明两种模式一致？

下面的 CPU 程序同时构造两种执行方式。MHA 路径显式展开 K/V，MQA 路径吸收相同的矩阵，并且两边都加入共享的 RoPE score term。使用 Float64 可以让等价性检查足够敏感，从而发现 index 转置或 contraction 写错之类的问题。

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

注意，这里的 scale 仍然由模型语义中的 QK head dimension 决定。不能仅仅因为吸收权重后的 dot product 恰好有 $D_{latent}$ 个 coordinates，就把它改成 $1/\sqrt{D_{latent}}$。

:::{admonition} Query 侧也可以压缩，但它不是本章的主线
:class: note

MLA 还可以通过一个独立的低秩 latent 分解 query projection：

$$
c_t^Q=W^{DQ}h_t,\qquad q_t^C=W^{UQ}c_t^Q.
$$

这项分解主要降低训练时的 activation memory，不会进一步缩小 KV cache。而且，它发生在 FlashMLA core attention 之前：传给 kernel 的 `q` 已经是 projection 后的 query。所以下文把 $q^C$ 当作输入，继续沿着 KV cache 这条主线分析。
:::

Weight absorption 回答了“一条共享 KV row 如何服务多个 query heads”。接下来还要回答另一个独立问题：sparse-prefill operator 收到的 KV row list 究竟由谁选择？

## Sparse-prefill operator 会自己选择 token 吗？

Dense attention 会访问每个合法的 KV token。DSA 先用轻量的 *lightning indexer* 为候选 token 打分，并为每个 query 选出一个 top-$k$ 集合；sparse core attention 随后只读取这些 latent KV entries。若原始 context length 为 $L$，prefill 的 core-attention 计算量会从 $O(L^2)$ 变成 $O(Lk)$，不过 indexer 自身也有开销。

```{figure} ../../img/flashmla_sparse_story_zh.svg
:width: 100%
:alt: Lightning indexer 先选择 token，sparse-prefill operator 再执行 attention

Token 选择与 sparse core attention 是两个相互独立的算子。Indexer 输出 row addresses；sparse-prefill operator 根据这些地址 gather 对应的 rows，再完成 QK--softmax--PV 计算。
```

本章研究的 sparse-prefill operator 不计算 index scores，也不执行 top-k selection，而是直接接收 `indices` tensor。因此它的语义 contract 是：

1. gather 指定的 KV rows；
2. 将越界位置和被 length mask 的位置标为 invalid；
3. 对剩余 rows 计算 attention；如果 caller 给出了重复 indices，重复项也会参与计算；
4. 返回 output、maximum logit 和 log-sum-exp。

这个 interface 没有 causal flag。若 caller 需要 causal attention，就必须生成一个只含允许访问的 keys 的 index list。稀疏性本身并不等同于 causal mask。

如果提供 `topk_length`，FlashMLA sparse-prefill contract 要求每个 query 的值都满足 `0 <= topk_length[q] <= topk`。这是 caller 应保证的前置条件。

这个 prefill 接口也没有 batch 维度。每个 query token 提供一份 selected-token list，它的 `h_q` 个 query heads 共用这份 list。Serving system 必须在调用前 flatten batch，或用其他方式完成 batch mapping。

这些规则已经回答了 operator 接受什么、拒绝什么。接下来把它们写成 CPU reference，先验证语义能够独立运行，再讨论 FlashMLA 如何实现。

## 能否先把 sparse contract 写成可执行 reference？

阅读优化实现之前，先统一通用 contract 中的 shape 符号：

| 符号 | 含义 |
| --- | --- |
| `s_q` | query rows 的数量 |
| `s_kv` | 可寻址 KV rows 的数量 |
| `h_q` | 每个 query row 的 query-head 数 |
| `h_kv` | 每个 KV row 的 KV-head 数；本接口要求为 1 |
| `d_qk` | QK 使用的 query/key 宽度 |
| `d_v` | value 与 output 的宽度，且 `d_v <= d_qk` |
| `topk` | 每个 query row 提供的 index slots 数量 |

对应的 tensors 为 `q[s_q,h_q,d_qk]`、`kv[s_kv,h_kv,d_qk]`、`indices[s_q,1,topk]` 和 `out[s_q,h_q,d_v]`。可选 sink 的 shape 是 `[h_q]`，可选 `topk_length` 的 shape 是 `[s_q]`，返回的 `max_logits` 和 `lse` 都是 `[s_q,h_q]`。后文的 regular head-128 specialization 只会进一步固定 `h_q=128` 和 `d_v=512`；`h_kv=1` 已经是通用 sparse-prefill contract 的一部分。

现在把“应该算什么”写成可执行 reference。在吸收权重后的 MQA contract 中，`kv[:, 0, :]` 同时提供 K 和 V：全部 `d_qk` 个坐标参与 QK，前 `d_v` 个坐标则作为 latent value。

这里还有两个容易在代码中突然出现的边界语义。第一，**attention sink** 可以看成额外加入一个 logit $a_i$，但它对应的 value vector 为 0。下面令 $x_{ij}$ 表示 head $i$ 对第 $j$ 个 KV row 的普通 logit，$v_j$ 表示对应的 value，$m_i$ 表示数值稳定所用的普通 logits 最大值。Sink 只进入 output 的 denominator：

$$
O_i=\frac{\sum_j e^{x_{ij}-m_i}v_j}
{\sum_j e^{x_{ij}-m_i}+e^{a_i-m_i}}.
$$

Sink 不参与返回的 `max_logits` 或 `lse`。第二，如果一个 query 的 selected rows 全部无效，reference 约定 `output=0`、`max_logits=-inf`、`lse=+inf`。显式写出这项约定，可以避免在 softmax 中计算 `(-inf)-(-inf)`。

下面的 CPU 程序按四步实现这个 contract：

1. 将 `indices` clamp 成可安全读取的地址，再 gather 对应 KV rows；
2. 合并地址边界与 `topk_length`，得到 validity mask；
3. 计算 QK、mask、softmax 与 PV，并把 sink 加入最终 denominator；
4. 返回 output、maximum logit 和不含 sink 的 log-sum-exp。

第一步只解决“地址可以安全读取”，不能代替后面的 mask。越界地址对应的 V row 还要在 PV 前清零：按照 IEEE arithmetic，softmax weight 为 0 并不能消除 NaN。完整代码如下：

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

这段 reference 明确了 FlashMLA sparse-prefill operator 对合法输入和返回值的数值 contract。任何实现都必须在其 QK scale 能力边界内复现已经定义的结果；tile 切分、storage reuse 和 pipeline overlap 则属于实现，而不属于 operator contract。

## Sparse prefill 在 FlashMLA 算子族中的位置

前面已经明确了 sparse-prefill operator 的语义。现在把它放回完整的算子族。[FlashMLA 官方仓库](https://github.com/deepseek-ai/FlashMLA)覆盖下面四类组合：

| Selection | Sequence stage | 代表性用途 |
| --- | --- | --- |
| dense | prefill | MHA forward 与 backward |
| dense | decode | 为新生成的 queries 读取 MLA KV cache |
| token-sparse | prefill | 对 selected-token list 执行 DSA core attention |
| token-sparse | decode | 对 selected FP8 KV cache 执行 DSA inference |

FlashMLA 不等同于本章研究的 sparse-prefill operator，更不等同于后半章聚焦的 regular head-128 specialization。本章选择 sparse prefill，是因为它能把算子语义与一个关键的实现问题连起来：如何将不规则 row addresses 整理成规则的 tensor-core tiles？其他三类算子不在本章范围内。

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

这段 public call 定义 FlashMLA sparse-prefill API；前面的 reference 则把其中与本章相关的数值语义写成了可执行形式。后文使用的 TIRx 是建立在 TVM 0.26 TIR 之上的 Python DSL 扩展。TIRx 中同名的 `flash_mla_sparse_fwd` 是 registry/dispatch bridge，而不是这个 Python API 的完整复刻：它按 shape 选择三个 SM100 phase-1 specializations 之一，并把调用转交给相应实现。

这个入口还把两项责任留给 caller。第一，每个 `topk_length[q]` 都必须位于 `[0, topk]`；TIRx prefill specializations 不会 clip 或逐项验证该值，大于 `topk` 会使实现读取到 `indices` storage 之外。第二，对于只因 `topk_length` 被 mask、地址本身仍合法的 row，优化路径不承诺净化其中的 NaN。前面的 reference 有意保留这项行为，因此没有为接口增加额外保证；越界地址则仍按 reference 的规则安全处理。

:::{admonition} TIRx 入口中的 `sm_scale`
:class: warning

FlashMLA public call 会在 runtime 接收 `sm_scale`。TIRx 入口保留了 registry name 和 shape dispatch，却**没有**保留完整的 call signature：三个 prefill specializations 都将 `sm_scale` specialize 为 `1 / sqrt(d_qk)`，launch ABI 中没有 scale argument。通过 `**kwargs` wrappers 传入 `sm_scale=...` 会被静默忽略。因此，下文的 B200 示例验证的是 TIRx 入口所覆盖的 QK scale；它不是 FlashMLA `flash_mla_sparse_fwd` 的 drop-in replacement。若模型语义中的 QK scale 不同，就必须将正确数值暴露为参数，或在编译时将其 specialize。

Weight absorption 并不构成改变该 scale 的理由。
:::

明确 operator family、FlashMLA API 与 TIRx 入口的边界后，下一步才是选择一条具体 specialization。

## 本章具体研究哪一个 Blackwell 案例？

Regular head-128 案例可以自然衔接前面的 FlashAttention 章节：它保留熟悉的 QK--softmax--PV 主链，再加入 irregular gather、吸收权重后的 latent KV 和两个线程块之间的协作分工，让读者不必同时面对所有新概念。

### 哪些 shape 会进入 regular head-128 路径？

算法部分只需要知道“共享 latent KV”；现在进入具体实现，才需要把每个 tensor 的 shape 写完整。Regular head-128 specialization 的输入输出 contract 是：

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

这里的 128 正是开篇问题中的 query-head 数，而 `kv` 中的 1 表示所有 heads 共享同一条 KV row。常见的 $d_{qk}=576$ 由 512 个 latent-content coordinates 和 64 个 RoPE coordinates 组成，`d_v=512` 则对应 latent value 宽度。这是本章 absorbed MQA representation 的 shape，不是所有 MLA operator 的通用 shape。

给前面定性的 mode 成本一个数值锚点：对同一 MLA layer，等价的 MHA representation 具有 $128+64=192$ 的 QK feature width 和 128 的 value/output feature width；这里的 absorbed MQA representation 则为 $512+64=576$ 和 512。若只粗略统计每个 query--key pair 在 QK 点积与 value 累加中涉及的乘加坐标数，两者是 $192+128=320$ 对 $576+512=1088$，后者约为前者的 3.4 倍。这不是 kernel runtime 的预测。[DeepSeek-V3.2 report](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf) Appendix A 也说明，同一模型会根据阶段和算法采用不同 mode：DeepSeek-V3.1-Terminus 在 training 和 prefill 时使用 MHA mode，在 decode 时使用 MQA mode；DSA sparse prefill 则使用 MQA mode。

下面用一组具体 shape 贯穿本节：`s_q=1`、`s_kv=8192`、`h_q=128`、`h_kv=1`、`d_qk=576`、`d_v=512`、`topk=2048`。它表示一个 query row、128 个 query heads，以及由这些 heads 共享的 2048 个 selected-index slots。这些 slots 可能包含重复或越界地址。没有更短的 `topk_length` 时，物理调度会按 128 个 slots 一组访问 $N=16$ 个 tiles；若给出 `topk_length`，实际访问的 tile 数是 `max(ceil(topk_length / 128), 1)`。

先把这些 selected-index tiles 在 kernel 中的算术过程归纳为六步。为与后文源码对应，$L$ 表示原始 QK logits，$W$ 表示 BF16 未归一化指数权重，`mi` 是 online-softmax 的指数参考值，`li` 是相对于该参考值累积的 denominator，$\widetilde O$ 是尚未除以 denominator 的累计 output：

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

在上面的具体例子中，前五步重复 16 轮，最后才执行第六步。算术主线明确后，再看 TIRx 实现如何分配硬件角色。

下面聚焦 TIRx regular head-128 实现中的 [`sparse_prefill_head128_phase1.py`](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py)。代码里的 `T` 是 TIR script namespace，`Tx` 是 GPU kernel helpers。虽然文件名中含有 `phase1`，这个 specialization 仍会在一个 kernel 中生成完整的 `(out, max_logits, lse)`；它不是等待 combine kernel 的 partial split-KV output。

后文会频繁使用三个执行层级。一个 **CTA**（cooperative thread array）就是一个 CUDA thread block（线程块）；相邻两个 CTA 组成一个 **cluster**，可以共同发起 CTA-group tensor-core operation。一个 **warpgroup** 由 4 个 warps、共 128 个 threads 组成，并在这条 kernel 中承担一个专门角色。这里的一个 cluster 将负责一个 query row。

在数学推导中，$p$ 表示归一化后的 softmax probability；而在源码里，`tmem_p` 和 register 变量 `p` 保存的是前面记作 $L$ 的原始 QK logits。源码中的 `s_frag` 和 `s_smem_gemm` 保存前面记作 $W$ 的未归一化指数权重。只有在 epilogue 中用 `li`（以及可选的 sink term）除 accumulated output 后，最终 output 才完成归一化。

下面较短的 TIRx 代码块都是 regular head-128 kernel 的上下文摘录，不是 standalone program。最后一节会编译并做数值验证；可以独立执行的 blocks 会显式说明。

Tensor contract 确定后，再看决定一次 tile 如何执行的常量：

```python
B_H = 128
B_TOPK = 128
D_V = 512
NUM_BUFS = 2
NUM_THREADS = 512
D_TQ = 384
```

这些名字分别对应后文会反复出现的结构：`B_H=128` 是每个 logical tile 的 query-head 数，`B_TOPK=128` 是每次处理的 selected-index slots 数，`D_V=512` 是 output feature 宽度，`NUM_THREADS=512` 表示每个 CTA 有四个 warpgroups，`D_TQ=384` 是稍后移入专用片上存储的 Q suffix 宽度。`NUM_BUFS=2` 只为两槽同步状态和小型 validity mask 提供循环槽位，并不表示有两份完整 K/V tile；数据驻留与 pipeline 两节会说明这一点。

这个 specialization 接受 512 或 576 的 `d_qk`，要求 `h_kv=1`、`d_v=512`，并要求 regular path 的 `topk` 是 128 的正整数倍。对于 128 heads，统一 front door 还会多做一次选择：`d_qk=512` 且 `topk<=1280` 时进入 small-top-k specialization；其他支持的 head-128 shapes 进入本章的 regular specialization。Head-64 shapes 使用 head-64 specialization。

`topk > 0` 是调用者必须满足的前置条件。统一 front door 会拒绝 `topk<=0`，但各 specialization 的 `_cfg().validate()` 只检查整除性，没有检查正数条件。因此，直接 import 某个 specialization 时仍须拒绝或避开非正的 `topk`；local validator 接受它并不意味着这种 launch 有效。

无需 launch GPU kernel 就能检查 dispatch。下面的代码块本身可独立执行，但需要先按本章末节安装 `tirx-kernels`：

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

Dispatch 本身记录在 [`flash_mla_sparse_fwd.py`](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/flash_mla_sparse_fwd.py#L66-L120) 中。把 dispatch 与 device schedule 分开，可以避免把一个 specialization 错当成整个 operator。

Dispatch 确定了实现入口。下一个问题是：两个 thread blocks 怎样协作完成前面的六步，又不重复搬运整个 tile？

### 为什么一个 query row 需要两个 CTA？

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

所以，2-CTA tensor-core operation 不只是“两个 CTA 各做同一个循环的一半”。QK 形成 128 个 heads 与 128 个 selected tokens 的两两组合，PV 再沿 token 维归约，得到 512 个 value coordinates。Collective `cta_group=2` MMA、配对的片上布局和跨 CTA 同步共同组成一个 logical tile。

现在再回到源码验证 launch topology。Launch grid 包含 `2 * s_q` 个 CTA，并将相邻 CTA 两两组成 cluster：

```python
block_idx = T.cta_id([2 * s_q])
T.cta_id_in_cluster([2])
cta_idx: T.let = block_idx % 2
s_q_idx: T.let = block_idx // 2
thread_idx = T.thread_id([512])
T.warpgroup_id([4])
```

因此，一个 cluster 负责一个 query row，每个 CTA 含 4 个 warpgroups。这个划分还可以从后面的数据索引直接看出：Q 按 `cta_idx` chunk，K producer 选择每个 top-k block 的 `cta_idx` 半块，V producer 则从 `cta_idx * 256` 开始。

## 各个 tile 放在哪里？

进入数据驻留图之前，先统一三种硬件存储：**global memory（GMEM）** 保存 kernel 的输入输出；**shared memory（SMEM）** 是 CTA 内线程共同访问的片上存储；**tensor memory（TMEM）** 是 Blackwell tensor cores 附近用于 operands 与 accumulators 的专用片上存储。

```{figure} ../../img/flashmla_dataflow_zh.svg
:width: 100%
:alt: QK、softmax、PV 与 epilogue 期间，global memory、shared memory、tensor memory 和 WG0 registers 中的数据驻留与生命周期复用

Q 被拆成 SMEM prefix 和 TMEM suffix。Gather 后的 K/V 进入 SMEM；原始 QK logits 与 output 在 TMEM 中累积；未归一化 softmax weights 再经过 SMEM 交给 PV。
```

图中的 registers 归当前线程所有。**Tensor Memory Accelerator（TMA）** 负责在 GMEM 与 SMEM 之间异步搬运规则 tile，也能根据地址列表执行 gather；`tcgen05` tensor-core operation 则从 SMEM/TMEM 读取 operands，并把大型 accumulators 留在 TMEM。

后文还会用两个简写描述 QK 的 operand 来源：**SS** 表示 Q、K 都从 SMEM 读取；**TS** 表示 Q 从 TMEM 读取、K 仍从 SMEM 读取。这条 kernel 将 Q 的 384-column suffix 搬到 TMEM，只把 prefix 留在 SMEM，所以 QK 要先做 SS prefix，再做 TS suffix。Softmax 产生的 BF16 未归一化权重则要写入 SMEM，供后面的 PV GEMM 使用。

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

**Completion barrier** 是一个小型硬件状态对象，记录异步 producer 何时完成；phase bit 则用来区分同一 barrier slot 的前后两次使用。后文再展开完整的 producer--consumer protocol。

这里要特别澄清一个容易误解的点：`NUM_BUFS = 2` **没有**分配两份完整的 K、V、$L$ 或 $W$ tile，这些 arrays 都没有 stage axis。`NUM_BUFS` 用于两槽 barrier/phase ring，也为小型 packed-validity mask 提供两个 slots；completion barrier 则保证可以安全地分段覆盖同一个大型 tile 的 physical storage。把这种 layout 称作“double-buffered K/V”，就会让人误以为存在两份实际上并不存在的 storage。

## 每次数据交接由哪个 warpgroup 负责？

数据放置确定后，下一问是谁生产每块 tile、谁消费它，以及谁归还可复用空间。四个 warpgroups 各做不同工作，而不是齐步推进：

| Warpgroup | Warps | 职责 |
| --- | --- | --- |
| WG0 | 0--3 | 从 TMEM 加载原始 logits $L$，mask、online softmax、写权重 $W$、rescale O、执行 epilogue |
| WG1 | 4--7 | 加载 index fragments，并为 K 发起 gather4 TMA |
| WG2 | 8--11 | 加载 index fragments，并为 V 发起 gather4 TMA |
| WG3 | 12--15 | CTA 0 的 warp 12 发起 CTA-group QK/PV MMA；每个 CTA 的 warp 13 构造 validity mask |

WG3 中剩下的 warps 并没有承担另一个隐藏 stage。这种不对称 role assignment 是有意设计的：一个 elected lane 就能为 CTA pair 发起 asynchronous MMA，而 exponentiation、row reduction、packing 和 epilogue conversion 则适合使用较多 lanes。

:::{admonition} Register budget 也跟着角色分配
:class: note

WG0 将上限提高到 144 registers，WG3 提高到 168；producer groups 则降到 96。TIRx API 用 `T.ptx.setmaxnreg(True, ...)` 表示提高，用 `T.ptx.setmaxnreg(False, ...)` 表示降低。这个配额解释了为什么不同 warpgroups 适合承担不同工作，但不改变下面的数据交接顺序。
:::

### 不规则的 rows 如何变成规则的 tiles？

稀疏的 row addresses 破坏了 dense attention 所用的 contiguous 2-D copy pattern。WG1 和 WG2 使用显式 TMA `gather4`：一次 issue 提供恰好 4 个 row coordinates，让一个 warp 可以把不连续的 KV rows 搬进规则的 SMEM tile。共享 helper 固定了 CTA-pair policy。

这里先使用 barrier 最基本的 producer--consumer 语义：producer 完成数据写入后，通过 ready/completion barrier 通知 consumer；consumer 等待后才能读取，使用完这段 storage 后再通过 done/free barrier 把复用权还给 producer。完整的 phase ring 会在 pipeline 一节说明。

下面的上下文摘录回答两个具体问题：一次 `gather4` issue 读取哪些 addresses，它的 completion 又交给哪个 barrier。读代码时先记住三个名字：`cur_buf` 是当前 tile 在两槽 barrier ring 中使用的槽位；`bar` 是 producer 与 consumer 共享的 completion barrier；`leader_mbar(...)` 取得 CTA pair 中负责汇总 TMA completion 的 leader 地址。Index names 和 slices 与链接源码一致：

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

Gather 与 validity 相关，但两者是分开的。Warp 13 的每个 active lane 加载 8 个 indices，并调用 `pack_valid_mask8`。当且仅当下面两个条件同时满足时，bit $i$ 才为 1：

$$
0\leq\text{index}_i<s_{kv}
\quad\text{and}\quad
\text{absolute_topk_position}_i<\text{topk_length}.
$$

WG0 等待 packed mask，再把对应的原始 logit $L$ 替换成 negative infinity，之后才求 maximum 或 exponential。因此，决定一个数值有限的 KV row 是否参与 attention 的是 packed mask，而不是假设 gather 会填 0。Mask 必须发生在 online-softmax state 更新之前。

源码清楚分开了这些 roles：[K producer 位于 lines 608--676](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L608-L676)， [V producer 位于 lines 679--729](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L679-L729)， [validity packing 位于 lines 841--865](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L841-L865)。

至此，不规则的 addresses 已经变成规则的 K/V SMEM tiles。下一问是 tensor core 为什么要从两个 memory spaces 读取同一个 QK dot product。

## 为什么 QK 要拆成 SMEM--SMEM 与 TMEM--SMEM 两段？

前面的驻留图已经定义了 SS 与 TS。拆分的目的，是让 Q 的大块 suffix 尽早离开 SMEM，从而给 K/V 和 epilogue 让出可复用空间，同时仍让 tensor core 完成完整 dot product。QK 在 $d_{sq}=d_{qk}-384$ 处分成两部分；下面的源码摘录展示这两个 partial products 如何写入同一个 accumulator：

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

第一步是 SS：Q 和 K operands 都由 SMEM 描述。第二步是 TS：Q 的 384-column suffix 来自 TMEM，K 仍在 SMEM。两步写入同一个 FP32 raw-logit accumulator（源码中的 `tmem_p`）；第一步清零，第二步累加。这样拆分 Q 后，SMEM 中只需保留较小的 Q prefix，使 union allocation 成为可能，同时不必放弃较大 suffix 所使用的 TS path。

Softmax 之后，PV 是 SS GEMM：BF16 $W$ 和 V 都在 SMEM，FP32 O accumulator 则留在 TMEM。Kernel 将 V rows 和 output columns 各分成两半，四种组合共同更新全部 512 个 value coordinates。

## Online softmax 如何避免不必要的 O 重缩放？

当 `topk` 大于 `B_TOPK=128` 时，一个 query row 会连续处理多个 selected-token tiles。每个 tile 都只看到一部分 scores，不能各自独立做完整 softmax；kernel 必须把前面 tiles 的状态带到下一轮。前面的具体例子在没有更短 `topk_length` 时有 $N=16$ 个 tiles，因此会递推合并 16 轮状态。

为了直接使用硬件 `exp2`，先把原始 QK dot product $x$ 乘以模型规定的 QK scale，再转换到以 2 为底的指数单位：

$$
r=x\cdot\text{semantic\_QK\_scale}\cdot\log_2(e).
$$

对于连续到来的 score tiles，普通 online-softmax recurrence 会保存指数参考值 $m$、denominator $\ell$ 和未归一化 output $\widetilde O$。若下一块 tile 含有 base-2 scaled scores $r$，则：

$$
m'=\max(m,\max r),\qquad
\alpha=2^{m-m'},
$$

$$
\ell'=\alpha\ell+\sum_j2^{r_j-m'},\qquad
\widetilde O'=\alpha\widetilde O+\sum_j2^{r_j-m'}v_j.
$$

在 TIRx regular head-128 specialization 中，编译期 `sm_scale_div_log2` 就是 `(1 / sqrt(d_qk)) * log2(e)`。这与原来的 softmax 相同，只是改写成直接映射到快速 base-2 exponential instructions 的形式。

对应到源码，`cur_pi_max` 是当前 128-token tile 的 base-2 最大值，`mi` 是递推中实际采用的指数参考值，`li` 是相对于 `mi` 累积的 denominator。`real_mi` 则单独保存迄今为止真实的最大值，用于最终报告 `max_logits`。区分 `mi` 与 `real_mi`，正是下面 lazy rescaling 能不改变输出统计量的关键。

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

如果当前 tile 的最大值比保存的指数参考值至多高 6 个 base-2 units，kernel 会继续保留旧参考值。新的 exponential 此时最大可能达到 $2^6=64$，但已累积的 O 无需 rescale。一旦差值超过 6，kernel 就更新参考值，同时 rescale $\ell$ 和已经存在的 O。Warp-wide `any_sync` 让参与计算的 rows 使用一致的决策。

`real_mi` 始终维护迄今为止真实的最大值，所以这项优化不会改变 `max_logits`。结束时，两个 64-token half 对每个 logical row 的贡献会被合并，kernel 输出：

$$
\mathrm{lse}=m\ln 2+\ln\ell.
$$

可选的 attention sink 将最终 output scale 改成：

```python
output_scale: T.float32 = T.cuda.fdividef(
    T.float32(1.0), li + T.ptx.exp2(attn_sink_log2 - mi)
)
```

但它有意不改变报告的 LSE。对于全 invalid row，特殊分支会输出 0，同时令 `max_logits=-inf`、`lse=+inf`，与前面的 reference 一致。

## 各个阶段如何安全重叠？

把这个 schedule 看成“数据所有权何时交给下一角色”最容易理解。宏观上，每个 tile 依次经历 `K ready → QK done → L consumed → W ready → PV done → V/O reusable`。QK 必须先完成，softmax 才能消费 logits；softmax 必须先产生 weights，PV 才能开始。但是，只要原位复用的 K 或 V segment 已经安全释放，负责 gather 的 warpgroups 就应该继续向前推进。

源码用 **memory barrier（mbarrier）** 表示这些交接。Producer 在数据或异步操作完成时贡献 arrival，consumer 等待相应 phase；consumer 用完后，再通过 done/free barrier 把覆盖这段 storage 的权利交还 producer。也就是说，ready 保护“不要过早读取”，done/free 保护“不要过早覆盖”。

```{figure} ../../img/flashmla_pipeline_stages_zh.svg
:width: 100%
:alt: 相差一个 tile 的 pipeline 填充、稳态与排空，以及 QK、softmax、PV 的重叠

填充阶段只发起 QK(0)，没有上一块 PV；排空阶段只发起 PV($N-1$)，随后进入最终 epilogue。稳态中，softmax($k-1$) 可与 QK($k$) 重叠；唯一的 MMA issuer 串行发出 QK($k$) 与 PV($k-1$)；QK($k$) 完成后，softmax($k$) 可与仍在异步执行的 PV($k-1$) 重叠。这不是两条同时发射 QK 与 PV 的 tensor-core streams；前面的具体例子中 $N=16$。
```

有了这张宏观时间线，再看初始化和具体 barrier 名称。Kernel 先由 warp 0 初始化 mbarriers，执行 cluster sync，launch Q prologue，分配 CTA-group TMEM，再进入 specialized loops。

CTA-group gather 的 TMA completion 会被路由到指定的 leader barrier，使 CTA pair 发出的操作共同满足同一 expected byte count。

把主要 barrier edges 写成 storage ownership transfer 会更容易理解：

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

Barrier 槽位按下面的方式循环使用：

```python
cur_buf = k % 2
cur_phase = (k // 2) & 1
```

这样，复用的 barrier slot 可以区分本轮到达与两轮之前的旧到达。

再次强调，这是一条 two-slot *barrier/phase ring*，并不是让两个完整 KV tiles 同时 resident。

`bar_qk_part_done` 允许 producer 在 K suffix 可以复用之前先替换 K prefix。两条 `bar_sv_*` edge 对 V 做同样的事。

```{figure} ../../img/flashmla_pipeline_zh.svg
:width: 100%
:alt: Sparse-prefill 详细 pipeline，展示 QK 与 PV 的串行发起、K/V 分段复用、mask-slot ring 和 WG0 交接

在稳态中，QK($k$) 与 PV($k-1$) 交错发起，其他 roles 同时执行分段 K/V gather 和 softmax。Barrier phase 保护 in-place storage reuse，而不是选择两份完整 tile buffer。
```

Barrier 只说明“某项工作已经完成”，还要处理不同 memory proxy 的可见性。这里有两类 proxy：线程执行的普通 SMEM load/store 属于 **generic proxy**，TMA 与 tcgen05 的异步访问属于 **async proxy**。如果一边写、另一边读同一片 SMEM，仅有 barrier arrival 并不能自动约束两个 proxy 观察 memory effects 的先后关系。

因此，`T.ptx.tcgen05.fence.*` 约束 TMEM access 与线程可见操作的先后关系；`T.ptx.fence.proxy_async("shared::cta")` 则在 SMEM 的 generic 与 async proxy 之间建立顺序。它既用于普通 store 写入 $W$ 或 epilogue tile 后、tcgen05/TMA 执行异步读取之前，也用于异步 SMEM read 完成后、普通代码覆写共用 storage 之前。

Mbarrier 传达 completion 并移交 storage 使用权，proxy fence 则约束不同 proxy 的 memory effects；两者不能互相替代。

## 如何编译并验证 regular head-128 specialization？

到这里需要验证三件不同的事：regular head-128 实现能否用 TVM 0.26 编译，生成的 kernel 能否在 B200 上实际 launch，以及它的 output、maximum logits 和 LSE 是否与前面的 reference 一致。只完成第一项并不能证明后两项。

这个 specialization 面向 compute capability 10，其 TMA/tcgen05 形式要求 SM100 class GPU。环境应使用 B200、CUDA 12.9 或更高版本，以及官方 Apache TVM 0.26.0 package。

首先通过 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/)安装支持 B200 的 CUDA-enabled PyTorch build。`tirx-kernels` 仓库会 import PyTorch，但没有把它声明为 package dependency。

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

`run_test` 不只是 compile。它会分配随机 BF16 Q/KV 和随机 indices，运行生成的 kernel，逐个 query row 求 FP32 PyTorch oracle，再用明确的 tolerance 检查 output、maximum logits 和 LSE。这才是默认应使用的验证路径；仅仅编译出 PTX 无法发现 head partition 错误或遗漏 validity bit。

`tirx-kernels` CLI 可以运行完整的 registered configuration：

```bash
python -m tirx_kernels.test \
  --kernel sparse_flashmla_prefill_head128_phase1 \
  --config bench_regular_dqk576_hq128_s4096_kv8192_topk2048
```

Negative tests 同样重要。设置 `inject_invalid_indices=True` 可以覆盖负数和过大的 row ID，设置 `have_topk_length=True` 可以覆盖 position predicate。还应测试全 invalid row，并确认约定的 0/-infinity/+infinity 行为。最后，对 head-64 和 small-top-k shapes 调用统一 dispatch entry，避免把一次成功的 regular-head128 运行误认为已经覆盖全部 prefill specializations。即便这些 dispatch 都能通过，也不能据此声称与完整 FlashMLA API 等价。

这些正向与异常输入测试，分别对应下面要总结的 operator semantics 与 specialization schedule contract。

## 哪些不变量属于 operator，哪些属于 specialization？

1. **缓存不变量（cache invariant）。** 一条 `h_kv=1` 的 latent KV row 可以服务全部 `h_q` 个 query heads，是因为 key up-projection 已吸收到各 head 的 query 路径，而 value up-projection 被移到 core attention 之后。RoPE channel 仍保持显式，QK scale 仍是模型语义规定的 scale。

2. **稀疏契约不变量（sparse-contract invariant）。** Token selection 发生在 sparse-prefill operator 之前；`indices` 提供 rows，重复项仍按重复项计算，因果合法性由 caller 保证，并且每个 `topk_length` 都必须位于 `[0, topk]`。优化路径与 reference path 必须保持相同的 attention sink 和全 invalid 约定。

3. **所有权不变量（ownership invariant）。** 在 regular head-128 specialization 中，一个 2-CTA cluster 负责一个 query row。这对 CTA 沿不同的轴切分 Q/output heads、selected K rows 和 V features；CTA 0 的 warp 12 是 CTA-group MMA 的唯一 issuer。

4. **驻留不变量（residency invariant）。** 在这个 specialization 中，$L$ 表示 FP32 原始 logits，$W$ 表示 BF16 未归一化指数权重。K、V、$L$ 和 $W$ 的大型 workspace 都是原位复用的单份 tile，而 `NUM_BUFS=2` 驱动的是 barrier/phase ring。在数据缓冲区（data buffers）中，只有小型 packed-validity mask 才有两个 physical slots。

5. **交接不变量（handoff invariant）。** 在这个 specialization 中，ready/done barriers 转移每个可复用 segment 的 ownership，proxy fences 则对 generic 与 asynchronous SMEM access 排序。唯一 issuer 保证 QK($k$) 先于 PV($k-1$) 发射，而 threshold 为 6 的优化保证上报 maximum 的语义不变，并保持 online LSE 的递推关系。

前两条不变量定义 FlashMLA sparse-prefill operator semantics，后三条定义 regular head-128 specialization 的 schedule contract。另一条由 dispatch 选中的 specialization 可以修改 tile sizes、register budgets、ownership 或 barrier topology，但必须显式给出替代 contract，而不能悄悄改变 operator 所计算的结果。

## 接下来应该测试什么？

Regular head-128 specialization 只是 dispatch space 中的一个点。源码树还包含 head-64 phase-1 specialization，以及 head-128 `d_qk=512` small-top-k specialization。它们使用不同 schedule，这说明 sparse attention 应根据 tile economics 做 dispatch，而不应被强行塞进一个 universal template。

1. **复现 weight absorption。** 给可执行的 absorption proof 加入 causal mask。确认 MHA mode 与 MQA mode 仍然一致，再故意把 absorbed path 的 scale 改为 $1/\sqrt{D_{latent}}$，测量产生的误差。

2. **压力测试 sparse validity。** 在 `sparse_prefill_reference` 中加入全 invalid query、重复 indices、`topk_length=0`，以及正负无穷的 sink values。分别写出预期的 `(out, max_logits, lse)`。

3. **追踪 ownership。** 对一个 top-k tile，给 Q、K、$L$、$W$、V 和 O 的每个 dimension 标注 `(CTA, local row, local column)`，找出一个 logical row 在哪些位置需要另一个 CTA 拥有的信息。

4. **审计 residency。** 从 `SMEMPool` allocation 开始画出每个 alias interval。验证为什么 $d_{sq}$ Q prefix 必须保持 live，而 384-column suffix 可以移到 TMEM，并找出结束每个 reuse hazard 的 barrier。

5. **测量 threshold。** 对 random 和 adversarial logits 加入 instrumentation，统计 `should_scale_o` 取 true 的频率。比较 threshold 6、always-rebase 和 never-rebase 三种策略的 numerical error 与 TMEM O traffic。

6. **比较 dispatch。** 对 head count 64/128、`d_qk` 512/576，以及 1280 附近的 top-k values 调用 `select_kernel`。运行前先预测选择的 specialization，再检查哪些 constraint 属于 front door，哪些由单个 specialization 强制执行。

7. **阅读 generated program。** 编译 smoke shape，在生成的 PTX 中找到 `tcgen05` MMA、TMA gather、mbarrier 和 proxy-fence instructions，再把它们映射回对应 TIRx line。最后重新运行 numerical check：source、generated code 和实际观察值是三类互补的证据。

本章的核心结论并不局限于 FlashMLA。高性能 irregular operator 往往会分阶段把工作规则化：indexer 生成 sparse addresses，TMA 把这些 addresses gather 成 dense tiles，tensor cores 消费 tiles，显式 barrier 则保护激进的 storage reuse。只有同时理解 algorithm、dispatch contract、ownership map 和 memory protocol，才能把高速 kernel 从难以理解的产物变成可以解释的程序。
