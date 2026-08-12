(chap_flashmla)=
# FlashMLA

:::{admonition} 概览
:class: overview

- 从普通 MHA 的 KV cache 出发，理解 MLA 为什么只需为每个 token 保存一份共享的压缩状态，以及各 head 特有的 K/V 变换被移到了哪里。
- 理解 FlashMLA sparse-prefill operator 的输入：外部 indexer 先选出 KV rows，FlashMLA 再对这些 rows 完成 attention；随后用一个可执行 reference 明确其数值语义和边界条件。
- 通过 TIRx `flash_mla_sparse_fwd` 入口理解 shape dispatch 与 QK scale 的 API 差异，再沿 regular head-128 specialization 分析 2-CTA 分工、数据驻留、warpgroup 角色与 pipeline，最后在 B200 上编译并验证。
:::

在介绍 MLA 之前，先回顾普通 attention 为什么需要 **KV cache**。计算一个新的 query 时，attention 需要读取更早 tokens 的 key 和 value。如果每次都重新计算这些 K/V，就会产生大量重复工作，因此系统会把已经得到的 K/V 缓存起来，供后续 query 直接读取。

普通 multi-head attention（MHA）中，每个 attention head 都有自己的 K 和 V。因此，KV cache 必须为每个已经处理过的 token 保存各个 head 的 K/V。上下文越长，需要保存和读取的数据就越多。MLA 要解决的第一个问题，就是怎样缩小这部分 cache。

这种方法称为 **Multi-head Latent Attention（MLA）**。它的核心做法是换一种 cache 表示：不再保存各个 head 展开后的 K/V，而只为每个 token 保存一份由所有 heads 共享的压缩状态。这并不是把所有 heads 合并成一个 head。每个 head 仍有自己的变换，只是 cache 不再需要长期保存这些变换产生的 K/V。至于一份共享状态怎样保留各 head 的差异，后面会用公式逐步推导。

MLA 说明的是 cache 中保存什么，以及 attention 应该怎样计算；FlashMLA 解决的则是怎样在 GPU 上高效完成这些计算。DeepSeek 将这套高性能 GPU kernel library 称为 **FlashMLA**。其中包含适用于不同 attention 阶段和 cache 格式的多类算子。

本章先以 FlashMLA `flash_mla_sparse_fwd` 的 sparse-prefill contract 为语义主线。Prefill 表示并行处理 prompt 中的 query tokens；sparse 表示先由外部 indexer 选出需要关注的 KV rows，再对选中的 rows 计算 attention。

TIRx 也注册了一个同名的 `flash_mla_sparse_fwd` 入口。它是一座实现桥梁：按 shape dispatch 到 head-64、regular head-128 和 small-top-k head-128 三个 SM100 specializations，但其 QK scale 已 specialize 为 `1 / sqrt(d_qk)`。后半章再以其中的 regular head-128 specialization 为实现主线，分析它在 Blackwell 上的 schedule。这样，FlashMLA sparse-prefill contract 回答“算什么”，TIRx 入口回答“选哪条实现”，被选中的 specialization 回答“怎样在 GPU 上执行”。其他实现只在后文比较算子族、dispatch 和设计取舍时出现。

先预告后半章的 regular head-128 案例：一个 query token 对应一行 query，这一行包含 128 个 query heads；每个 cached token 则对应 cache 中的一行共享 KV。于是问题变成了：128 个 heads 读取同一行 KV，为什么仍能得到不同的结果？下面这张图先给出直观线索：共享的是生成 K/V 所需的源状态，而不是各 head 最终参与 attention 的表示。各 head 特有的变换仍然位于点积、softmax 和加权求和这些核心 attention 计算的两侧。精确的 tensor shapes 会在完成 MLA 推导后统一列出。

```{figure} ../../img/flashmla_cache_story_zh.svg
:width: 100%
:alt: 普通 MHA 为每个 head 分别缓存 key 和 value；MLA 只保存一份共享压缩状态，并把各 head 特有的计算放在 attention 两侧

普通 MHA 为每个 head 分别保存一份 key/value。MLA 为每个 token 只保存一份共享的压缩内容状态和位置信息；各 head 特有的 query 与 output 变换分别在 attention 前后完成。
```

这张图只给出了直觉，还没有说明共享状态怎样保留各 head 的差异，也没有证明移动这些变换为什么不改变计算结果。下面先算清普通 MHA 的 cache 开销，再推导 MLA 到底缓存了什么，以及各 head 的变换为什么可以移动。等 cache 表示和等价变换都讲清楚之后，我们再进入 FlashMLA sparse-prefill contract 和 regular head-128 schedule。

## 先从普通 MHA 的 KV cache 算起

上面的直观描述还没有回答 cache 到底有多大。设
$h_t\in\mathbb{R}^{d_{model}}$ 是 token $t$ 的 hidden state，$n_h$ 是 head
数量，$d_h$ 是每个 head 的宽度；$i$ 表示某个 head，$s$ 表示一个已经缓存的
key-token 位置。普通 MHA 会为每个 head $i$ 分别生成 query、key 和 value：

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

Output projection 会混合各个 head 的结果。但在 autoregressive generation
期间，所有更早位置的 $k_{s,i}$ 和 $v_{s,i}$ 都必须继续可用。因此，每层的
cache 需要为每个 token 保存 $2n_h d_h$ 个元素，也就是为每个 head 分别保存
一份 K slice 和一份 V slice。

若固定 $d_{model}=n_h d_h$，仅改变 head 数并不一定改变这项总宽度。结构上的
代价在于 cache 仍需 materialize 各 head 的独立状态。

在长序列 decode 中，读取这些随 token history 增长的 slices 会主导整个 step。

Multi-query attention（MQA）让所有 query heads 共享一个 K/V head，以此
减小 cache；grouped-query attention 则在组内共享。它们都是有用的模型架构，
但 MLA 采用了另一条路径：保留具有表达能力的 per-head projection，同时只
cache 一个共享的低维数据源，需要时可以从中恢复各 head 的表示。

### MLA 如何只缓存一次，却恢复 per-head 行为？

最初的 [DeepSeek-V2 MLA 推导](https://arxiv.org/abs/2405.04434)先定义一个
联合潜在 KV 向量（joint latent KV vector）：

$$
c_t^{KV}=W^{DKV}h_t,
$$

再分别做 up-projection：

$$
k_{t,i}^{C}=W_i^{UK}c_t^{KV},\qquad
v_{t,i}^{C}=W_i^{UV}c_t^{KV}.
$$

上标 $C$ 表示不含位置信息的 *content* channel。关键在于，cache 不需要为
每个 head 保存展开后的 $k^C$ 和 $v^C$，只需为每个 token 保存一份
$c^{KV}$。

Rotary positional embedding（RoPE）会旋转 Q/K 中负责表示位置的子空间。
这让上面的简单图景变得复杂：如果在 $W^{UK}$ 与 dot product 之间加入这种
position-dependent rotation，矩阵就无法再重新结合。为此，MLA 使用
decoupled positional channel：每个 query head 有自己的 $q_{t,i}^{R}$，
而所有 heads 共享一个 cached $k_t^R$。真正以 MHA 形式计算的 score 使用：

$$
q_{t,i}=[q_{t,i}^{C};q_{t,i}^{R}],\qquad
k_{s,i}=[k_{s,i}^{C};k_s^R].
$$

此时 cache 保存的是 $[c_s^{KV};k_s^R]$，仍然由所有 heads 共享。
Positional channel 保持显式，weight absorption 只作用于 content channel。

:::{admonition} Query 侧也可以压缩，但它不是本章的主线
:class: note

MLA 还可以通过一个独立的低秩 latent 分解 query projection：

$$
c_t^Q=W^{DQ}h_t,\qquad q_t^C=W^{UQ}c_t^Q.
$$

这项分解主要降低训练时的 activation memory，不会进一步缩小 KV cache。
而且，它发生在 FlashMLA core attention 之前：传给 kernel 的 `q` 已经是
projection 后的 query。所以下文把 $q^C$ 当作输入，继续沿着 KV cache 这条
主线分析。
:::

### 各 head 的 up-projection 去了哪里？

到这里，我们知道了 cache 里保存什么，但还没有回答一个关键问题：各 head 的
$W_i^{UK}$ 和 $W_i^{UV}$ 是否仍要在 attention 前显式执行？MLA 的 core
attention 可以用两种代数上等价的模式执行。这里的 “MQA mode” 是一种执行
方式，不能据此把所有 MLA 模型都理解成普通 MQA 模型。

| MLA 执行模式 | 提交给 kernel 的 core-attention K/V | Up-projection 发生的位置 |
| --- | --- | --- |
| MHA mode | 每个 head 的 $[W_i^{UK}c^{KV};k^R]$ 和 $W_i^{UV}c^{KV}$ | Core attention 之前 |
| MQA mode | 共享的 $[c^{KV};k^R]$ 和共享 latent value $c^{KV}$ | 吸收到 query 与 output 路径 |

```{figure} ../../img/flashmla_mla_modes_zh.svg
:width: 100%
:alt: MLA 的 MHA 与 MQA 执行模式，以及两条 weight-absorption 路径

MLA 的 MHA mode 在 core attention 之前展开 latent KV；MQA mode 则把 key
up-projection 移到 query 路径，把 value up-projection 移到 output 路径。
两种模式都显式保留共享的 RoPE key。
```

图中已经给出答案，下面两个恒等式会证明它。Key 一侧先重新结合矩阵乘法：

$$
(q_{t,i}^{C})^{\mathsf T}W_i^{UK}c_s^{KV}
=\left((W_i^{UK})^{\mathsf T}q_{t,i}^{C}\right)^{\mathsf T}c_s^{KV}.
$$

定义吸收权重后的 query
$q_{t,i}^{A}=(W_i^{UK})^{\mathsf T}q_{t,i}^{C}$，content score 就可以直接
与 cached latent 做 dot product。Value 一侧则利用线性关系：

$$
\sum_s p_{t,s,i}W_i^{UV}c_s^{KV}
=W_i^{UV}\left(\sum_s p_{t,s,i}c_s^{KV}\right).
$$

因此，$W_i^{UV}$ 可以与模型的 output projection 合并。Attention
kernel 不必真正 materialize 展开的 per-head K 或 per-head V。

本章研究的 DeepSeek Sparse Attention（DSA）sparse-prefill 路径采用的正是
这个 MQA mode：每个被选中的 latent KV entry 由所有 query heads 共享。因此，
下面先用一个小程序验证两种计算确实等价。具体模型为什么选择不同 mode，
以及 512/576 等实现 shapes 的含义，留到进入 kernel contract 时再说明。

### 一个小程序能否证明两种模式一致？

下面的 CPU 程序同时构造两种执行方式。MHA 路径显式展开 K/V，MQA 路径吸收
相同的矩阵，并且两边都加入共享的 RoPE score term。使用 Float64 可以
让等价性检查足够敏感，从而发现 index 转置或 contraction 写错之类的问题。

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

注意，这里的 scale 仍然由模型语义中的 QK head dimension 决定。不能仅仅因为
吸收权重后的 dot product 恰好有 $D_{latent}$ 个 coordinates，就把它改成
$1/\sqrt{D_{latent}}$。

Weight absorption 回答了“一条共享 KV row 如何服务多个 query heads”。接下来
还要回答另一个独立问题：sparse-prefill operator 收到的 KV row list 究竟由谁选择？

## Sparse-prefill operator 会自己选择 token 吗？

Dense attention 会访问每个合法的 KV token。DSA 先用轻量的 *lightning
indexer* 为候选 token 打分，并为每个 query 选出一个 top-$k$ 集合；sparse
core attention 随后只读取这些 latent KV entries。若原始 context length 为
$L$，prefill 的 core-attention 计算量会从 $O(L^2)$ 变成 $O(Lk)$，不过
indexer 自身也有开销。

```{figure} ../../img/flashmla_sparse_story_zh.svg
:width: 100%
:alt: Lightning indexer 先选择 token，sparse-prefill operator 再执行 attention

Token 选择与 sparse core attention 是两个相互独立的算子。Indexer 输出
row addresses；sparse-prefill operator 根据这些地址 gather 对应的 rows，
再完成 QK--softmax--PV 计算。
```

本章研究的 sparse-prefill operator 不计算 index scores，也不执行 top-k
selection，而是直接接收 `indices` tensor。因此它的语义 contract 是：

1. gather 指定的 KV rows；
2. 将越界位置和被 length mask 的位置标为 invalid；
3. 对剩余 rows 计算 attention；如果 caller 给出了重复 indices，重复项也会参与计算；
4. 返回 output、maximum logit 和 log-sum-exp。

这个 interface 没有 causal flag。若 caller 需要 causal attention，就必须生成
一个只含允许访问的 keys 的 index list。稀疏性本身并不等同于 causal mask。

如果提供 `topk_length`，FlashMLA sparse-prefill contract 要求每个 query 的值
都满足 `0 <= topk_length[q] <= topk`。这是 caller 应保证的前置条件。TIRx
prefill specializations 不会 clip 或逐项验证该值；大于 `topk` 会使它们读取到
`indices` storage 之外。

这个 prefill 接口也没有 batch 维度。每个 query token 提供一份
selected-token list，它的 `h_q` 个 query heads 共用这份 list。Serving system
必须在调用前 flatten batch，或用其他方式完成 batch mapping。`s_q`、`h_kv`
和 `topk` 对应的精确 tensor shape 会在介绍 public call 时统一定义；后半章的
regular specialization 再令 `h_q=128`。

这些规则已经回答了 operator 接受什么、拒绝什么。接下来把它们写成 CPU
reference，先验证语义能够独立运行，再讨论 FlashMLA 如何实现。

## 能否先把 sparse contract 写成可执行 reference？

阅读优化实现之前，先把“应该算什么”写成可执行 reference。在吸收权重后的
MQA contract 中，`kv[:, 0, :]` 同时提供 K 和 V：全部 `d_qk` 个坐标参与 QK，
前 `d_v` 个坐标则作为 latent value。

这里还有两个容易在代码中突然出现的边界语义。第一，**attention sink** 可以
看成额外加入一个 logit $a_i$，但它对应的 value vector 为 0。因此它只进入
output 的 denominator：

$$
O_i=\frac{\sum_j e^{x_{ij}-m_i}v_j}
{\sum_j e^{x_{ij}-m_i}+e^{a_i-m_i}}.
$$

Sink 不参与返回的 `max_logits` 或 `lse`。第二，如果一个 query 的 selected
rows 全部无效，reference 约定 `output=0`、`max_logits=-inf`、`lse=+inf`。
显式写出这项约定，可以避免在 softmax 中计算 `(-inf)-(-inf)`。

下面的 CPU 程序按四步实现这个 contract：

1. 将 `indices` clamp 成可安全读取的地址，再 gather 对应 KV rows；
2. 合并地址边界与 `topk_length`，得到 validity mask；
3. 计算 QK、mask、softmax 与 PV，并把 sink 加入最终 denominator；
4. 返回 output、maximum logit 和不含 sink 的 log-sum-exp。

第一步只解决“地址可以安全读取”，不能代替后面的 mask。越界地址对应的 V row
还要在 PV 前清零：按照 IEEE arithmetic，softmax weight 为 0 并不能消除 NaN。
对于只因 `topk_length` 而被 mask、地址本身仍合法的 row，这里有意不做净化；
FlashMLA sparse-prefill contract 和三个 TIRx prefill specializations 都没有
承诺净化这种异常输入。完整代码如下：

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

这段 reference 明确了 FlashMLA sparse-prefill operator 对合法输入和返回值的
数值 contract。上面对异常 NaN 输入的处理与 TIRx specializations 一致，但不会
为接口增加额外的净化保证。所有由 TIRx 入口选中的 specializations 都必须在其
QK scale 能力边界内复现已经定义的结果；tile 切分、storage reuse 和 pipeline
overlap 则属于实现，而不属于 operator contract。

## Sparse prefill 在 FlashMLA 算子族中的位置

前面已经明确了 sparse-prefill operator 的语义。现在把它放回完整的算子族：
*decode* 会在复用 KV cache 的同时，每一步加入一个新 query（或一小组
speculative queries），与前面处理整段 prompt 的 *prefill* 不同。
[FlashMLA 官方仓库](https://github.com/deepseek-ai/FlashMLA)将相关实现组织成
四类：

| Selection | Sequence stage | 代表性用途 |
| --- | --- | --- |
| dense | prefill | MHA forward 与 backward |
| dense | decode | 为新生成的 queries 读取 MLA KV cache |
| token-sparse | prefill | 对 selected-token list 执行 DSA core attention |
| token-sparse | decode | 对 selected FP8 KV cache 执行 DSA inference |

所以，FlashMLA 不等同于本章研究的 sparse-prefill operator，更不等同于后半章
聚焦的 regular head-128 specialization。本章选择 sparse prefill，是因为它能把
算子语义与一个关键的实现问题连起来：如何将不规则 row addresses 整理成规则的
tensor-core tiles？Sparse decode 有自己的 paged-cache、scheduling 和 reduction
contract，不在本章范围内。在 TIRx 中，它是单独注册的实现，内部
先运行 split-KV main kernel，再运行独立的 combine kernel；它不是
`flash_mla_sparse_fwd` 的第四个 dispatch target。

FlashMLA sparse-prefill public call 在概念上是：

```text
out, max_logits, lse = flash_mla_sparse_fwd(
    q, kv, indices, sm_scale,
    d_v=512,
    attn_sink=attn_sink,       # optional [h_q], float32
    topk_length=topk_length,   # optional [s_q], int32
)
```

这些参数对应前面 reference 明确的语义：`h_q` 是 query-head 数，`s_q` 是
query-row 数，`sm_scale` 用来缩放 QK scores，`d_v` 决定 value 和 output 的
宽度，`attn_sink` 可以为每个 query head 加入一个 value 为 0 的额外 logit，
`topk_length` 则限定每个 query 中有效 indices 的前缀长度。这个调用返回归一化
后的 output、经过 scale 的最大 logit，以及不包含 sink 的 log-sum-exp。

这段 public call 定义 FlashMLA sparse-prefill API；前面的 reference 则把其中
与本章相关的数值语义写成了可执行形式。TIRx 中同名的
`flash_mla_sparse_fwd` 是 registry/dispatch bridge，而不是这个 Python API 的
完整复刻：它按 shape 选择三个 SM100 phase-1 specializations 之一，并把调用
转交给相应实现。

:::{admonition} TIRx 入口中的 `sm_scale`
:class: warning

FlashMLA public call 会在 runtime 接收 `sm_scale`。TIRx 入口保留了 registry
name 和 shape dispatch，却**没有**保留完整的 call signature：三个 prefill
specializations 都将 `sm_scale` specialize 为 `1 / sqrt(d_qk)`，launch ABI 中
没有 scale argument。通过 `**kwargs` wrappers 传入 `sm_scale=...` 会被静默
忽略。因此，下文的 B200 示例验证的是 TIRx 入口所覆盖的 QK scale；它不是
FlashMLA `flash_mla_sparse_fwd` 的 drop-in replacement。若模型语义中的 QK
scale 不同，就必须将正确数值暴露为参数，或在编译时将其 specialize。

Weight absorption 并不构成改变该 scale 的理由。
:::

明确 operator family、FlashMLA API 与 TIRx 入口的边界后，下一步才是选择一条
具体 specialization。

## 本章具体研究哪一个 Blackwell 案例？

Regular head-128 案例可以自然衔接前面的 FlashAttention 章节：它保留熟悉的
QK--softmax--PV 主链，再加入 irregular gather、吸收权重后的 latent KV 和
2-CTA cooperative ownership，让读者不必同时面对所有新概念。

下面聚焦 TIRx regular head-128 实现中的
[`sparse_prefill_head128_phase1.py`](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py)。
TIRx 是建立在 TVM 0.26 TIR 之上的 Python DSL 扩展；代码里的 `T` 是 TIR
script namespace，`Tx` 是 GPU kernel helpers。

虽然文件名中含有 `phase1`，这个 regular head-128 specialization 仍会在一个
kernel 中生成完整的 `(out, max_logits, lse)`；它不是等待 combine kernel 的
partial split-KV output。

后文还会频繁使用三个执行层级。一个 **CTA**（cooperative thread array）就是
一个 CUDA thread block（线程块）；相邻两个 CTA 组成一个 **cluster**，可以
共同发起 CTA-group tensor-core operation。一个 **warpgroup** 由 4 个 warps、
共 128 个 threads 组成，并在这条 kernel 中承担一个专门角色。这里的一个
cluster 将负责一个 query row。

先明确一组命名，避免后文混淆。在数学推导中，$p$ 表示归一化后的 softmax
probability；而在源码里，`tmem_p` 和 register 变量 `p` 保存的是
**原始 QK logits**，下文将它们记作 $L$。源码中的 `s_frag` 和
`s_smem_gemm` 保存 BF16 **未归一化指数权重**，下文记作 $W$。只有在
epilogue 中用 `li`（以及可选的 sink term）除 accumulated output 后，
最终 output 才完成归一化。

下面较短的 TIRx 代码块都是 regular head-128 kernel 的上下文摘录，不是 standalone
program。最后一节会编译并做数值验证；可以独立执行的 blocks 会显式说明。

### 哪些 shape 会进入 regular head-128 路径？

算法部分只需要知道“共享 latent KV”；现在进入具体实现，才需要把每个 tensor
的 shape 写完整。Regular head-128 specialization 的输入输出 contract 是：

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

这里的 128 正是开篇问题中的 query-head 数，而 `kv` 中的 1 表示所有 heads
共享同一条 KV row。常见的 $d_{qk}=576$ 由 512 个 latent-content coordinates
和 64 个 RoPE coordinates 组成，`d_v=512` 则对应 latent value 宽度。这是本章
absorbed MQA representation 的 shape，不是所有 MLA operator 的通用 shape。

这个区别也会随模型和阶段变化。
[DeepSeek-V3.2 report](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)
Appendix A 说明，DeepSeek-V3.1-Terminus 在 training 和 prefill 时使用 MHA
mode，在 decode 时使用 MQA mode；DSA sparse prefill 则使用 MQA mode。

Tensor contract 确定后，再看决定一次 tile 如何执行的常量：

```python
B_H = 128
B_TOPK = 128
D_V = 512
NUM_BUFS = 2
NUM_THREADS = 512
D_TQ = 384
```

这些名字分别对应后文会反复出现的结构：`B_H=128` 是每个 logical tile 的
query-head 数，`B_TOPK=128` 是每次处理的 selected-token 数，`D_V=512` 是
output feature 宽度，`NUM_THREADS=512` 表示每个 CTA 有四个 warpgroups，
`D_TQ=384` 是移入 Blackwell tensor memory（TMEM）的 Q suffix 宽度。
`NUM_BUFS=2` 只为 barrier phases
和小型 validity mask 提供两个循环槽位，并不表示有两份完整 K/V tile；数据
驻留与 pipeline 两节会说明这一点。

这个 specialization 接受 512 或 576 的 `d_qk`，要求 `h_kv=1`、`d_v=512`，并要求
regular path 的 `topk` 是 128 的正整数倍。对于 128 heads，统一 front door 还会
多做一次选择：`d_qk=512` 且 `topk<=1280` 时进入 small-top-k specialization；
其他支持的 head-128 shapes 进入本章的 regular specialization。Head-64 shapes
使用 head-64 specialization。

`topk > 0` 是调用者必须满足的前置条件。统一 front door 会拒绝
`topk<=0`，但各 specialization 的 `_cfg().validate()` 只检查整除性，
没有检查正数条件。因此，直接 import 某个 specialization 时仍须拒绝或避开
非正的 `topk`；local validator 接受它并不意味着这种 launch 有效。

无需 launch GPU kernel 就能检查 dispatch。下面的代码块本身可独立执行，但
需要先按本章末节安装 `tirx-kernels`：

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

Dispatch 本身记录在
[`flash_mla_sparse_fwd.py`](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/flash_mla_sparse_fwd.py#L66-L120)
中。把 dispatch 与 device schedule 分开，可以避免把一个 specialization 错当成
整个 operator。

在分配硬件角色之前，先把 selected tokens 如何流过 kernel 化成六步。对每个
包含 128 个 selected tokens 的 tile，重复前五步；所有 tiles 都处理完后，才
执行第六步：

```text
对每个包含 128 个 selected tokens 的 tile：
    1. gather 该 tile 的 K rows
    2. gather 该 tile 的 V rows，并构造 validity mask
    3. QK：128 个 query heads × 128 个 selected tokens → logits L
    4. 更新 mask 后的 online softmax，得到 W、参考值 mi 和分母 li
    5. 必要时 rescale running state，再累加 O~ += W @ V
所有 selected tiles 完成后：
    6. 用 li + sink 归一化 O~，写回 out、max_logits 和 lse
```

现在算术主线已经清楚了。下一个问题是：两个 thread blocks 怎样协作完成这六步，
又不重复搬运整个 tile？

### 为什么一个 query row 需要两个 CTA？

难点出现在第 3、5 步的切分轴不同：QK 要形成所有 head 与 selected token 的
两两组合，PV 又要沿 token 维归约，产生 512 个 value coordinates。Regular
head-128 实现让两个 CTA 通过 `cta_group=2` 共同形成这块 logical tile，并在
QK 与 PV 之间旋转切分轴。

这个关系先用所有权图（ownership map）表示：

```{figure} ../../img/flashmla_cta_ownership_zh.svg
:width: 100%
:alt: 两个 CTA 对 query heads、selected K rows 和 V feature columns 的 ownership

对于每个 query row，CTA pair 会在 QK 与 PV 之间改变 logical partition。
每个 CTA 最终写出 64 个完整的 output heads，每个 head 含 512 个
coordinates。
```

这个 CTA pair 会用三种不同方式切分三个轴：

| 一个 128-token top-k tile 中的资源 | CTA 0 | CTA 1 |
| --- | --- | --- |
| Query/output head ownership | heads 0--63 | heads 64--127 |
| K-row gather ownership | selected tokens 0--63 | selected tokens 64--127 |
| V-feature gather ownership | value columns 0--255 | value columns 256--511 |

所以，2-CTA tensor-core operation 不只是“两个 CTA 各做同一个循环的一半”。
QK 形成 128 个 heads 与 128 个 selected tokens 的两两组合，PV 再沿 token 维
归约，得到 512 个 value coordinates。Collective `cta_group=2` MMA、配对的
SMEM/TMEM layouts 和 cross-CTA barriers 共同组成一个 logical tile。

现在再回到源码验证 launch topology。Launch grid 包含 `2 * s_q` 个 CTA，并将
相邻 CTA 两两组成 cluster：

```python
block_idx = T.cta_id([2 * s_q])
T.cta_id_in_cluster([2])
cta_idx: T.let = block_idx % 2
s_q_idx: T.let = block_idx // 2
thread_idx = T.thread_id([512])
T.warpgroup_id([4])
```

因此，一个 cluster 负责一个 query row，每个 CTA 含 4 个 warpgroups。这个
划分还可以从后面的数据索引直接看出：Q 按 `cta_idx` chunk，K producer 选择
每个 top-k block 的 `cta_idx` 半块，V producer 则从 `cta_idx * 256` 开始。

## 各个 tile 放在哪里？

进入数据驻留图之前，先统一硬件词汇。**Global memory（GMEM）** 保存 kernel
的输入输出；**shared memory（SMEM）** 是 CTA 内线程和异步搬运共同访问的片上
存储；**tensor memory（TMEM）** 是 Blackwell tensor cores 附近用于 operands
与 accumulators 的专用片上存储；registers 则归当前线程所有。**Tensor Memory
Accelerator（TMA）** 负责在 GMEM 与 SMEM 之间异步搬运规则 tile，也能根据
地址列表执行 gather。`tcgen05` tensor-core operation 则从 SMEM/TMEM 读取
operands，并把大型 accumulators 留在 TMEM。

后文还会用两个简写描述 QK 的 operand 来源：**SS** 表示 Q、K 都从 SMEM
读取；**TS** 表示 Q 从 TMEM 读取、K 仍从 SMEM 读取。这条 kernel 将 Q 的
384-column suffix 搬到 TMEM，只把 prefix 留在 SMEM，所以 QK 要先做 SS
prefix，再做 TS suffix。Softmax 产生的 BF16 未归一化权重则要写入 SMEM，供
后面的 PV GEMM 使用。

```{figure} ../../img/flashmla_dataflow_zh.svg
:width: 100%
:alt: QK、softmax、PV 与 epilogue 期间，global memory、shared memory、tensor memory 和 WG0 registers 中的数据驻留与生命周期复用

Q 被拆成 SMEM prefix 和 TMEM suffix。Gather 后的 K/V 进入 SMEM；原始 QK
logits 与 output 在 TMEM 中累积；未归一化 softmax weights 再经过 SMEM 交给
PV。
```

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

这些都是逻辑视图（logical views）；CTA-group TMEM layout 和 rearrangement
决定 MMA 与 load/store instructions 的实际 lane mapping。源码中的
`SMEMPool` 是共享内存分配器，用来从同一片动态 SMEM 中切出带对齐和生命周期
约束的 views。源码还另行分配一个 512-column CTA-group TMEM pool，再从中
切出 O、raw-logit 和 Q views。

SMEM 采用激进的 alias 与复用策略。只要 lifetime 允许，`q_full`、gather 后的
K/V region 和 output epilogue 就会复用类似 union 的 base。最后 384 个 Q columns 移到
TMEM 后，只有 $d_{sq}=d_{qk}-384$ 的 prefix 仍需保持 live，供 QK 第一部分
使用。`d_qk=512` 时 $d_{sq}=128$；`d_qk=576` 时则为 192。具体 allocation
plan 见
[`sparse_prefill_head128_phase1.py` lines 302--365](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L302-L365)。

这里要特别澄清一个容易误解的点：`NUM_BUFS = 2` **没有**分配两份完整的 K、V、
$L$ 或 $W$ tile，这些 arrays 都没有 stage axis。`NUM_BUFS` 用于两槽
barrier/phase ring，也为小型 packed-validity mask 提供两个 slots；completion
barrier 则保证可以安全地分段覆盖同一个大型 tile 的 physical storage。把这种
layout 称作“double-buffered K/V”，就会让人误以为存在两份实际上并不存在的
storage。

## 每次数据交接由哪个 warpgroup 负责？

数据放置确定后，下一问是谁生产每块 tile、谁消费它，以及谁归还可复用空间。
四个 warpgroups 各做不同工作，而不是齐步推进：

| Warpgroup | Warps | 职责 |
| --- | --- | --- |
| WG0 | 0--3 | 从 TMEM 加载原始 logits $L$，mask、online softmax、写权重 $W$、rescale O、执行 epilogue |
| WG1 | 4--7 | 加载 index fragments，并为 K 发起 gather4 TMA |
| WG2 | 8--11 | 加载 index fragments，并为 V 发起 gather4 TMA |
| WG3 | 12--15 | CTA 0 的 warp 12 发起 CTA-group QK/PV MMA；每个 CTA 的 warp 13 构造 validity mask |

WG3 中剩下的 warps 并没有承担另一个隐藏 stage。这种不对称 role assignment
是有意设计的：一个 elected lane 就能为 CTA pair 发起 asynchronous MMA，
而 exponentiation、row reduction、packing 和 epilogue conversion 则适合使用
较多 lanes。

:::{admonition} Register budget 也跟着角色分配
:class: note

WG0 将上限提高到 144 registers，WG3 提高到 168；producer groups 则降到
96。TIRx API 用 `T.ptx.setmaxnreg(True, ...)` 表示提高，用
`T.ptx.setmaxnreg(False, ...)` 表示降低。这个配额解释了为什么不同
warpgroups 适合承担不同工作，但不改变下面的数据交接顺序。
:::

### 不规则的 rows 如何变成规则的 tiles？

稀疏的 row addresses 破坏了 dense attention 所用的 contiguous 2-D copy pattern。
WG1 和 WG2 使用显式 TMA `gather4`：一次 issue 提供恰好 4 个 row coordinates，
让一个 warp 可以把不连续的 KV rows 搬进规则的 SMEM tile。共享 helper 固定了
CTA-pair policy。

下面的上下文摘录回答两个具体问题：一次 `gather4` issue 读取哪些 addresses，
它的 completion 又交给哪个 barrier。读代码时先记住三个名字：`cur_buf` 是
当前 tile 在两槽 barrier ring 中使用的槽位；`bar` 是 producer 与 consumer
共享的 completion barrier；`leader_mbar(...)` 取得 CTA pair 中负责汇总 TMA
completion 的 leader 地址。Index names 和 slices 与链接源码一致：

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

Gather 与 validity 相关，但两者是分开的。Warp 13 的每个 active lane 加载 8 个
indices，并调用 `pack_valid_mask8`。当且仅当下面两个条件同时满足时，bit $i$
才为 1：

$$
0\leq\text{index}_i<s_{kv}
\quad\text{and}\quad
\text{absolute_topk_position}_i<\text{topk_length}.
$$

WG0 等待 packed mask，再把对应的原始 logit $L$ 替换成 negative infinity，
之后才求 maximum 或 exponential。因此，决定一个数值有限的 KV row 是否参与
attention 的是 packed mask，而不是假设 gather 会填 0。Mask 必须发生在
online-softmax state 更新之前。

对于 TIRx regular head-128 specialization，caller 应避免让 `topk_length`
之后的 in-range KV rows 含有 NaN；这条优化数据路径不保证屏蔽这类异常输入。

源码见：[`sparse_prefill_head128_phase1.py`](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py)。其中 K producer 位于 lines 608--676，V producer 位于 lines 679--729，validity packing 位于 lines 841--865。

至此，不规则的 addresses 已经变成规则的 K/V SMEM tiles。下一问是 tensor core
为什么要从两个 memory spaces 读取同一个 QK dot product。

## 为什么 QK 要拆成 SMEM--SMEM 与 TMEM--SMEM 两段？

前面的驻留图已经定义了 SS 与 TS。拆分的目的，是让 Q 的大块 suffix 尽早离开
SMEM，从而给 K/V 和 epilogue 让出可复用空间，同时仍让 tensor core 完成完整
dot product。QK 在 $d_{sq}=d_{qk}-384$ 处分成两部分；下面的源码摘录展示这
两个 partial products 如何写入同一个 accumulator：

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

第一步是 SS：Q 和 K operands 都由 SMEM 描述。第二步是 TS：Q 的 384-column
suffix 来自 TMEM，K 仍在 SMEM。两步写入同一个 FP32 raw-logit accumulator
（源码中的 `tmem_p`）；第一步清零，第二步累加。这样拆分 Q 后，SMEM 中只需
保留较小的 Q prefix，使 union allocation 成为可能，同时不必放弃较大 suffix
所使用的 TS path。

Softmax 之后，PV 是 SS GEMM：BF16 $W$ 和 V 都在 SMEM，FP32 O accumulator
则留在 TMEM。Kernel 将 V rows 和 output columns 各分成两半，四种组合共同
更新全部 512 个 value coordinates。

## Online softmax 如何避免不必要的 O 重缩放？

当 `topk` 大于 `B_TOPK=128` 时，一个 query row 会连续处理多个 selected-token
tiles。每个 tile 都只看到一部分 scores，不能各自独立做完整 softmax；kernel
必须把前面 tiles 的状态带到下一轮。

为了直接使用硬件 `exp2`，先把原始 QK dot product $x$ 乘以模型规定的 QK
scale，再转换到以 2 为底的指数单位：

$$
r=x\cdot\text{semantic\_QK\_scale}\cdot\log_2(e).
$$

对于连续到来的 score tiles，普通 online-softmax recurrence 会保存指数参考值
$m$、denominator $\ell$ 和未归一化 output $\widetilde O$。若下一块 tile 含有
base-2 scaled scores $r$，则：

$$
m'=\max(m,\max r),\qquad
\alpha=2^{m-m'},
$$

$$
\ell'=\alpha\ell+\sum_j2^{r_j-m'},\qquad
\widetilde O'=\alpha\widetilde O+\sum_j2^{r_j-m'}v_j.
$$

在 TIRx regular head-128 specialization 中，编译期
`sm_scale_div_log2` 就是 `(1 / sqrt(d_qk)) * log2(e)`。这与原来的
softmax 相同，只是改写成直接映射到快速 base-2 exponential instructions
的形式。

对应到源码，`cur_pi_max` 是当前 128-token tile 的 base-2 最大值，`mi` 是递推
中实际采用的指数参考值，`li` 是相对于 `mi` 累积的 denominator。`real_mi`
则单独保存迄今为止真实的最大值，用于最终报告 `max_logits`。区分 `mi` 与
`real_mi`，正是下面 lazy rescaling 能不改变输出统计量的关键。

每次 row maximum 增加时都重缩放完整的 512-coordinate O tile，代价会很高。
Head-128 kernel 因此使用 lazy threshold：

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

如果当前 tile 的最大值比保存的指数参考值至多高 6 个 base-2 units，kernel
会继续保留旧参考值。新的 exponential 此时最大可能达到 $2^6=64$，但已累积
的 O 无需 rescale。一旦差值超过 6，kernel 就更新参考值，同时 rescale $\ell$
和已经存在的 O。Warp-wide `any_sync` 让参与计算的 rows 使用一致的决策。

`real_mi` 始终维护迄今为止真实的最大值，所以这项优化不会改变
`max_logits`。结束时，两个 64-token half 对每个 logical row 的贡献会被合并，
kernel 输出：

$$
\mathrm{lse}=m\ln 2+\ln\ell.
$$

可选的 attention sink 将最终 output scale 改成：

```python
output_scale: T.float32 = T.cuda.fdividef(
    T.float32(1.0), li + T.ptx.exp2(attn_sink_log2 - mi)
)
```

但它有意不改变报告的 LSE。对于全 invalid row，特殊分支会输出 0，同时令
`max_logits=-inf`、`lse=+inf`，与前面的 reference 一致。

## 各个阶段如何安全重叠？

把这个 schedule 看成“数据所有权何时交给下一角色”最容易理解。宏观上，每个
tile 依次经历 `K ready → QK done → L consumed → W ready → PV done → V/O
reusable`。QK 必须先完成，softmax 才能消费 logits；softmax 必须先产生 weights，
PV 才能开始。但是，只要原位复用的 K 或 V segment 已经安全释放，负责 gather
的 warpgroups 就应该继续向前推进。

源码用 **memory barrier（mbarrier）** 表示这些交接。Producer 在数据或异步
操作完成时贡献 arrival，consumer 等待相应 phase；consumer 用完后，再通过
done/free barrier 把覆盖这段 storage 的权利交还 producer。也就是说，ready
保护“不要过早读取”，done/free 保护“不要过早覆盖”。

```{figure} ../../img/flashmla_pipeline_stages_zh.svg
:width: 100%
:alt: 相差一个 tile 的 pipeline 填充、稳态与排空，以及 QK、softmax、PV 的重叠

填充阶段只发起 QK(0)，没有上一块 PV；排空阶段只发起 PV($N-1$)，随后进入
最终 epilogue。稳态中，softmax($k-1$) 可与 QK($k$) 重叠；唯一的 MMA issuer
串行发出 QK($k$) 与 PV($k-1$)；QK($k$) 完成后，softmax($k$) 可与仍在异步
执行的 PV($k-1$) 重叠。这不是两条同时发射 QK 与 PV 的 tensor-core streams。
```

有了这张宏观时间线，再看初始化和具体 barrier 名称。Kernel 先由 warp 0 初始化
mbarriers，执行 cluster sync，launch Q prologue，分配 CTA-group TMEM，再进入
specialized loops。

CTA-group gather 的 TMA completion 会被路由到指定的 leader barrier，使
CTA pair 发出的操作共同满足同一 expected byte count。

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

对于 tile $k$，唯一的 MMA issuer 先 launch QK($k$)。WG0 把 $L(k-1)$ 转成
$W(k-1)$ 后，同一个 issuer 才在 serial loop 后部 launch PV($k-1$)。因此，
QK 与 PV 由同一 instruction stream 交错发起，而不是两条同时发射的
tensor-core streams。

在 steady state 中，softmax($k-1$) 可以与 QK($k$) 重叠。唯一 issuer 在
QK($k$) 之后发出 PV($k-1$)；QK($k$) 完成后，softmax($k$) 又可以与仍在异步
执行的 PV($k-1$) 重叠。所有这些重叠仍受上述 part-level edges 约束。

Barrier 槽位按下面的方式循环使用：

```python
cur_buf = k % 2
cur_phase = (k // 2) & 1
```

这样，复用的 barrier slot 可以区分本轮到达与两轮之前的旧到达。

再次强调，这是一条 two-slot *barrier/phase ring*，并不是让两个完整 KV tiles
同时 resident。

`bar_qk_part_done` 允许 producer 在 K suffix 可以复用之前先替换 K prefix。
两条 `bar_sv_*` edge 对 V 做同样的事。

```{figure} ../../img/flashmla_pipeline_zh.svg
:width: 100%
:alt: Sparse-prefill 详细 pipeline，展示 QK 与 PV 的串行发起、K/V 分段复用、mask-slot ring 和 WG0 交接

在稳态中，QK($k$) 与 PV($k-1$) 交错发起，其他 roles 同时执行分段 K/V
gather 和 softmax。Barrier phase 保护 in-place storage reuse，而不是选择两份
完整 tile buffer。
```

Barrier 只说明“某项工作已经完成”，还要处理不同 memory proxy 的可见性。
这里有两类 proxy：线程执行的普通 SMEM load/store 属于 **generic proxy**，
TMA 与 tcgen05 的异步访问属于 **async proxy**。如果一边写、另一边读同一片
SMEM，仅有 barrier arrival 并不能自动约束两个 proxy 观察 memory effects 的
先后关系。

因此，`T.ptx.tcgen05.fence.*` 约束 TMEM access 与线程可见操作的先后关系；
`T.ptx.fence.proxy_async("shared::cta")` 则在 SMEM 的 generic 与 async proxy
之间建立顺序。它既用于普通 store 写入 $W$ 或 epilogue tile 后、tcgen05/TMA
执行异步读取之前，也用于异步 SMEM read 完成后、普通代码覆写共用 storage
之前。

Mbarrier 传达 completion 并移交 storage 使用权，proxy fence 则约束不同 proxy
的 memory effects；两者不能互相替代。

## 如何编译并验证 regular head-128 specialization？

到这里需要验证三件不同的事：regular head-128 实现能否用 TVM 0.26
编译，生成的 kernel 能否在 B200 上实际 launch，以及它的 output、maximum
logits 和 LSE 是否与前面的 reference 一致。只完成第一项并不能证明后两项。

这个 specialization 面向 compute capability 10，其 TMA/tcgen05 形式要求
SM100 class GPU。环境应使用 B200、CUDA 12.9 或更高版本，以及官方 Apache
TVM 0.26.0 package。

首先通过 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/)安装
支持 B200 的 CUDA-enabled PyTorch build。Companion repository 会 import
PyTorch，但没有把它声明为 package dependency。

随后安装 TVM 和 kernel repository，再运行示例：

```bash
python -m pip install "apache-tvm==0.26.0" cuda-bindings
git clone https://github.com/mlc-ai/tirx-kernels.git
cd tirx-kernels
git checkout 5be39749e7dfd2c4bdae9b4d396f8ec35af07126
pip install -e .
```

只用一个 query row 的 smoke test 仍会覆盖完整的 128-head、top-k-2048
kernel，同时保持 reference 时间较短：

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

`run_test` 不只是 compile。它会分配随机 BF16 Q/KV 和随机 indices，运行生成的
kernel，逐个 query row 求 FP32 PyTorch oracle，再用明确的 tolerance 检查
output、maximum logits 和 LSE。这才是默认应使用的验证路径；仅仅编译出 PTX
无法发现 head partition 错误或遗漏 validity bit。

Repository CLI 可以运行完整的 registered configuration：

```bash
python -m tirx_kernels.test \
  --kernel sparse_flashmla_prefill_head128_phase1 \
  --config bench_regular_dqk576_hq128_s4096_kv8192_topk2048
```

Negative tests 同样重要。设置 `inject_invalid_indices=True` 可以覆盖负数和过大的
row ID，设置 `have_topk_length=True` 可以覆盖 position predicate。还应测试全
invalid row，并确认约定的 0/-infinity/+infinity 行为。最后，对 head-64 和
small-top-k shapes 调用统一 dispatch entry，避免把一次成功的 regular-head128
运行误认为已经覆盖全部 prefill specializations。即便这些 dispatch 都能通过，
也不能据此声称与完整 FlashMLA API 等价。

这些正向与异常输入测试，分别对应下面要总结的 operator semantics 与
specialization schedule contract。

## 哪些不变量属于 operator，哪些属于 specialization？

1. **缓存不变量（cache invariant）。** 一条 `h_kv=1` 的 latent KV row 可以
   服务全部 `h_q` 个 query heads，是因为 key up-projection 已吸收到各 head 的
   query 路径，而 value up-projection 被移到 core attention 之后。RoPE channel
   仍保持显式，QK scale 仍是模型语义规定的 scale。

2. **稀疏契约不变量（sparse-contract invariant）。** Token selection 发生在
   sparse-prefill operator 之前；`indices` 提供 rows，重复项仍按重复项计算，
   因果合法性由 caller 保证，并且每个 `topk_length` 都必须位于 `[0, topk]`。
   优化路径与 reference path 必须保持相同的 attention sink 和全 invalid 约定。

3. **所有权不变量（ownership invariant）。** 在 regular head-128
   specialization 中，一个 2-CTA cluster 负责一个 query row。这对 CTA 沿
   不同的轴切分 Q/output heads、selected K rows 和 V features；CTA 0 的
   warp 12 是 CTA-group MMA 的唯一 issuer。

4. **驻留不变量（residency invariant）。** 在这个 specialization 中，$L$
   表示 FP32 原始 logits，$W$ 表示 BF16 未归一化指数权重。K、V、$L$ 和 $W$
   的大型 workspace 都是原位复用的单份 tile，而 `NUM_BUFS=2` 驱动的是
   barrier/phase ring。在数据缓冲区（data buffers）中，只有小型
   packed-validity mask 才有两个 physical slots。

5. **交接不变量（handoff invariant）。** 在这个 specialization 中，ready/done
   barriers 转移每个可复用 segment 的 ownership，proxy fences 则对 generic
   与 asynchronous SMEM access 排序。唯一 issuer 保证 QK($k$) 先于
   PV($k-1$) 发射，而 threshold 为 6 的优化保证上报 maximum 的语义不变，
   并保持 online LSE 的递推关系。

前两条不变量定义 FlashMLA sparse-prefill operator semantics，后三条定义
regular head-128 specialization 的 schedule contract。另一条由 dispatch 选中的
specialization 可以修改 tile sizes、register budgets、ownership 或 barrier
topology，但必须显式给出替代 contract，而不能悄悄改变 operator 所计算的结果。

## 接下来应该测试什么？

Regular head-128 specialization 只是 dispatch space 中的一个点。源码树还包含
head-64 phase-1 specialization，以及 head-128 `d_qk=512` small-top-k
specialization。它们使用不同 schedule，这说明 sparse attention 应根据 tile
economics 做 dispatch，而不应被强行塞进一个 universal template。

1. **复现 weight absorption。** 给可执行的 absorption proof 加入 causal mask。
   确认 MHA mode 与 MQA mode 仍然一致，再故意把 absorbed path 的 scale 改为
   $1/\sqrt{D_{latent}}$，测量产生的误差。

2. **压力测试 sparse validity。** 在 `sparse_prefill_reference` 中加入全 invalid
   query、重复 indices、`topk_length=0`，以及正负无穷的 sink values。分别写出
   预期的 `(out, max_logits, lse)`。

3. **追踪 ownership。** 对一个 top-k tile，给 Q、K、$L$、$W$、V 和 O 的每个
   dimension 标注 `(CTA, local row, local column)`，找出一个 logical row 在哪些
   位置需要另一个 CTA 拥有的信息。

4. **审计 residency。** 从 `SMEMPool` allocation 开始画出每个 alias interval。
   验证为什么 $d_{sq}$ Q prefix 必须保持 live，而 384-column suffix 可以移到
   TMEM，并找出结束每个 reuse hazard 的 barrier。

5. **测量 threshold。** 对 random 和 adversarial logits 加入 instrumentation，
   统计 `should_scale_o` 取 true 的频率。比较 threshold 6、always-rebase 和
   never-rebase 三种策略的 numerical error 与 TMEM O traffic。

6. **比较 dispatch。** 对 head count 64/128、`d_qk` 512/576，以及 1280 附近的
   top-k values 调用 `select_kernel`。运行前先预测选择的 specialization，再检查哪些
   constraint 属于 front door，哪些由单个 specialization 强制执行。

7. **阅读 generated program。** 编译 smoke shape，在生成的 PTX 中找到
   `tcgen05` MMA、TMA gather、mbarrier 和 proxy-fence instructions，再把它们
   映射回对应 TIRx line。最后重新运行 numerical check：source、generated
   code 和实际观察值是三类互补的证据。

本章的核心结论并不局限于 FlashMLA。高性能 irregular operator 往往会分阶段
把工作规则化：indexer 生成 sparse addresses，TMA 把这些 addresses gather 成
dense tiles，tensor cores 消费 tiles，显式 barrier 则保护激进的 storage
reuse。只有同时理解 algorithm、dispatch contract、ownership map 和 memory
protocol，才能把高速 kernel 从难以理解的产物变成可以解释的程序。
