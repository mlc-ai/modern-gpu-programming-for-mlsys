(chap_flashmla)=
# FlashMLA

:::{admonition} Overview
:class: overview

- Start from an ordinary MHA KV cache and derive why MLA stores one shared
  compressed state per token, including where the head-specific K/V
  transformations move.
- Define the FlashMLA sparse-prefill operator: an external indexer selects KV
  rows, then FlashMLA attends over those rows. An executable reference fixes its
  numerical behavior and edge cases.
- Use a concrete implementation to understand how sparse attention maps to
  Blackwell, then compile and verify it on B200.
:::

Before introducing MLA, recall why ordinary attention needs a **KV cache**. To
compute a new query, attention must read the keys and values of earlier tokens.
Recomputing them every time would repeat a large amount of work, so the system
caches the K/V states and lets later queries read them directly.

Those states remain reusable because generative models normally use causal
attention: the K/V at position $s$ depend only on tokens up to position $s$.
Appending later tokens cannot change them. Prefill computes the prompt tokens
in parallel and fills the cache; decode then reads that cache step by step and
appends the new token's K/V. A KV cache therefore trades storage capacity and
read bandwidth for avoiding repeated computation over the whole prefix.

In ordinary multi-head attention (MHA), every attention head has its own K and
V. The KV cache must therefore store every head's K/V for every processed token.
As the context grows, the amount of state that must be stored and read grows
with it. The next problem is how to reduce this cache.

**Multi-head Latent Attention (MLA)** uses a different cache representation.
Instead of storing expanded K/V for every head, it stores one compressed state
per token and shares that state across all heads. This does not merge all heads
into one. Each head still has its own transformations; the cache simply no
longer stores the K/V produced by those transformations as long-lived state. We
will derive how one shared state can preserve head-specific behavior.

MLA specifies what the cache stores and how attention is computed; FlashMLA
addresses how to perform those computations efficiently on a GPU. **FlashMLA**
is DeepSeek's library of optimized GPU kernels for MLA, with operators for
different attention stages and cache formats.

This chapter focuses on sparse attention during prefill. Unlike dense attention,
which visits the whole history, an external indexer first selects the relevant
KV rows and FlashMLA attends only to those rows. We will first establish the
shared operator semantics and postpone implementation choices until the
algorithm is clear.

This raises the central MLA question: if multiple query heads read one shared
cached state, how can they still produce different results? The key is that only
the compressed source state is shared. Head-specific transformations do not
disappear; they move to the two sides of the core attention computation. The
next figure uses 128 heads to make the question concrete, while the derivation
that follows keeps the head count general.

```{figure} ../img/flashmla_cache_story.png
:width: 100%
:alt: Ordinary MHA caches separate key and value data for every head; MLA stores one shared compressed state and keeps head-specific work around attention

Ordinary MHA stores a separate key/value slice per head. MLA stores one shared
compressed content state plus shared position information per token;
head-specific query and output transformations happen before and after
attention.
```

Why is the transformation in the figure valid? Start with the ordinary MHA KV
cache.

## Start with the ordinary MHA KV cache

The intuition above does not yet tell us how large the cache is. Let
$h_t\in\mathbb{R}^{d_{model}}$ be the hidden state of token $t$. Write $n_h$
for the number of heads and $d_h$ for the width of one head; $i$ selects a head,
while $s$ will select a cached key-token position. Ordinary MHA produces a
separate query, key, and value for every head $i$:

$$
q_{t,i}=W_i^Q h_t,\qquad
k_{t,i}=W_i^K h_t,\qquad
v_{t,i}=W_i^V h_t.
$$

For a query at position $t$, head $i$ computes

$$
p_{t,s,i}=\operatorname{softmax}_s
\left(\frac{q_{t,i}^{\mathsf T}k_{s,i}}{\sqrt{d_h}}\right),
\qquad
o_{t,i}=\sum_s p_{t,s,i}v_{s,i}.
$$

The output projection mixes the per-head results. During autoregressive
generation, however, every earlier $k_{s,i}$ and $v_{s,i}$ must remain available.
The cache therefore stores $2n_h d_h$ elements per token per layer: one K slice
and one V slice for every head.

Batch size, sequence length, layer count, and dtype quickly amplify that number.
For a batch of $B$ equal-length sequences of length $S$, a model with
$N_{layer}$ layers, and $b_{elem}$ bytes per element, an ordinary MHA KV cache
occupies

$$
M_{MHA}=2N_{layer}BSn_hd_hb_{elem}\quad\text{bytes}.
$$

For a variable-length batch, replace $BS$ with $\sum_{j=1}^{B}S_j$, where $S_j$
is the length of sequence $j$. As a concrete example, 32 layers, batch size 1,
context length 4096, 32 heads of width 128, and BF16 require 2 GiB for the KV
cache alone. This estimate excludes allocator metadata, model weights, and
other intermediate state.

For a fixed $d_{model}=n_h d_h$, changing only the head count need not change
that total width. The structural cost is that the cache still materializes
separate per-head state.

During dense decode, each new query reads the growing K/V history, so those
reads often become a bottleneck. A smaller representation not only permits a
longer context or larger batch; it also reduces HBM traffic on every decode
step.

Multi-query attention (MQA) reduces the cache by sharing one K/V head across all
query heads. Grouped-query attention shares within groups. Those are useful model
architectures, but MLA takes a different route: retain the expressive per-head
projections while caching a shared low-dimensional source from which they can be
recovered.

### What if we cache one shared state first?

Temporarily ignore RoPE and low-rank compression. Let $h_s$ denote the hidden
state fed to the attention projections at position $s$ in one layer. It is not
a token embedding from the vocabulary. The superscript $C$ denotes the
non-positional content channel; ordinary MHA content keys and values are

$$
k_{s,i}^{C}=W_i^Kh_s,\qquad v_{s,i}^{C}=W_i^Vh_s.
$$

If we cache one shared $h_s$, associativity and the distributivity of a linear
map over a sum let us change the evaluation order:

$$
(q_{t,i}^{C})^{\mathsf T}W_i^Kh_s
=\left((W_i^K)^{\mathsf T}q_{t,i}^{C}\right)^{\mathsf T}h_s,
\qquad
\sum_s p_{t,s,i}W_i^Vh_s
=W_i^V\left(\sum_s p_{t,s,i}h_s\right).
$$

The left side expands K/V for every key and every head. The right side moves
the key projection to the current query and the value projection after the
weighted sum. The values are identical, yet only one $h_s$ must persist. Think
of this as an uncompressed latent-cache thought experiment.

The problem is that $h_s$ is still $d_{model}$ coordinates wide. The cache is
shared across heads, but QK (the query--key dot product) and PV (the
probability--value weighted sum) now operate in that wide space, so this is
usually not a practical design. MLA adds a learned low-rank bottleneck that
shrinks the shared state to $d_c$ coordinates. The original
[DeepSeek-V2 MLA derivation](https://arxiv.org/abs/2405.04434) defines

$$
c_t^{KV}=W^{DKV}h_t,
$$

followed by separate up-projections

$$
k_{t,i}^{C}=W_i^{UK}c_t^{KV},\qquad
v_{t,i}^{C}=W_i^{UV}c_t^{KV}.
$$

Here $c^{KV}\in\mathbb{R}^{d_c}$ is the shared latent state. The thought
experiment above corresponds to $d_c=d_{model}$, $W^{DKV}=I$, $c^{KV}=h$,
$W_i^{UK}=W_i^K$, and $W_i^{UV}=W_i^V$; MLA learns a much narrower $c^{KV}$.

This low-rank step is not a lossless compression of an arbitrary pretrained
MHA. It is a joint low-rank structure imposed during training. If we write the
MLA content path as an equivalent projection from $h_t$, its effective K/V
projection matrices factor as

$$
W_i^K=W_i^{UK}W^{DKV},\qquad
W_i^V=W_i^{UV}W^{DKV}.
$$

Thus every head's content K/V projections share the right factor $W^{DKV}$.
These $W_i^K$ and $W_i^V$ denote the effective MLA projections defined by this
factorization; the equations do not claim that arbitrary MHA projection
matrices can always be factored this way. Under the MLA training constraint,
the model learns what information to preserve in $c^{KV}$.

The construction therefore has two steps: use associativity and linearity to
move per-head projections to the query and output paths so that one shared state
can participate directly in core attention, then use a learned bottleneck to
make that state narrow. The cache stores one $c^{KV}$ per token instead of
expanded $k^C$ and $v^C$ for every head.

### Why does RoPE need a separate positional channel?

Rotary positional embedding (RoPE) rotates a position-carrying subspace of Q
and K. Let $R_u$ denote the RoPE rotation at position $u$. Suppose we applied
RoPE directly to the content query and key. Their dot product would become

$$
\left(R_tq_{t,i}^{C}\right)^{\mathsf T}
\left(R_sW_i^{UK}c_s^{KV}\right)
=
(q_{t,i}^{C})^{\mathsf T}R_t^{\mathsf T}R_sW_i^{UK}c_s^{KV}.
$$

The factor $R_t^{\mathsf T}R_s$ changes with the relative query/key position.
Matrix multiplication can still be reassociated, of course, but $W_i^{UK}$ can
no longer be folded into one projection that the current query can reuse for
every key position. MLA therefore uses a decoupled positional channel: each query head has
$q_{t,i}^{R}$, while all heads share a cached $k_t^R$. The actual MHA-form query
and key are

$$
q_{t,i}=[q_{t,i}^{C};q_{t,i}^{R}],\qquad
k_{s,i}=[k_{s,i}^{C};k_s^R].
$$

The unscaled score therefore separates into

$$
\operatorname{score}_{t,s,i}
=(q_{t,i}^{C})^{\mathsf T}W_i^{UK}c_s^{KV}
+(q_{t,i}^{R})^{\mathsf T}k_s^R.
$$

The cache is now $[c_s^{KV};k_s^R]$, still shared across heads. The positional
channel remains explicit; weight absorption applies to the content channel.

This lets us compare MLA with ordinary MQA precisely. The table counts cached
elements per token per layer; it does not compare model quality or include dtype
and allocator metadata.

| Mechanism | Cached elements per token per layer | Cached state |
| --- | ---: | --- |
| MHA | $2n_hd_h$ | $n_h$ expanded K/V heads |
| GQA | $2n_{kv}d_h$ | $n_{kv}$ expanded K/V heads, $1<n_{kv}<n_h$ |
| MQA | $2d_h$ | one expanded K/V head |
| MLA | $d_c+d_h^R$ | shared latent content and RoPE key |

Here $n_{kv}$ is the number of GQA KV heads, $d_c$ is the width of $c^{KV}$,
and $d_h^R$ is the shared RoPE-key width. A common configuration used later has
$d_c=512$ and $d_h^R=64$, so one cached row has $512+64=576$ coordinates.
Compared only with ordinary MQA at $d_h=128$, 576 is 2.25 times the
$2d_h=256$ cached elements. That ratio
describes this one set of dimensions; it neither makes MLA a form of GQA nor
implies a model-quality ordering.

### Where do the per-head up-projections go?

The uncompressed thought experiment already showed how to move the projections.
Now apply the same regrouping to MLA's actual $W_i^{UK}$ and $W_i^{UV}$. MLA
admits two algebraically equivalent core-attention modes. Here “MQA mode” names
an execution strategy; it does not mean that every MLA model is an ordinary
MQA model.

| MLA execution mode | K/V used by core attention | Where up-projection occurs |
| --- | --- | --- |
| MHA mode | Per-head $[W_i^{UK}c^{KV};k^R]$ and $W_i^{UV}c^{KV}$ | Before core attention |
| MQA mode | Shared $[c^{KV};k^R]$ and shared latent values $c^{KV}$ | Absorbed into the query and output paths |

```{figure} ../img/flashmla_mla_modes.png
:width: 100%
:alt: MHA and MQA execution modes of MLA and the two weight-absorption paths

MLA's MHA mode expands latent KV before core attention. Its MQA mode moves the key
up-projection to the query path and the value up-projection to the output path.
The shared RoPE key stays explicit in both modes.
```

The two modes compute the same result, but their execution costs need not be
the same. MHA mode expands per-head K/V and then uses a narrower core-attention
feature width; that expansion can be amortized when many query rows reuse the
same expanded state. Absorbed MQA mode makes core attention operate on the wider
latent representation, but avoids materializing and rereading per-head K/V for
the history. The best choice therefore depends on sequence stage, sparsity,
shape, data movement, and hardware schedule--not on a rule that prefill must
always use MHA or decode must always use MQA.

This distinction appears in practice as well. Appendix A of the
[DeepSeek-V3.2 report](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)
states that DeepSeek-V3.1-Terminus used MHA mode for training and prefill and
MQA mode for decoding, whereas DSA sparse prefill uses MQA mode.

The picture suggests the answer; the following two identities prove it. Here
“absorption” does not delete a weight or swap matrix order. It uses associativity
to regroup fixed linear maps so that expanded K/V need not be materialized. On
the key side, reassociate the matrix multiplication:

$$
(q_{t,i}^{C})^{\mathsf T}W_i^{UK}c_s^{KV}
=\left((W_i^{UK})^{\mathsf T}q_{t,i}^{C}\right)^{\mathsf T}c_s^{KV}.
$$

Define the absorbed query $q_{t,i}^{A}=(W_i^{UK})^{\mathsf T}q_{t,i}^{C}$.
The content score can now be computed directly against the cached latent.

A two-coordinate example makes the regrouping concrete. Let
$q=(1,2)^{\mathsf T}$, $c=(3,4)^{\mathsf T}$, and
$W=\begin{bmatrix}1&2\\0&1\end{bmatrix}$. Expanding the key first gives
$q^{\mathsf T}(Wc)=19$; transforming the query first gives
$(W^{\mathsf T}q)^{\mathsf T}c=19$. The first form computes $Wc$ for every
cached row, while the second computes $W^{\mathsf T}q$ once for the current
query.

On the value side, linearity gives

$$
\sum_s p_{t,s,i}W_i^{UV}c_s^{KV}
=W_i^{UV}\left(\sum_s p_{t,s,i}c_s^{KV}\right).
$$

Consequently, $W_i^{UV}$ can be composed with the model's output projection.
Neither expanded per-head K nor expanded per-head V needs to be materialized by
the core-attention computation.

The DeepSeek Sparse Attention (DSA) sparse-prefill operator studied here uses
this MQA mode: every selected latent KV entry is shared by all query heads.
Before mapping these widths to the complete tensor contract and dispatch, we
can verify the algebraic equivalence with a small program.

### Can a small program prove the two modes agree?

The following CPU program constructs both executions. It includes a shared RoPE
score term, expands K/V in the MHA path, and absorbs the same matrices in the MQA
path. Float64 makes the equality check sensitive enough to catch a transposed
index or an incorrect contraction.

```python
import math
import torch

torch.manual_seed(0)
Q, K, H = 3, 5, 4
D_CONTENT, D_LATENT, D_VALUE, D_ROPE, D_MODEL = 7, 6, 8, 3, 11

# Per-head queries, one shared latent KV per key token, and shared RoPE keys.
q_content = torch.randn(Q, H, D_CONTENT, dtype=torch.float64)
q_rope = torch.randn(Q, H, D_ROPE, dtype=torch.float64)
c_kv = torch.randn(K, D_LATENT, dtype=torch.float64)
k_rope = torch.randn(K, D_ROPE, dtype=torch.float64)

W_UK = torch.randn(H, D_CONTENT, D_LATENT, dtype=torch.float64)
W_UV = torch.randn(H, D_VALUE, D_LATENT, dtype=torch.float64)
W_O = torch.randn(D_MODEL, H, D_VALUE, dtype=torch.float64)

# MHA mode: explicitly expand a key and value for every head.
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

# MQA mode: move W_UK to Q, attend to c_kv, then move W_UV to output.
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

Notice that the scale remains the scale of the model's semantic QK head, not
$1/\sqrt{D_{latent}}$ merely because the absorbed dot product happens to have
$D_{latent}$ coordinates.

:::{admonition} The query side can also be compressed
:class: note

MLA may factor the query projection through a separate low-rank latent,

$$
c_t^Q=W^{DQ}h_t,\qquad q_t^C=W^{UQ}c_t^Q.
$$

This factorization mainly reduces activation memory during training; it does
not shrink the KV cache further. It also happens before FlashMLA's core
attention: the `q` tensor presented to the sparse-prefill operator is already
projected. We therefore treat $q^C$ as an input and continue along the KV-cache
path.
:::

Weight absorption has answered how one shared KV row can serve many query
heads. The next, independent question is who selects the list of KV rows that
the sparse-prefill operator receives.

## Does the sparse-prefill operator choose the tokens?

Dense attention visits every eligible KV token. DSA first uses a lightweight
*lightning indexer* to score candidate tokens and chooses a top-$k$ set for each
query. The sparse core attention then reads only those latent KV entries. If the
original context length is $L$, this changes the core-attention work from
$O(L^2)$ to $O(Lk)$ for prefill, although the indexer has its own cost.

```{figure} ../img/flashmla_sparse_story.png
:width: 100%
:alt: A lightning indexer selecting token rows before the sparse-prefill attention operator

Selection and attention are separate operators. The indexer produces row
addresses; the sparse-prefill operator gathers those rows and performs the
QK--softmax--PV computation.
```

The sparse-prefill operator studied here does not compute the index scores and
does not run top-k selection. It receives an `indices` tensor. Therefore its
semantic contract is:

1. gather the requested KV rows;
2. mark out-of-range and length-masked positions invalid;
3. compute attention over the remaining rows, including duplicates if the caller
   supplied duplicates;
4. return the output, maximum logit, and log-sum-exp.

There is no causal flag in this interface. A caller that requires causal
attention must produce an index list containing only allowed keys. Sparsity is
not itself a causal mask.

When `topk_length` is present, every query must satisfy
`0 <= topk_length[q] <= topk`. This is a caller precondition.

This prefill interface also has no batch dimension. Each query token supplies
one selected-token list, and all `h_q` query heads share that list. A serving
system must flatten or otherwise map batches before making this call.

These rules now say what the operator accepts and rejects. Next we turn them
into an executable CPU reference, so that the semantics stand on their own
before we discuss the FlashMLA implementation.

## Can we make the sparse contract executable first?

Before examining any GPU specialization, first fix the general shape notation:

| Symbol | Meaning |
| --- | --- |
| `s_q` | number of query rows |
| `s_kv` | number of addressable KV rows |
| `h_q` | query heads in each query row |
| `h_kv` | KV heads in each KV row; this interface requires 1 |
| `d_qk` | query/key width used by QK |
| `d_v` | value and output width, with `d_v <= d_qk` |
| `topk` | index slots supplied for each query row |

The corresponding tensors are `q[s_q,h_q,d_qk]`,
`kv[s_kv,h_kv,d_qk]`, `indices[s_q,1,topk]`, and
`out[s_q,h_q,d_v]`. The optional sink has shape `[h_q]`, the optional
`topk_length` has shape `[s_q]`, and both returned statistics have shape
`[s_q,h_q]`. Only the later regular head-128 specialization fixes
`h_q=128` and `d_v=512`; `h_kv=1` is already part of the general
sparse-prefill contract.

For the absorbed MQA contract, `kv[:, 0, :]` supplies both K and V: all
`d_qk` coordinates participate in QK, while the first `d_v` coordinates are
the latent value. Here `sm_scale` is the model's semantic QK scale.

Invalid indices must be clamped *before* a PyTorch gather and then masked out.
Directly indexing with `-1` would incorrectly select the last row.

The gathered V row for an out-of-range address is also cleared before PV,
because a zero softmax weight does not neutralize a NaN under IEEE arithmetic.

An attention sink is equivalent to adding a logit whose value vector is zero.
It changes only the output denominator:

$$
O_i=\frac{\sum_j e^{x_{ij}-m_i}v_j}
{\sum_j e^{x_{ij}-m_i}+e^{a_i-m_i}}.
$$

Here $x_{ij}$ is head $i$'s ordinary scaled logit for selected KV row $j$,
$v_j$ is that row's value, $m_i$ is a numerical origin chosen as the maximum of
the ordinary logits, and $a_i$ is the per-head sink logit. The sink has no
effect on `max_logits` or the returned log-sum-exp (`lse`). If every selected
position is invalid, the operator instead uses the explicit convention
`out=0`, `max_logits=-inf`, and `lse=+inf`.

The executable oracle follows four stages:

1. validate and safely gather the requested rows;
2. construct the validity predicate and clear only out-of-range V sentinels;
3. compute scaled logits, unnormalized weights, numerator, denominator, and the
   optional sink term; and
4. normalize the output and return the two statistics, including the all-invalid
   convention.

The complete CPU block puts those stages in one place:

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

    # OOB indices use a clamped boundary row only as a safe gather address. Clear
    # their V rows before PV so that 0 * NaN cannot leak from that sentinel row.
    # Only OOB sentinel rows need clearing for the defined-input oracle.
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

    # Avoid (-inf)-(-inf) on rows for which every selected index is invalid.
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

    # FlashMLA's reported LSE excludes the attention sink. Its all-invalid
    # convention is max_logits=-inf, lse=+inf, output=0.
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

# An OOB sentinel must not inherit NaNs from the row used as its safe address.
nan_q = torch.ones(1, 1, 2)
nan_kv = torch.tensor([[[torch.nan, torch.nan]], [[2.0, 3.0]]])
nan_indices = torch.tensor([[[-1, 1]]], dtype=torch.int32)
nan_out, _, _ = sparse_prefill_reference(nan_q, nan_kv, nan_indices, 1.0, 2)
torch.testing.assert_close(nan_out, torch.tensor([[[2.0, 3.0]]]))
print(out.shape, max_logits.shape, lse.shape)
```

The reference makes the sparse-prefill operator's numerical contract executable
for defined inputs and return conventions. Tile partitioning, storage reuse,
and pipeline overlap belong to an implementation, not to the operator
contract.

## Where does sparse prefill sit in the FlashMLA family?

FlashMLA spans both the sequence stage and the selection pattern. Unlike
prefill, *decoding* adds a new query (or a small speculative group) step by step
while reusing the KV cache. The
[official FlashMLA repository](https://github.com/deepseek-ai/FlashMLA)
organizes its operators and their implementations into four broad families:

| Selection | Sequence stage | Representative purpose |
| --- | --- | --- |
| dense | prefill | MHA forward and backward |
| dense | decoding | read an MLA KV cache for newly generated queries |
| token-sparse | prefill | DSA core attention over a selected token list |
| token-sparse | decoding | DSA inference over a selected FP8 KV cache |

FlashMLA is therefore neither synonymous with sparse prefill nor with one
head-128 specialization. This chapter follows the token-sparse prefill cell
because its externally selected, irregular KV row addresses lead directly to
an instructive Blackwell scheduling problem. Sparse decode has a different
paged-cache, scheduling, and reduction contract and remains outside the
chapter's boundary.

The FlashMLA sparse-prefill interface is conceptually

```text
out, max_logits, lse = flash_mla_sparse_fwd(
    q, kv, indices, sm_scale,
    d_v=512,
    attn_sink=attn_sink,       # optional [h_q], float32
    topk_length=topk_length,   # optional [s_q], int32
)
```

The tensor dimensions and explicit arguments have the semantics exercised by
the oracle: `h_q` is the query-head count, `s_q` is the query-row count,
`sm_scale` scales QK scores, `d_v` selects the value/output width, `attn_sink`
optionally adds one zero-valued logit per query head, and `topk_length` limits
each query's valid index prefix. The call returns the normalized output, maximum
scaled logit, and log-sum-exp without the sink.

The TIRx implementation exposes a registry/dispatch bridge with the same
`flash_mla_sparse_fwd` name; it is not a complete replica of the FlashMLA Python
API above. By shape, this bridge selects one of three SM100 phase-1
specializations of the same sparse-prefill computation.

:::{admonition} The TIRx entry specializes `sm_scale`
:class: warning

The FlashMLA interface above accepts `sm_scale` at runtime. The TIRx dispatch
entry exposes a narrower call signature: all three prefill specializations bind
the scale to `1 / sqrt(d_qk)`, and the launch application binary interface
(ABI) has no scale argument.

Passing `sm_scale=...` through the `**kwargs` wrappers is silently ignored. The
B200 examples below therefore validate the computation only at
`sm_scale = 1 / sqrt(d_qk)`; they do not demonstrate runtime-scale parity with
the complete FlashMLA interface.

A model whose semantic QK scale differs must expose or specialize the correct
value. Weight absorption does not justify changing that scale.
:::

Two more boundary conditions belong specifically to this TIRx path. Its
prefill specializations use `topk_length` to decide how many selected-row tiles
to visit but do not clip it, so a value above `topk` can step beyond the logical
`indices` storage. They also do not promise to sanitize NaNs in an in-range KV
row that lies beyond `topk_length`; callers should keep such length-masked rows
finite. Within these preconditions and at the specialized scale, every
dispatched implementation must reproduce the reference contract.

## Which Blackwell case are we studying?

This regular head-128 case is a useful bridge from the preceding FlashAttention
chapter. It retains the familiar QK--softmax--PV chain, then adds irregular
gather, absorbed latent KV, and cooperative thread-block ownership.

### Which shapes reach the regular head-128 path?

The algorithmic discussion only needed “one shared latent KV.” Now that we are
entering a concrete implementation, we can state every tensor shape. The
regular head-128 module has this shape-specialized signature:

| Tensor | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `q` | `[s_q, 128, d_qk]` | BF16 | absorbed queries |
| `kv` | `[s_kv, 1, d_qk]` | BF16 | shared latent/positional KV rows |
| `indices` | `[s_q, 1, topk]` | int32 | direct KV row indices |
| `attn_sink` | `[128]` | FP32 | optional per-head sink logits |
| `topk_length` | `[s_q]` | int32 | optional valid prefix length; every entry is in `[0, topk]` |
| `out` | `[s_q, 128, 512]` | BF16 | sparse-attention result |
| `max_logits` | `[s_q, 128]` | FP32 | maximum scaled logit |
| `lse` | `[s_q, 128]` | FP32 | natural-log sum-exp, without sink |

The 128 is the query-head count from the opening question, while the 1 in `kv`
means that every head shares the same KV row. A common $d_{qk}=576$ case combines
512 latent-content coordinates with 64 RoPE coordinates; `d_v=512` is the
latent-value width. These are shapes of this absorbed MQA representation, not a
universal MLA contract.

To give the earlier qualitative cost comparison one numerical anchor, the
equivalent MHA representation of this MLA layer has QK feature width
$128+64=192$ and value/output width 128. The absorbed MQA representation used
here has widths $512+64=576$ and 512. A rough count of multiply-add
coordinates per query--key pair is therefore $192+128=320$ versus
$576+512=1088$, about 3.4 times as many for the absorbed representation.
This is only an arithmetic-width intuition, not a prediction of kernel runtime.

Keep one concrete shape in mind for the rest of the chapter: `s_q=1`,
`s_kv=8192`, `h_q=128`, `h_kv=1`, `d_qk=576`, `d_v=512`, and `topk=2048`. It is
one query row with 128 query heads sharing 2048 selected-index slots. Those
slots may contain duplicate or out-of-range addresses. Without a shorter
`topk_length`, the physical schedule visits $N=16$ tiles of 128 slots. When
`topk_length` is present, it visits
`max(ceil(topk_length / 128), 1)` tiles.

Before assigning hardware roles, reduce those selected-index tiles to six
semantic steps. To match the source notation used later, $L$ denotes raw QK
logits, $W$ denotes BF16 unnormalized exponential weights, `mi` is the
online-softmax exponent origin, `li` is the denominator accumulated relative to
that origin, and $\widetilde O$ is the accumulated output before division by
the denominator:

```text
for each 128-slot selected-index tile:
    1. gather the tile's K rows
    2. gather its V rows and build the validity mask
    3. QK: 128 query heads x 128 selected-index slots -> logits L
    4. update masked online softmax, producing W, origin mi, and denominator li
    5. rescale the running state when needed, then accumulate O~ += W @ V
after all selected-index tiles:
    6. normalize O~ by li + sink; store out, max_logits, and lse
```

In the concrete example, the first five steps repeat 16 times before the sixth
step runs. With the arithmetic thread fixed, we can now ask how TIRx assigns
the hardware roles.

We now focus on TIRx's regular head-128 implementation in
[`sparse_prefill_head128_phase1.py`](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py).
TIRx extends TVM 0.26's TIR Python DSL: `T` denotes the TIR script namespace,
while `Tx` contains GPU-kernel helpers. Despite the internal `phase1` name,
this specialization produces the complete `(out, max_logits, lse)` result in
one kernel; it is not a partial split-KV output awaiting a combine kernel.

Three execution levels recur below. A cooperative thread array (CTA) is one
CUDA thread block. Two adjacent CTAs form a cluster that can participate in
CTA-group tensor-core operations. A warpgroup is four warps, or 128 threads,
assigned one specialized role. Here one two-CTA cluster owns one query row.

In the mathematical introduction, $p$ denoted a normalized softmax
probability. In the source, however, `tmem_p` and the register variable `p`
hold the raw QK logits already named $L$. The source's `s_frag` and
`s_smem_gemm` hold the unnormalized exponential weights already named $W$.
Only the epilogue divides the accumulated output by `li` plus the optional sink
term.

All shorter TIRx code blocks below are contextual excerpts from the linked
implementation, not standalone programs. The complete module is compiled and
numerically verified in the final section. Blocks intended to run independently
are called out explicitly.

With the tensor contract fixed, the following constants determine one tile's
execution:

```python
B_H = 128
B_TOPK = 128
D_V = 512
NUM_BUFS = 2
NUM_THREADS = 512
D_TQ = 384
```

The names map directly to the work we will trace: `B_H` is the 128-head logical
tile, `B_TOPK` is the 128 selected-index slots processed per streaming tile,
`D_V` is the value/output width, and `D_TQ` is the 384-coordinate Q suffix moved
to dedicated on-chip storage. `NUM_THREADS=512` gives each CTA four
warpgroups. `NUM_BUFS=2` names two synchronization slots (and two small
validity-mask slots), not two full K/V data stages; the residency and pipeline
sections will make that distinction concrete.

The regular head-128 specialization accepts `d_qk` 512 or 576, requires
`h_kv=1`, `d_v=512`, and requires `topk` to be a positive multiple of 128. The
TIRx dispatch entry makes one additional choice for 128 heads: `d_qk=512`
with `topk<=1280` selects the small-top-k head-128 specialization; other
supported head-128 shapes select this regular head-128 specialization. Head-64
shapes select the head-64 specialization.

The positive-`topk` condition is a required caller precondition. The TIRx
dispatch entry rejects `topk<=0`, but the per-specialization `_cfg().validate()`
methods check divisibility without checking positivity. A direct import of one
specialization must therefore still reject or avoid nonpositive `topk`;
acceptance by that local validator does not make such a launch valid.

After completing the repository setup in the final section, this dispatch can
be inspected without launching a GPU kernel:

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

The dispatch itself is documented in
[`flash_mla_sparse_fwd.py`](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/flash_mla_sparse_fwd.py#L66-L120).
Keeping dispatch separate from the device schedule prevents a tutorial from
mistaking one specialization for the entire operator.

Dispatch has selected the implementation. The next question is how two thread
blocks cooperate on the six steps without duplicating the whole tile.

### Why does one query row need two CTAs?

The difficulty lies in steps 3 and 5 of the tile skeleton. QK must form every
pair of 128 query heads and 128 selected tokens; PV then contracts the token
axis to produce 512 value coordinates. The implementation therefore lets two
CTAs form one logical tile and changes their partition axis between QK and PV.
Start with that ownership map before reading the launch code.

```{figure} ../img/flashmla_cta_ownership.png
:width: 100%
:alt: Two-CTA ownership of query heads, selected K rows, and V feature columns

For each query row, the CTA pair changes its logical partition between QK and
PV. Each CTA ultimately writes 64 complete output-head vectors, each with 512
coordinates.
```

The pair divides three different axes in three different ways:

| Resource within a 128-token top-k tile | CTA 0 | CTA 1 |
| --- | --- | --- |
| query/output head ownership | heads 0--63 | heads 64--127 |
| K-row gather ownership | selected tokens 0--63 | selected tokens 64--127 |
| V-feature gather ownership | value columns 0--255 | value columns 256--511 |

This is why a 2-CTA tensor-core operation is more than “two CTAs doing half the
same loop.” QK needs the cross-product of 128 heads and 128 selected tokens; PV
then contracts those tokens into 512 value coordinates. The partition rotates
between the two GEMMs. Collective `cta_group=2` MMA, paired on-chip layouts, and
cross-CTA synchronization together form the logical tile.

Now return to the source to verify the launch topology. The grid contains
`2 * s_q` CTAs and clusters adjacent CTAs in pairs:

```python
block_idx = T.cta_id([2 * s_q])
T.cta_id_in_cluster([2])
cta_idx: T.let = block_idx % 2
s_q_idx: T.let = block_idx // 2
thread_idx = T.thread_id([512])
T.warpgroup_id([4])
```

One cluster therefore owns one query row, and each CTA has four warpgroups. The
same division appears in the data indexing: Q is chunked by `cta_idx`, the K
producer selects the `cta_idx` half of every top-k block, and the V producer
starts at `cta_idx * 256`.

## Where do the tiles live?

The ownership map says *who* computes each piece; residency explains where a
piece waits between producers and consumers. Three storage terms are enough to
read the next figure:

- global memory (GMEM) holds the input and output tensors;
- shared memory (SMEM) is the ordinary on-chip scratchpad visible to a CTA;
- Blackwell tensor memory (TMEM) is a separate on-chip space near the tensor
  cores, used for operands and large accumulators.

```{figure} ../img/flashmla_dataflow.png
:width: 100%
:alt: Data residency and lifetime reuse across global memory, shared memory, tensor memory, and WG0 registers during QK, softmax, PV, and the epilogue

Q is split between an SMEM prefix and a TMEM suffix. Gathered K/V enter SMEM,
raw QK logits and the output accumulate in TMEM, and unnormalized softmax
weights cross back through SMEM for PV.
```

The arrows in the figure correspond to two Blackwell mechanisms. The Tensor
Memory Accelerator (TMA) moves data asynchronously between GMEM and SMEM and
provides the sparse `gather4` path used here. The `tcgen05` tensor-core
instruction family reads operands from SMEM or TMEM and keeps its large
accumulators in TMEM. TMEM therefore complements rather than replaces SMEM:
TMA gathers land in SMEM, and the softmax warpgroup materializes BF16
unnormalized weights there for PV.

The source abbreviates tensor-core operand residency as **SS** when both matrix
operands come from SMEM and **TS** when the first comes from TMEM and the second
from SMEM. The Q prefix follows the SS path; its suffix follows the TS path.

For one CTA, the important logical views are:

| Storage | Logical tile | Lifetime and purpose |
| --- | --- | --- |
| SMEM `q_full` | `64 x d_qk` BF16 | Q prologue; its prefix remains for SS QK |
| TMEM `q_tmem` | `64 x 384` BF16 | suffix of Q used by TS QK |
| SMEM `k_smem` | `64 x d_qk` BF16 | this CTA's gathered half of a 128-row K tile |
| TMEM `tmem_p` | `64 x 128` FP32 logical view | raw QK logits $L$ consumed by softmax |
| SMEM `s_smem_gemm` | `64 x 128` BF16 | unnormalized exponential weights $W$ for PV |
| SMEM `v_smem_gemm` | `128 x 256` BF16 logical view | rearranged view of `v_smem`: all tile rows, this CTA's V columns |
| TMEM `o_tmem` | `64 x 512` FP32 logical view | running unnormalized output |
| SMEM `o_smem` | `64 x 512` BF16 | epilogue staging before TMA store |

These are logical views; CTA-group TMEM layouts and rearrangements give the MMA
and load/store instructions their physical lane mapping. The source allocates
one 512-column CTA-group TMEM pool, then carves out O, raw-logit, and Q views.

The source object named `SMEMPool` describes the corresponding shared-memory
allocation. SMEM is aggressively aliased: `q_full`, the gathered K/V region,
and the output epilogue reuse a union-like base when their lifetimes permit.

After the final 384 Q columns have moved to TMEM, only the
$d_{sq}=d_{qk}-384$ prefix must stay live for the first QK part. For
`d_qk=512`, $d_{sq}=128$; for 576, it is 192.

The allocation plan is visible in
[`sparse_prefill_head128_phase1.py` lines 302--365](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L302-L365).

A **completion barrier** is a small hardware state object that records when an
asynchronous producer has finished. Its phase bit distinguishes successive
uses of the same barrier slot; later sections will spell out the complete
producer--consumer protocol.

One important caveat is that `NUM_BUFS = 2` does **not** allocate two complete K,
V, $L$, or $W$ tiles. Those arrays have no stage axis. `NUM_BUFS` drives the
two-slot barrier/phase ring and also gives the small packed-validity mask two
slots; completion barriers make it safe to overwrite parts of the single large
tile storage. Calling this layout “double-buffered K/V” would describe storage
that does not exist.

## Which warpgroup owns each transition?

The four warpgroups do different jobs rather than advancing in lockstep:

| Warpgroup | Warps | Responsibility |
| --- | --- | --- |
| WG0 | 0--3 | load raw logits $L$ from TMEM, mask, online softmax, write weights $W$, rescale O, epilogue |
| WG1 | 4--7 | load index fragments and issue gather4 TMA for K |
| WG2 | 8--11 | load index fragments and issue gather4 TMA for V |
| WG3 | 12--15 | warp 12 of CTA 0 issues CTA-group QK/PV MMA; warp 13 in each CTA builds validity masks |

The remaining WG3 warps do not acquire another hidden stage. This asymmetric
role assignment is deliberate: one elected lane can issue asynchronous MMA for
the CTA pair, while many lanes are useful for exponentiation, row reductions,
packing, and epilogue conversion.

:::{admonition} Why the register limits differ
:class: note

The register budgets match the roles. WG0 raises its limit to 144 registers and
WG3 to 168; producer groups lower theirs to 96. The TIRx API spells these calls
as `T.ptx.setmaxnreg(True, ...)` for an increase and
`T.ptx.setmaxnreg(False, ...)` for a decrease.
:::

Before following a gather, keep one minimal synchronization model in mind. A
producer waits until a tile's storage is **free**, writes or asynchronously
fills the tile, and signals **ready**. The consumer waits for ready, uses the
tile, and signals free when the storage may be overwritten. These barriers
transfer ownership of storage; they do not imply that two complete data tiles
exist.

### How do irregular rows become regular tiles?

Sparse row addresses destroy the contiguous 2-D copy pattern used by dense
attention. WG1 and WG2 use explicit TMA `gather4`: one issue supplies exactly
four row coordinates, so a warp can bring noncontiguous KV rows into a regular
SMEM tile.

The next contextual excerpt answers which addresses one `gather4` issue reads
and which barrier receives its completion. Its index names and slices are kept
exactly as they appear in the linked implementation:

- `gather4=[...]` supplies the four KV source-row coordinates for this issue;
- `cur_buf = k % NUM_BUFS` selects the current slot in the two-slot barrier ring;
  and
- `bar` is the ready-barrier array passed to this copy helper. `leader_mbar`
  selects the CTA-pair leader's slot, where TMA reports asynchronous completion.

The remaining names describe the surrounding layouts; they are not additional
stages hidden from the diagram.

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

Gathering and validity are related but separate. Warp 13 loads eight indices per
active lane and calls `pack_valid_mask8`. Bit $i$ is one only if

$$
0\leq\text{index}_i<s_{kv}
\quad\text{and}\quad
\text{absolute_topk_position}_i<\text{topk_length}.
$$

WG0 waits for the packed mask and replaces the corresponding raw logit $L$ with
negative infinity before taking a maximum or exponent. Thus the packed mask,
not an assumed zero value from the gather, decides whether a finite KV row
participates in attention. Masking must happen before online-softmax state is
updated.

The implementation source separates these roles cleanly: the
[K producer is at lines 608--676](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L608-L676),
the [V producer is at lines 679--729](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L679-L729),
and [validity packing is at lines 841--865](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L841-L865).

## Why is Q split between SMEM and TMEM?

The gather path has turned irregular K rows into a regular SMEM tile. Q must now
meet that tile without occupying all of the aliased SMEM workspace. Using the
SS/TS terms defined with the residency map, the QK dot product is split at
$d_{sq}=d_{qk}-384$:

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

The first operation is SS: both Q and K operands are described from SMEM. The
second is TS: Q's 384-column suffix comes from TMEM while K remains in SMEM. Both
write the same FP32 raw-logit accumulator (`tmem_p` in the source); the first
clears it and the second accumulates. Splitting Q this way preserves only the
smaller Q prefix in SMEM, making the union allocation possible without giving
up the TS path for the larger suffix.

After softmax, PV is an SS GEMM: BF16 $W$ and V are both in SMEM, while the FP32 O
accumulator stays in TMEM. The kernel splits the V rows and output columns into
two halves each; the four combinations collectively update all 512 value
coordinates.

## How does online softmax avoid needless O rescaling?

QK has produced raw logits $L$; softmax must now turn each tile into $W$ while
preserving state from earlier selected tiles. In the running shape, this is why
the state must be merged across $N=16$ tiles. Let $x$ be one raw QK dot product.
The kernel first places it in the base-2 exponent domain:

$$
r=x\cdot\text{semantic\_QK\_scale}\cdot\log_2(e).
$$

The TIRx specialization binds the multiplier named `sm_scale_div_log2` to
`(1 / sqrt(d_qk)) * log2(e)`. Despite that source name, $r$ is simply the
model-scaled score expressed in base-2 units so that the kernel can use `exp2`.

For a stream of such score tiles, online softmax stores a base-2 row origin $m$,
denominator $\ell$, and unnormalized output $\widetilde O$. When the next tile
contains scores $r$,

$$
m'=\max(m,\max r),\qquad
\alpha=2^{m-m'},
$$

$$
\ell'=\alpha\ell+\sum_j2^{r_j-m'},\qquad
\widetilde O'=\alpha\widetilde O+\sum_j2^{r_j-m'}v_j.
$$

Rescaling the full 512-coordinate O tile whenever the row maximum increases
would be expensive. Before reading the lazy-threshold excerpt, map its state back
to the recurrence:

- `cur_pi_max` is the current tile's maximum in the base-2 exponent domain;
- `mi` is the retained numerical origin $m$;
- `real_mi` is the exact maximum retained for the reported `max_logits`;
- `li` is the running denominator $\ell$; and
- `attn_sink_log2` is the optional sink logit in the same base-2 domain.

The head-128 kernel then makes one warp-uniform decision:

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

If the new tile maximum is at most 6 base-2 units above the stored origin, the
kernel keeps the old origin. New exponentials may then be as large as $2^6=64$,
but the accumulated O does not need a rescale. Once the difference exceeds 6,
it rebases and rescales both $\ell$ and, when it already exists, O. The warp-wide
`any_sync` keeps the decision uniform for the participating rows.

Because `real_mi` is maintained separately, the optimization does not change
`max_logits`. At the end, the two 64-token contributions to each logical row are
combined, and the kernel emits

$$
\mathrm{lse}=m\ln 2+\ln\ell.
$$

The optional attention sink changes the final output scale to

```python
output_scale: T.float32 = T.cuda.fdividef(
    T.float32(1.0), li + T.ptx.exp2(attn_sink_log2 - mi)
)
```

but deliberately leaves the reported LSE untouched. All-invalid rows are
special-cased to output zero with `max_logits=-inf` and `lse=+inf`, matching the
executable reference.

## How can the stages overlap without racing?

The ready/free ownership model introduced before gather now expands into a
pipeline. QK must finish before softmax consumes its logits, and PV must wait
until softmax has produced its weights. Meanwhile, the gather warpgroups move
forward whenever an in-place K or V segment becomes reusable.

```{figure} ../img/flashmla_pipeline_stages.png
:width: 100%
:alt: Sparse-prefill pipeline fill, steady state, and drain, including the overlap among QK, softmax, and PV for adjacent tiles

The fill iteration issues QK(0) without a previous PV, while the drain iteration
issues PV($N-1$) without a new QK before the final epilogue. In steady state,
softmax($k-1$) may overlap QK($k$). The sole MMA issuer then issues PV($k-1$)
after QK($k$); once QK($k$) completes, softmax($k$) may overlap that PV.
Producer warpgroups gather the next safe K/V segments around this serial issuer
order. For the running shape, $N=16$.
```

At the coarsest level, four ownership handoffs repeat for each tile:

1. WG1/WG2 publish gathered K/V segments, while warp 13 publishes validity;
2. WG3 completes QK, publishes $L$ to WG0, and returns consumed K segments;
3. WG0 masks $L$, updates online softmax, publishes $W$ to WG3, and releases the
   raw-logit and validity storage; and
4. WG3 completes PV, returning V storage to WG2 and making O safe for WG0 to
   rescale or finalize.

An **mbarrier** is the hardware completion object behind these handoffs. It
tracks expected arrivals or TMA bytes and carries a phase, so a wait identifies
the intended reuse of a slot. This four-step view gives the causal chain; the
exact table below splits K and V into parts that can be released early.

The kernel initializes its mbarriers in warp 0, performs a cluster sync, launches
the Q prologue, allocates CTA-group TMEM, and then enters specialized loops.

TMA completion for CTA-group gathers is routed to a named leader barrier, so
issues from the pair contribute to one expected byte count.

The main barrier edges are easier to read as ownership transfers:

| Barrier | Producer to consumer | Storage protected |
| --- | --- | --- |
| `bar_k_part0_ready` | WG1 to WG3 | K prefix for SS QK |
| `bar_qk_part_done` | WG3 to WG1 | permission to overwrite K prefix after SS QK completion |
| `bar_k_part1_ready` | WG1 to WG3 | K suffix for TS QK |
| `bar_qk_done` | WG3 to WG0 and WG1 | raw logits $L$ ready; K suffix reusable after QK completion |
| `bar_p_free` | WG0 to WG3 | TMEM raw-logit tile consumed before next overwrite |
| `bar_k_valid_ready/free` | warp 13 to/from WG0 | packed validity mask |
| `bar_so_ready` | WG0 to WG3 | BF16 weights $W$ ready for PV |
| `bar_v_part0_ready` / `bar_sv_part_done` | WG2 to/from WG3 | first V half |
| `bar_v_part1_ready` / `bar_sv_done` | WG2 to WG3, then WG3 to WG2 and WG0 | second V half; PV/O completion before V reuse, O rescale, or epilogue |

The ring index is

```python
cur_buf = k % 2
cur_phase = (k // 2) & 1
```

so a reused barrier slot can distinguish a new arrival from one made two
iterations earlier. Again, this is a two-slot *barrier/phase ring*, not two full
KV tiles resident at once.

`bar_qk_part_done` allows the producer to replace K's prefix before its suffix
is reusable. The two `bar_sv_*` edges do the analogous job for V.

```{figure} ../img/flashmla_pipeline.png
:width: 100%
:alt: Detailed sparse-prefill pipeline showing serial QK and PV issue, part-wise K and V reuse, the mask-slot ring, and the WG0 handoff

This detailed view names the part-level reuse edges and the mask-slot ring.
Barrier phases protect in-place storage reuse rather than selecting separate
full-tile buffers.
```

The memory model has one more distinction. Ordinary thread loads and stores see
SMEM through the **generic proxy**; TMA and tensor-core asynchronous accesses use
an **asynchronous proxy**. A barrier reports completion, but completion alone
does not establish the required visibility and ordering between those proxies.

Two kinds of fences therefore appear around the barrier edges.
`T.ptx.tcgen05.fence.*` orders TMEM accesses relative to thread-visible work,
while `T.ptx.fence.proxy_async("shared::cta")` establishes cross-proxy ordering
between generic and asynchronous accesses to SMEM.

Here the proxy fence is needed both when generic stores of $W$ or the epilogue
tile precede tcgen05/TMA asynchronous reads, and after an asynchronous SMEM read
completes before generic code overwrites aliased storage.

Thus an mbarrier communicates completion and an ownership handoff, whereas the
proxy fence orders memory effects across proxies. Neither is a substitute for
the other.

## How do we compile and verify the regular head-128 implementation?

The regular head-128 specialization targets compute capability 10, and its
TMA/tcgen05 forms require an SM100-class GPU. Use B200, CUDA 12.9 or newer, and
the dependencies installed below.

First install a CUDA-enabled PyTorch build that supports B200 using the
[official PyTorch selector](https://pytorch.org/get-started/locally/). The
`tirx-kernels` repository imports PyTorch but does not declare it as a package
dependency.

Install TVM and `tirx-kernels` as follows:

```bash
python -m pip install "apache-tvm==0.26.0" cuda-bindings
git clone https://github.com/mlc-ai/tirx-kernels.git
cd tirx-kernels
git checkout 5be39749e7dfd2c4bdae9b4d396f8ec35af07126
pip install -e .
```

A one-row smoke test still exercises the full 128-head, top-k-2048 kernel while
keeping reference time modest:

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

`run_test` does more than compile. It allocates randomized BF16 Q/KV, random
indices, runs the generated kernel, evaluates the FP32 PyTorch oracle one query
row at a time, and checks output, maximum logits, and LSE with explicit
tolerances. This is the right default verification path; compiling generated
PTX alone cannot find a wrong head partition or a missing validity bit.

The `tirx-kernels` CLI can run the registered regular-head128 configuration:

```bash
python -m tirx_kernels.test \
  --kernel sparse_flashmla_prefill_head128_phase1 \
  --config bench_regular_dqk576_hq128_s4096_kv8192_topk2048
```

Useful negative tests are just as important. Set `inject_invalid_indices=True`
to cover negative and too-large row IDs, and `have_topk_length=True` to exercise
the position predicate. Test an all-invalid row and confirm the documented
zero/-infinity/+infinity convention. Finally, call the TIRx dispatch entry for
head-64 and small-top-k shapes so that a successful regular-head128 run is not
mistaken for coverage of every prefill specialization. Even that broader
dispatch check does not establish parity with the complete FlashMLA interface.

## Which invariants belong to the operator and the specialization?

1. **The cache invariant.** One `h_kv=1` latent KV row can serve multiple query
   heads because key up-projection is absorbed into each query and value
   up-projection is moved after core attention. The RoPE channel stays explicit,
   and the QK scale remains the model's semantic scale.

2. **The sparse-contract invariant.** Selection happens before this operator.
   `indices` supplies the rows; duplicates remain duplicates; causal legality is
   the caller's responsibility; and every `topk_length` lies in `[0, topk]`.
   Sink and all-invalid conventions must remain identical in optimized and
   reference paths.

3. **The ownership invariant.** One two-CTA cluster owns one query row. The pair
   partitions Q/output heads, selected K rows, and V features on different axes,
   while CTA 0 warp 12 is the sole CTA-group MMA issuer.

4. **The residency invariant.** $L$ means raw FP32 logits and $W$ means BF16
   unnormalized exponential weights. Large K, V, $L$, and $W$ workspaces are
   single in-place tiles, while `NUM_BUFS=2` drives a barrier/phase ring. Among
   data buffers, only the small packed-validity mask has two physical slots.

5. **The handoff invariant.** Ready/done barriers transfer ownership of each
   reusable segment, and proxy fences order generic and asynchronous SMEM
   accesses. The sole issuer orders QK($k$) before PV($k-1$), while the threshold-6
   optimization preserves the reported-maximum semantics and the online LSE
   recurrence.

The first two invariants define the sparse-prefill operator semantics. The last
three define the regular head-128 specialization's schedule contract. Another
specialization selected by the dispatch bridge may change tile sizes, register
budgets, ownership, or barrier topology, but it must preserve the operator
semantics while defining its own schedule contract explicitly.

## What should you test next?

The regular head-128 specialization is one point in a dispatch space. The
TIRx dispatch tree also contains a head-64 phase-1 specialization and a head-128
`d_qk=512` small-top-k specialization. Their different schedules are evidence
that sparse attention should be dispatched by tile economics, not forced
through one universal template.

1. **Reproduce weight absorption.** Add a causal mask to the runnable absorption
   proof. Confirm that MHA and MQA modes still agree, then intentionally change
   the absorbed path's scale to $1/\sqrt{D_{latent}}$ and measure the error.

2. **Stress sparse validity.** Extend `sparse_prefill_reference` with an
   all-invalid query, duplicated indices, `topk_length=0`, and sink values of
   both infinities. Write down the expected `(out, max_logits, lse)` for each.

3. **Trace ownership.** For one top-k tile, label every dimension of Q, K, $L$, $W$,
   V, and O with `(CTA, local row, local column)`. Show where a logical row needs
   information owned by the other CTA.

4. **Audit residency.** Starting at the `SMEMPool` allocation, draw every alias
   interval. Verify why the $d_{sq}$ Q prefix remains live while the 384-column
   suffix can move to TMEM, and identify the barrier that ends each reuse hazard.

5. **Measure the threshold.** Instrument how often `should_scale_o` is true for
   random and adversarial logits. Compare threshold 6 with always-rebase and
   never-rebase versions in both numerical error and TMEM O traffic.

6. **Compare dispatches.** Use `select_kernel` on head counts 64 and 128,
   `d_qk` 512 and 576, and top-k values around 1280. Predict the selected module
   before running it, then inspect which constraints belong to the front door and
   which are enforced by an individual specialization.

7. **Read the generated program.** Compile the smoke shape, inspect the emitted
   PTX for `tcgen05` MMA, TMA gather, mbarrier, and proxy-fence instructions, and
   map each one back to a TIRx line. Then run the numerical check again: source,
   generated code, and observed values are three complementary kinds of proof.

The central lesson is broader than FlashMLA. A high-performance irregular
operator often regularizes work in stages: an indexer creates sparse addresses,
TMA gathers those addresses into dense tiles, tensor cores consume the tiles,
and explicit barriers protect aggressive storage reuse. Understanding the
algorithm, dispatch contract, ownership map, and memory protocol together is
what turns a fast kernel from an opaque artifact into an explainable program.
