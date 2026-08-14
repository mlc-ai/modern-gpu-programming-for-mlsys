(chap_flashmla)=
# FlashMLA

:::{admonition} Overview
:class: overview

- Start from an ordinary MHA KV cache and derive why MLA stores one shared
  compressed state per token, including where the head-specific K/V
  transformations move.
- Define FlashMLA sparse attention: an external indexer selects KV rows, then
  the kernel performs QK, softmax, and PV over those rows. An executable
  reference fixes its numerical behavior and edge cases.
- Use a concrete implementation to understand how sparse attention maps to
  Blackwell, then compile and verify it on B200.
:::

A language model normally predicts one new token at a time. At every step,
attention compares the query at the current position with the keys and values
of all earlier tokens, then uses the attention weights to gather information
from that history. Without saving the historical K/V states, the model would
recompute K/V for the entire prefix at every step. A **KV cache** avoids that
repeated work by storing those states once and letting later attention steps
read them directly. Generative models normally use causal attention, so the K/V
at position $s$ depend only on tokens up to position $s$ and remain unchanged
when later tokens are appended. Prefill computes the prompt tokens in parallel
and fills the cache; decode then reads that cache step by step and appends the
new token's K/V.

In ordinary multi-head attention (MHA), every attention head has its own K and
V, so the KV cache must store every head's K/V for every processed token. As the
context grows, the cache consumes more GPU memory, and attention must read an
ever-longer K/V history for every newly generated token. It can therefore
become both a capacity and a memory-bandwidth bottleneck. **Multi-head Latent
Attention (MLA)** reduces these costs by changing the cache representation: it
compresses the K/V-related information for each token into one state shared by
all heads. During attention, every head still applies its own transformations
and therefore retains distinct attention behavior.

FlashMLA is DeepSeek's library of optimized GPU kernels for MLA. This chapter
studies one sparse-attention forward kernel for Blackwell that is used during
prefill. While the prompt tokens are processed in parallel, an external indexer
first selects the history positions that each query token should attend to. The
kernel loads the corresponding KV rows, then performs QK, softmax, and PV. Compared with
dense attention, it reduces only the history positions that participate in the
calculation; the main attention chain remains unchanged.

In the kernel studied here, each query token corresponds to 128 query heads,
while every selected history token has one KV-cache row shared by those heads.
But if 128 heads read the same row, how can they still produce different
results? The cached compressed state is shared, but every head retains its own
transformations on the query and output paths. To see how those transformations
move, start with the KV cache of ordinary MHA.

## Ordinary MHA KV-cache cost

Ordinary MHA has $n_h$ heads of width $d_h$, each with its own query, key, and
value projections. Fix one head and write its projection matrices as $W^Q$,
$W^K$, and $W^V$. Within one layer, write the current token vector entering the
attention projections as $h_t$ and the input vector for the earlier token at
position $s$ as $h_s$. Then

$$
q=W^Q h_t,\qquad
k_s=W^K h_s,\qquad
v_s=W^V h_s.
$$

This head computes

$$
p_s=\operatorname{softmax}_s
\left(\frac{q^{\mathsf T}k_s}{\sqrt{d_h}}\right),
\qquad
o=\sum_s p_s v_s.
$$

During autoregressive generation, the current query can be discarded after use,
but later queries repeatedly read the historical $k_s$ and $v_s$.
Each layer therefore caches $2n_hd_h$ elements per token: one K and one V for
every head.

The total grows linearly with batch size, context length, and layer count. For
example, 32 layers, context length 4096, 32 heads of width 128, batch size 1,
and BF16 require about 2 GB of KV cache. Generating every new token
also rereads this growing history, so KV cache creates both memory-capacity and
bandwidth pressure.

One direct way to reduce this cost is to share more state across query heads.
Multi-query attention (MQA) shares one K/V pair across all query heads, while
grouped-query attention (GQA) shares within groups. MLA uses a different form of
sharing: every head retains its own projection, but the cache stores only one
shared low-dimensional state.

### From a shared state to a low-dimensional latent cache

Start with the non-positional content path and write the current query content
as $q^C$. Ordinary MHA first forms $k_s^C=W^Kh_s$, then computes the content
score in QK. Associativity lets us rewrite that step as

$$
\mathrm{QK}_s
=(q^{C})^{\mathsf T}k_s^C
=(q^{C})^{\mathsf T}W^Kh_s
=\left((W^K)^{\mathsf T}q^{C}\right)^{\mathsf T}h_s.
$$

PV likewise forms $v_s=W^Vh_s$ before taking the attention-weighted sum with
weights $p_s$. Linearity gives

$$
\mathrm{PV}
=\sum_s p_s v_s
=\sum_s p_s W^Vh_s
=W^V\left(\sum_s p_s h_s\right).
$$

The first identity moves the key projection to the current query; the second
moves the value projection after the weighted sum. The numerical results stay
the same, while the cache only needs to retain one $h_s$. Think of this as an
uncompressed latent-cache thought experiment.

$h_s$ is still $d_{model}$ coordinates wide. Caching it shares state across
heads, but QK scoring and PV aggregation still operate in that wide space.
Actual MLA uses two steps.

First, every head shares a matrix $D$ that compresses $h_s$ into a
$d_c$-dimensional vector $c_s$. For the content path, the KV cache stores only
this $c_s$:

$$
c_s=Dh_s,\qquad c_s\in\mathbb{R}^{d_c}.
$$

Second, each head uses its own matrices $U_K$ and $U_V$ to obtain that head's
content key and value from $c_s$:

$$
k_s^C=U_Kc_s,\qquad
v_s=U_Vc_s.
$$

$D$, $U_K$, and $U_V$ are learned together during model training. Because $d_c$
is much smaller than $d_{model}$, the content state cached for each token is
correspondingly smaller.

### RoPE and the separate positional channel

The preceding content-only QK path has no position-dependent transform on the
query side, so one transformed query can be reused for every cached row. RoPE
rotates Q/K according to token position; let $R_t$ and $R_s$ be the rotations
at positions $t$ and $s$. Applying it directly to the content query and content
key would give

$$
\left(R_tq^{C}\right)^{\mathsf T}
\left(R_sU_Kc_s\right)
=
\left(U_K^{\mathsf T}R_s^{\mathsf T}R_tq^{C}\right)^{\mathsf T}c_s.
$$

The query position $t$ is fixed, but the historical position $s$ changes from
one cached row to another. The transformed query in parentheses therefore also
changes with $s$ and cannot be computed once and reused across the history.

MLA instead separates content and position into two paths. The content path
does not apply RoPE and continues to compute $(q^C)^{\mathsf T}U_Kc_s$. The
positional channel is a small extra block of Q/K coordinates, not another
attention head. Let $q_t^R$ and $k_s^R$ denote the positional query and key
after RoPE has been applied at positions $t$ and $s$. $q_t^R$ belongs to the
current head, whereas $k_s^R$ is shared by all heads and cached.

The two paths produce

$$
\mathrm{content}_s=(q^{C})^{\mathsf T}U_Kc_s,
\qquad
\mathrm{position}_s=(q_t^{R})^{\mathsf T}k_s^R,
$$

and the final score is

$$
\mathrm{score}_s=\mathrm{content}_s+\mathrm{position}_s.
$$

Each historical token therefore caches $[c_s;k_s^R]$: $c_s$ carries shared
content, while $k_s^R$ carries shared position information. The content path
can still move $U_K$ to the query side; the positional channel remains
explicit.

### Per-head up-projections and weight absorption

The uncompressed thought experiment already showed how to move the projections.
Now apply the same regrouping to $U_K$ and $U_V$. MLA
admits two algebraically equivalent core-attention modes. Here “MQA mode” names
an execution strategy; it does not mean that every MLA model is an ordinary
MQA model.

| MLA execution mode | K/V used by core attention | Where up-projection occurs |
| --- | --- | --- |
| MHA mode | Per-head $[U_Kc;k^R]$ and $U_Vc$ | Before core attention |
| MQA mode | Shared $[c;k^R]$ and latent values $c$ | Absorbed into the query and output paths |

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

**Weight absorption** uses associativity to regroup fixed linear maps so that
expanded K/V need not be materialized. On the key side, reassociate the matrix
multiplication:

$$
(q^{C})^{\mathsf T}U_Kc_s
=\left(U_K^{\mathsf T}q^{C}\right)^{\mathsf T}c_s
=(q^A)^{\mathsf T}c_s,
\qquad q^A=U_K^{\mathsf T}q^C.
$$

The absorbed query $q^A$ can now be dotted directly with the cached latent.

A two-coordinate example makes the regrouping concrete. Let
$q=(1,2)^{\mathsf T}$, $c=(3,4)^{\mathsf T}$, and
$U=\begin{bmatrix}1&2\\0&1\end{bmatrix}$. Expanding the key first gives
$q^{\mathsf T}(Uc)=19$; transforming the query first gives
$(U^{\mathsf T}q)^{\mathsf T}c=19$. The first form computes $Uc$ for every
cached row, while the second computes $U^{\mathsf T}q$ once for the current
query.

On the value side, linearity gives

$$
\sum_s p_s U_Vc_s
=U_V\left(\sum_s p_s c_s\right).
$$

Consequently, $U_V$ can be composed with the model's output projection,
and core attention can operate on the latent states without materializing
expanded per-head K/V.

The following figure brings together the ordinary MHA cache, the shared MLA
cache, and the two weight-absorption paths:

```{figure} ../img/flashmla_cache_story.png
:width: 100%
:alt: Ordinary MHA caches separate key and value data for every head; MLA stores one shared compressed state and keeps head-specific work around attention

*Ordinary MHA stores a separate key/value slice per head. MLA stores one shared
compressed content state plus shared position information per token;
head-specific query and output transformations happen before and after
attention.*
```

We can now compare what each mechanism actually caches. The table counts scalar
elements stored per token per layer, excluding dtype and allocator metadata:

| Mechanism | Cached elements | Cached state |
| --- | ---: | --- |
| MHA | $2n_hd_h$ | Complete K and V for every head |
| GQA | $2n_{kv}d_h$ | Complete K and V for every KV head |
| MQA | $2d_h$ | One complete K/V pair shared by all query heads |
| MLA | $d_c+d_h^R$ | One $c_s$ and $k_s^R$ shared by all heads |

Here $n_{kv}$ is the number of GQA KV heads, and $d_h^R$ is the width of
$k_s^R$. The MLA row does not contain $2d_c$: the same $c_s$ supplies the
information used to form both the content K and V, so it is cached only once.
This chapter uses $d_c=512$ and $d_h^R=64$, giving $512+64=576$ scalar elements
in $[c_s;k_s^R]$. The later kernel parameter `d_qk=576` is the width of this
cached row.

The DeepSeek Sparse Attention (DSA) sparse-prefill operator studied here uses
this MQA mode: every selected latent KV entry is shared by all query heads.

### Numerical verification of the two execution modes

The CPU program constructs both executions. It includes a shared RoPE score
term, expands K/V in the MHA path, and absorbs the same matrices in the MQA path.
Float64 makes the equality check sensitive enough to catch a transposed index or
an incorrect contraction.

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

The query projection may use a separate low-rank latent as well. Write its
down- and up-projections as $D_Q$ and $U_Q$:

$$
c^Q=D_Qh,\qquad q^C=U_Qc^Q.
$$

This factorization mainly reduces activation memory during training; it does
not shrink the KV cache further. It happens before core attention: the `q`
tensor presented to the attention implementation is already projected, so the
KV-path analysis treats $q^C$ as an input.
:::

For the complete MLA architecture, training design, and original notation, see
the [DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434). The discussion here
keeps the KV compression, decoupled RoPE, and weight absorption needed to
understand the FlashMLA kernel that follows.

Weight absorption explains how one shared KV row can serve many query heads.
Sparse prefill introduces a separate boundary between token selection and the
sparse core-attention operator that consumes the selected rows.

## Token-selection boundary of the sparse-prefill operator

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

The sparse-prefill operator receives the indexer's result as an `indices`
tensor. Its semantic contract is:

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

An executable CPU reference makes these rules testable independently of any GPU
implementation.

## An executable sparse-attention reference

The general shape notation is:

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
`[s_q,h_q]`. The general sparse-prefill contract already requires `h_kv=1`;
the regular head-128 specialization additionally fixes `h_q=128` and
`d_v=512`.

For the absorbed MQA contract, `kv[:, 0, :]` supplies both K and V: all
`d_qk` coordinates participate in QK, while the first `d_v` coordinates are
the latent value. Here `sm_scale` is the model's semantic QK scale.

Invalid indices must be clamped *before* a PyTorch gather and then masked out.
Directly indexing with `-1` would incorrectly select the last row.

The gathered V row for an out-of-range address is also cleared before PV,
because a zero softmax weight does not neutralize a NaN under IEEE arithmetic.

An attention sink is equivalent to adding a logit whose value vector is zero.
Fix one query and head. Let $x_j$ be the ordinary scaled logit for selected KV
row $j$, $v_j$ its value, $a$ the sink logit, and $m$ the maximum of the
ordinary logits. The sink changes only the output denominator:

$$
O=\frac{\sum_j e^{x_j-m}v_j}
{\sum_j e^{x_j-m}+e^{a-m}}.
$$

The sink has no effect on `max_logits` or the returned log-sum-exp (`lse`). If every selected
position is invalid, the operator instead uses the explicit convention
`out=0`, `max_logits=-inf`, and `lse=+inf`.

The executable oracle follows four stages:

1. validate and safely gather the requested rows;
2. construct the validity predicate and clear only out-of-range V sentinels;
3. compute scaled logits, unnormalized weights, numerator, denominator, and the
   optional sink term; and
4. normalize the output and return the two statistics, including the all-invalid
   convention.

The CPU implementation is:

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
for defined inputs and return conventions. An implementation supplies tile
partitioning, storage reuse, and pipeline overlap while preserving that
contract.

## Sparse prefill in the FlashMLA operator family

FlashMLA spans both the sequence stage and the selection pattern. *Decoding*
adds a new query (or a small speculative group) step by step while reusing the
KV cache. The
[official FlashMLA repository](https://github.com/deepseek-ai/FlashMLA)
organizes its operators and their implementations into four broad families:

| Selection | Sequence stage | Representative purpose |
| --- | --- | --- |
| dense | prefill | MHA forward and backward |
| dense | decoding | read an MLA KV cache for newly generated queries |
| token-sparse | prefill | DSA core attention over a selected token list |
| token-sparse | decoding | DSA inference over a selected FP8 KV cache |

The token-sparse prefill cell combines externally selected, irregular KV-row
addresses with core attention, creating the Blackwell scheduling problem
studied here. Sparse decode uses a different paged-cache, scheduling, and
reduction contract.

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
B200 examples in this chapter therefore validate the computation only at
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

## The Blackwell regular head-128 case

The regular head-128 case retains the QK--softmax--PV chain and adds irregular
gather, absorbed latent KV, and cooperative thread-block ownership.

### Shape and dispatch conditions for regular head-128

The regular head-128 module has this shape-specialized signature:

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

In this specialization, 128 is the query-head count, while the 1 in `kv` means
that every head shares the same KV row. A common $d_{qk}=576$ case combines 512
latent-content coordinates with 64 RoPE coordinates; `d_v=512` is the
latent-value width. Other MLA operators may use different shapes.

For this MLA layer, the equivalent MHA representation has QK feature width
$128+64=192$ and value/output width 128. The absorbed MQA representation used
here has widths $512+64=576$ and 512. A rough count of multiply-add
coordinates per query--key pair is therefore $192+128=320$ versus
$576+512=1088$, about 3.4 times as many for the absorbed representation.
This is only an arithmetic-width intuition, not a prediction of kernel runtime.

The running shape is `s_q=1`,
`s_kv=8192`, `h_q=128`, `h_kv=1`, `d_qk=576`, `d_v=512`, and `topk=2048`. It is
one query row with 128 query heads sharing 2048 selected-index slots. Those
slots may contain duplicate or out-of-range addresses. Without a shorter
`topk_length`, the physical schedule visits $N=16$ tiles of 128 slots. When
`topk_length` is present, it visits
`max(ceil(topk_length / 128), 1)` tiles.

Each selected-index tile follows six semantic steps. In the source notation,
$L$ denotes raw QK logits, $W$ denotes BF16 unnormalized exponential weights,
`mi` is the online-softmax exponent origin, `li` is the denominator accumulated
relative to that origin, and $\widetilde O$ is the accumulated output before
division by the denominator:

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

In the running example, the first five steps repeat 16 times before the sixth
step runs.

### Complete source navigation

The chapter does not reproduce the entire device function. Instead, it presents
short excerpts organized around QK, softmax, PV, data movement, and
synchronization. Read the complete source through these entry points:

| Goal | Source entry point |
| --- | --- |
| Unified entry and shape dispatch | [`flash_mla_sparse_fwd.py` lines 66--125](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/flash_mla_sparse_fwd.py#L66-L125) |
| Config, test data, PyTorch reference, and launch ABI | [`sparse_prefill_head128_phase1.py` lines 66--244](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L66-L244) |
| Complete regular head-128 device kernel | [`sparse_prefill_head128_phase1.py` lines 247--865](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L247-L865) |
| CTA-pair TMA, tcgen05 MMA, and validity-mask helpers | [`_tma.py` lines 10--60](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/_tma.py#L10-L60), [`_gemm.py` lines 8--29](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/_gemm.py#L8-L29), and [`_mask.py` lines 10--29](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/_mask.py#L10-L29) |
| Specialization, compilation, launch, and numerical checks | [`sparse_prefill_head128_phase1.py` lines 868--905](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L868-L905) |

A productive order is dispatch and tensor ABI first, then the WG0, WG1, WG2,
and WG3 branches in `_kernel`. Follow TMA, MMA, and mask calls into their helpers
only when they appear, and finish with `run_test` to connect inputs, outputs, and
the reference. Each excerpt in the chapter preserves the source variable names
and slices and links to its full context.

The TIRx regular head-128 implementation is in
[`sparse_prefill_head128_phase1.py`](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py).
TIRx extends TVM 0.26's TIR Python DSL: `T` denotes the TIR script namespace,
while `Tx` contains GPU-kernel helpers. The `phase1` name follows the
corresponding CUDA implementation's file naming; on this regular prefill path,
one kernel produces the complete `(out, max_logits, lse)` result.

The implementation uses three execution levels. A cooperative thread array
(CTA) is one CUDA thread block. Two adjacent CTAs form a cluster that can
participate in CTA-group tensor-core operations. A warpgroup is four warps, or
128 threads, assigned one specialized role. Here one two-CTA cluster owns one
query row.

In the mathematical introduction, $p$ denoted a normalized softmax
probability. In the source, however, `tmem_p` and the register variable `p`
hold the raw QK logits already named $L$. The source's `s_frag` and
`s_smem_gemm` hold the unnormalized exponential weights already named $W$.
Only the epilogue divides the accumulated output by `li` plus the optional sink
term.

Short TIRx code blocks are contextual excerpts from the linked implementation.
Independently runnable blocks are labeled explicitly.

The following constants describe the tile sizes, thread count, and
synchronization slots:

```python
B_H = 128
B_TOPK = 128
D_V = 512
NUM_BUFS = 2
NUM_THREADS = 512
D_TQ = 384
```

`B_H` is the 128-head logical tile, `B_TOPK` is the 128 selected-index slots
processed per streaming tile, `D_V` is the value/output width, and `D_TQ` is the
384-coordinate Q suffix moved to dedicated on-chip storage. `NUM_THREADS=512`
gives each CTA four warpgroups. `NUM_BUFS=2` provides two slots for
synchronization state and two small packed-validity-mask slots.

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

Once `tirx-kernels` is installed, the dispatch can be inspected without
launching a GPU kernel:

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
The registry keeps dispatch separate from the device schedule, preserving the
distinction between operator selection and schedule implementation.

The selected implementation uses two thread blocks to cooperate on the six
steps without duplicating the whole tile.

### Query-row ownership across two CTAs

The difficulty lies in steps 3 and 5 of the tile skeleton. QK must form every
pair of 128 query heads and 128 selected tokens; PV then contracts the token
axis to produce 512 value coordinates. The implementation therefore lets two
CTAs form one logical tile and changes their partition axis between QK and PV.
The ownership map is:

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

A 2-CTA tensor-core operation forms one collective logical tile. QK computes the
cross-product of 128 heads and 128 selected tokens; PV then contracts those
tokens into 512 value coordinates. The partition rotates between the two GEMMs,
supported by collective `cta_group=2` MMA, paired on-chip layouts, and cross-CTA
synchronization.

The launch topology implements this ownership map. The grid contains
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

## Tile residency and lifetime

The ownership map says *who* computes each piece; residency says where a piece
waits between producers and consumers:

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
accumulators in TMEM. TMEM and SMEM have complementary roles: TMA gathers land
in SMEM, and the softmax warpgroup materializes BF16 unnormalized weights there
for PV.

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

Safe in-place reuse of these storage regions relies on a **completion barrier**,
a small hardware state object that records when an asynchronous producer has
finished. Its phase bit distinguishes successive uses of each synchronization
slot, allowing released portions of the single K/V/$L$/$W$ tile storage to be
overwritten safely.

## Warpgroup responsibilities

The four warpgroups have specialized roles:

| Warpgroup | Warps | Responsibility |
| --- | --- | --- |
| WG0 | 0--3 | load raw logits $L$ from TMEM, mask, online softmax, write weights $W$, rescale O, epilogue |
| WG1 | 4--7 | load index fragments and issue gather4 TMA for K |
| WG2 | 8--11 | load index fragments and issue gather4 TMA for V |
| WG3 | 12--15 | warp 12 of CTA 0 issues CTA-group QK/PV MMA; warp 13 in each CTA builds validity masks |

WG3 concentrates its active responsibilities in warp 12 for asynchronous MMA
issue and warp 13 for validity masks. This asymmetric role assignment matches
the parallelism of each operation: one elected lane issues MMA for the CTA pair,
warp 13 packs validity bits, and WG0 uses many lanes for exponentiation, row
reductions, and epilogue conversion.

:::{admonition} Role-specific register limits
:class: note

The register budgets match the roles. WG0 raises its limit to 144 registers and
WG3 to 168; producer groups lower theirs to 96. The TIRx API spells these calls
as `T.ptx.setmaxnreg(True, ...)` for an increase and
`T.ptx.setmaxnreg(False, ...)` for a decrease.
:::

The gather handoff follows a ready/free protocol. A producer waits until a
tile's storage is **free**, writes or asynchronously fills the tile, and signals
**ready**. The consumer waits for ready, uses the tile, and signals free when the
storage may be overwritten. These barriers transfer ownership of the in-place
storage.

### From irregular rows to regular tiles

Sparse row addresses destroy the contiguous 2-D copy pattern used by dense
attention. WG1 and WG2 use explicit TMA `gather4`: one issue supplies exactly
four row coordinates, so a warp can bring noncontiguous KV rows into a regular
SMEM tile.

The contextual excerpt identifies the addresses read by one `gather4` issue and
the barrier that receives its completion. Its index names and slices match the
linked implementation:

- `gather4=[...]` supplies the four KV source-row coordinates for this issue;
- `cur_buf = k % NUM_BUFS` selects the current slot in the two-slot barrier ring;
  and
- `bar` is the ready-barrier array passed to this copy helper. `leader_mbar`
  selects the CTA-pair leader's slot, where TMA reports asynchronous completion.

The remaining names describe the surrounding layouts.

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

## The SMEM--SMEM and TMEM--SMEM decomposition of QK

The gather path has turned irregular K rows into a regular SMEM tile. Why split
one QK dot product into SMEM--SMEM and TMEM--SMEM pieces? Q must meet that tile
without occupying all of the aliased SMEM workspace. Using the SS/TS terms
defined with the residency map, the QK dot product is split at
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

## Lazy O rescaling in online softmax

QK has produced raw logits $L$; softmax must now turn each tile into $W$ while
preserving state from earlier selected tiles. In the running shape, this is why
the state must be merged across $N=16$ tiles. For every raw QK dot product $x_j$
in the next tile, the kernel first computes

$$
r_j=x_j\cdot\text{semantic\_QK\_scale}\cdot\log_2(e).
$$

The TIRx specialization binds the multiplier named `sm_scale_div_log2` to
`(1 / sqrt(d_qk)) * log2(e)`. Despite that source name, each $r_j$ is simply a
model-scaled score expressed in base-2 units so that the kernel can use `exp2`.

For a stream of such score tiles, online softmax stores a base-2 row origin $m$,
denominator $\ell$, and unnormalized output $\widetilde O$. To merge the next
tile,

$$
m'=\max(m,\max_j r_j),\qquad
\alpha=2^{m-m'},
$$

$$
\ell'=\alpha\ell+\sum_j2^{r_j-m'},\qquad
\widetilde O'=\alpha\widetilde O+\sum_j2^{r_j-m'}v_j.
$$

Rescaling the full 512-coordinate O tile whenever the row maximum increases
would be expensive. The lazy-threshold excerpt maps to the recurrence as
follows:

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

## Race-free pipeline overlap

The pipeline repeats the ready/free ownership handoffs across tiles. QK must
finish before softmax consumes its logits, and PV must wait until softmax has
produced its weights. Meanwhile, the gather warpgroups move forward whenever an
in-place K or V segment becomes reusable.

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
the intended reuse of a slot. The four-step view gives the causal chain; the
part-level edges release K and V segments as early as their consumers finish.

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
iterations earlier.

`bar_qk_part_done` allows the producer to replace K's prefix before its suffix
is reusable. The two `bar_sv_*` edges do the analogous job for V.

```{figure} ../img/flashmla_pipeline.png
:width: 100%
:alt: Detailed sparse-prefill pipeline showing serial QK and PV issue, part-wise K and V reuse, the mask-slot ring, and the WG0 handoff

This detailed view names the part-level reuse edges and the mask-slot ring.
Barrier phases protect reuse of the single in-place tile storage.
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

## Compiling and numerically verifying regular head-128

The regular head-128 specialization targets compute capability 10, and its
TMA/tcgen05 forms require an SM100-class GPU. The environment uses B200, CUDA
12.9 or newer, and the dependencies specified here.

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

`run_test` covers three levels of verification: compilation checks code
generation, launch checks execution on the GPU, and the FP32 PyTorch oracle
checks output, maximum logits, and LSE with explicit tolerances. The numerical
comparison can expose head-partition and validity-bit errors beyond code
generation itself.

The `tirx-kernels` CLI can run the registered regular-head128 configuration:

```bash
python -m tirx_kernels.test \
  --kernel sparse_flashmla_prefill_head128_phase1 \
  --config bench_regular_dqk576_hq128_s4096_kv8192_topk2048
```

Useful negative tests are just as important. Set `inject_invalid_indices=True`
to cover negative and too-large row IDs, and `have_topk_length=True` to exercise
the position predicate. Test an all-invalid row and confirm the documented
zero/-infinity/+infinity convention. Calling the TIRx dispatch entry for head-64
and small-top-k shapes extends coverage to every prefill specialization in this
dispatch tree. Parity with the complete FlashMLA interface remains a separate
verification target.

## Operator and specialization invariants

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

## Exercises and further validation

The regular head-128 specialization is one point in a dispatch space. The
TIRx dispatch tree also contains a head-64 phase-1 specialization and a head-128
`d_qk=512` small-top-k specialization. Their different schedules demonstrate
how tile economics drive dispatch across that space.

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

The central lesson applies beyond FlashMLA. A high-performance irregular
operator often regularizes work in stages: an indexer creates sparse addresses,
TMA gathers those addresses into dense tiles, tensor cores consume the tiles,
and explicit barriers protect aggressive storage reuse. Understanding the
algorithm, dispatch contract, ownership map, and memory protocol together is
what turns a fast kernel from an opaque artifact into an explainable program.
