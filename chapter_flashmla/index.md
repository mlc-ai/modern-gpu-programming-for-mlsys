(chap_flashmla)=
# FlashMLA

Start with an apparent contradiction. The Blackwell kernel studied in this
chapter accepts 128 query heads, but its KV tensor has only one head. One cached
KV entry must somehow serve all 128 heads, preserve head-specific attention, and
produce 128 different 512-coordinate outputs.

That puzzle has three layers. MLA explains why the cache can be shared.
Weight absorption explains where the per-head projections went. The FlashMLA
kernel explains how a sparse list of shared cache rows becomes a dense,
two-CTA tensor-core computation.

```{figure} ../img/flashmla_cache_story.png
:width: 100%
:alt: One latent KV cache entry serving 128 query heads through absorbed projections

The cache stores one latent content vector and one shared RoPE key per token.
Per-head behavior survives on the query and output sides rather than as 128
materialized K/V cache entries.
```

:::{admonition} Overview
:class: overview

- Derive MLA's shared latent cache, MHA/MQA execution modes, and weight absorption.
- Turn the sparse-attention contract into a 2-CTA Blackwell dataflow program.
- Run executable references, then compile and verify the pinned kernel on B200.
:::

The name *FlashMLA* combines two ideas that are easy to conflate. MLA is an
attention architecture that compresses key-value state into a latent vector.
FlashMLA is DeepSeek's library of optimized attention kernels, spanning dense
and sparse operators across prefill and decoding.

We will derive the architecture, make the sparse contract executable, and then
read one specific library path with the same algorithm-to-schedule depth used
for FA4: the regular 128-query-head sparse-prefill forward kernel on Blackwell.

Our implementation reference is pinned to
[`mlc-ai/tirx-kernels@5be39749`](https://github.com/mlc-ai/tirx-kernels/tree/5be39749e7dfd2c4bdae9b4d396f8ec35af07126).
All TIRx API spellings and excerpts below come from that revision. This matters:
the compiler APIs are evolving, while a textbook example must be reproducible.

## Why does MHA cache separate K/V for every head?

Let $h_t\in\mathbb{R}^{d_{model}}$ be the hidden state of token $t$. Ordinary
multi-head attention (MHA) projects it independently for every head $i$:

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

For a fixed $d_{model}=n_h d_h$, changing only the head count need not change
that total width. The structural cost is that the cache still materializes
separate per-head state.

During a long decode, reading those slices for the growing token history can
dominate the step.

Multi-query attention (MQA) reduces the cache by sharing one K/V head across all
query heads. Grouped-query attention shares within groups. Those are useful model
architectures, but MLA takes a different route: retain the expressive per-head
projections while caching a shared low-dimensional source from which they can be
recovered.

### How does MLA cache once and recover per-head behavior?

The original [DeepSeek-V2 MLA derivation](https://arxiv.org/abs/2405.04434)
introduces a joint latent KV vector

$$
c_t^{KV}=W^{DKV}h_t,
$$

followed by separate up-projections

$$
k_{t,i}^{C}=W_i^{UK}c_t^{KV},\qquad
v_{t,i}^{C}=W_i^{UV}c_t^{KV}.
$$

The superscript $C$ denotes the non-positional, or *content*, channel. The key
observation is that the cache need not store the expanded $k^C$ and $v^C$ for
every head. It stores one $c^{KV}$ per token instead.

MLA may also factor the query projection through a separate low-rank latent,

$$
c_t^Q=W^{DQ}h_t,\qquad q_t^C=W^{UQ}c_t^Q.
$$

This query-side factorization reduces activation memory during training, but it
does not reduce the KV cache.

It also happens before the core-attention API: the `q` tensor passed to FlashMLA
already contains the projected query. We therefore treat $q^C$ as an input and
focus on the KV-side compression that determines the cache and kernel operands.

RoPE complicates this simple picture. A position-dependent rotation between
$W^{UK}$ and a dot product would prevent the matrices from being reassociated.
MLA therefore uses a decoupled positional channel: each query head has
$q_{t,i}^{R}$, while all heads share a cached $k_t^R$. The actual MHA-form score
uses

$$
q_{t,i}=[q_{t,i}^{C};q_{t,i}^{R}],\qquad
k_{s,i}=[k_{s,i}^{C};k_s^R].
$$

The cache is now $[c_s^{KV};k_s^R]$, still shared across heads. The positional
channel remains explicit; weight absorption applies to the content channel.

### Where do the per-head up-projections go?

MLA admits two algebraically related core-attention modes. This should not be
taken to mean that every MLA model is an ordinary MQA model.

| MLA execution mode | Core-attention K/V presented to the kernel | Where up-projection occurs |
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

The picture suggests the answer; the following two identities prove it. On the
key side, reassociate the matrix multiplication:

$$
(q_{t,i}^{C})^{\mathsf T}W_i^{UK}c_s^{KV}
=\left((W_i^{UK})^{\mathsf T}q_{t,i}^{C}\right)^{\mathsf T}c_s^{KV}.
$$

Define the absorbed query $q_{t,i}^{A}=(W_i^{UK})^{\mathsf T}q_{t,i}^{C}$.
The content score can now be computed directly against the cached latent. On the
value side, linearity gives

$$
\sum_s p_{t,s,i}W_i^{UV}c_s^{KV}
=W_i^{UV}\left(\sum_s p_{t,s,i}c_s^{KV}\right).
$$

Consequently, $W_i^{UV}$ can be composed with the model's output projection.
Neither expanded per-head K nor expanded per-head V needs to be materialized by
the attention kernel.

The distinction is operational, not merely notational. Appendix A of the
[DeepSeek-V3.2 report](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)
states that DeepSeek-V3.1-Terminus used MHA mode for training and prefill and MQA
mode for decoding.

DeepSeek Sparse Attention (DSA), by contrast, instantiates MLA in MQA mode even
for sparse prefill because each selected latent KV entry is shared by all query
heads.

Thus the $d_{qk}=576$, $d_v=512$ contract commonly seen in this sparse-prefill
path is an absorbed MQA representation: 512 latent content coordinates plus a
64-coordinate RoPE channel. It is not a universal shape for MLA or every
FlashMLA operator. The implementation also supports a $d_{qk}=512$ variant.

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

## Does the sparse kernel choose the tokens?

Dense attention visits every eligible KV token. DSA first uses a lightweight
*lightning indexer* to score candidate tokens and chooses a top-$k$ set for each
query. The sparse core attention then reads only those latent KV entries. If the
original context length is $L$, this changes the core-attention work from
$O(L^2)$ to $O(Lk)$ for prefill, although the indexer has its own cost.

```{figure} ../img/flashmla_sparse_story.png
:width: 100%
:alt: A lightning indexer selecting token rows before the sparse FlashMLA attention kernel

Selection and attention are separate operators. The indexer produces row
addresses; the sparse-prefill kernel gathers those rows and performs the
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
`0 <= topk_length[q] <= topk`. This is a caller precondition, not a value that
the kernel clips automatically: a value above `topk` makes the pinned kernel
walk beyond the logical `indices` storage.

This prefill interface also has no batch dimension. In the port studied below,
`h_kv=1`, so `indices` has shape `[s_q, 1, topk]`: each query token supplies one
selected-token list, and all 128 of its query heads share that list. A serving
system must flatten or otherwise map batches before making this call.

### Is FlashMLA one kernel or an operator family?

*Prefill* processes the prompt's query tokens in parallel; *decoding* adds a new
query (or a small speculative group) step by step while reusing the KV cache.
The [official FlashMLA repository](https://github.com/deepseek-ai/FlashMLA)
organizes its kernels into four broad families:

| Selection | Sequence stage | Representative purpose |
| --- | --- | --- |
| dense | prefill | MHA forward and backward |
| dense | decoding | read an MLA KV cache for newly generated queries |
| token-sparse | prefill | DSA core attention over a selected token list |
| token-sparse | decoding | DSA inference over a selected FP8 KV cache |

FlashMLA is therefore neither synonymous with sparse attention nor with the
kernel in this chapter.

We use its sparse prefill operator because it makes one systems question
especially visible: how can irregular TMA gathers feed regular tensor-core
tiles? Sparse decode has a different paged-cache, scheduling, and reduction
contract and is outside this chapter's boundary.

The public upstream sparse-prefill call is conceptually

```text
out, max_logits, lse = flash_mla_sparse_fwd(
    q, kv, indices, sm_scale,
    d_v=512,
    attn_sink=attn_sink,       # optional [h_q], float32
    topk_length=topk_length,   # optional [s_q], int32
)
```

The pinned TIRx repository reuses the public name in its unified
`flash_mla_sparse_fwd` registry entry, then dispatches to one of three SM100
phase-1 modules by shape.

:::{admonition} Port scope: `sm_scale` is fixed
:class: warning

The upstream call above accepts `sm_scale` at runtime. The pinned TIRx entry
preserves the registry name and shape dispatch, **not** the complete call
signature. All three prefill modules specialize the scale to `1 / sqrt(d_qk)`,
and the launch ABI has no scale argument.

In this revision, passing `sm_scale=...` through the `**kwargs` wrappers is
silently ignored. The B200 examples below therefore validate a fixed-scale
teaching port, not a drop-in replacement for upstream `flash_mla_sparse_fwd`.

A model whose semantic QK scale differs must expose or specialize the correct
value. Weight absorption does not justify changing that scale.
:::

## Can we make the sparse contract executable first?

Before reading the kernel, we need a reference that makes edge cases explicit.
For the absorbed MQA contract, `kv[:, 0, :]` supplies both K and V: all `d_qk`
coordinates participate in QK, while the first `d_v` coordinates are the latent
value.

Invalid indices must be clamped *before* a PyTorch gather and then masked out.
Directly indexing with `-1` would incorrectly select the last row.

The gathered V row for an out-of-range address is also cleared before PV,
because a zero softmax weight does not neutralize a NaN under IEEE arithmetic.
We deliberately leave an in-range row masked only by `topk_length` unsanitized
to match the pinned kernel behavior discussed below.

The complete CPU block answers whether those rules, the attention sink, and the
all-invalid convention can coexist in one executable oracle:

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
    # Deliberately keep in-range rows beyond topk_length unchanged to match the
    # kernel's documented exceptional NaN behavior.
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

An attention sink is equivalent to adding a logit whose value vector is zero.
It changes only the output denominator:

$$
O_i=\frac{\sum_j e^{x_{ij}-m_i}v_j}
{\sum_j e^{x_{ij}-m_i}+e^{a_i-m_i}}.
$$

Here $a_i$ is the per-head sink logit. It has no effect on `max_logits` or the
returned `lse`, a detail that is easy to miss when writing an independent
oracle.

## Which Blackwell case are we studying?

We now focus on
[`sparse_prefill_head128_phase1.py`](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py).
Despite the internal `phase1` name inherited from upstream FlashMLA, this
prefill path produces the complete `(out, max_logits, lse)` result in one kernel;
it is not a partial split-KV output awaiting a combine kernel.

This regular head-128 case is a useful bridge from the preceding FlashAttention
chapter. It retains the familiar QK--softmax--PV chain, then adds irregular
gather, absorbed latent KV, and cooperative 2-CTA ownership.

One naming warning prevents confusion later. In the mathematical introduction,
$p$ denotes a normalized softmax probability. In the pinned source, however,
`tmem_p` and the register variable `p` hold **raw QK logits**. In the schedule
discussion we will call those logits $L$.

The source's `s_frag` and `s_smem_gemm` hold BF16 **unnormalized exponential
weights**, which we will call $W$.

The weights are not normalized in place. The final output becomes normalized
only when the accumulated output is divided by `li` plus the optional sink term
in the epilogue.

All shorter TIRx code blocks below are contextual excerpts from the pinned
kernel, not standalone programs. The complete module is compiled and
numerically verified in the final section. Blocks intended to run independently
are called out explicitly.

### Which shapes reach the regular head-128 path?

The regular head-128 module fixes the following tile constants:

```python
B_H = 128
B_TOPK = 128
D_V = 512
NUM_BUFS = 2
NUM_THREADS = 512
D_TQ = 384
```

Its input and output contract is:

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

The module accepts `d_qk` 512 or 576, requires `h_kv=1`, `d_v=512`, and requires
the regular path's `topk` to be a positive multiple of 128. The unified front
door makes one additional choice for 128 heads: `d_qk=512` with `topk<=1280` goes to the
small-top-k implementation; other supported head-128 shapes go to this regular
implementation. Head-64 shapes use a separate kernel.

The positive-`topk` condition is a required caller precondition. The unified front
door rejects `topk<=0`, but the pinned per-implementation `_cfg().validate()`
methods check divisibility without checking positivity. A direct import of one
specialization must therefore still reject or avoid nonpositive `topk`; acceptance
by that local validator does not make such a launch valid.

This dispatch can be inspected without launching a GPU kernel:

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

The dispatch itself is documented in the pinned
[`flash_mla_sparse_fwd.py`](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/flash_mla_sparse_fwd.py#L66-L120).
Keeping dispatch separate from the device schedule prevents a tutorial from
mistaking one specialization for the entire operator.

### Why does one query row need two CTAs?

The launch grid contains `2 * s_q` CTAs and clusters adjacent CTAs in pairs:

```python
block_idx = T.cta_id([2 * s_q])
T.cta_id_in_cluster([2])
cta_idx: T.let = block_idx % 2
s_q_idx: T.let = block_idx // 2
thread_idx = T.thread_id([512])
T.warpgroup_id([4])
```

One cluster therefore owns one query row, and each CTA has four warpgroups.
Start with the ownership map: the unusual point is that the partition axis
changes between QK and PV.

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
between the two GEMMs. Collective `cta_group=2` MMA, paired SMEM/TMEM layouts,
and cross-CTA barriers together form the logical tile.

The division follows directly from the pinned source: Q is chunked by
`cta_idx`, the K producer selects the `cta_idx` half of every top-k block, and
the V producer starts at `cta_idx * 256`.

## Where do the tiles live?

Blackwell's tensor memory (TMEM) lets `tcgen05` keep large accumulators close to
the tensor cores. It does not replace shared memory (SMEM). TMA gathers arrive in
SMEM, and the softmax warpgroup must materialize BF16 unnormalized weights in
SMEM for the PV GEMM.

```{figure} ../img/flashmla_dataflow.png
:width: 100%
:alt: Data movement between global memory, shared memory, tensor memory, and registers

Q is split between an SMEM prefix and a TMEM suffix. Gathered K/V enter SMEM,
raw QK logits and the output accumulate in TMEM, and unnormalized softmax
weights cross back through SMEM for PV.
```

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

SMEM is aggressively aliased. `q_full`, the gathered K/V region, and the output
epilogue reuse a union-like base when their lifetimes permit.

After the final 384 Q columns have moved to TMEM, only the
$d_{sq}=d_{qk}-384$ prefix must stay live for the first QK part. For
`d_qk=512`, $d_{sq}=128$; for 576, it is 192.

The allocation plan is visible in
[`sparse_prefill_head128_phase1.py` lines 302--365](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L302-L365).

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

The register budgets match the roles. WG0 raises its limit to 144 registers and
WG3 to 168; producer groups lower theirs to 96. The pinned API spells these calls
as `T.ptx.setmaxnreg(True, ...)` for an increase and
`T.ptx.setmaxnreg(False, ...)` for a decrease.

### How do irregular rows become regular tiles?

Sparse row addresses destroy the contiguous 2-D copy pattern used by dense
attention. WG1 and WG2 use explicit TMA `gather4`: one issue supplies exactly
four row coordinates, so a warp can bring noncontiguous KV rows into a regular
SMEM tile.

The next contextual excerpt answers which addresses one `gather4` issue reads
and which barrier receives its completion. Its index names and slices are kept
exactly as they appear at the pinned revision:

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

As in the upstream API, callers should still avoid NaNs in in-range KV rows
referenced beyond `topk_length`. The optimized data path does not promise to
sanitize that exceptional case.

The pinned source separates these roles cleanly: the
[K producer is at lines 608--676](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L608-L676),
the [V producer is at lines 679--729](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L679-L729),
and [validity packing is at lines 841--865](https://github.com/mlc-ai/tirx-kernels/blob/5be39749e7dfd2c4bdae9b4d396f8ec35af07126/tirx_kernels/flashmla/sparse_prefill_head128_phase1.py#L841-L865).

## Why does QK use both SS and TS?

The next source excerpt answers how the kernel keeps only part of Q in SMEM. The
QK dot product is split at $d_{sq}=d_{qk}-384$:

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

For a stream of score tiles, the usual online-softmax recurrence stores a row
origin $m$, denominator $\ell$, and unnormalized output $\widetilde O$. In base
2, when the next tile has scaled scores $r$,

$$
m'=\max(m,\max r),\qquad
\alpha=2^{m-m'},
$$

$$
\ell'=\alpha\ell+\sum_j2^{r_j-m'},\qquad
\widetilde O'=\alpha\widetilde O+\sum_j2^{r_j-m'}v_j.
$$

Conceptually, the implementation multiplies natural-unit logits by the model's
semantic QK scale and `log2(e)`, then uses `exp2`. In this fixed-scale port, the
compile-time `sm_scale_div_log2` is
`(1 / sqrt(d_qk)) * log2(e)`. This is the same softmax in a form that maps
directly to fast base-2 exponent instructions.

Rescaling the full 512-coordinate O tile whenever the row maximum increases
would be expensive. The head-128 kernel uses a lazy threshold:

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

`real_mi` is maintained separately as the exact maximum seen so far. Therefore
the optimization does not change `max_logits`; `mi` is the numerical origin for
the recurrence, while `real_mi` is the reported statistic. At the end, the two
64-token contributions to each logical row are combined, and the kernel emits

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
pinned reference.

## How can the stages overlap without racing?

The schedule is easiest to understand as a dependency puzzle. QK must finish
before softmax consumes its logits. PV must wait until softmax has produced its
weights. Yet the gather warpgroups should keep moving whenever the in-place K or
V segment they need has become reusable.

```{figure} ../img/flashmla_pipeline_stages.png
:width: 100%
:alt: Conceptual sparse-prefill stages and the dependencies that permit steady-state overlap

In steady state, softmax($k-1$) may overlap QK($k$). The sole MMA issuer then
issues PV($k-1$) after QK($k$); once QK($k$) completes, softmax($k$) may overlap
that PV. Producer warpgroups gather the next safe K/V segments around this
serial issuer order.
```

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

For tile $k$, the sole MMA issuer first launches QK($k$). Once WG0 has turned
$L(k-1)$ into $W(k-1)$, that same issuer later launches PV($k-1$) in the serial
loop.

QK and PV are therefore interleaved by one instruction stream, not issued as
two simultaneous tensor-core streams. Their asynchronous execution overlaps
the other warpgroups' gathers and scalar softmax work, subject to the part-level
edges above.

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
:alt: Sparse-prefill pipeline showing QK of the current tile and PV of the previous tile

The steady state interleaves QK($k$) and PV($k-1$) while other roles perform
part-wise K/V gathers and softmax work. Barrier phases protect in-place storage
reuse rather than selecting separate full-tile buffers.
```

Two kinds of fences appear around these edges. `T.ptx.tcgen05.fence.*` orders
TMEM accesses relative to thread-visible work, while
`T.ptx.fence.proxy_async("shared::cta")` establishes cross-proxy ordering between
generic and asynchronous accesses to SMEM.

Here the proxy fence is needed both when generic stores of $W$ or the epilogue
tile precede tcgen05/TMA asynchronous reads, and after an asynchronous SMEM read
completes before generic code overwrites aliased storage.

An mbarrier communicates completion and an ownership handoff; the proxy fence
orders memory effects across proxies. Neither is a substitute for the other.

## How do we compile and verify the entire path?

The pinned module targets compute capability 10 and its TMA/tcgen05 forms require
an SM100-class GPU. Use B200, CUDA 12.9 or newer, and the official Apache TVM
0.26.0 package. The pinned kernel uses that release's TIRx API rather than the
spellings on a moving development checkout.

First install a CUDA-enabled PyTorch build that supports B200 using the
[official PyTorch selector](https://pytorch.org/get-started/locally/). The
companion repository imports PyTorch but does not declare it as a package
dependency.

The following commands answer exactly which TVM release and kernel revision the
examples require:

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

The repository CLI can run the full registered configuration:

```bash
python -m tirx_kernels.test \
  --kernel sparse_flashmla_prefill_head128_phase1 \
  --config bench_regular_dqk576_hq128_s4096_kv8192_topk2048
```

Useful negative tests are just as important. Set `inject_invalid_indices=True`
to cover negative and too-large row IDs, and `have_topk_length=True` to exercise
the position predicate. Test an all-invalid row and confirm the documented
zero/-infinity/+infinity convention. Finally, call the unified dispatch entry
for head-64 and small-top-k shapes so that a successful regular-head128 run is
not mistaken for full API coverage.

## Which five invariants define this path?

1. **The cache invariant.** One `h_kv=1` latent KV row can serve 128 query heads
   because key up-projection is absorbed into each query and value up-projection
   is moved after core attention. The RoPE channel stays explicit, and the QK
   scale remains the model's semantic scale.

2. **The sparse-contract invariant.** Selection happens before this kernel.
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

The first two invariants define operator semantics; the last three define this
specialization's schedule contract. Another dispatch may change tile sizes,
register budgets, ownership, or barrier topology, but it must replace those
contracts explicitly rather than silently changing the operator it computes.

## What should you test next?

The regular head-128 kernel is one point in a dispatch space. The pinned tree
also contains a 64-head phase-1 kernel and a 128-head `d_qk=512` small-top-k
kernel. Their different schedules are evidence that sparse attention should be
dispatched by tile economics, not forced through one universal template.

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
