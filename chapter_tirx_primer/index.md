(chap_tirx_primer)=
# TIRx Language and Compile Pipeline

This appendix explains how TIRx source becomes generated CUDA. Use it when a tutorial kernel contains an unfamiliar parser feature, buffer form, or generated-code check.

## What TIRx Source Represents

A TIRx function is Python syntax that builds TVM IR. The Python code is not executed as a normal GPU program. The parser reads the function body and turns supported constructs into IR nodes: buffers, coordinates, scopes, allocations, loops, conditions, layouts, and tile primitives.

The current entry pattern is:

```python
@T.prim_func
def kernel(a_ptr: T.handle, b_ptr: T.handle):
    n = T.int32()
    A = T.match_buffer(a_ptr, [n], "float32")
    B = T.match_buffer(b_ptr, [n], "float32")

    T.device_entry()
    bx = T.cta_id([(n + 255) // 256])
    tid = T.thread_id([256])

    i = bx * 256 + tid
    if i < n:
        B[i] = A[i] * 2.0
```

Read the structure as:

1. Host-visible arguments and buffer binding.
2. `T.device_entry()` starts device-side IR.
3. Coordinate calls name the launch and thread positions.
4. Scope prefixes tell TIRx which execution team owns the operation.
5. Primitive calls describe tile movement, compute, or scalar/register work.

## `@T.prim_func` and `@T.jit`

Use `@T.prim_func` when the kernel shape is already concrete or when dynamic arguments are represented with `T.int32()` and `T.match_buffer`.

Use `@T.jit(...)` when the kernel has compile-time parameters. Current `tirx-kernels` reference kernels use `@T.jit(persistent=True)` for shape-specialized persistent kernels:

```python
@T.jit(persistent=True)
def _kernel(
    Q: T.Buffer((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"),
    O: T.Buffer((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"),
    *,
    BATCH_SIZE: T.constexpr,
    SEQ_LEN_Q: T.constexpr,
    NUM_QO_HEADS: T.constexpr,
    HEAD_DIM: T.constexpr,
):
    T.device_entry()
    ...
```

A JIT function is specialized before compile:

```python
kernel = _kernel.specialize(
    BATCH_SIZE=batch_size,
    SEQ_LEN_Q=seq_len_q,
    NUM_QO_HEADS=num_heads,
    HEAD_DIM=head_dim,
)
```

## Host and Device: `T.device_entry()`

The example above has a bare `T.device_entry()` line. It marks a boundary, because a TIRx PrimFunc
is a *single* function that holds both **host** and **device** code:

- Everything **before** the marker is **host** code — the signature and the `T.match_buffer` calls
  that bind the array arguments.
- Everything **after** it is **device** code — the kernel body: scope-id definitions (`T.cta_id`,
  `T.warpgroup_id`), tile primitives, and the compute loop.

At compile time the **SplitHostDevice** pass ("annotate and split device functions from host, then
lower kernel launches") turns that one PrimFunc into two: a **host** function that launches the
kernel and the **device** function that *is* the kernel. The launch geometry — grid, cluster, and
block dimensions — comes from the device region's scope-id extents: `T.cta_id([SM_COUNT])` becomes
a grid of `SM_COUNT` CTAs, and a 2-CTA cluster (what a `cta_group::2` MMA needs,
{ref}`chap_tensor_cores`) becomes the launch's cluster dimension. So `T.device_entry()` is the one
line that tells the compiler where host setup ends and the GPU kernel begins.

## Coordinates and Scopes

Coordinates name positions. Scopes name the team that cooperates on the following operation.

```python
T.device_entry()
bx, by = T.cta_id([M // BLK_M, N // BLK_N])
wg_id = T.warpgroup_id([2])
warp_id = T.warp_id_in_wg([4])
lane_id = T.lane_id([32])
tid_in_wg = T.thread_id_in_wg([128])
```

The relationship is in the name: `warp_id_in_wg` means warp inside a warpgroup, and `cta_id_in_cluster` means CTA inside a cluster.

Scope examples:

```python
Tx.cta.copy(Asmem[:, :], A_tile)

Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()

Tx.copy(D[row, :], Dreg_f16[:])
```

A scope prefix does not create threads. The launch already did that. The scope tells TIRx how to lower the prefixed operation.

## Buffers

There are two common ways to describe kernel arguments.

The handle form is useful when a runtime dimension appears in the buffer shape:

```python
def kernel(a_ptr: T.handle, b_ptr: T.handle):
    n = T.int32()
    A = T.match_buffer(a_ptr, [n], "float32")
    B = T.match_buffer(b_ptr, [n], "float32")
```

The typed-buffer form is useful for constexpr-specialized kernels:

```python
def kernel(A: T.Buffer((M, K), "float16"), D: T.Buffer((M, N), "float16")):
    ...
```

Inside the kernel, these are device buffers. They are not Python arrays.

## Shared Memory and TMEM

Shared memory is usually allocated through `T.SMEMPool()`:

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc([1], "uint32")
mma_bar = pool.alloc([1], "uint64", align=8)
pool.move_base_to(1024)
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
pool.commit()
```

The pool is a shared-memory allocation plan. It is not an automatic live-range optimizer. Put control objects and tile storage in the order you need, then call `commit()`.

TMEM can be managed manually. The layout vocabulary (`TileLayout`, `S`, `TLane`, `TCol`) lives in `tvm.tirx.layout`, not on `Tx`, so import it explicitly:

```python
from tvm.tirx.layout import S, TileLayout, TLane, TCol

T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
tmem = T.decl_buffer(
    (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
    layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
)
```

Or through `T.TMEMPool(...)`, which is the style used in current larger kernels:

```python
tmem_pool = T.TMEMPool(pool, total_cols=N_COLS_TMEM, cta_group=CTA_GROUP, tmem_addr=tmem_addr)
tmem = tmem_pool.alloc((128, N_COLS_TMEM), "float32")
tmem_as_f16 = tmem_pool.alloc((128, N_COLS_TMEM * 2), "float16")
tmem_pool.commit()
...
tmem_pool.dealloc()
```

`T.decl_buffer(..., scope="tmem")` gives TIRx a typed view of already allocated TMEM. It does not allocate TMEM by itself.

## Pipeline Objects

Software-pipelined kernels need stage and phase tracking. Current TIRx uses `PipelineState` and `Pipeline`. These helpers are *not* part of the `Tx` namespace; import them explicitly from `tvm.tirx.lang.pipeline` (the same module also exports `MBarrier`, `TMABar`, and `TCGen05Bar`):

```python
from tvm.tirx.lang.pipeline import Pipeline, PipelineState

kv_state = PipelineState(SMEM_PIPE_DEPTH_KV)
kv_load = Pipeline(pool, SMEM_PIPE_DEPTH_KV, full="tma", empty="tcgen05", empty_phase_offset=1)
```

`PipelineState` carries `stage` and `phase`. Calling `advance()` moves to the next stage and flips the phase when the ring wraps.

`Pipeline` allocates the barriers that protect each stage. The `full` endpoint is usually waited by the consumer; the `empty` endpoint is usually waited by the producer before reusing a stage. The exact names depend on the producer-consumer direction.

## Metaprogramming Helpers

`T.meta_var(expr)` is a parser-only value: the wrapped expression is substituted inline wherever the name is used, so it never becomes a separate IR variable in the final TIR:

```python
m_st = T.meta_var(bx * BLK_M)
n_st = T.meta_var(by * BLK_N)
```

Use it for index arithmetic and small compile-time-derived expressions, not for storage allocation.

`@T.inline` marks a helper that is expanded at the call site:

```python
@T.inline
def tma_load(stage, k_start):
    if tid == 0:
        Tx.copy_async(Asmem[stage, :, :], A[:, k_start:k_start + BLK_K], dispatch="tma")
```

Inlining is useful when the same tile operation appears in prefetch and loop bodies.

## Compile Pipeline

Compile TIRx kernels with the TIRx pipeline selected:

```python
target = tvm.target.Target("cuda")
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
```

At a high level:

```text
TIRx Python DSL
  -> TVM IR with TIRx-specific nodes
  -> TIRx lowering passes
  -> ordinary TIR
  -> CUDA host/device code
  -> compiled GPU module
```

The default TIR pipeline is not a replacement for `tir_pipeline="tirx"` when the function contains TIRx tile primitives.

## Generated CUDA Inspection

Generated CUDA is the fastest way to check whether scopes and hardware helpers lowered as intended:

```python
cuda_source = ex.mod.imports[0].inspect_source("cuda")
print(cuda_source)
```

Useful checks:

| TIRx intent | Generated-code clue |
|-------------|---------------------|
| CTA coordinate | `blockIdx.x`, `blockIdx.y` |
| CTA thread coordinate | `threadIdx.x` |
| Warpgroup branch | `warp_id_in_cta >> 2` |
| Warp branch | `warp_id_in_cta & 3` |
| Lane-0 branch | `threadIdx.x % 32 == 0` |
| Elected issue | `tvm_builtin_elect_one_sync_op()` |
| TMA path | `cp.async.bulk` |
| Blackwell MMA | `tcgen05.mma` |
| mbarrier | `mbarrier.init`, `mbarrier.arrive`, `mbarrier.try_wait` |

When the TIRx source and generated CUDA disagree, debug the generated CUDA first. It shows which threads actually issue an operation and which waits are present.
