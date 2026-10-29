# Copyright (c) 2026 TIRx authors
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# Adapted from the TIRx port of FlashAttention (Dao-AILab/flash-attention).
# See examples/licenses/ for the source licenses and contributor notices.

"""TVMScript FA4 from the book's original reference, migrated to apache-tvm-tirx.

Source: mlc-ai/tirx-kernels@5be39749e7dfd2c4bdae9b4d396f8ec35af07126,
tirx_kernels/attention/flash_attention4.py. The tile helpers below are ordinary
TVMScript macros using the compiler's CUDA tile API.
"""

from __future__ import annotations

import math
import os

from tvm_tirx.script import tirx as T
from tvm_tirx.backend.cuda.script import tile as Tx
from tvm_tirx.backend.cuda.axis import TLane, TCol, tid_in_wg as thread_in_wg
from tvm_tirx.backend.cuda.layout import mma_shared_layout, SwizzleMode
from tvm_tirx.backend.cuda.lang import SMEMPool
from tvm_tirx.backend.cuda.lang.pipeline import MBarrier, Pipeline, PipelineState, TCGen05Bar
from tvm_tirx.backend.cuda.lang.tile_scheduler import (
    FlashAttentionLinearScheduler,
    FlashAttentionLPTScheduler,
)
from tvm_tirx.backend.cuda.iket import IketProfiler
from tvm_tirx.tirx.layout import TileLayout, S

EMU_PAIRS_CAUSAL = 2
EMU_START_CAUSAL = 0
EMU_PAIRS_NC = 2
EMU_START_NC = 1


def wg_local_layout(width):
    return TileLayout(S[(128, width) : (1 @ thread_in_wg, 1)])


def alloc_wg_tile(width, dtype="float32"):
    return T.alloc_local((width,), dtype)


@T.inline
def vote_any(value):
    result: T.uint32
    T.ptx.vote_sync.any.pred(result, T.ptx.pred(T.cast(value, ty="bool")), T.uint32(0xFFFFFFFF))
    return result


@T.inline
def reciprocal(value):
    result: T.float32
    T.ptx.rcp.approx.ftz.f32(result, value)
    return result


@T.inline
def copy_tmem(dst, src):
    if T.constexpr(str(src.scope()) == "tmem"):
        reg = T.decl_tensor(
            dst.ptr_to([0]),
            ty_args=[
                T.Tensor(
                    (128, int(src.shape[-1])),
                    "float32",
                    scope="local",
                    layout=wg_local_layout(int(src.shape[-1])),
                )
            ],
        )
        T.ptx.tcgen05.fence__after_thread_sync()
        Tx.apply(
            T.ptx[f"tcgen05.ld.sync.aligned.32x32b.x{int(src.shape[-1])}.b32"],
            reg.view(4, 32, int(src.shape[-1])),
            src.view(4, 32, int(src.shape[-1])),
            scope="warpgroup",
        )
        T.ptx.tcgen05.wait__ld.sync.aligned()
    else:
        reg = T.decl_tensor(
            src.ptr_to([0]),
            ty_args=[
                T.Tensor(
                    (128, int(dst.shape[-1])),
                    src.dtype,
                    scope="local",
                    layout=wg_local_layout(int(dst.shape[-1])),
                )
            ],
        )
        Tx.apply(
            T.ptx[f"tcgen05.st.sync.aligned.32x32b.x{int(dst.view('uint32').shape[-1])}.b32"],
            dst.view("uint32").view(4, 32, int(dst.view("uint32").shape[-1])),
            reg.view("uint32").view(4, 32, int(dst.view("uint32").shape[-1])),
            scope="warpgroup",
        )


@T.inline
def wg_mul(dst, src, scale):
    for j in T.unroll(int(dst.shape[0])):
        dst[j] = src[j] * scale


@T.inline
def wg_fma(dst, src, scale, bias):
    for j in T.unroll(int(dst.shape[0])):
        T.ptx.fma.rn.f32(dst[j], src[j], T.float32(scale), bias)


@T.inline
def wg_cast(dst, src):
    for j in T.unroll(int(dst.shape[0])):
        dst[j] = T.cast(src[j], ty="float16")


@T.inline
def store_shared(dst, src, column):
    row = T.cuda.thread_idx("x") % 128
    for j in T.unroll(int(src.shape[0])):
        dst[row, column + j] = src[j]


@T.inline
def add_tile(dst, src, value, rounding_mode="rn"):
    for j in T.unroll(int(dst.shape[0])):
        T.ptx[f"add.{rounding_mode}.f32"](dst[j], src[j], T.float32(value))


@T.inline
def sub_tile(dst, a, b, rounding_mode="rn"):
    for j in T.unroll(int(dst.shape[0])):
        T.ptx[f"sub.{rounding_mode}.f32"](
            dst[j], a[j], b[j] if T.constexpr(hasattr(b, "shape")) else T.float32(b)
        )


@T.inline
def fma_tile(dst, a, b, value):
    for j in T.unroll(int(dst.shape[0])):
        T.ptx.fma.rn.f32(dst[j], a[j], b[j], T.float32(value))


@T.inline
def max_tile(dst, src, accum=False):
    if not T.constexpr(accum):
        dst[0] = T.float32(-float("inf"))
    for j in T.unroll(int(src.shape[0])):
        dst[0] = T.max(dst[0], src[j])


@T.inline
def sum_tile(dst, src, accum=False):
    if not T.constexpr(accum):
        dst[0] = T.float32(0)
    for j in T.unroll(int(src.shape[0])):
        dst[0] = dst[0] + src[j]


@T.inline
def gemm_tile(dst, a, b, accum=False, trans_b=False):
    # Each shared stage is a fresh, aligned 2D view for descriptor extraction.
    if T.constexpr(str(a.scope()) != "tmem"):
        aa = T.decl_tensor(
            a.ptr_to([0, 0]),
            ty_args=[
                T.Tensor(
                    (int(a.shape[0]), int(a.shape[1])),
                    "float16",
                    scope="shared.dyn",
                    align=1024,
                    layout=mma_shared_layout(
                        "float16", SwizzleMode.SWIZZLE_128B_ATOM, tuple(a.shape)
                    ),
                )
            ],
        )
    else:
        aa = a
    if T.constexpr(trans_b):
        # A partial V tile retains the full stage's column-plane stride.
        bt = b.permute(1, 0)
    else:
        bt = T.decl_tensor(
            b.ptr_to([0, 0]),
            ty_args=[
                T.Tensor(
                    (int(b.shape[0]), int(b.shape[1])),
                    "float16",
                    scope="shared.dyn",
                    align=1024,
                    layout=mma_shared_layout("float16", 3, tuple(b.shape)),
                )
            ],
        )
    T.ptx.tcgen05.fence__after_thread_sync()
    if T.cuda.elect_sync():
        Tx.apply(
            T.ptx["tcgen05.mma.cta_group::1.kind::f16"],
            dst.rearrange("(mr m) (nr n) -> mr nr m n", m=128, n=128),
            aa.rearrange("(mr m) (kr k) -> mr kr m k", m=128, k=16),
            bt.rearrange("(nr n) (kr k) -> nr kr n k", n=128, k=16),
            instruction_shape=(128, 128, 16),
            accum=accum,
            b_major="mn" if T.constexpr(trans_b) else "k",
        )


@T.inline
def load_q_tile(dst, source, batch, seq, head, bar, seq_rows, heads):
    for part in T.unroll(2):
        tile = T.decl_tensor(
            dst.ptr_to([0, part * 64]),
            ty_args=[
                T.Tensor(
                    (128, 64),
                    "float16",
                    scope="shared.dyn",
                    align=1024,
                    layout=mma_shared_layout("float16", 3, (128, 64)),
                )
            ],
        )
        Tx.apply(
            T.ptx["cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes"],
            tile.view(1, seq_rows, heads, 64),
            source,
            box=(1, seq_rows, heads, 64),
            coords=(batch, seq, head, part * 64),
            swizzle=3,
            mbar=bar,
        )


@T.inline
def load_kv_tile(dst, source, batch, seq, head, bar):
    for part in T.unroll(2):
        tile = T.decl_tensor(
            dst.ptr_to([0, part * 64]),
            ty_args=[
                T.Tensor(
                    (128, 64),
                    "float16",
                    scope="shared.dyn",
                    align=1024,
                    layout=mma_shared_layout("float16", 3, (128, 64)),
                )
            ],
        )
        Tx.apply(
            T.ptx["cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes"],
            tile.view(1, 128, 1, 64),
            source,
            box=(1, 128, 1, 64),
            coords=(batch, seq, head, part * 64),
            swizzle=3,
            mbar=bar,
        )


@T.inline
def store_o_tile(destination, src, batch, seq, head, seq_rows, heads):
    for part in T.unroll(2):
        tile = T.decl_tensor(
            src.ptr_to([0, part * 64]),
            ty_args=[
                T.Tensor(
                    (128, 64),
                    "float16",
                    scope="shared.dyn",
                    align=1024,
                    layout=mma_shared_layout("float16", 3, (128, 64)),
                )
            ],
        )
        Tx.apply(
            T.ptx["cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"],
            destination,
            tile.view(1, seq_rows, heads, 64),
            box=(1, seq_rows, heads, 64),
            coords=(batch, seq, head, part * 64),
            swizzle=3,
        )


def ceildiv(a, b):
    return (a + b - 1) // b


def shl_u32_clamp(val, shift):
    """Left shift with PTX clamping (shift>=32 -> 0). Lets the causal mask keep
    the fused ``max(col_limit - s*CHUNK, 0)`` (one VIADDMNMX, like cutedsl) and
    drop the ``min(.,CHUNK)`` clamp (which adds a VIMNMX above cutedsl's count):
    ``~shl_clamp(0xFFFFFFFF, k)`` is the low-k-bits mask for any k, no min."""
    func_name = "shl_u32_clamp"
    source_code = (
        f"\n__device__ __forceinline__ unsigned int {func_name}"
        "(unsigned int val, unsigned int shift) {\n"
        "  unsigned int r;\n"
        '  asm("shl.b32 %0, %1, %2;" : "=r"(r) : "r"(val), "r"(shift));\n'
        "  return r;\n}\n"
    )
    return T.cuda.func_call(func_name, val, shift, source_code, ty="uint32")


def combine_int_frac_ex2(x_rounded, frac_ex2):
    func_name = "combine_int_frac_ex2"
    source_code = f'\n__device__ __forceinline__ float {func_name}(float x_rounded, float frac_ex2) {{\n  float out;\n  asm volatile(\n    "{{\\n\\t"\n    ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\\n\\t"\n    "mov.b32 x_rounded_i, %1;\\n\\t"\n    "mov.b32 frac_ex_i, %2;\\n\\t"\n    "shl.b32 x_rounded_e, x_rounded_i, 23;\\n\\t"\n    "add.s32 out_i, x_rounded_e, frac_ex_i;\\n\\t"\n    "mov.b32 %0, out_i;\\n\\t"\n    "}}\\n"\n    : "=f"(out) : "f"(x_rounded), "f"(frac_ex2));\n  return out;\n}}\n'
    return T.cuda.func_call(func_name, x_rounded, frac_ex2, source_code, ty="float32")


def get_n_block_max(m_block_idx, causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE):
    """Maximum KV block index (exclusive) for this Q block."""
    n_block_max = ceildiv(SEQ_LEN_KV, BLK_N)
    if not causal:
        return n_block_max
    m_idx_max = (m_block_idx + 1) * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
    n_idx = m_idx_max + SEQ_LEN_KV - SEQ_LEN_Q
    return T.min(n_block_max, ceildiv(n_idx, BLK_N))


def get_n_block_min_causal_mask(m_block_idx, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE):
    """KV block index where causal masking stops being needed."""
    m_idx_min = m_block_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
    n_idx = m_idx_min + SEQ_LEN_KV - SEQ_LEN_Q
    return T.max(0, n_idx // BLK_N)


@T.inline
def ex2_emulation_2(out, idx, x, y):
    poly_ex2_deg3 = T.meta_var((1.0, 0.6951461434364319, 0.22756439447402954, 0.07711908966302872))
    fp32_round_int = T.meta_var(float(2**23 + 2**22))
    xy_clamped: T.f32[2]
    xy_clamped[0] = T.max(x, -127.0)
    xy_clamped[1] = T.max(y, -127.0)
    xy_rounded: T.f32[2]
    add_tile(xy_rounded, xy_clamped, fp32_round_int, rounding_mode="rm")
    xy_rounded_back: T.f32[2]
    sub_tile(xy_rounded_back, xy_rounded, fp32_round_int, rounding_mode="rn")
    xy_frac: T.f32[2]
    sub_tile(xy_frac, xy_clamped, xy_rounded_back, rounding_mode="rn")
    xy_frac_ex2: T.f32[2]
    xy_frac_ex2[0] = poly_ex2_deg3[3]
    xy_frac_ex2[1] = poly_ex2_deg3[3]
    fma_tile(xy_frac_ex2, xy_frac_ex2, xy_frac, poly_ex2_deg3[2])
    fma_tile(xy_frac_ex2, xy_frac_ex2, xy_frac, poly_ex2_deg3[1])
    fma_tile(xy_frac_ex2, xy_frac_ex2, xy_frac, poly_ex2_deg3[0])
    out[idx] = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0])
    out[idx + 1] = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1])


WG_NUMBER = 4
WARP_NUMBER = 4
NUM_THREADS = 32 * WARP_NUMBER * WG_NUMBER
N_COLS_TMEM = 512
TMEM_PIPE_DEPTH = 2
SMEM_PIPE_DEPTH_Q = 2
SMEM_PIPE_DEPTH_KV = 3
BLK_M = 128
BLK_N = 128
BLK_K = 64
SOFTMAX_LD_CHUNK = 32
SOFTMAX_ST_CHUNK = 32
EPI_TILE = 64
TMEM_EPI_LD_SIZE = 16
USE_S0_S1_BARRIER = False
MMA_M = 128
MMA_N = 128
MMA_K = 16
F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16
a_type_qk = "float16"
b_type_qk = "float16"
d_type_qk = "float32"
a_type_pv = "float16"
b_type_pv = "float16"
d_type_pv = "float32"


def _kernel(
    BATCH_SIZE,
    SEQ_LEN_Q,
    SEQ_LEN_KV,
    NUM_QO_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    is_causal=False,
    TMEM_PIPE_DEPTH=2,
    SMEM_PIPE_DEPTH_KV=3,
    reg_level="10",
):
    CTA_GROUP = 1
    GQA_RATIO = NUM_QO_HEADS // NUM_KV_HEADS
    SEQ_Q_PER_TILE = BLK_M // GQA_RATIO
    STATS_BAR_PAIRWISE = GQA_RATIO == 1
    L2_SIZE = 50 * 1024 * 1024
    SIZE_ONE_KV_HEAD = SEQ_LEN_KV * HEAD_DIM * 2 * F16_BYTES
    L2_SWIZZLE = (
        1 if L2_SIZE < SIZE_ONE_KV_HEAD else 1 << int(math.log2(L2_SIZE // SIZE_ONE_KV_HEAD))
    )
    SSCALE_TOTAL_SIZE = 2 * SMEM_PIPE_DEPTH_Q * BLK_M
    num_q_blocks_total = ceildiv(SEQ_LEN_Q, SEQ_Q_PER_TILE)
    num_q_blocks = ceildiv(num_q_blocks_total, SMEM_PIPE_DEPTH_Q)
    num_total_tasks = BATCH_SIZE * NUM_KV_HEADS * num_q_blocks
    EPI_ON_SOFTMAX = is_causal
    EARLY_Q_RELEASE = not is_causal
    cta_count = num_total_tasks if is_causal else min(148, num_total_tasks)
    K_SPLIT = (4 if is_causal else 6) * MMA_K
    P_SPLIT_Q = 2 if is_causal else 3
    scale_log2 = math.log2(math.e) / math.sqrt(HEAD_DIM)
    rescale_threshold = 8.0
    EPI_LD_SM = 32
    RESCALE_TILE = 16
    emu_pairs = EMU_PAIRS_CAUSAL if is_causal else EMU_PAIRS_NC
    emu_start = EMU_START_CAUSAL if is_causal else EMU_START_NC

    @T.function
    def kernel(
        Q: T.Tensor((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"),
        K: T.Tensor((BATCH_SIZE, SEQ_LEN_KV, NUM_KV_HEADS, HEAD_DIM), "float16"),
        V: T.Tensor((BATCH_SIZE, SEQ_LEN_KV, NUM_KV_HEADS, HEAD_DIM), "float16"),
        O: T.Tensor((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"),
    ):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=cta_count, block=512),
            kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=1),
            backend_config={"cuda": {"ptxas": [f"--register-usage-level={reg_level}"]}},
        )
        bx = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        warp_id = T.cuda.warp_in_warpgroup()
        tid_in_wg = T.cuda.thread_idx("x") % 128
        pool = SMEMPool()
        Q_smem = pool.alloc_tcgen05_mma_AB((SMEM_PIPE_DEPTH_Q, BLK_M, HEAD_DIM), "float16")
        K_smem = pool.alloc_tcgen05_mma_AB((SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM), "float16")
        V_smem = K_smem.view(SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM)
        O_smem = pool.alloc_tcgen05_mma_AB((TMEM_PIPE_DEPTH, BLK_M, HEAD_DIM), "float16")
        sScale = pool.alloc((SSCALE_TOTAL_SIZE,), "float32", align=1024)
        tmem_addr = pool.alloc([1], "uint32")
        ACC_SCALE_BASE: T.let = 0
        ROW_SUM_BASE: T.let = 0
        kv_pipe = PipelineState(SMEM_PIPE_DEPTH_KV)
        phase_q = T.alloc_local((1,), "int32")
        phase_s_full = T.alloc_local((1,), "int32")
        phase_tmem = T.alloc_local((1,), "int32")
        phase_s0_s1 = T.alloc_local((1,), "int32")
        phase_q_load = T.alloc_local((1,), "int32")
        phase_oepi = T.alloc_local((1,), "int32")
        q_load = Pipeline(
            pool, SMEM_PIPE_DEPTH_Q, full="tma", empty="tcgen05", empty_phase_offset=1
        )
        kv_load = Pipeline(
            pool, SMEM_PIPE_DEPTH_KV, full="tma", empty="tcgen05", empty_phase_offset=1
        )
        p_o_rescale = MBarrier(pool, 2)
        p_o_rescale.init(256)
        s_ready = TCGen05Bar(pool, 2)
        s_ready.init(1)
        o_ready = TCGen05Bar(pool, 2)
        o_ready.init(1)
        softmax_corr = Pipeline(
            pool, 2, full="mbar", empty="mbar", init_full=128, init_empty=128, empty_phase_offset=1
        )
        corr_epi = Pipeline(
            pool,
            TMEM_PIPE_DEPTH,
            full="mbar",
            empty="mbar",
            init_full=128,
            init_empty=32,
            empty_phase_offset=1,
        )
        p_ready_2 = MBarrier(pool, 2)
        p_ready_2.init(128)
        bar_s0_s1_sequence = MBarrier(pool, 8)
        # Init in the FIRST init group so the single prologue fence (below) covers
        # it; lets us drop the redundant 2nd fence+cta_sync (MEMBAR/FENCE/BSSY
        # excess vs cutedsl, which uses one prologue init barrier).
        bar_s0_s1_sequence.init(32)
        pool.commit()
        iket = IketProfiler()
        # TMEM is allocated by the MMA warp (cta-scope warp 12 = wg3/warp0) and
        # the alloc deliberately sits AFTER the prologue cta_syncs (commit is
        # emitted further down): every other warp's first TMEM access is
        # transitively gated behind this warp's progress (softmax via s_ready
        # from the first QK gemm, correction via the softmax handshake, TMA
        # store via corr_epi), so no CTA-wide sync on the alloc is needed, and
        # the TMA load warp issues the first Q/K copies without waiting for it.
        # FA4 CuTeDSL structures its prologue the same way (alloc inside the
        # MMA warp's branch; load warps never wait for it).
        # The original kernel requires the full 512-column allocation at base zero.
        # Its first TMEM accesses wait transitively for the MMA warp's allocation.
        tmem = T.cuda.decl_tmem(
            T.uint32(0),
            ty_args=[
                T.Tensor(
                    (128, N_COLS_TMEM),
                    "float32",
                    scope="tmem",
                    layout=TileLayout(S[(128, N_COLS_TMEM) : (1 @ TLane, 1 @ TCol)]),
                )
            ],
        )
        tmem_as_f16 = tmem.view("float16")
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()
        # S and O share the (128, N_COLS_TMEM) f32 tmem as 2*SMEM_PIPE_DEPTH_Q
        # MMA_N-wide stages: S in the low SMEM_PIPE_DEPTH_Q, O in the high ones
        # (indexed SMEM_PIPE_DEPTH_Q + stage). Indexing a constant stage at the use
        # site keeps the column offset in the region so allocated_addr stays 0.
        S_region = T.meta_var(tmem.rearrange("m (s n) -> s m n", n=MMA_N))
        O_region = S_region
        # P overlays the f16 view of the S stages: the high MMA_N-wide half (two=1)
        # of each stage's 2*MMA_N f16 columns, indexed [stage, 1, :, cols].
        P_region = T.meta_var(tmem_as_f16.rearrange("m (s two n) -> s two m n", two=2, n=MMA_N))
        scheduler = (
            FlashAttentionLPTScheduler(
                "fa_scheduler",
                num_batches=BATCH_SIZE,
                num_heads=NUM_KV_HEADS,
                num_m_blocks=num_q_blocks,
                l2_swizzle=L2_SWIZZLE,
            )
            if is_causal
            else FlashAttentionLinearScheduler(
                "fa_scheduler",
                num_batches=BATCH_SIZE,
                num_heads=NUM_KV_HEADS,
                num_m_blocks=num_q_blocks,
                num_ctas=cta_count,
            )
        )
        scheduler.init(bx)
        kv_pipe.init(0)
        phase_q[0] = 0
        phase_oepi[0] = 0
        phase_tmem[0] = 0
        phase_s_full[0] = 0
        if T.constexpr(USE_S0_S1_BARRIER):
            phase_s0_s1[0] = T.if_then_else(wg_id == 1, 0, 1)
        phase_q_load[0] = 0
        if (wg_id == 3) & (warp_id == 0):
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                tmem_addr.ptr_to([0]), T.uint32(N_COLS_TMEM)
            )
            T.cuda.warp_sync()
        if (wg_id == 3) & (warp_id == 0):
            T.cuda.trap_when_assert_failed(tmem_addr[0] == T.uint32(0))
        if wg_id == 2:
            for i_q in T.unroll(2):
                p_o_rescale.arrive(i_q)
        num_kv_blocks: T.let = ceildiv(SEQ_LEN_KV, BLK_N)
        while scheduler.valid():
            m_block_idx = T.meta_var(scheduler.m_block_idx)
            batch_idx = T.meta_var(scheduler.batch_idx)
            kv_head_idx = T.meta_var(scheduler.head_idx)
            m_start = T.meta_var(m_block_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q)
            if wg_id == 3:
                T.ptx.setmaxnreg.dec.sync.aligned.u32(48)
                if warp_id == 1:

                    @T.inline
                    def load_q(i_q):
                        q_load.empty.wait(i_q, phase_q_load[0])
                        tma_q_token: T.uint32 = iket.range_start("issue-tma-q")
                        Q_smem_4d = Q_smem.view(
                            SMEM_PIPE_DEPTH_Q, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM
                        )
                        if T.cuda.elect_sync():
                            load_q_tile(
                                Q_smem.sub[i_q, :, :],
                                Q,
                                batch_idx,
                                m_start + i_q * SEQ_Q_PER_TILE,
                                kv_head_idx * GQA_RATIO,
                                q_load.full.ptr_to([i_q]),
                                SEQ_Q_PER_TILE,
                                GQA_RATIO,
                            )
                            q_load.full.arrive(i_q, CTA_GROUP * BLK_M * HEAD_DIM * F16_BYTES)
                        iket.range_end(tma_q_token)

                    @T.inline
                    def load_k(i_kv):
                        kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase)
                        tma_k_token: T.uint32 = iket.range_start("issue-tma-k")
                        if T.cuda.elect_sync():
                            load_kv_tile(
                                K_smem.sub[kv_pipe.stage % SMEM_PIPE_DEPTH_KV, :, :],
                                K,
                                batch_idx,
                                i_kv * BLK_N,
                                kv_head_idx,
                                kv_load.full.ptr_to([kv_pipe.stage % SMEM_PIPE_DEPTH_KV]),
                            )
                            kv_load.full.arrive(
                                kv_pipe.stage, CTA_GROUP * BLK_N * HEAD_DIM * F16_BYTES
                            )
                        iket.range_end(tma_k_token)
                        kv_pipe.advance()

                    @T.inline
                    def load_v(i_kv):
                        kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase)
                        tma_v_token: T.uint32 = iket.range_start("issue-tma-v")
                        if T.cuda.elect_sync():
                            load_kv_tile(
                                V_smem.sub[kv_pipe.stage % SMEM_PIPE_DEPTH_KV, :, :],
                                V,
                                batch_idx,
                                i_kv * BLK_N,
                                kv_head_idx,
                                kv_load.full.ptr_to([kv_pipe.stage % SMEM_PIPE_DEPTH_KV]),
                            )
                            kv_load.full.arrive(
                                kv_pipe.stage, CTA_GROUP * BLK_N * HEAD_DIM * F16_BYTES
                            )
                        iket.range_end(tma_v_token)
                        kv_pipe.advance()

                    load_trip_count: T.int32
                    load_trip_count = (
                        get_n_block_max(
                            m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE
                        )
                        if is_causal
                        else num_kv_blocks
                    )
                    load_q(0)
                    load_k(load_trip_count - 1)
                    load_q(1)
                    phase_q_load[0] ^= 1
                    load_v(load_trip_count - 1)
                    for _i in T.serial(load_trip_count - 1, unroll=False):
                        i_kv: T.let = load_trip_count - 2 - _i
                        load_k(i_kv)
                        load_v(i_kv)
                if warp_id == 2:
                    corr_epi.full.wait(0, phase_tmem[0])
                    tma_store_token: T.uint32 = iket.range_start("tma-store")
                    for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                        if i_q != 0:
                            corr_epi.full.wait(i_q, phase_tmem[0])
                        m_start_global = T.meta_var(m_start + i_q * SEQ_Q_PER_TILE)
                        O_smem_4d = O_smem.view(
                            TMEM_PIPE_DEPTH, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM
                        )
                        if T.cuda.elect_sync():
                            store_o_tile(
                                O,
                                O_smem.sub[i_q, :, :],
                                batch_idx,
                                m_start_global,
                                kv_head_idx * GQA_RATIO,
                                SEQ_Q_PER_TILE,
                                GQA_RATIO,
                            )
                        T.ptx.cp.async_.bulk.commit_group()
                    for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                        T.ptx.cp.async_.bulk.wait_group(1 - i_q)
                        corr_epi.empty.arrive(i_q)
                    iket.range_end(tma_store_token)
                    phase_tmem[0] ^= 1
                if warp_id == 0:
                    acc: T.int32
                    acc = 0

                    @T.inline
                    def gemm_qk(q_stage, kv_stage):
                        gemm_tile(
                            S_region.sub[q_stage, :, :],
                            Q_smem.sub[q_stage, 0:BLK_M, 0:HEAD_DIM],
                            K_smem.sub[kv_stage, 0:BLK_N, 0:HEAD_DIM],
                            accum=False,
                            trans_b=False,
                        )
                        if T.cuda.elect_sync():
                            s_ready.arrive(q_stage)

                    # PV gemm split point, regime-tuned: for causal, 64 cols
                    # (p_o_rescale waits TWO P quarter-stores instead of three,
                    # releasing the MMA warp earlier in the
                    # softmax->P->PV->QK->s_ready loop that binds the small-shape
                    # cadence; single-task phase tables vs cutedsl showed TIRx
                    # softmax idling ~26% of its cadence on s_ready vs baseline
                    # 14%). For non-causal the earlier release measured 2-3pp
                    # WORSE on s1024 shapes — keep 96.

                    @T.inline
                    def gemm_pv_part1(i_q, kv_stage, should_accumulate):
                        v_stage = T.decl_tensor(
                            V_smem.ptr_to([kv_stage % SMEM_PIPE_DEPTH_KV, 0, 0]),
                            ty_args=[
                                T.Tensor(
                                    (128, 128),
                                    "float16",
                                    scope="shared.dyn",
                                    align=1024,
                                    layout=mma_shared_layout("float16", 3, (128, 128)),
                                )
                            ],
                        )
                        gemm_tile(
                            O_region.sub[SMEM_PIPE_DEPTH_Q + i_q, :, :],
                            P_region.sub[i_q, 1, :, 0:K_SPLIT],
                            v_stage.sub[0:K_SPLIT, 0:HEAD_DIM],
                            accum=should_accumulate,
                            trans_b=True,
                        )

                    @T.inline
                    def gemm_pv_part2(i_q, kv_stage):
                        v_stage = T.decl_tensor(
                            V_smem.ptr_to([kv_stage % SMEM_PIPE_DEPTH_KV, 0, 0]),
                            ty_args=[
                                T.Tensor(
                                    (128, 128),
                                    "float16",
                                    scope="shared.dyn",
                                    align=1024,
                                    layout=mma_shared_layout("float16", 3, (128, 128)),
                                )
                            ],
                        )
                        p_ready_2.wait(i_q, phase_tmem[0])
                        gemm_tile(
                            O_region.sub[SMEM_PIPE_DEPTH_Q + i_q, :, :],
                            P_region.sub[i_q, 1, :, K_SPLIT:BLK_N],
                            v_stage.sub[K_SPLIT:BLK_N, 0:HEAD_DIM],
                            accum=True,
                            trans_b=True,
                        )

                    @T.inline
                    def gemm_pv(i_q, kv_stage, should_accumulate):
                        gemm_pv_part1(i_q, kv_stage, should_accumulate)
                        gemm_pv_part2(i_q, kv_stage)

                    for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                        q_load.full.wait(i_q, phase_q_load[0])
                        if i_q == 0:
                            kv_load.full.wait(kv_pipe.stage, kv_pipe.phase)
                        gemm_qk(i_q, kv_pipe.stage)
                        if i_q == 1:
                            if T.cuda.elect_sync():
                                kv_load.empty.arrive(kv_pipe.stage)
                    kv_pipe.advance()
                    mma_trip_count: T.int32
                    mma_trip_count = (
                        get_n_block_max(
                            m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE
                        )
                        if is_causal
                        else num_kv_blocks
                    )
                    for i_kv in T.serial(mma_trip_count - 1, unroll=False):
                        stage_v: T.let = kv_pipe.stage
                        phase_v: T.let = kv_pipe.phase
                        kv_pipe.advance()
                        stage_k: T.let = kv_pipe.stage
                        phase_k: T.let = kv_pipe.phase
                        for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                            if i_q == 0:
                                kv_load.full.wait(stage_v, phase_v)
                            p_o_rescale.wait(i_q, phase_tmem[0])
                            gemm_pv(i_q, stage_v, acc)
                            if i_q == 1:
                                if T.cuda.elect_sync():
                                    kv_load.empty.arrive(stage_v)
                            if i_q == 0:
                                kv_load.full.wait(stage_k, phase_k)
                            gemm_qk(i_q, stage_k)
                            # Early Q release (non-causal / multi-task CTAs):
                            # Q[i_q]'s LAST reader is this final QK — committing
                            # here fires when it completes, ~2 tail-PV gemms
                            # before the end-of-task commit, so the next task's
                            # Q TMA load overlaps the PV tail + epilogue.
                            # Causal keeps the tail commit: one task per CTA
                            # (nothing to overlap) and packed-causal tasks can
                            # have mma_trip_count == 1, where this steady-loop
                            # site never executes.
                            if T.constexpr(EARLY_Q_RELEASE):
                                if i_kv == mma_trip_count - 2:
                                    if T.cuda.elect_sync():
                                        q_load.empty.arrive(i_q)
                            if i_q == 1:
                                if T.cuda.elect_sync():
                                    kv_load.empty.arrive(stage_k)
                        acc = 1
                        kv_pipe.advance()
                        phase_tmem[0] ^= 1
                    for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                        if i_q == 0:
                            kv_load.full.wait(kv_pipe.stage, kv_pipe.phase)
                        p_o_rescale.wait(i_q, phase_tmem[0])
                        gemm_pv(i_q, kv_pipe.stage, acc)
                        if i_q == 1:
                            if T.cuda.elect_sync():
                                kv_load.empty.arrive(kv_pipe.stage)
                        if T.cuda.elect_sync():
                            o_ready.arrive(i_q)
                    kv_pipe.advance()
                    phase_tmem[0] ^= 1
                    if T.constexpr(not EARLY_Q_RELEASE):
                        for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                            if T.cuda.elect_sync():
                                q_load.empty.arrive(i_q)
                    phase_q_load[0] ^= 1
            elif wg_id < 2:
                T.ptx.setmaxnreg.inc.sync.aligned.u32(200)

                row_max: T.f32[1]
                row_sum: T.f32[1]
                if warp_id == 0:
                    iket.mark("softmax-baseline")

                @T.inline
                def mask_r2p(s_chunk_buf, col_limit, ncol: T.int32):
                    """Apply mask using R2P-style bit manipulation.

                    Optimizes: for j in range(N): buf[j] = -inf if j >= col_limit else buf[j]
                    Into: bitmask operations that compile to R2P PTX instruction.

                    Following flash_attn/cute/mask.py mask_r2p(): process in 32-col
                    chunks (shl_u32_clamp tolerates shift>=32, so no 24-col split).

                    The bit test `mask & (1 << i)` compiles to the R2P (Register to Predicate)
                    PTX instruction, which is more efficient than per-column comparisons.
                    """

                    # Subtract-free low-k bitmask: ~(0xFFFFFFFF<<k) gives the low-k-bits
                    # mask with NO `-1`; the ~ fuses into the per-column `& (1<<i)` test
                    # as ANDN (one LOP3), so there is no VIADD per chunk. shl clamps
                    # k>=32 -> 0 => mask_inv=0 => ~=all-ones => keep all (k>=32 correct),
                    # so no VIMNMX either. The bit test compiles to R2P.
                    CHUNK_SIZE: T.let = 32
                    num_chunks: T.let = ceildiv(ncol, CHUNK_SIZE)
                    s_chunk_local = s_chunk_buf
                    for s in T.unroll(num_chunks):
                        k_keep: T.let = T.max(col_limit - s * CHUNK_SIZE, 0)
                        mask_inv: T.uint32
                        mask_inv = shl_u32_clamp(T.uint32(0xFFFFFFFF), T.cast(k_keep, "uint32"))
                        for i in T.unroll(CHUNK_SIZE):
                            if i < ncol - s * CHUNK_SIZE:
                                c: T.let = s * CHUNK_SIZE + i
                                in_bound: T.let = T.bitwise_and(
                                    T.bitwise_not(mask_inv), T.shift_left(T.uint32(1), i)
                                )
                                s_chunk_local[c] = T.Select(
                                    T.cast(in_bound, "bool"),
                                    s_chunk_local[c],
                                    T.float32(-float("inf")),
                                )

                @T.inline
                def apply_causal_mask(s_chunk_buf, m_blk_idx, n_blk_idx):
                    """Apply causal mask to attention scores.

                    Following flash_attn/cute/mask.py apply_mask_sm100() lines 384-400:
                    causal_row_offset = 1 + seqlen_k - n_block * tile_n - seqlen_q
                    row_idx = thread_row + m_block * tile_m
                    col_limit_right = row_idx + causal_row_offset
                    Mask if col >= col_limit_right

                    Coordinate Mapping:
                    - BLK_M = 128 packed rows per tile
                    - SEQ_Q_PER_TILE = BLK_M // GQA_RATIO (e.g., 32 for GQA_RATIO=4)
                    - Each warpgroup handles one Q stage with SEQ_Q_PER_TILE sequence positions
                    - tid_in_wg (0-127) maps to packed rows: (seq_pos, head) = (tid//GQA_RATIO, tid%GQA_RATIO)
                    """
                    seq_pos_in_wg: T.let = tid_in_wg // GQA_RATIO
                    row_idx: T.let = (
                        m_blk_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
                        + wg_id * SEQ_Q_PER_TILE
                        + seq_pos_in_wg
                    )
                    causal_row_offset: T.let = 1 + SEQ_LEN_KV - n_blk_idx * BLK_N - SEQ_LEN_Q
                    col_limit_right: T.let = row_idx + causal_row_offset
                    mask_r2p(s_chunk_buf, col_limit_right, BLK_N)

                @T.inline
                def softmax_step(i_kv, apply_mask=False, is_first=False):
                    s_chunk_buf: T.f32[BLK_N]
                    s_chunk = s_chunk_buf.view(128, BLK_N, layout=wg_local_layout(BLK_N))
                    p_chunk_buf_f32: T.f32[BLK_N // 2]
                    p_chunk_buf = p_chunk_buf_f32.view("float16")
                    p_chunk = p_chunk_buf.view(128, BLK_N, layout=wg_local_layout(BLK_N))
                    s_ready.wait(wg_id, phase_s_full[0])
                    if warp_id == 0:
                        iket.mark("softmax-phase-0")
                    softmax_max_token: T.uint32 = iket.sentinel_token("softmax-max")
                    if warp_id == 0:
                        softmax_max_token = iket.range_start("softmax-max")
                    tile_max: T.f32[1]
                    for chunk_idx in T.unroll(BLK_N // SOFTMAX_LD_CHUNK):
                        copy_tmem(
                            s_chunk_buf.sub[
                                chunk_idx * SOFTMAX_LD_CHUNK : (chunk_idx + 1) * SOFTMAX_LD_CHUNK
                            ],
                            S_region.sub[
                                wg_id,
                                :,
                                chunk_idx * SOFTMAX_LD_CHUNK : (chunk_idx + 1) * SOFTMAX_LD_CHUNK,
                            ],
                        )
                    if T.constexpr(apply_mask):
                        apply_causal_mask(s_chunk_buf, m_block_idx, i_kv)
                    row_max_old: T.f32
                    if T.constexpr(is_first):
                        max_tile(tile_max, s_chunk_buf)
                    else:
                        # row_max is initialized by the first softmax step.  Keep
                        # its load inside the non-first specialization so the
                        # first step does not read an uninitialized local value
                        # whose result is dead on that path.
                        row_max_old = row_max[0]
                        tile_max[0] = row_max_old
                        max_tile(tile_max, s_chunk_buf, accum=True)
                    row_max_new: T.f32
                    acc_scale: T.f32
                    acc_scale_: T.f32
                    row_max_safe: T.f32
                    row_max_new = tile_max[0]
                    row_max_safe = T.if_then_else(tile_max[0] == -float("inf"), 0.0, tile_max[0])
                    if T.constexpr(is_first):
                        acc_scale = T.float32(1.0)
                    else:
                        acc_scale_ = (row_max_old - row_max_safe) * scale_log2
                        if acc_scale_ >= -rescale_threshold:
                            row_max_new = row_max_old
                            row_max_safe = row_max_old
                            acc_scale = T.float32(1.0)
                        else:
                            acc_scale = T.exp2(acc_scale_)
                    row_max[0] = row_max_new
                    row_max_scaled: T.let = row_max_safe * scale_log2
                    if warp_id == 0:
                        iket.mark("softmax-phase-1")
                    iket.range_end(softmax_max_token)
                    if tid_in_wg < BLK_M and (not is_first):
                        sScale_idx: T.let = ACC_SCALE_BASE + tid_in_wg + wg_id * BLK_M
                        sScale[sScale_idx] = acc_scale
                    # Stats-ready handshake to the correction wg via HW named
                    # barrier (FA4 CuTeDSL sm_stats_barrier): softmax warp w of
                    # stage wg_id pairs with correction warp w on barrier
                    # 1 + wg_id*4 + w (64 threads). bar.arrive is non-blocking and
                    # the correction side's bar.sync wakes without the mbarrier
                    # try_wait + nanosleep latency — this wait gates p_o_rescale
                    # and therefore the PV MMA. The reverse (sScale slot reuse)
                    # direction stays on softmax_corr.empty.
                    if T.constexpr(STATS_BAR_PAIRWISE):
                        T.ptx.bar.arrive(T.uint32(1 + wg_id * 4 + warp_id), T.uint32(64))
                    else:
                        T.ptx.bar.arrive(T.uint32(1 + wg_id), T.uint32(256))
                    softmax_fma_token: T.uint32 = iket.sentinel_token("softmax-fma")
                    if warp_id == 0:
                        softmax_fma_token = iket.range_start("softmax-fma")
                    wg_fma(s_chunk_buf, s_chunk_buf, scale_log2, -row_max_scaled)
                    iket.range_end(softmax_fma_token)
                    if T.constexpr(USE_S0_S1_BARRIER):
                        bar_s0_s1_sequence.wait(wg_id * 4 + warp_id, phase_s0_s1[0])
                    softmax_exp2_token: T.uint32 = iket.sentinel_token("softmax-exp2")
                    if warp_id == 0:
                        softmax_exp2_token = iket.range_start("softmax-exp2")
                    for frag_idx in T.unroll(4):
                        s_chunk_local = s_chunk_buf
                        for i in T.unroll(BLK_N // 4 // 2):
                            idx = T.meta_var(frag_idx * BLK_N // 4 + 2 * i)
                            # ex2 emulation ratio is shape-tuned (cf. FA4 CuTeDSL
                            # _TUNING_CONFIG keyed on is_causal); see module-level
                            # EMU_* knobs.

                            if (
                                i * 2 % 16 < 16 - 2 * emu_pairs
                                or frag_idx >= 4 - 1
                                or frag_idx < emu_start
                                or apply_mask
                            ):
                                s_chunk_local[idx] = T.exp2(s_chunk_local[idx])
                                s_chunk_local[idx + 1] = T.exp2(s_chunk_local[idx + 1])
                            else:
                                ex2_emulation_2(
                                    s_chunk_local, idx, s_chunk_local[idx], s_chunk_local[idx + 1]
                                )
                        wg_cast(
                            p_chunk_buf.sub[frag_idx * BLK_N // 4 : (frag_idx + 1) * BLK_N // 4],
                            s_chunk_buf.sub[frag_idx * BLK_N // 4 : (frag_idx + 1) * BLK_N // 4],
                        )
                    if T.constexpr(USE_S0_S1_BARRIER):
                        bar_s0_s1_sequence.arrive((1 - wg_id) * 4 + warp_id)
                    iket.range_end(softmax_exp2_token)
                    softmax_tmem_st_token: T.uint32 = iket.sentinel_token("softmax-tmem-st")
                    if warp_id == 0:
                        softmax_tmem_st_token = iket.range_start("softmax-tmem-st")

                    for i in T.unroll(P_SPLIT_Q):
                        copy_tmem(
                            P_region.sub[wg_id, 1, :, i * BLK_N // 4 : (i + 1) * BLK_N // 4],
                            p_chunk_buf.sub[i * BLK_N // 4 : (i + 1) * BLK_N // 4],
                        )
                    T.ptx.tcgen05.wait__st.sync.aligned()
                    p_o_rescale.arrive(wg_id)
                    for i in T.unroll(4 - P_SPLIT_Q):
                        copy_tmem(
                            P_region.sub[
                                wg_id,
                                1,
                                :,
                                (P_SPLIT_Q + i) * BLK_N // 4 : (P_SPLIT_Q + 1 + i) * BLK_N // 4,
                            ],
                            p_chunk_buf.sub[
                                (P_SPLIT_Q + i) * BLK_N // 4 : (P_SPLIT_Q + 1 + i) * BLK_N // 4
                            ],
                        )
                    if warp_id == 0:
                        iket.mark("softmax-phase-2")
                    T.ptx.tcgen05.wait__st.sync.aligned()
                    p_ready_2.arrive(wg_id)
                    if warp_id == 0:
                        iket.mark("softmax-phase-3")
                    iket.range_end(softmax_tmem_st_token)
                    softmax_corr.empty.wait(wg_id, phase_q[0])
                    if warp_id == 0:
                        iket.mark("softmax-phase-4")
                    softmax_sum_token: T.uint32 = iket.sentinel_token("softmax-sum")
                    if warp_id == 0:
                        softmax_sum_token = iket.range_start("softmax-sum")
                    phase_s_full[0] ^= 1
                    phase_q[0] ^= 1
                    if T.constexpr(is_first):
                        sum_tile(row_sum, s_chunk_buf)
                    else:
                        row_sum[0] = row_sum[0] * acc_scale
                        sum_tile(row_sum, s_chunk_buf, accum=True)
                    if warp_id == 0:
                        iket.mark("softmax-phase-5")
                    iket.range_end(softmax_sum_token)
                    if T.constexpr(USE_S0_S1_BARRIER):
                        phase_s0_s1[0] ^= 1

                # Pre-loop empty.wait guards this task's first sScale write
                # against the PREVIOUS task's tail row_sum read. Under
                # EPI_ON_SOFTMAX there is no tail row_sum slot use and no
                # previous task (one task per CTA), and correction's tail
                # empty credits are gone too — the wait must go with them
                # (credit/wait audit: n arrives vs n in-step waits per slot).
                if T.constexpr(not EPI_ON_SOFTMAX):
                    softmax_corr.empty.wait(wg_id, phase_q[0])
                # Keep the phase flip even when the wait is gone: parities are
                # absolute, and correction's prologue empty.arrive can land
                # BEFORE this wg reaches its step-1 wait — starting the in-step
                # waits at phase 1 makes step-1 consume that prologue credit
                # exactly like the original protocol (synccheck-verified; the
                # phase-0 "free pass" is NOT sticky and deadlocks if the flip
                # outruns the waiter).
                phase_q[0] ^= 1
                n_block_max: T.let = get_n_block_max(
                    m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE
                )
                n_block_min_causal: T.let = (
                    get_n_block_min_causal_mask(m_block_idx, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE)
                    if is_causal
                    else n_block_max
                )
                softmax_step(n_block_max - 1, apply_mask=is_causal, is_first=True)
                n_block_max_after_p1: T.let = n_block_max - 1
                num_phase2_blocks: T.let = T.max(n_block_max_after_p1 - n_block_min_causal, 0)
                for i in T.serial(num_phase2_blocks, unroll=False):
                    n_block: T.let = n_block_max_after_p1 - 1 - i
                    softmax_step(n_block, apply_mask=True)
                n_block_max_after_p2: T.let = T.min(n_block_max_after_p1, n_block_min_causal)
                for i in T.serial(n_block_max_after_p2, unroll=False):
                    n_block: T.let = n_block_max_after_p2 - 1 - i
                    softmax_step(n_block, apply_mask=False)
                if T.constexpr(EPI_ON_SOFTMAX):
                    # Stage-parallel epilogue on this wg: row_sum is already in
                    # a register (no sScale round-trip / tail stats trip), and
                    # the post-loop softmax register file is idle — 32-wide
                    # TMEM loads are free (4 round trips, vs 8x16 under the
                    # correction wg's 64-reg budget).

                    o_ready.wait(wg_id, phase_oepi[0])
                    corr_epi.empty.wait(wg_id, phase_oepi[0])
                    epi_ld_tmem_token: T.uint32 = iket.sentinel_token("epi-ld-tmem")
                    if warp_id == 0:
                        epi_ld_tmem_token = iket.range_start("epi-ld-tmem")
                    acc_O_row_is_zero_or_nan: T.let = T.Or(
                        row_sum[0] == T.float32(0.0), row_sum[0] != row_sum[0]
                    )
                    norm_scale_sm: T.let = reciprocal(
                        T.Select(acc_O_row_is_zero_or_nan, T.float32(1.0), row_sum[0])
                    )
                    o_row_f32_sm = alloc_wg_tile(EPI_LD_SM)
                    o_row_f16_sm = alloc_wg_tile(EPI_LD_SM, "float16")
                    for epi_q in T.unroll(2):
                        if wg_id == epi_q:
                            for d_tile in T.unroll(ceildiv(HEAD_DIM, EPI_LD_SM)):
                                copy_tmem(
                                    o_row_f32_sm,
                                    O_region.sub[
                                        SMEM_PIPE_DEPTH_Q + epi_q,
                                        :,
                                        d_tile * EPI_LD_SM : (d_tile + 1) * EPI_LD_SM,
                                    ],
                                )
                                wg_mul(o_row_f32_sm, o_row_f32_sm, norm_scale_sm)
                                wg_cast(o_row_f16_sm, o_row_f32_sm)
                                store_shared(
                                    O_smem.sub[epi_q, :, :], o_row_f16_sm, d_tile * EPI_LD_SM
                                )
                    iket.range_end(epi_ld_tmem_token)
                    T.ptx.fence.proxy.async_.shared__cta()
                    corr_epi.full.arrive(wg_id)
                    p_o_rescale.arrive(wg_id)
                    phase_oepi[0] ^= 1
                else:
                    if tid_in_wg < BLK_M:
                        sScale[ROW_SUM_BASE + tid_in_wg + wg_id * BLK_M] = row_sum[0]
                    if T.constexpr(STATS_BAR_PAIRWISE):
                        T.ptx.bar.arrive(T.uint32(1 + wg_id * 4 + warp_id), T.uint32(64))
                    else:
                        T.ptx.bar.arrive(T.uint32(1 + wg_id), T.uint32(256))
            if wg_id == 2:
                T.ptx.setmaxnreg.dec.sync.aligned.u32(64)
                if T.constexpr(STATS_BAR_PAIRWISE):
                    T.ptx.bar.sync(T.uint32(1 + 0 * 4 + warp_id), T.uint32(64))
                else:
                    T.ptx.bar.sync(T.uint32(1 + 0), T.uint32(256))
                softmax_corr.empty.arrive(0)
                if T.constexpr(STATS_BAR_PAIRWISE):
                    T.ptx.bar.sync(T.uint32(1 + 1 * 4 + warp_id), T.uint32(64))
                else:
                    T.ptx.bar.sync(T.uint32(1 + 1), T.uint32(256))
                phase_q[0] ^= 1
                corr_trip_count: T.let = (
                    get_n_block_max(m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE)
                    if is_causal
                    else num_kv_blocks
                )
                for i_kv in T.serial(corr_trip_count - 1, unroll=False):
                    for i_q in T.unroll(2):
                        if T.constexpr(STATS_BAR_PAIRWISE):
                            T.ptx.bar.sync(T.uint32(1 + i_q * 4 + warp_id), T.uint32(64))
                        else:
                            T.ptx.bar.sync(T.uint32(1 + i_q), T.uint32(256))
                        correction_token: T.uint32 = iket.sentinel_token("correction")
                        if warp_id == 0:
                            correction_token = iket.range_start("correction")
                        acc_scale: T.f32
                        should_rescale: T.i32
                        if tid_in_wg < BLK_M:
                            acc_scale = sScale[ACC_SCALE_BASE + tid_in_wg + i_q * BLK_M]
                            should_rescale = T.Select(acc_scale < T.float32(1.0), 1, 0)
                        else:
                            should_rescale = 0
                        any_needs_rescale: T.let = vote_any(should_rescale)
                        if any_needs_rescale != 0:
                            if tid_in_wg < BLK_M:
                                o_row = alloc_wg_tile(RESCALE_TILE)
                                for d_tile in T.unroll(ceildiv(HEAD_DIM, RESCALE_TILE)):
                                    d_start: T.let = d_tile * RESCALE_TILE
                                    if d_start < HEAD_DIM:
                                        copy_tmem(
                                            o_row,
                                            O_region.sub[
                                                SMEM_PIPE_DEPTH_Q + i_q,
                                                :,
                                                d_start : d_start + RESCALE_TILE,
                                            ],
                                        )
                                        wg_mul(o_row, o_row, acc_scale)
                                        copy_tmem(
                                            O_region.sub[
                                                SMEM_PIPE_DEPTH_Q + i_q,
                                                :,
                                                d_start : d_start + RESCALE_TILE,
                                            ],
                                            o_row,
                                        )
                                T.ptx.tcgen05.wait__st.sync.aligned()
                        p_o_rescale.arrive(i_q)
                        softmax_corr.empty.arrive(1 - i_q)
                        iket.range_end(correction_token)
                    phase_q[0] ^= 1
                softmax_corr.empty.arrive(1)
                if T.constexpr(not EPI_ON_SOFTMAX):
                    for i_q in T.unroll(2):
                        if T.constexpr(STATS_BAR_PAIRWISE):
                            T.ptx.bar.sync(T.uint32(1 + i_q * 4 + warp_id), T.uint32(64))
                        else:
                            T.ptx.bar.sync(T.uint32(1 + i_q), T.uint32(256))
                        corr_row_sum: T.let = sScale[ROW_SUM_BASE + tid_in_wg + i_q * BLK_M]
                        softmax_corr.empty.arrive(i_q)
                        o_ready.wait(i_q, phase_tmem[0])
                        corr_epi.empty.wait(i_q, phase_tmem[0])
                        epi_ld_tmem_token: T.uint32 = iket.sentinel_token("epi-ld-tmem")
                        if warp_id == 0:
                            epi_ld_tmem_token = iket.range_start("epi-ld-tmem")
                        acc_O_mn_row_is_zero_or_nan: T.let = T.Or(
                            corr_row_sum == T.float32(0.0), corr_row_sum != corr_row_sum
                        )
                        norm_scale: T.let = reciprocal(
                            T.Select(acc_O_mn_row_is_zero_or_nan, T.float32(1.0), corr_row_sum)
                        )
                        o_row_f32 = alloc_wg_tile(TMEM_EPI_LD_SIZE)
                        o_row_f16 = alloc_wg_tile(TMEM_EPI_LD_SIZE, "float16")
                        for d_tile in T.unroll(ceildiv(HEAD_DIM, TMEM_EPI_LD_SIZE)):
                            d_start: T.let = d_tile * TMEM_EPI_LD_SIZE
                            if d_start < HEAD_DIM:
                                copy_tmem(
                                    o_row_f32,
                                    O_region.sub[
                                        SMEM_PIPE_DEPTH_Q + i_q,
                                        :,
                                        d_start : d_start + TMEM_EPI_LD_SIZE,
                                    ],
                                )
                                wg_mul(o_row_f32, o_row_f32, norm_scale)
                                wg_cast(o_row_f16, o_row_f32)
                                store_shared(
                                    O_smem.sub[i_q, :, :], o_row_f16, d_tile * TMEM_EPI_LD_SIZE
                                )
                        iket.range_end(epi_ld_tmem_token)
                        T.ptx.fence.proxy.async_.shared__cta()
                        corr_epi.full.arrive(i_q)
                        p_o_rescale.arrive(i_q)
                    phase_tmem[0] ^= 1
                phase_q[0] ^= 1
            scheduler.next_tile()
        T.cuda.cta_sync()
        if (wg_id == 0) & (warp_id == 0):
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(N_COLS_TMEM))

    return kernel


def get_flash_attention4_kernel(
    batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim, is_causal=False
):
    # ptxas --register-usage-level: the default 10 over-allocates under the
    # setmaxnreg caps and SPILLS (35K LDL/STL; cutedsl has 0). On the latency-
    # bound causal GQA=1 (kv32) shapes those spills sit on the critical path.
    # Clean isolated run_bench A/B (warm-L2) picks the best NON-spilling level
    # per regime (swept 3..8; >=6 re-spills): multi-wave s2048/s4096_h32kv32_c
    # want 5 (0.967->0.992, 0.974->0.993); the single-wave s1024_h32kv32_c
    # wants 4 (0.954->0.962, won all 3 rounds, s2048/s4096 unaffected ~tie).
    # GQA=2 (s1024kv16_c) and throughput-bound non-causal keep 10's ILP. Read
    # through device_entry backend_config. FA4_REG_LEVEL overrides this selection.
    _reg_override = os.environ.get("FA4_REG_LEVEL", "")
    if _reg_override:
        _reg_level = _reg_override
    elif is_causal and num_qo_heads == num_kv_heads:
        _reg_level = "4" if seq_len_q <= 1024 else "5"
    else:
        _reg_level = "10"
    # Pipeline-depth split by wave count. Single-wave causal shapes (s1024) are
    # warpgroup-handshake-bound (stall-PC: ~40% mbarrier sync); a DEEPER
    # O-accumulator TMEM pipeline (TMEM_PIPE_DEPTH 3, funded by SMEM_PIPE_DEPTH_KV
    # 2 — SMEM-neutral) hides the correction-wg handshake bubble. Clean A/B
    # (warmup=100): s1024_h32kv32_c 0.953->0.998, s1024_h32kv16_c 0.985->1.002.
    # Multi-wave shapes (s2048+) instead need KV depth 3 for the long KV stream:
    # 3/2 REGRESSES s2048 (0.991->0.951) and s4096 (0.992->0.968), so they keep
    # the default 2/3. KV depth 1 deadlocks (pipeline needs >=2 stages).
    _deep_o = is_causal and seq_len_q <= 1024
    _tmem_depth = 3 if _deep_o else TMEM_PIPE_DEPTH
    _kv_depth = 2 if _deep_o else SMEM_PIPE_DEPTH_KV
    return _kernel(
        BATCH_SIZE=batch_size,
        SEQ_LEN_Q=seq_len_q,
        SEQ_LEN_KV=seq_len_kv,
        NUM_QO_HEADS=num_qo_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        is_causal=is_causal,
        TMEM_PIPE_DEPTH=_tmem_depth,
        SMEM_PIPE_DEPTH_KV=_kv_depth,
        reg_level=_reg_level,
    )
