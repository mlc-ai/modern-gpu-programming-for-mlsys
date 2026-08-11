#!/usr/bin/env python3
"""Generate the sparse FlashMLA head128 regular SMEM/TMEM data-flow diagram."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from flashmla_diagram_common import (
    COLORS,
    arrow,
    configure_style,
    plain_rect,
    rounded_box,
    save_figure,
    tr,
)


def draw(lang: str, output: str, font_path: str | None = None) -> None:
    configure_style(lang, font_path)
    fig, ax = plt.subplots(figsize=(19.0, 9.4))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    ax.text(
        9.5,
        9.08,
        tr(
            lang,
            "Sparse FlashMLA Head128 Regular: Data Residency",
            "Sparse FlashMLA Head128 Regular：数据驻留",
        ),
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
    )
    ax.text(
        9.5,
        8.68,
        tr(
            lang,
            "one CTA's view; cta_group::2 combines the pair's operand halves",
            "单个 CTA 的视图；cta_group::2 会组合 CTA pair 的 operand halves",
        ),
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    # Q path: one prefix remains in SMEM while the fixed 384-column suffix moves to TMEM.
    rounded_box(ax, 0.35, 6.63, 1.65, 0.86, "Q\nGMEM", COLORS["gmem"], fontsize=9.5)
    rounded_box(
        ax,
        2.55,
        6.43,
        2.35,
        1.25,
        tr(
            lang,
            "q_full · SMEM\n64 × d_qk per CTA",
            "q_full · SMEM\n每个 CTA 为 64 × d_qk",
        ),
        COLORS["smem"],
        fontsize=9,
    )
    rounded_box(
        ax,
        5.63,
        7.02,
        2.18,
        0.78,
        tr(
            lang,
            "Q prefix · SMEM\nd_sq = d_qk − 384",
            "Q prefix · SMEM\nd_sq = d_qk − 384",
        ),
        COLORS["smem"],
        fontsize=8.5,
    )
    rounded_box(
        ax,
        5.63,
        5.94,
        2.18,
        0.78,
        tr(lang, "q_tmem · TMEM\nfixed suffix 384", "q_tmem · TMEM\n固定 suffix 384"),
        COLORS["tmem"],
        fontsize=8.5,
    )
    rounded_box(
        ax,
        8.48,
        6.34,
        1.92,
        1.18,
        tr(lang, "QK MMA\nSS prefix + TS suffix", "QK MMA\nSS prefix + TS suffix"),
        COLORS["mma"],
        fontsize=9,
    )
    arrow(ax, (2.0, 7.06), (2.55, 7.06), label="TMA")
    arrow(ax, (4.9, 7.2), (5.63, 7.4), label=tr(lang, "stays", "保留"), rad=-0.05)
    arrow(ax, (4.9, 6.86), (5.63, 6.33), label="SMEM→TMEM", rad=0.08)
    arrow(ax, (7.81, 7.4), (8.48, 7.18))
    arrow(ax, (7.81, 6.33), (8.48, 6.68))

    # Sparse gather path and validity mask.
    rounded_box(ax, 0.35, 4.66, 1.65, 0.82, "indices\nGMEM", COLORS["gmem"], fontsize=9)
    rounded_box(
        ax,
        0.35,
        3.34,
        1.65,
        0.92,
        "KV cache\nGMEM\n[s_kv, 1, d_qk]",
        COLORS["gmem"],
        fontsize=8.5,
    )
    rounded_box(
        ax,
        2.65,
        4.56,
        2.5,
        0.92,
        tr(
            lang,
            "WG1 gathered K · SMEM\n64 slots × d_qk per CTA",
            "WG1 gather K · SMEM\n每个 CTA：64 slots × d_qk",
        ),
        COLORS["smem"],
        fontsize=8.3,
    )
    rounded_box(
        ax,
        2.65,
        3.10,
        2.5,
        0.98,
        tr(
            lang,
            "WG2 gathered V · SMEM\n128 slots × 256 features per CTA",
            "WG2 gather V · SMEM\n每个 CTA：128 slots × 256 features",
        ),
        COLORS["smem"],
        fontsize=8.1,
    )
    rounded_box(
        ax,
        5.72,
        4.45,
        1.82,
        0.93,
        tr(
            lang,
            "is_k_valid · SMEM\n2 slots × 16 bytes",
            "is_k_valid · SMEM\n2 个 slot × 16 bytes",
        ),
        COLORS["barrier"],
        fontsize=8.1,
    )
    arrow(ax, (2.0, 5.08), (2.65, 5.08), label=tr(lang, "selected slots", "选中 slots"))
    arrow(
        ax,
        (2.0, 4.76),
        (2.65, 3.83),
        color=COLORS["line"],
        rad=-0.05,
    )
    # Route mask metadata above the K box so the dependency does not cut through its label.
    ax.plot(
        [2.0, 2.25, 5.28], [5.32, 5.76, 5.76], color=COLORS["line"], lw=1.25, zorder=2
    )
    arrow(ax, (5.28, 5.76), (6.22, 5.38), color=COLORS["line"], rad=0.05)
    ax.text(
        3.82,
        5.86,
        tr(lang, "bounds + topk_length", "边界 + topk_length"),
        ha="center",
        va="center",
        fontsize=7.3,
        color=COLORS["line"],
    )
    arrow(ax, (2.0, 3.72), (2.65, 4.83), label="gather4", rad=-0.12)
    arrow(ax, (2.0, 3.72), (2.65, 3.59), label="gather4", rad=0.05)
    # K takes a low elbow under q_tmem, then enters QK from below.
    ax.plot(
        [5.15, 5.5, 7.82], [5.32, 5.61, 5.61], color=COLORS["line"], lw=1.25, zorder=2
    )
    arrow(ax, (7.82, 5.61), (8.82, 6.34), label="K", color=COLORS["line"], rad=-0.05)

    # Logits, online softmax, weights, and public statistics.
    rounded_box(
        ax,
        10.92,
        6.36,
        1.86,
        1.15,
        "tmem_p · TMEM\nL logits\n64 × 128 view",
        COLORS["tmem"],
        fontsize=8.5,
    )
    rounded_box(
        ax,
        13.25,
        6.10,
        1.94,
        1.48,
        tr(
            lang,
            "WG0 online softmax\nmask · max · exp · sum",
            "WG0 online softmax\nmask · max · exp · sum",
        ),
        COLORS["softmax"],
        fontsize=8.7,
    )
    rounded_box(
        ax,
        13.18,
        4.42,
        2.08,
        0.95,
        "s_smem_gemm · SMEM\nW weights · 64 × 128",
        COLORS["smem"],
        fontsize=8.3,
    )
    rounded_box(
        ax,
        16.0,
        6.29,
        2.05,
        1.08,
        tr(
            lang,
            "max_logits / lse\nGMEM · sink excluded",
            "max_logits / lse\nGMEM · 不含 sink",
        ),
        COLORS["gmem"],
        fontsize=8.2,
    )
    rounded_box(
        ax,
        15.62,
        5.12,
        2.35,
        0.75,
        tr(
            lang,
            "mi · real_mi · li\nWG0 registers",
            "mi · real_mi · li\nWG0 registers",
        ),
        COLORS["neutral"],
        fontsize=8.0,
    )
    rounded_box(
        ax,
        16.10,
        4.22,
        1.87,
        0.64,
        tr(lang, "optional attn_sink\nGMEM", "可选 attn_sink\nGMEM"),
        COLORS["gmem"],
        fontsize=7.7,
    )
    arrow(ax, (10.4, 6.93), (10.92, 6.93))
    arrow(ax, (12.78, 6.93), (13.25, 6.93))
    arrow(
        ax,
        (7.54, 4.92),
        (13.25, 6.42),
        label=tr(lang, "mask ready", "mask 就绪"),
        rad=-0.10,
    )
    arrow(
        ax, (14.22, 6.10), (14.22, 5.37), label=tr(lang, "cast/store W", "转换/写入 W")
    )
    arrow(ax, (15.19, 6.93), (16.0, 6.93))
    arrow(ax, (15.10, 6.36), (15.78, 5.76), color=COLORS["line"], rad=0.05)

    # PV path and epilogue.
    rounded_box(
        ax,
        8.52,
        2.96,
        1.78,
        1.08,
        tr(lang, "PV MMA\nW × V", "PV MMA\nW × V"),
        COLORS["mma"],
        fontsize=9.2,
    )
    rounded_box(
        ax,
        10.92,
        2.92,
        1.86,
        1.14,
        "o_tmem · TMEM\nO accumulator\n64 × 512 view",
        COLORS["tmem"],
        fontsize=8.4,
    )
    rounded_box(
        ax,
        13.25,
        2.92,
        1.94,
        1.14,
        tr(
            lang,
            "WG0 epilogue\nO~ / (li + sink term)\n→ bf16",
            "WG0 epilogue\nO~ /（li + sink 项）\n→ bf16",
        ),
        COLORS["softmax"],
        fontsize=7.7,
    )
    rounded_box(
        ax,
        15.65,
        2.92,
        1.55,
        1.14,
        "o_smem\nSMEM\n64 × 512",
        COLORS["smem"],
        fontsize=8.3,
    )
    rounded_box(ax, 17.64, 3.04, 1.05, 0.9, "out\nGMEM", COLORS["gmem"], fontsize=8.7)
    arrow(ax, (5.15, 3.59), (8.52, 3.50), label="V")
    arrow(ax, (13.18, 4.86), (10.0, 4.04), label="W", rad=0.10)
    arrow(ax, (10.3, 3.5), (10.92, 3.5))
    arrow(ax, (12.78, 3.5), (13.25, 3.5), label="O~")
    arrow(
        ax,
        (16.05, 5.12),
        (14.84, 4.06),
        label="li",
        color=COLORS["line"],
        rad=0.05,
    )
    arrow(
        ax,
        (16.10, 4.54),
        (15.19, 3.75),
        label=tr(lang, "denominator only", "仅进入分母"),
        color=COLORS["line"],
        rad=0.03,
        label_offset=(0.30, -0.04),
    )
    arrow(ax, (15.19, 3.5), (15.65, 3.5), label="TMEM→SMEM")
    arrow(ax, (17.2, 3.5), (17.64, 3.5), label="TMA")

    # Exact lifetime aliasing from the pinned SharedMemoryPlan.
    outer = FancyBboxPatch(
        (0.42, 0.22),
        18.16,
        1.82,
        boxstyle="round,pad=0.05,rounding_size=0.06",
        linewidth=1.2,
        linestyle="--",
        edgecolor="#7c3aed",
        facecolor="#faf5ff",
        zorder=1,
    )
    ax.add_patch(outer)
    ax.text(
        0.72,
        1.80,
        tr(
            lang,
            "Per-CTA SMEM union: the same address range is reused by lifetime",
            "每个 CTA 的 SMEM union：同一地址范围按生命周期复用",
        ),
        ha="left",
        va="center",
        fontsize=10,
        weight="bold",
        color="#6d28d9",
        zorder=3,
    )
    rounded_box(
        ax,
        0.72,
        0.52,
        3.0,
        0.88,
        tr(
            lang,
            "prologue\nq_full = prefix | suffix384",
            "prologue\nq_full = prefix | suffix384",
        ),
        COLORS["smem"],
        fontsize=8.3,
    )
    arrow(
        ax, (3.72, 0.96), (4.36, 0.96), label=tr(lang, "reuse", "复用"), color="#7c3aed"
    )
    plain_rect(ax, 4.42, 0.52, 1.65, 0.88, COLORS["smem"], edgecolor=COLORS["ink"])
    plain_rect(ax, 6.07, 0.52, 2.0, 0.88, "#ddd6fe", edgecolor=COLORS["ink"])
    plain_rect(ax, 8.07, 0.52, 2.0, 0.88, "#c4b5fd", edgecolor=COLORS["ink"])
    ax.text(
        5.24,
        0.96,
        tr(lang, "Q prefix\nlive", "Q prefix\n存活"),
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
        zorder=4,
    )
    ax.text(
        7.07,
        0.96,
        tr(lang, "one V\nworkspace", "单份 V\nworkspace"),
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
        zorder=4,
    )
    ax.text(
        9.07,
        0.96,
        tr(lang, "one K\nworkspace", "单份 K\nworkspace"),
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
        zorder=4,
    )
    ax.text(
        7.24,
        0.32,
        tr(
            lang,
            "steady-state views are adjacent, not duplicate stages",
            "稳态 views 相邻，并非复制的 stages",
        ),
        ha="center",
        fontsize=7.3,
        color=COLORS["muted"],
    )
    arrow(
        ax,
        (10.07, 0.96),
        (10.75, 0.96),
        label=tr(lang, "reuse", "复用"),
        color="#7c3aed",
    )
    rounded_box(
        ax,
        10.82,
        0.52,
        2.72,
        0.88,
        tr(lang, "epilogue\no_smem · 64 × 512", "epilogue\no_smem · 64 × 512"),
        COLORS["smem"],
        fontsize=8.3,
    )
    ax.text(
        14.0,
        0.96,
        tr(
            lang,
            "K and V are adjacent in the steady-state region.\nThat region reuses Q's tail; later o_smem reuses the union base for the O epilogue.",
            "K 与 V 位于相邻的稳态区域。\n该区域复用 Q tail；随后 o_smem 再复用 union base 完成 O epilogue。",
        ),
        ha="left",
        va="center",
        fontsize=8.1,
        color=COLORS["muted"],
    )

    save_figure(fig, output, dpi=160)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--output", required=True)
    parser.add_argument("--font-path")
    args = parser.parse_args()
    draw(args.lang, args.output, args.font_path)
