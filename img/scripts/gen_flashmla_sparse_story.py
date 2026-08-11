#!/usr/bin/env python3
"""Generate the selection-versus-attention story for sparse FlashMLA prefill."""

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


def _panel(ax, x, y, w, h, title, subtitle, color) -> None:
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        linewidth=1.35,
        edgecolor=color,
        facecolor="#ffffff",
        linestyle="--",
        zorder=0,
    )
    ax.add_patch(panel)
    ax.text(
        x + 0.25,
        y + h - 0.34,
        title,
        ha="left",
        va="center",
        fontsize=11.3,
        weight="bold",
        color=color,
    )
    ax.text(
        x + 0.25,
        y + h - 0.72,
        subtitle,
        ha="left",
        va="center",
        fontsize=7.8,
        color=COLORS["muted"],
    )


def draw(lang: str, output: str, font_path: str | None = None) -> None:
    configure_style(lang, font_path)
    fig, ax = plt.subplots(figsize=(18.0, 9.2))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    ax.text(
        9,
        8.86,
        tr(
            lang,
            "Who Chooses the Tokens in Sparse FlashMLA?",
            "Sparse FlashMLA 中由谁选择 Tokens？",
        ),
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
    )
    ax.text(
        9,
        8.46,
        tr(
            lang,
            "the indexer ranks candidates; sparse prefill receives row addresses and only performs attention",
            "indexer 对 candidates 排序；sparse prefill 接收 row addresses，只负责 attention",
        ),
        ha="center",
        va="center",
        fontsize=9.3,
        color=COLORS["muted"],
    )

    _panel(
        ax,
        0.35,
        1.34,
        4.55,
        6.65,
        tr(lang, "1 · Select outside the kernel", "1 · 在 kernel 外完成选择"),
        tr(
            lang,
            "lightning indexer scores candidate rows",
            "lightning indexer 为 candidate rows 打分",
        ),
        COLORS["cta0_dark"],
    )
    _panel(
        ax,
        5.18,
        1.34,
        5.80,
        6.65,
        tr(lang, "2 · Pass an address list", "2 · 传入 address list"),
        tr(
            lang,
            "one list is shared by all 128 query heads",
            "同一份 list 由全部 128 个 query heads 共享",
        ),
        "#7c3aed",
    )
    _panel(
        ax,
        11.26,
        1.34,
        6.39,
        6.65,
        tr(lang, "3 · Attend over selected rows", "3 · 对选中 rows 执行 attention"),
        tr(
            lang,
            "gather first; then run the dense tile chain",
            "先 gather，再执行规则的 tile 计算链",
        ),
        COLORS["cta1_dark"],
    )

    # Selection is an upstream operator, not part of this sparse-prefill kernel.
    rounded_box(
        ax,
        0.75,
        6.10,
        1.26,
        0.74,
        "query q",
        COLORS["neutral"],
        fontsize=8.7,
    )
    rounded_box(
        ax,
        0.75,
        4.65,
        1.26,
        0.90,
        tr(lang, "candidate\nKV rows", "candidate\nKV rows"),
        COLORS["gmem"],
        fontsize=8.4,
    )
    rounded_box(
        ax,
        2.40,
        5.12,
        1.87,
        1.30,
        tr(
            lang,
            "lightning indexer\nscore + top-k",
            "lightning indexer\nscore + top-k",
        ),
        "#fef3c7",
        fontsize=9.0,
        edgecolor="#d97706",
    )
    arrow(ax, (2.01, 6.47), (2.40, 6.05))
    arrow(ax, (2.01, 5.10), (2.40, 5.55))
    rounded_box(
        ax,
        1.03,
        3.32,
        3.18,
        0.90,
        tr(
            lang,
            "output: selected row addresses\nnot gathered K/V values",
            "输出：选中 rows 的 addresses\n不是已 gather 的 K/V values",
        ),
        COLORS["barrier"],
        fontsize=8.4,
        weight="normal",
    )
    arrow(ax, (3.34, 5.12), (2.91, 4.22))
    rounded_box(
        ax,
        0.78,
        1.78,
        3.68,
        0.78,
        tr(
            lang,
            "different operator · its own cost",
            "独立 operator · 有自己的计算成本",
        ),
        COLORS["note"],
        fontsize=8.4,
        weight="normal",
        edgecolor="#ca8a04",
    )

    # Concrete semantics for one query row.
    ax.text(
        8.08,
        6.67,
        "indices[q, 0, :]",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    values = ("42", "7", "42", "−1", "91", "13")
    states = (
        tr(lang, "valid", "有效"),
        tr(lang, "valid", "有效"),
        tr(lang, "duplicate", "重复"),
        "OOB",
        tr(lang, "valid", "有效"),
        tr(lang, "length tail", "长度 tail"),
    )
    fills = (
        COLORS["cta0"],
        COLORS["cta0"],
        COLORS["cross01"],
        "#fecaca",
        COLORS["cta0"],
        "#e5e7eb",
    )
    x0 = 5.60
    cell_w = 0.82
    for idx, (value, state, fill) in enumerate(zip(values, states, fills, strict=True)):
        x = x0 + idx * cell_w
        plain_rect(ax, x, 5.82, cell_w, 0.64, fill)
        ax.text(
            x + cell_w / 2,
            6.14,
            value,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
        )
        ax.text(
            x + cell_w / 2,
            5.56,
            state,
            ha="center",
            va="center",
            fontsize=6.5,
            color=COLORS["muted"],
        )
    ax.plot([x0, x0 + 5 * cell_w], [5.30, 5.30], color="#7c3aed", lw=1.3)
    ax.plot([x0, x0], [5.24, 5.36], color="#7c3aed", lw=1.3)
    ax.plot([x0 + 5 * cell_w, x0 + 5 * cell_w], [5.24, 5.36], color="#7c3aed", lw=1.3)
    ax.text(
        x0 + 2.5 * cell_w,
        5.10,
        "topk_length[q] = 5",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#6d28d9",
        weight="bold",
    )

    rounded_box(
        ax,
        5.70,
        4.05,
        4.82,
        0.76,
        tr(
            lang,
            "valid = 0 ≤ idx < s_kv  AND  slot < topk_length",
            "valid = 0 ≤ idx < s_kv  且  slot < topk_length",
        ),
        COLORS["barrier"],
        fontsize=8.3,
        weight="normal",
    )
    rounded_box(
        ax,
        5.82,
        2.93,
        2.18,
        0.70,
        "h_kv = 1",
        COLORS["gmem"],
        fontsize=8.8,
    )
    rounded_box(
        ax,
        8.32,
        2.93,
        2.06,
        0.70,
        "Q heads 0…127",
        COLORS["neutral"],
        fontsize=8.3,
    )
    arrow(
        ax,
        (8.00, 3.28),
        (8.32, 3.28),
        label=tr(lang, "same list", "同一份 list"),
        color="#7c3aed",
        label_offset=(0.0, 0.17),
    )
    ax.text(
        8.08,
        2.20,
        tr(
            lang,
            "duplicates participate twice; invalid slots are masked",
            "重复 row 会参与两次；invalid slots 被 mask",
        ),
        ha="center",
        va="center",
        fontsize=8.1,
        color=COLORS["muted"],
        weight="bold",
    )

    # The kernel turns irregular addresses into regular QK/softmax/PV tiles.
    rounded_box(
        ax,
        11.66,
        6.11,
        1.46,
        0.74,
        "Q\n[128, d_qk]",
        COLORS["neutral"],
        fontsize=8.0,
    )
    rounded_box(
        ax,
        11.66,
        4.72,
        1.46,
        0.92,
        "KV cache\n[s_kv, 1, d_qk]",
        COLORS["gmem"],
        fontsize=7.7,
    )
    rounded_box(
        ax,
        13.51,
        4.94,
        1.32,
        1.46,
        tr(
            lang,
            "gather\nselected rows\n+ validity",
            "gather\n选中 rows\n+ validity",
        ),
        COLORS["smem"],
        fontsize=7.8,
    )
    arrow(ax, (13.12, 5.18), (13.51, 5.40), label="rows", rad=-0.05)
    arrow(
        ax,
        (10.98, 5.86),
        (13.51, 6.07),
        label="indices",
        color="#7c3aed",
        rad=-0.06,
    )

    rounded_box(ax, 15.25, 5.95, 1.04, 0.72, "QK", COLORS["mma"], fontsize=9.0)
    rounded_box(
        ax,
        16.55,
        5.95,
        0.78,
        0.72,
        "L",
        COLORS["tmem"],
        fontsize=9.0,
    )
    arrow(ax, (13.12, 6.48), (15.25, 6.31), label="Q")
    arrow(ax, (14.83, 5.84), (15.25, 6.13), label="K", rad=-0.05)
    arrow(ax, (16.29, 6.31), (16.55, 6.31))

    rounded_box(
        ax,
        15.02,
        4.48,
        2.10,
        0.92,
        tr(
            lang,
            "mask L → softmax\n→ BF16 weights W",
            "mask L → softmax\n→ BF16 weights W",
        ),
        COLORS["softmax"],
        fontsize=8.0,
    )
    arrow(ax, (16.94, 5.95), (16.25, 5.40), label="L", rad=0.05)
    rounded_box(ax, 14.15, 3.02, 1.16, 0.76, "PV\nW × V", COLORS["mma"], fontsize=8.5)
    arrow(ax, (15.65, 4.48), (14.98, 3.78), label="W", rad=0.04)
    arrow(
        ax,
        (14.18, 4.94),
        (14.47, 3.78),
        label=tr(lang, "V = first 512 cols", "V = 前 512 cols"),
        color=COLORS["line"],
        rad=-0.12,
        label_offset=(-0.60, -0.02),
    )
    rounded_box(
        ax,
        15.72,
        2.82,
        1.52,
        1.16,
        "out [128,512]\nmax_logits\nlse",
        COLORS["gmem"],
        fontsize=7.8,
    )
    arrow(ax, (15.31, 3.40), (15.72, 3.40))

    rounded_box(
        ax,
        11.80,
        1.76,
        5.32,
        0.66,
        tr(
            lang,
            "the kernel performs no ranking; it consumes addresses and computes attention",
            "kernel 不执行排序；它消费 addresses 并计算 attention",
        ),
        COLORS["note"],
        fontsize=8.2,
        weight="normal",
        edgecolor="#ca8a04",
    )

    rounded_box(
        ax,
        1.55,
        0.30,
        14.90,
        0.66,
        tr(
            lang,
            "No causal flag exists in this interface: the caller's list determines allowed keys; the kernel only applies bounds and optional length masking.",
            "该接口没有 causal flag：允许访问哪些 keys 由 caller 的 list 决定；kernel 只应用边界检查与可选的 length mask。",
        ),
        COLORS["note"],
        fontsize=8.3,
        weight="normal",
        edgecolor="#ca8a04",
    )

    save_figure(fig, output, dpi=160)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--output", required=True)
    parser.add_argument("--font-path")
    args = parser.parse_args()
    draw(args.lang, args.output, args.font_path)
