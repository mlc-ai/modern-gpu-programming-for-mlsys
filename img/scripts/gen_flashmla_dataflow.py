#!/usr/bin/env python3
"""Generate the sparse FlashMLA head128 regular SMEM/TMEM data-flow diagram."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from flashmla_diagram_common import (
    COLORS,
    arrow as _arrow,
    configure_style,
    plain_rect,
    rounded_box as _rounded_box,
    save_figure,
    tr,
)


def box(ax, x, y, w, h, text, color, *, fontsize=11.0, **kwargs):
    """Draw a node sized for the book's roughly 790 px content column."""

    return _rounded_box(
        ax,
        x,
        y,
        w,
        h,
        text,
        color,
        fontsize=fontsize,
        **kwargs,
    )


def edge(
    ax,
    start,
    end,
    *,
    label=None,
    label_offset=(0.0, 0.14),
    fontsize=10.0,
    **kwargs,
):
    """Draw an edge and place its label in surrounding whitespace."""

    line_color = kwargs.get("color") or COLORS["line"]
    zorder = kwargs.get("zorder", 2)
    patch = _arrow(ax, start, end, label=None, **kwargs)
    if label:
        ax.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=line_color,
            bbox=dict(
                boxstyle="round,pad=0.10",
                facecolor="white",
                edgecolor="none",
                alpha=0.94,
            ),
            zorder=zorder + 1,
        )
    return patch


def panel(ax, x, y, w, h, title, subtitle):
    """Draw one readable layer of the single data-flow figure."""

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.05,rounding_size=0.06",
        linewidth=1.15,
        linestyle="--",
        edgecolor="#c4b5fd",
        facecolor="#ffffff",
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.20,
        y + h - 0.24,
        title,
        ha="left",
        va="center",
        fontsize=13.0,
        weight="bold",
        color="#6d28d9",
    )
    ax.text(
        x + w - 0.20,
        y + h - 0.24,
        subtitle,
        ha="right",
        va="center",
        fontsize=10.0,
        color=COLORS["muted"],
    )


def draw(lang: str, output: str, font_path: str | None = None) -> None:
    configure_style(lang, font_path)
    fig, ax = plt.subplots(figsize=(12.0, 13.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 13)
    ax.axis("off")

    ax.text(
        7,
        12.62,
        tr(
            lang,
            "Sparse FlashMLA Head128 Regular: Data Residency",
            "Sparse FlashMLA Head128 Regular：数据驻留",
        ),
        ha="center",
        va="center",
        fontsize=19,
        weight="bold",
    )
    ax.text(
        7,
        12.25,
        tr(
            lang,
            "one CTA's logical view; cta_group::2 combines the pair's operand halves",
            "单个 CTA 的 logical view；cta_group::2 组合 CTA pair 的 operand halves",
        ),
        ha="center",
        va="center",
        fontsize=11.0,
        color=COLORS["muted"],
    )

    panel(
        ax,
        0.25,
        8.20,
        13.50,
        3.55,
        tr(lang, "1 · QK and online softmax", "1 · QK 与 online softmax"),
        tr(lang, "BF16 operands → FP32 L → BF16 W", "BF16 operands → FP32 L → BF16 W"),
    )

    box(ax, 0.48, 9.20, 0.78, 0.92, "Q\nGMEM", COLORS["gmem"], fontsize=12.0)
    box(
        ax,
        1.52,
        9.08,
        1.55,
        1.16,
        tr(
            lang,
            "q_full\nSMEM · TMA\n64×d_qk / CTA",
            "q_full\nSMEM · TMA\n64×d_qk / CTA",
        ),
        COLORS["smem"],
        fontsize=10.5,
    )
    box(
        ax,
        3.35,
        9.86,
        1.90,
        0.80,
        "Q prefix\nSMEM\nd_sq columns",
        COLORS["smem"],
        fontsize=10.2,
    )
    box(
        ax,
        3.35,
        8.66,
        1.90,
        0.80,
        tr(lang, "Q suffix · TMEM\n384 columns", "Q suffix · TMEM\n384 列"),
        COLORS["tmem"],
        fontsize=10.2,
    )
    box(
        ax,
        5.57,
        9.01,
        1.42,
        1.30,
        "QK MMA\nSS prefix\n+ TS suffix",
        COLORS["mma"],
        fontsize=10.5,
    )
    box(
        ax,
        7.37,
        9.01,
        1.42,
        1.30,
        "L (tmem_p)\nFP32 · TMEM\n64×128",
        COLORS["tmem"],
        fontsize=9.5,
    )
    box(
        ax,
        9.17,
        8.93,
        1.90,
        1.46,
        tr(
            lang,
            "WG0 softmax\nmask · max\nexp · row sum",
            "WG0 softmax\nmask · max\nexp · row sum",
        ),
        COLORS["softmax"],
        fontsize=10.5,
    )
    box(
        ax,
        11.42,
        9.83,
        2.05,
        0.78,
        tr(
            lang,
            "max_logits / lse\nGMEM\nsink excluded",
            "max_logits / lse\nGMEM\n不含 sink",
        ),
        COLORS["gmem"],
        fontsize=9.5,
    )
    box(
        ax,
        11.42,
        8.69,
        2.05,
        0.78,
        "mi / real_mi / li\nWG0 registers",
        COLORS["neutral"],
        fontsize=10.0,
    )

    edge(ax, (1.26, 9.66), (1.52, 9.66))
    edge(ax, (3.07, 9.82), (3.35, 10.26))
    edge(ax, (3.07, 9.48), (3.35, 9.06))
    edge(ax, (5.25, 10.26), (5.57, 9.98))
    edge(ax, (5.25, 9.06), (5.57, 9.34))
    edge(ax, (6.99, 9.66), (7.37, 9.66))
    edge(ax, (8.79, 9.66), (9.17, 9.66))
    edge(ax, (11.07, 9.86), (11.42, 10.22))
    edge(ax, (11.07, 9.40), (11.42, 9.08))

    panel(
        ax,
        0.25,
        3.20,
        13.50,
        4.70,
        tr(
            lang,
            "2 · Sparse gather, PV, and epilogue",
            "2 · Sparse gather、PV 与 epilogue",
        ),
        tr(
            lang,
            "indices + KV feed both gather4 paths; O~ / (li + sink)",
            "indices + KV 同时送入两条 gather4；O~ / (li + sink)",
        ),
    )

    box(ax, 0.52, 6.00, 1.05, 0.88, "indices\nGMEM", COLORS["gmem"], fontsize=11.5)
    box(
        ax,
        0.52,
        4.00,
        1.15,
        1.05,
        "KV cache\nGMEM\n[s_kv,1,\nd_qk]",
        COLORS["gmem"],
        fontsize=8.8,
    )
    box(
        ax,
        2.15,
        5.90,
        2.05,
        1.08,
        tr(
            lang,
            "WG1 K · SMEM\ngather4\n64×d_qk / CTA",
            "WG1 K · SMEM\ngather4\n64×d_qk / CTA",
        ),
        COLORS["smem"],
        fontsize=10.2,
    )
    box(
        ax,
        2.15,
        4.00,
        2.05,
        1.05,
        tr(
            lang,
            "WG2 V · SMEM\ngather4\n128×256 / CTA",
            "WG2 V · SMEM\ngather4\n128×256 / CTA",
        ),
        COLORS["smem"],
        fontsize=10.2,
    )
    box(
        ax,
        4.55,
        5.90,
        1.85,
        1.08,
        tr(
            lang,
            "validity · SMEM\nbounds + length\n2 × 16 bytes",
            "validity · SMEM\nbounds + length\n2 × 16 bytes",
        ),
        COLORS["barrier"],
        fontsize=10.0,
    )
    box(
        ax,
        6.55,
        5.90,
        1.95,
        1.08,
        "W · BF16\ns_smem_gemm\n64 × 128",
        COLORS["smem"],
        fontsize=9.4,
    )
    box(ax, 6.82, 4.00, 1.40, 1.05, "PV MMA\nW × V", COLORS["mma"], fontsize=11.4)
    box(
        ax,
        8.62,
        4.00,
        1.42,
        1.05,
        "O~ · FP32\nTMEM\n64×512",
        COLORS["tmem"],
        fontsize=9.8,
    )
    box(
        ax,
        10.30,
        3.90,
        1.75,
        1.25,
        tr(
            lang,
            "WG0 epilogue\nO~/(li + sink)\n→ BF16",
            "WG0 epilogue\nO~/(li + sink)\n→ BF16",
        ),
        COLORS["softmax"],
        fontsize=9.5,
    )
    box(
        ax,
        8.85,
        5.95,
        1.55,
        0.95,
        tr(
            lang,
            "attn_sink\nGMEM",
            "attn_sink\nGMEM",
        ),
        COLORS["gmem"],
        fontsize=9.5,
    )
    box(
        ax,
        12.34,
        3.96,
        1.20,
        1.13,
        tr(
            lang,
            "o_smem\nBF16\n64×512\nTMA → out",
            "o_smem\nBF16\n64×512\nTMA → out",
        ),
        COLORS["smem"],
        fontsize=9.2,
    )

    # Both gather producers consume both the sparse row coordinates and KV source.
    # The junction makes that all-to-all relationship explicit without four labels.
    shared = (1.82, 5.48)
    edge(ax, (1.57, 6.25), shared)
    edge(ax, (1.67, 4.78), shared)
    edge(ax, shared, (2.15, 6.25))
    edge(ax, shared, (2.15, 4.78))
    ax.plot(*shared, marker="o", markersize=4.0, color=COLORS["line"], zorder=4)
    edge(ax, (4.20, 6.44), (4.55, 6.44))

    # K enters the QK panel; validity and W cross the layer boundary separately.
    ax.plot([4.20, 4.38, 6.28], [6.56, 8.02, 8.02], color=COLORS["line"], lw=1.25)
    edge(ax, (6.28, 8.02), (6.28, 9.01))
    ax.plot([6.40, 6.40, 10.10], [6.62, 7.98, 7.98], color=COLORS["line"], lw=1.25)
    edge(ax, (10.10, 7.98), (10.10, 8.93))
    ax.plot([9.60, 9.60, 7.52], [8.93, 8.08, 8.08], color=COLORS["line"], lw=1.25)
    edge(ax, (7.52, 8.08), (7.52, 6.98))
    edge(ax, (4.20, 4.52), (6.82, 4.52))
    edge(ax, (7.52, 5.90), (7.52, 5.05))
    edge(ax, (8.22, 4.52), (8.62, 4.52))
    edge(ax, (10.04, 4.52), (10.30, 4.52))
    edge(ax, (10.40, 6.28), (10.88, 5.15))
    edge(ax, (12.05, 4.52), (12.34, 4.52))

    panel(
        ax,
        0.25,
        0.35,
        13.50,
        2.45,
        tr(
            lang,
            "3 · Per-CTA SMEM lifetime aliasing",
            "3 · 每个 CTA 的 SMEM 生命周期复用",
        ),
        tr(
            lang,
            "one in-place K/V region; o_smem later reuses its base",
            "单份原位 K/V 区域；随后 o_smem 复用其基址",
        ),
    )
    box(
        ax,
        0.55,
        0.77,
        2.10,
        0.92,
        "q_full · SMEM\nprefix | suffix384",
        COLORS["smem"],
        fontsize=10.2,
    )
    edge(ax, (2.65, 1.23), (3.08, 1.23))

    plain_rect(ax, 3.08, 0.77, 1.45, 0.92, COLORS["smem"], edgecolor=COLORS["ink"])
    plain_rect(ax, 4.53, 0.77, 1.75, 0.92, "#ddd6fe", edgecolor=COLORS["ink"])
    plain_rect(ax, 6.28, 0.77, 1.75, 0.92, "#c4b5fd", edgecolor=COLORS["ink"])
    ax.text(
        3.80,
        1.23,
        tr(lang, "Q prefix\nlive", "Q prefix\n存活"),
        ha="center",
        va="center",
        fontsize=11.0,
        weight="bold",
    )
    ax.text(
        5.40,
        1.23,
        tr(lang, "one V\nworkspace", "单份 V\nworkspace"),
        ha="center",
        va="center",
        fontsize=11.0,
        weight="bold",
    )
    ax.text(
        7.15,
        1.23,
        tr(lang, "one K\nworkspace", "单份 K\nworkspace"),
        ha="center",
        va="center",
        fontsize=11.0,
        weight="bold",
    )

    edge(ax, (8.03, 1.23), (8.46, 1.23))
    box(
        ax,
        8.46,
        0.77,
        1.95,
        0.92,
        "o_smem · BF16\n64×512",
        COLORS["smem"],
        fontsize=10.2,
    )
    ax.text(
        10.82,
        1.23,
        tr(
            lang,
            "Adjacent K/V workspaces\nbegin at Q's released suffix.",
            "相邻的 K/V workspace\n从已释放的 Q suffix 开始。",
        ),
        ha="left",
        va="center",
        fontsize=11.0,
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
