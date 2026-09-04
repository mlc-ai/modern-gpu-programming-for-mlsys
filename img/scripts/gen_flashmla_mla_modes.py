#!/usr/bin/env python3
"""Generate the two equivalent MLA attention-core views used by the tutorial."""

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


def _panel(
    ax, x: float, y: float, w: float, h: float, title: str, subtitle: str, color: str
) -> None:
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
        x + 0.28,
        y + h - 0.34,
        title,
        ha="left",
        va="center",
        fontsize=12,
        weight="bold",
        color=color,
    )
    ax.text(
        x + 0.28,
        y + h - 0.72,
        subtitle,
        ha="left",
        va="center",
        fontsize=7.8,
        color=COLORS["muted"],
    )


def draw(lang: str, output: str, font_path: str | None = None) -> None:
    configure_style(lang, font_path)
    fig, ax = plt.subplots(figsize=(18.0, 9.4))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    ax.text(
        9,
        9.08,
        tr(
            lang,
            "MLA: Two Equivalent Attention-Core Views of One Cache",
            "MLA：同一份 Cache 的两种等价 Attention Core 视图",
        ),
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
    )
    ax.text(
        9,
        8.70,
        tr(
            lang,
            "move the up-projections across the attention core without changing the cached latent state",
            "在 attention core 两侧移动 up-projection，而不改变缓存的 latent state",
        ),
        ha="center",
        va="center",
        fontsize=9.3,
        color=COLORS["muted"],
    )

    # One common compressed cache is the source for both views.
    ax.text(
        9,
        8.26,
        tr(lang, "one MLA KV cache entry", "同一条 MLA KV cache entry"),
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    plain_rect(ax, 6.15, 7.35, 4.25, 0.72, COLORS["gmem"], edgecolor=COLORS["ink"])
    plain_rect(ax, 10.40, 7.35, 1.45, 0.72, "#bfdbfe", edgecolor=COLORS["ink"])
    ax.text(
        8.27,
        7.71,
        "c_KV  [latent 512]",
        ha="center",
        va="center",
        fontsize=9.3,
        weight="bold",
        zorder=4,
    )
    ax.text(
        11.12,
        7.71,
        "k_R  [RoPE 64]",
        ha="center",
        va="center",
        fontsize=8.8,
        weight="bold",
        zorder=4,
    )
    ax.text(
        9,
        7.15,
        tr(
            lang,
            "stored once; the diagrams below differ only in where W_UK and W_UV run",
            "只缓存一份；下方两种视图只改变 W_UK 与 W_UV 的执行位置",
        ),
        ha="center",
        va="center",
        fontsize=7.8,
        color=COLORS["muted"],
    )

    _panel(
        ax,
        0.35,
        1.42,
        8.32,
        5.30,
        tr(lang, "MHA mode", "MHA mode"),
        tr(
            lang,
            "expand per-head K and V before the core",
            "在 core 之前展开每个 head 的 K 与 V",
        ),
        COLORS["cta0_dark"],
    )
    _panel(
        ax,
        9.33,
        1.42,
        8.32,
        5.30,
        tr(lang, "Absorbed MQA mode", "Absorbed MQA mode"),
        tr(
            lang,
            "absorb W_UK into q_C; q_R bypasses; move W_UV after the core",
            "将 W_UK 吸收到 q_C；q_R 绕过；把 W_UV 移到 core 之后",
        ),
        COLORS["cta1_dark"],
    )
    arrow(ax, (7.52, 7.35), (4.58, 6.43), color=COLORS["cta0_dark"], rad=0.08)
    arrow(ax, (10.47, 7.35), (13.55, 6.43), color=COLORS["cta1_dark"], rad=-0.08)

    # Left: explicit per-head K/V expansion before a conventional MHA-shaped core.
    rounded_box(
        ax,
        0.82,
        4.82,
        2.05,
        0.82,
        "per-head Q\n[q_C 128 ; q_R 64]",
        COLORS["neutral"],
        fontsize=8.3,
    )
    rounded_box(
        ax, 0.82, 3.78, 1.72, 0.75, "c_KV\nlatent 512", COLORS["gmem"], fontsize=8.4
    )
    rounded_box(
        ax, 0.82, 2.76, 1.72, 0.70, "k_R\nRoPE 64", COLORS["gmem"], fontsize=8.3
    )
    rounded_box(ax, 3.05, 4.02, 1.28, 0.70, "W_UK", COLORS["projection"], fontsize=9)
    rounded_box(ax, 3.05, 3.00, 1.28, 0.70, "W_UV", COLORS["projection"], fontsize=9)
    rounded_box(
        ax,
        4.86,
        3.97,
        2.20,
        0.82,
        "per-head K\n[k_C 128 ; k_R 64]",
        COLORS["smem"],
        fontsize=8.2,
    )
    rounded_box(
        ax, 4.86, 2.92, 2.20, 0.76, "per-head V\nv_C 128", COLORS["smem"], fontsize=8.4
    )
    rounded_box(
        ax,
        3.08,
        1.72,
        2.70,
        0.84,
        tr(
            lang,
            "MHA attention core\nper-head K / V",
            "MHA attention core\nper-head K / V",
        ),
        COLORS["mma"],
        fontsize=8.7,
    )
    rounded_box(
        ax, 6.34, 1.76, 1.75, 0.76, "per-head out\n128", COLORS["neutral"], fontsize=8.5
    )
    arrow(ax, (2.54, 4.15), (3.05, 4.37))
    arrow(ax, (2.54, 4.00), (3.05, 3.35))
    arrow(ax, (4.33, 4.37), (4.86, 4.37))
    # k_R bypasses W_UK and joins K below that projection box.
    ax.plot(
        [2.54, 2.76, 4.55], [3.11, 3.88, 3.88], color=COLORS["line"], lw=1.25, zorder=2
    )
    arrow(
        ax,
        (4.55, 3.88),
        (5.15, 3.97),
        label=tr(lang, "concat", "拼接"),
        color=COLORS["line"],
        rad=-0.03,
    )
    arrow(ax, (4.33, 3.35), (4.86, 3.30))
    # Route Q around the left edge so it does not cut through c_KV/W_UV.
    ax.plot(
        [1.84, 0.62, 0.62, 2.72],
        [4.82, 4.82, 2.14, 2.14],
        color=COLORS["line"],
        lw=1.25,
        zorder=2,
    )
    arrow(ax, (2.72, 2.14), (3.08, 2.14), color=COLORS["line"])
    arrow(ax, (5.70, 3.97), (4.82, 2.56), color=COLORS["line"], rad=0.04)
    arrow(ax, (5.70, 2.92), (5.18, 2.56), color=COLORS["line"], rad=-0.02)
    arrow(ax, (5.78, 2.14), (6.34, 2.14))

    # Right: only the content query absorbs W_UK.  The per-head RoPE query
    # bypasses that projection and is concatenated immediately before QK.
    rounded_box(
        ax,
        9.78,
        5.18,
        1.48,
        0.66,
        "q_C · per head\ncontent 128",
        COLORS["neutral"],
        fontsize=8.0,
    )
    rounded_box(
        ax,
        9.78,
        4.37,
        1.48,
        0.62,
        "q_R · per head\nRoPE 64",
        "#bfdbfe",
        fontsize=8.0,
    )
    rounded_box(
        ax,
        11.72,
        5.14,
        1.45,
        0.74,
        tr(lang, "(W_UK)^T q_C\ncontent only", "(W_UK)^T q_C\n仅 content"),
        COLORS["projection"],
        fontsize=7.9,
    )
    rounded_box(
        ax,
        13.62,
        4.64,
        2.78,
        1.02,
        "absorbed Q\n[q_abs 512 ; q_R 64] = 576",
        COLORS["tmem"],
        fontsize=8.2,
    )
    rounded_box(
        ax,
        9.83,
        3.30,
        2.30,
        0.94,
        "shared KV · h_kv=1\n[c_KV 512 ; k_R 64]\nV uses c_KV 512",
        COLORS["gmem"],
        fontsize=8.0,
    )
    rounded_box(
        ax,
        12.35,
        2.44,
        2.42,
        0.94,
        tr(lang, "absorbed MQA\nattention core", "absorbed MQA\nattention core"),
        COLORS["mma"],
        fontsize=8.8,
    )
    rounded_box(
        ax, 15.18, 2.51, 1.55, 0.80, "latent out\n512", COLORS["tmem"], fontsize=8.5
    )
    rounded_box(
        ax,
        14.30,
        1.58,
        1.45,
        0.70,
        tr(lang, "W_UV\nafter core", "W_UV\n移到 core 后"),
        COLORS["projection"],
        fontsize=8.0,
    )
    rounded_box(
        ax, 16.18, 1.58, 1.05, 0.70, "out\n128", COLORS["neutral"], fontsize=8.3
    )
    arrow(ax, (11.26, 5.51), (11.72, 5.51))
    arrow(ax, (13.17, 5.51), (13.62, 5.30))
    # The explicit positional channel never passes through W_UK.
    arrow(
        ax,
        (11.26, 4.68),
        (13.62, 4.91),
        label=tr(lang, "bypass W_UK", "绕过 W_UK"),
        color=COLORS["cta0_dark"],
        rad=-0.06,
        label_offset=(0.0, -0.15),
    )
    arrow(ax, (15.05, 4.64), (13.83, 3.38), color=COLORS["line"], rad=-0.05)
    arrow(ax, (10.98, 3.30), (12.64, 3.08), color=COLORS["line"], rad=0.03)
    arrow(ax, (14.77, 2.91), (15.18, 2.91))
    arrow(ax, (15.95, 2.51), (15.02, 2.28), color=COLORS["line"], rad=0.06)
    arrow(ax, (15.75, 1.93), (16.18, 1.93))

    # Explicit equivalence and chapter scope.
    ax.text(
        9,
        1.23,
        tr(
            lang,
            "Equivalent after projection reassociation: both produce the same per-head output 128",
            "重新结合 projection 后等价：两种视图都产生相同的 per-head output 128",
        ),
        ha="center",
        va="center",
        fontsize=8.0,
        color="#7c3aed",
        weight="bold",
    )
    rounded_box(
        ax,
        2.1,
        0.24,
        13.8,
        0.76,
        tr(
            lang,
            "The d_qk=576 sparse-prefill case studied here uses the absorbed MQA view:\nshared KV [512+64], absorbed Q 576, latent output 512. These dimensions do not describe every FlashMLA kernel.",
            "本章重点分析的 d_qk=576 sparse-prefill case 采用 absorbed MQA 视图：\nshared KV [512+64]、absorbed Q 576、latent output 512。这些维度并不代表所有 FlashMLA kernel。",
        ),
        COLORS["note"],
        fontsize=8.2,
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
