#!/usr/bin/env python3
"""Generate the introductory MHA-to-MLA cache story for the FlashMLA tutorial."""

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
        fontsize=11.5,
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
    fig, ax = plt.subplots(figsize=(18.0, 9.0))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        9,
        8.68,
        tr(
            lang,
            "How Can One MLA Cache Entry Serve 128 Query Heads?",
            "一条 MLA Cache Entry 如何服务 128 个 Query Heads？",
        ),
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
    )
    ax.text(
        9,
        8.28,
        tr(
            lang,
            "cache the shared source once; preserve per-head behavior on the query and output paths",
            "共享 source 只缓存一次；各 head 的差异保留在 query 与 output 路径",
        ),
        ha="center",
        va="center",
        fontsize=9.3,
        color=COLORS["muted"],
    )

    _panel(
        ax,
        0.35,
        1.30,
        4.45,
        6.45,
        tr(lang, "1 · MHA cache", "1 · MHA cache"),
        tr(lang, "materialize K and V for every head", "为每个 head 物化 K 与 V"),
        COLORS["cta0_dark"],
    )
    _panel(
        ax,
        5.15,
        1.30,
        5.05,
        6.45,
        tr(lang, "2 · MLA cache", "2 · MLA cache"),
        tr(
            lang,
            "store one latent source per token",
            "每个 token 只保存一份 latent source",
        ),
        "#7c3aed",
    )
    _panel(
        ax,
        10.55,
        1.30,
        7.10,
        6.45,
        tr(lang, "3 · Absorbed execution", "3 · Absorbed execution"),
        tr(
            lang,
            "head-specific work moves around the core",
            "各 head 的特定计算移到 core 两侧",
        ),
        COLORS["cta1_dark"],
    )

    # MHA: a visual stack of materialized per-head cache entries.
    ax.text(
        2.58,
        6.53,
        tr(lang, "one cached token", "一个 cached token"),
        ha="center",
        fontsize=9.5,
        weight="bold",
    )
    row_entries = (
        ("head 0", "K_0", "V_0"),
        ("head 1", "K_1", "V_1"),
        ("...", "...", "..."),
        ("head 127", "K_127", "V_127"),
    )
    for idx, (label, k_label, v_label) in enumerate(row_entries):
        y = 5.70 - idx * 0.83
        ax.text(0.72, y + 0.29, label, ha="left", va="center", fontsize=7.7)
        plain_rect(ax, 1.55, y, 1.18, 0.58, COLORS["gmem"])
        plain_rect(ax, 2.73, y, 1.18, 0.58, "#bfdbfe")
        ax.text(
            2.14,
            y + 0.29,
            k_label,
            ha="center",
            va="center",
            fontsize=8.0,
            weight="bold",
        )
        ax.text(
            3.32,
            y + 0.29,
            v_label,
            ha="center",
            va="center",
            fontsize=8.0,
            weight="bold",
        )
    rounded_box(
        ax,
        0.82,
        1.72,
        3.50,
        0.88,
        tr(
            lang,
            "cache stores a separate K/V slice\nfor every head",
            "cache 为每个 head 保存\n独立的 K/V slice",
        ),
        COLORS["note"],
        fontsize=8.5,
        weight="normal",
        edgecolor="#ca8a04",
    )

    # MLA: down-project content once and store it with one shared RoPE key.
    rounded_box(ax, 5.55, 5.83, 1.20, 0.72, "h_s", COLORS["neutral"], fontsize=9.2)
    rounded_box(ax, 7.12, 5.83, 1.35, 0.72, "W_DKV", COLORS["projection"], fontsize=8.8)
    rounded_box(
        ax,
        8.78,
        5.69,
        1.05,
        1.00,
        "c_KV\nlatent\n512",
        COLORS["smem"],
        fontsize=8.0,
    )
    arrow(ax, (6.75, 6.19), (7.12, 6.19))
    arrow(ax, (8.47, 6.19), (8.78, 6.19))
    rounded_box(
        ax,
        5.55,
        4.47,
        2.05,
        0.78,
        tr(lang, "shared RoPE key", "共享 RoPE key"),
        "#bfdbfe",
        fontsize=8.4,
    )
    rounded_box(ax, 8.02, 4.47, 1.81, 0.78, "k_R\nRoPE 64", "#bfdbfe", fontsize=8.4)
    arrow(ax, (7.60, 4.86), (8.02, 4.86))

    ax.text(
        7.68,
        3.94,
        tr(lang, "stored once per token", "每个 token 只存一次"),
        ha="center",
        va="center",
        fontsize=9.2,
        weight="bold",
        color="#6d28d9",
    )
    plain_rect(ax, 5.82, 2.90, 2.76, 0.76, COLORS["gmem"])
    plain_rect(ax, 8.58, 2.90, 1.20, 0.76, "#bfdbfe")
    ax.text(
        7.20, 3.28, "c_KV · 512", ha="center", va="center", fontsize=9.0, weight="bold"
    )
    ax.text(
        9.18, 3.28, "k_R · 64", ha="center", va="center", fontsize=8.4, weight="bold"
    )
    rounded_box(
        ax,
        5.82,
        1.72,
        3.96,
        0.72,
        tr(
            lang,
            "one shared entry · not 128 expanded pairs",
            "一份 shared entry · 不是 128 份展开 pair",
        ),
        COLORS["note"],
        fontsize=8.2,
        weight="normal",
        edgecolor="#ca8a04",
    )
    arrow(ax, (9.78, 3.28), (11.15, 4.09), color="#7c3aed", linewidth=1.6, rad=-0.05)

    # Absorbed execution: q_C alone crosses W_UK; q_R remains explicit.
    rounded_box(
        ax,
        10.92,
        5.89,
        1.46,
        0.70,
        "q_C,i\nper head",
        COLORS["neutral"],
        fontsize=8.1,
    )
    rounded_box(
        ax,
        12.78,
        5.89,
        1.66,
        0.70,
        "(W_UK,i)^T\ncontent only",
        COLORS["projection"],
        fontsize=7.8,
    )
    rounded_box(
        ax,
        10.92,
        4.93,
        1.46,
        0.70,
        "q_R,i\nRoPE 64",
        "#bfdbfe",
        fontsize=8.1,
    )
    rounded_box(
        ax,
        14.88,
        5.34,
        2.28,
        1.02,
        "Q_i = [q_abs,i ; q_R,i]\n512 + 64",
        COLORS["tmem"],
        fontsize=8.1,
    )
    arrow(ax, (12.38, 6.24), (12.78, 6.24))
    arrow(ax, (14.44, 6.24), (14.88, 6.03))
    arrow(
        ax,
        (12.38, 5.28),
        (14.88, 5.62),
        label=tr(lang, "bypass W_UK", "绕过 W_UK"),
        color=COLORS["cta0_dark"],
        rad=-0.05,
        label_offset=(0.0, -0.14),
    )

    rounded_box(
        ax,
        11.15,
        3.63,
        2.56,
        0.92,
        "shared K = [c_KV ; k_R]\nshared V = c_KV",
        COLORS["gmem"],
        fontsize=8.0,
    )
    rounded_box(
        ax,
        14.15,
        3.48,
        2.12,
        1.10,
        tr(lang, "shared-cache\nattention core", "shared-cache\nattention core"),
        COLORS["mma"],
        fontsize=8.7,
    )
    arrow(ax, (16.02, 5.34), (15.56, 4.58), color=COLORS["line"], rad=0.04)
    arrow(ax, (13.71, 4.09), (14.15, 4.03))
    rounded_box(
        ax,
        13.22,
        2.22,
        1.60,
        0.72,
        "latent out_i\n512",
        COLORS["tmem"],
        fontsize=8.0,
    )
    rounded_box(
        ax,
        15.22,
        2.22,
        1.90,
        0.72,
        tr(
            lang,
            "per-head output path\nW_UV,i / W_O",
            "per-head output 路径\nW_UV,i / W_O",
        ),
        COLORS["projection"],
        fontsize=7.6,
    )
    arrow(ax, (15.05, 3.48), (14.10, 2.94), color=COLORS["line"], rad=0.04)
    arrow(ax, (14.82, 2.58), (15.22, 2.58))
    ax.text(
        15.18,
        1.74,
        tr(
            lang,
            "repeat the query/output paths for i=0…127\n→ 128 distinct head outputs",
            "query/output 路径对 i=0…127 各自执行\n→ 128 份不同的 head output",
        ),
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
        weight="bold",
    )

    rounded_box(
        ax,
        1.65,
        0.30,
        14.70,
        0.66,
        tr(
            lang,
            "Only [c_KV ; k_R] is cached. Per-head K/V need not be materialized; head-specific projections survive by reassociation around attention.",
            "Cache 中只有 [c_KV ; k_R]。无需物化 per-head K/V；通过在 attention 两侧重新结合运算，仍保留各 head 的特定 projection。",
        ),
        COLORS["note"],
        fontsize=8.4,
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
