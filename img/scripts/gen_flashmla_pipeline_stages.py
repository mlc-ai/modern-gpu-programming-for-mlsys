#!/usr/bin/env python3
"""Generate the fill, steady-state, and drain story for sparse FlashMLA."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from flashmla_diagram_common import (
    COLORS,
    arrow,
    configure_style,
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
        x + 0.24,
        y + h - 0.32,
        title,
        ha="left",
        va="center",
        fontsize=11.2,
        weight="bold",
        color=color,
    )
    ax.text(
        x + 0.24,
        y + h - 0.70,
        subtitle,
        ha="left",
        va="center",
        fontsize=7.6,
        color=COLORS["muted"],
    )


def _lane(ax, x0, x1, y, label, lang) -> None:
    ax.text(
        x0,
        y,
        label,
        ha="left",
        va="center",
        fontsize=7.1,
        color=COLORS["muted"],
        weight="bold",
    )
    ax.plot([x0 + 0.76, x1], [y, y], color="#e5e7eb", lw=1.0, zorder=0)


def draw(lang: str, output: str, font_path: str | None = None) -> None:
    configure_style(lang, font_path)
    fig, ax = plt.subplots(figsize=(18.0, 9.6))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    ax.text(
        9,
        9.27,
        tr(
            lang,
            "How Does the One-Step-Lag Pipeline Fill and Drain?",
            "相差一个 Tile 的 Pipeline 如何填充与排空？",
        ),
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
    )
    ax.text(
        9,
        8.88,
        tr(
            lang,
            "N = num_k_blocks; one WG3 issuer loop executes k=0…N and serially issues QK before PV",
            "N = num_k_blocks；一个 WG3 issuer loop 执行 k=0…N，并按顺序先发 QK、再发 PV",
        ),
        ha="center",
        va="center",
        fontsize=9.2,
        color=COLORS["muted"],
    )

    # The mathematical dependency is simple; scheduling staggers adjacent tiles.
    rounded_box(
        ax,
        2.02,
        7.83,
        2.02,
        0.62,
        "QK(t) → L(t)",
        COLORS["mma"],
        fontsize=8.6,
    )
    rounded_box(
        ax,
        5.35,
        7.83,
        2.60,
        0.62,
        "softmax(t) → W(t)",
        COLORS["softmax"],
        fontsize=8.6,
    )
    rounded_box(
        ax,
        9.24,
        7.83,
        2.30,
        0.62,
        "PV(t) → O~(t)",
        COLORS["mma"],
        fontsize=8.6,
    )
    arrow(ax, (4.04, 8.14), (5.35, 8.14), label="L(t)")
    arrow(ax, (7.95, 8.14), (9.24, 8.14), label="W(t)")
    ax.text(
        13.95,
        8.14,
        tr(
            lang,
            "logical dependency for one tile\nexecution overlaps neighboring tiles",
            "单个 tile 的逻辑依赖\n执行时与相邻 tiles 重叠",
        ),
        ha="center",
        va="center",
        fontsize=8.1,
        color=COLORS["muted"],
        style="italic",
    )

    y0 = 1.42
    h = 5.94
    _panel(
        ax,
        0.35,
        y0,
        4.40,
        h,
        tr(lang, "Fill · k=0", "填充 · k=0"),
        tr(
            lang,
            "the issuer has no previous tile to multiply",
            "issuer 尚无上一 tile 可执行 PV",
        ),
        COLORS["cta0_dark"],
    )
    _panel(
        ax,
        5.02,
        y0,
        8.33,
        h,
        tr(lang, "Steady state · 1 ≤ k < N", "稳态 · 1 ≤ k < N"),
        tr(
            lang,
            "adjacent softmax work overlaps asynchronous MMAs",
            "相邻 tile 的 softmax 与异步 MMA 重叠",
        ),
        "#7c3aed",
    )
    _panel(
        ax,
        13.62,
        y0,
        4.03,
        h,
        tr(lang, "Drain · k=N", "排空 · k=N"),
        tr(lang, "the issuer has no new QK tile", "issuer 不再有新的 QK tile"),
        COLORS["cta1_dark"],
    )

    # Fill: Q prologue, QK(0), and the first softmax; no PV is issued at k=0.
    rounded_box(
        ax,
        0.72,
        5.82,
        3.66,
        0.58,
        tr(
            lang,
            "prologue: Q TMA · TMEM alloc · initial K/V gather",
            "prologue：Q TMA · TMEM alloc · 首轮 K/V gather",
        ),
        COLORS["gmem"],
        fontsize=7.6,
        weight="normal",
    )
    _lane(ax, 0.68, 4.38, 5.25, "WG3", lang)
    _lane(ax, 0.68, 4.38, 4.08, "WG0", lang)
    _lane(ax, 0.68, 4.38, 2.87, "WG1/2", lang)
    rounded_box(ax, 1.65, 4.88, 1.56, 0.72, "QK(0)", COLORS["mma"], fontsize=8.8)
    rounded_box(
        ax,
        3.47,
        4.91,
        0.66,
        0.66,
        "PV\n—",
        COLORS["neutral"],
        fontsize=7.7,
        weight="normal",
        edgecolor="#9ca3af",
        linestyle="--",
    )
    rounded_box(
        ax,
        2.10,
        3.71,
        1.82,
        0.74,
        "softmax(0)",
        COLORS["softmax"],
        fontsize=8.4,
    )
    arrow(ax, (2.84, 4.88), (2.72, 4.45), label="L(0)", rad=0.03)
    rounded_box(
        ax,
        1.37,
        2.50,
        2.65,
        0.72,
        tr(lang, "prepare safe next segments", "准备下一批可安全复用的 segments"),
        COLORS["tma"],
        fontsize=7.8,
        weight="normal",
    )
    ax.text(
        2.55,
        1.91,
        tr(lang, "issuer branch: QK(0) only", "issuer 分支：仅 QK(0)"),
        ha="center",
        va="center",
        fontsize=8.1,
        color=COLORS["muted"],
        weight="bold",
    )

    # Steady state: make the two legal overlaps explicit without depicting two issuers.
    _lane(ax, 5.34, 13.00, 5.65, "WG3", lang)
    _lane(ax, 5.34, 13.00, 4.20, "WG0", lang)
    _lane(ax, 5.34, 13.00, 2.72, "WG1/2", lang)
    arrow(ax, (6.15, 6.37), (12.78, 6.37), color="#9ca3af", linewidth=1.0)
    ax.text(
        9.46,
        6.44,
        tr(lang, "issuer program order / time", "issuer 程序顺序 / 时间"),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=COLORS["muted"],
        style="italic",
    )

    # Light overlap bands sit behind the actual operation boxes.
    ax.axvspan(6.14, 8.58, ymin=0.41, ymax=0.66, color="#eff6ff", alpha=0.75, zorder=-1)
    ax.axvspan(
        9.62, 12.47, ymin=0.41, ymax=0.66, color="#fdf2f8", alpha=0.75, zorder=-1
    )
    rounded_box(ax, 6.38, 5.27, 1.86, 0.76, "QK(k)", COLORS["mma"], fontsize=9.0)
    rounded_box(
        ax,
        9.72,
        5.27,
        2.16,
        0.76,
        "PV(k−1)",
        COLORS["mma"],
        fontsize=9.0,
    )
    arrow(
        ax,
        (8.24, 5.65),
        (9.72, 5.65),
        label=tr(lang, "same issuer · serial issue", "同一 issuer · 串行发出"),
        color=COLORS["ink"],
        linewidth=1.5,
        label_offset=(0.0, 0.18),
    )
    rounded_box(
        ax,
        6.05,
        3.82,
        2.58,
        0.76,
        "softmax(k−1) → W(k−1)",
        COLORS["softmax"],
        fontsize=8.0,
    )
    rounded_box(
        ax,
        9.88,
        3.82,
        2.52,
        0.76,
        "softmax(k) → W(k)",
        COLORS["softmax"],
        fontsize=8.0,
    )
    arrow(
        ax,
        (8.63, 4.20),
        (10.03, 5.27),
        label="W(k−1) ready",
        color="#7c3aed",
        rad=-0.06,
        label_offset=(0.35, 0.25),
    )
    arrow(
        ax,
        (7.93, 5.27),
        (10.12, 4.58),
        label="L(k) ready",
        color="#7c3aed",
        rad=0.05,
        label_offset=(0.40, -0.34),
    )
    ax.text(
        7.34,
        3.43,
        tr(lang, "legal overlap A", "合法重叠 A"),
        ha="center",
        va="center",
        fontsize=7.4,
        color=COLORS["cta0_dark"],
        weight="bold",
    )
    ax.text(
        11.11,
        3.43,
        tr(lang, "legal overlap B", "合法重叠 B"),
        ha="center",
        va="center",
        fontsize=7.4,
        color=COLORS["cta1_dark"],
        weight="bold",
    )
    rounded_box(
        ax,
        6.02,
        2.35,
        2.82,
        0.72,
        tr(lang, "gather safe K(k+1) parts", "gather 可复用的 K(k+1) parts"),
        COLORS["tma"],
        fontsize=7.7,
        weight="normal",
    )
    rounded_box(
        ax,
        9.55,
        2.35,
        2.82,
        0.72,
        tr(lang, "gather safe V(k) parts", "gather 可复用的 V(k) parts"),
        COLORS["tma"],
        fontsize=7.7,
        weight="normal",
    )
    ax.text(
        9.18,
        1.87,
        tr(
            lang,
            "QK(k) and PV(k−1) are not concurrently issued",
            "QK(k) 与 PV(k−1) 并非同时发出",
        ),
        ha="center",
        va="center",
        fontsize=8.2,
        color="#6d28d9",
        weight="bold",
    )

    # Drain: the extra loop iteration performs only the final PV, then WG0 normalizes.
    _lane(ax, 13.92, 17.30, 5.65, "WG3", lang)
    _lane(ax, 13.92, 17.30, 4.20, "WG0", lang)
    _lane(ax, 13.92, 17.30, 2.72, "WG1/2", lang)
    rounded_box(
        ax,
        14.64,
        5.30,
        0.86,
        0.70,
        "QK\n—",
        COLORS["neutral"],
        fontsize=7.6,
        weight="normal",
        edgecolor="#9ca3af",
        linestyle="--",
    )
    rounded_box(ax, 15.78, 5.27, 1.34, 0.76, "PV(N−1)", COLORS["mma"], fontsize=8.1)
    arrow(ax, (15.50, 5.65), (15.78, 5.65), color=COLORS["ink"])
    rounded_box(
        ax,
        14.53,
        3.77,
        2.58,
        0.86,
        tr(
            lang,
            "wait sv_done(N−1)\nnormalize + epilogue",
            "等待 sv_done(N−1)\n归一化 + epilogue",
        ),
        COLORS["softmax"],
        fontsize=7.8,
    )
    arrow(ax, (16.48, 5.27), (16.20, 4.63), color="#7c3aed", rad=0.03)
    rounded_box(
        ax,
        14.62,
        2.37,
        2.42,
        0.70,
        tr(lang, "no more gathers", "不再发起 gather"),
        COLORS["neutral"],
        fontsize=8.0,
        weight="normal",
        edgecolor="#9ca3af",
        linestyle="--",
    )
    ax.text(
        15.64,
        1.87,
        tr(lang, "issuer branch: PV(N−1) only", "issuer 分支：仅 PV(N−1)"),
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
        weight="bold",
    )

    rounded_box(
        ax,
        1.25,
        0.30,
        15.50,
        0.72,
        tr(
            lang,
            "Source-order loop: for k in [0, N]:  if k < N issue QK(k);  if k > 0 issue PV(k−1).  MMAs execute asynchronously, but one warp issues them in that order.",
            "源码顺序：for k in [0, N]：若 k < N 则发 QK(k)；若 k > 0 则发 PV(k−1)。MMA 异步执行，但由同一个 warp 按此顺序发出。",
        ),
        COLORS["note"],
        fontsize=8.15,
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
