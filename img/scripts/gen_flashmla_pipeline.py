#!/usr/bin/env python3
"""Generate the sparse FlashMLA head128 regular steady-state pipeline diagram."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from flashmla_diagram_common import (
    COLORS,
    arrow,
    configure_style,
    rounded_box,
    save_figure,
    tr,
)


def _row(ax, y: float, title: str, role: str) -> None:
    rounded_box(
        ax, 0.12, y, 2.05, 0.82, f"{title}\n{role}", COLORS["neutral"], fontsize=8.4
    )
    ax.plot([2.42, 18.7], [y + 0.41, y + 0.41], color="#e5e7eb", lw=1.0, zorder=0)


def _barrier_note(ax, x: float, y: float, text: str, *, fontsize: float = 7.0) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#92400e",
        bbox=dict(
            boxstyle="round,pad=0.14", facecolor=COLORS["barrier"], edgecolor="#d97706"
        ),
        zorder=7,
    )


def draw(lang: str, output: str, font_path: str | None = None) -> None:
    configure_style(lang, font_path)
    fig, ax = plt.subplots(figsize=(19.0, 10.5))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    ax.text(
        9.5,
        10.16,
        tr(
            lang,
            "Sparse FlashMLA Head128 Regular: Steady-State Pipeline",
            "Sparse FlashMLA Head128 Regular：稳态 Pipeline",
        ),
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
    )
    ax.text(
        9.5,
        9.76,
        tr(
            lang,
            "softmax(k−1) may overlap QK(k); after QK(k) is issued, softmax(k) may overlap asynchronous PV(k−1); widths are not cycle measurements",
            "softmax(k−1) 可与 QK(k) 重叠；QK(k) 发出后，softmax(k) 可与异步 PV(k−1) 重叠；方框宽度不表示 cycle 数",
        ),
        ha="center",
        va="center",
        fontsize=9.4,
        color=COLORS["muted"],
    )
    arrow(ax, (2.42, 9.24), (18.62, 9.24), color="#9ca3af", linewidth=1.0)
    ax.text(
        10.5,
        9.31,
        tr(lang, "time", "时间"),
        ha="center",
        va="bottom",
        fontsize=8,
        color=COLORS["muted"],
        style="italic",
    )

    y_k = 7.83
    y_v = 6.23
    y_mma = 4.63
    y_mask = 3.17
    y_soft = 1.57
    _row(ax, y_k, "WG1", tr(lang, "K gather producer", "K gather producer"))
    _row(ax, y_v, "WG2", tr(lang, "V gather producer", "V gather producer"))
    _row(ax, y_mma, "WG3 · CTA0 warp 12", tr(lang, "QK / PV issue", "发起 QK / PV"))
    _row(ax, y_mask, "WG3 · warp 13", tr(lang, "validity mask", "有效性 mask"))
    _row(ax, y_soft, "WG0", tr(lang, "softmax + O rescale", "softmax + O 重缩放"))

    # Current K was gathered earlier into the one in-place K workspace.
    rounded_box(
        ax,
        2.54,
        y_k + 0.06,
        1.18,
        0.70,
        "K(k) part0\nready",
        "#dbeafe",
        fontsize=7.5,
        edgecolor="#60a5fa",
    )
    rounded_box(
        ax,
        3.92,
        y_k + 0.06,
        1.18,
        0.70,
        "K(k) part1\nready",
        "#dbeafe",
        fontsize=7.5,
        edgecolor="#60a5fa",
    )

    # The leader issues two QK segments, then two PV segments for the prior tile.
    rounded_box(
        ax,
        5.40,
        y_mma + 0.03,
        1.26,
        0.76,
        "QK(k)\nSS prefix",
        COLORS["mma"],
        fontsize=8.0,
    )
    rounded_box(
        ax,
        7.35,
        y_mma + 0.03,
        1.26,
        0.76,
        "QK(k)\nTS suffix384",
        COLORS["mma"],
        fontsize=8.0,
    )
    rounded_box(
        ax,
        10.28,
        y_mma + 0.03,
        1.35,
        0.76,
        "PV(k−1)\npart0",
        COLORS["mma"],
        fontsize=8.0,
    )
    rounded_box(
        ax,
        12.36,
        y_mma + 0.03,
        1.35,
        0.76,
        "PV(k−1)\npart1",
        COLORS["mma"],
        fontsize=8.0,
    )
    arrow(ax, (6.66, y_mma + 0.41), (7.35, y_mma + 0.41), color=COLORS["line"])
    arrow(ax, (8.61, y_mma + 0.41), (10.28, y_mma + 0.41), color=COLORS["line"])
    arrow(ax, (11.63, y_mma + 0.41), (12.36, y_mma + 0.41), color=COLORS["line"])
    ax.text(
        9.42,
        y_mma + 0.18,
        tr(lang, "same issue warp · source order", "同一个 issue warp · 源码顺序"),
        ha="center",
        va="center",
        fontsize=6.9,
        color=COLORS["muted"],
        style="italic",
    )

    # Current K readiness gates each QK segment; score-TMEM reuse has its own gate.
    arrow(ax, (3.72, y_k + 0.39), (5.73, y_mma + 0.79), color="#2563eb", rad=-0.06)
    _barrier_note(ax, 4.55, 6.74, "k_part0_ready[k]")
    arrow(ax, (5.10, y_k + 0.39), (7.68, y_mma + 0.79), color="#2563eb", rad=-0.08)
    _barrier_note(ax, 6.42, 6.98, "k_part1_ready[k]")
    rounded_box(
        ax,
        4.92,
        4.06,
        2.10,
        0.43,
        tr(
            lang,
            "incoming from prior WG0(k−1) · p_free[k−1]",
            "来自前一轮 WG0(k−1) · p_free[k−1]",
        ),
        "#f3e8ff",
        fontsize=6.5,
        weight="normal",
        edgecolor="#7c3aed",
        linestyle="--",
    )
    arrow(
        ax,
        (5.97, 4.49),
        (5.97, y_mma + 0.03),
        color="#7c3aed",
        linestyle="--",
        linewidth=1.1,
    )

    # QK segment completion releases the same K workspace for k+1, part by part.
    rounded_box(
        ax,
        6.70,
        y_k + 0.06,
        1.42,
        0.70,
        "gather K(k+1)\npart0",
        COLORS["tma"],
        fontsize=7.6,
    )
    rounded_box(
        ax,
        8.68,
        y_k + 0.06,
        1.42,
        0.70,
        "gather K(k+1)\npart1",
        COLORS["tma"],
        fontsize=7.6,
    )
    arrow(ax, (6.03, y_mma + 0.79), (7.12, y_k + 0.06), color="#059669", rad=0.04)
    _barrier_note(ax, 6.72, 7.25, "qk_part_done[k]")
    arrow(ax, (7.98, y_mma + 0.79), (9.10, y_k + 0.06), color="#059669", rad=0.04)
    _barrier_note(ax, 8.64, 7.32, "qk_done[k]")

    # The prior V parts are already ready for PV(k-1).
    rounded_box(
        ax,
        8.18,
        y_v + 0.06,
        1.15,
        0.70,
        "V(k−1)\npart0 ready",
        "#dbeafe",
        fontsize=7.2,
        edgecolor="#60a5fa",
    )
    rounded_box(
        ax,
        9.48,
        y_v + 0.06,
        1.15,
        0.70,
        "V(k−1)\npart1 ready",
        "#dbeafe",
        fontsize=7.2,
        edgecolor="#60a5fa",
    )
    arrow(ax, (9.33, y_v + 0.38), (10.62, y_mma + 0.79), color="#2563eb", rad=-0.04)
    arrow(ax, (10.63, y_v + 0.28), (12.70, y_mma + 0.79), color="#2563eb", rad=-0.08)
    rounded_box(
        ax,
        9.88,
        4.06,
        2.14,
        0.43,
        tr(
            lang,
            "incoming from prior WG0(k−1) · so_ready[k−1]",
            "来自前一轮 WG0(k−1) · so_ready[k−1]",
        ),
        "#f3e8ff",
        fontsize=6.4,
        weight="normal",
        edgecolor="#7c3aed",
        linestyle="--",
    )
    arrow(
        ax,
        (10.95, 4.49),
        (10.95, y_mma + 0.03),
        color="#7c3aed",
        linestyle="--",
        linewidth=1.1,
    )

    # Each completed PV half releases the one V workspace for the matching k half.
    rounded_box(
        ax,
        11.72,
        y_v + 0.06,
        1.38,
        0.70,
        "gather V(k)\npart0",
        COLORS["tma"],
        fontsize=7.6,
    )
    rounded_box(
        ax,
        13.80,
        y_v + 0.06,
        1.38,
        0.70,
        "gather V(k)\npart1",
        COLORS["tma"],
        fontsize=7.6,
    )
    arrow(ax, (10.95, y_mma + 0.79), (12.12, y_v + 0.06), color="#059669", rad=0.04)
    _barrier_note(ax, 11.78, 5.94, "sv_part_done[k−1]")
    arrow(ax, (13.03, y_mma + 0.79), (14.20, y_v + 0.06), color="#059669", rad=0.04)
    _barrier_note(ax, 13.87, 5.94, "sv_done[k−1]")

    # Mask production runs on a separate warp and feeds WG0.
    rounded_box(
        ax,
        5.72,
        y_mask + 0.06,
        1.92,
        0.70,
        tr(
            lang,
            "pack mask(k)\n8 indices / lane",
            "pack mask(k)\n每个 lane 处理 8 indices",
        ),
        COLORS["barrier"],
        fontsize=7.6,
    )

    # WG0 consumes L(k), computes W in registers while PV(k-1) is active, then safely reuses W/O.
    rounded_box(
        ax,
        8.72,
        y_soft + 0.03,
        1.55,
        0.76,
        tr(lang, "load L(k)\nfrom TMEM", "从 TMEM\n读取 L(k)"),
        COLORS["tmem"],
        fontsize=7.8,
    )
    rounded_box(
        ax,
        10.55,
        y_soft + 0.03,
        2.22,
        0.76,
        tr(
            lang,
            "mask L(k)\nmax / exp / row sum",
            "mask L(k)\nmax / exp / row sum",
        ),
        COLORS["softmax"],
        fontsize=7.8,
    )
    rounded_box(
        ax,
        13.78,
        y_soft + 0.03,
        3.05,
        0.76,
        tr(
            lang,
            "after sv_done(k−1): write W(k)\noptional O rescale · so_ready[k]",
            "等待 sv_done(k−1) 后：写入 W(k)\n按需重缩放 O · so_ready[k]",
        ),
        COLORS["softmax"],
        fontsize=7.7,
    )
    arrow(ax, (8.0, y_mma + 0.03), (9.32, y_soft + 0.79), color="#7c3aed", rad=0.05)
    _barrier_note(ax, 8.50, 3.76, "qk_done[k]")
    arrow(ax, (7.64, y_mask + 0.41), (10.90, y_soft + 0.79), color="#d97706", rad=0.07)
    _barrier_note(ax, 8.35, 2.86, "k_valid_ready[k]")
    arrow(ax, (12.77, y_soft + 0.41), (13.78, y_soft + 0.41), color="#7c3aed")
    _barrier_note(ax, 13.25, y_soft + 0.72, "wait sv_done[k−1]")
    arrow(ax, (13.03, y_mma + 0.03), (14.30, y_soft + 0.79), color="#059669", rad=-0.05)
    arrow(
        ax,
        (9.49, y_soft + 0.03),
        (9.49, 0.94),
        label=tr(
            lang,
            "TMEM L load complete → p_free[k] → future QK(k+1)",
            "TMEM L 读取完成 → p_free[k] → 后续 QK(k+1)",
        ),
        color="#7c3aed",
        linestyle="--",
        label_offset=(1.45, 0.0),
    )
    arrow(
        ax,
        (16.35, y_soft + 0.03),
        (17.52, 0.94),
        label=tr(lang, "so_ready[k] → future PV(k)", "so_ready[k] → 后续 PV(k)"),
        color="#7c3aed",
        linestyle="--",
        label_offset=(0.10, 0.0),
        rad=0.05,
    )
    rounded_box(
        ax,
        15.42,
        y_mask + 0.06,
        2.08,
        0.70,
        tr(
            lang,
            "pack mask(k+2)\nreuse slot k%2",
            "pack mask(k+2)\n复用 slot k%2",
        ),
        COLORS["barrier"],
        fontsize=7.4,
    )
    arrow(
        ax,
        (11.35, y_soft + 0.79),
        (15.80, y_mask + 0.06),
        color="#d97706",
        linestyle="--",
        rad=-0.08,
    )
    _barrier_note(ax, 13.55, 2.88, "k_valid_free[k]", fontsize=6.6)

    # Exact scope of the two-slot ring in the pinned regular kernel.
    rounded_box(
        ax,
        0.62,
        0.18,
        17.76,
        0.62,
        tr(
            lang,
            "NUM_BUFS=2 is a barrier/phase ring: slot=k%2, phase=(k//2)&1. K, V, and s_smem_gemm are single in-place workspaces (plus two small mask slots), not two complete K/V stages.",
            "NUM_BUFS=2 是 barrier/phase ring：slot=k%2，phase=(k//2)&1。K、V 与 s_smem_gemm 都是单份原位 workspace（另有两个小型 mask slots），并非两份完整 K/V stages。",
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
