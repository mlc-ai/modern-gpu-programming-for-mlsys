#!/usr/bin/env python3
"""Generate the sparse FlashMLA head128 regular steady-state pipeline diagram."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from flashmla_diagram_common import (
    COLORS,
    arrow as _arrow,
    configure_style,
    rounded_box as _rounded_box,
    save_figure,
    tr,
)


def box(ax, x, y, w, h, text, color, *, fontsize=10.8, **kwargs):
    """Draw a timeline node sized for the book's content column."""

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
    """Draw a dependency edge with an HTML-readable label."""

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
    """Draw one layer of the steady-state dependency story."""

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


def lane(ax, y, title, role):
    """Draw a named warpgroup lane and its baseline."""

    box(
        ax,
        0.48,
        y - 0.38,
        1.42,
        0.76,
        f"{title}\n{role}",
        COLORS["neutral"],
        fontsize=10.2,
    )
    ax.plot([2.08, 13.45], [y, y], color="#e5e7eb", lw=1.0, zorder=0)


def tag(ax, x, y, text, *, fontsize=9.8):
    """Place a completion/ready barrier name away from node text."""

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#92400e",
        bbox=dict(
            boxstyle="round,pad=0.10",
            facecolor=COLORS["barrier"],
            edgecolor="#d97706",
        ),
        zorder=7,
    )


def draw(lang: str, output: str, font_path: str | None = None) -> None:
    configure_style(lang, font_path)
    fig, ax = plt.subplots(figsize=(14.0, 13.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 13.2)
    ax.axis("off")

    ax.text(
        7,
        12.84,
        tr(
            lang,
            "Sparse FlashMLA Head128 Regular: Steady-State Pipeline",
            "Sparse FlashMLA Head128 Regular：稳态 Pipeline",
        ),
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
    )
    ax.text(
        7,
        12.30,
        tr(
            lang,
            "softmax(k−1) may overlap QK(k); once QK(k) completes,\nsoftmax(k) may overlap asynchronous PV(k−1) · box widths are not cycle measurements",
            "softmax(k−1) 可与 QK(k) 重叠；QK(k) 完成后，\nsoftmax(k) 可与异步 PV(k−1) 重叠 · 方框宽度不表示 cycle 数",
        ),
        ha="center",
        va="center",
        fontsize=11.5,
        color=COLORS["muted"],
        linespacing=1.25,
    )

    panel(
        ax,
        0.25,
        6.55,
        13.50,
        5.25,
        tr(
            lang,
            "A · One issuer order; four independent part handshakes",
            "A · 唯一 issuer 的顺序与四个独立分段握手",
        ),
        tr(lang, "box widths are not cycle measurements", "方框宽度不表示 cycle 数"),
    )

    ax.text(
        1.05,
        10.72,
        tr(
            lang,
            "WG3 · CTA0 warp 12\nsole MMA issuer",
            "WG3 · CTA0 warp 12\n唯一 MMA issuer",
        ),
        ha="center",
        va="center",
        fontsize=10.5,
        weight="bold",
        color=COLORS["ink"],
    )
    issuer_boxes = [
        (2.25, "wait p_free[k−1]\nQK(k) · SS prefix"),
        (5.10, "QK(k) · TS suffix384"),
        (7.95, "wait so_ready[k−1]\nPV(k−1) · part0"),
        (10.80, "PV(k−1) · part1"),
    ]
    for x, text in issuer_boxes:
        box(ax, x, 10.25, 2.25, 0.90, text, COLORS["mma"], fontsize=10.4)
    for x0, x1 in ((4.50, 5.10), (7.35, 7.95), (10.20, 10.80)):
        edge(ax, (x0, 10.70), (x1, 10.70), color=COLORS["ink"])
    ax.text(
        7.0,
        9.93,
        tr(lang, "strict issue order", "严格串行发起顺序"),
        ha="center",
        va="center",
        fontsize=10.0,
        color=COLORS["muted"],
        style="italic",
    )

    handshakes = [
        (
            0.70,
            8.33,
            tr(
                lang,
                "WG1: k_part0_ready[k]\n→ QK SS prefix\n→ qk_part_done[k]\n→ gather K(k+1) part0",
                "WG1：k_part0_ready[k]\n→ QK SS prefix\n→ qk_part_done[k]\n→ gather K(k+1) part0",
            ),
            "#dbeafe",
        ),
        (
            7.10,
            8.33,
            tr(
                lang,
                "WG1: k_part1_ready[k]\n→ QK TS suffix384\n→ qk_done[k]\n→ gather K(k+1) part1",
                "WG1：k_part1_ready[k]\n→ QK TS suffix384\n→ qk_done[k]\n→ gather K(k+1) part1",
            ),
            "#dbeafe",
        ),
        (
            0.70,
            6.88,
            tr(
                lang,
                "WG2: v_part0_ready[k−1]\n→ PV part0\n→ sv_part_done[k−1]\n→ gather V(k) part0",
                "WG2：v_part0_ready[k−1]\n→ PV part0\n→ sv_part_done[k−1]\n→ gather V(k) part0",
            ),
            COLORS["tma"],
        ),
        (
            7.10,
            6.88,
            tr(
                lang,
                "WG2: v_part1_ready[k−1]\n→ PV part1\n→ sv_done[k−1]\n→ gather V(k) part1",
                "WG2：v_part1_ready[k−1]\n→ PV part1\n→ sv_done[k−1]\n→ gather V(k) part1",
            ),
            COLORS["tma"],
        ),
    ]
    for x, y, text, color in handshakes:
        box(ax, x, y, 6.20, 1.12, text, color, fontsize=10.3, weight="normal")

    panel(
        ax,
        0.25,
        1.05,
        13.50,
        5.15,
        tr(
            lang,
            "B · Tile k: mask-slot reuse and WG0 handoff",
            "B · Tile k：mask slot 复用与 WG0 交接",
        ),
        tr(
            lang,
            "arrows name the barrier that guards reuse",
            "箭头标出保护复用的 barrier",
        ),
    )
    box(
        ax,
        0.70,
        4.35,
        2.45,
        0.92,
        tr(lang, "WG3 warp 13\npack mask(k)", "WG3 warp 13\npack mask(k)"),
        COLORS["barrier"],
        fontsize=10.8,
    )
    box(
        ax,
        5.35,
        4.35,
        3.10,
        0.92,
        tr(
            lang,
            "WG0 consumes mask(k)\nwhile processing L(k)",
            "WG0 处理 L(k) 时\n读取 mask(k)",
        ),
        COLORS["softmax"],
        fontsize=10.8,
    )
    box(
        ax,
        10.35,
        4.35,
        2.95,
        0.92,
        "WG3 warp 13\npack mask(k+2)\nreuse slot k%2",
        COLORS["barrier"],
        fontsize=10.6,
    )
    edge(
        ax,
        (3.15, 4.81),
        (5.35, 4.81),
        label="k_valid_ready[k]",
        label_offset=(0.0, 0.22),
        fontsize=9.7,
        color="#d97706",
    )
    edge(
        ax,
        (8.45, 4.81),
        (10.35, 4.81),
        label="k_valid_free[k]",
        label_offset=(0.0, 0.22),
        fontsize=9.7,
        color="#d97706",
        linestyle="--",
    )

    box(
        ax,
        0.70,
        2.35,
        2.45,
        1.00,
        tr(lang, "qk_done[k]\nload L(k) from TMEM", "qk_done[k]\n从 TMEM 读取 L(k)"),
        COLORS["tmem"],
        fontsize=10.8,
    )
    box(
        ax,
        5.05,
        2.35,
        3.25,
        1.00,
        tr(lang, "WG0: mask · max\nexp · row sum", "WG0：mask · max\nexp · row sum"),
        COLORS["softmax"],
        fontsize=10.8,
    )
    box(
        ax,
        9.65,
        2.20,
        3.65,
        1.30,
        tr(
            lang,
            "wait sv_done(k−1)\nwrite W(k) · optional O rescale\narrive so_ready[k]",
            "等待 sv_done(k−1)\n写 W(k) · 按需重缩放 O\narrive so_ready[k]",
        ),
        COLORS["softmax"],
        fontsize=10.5,
    )
    edge(
        ax,
        (3.15, 2.85),
        (5.05, 2.85),
        color=COLORS["line"],
    )
    edge(
        ax,
        (8.30, 2.85),
        (9.65, 2.85),
        color="#7c3aed",
    )
    ax.text(
        2.0,
        1.65,
        "p_free[k] → QK(k+1)",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#7c3aed",
        weight="bold",
    )
    edge(
        ax,
        (1.95, 2.35),
        (1.95, 1.87),
        color="#7c3aed",
        linestyle="--",
    )
    ax.text(
        11.5,
        1.65,
        "so_ready[k] → PV(k)",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#7c3aed",
        weight="bold",
    )
    edge(
        ax,
        (11.5, 2.20),
        (11.5, 1.87),
        color="#7c3aed",
        linestyle="--",
    )

    box(
        ax,
        0.52,
        0.10,
        12.96,
        0.72,
        tr(
            lang,
            "NUM_BUFS=2 is a barrier/phase ring: slot=k%2, phase=(k//2)&1.\nK, V, and W use one in-place workspace each; only the small validity mask has two data slots.",
            "NUM_BUFS=2 是 barrier/phase ring：slot=k%2，phase=(k//2)&1。\nK、V、W 各自只有一份原位 workspace；只有小型 validity mask 有两个 data slots。",
        ),
        COLORS["note"],
        fontsize=9.8,
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
