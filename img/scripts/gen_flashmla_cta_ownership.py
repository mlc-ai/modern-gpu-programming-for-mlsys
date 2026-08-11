#!/usr/bin/env python3
"""Generate the sparse FlashMLA head128 regular 2-CTA ownership diagram."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from flashmla_diagram_common import (
    COLORS,
    arrow,
    configure_style,
    plain_rect,
    rounded_box,
    save_figure,
    tr,
)


def _partition_bar(ax, x, y, w, h, *, labels, colors, split="vertical"):
    if split == "vertical":
        cell_w = w / len(labels)
        for idx, (label, color) in enumerate(zip(labels, colors, strict=True)):
            plain_rect(ax, x + idx * cell_w, y, cell_w, h, color)
            ax.text(
                x + (idx + 0.5) * cell_w,
                y + h / 2,
                label,
                ha="center",
                va="center",
                fontsize=8.6,
                weight="bold",
                color=COLORS["ink"],
                zorder=4,
            )
    else:
        cell_h = h / len(labels)
        for idx, (label, color) in enumerate(zip(labels, colors, strict=True)):
            plain_rect(ax, x, y + (len(labels) - idx - 1) * cell_h, w, cell_h, color)
            ax.text(
                x + w / 2,
                y + (len(labels) - idx - 0.5) * cell_h,
                label,
                ha="center",
                va="center",
                fontsize=8.6,
                weight="bold",
                color=COLORS["ink"],
                zorder=4,
            )


def _logical_matrix(ax, x, y, w, h, *, title, col_axis, lang, kind):
    ax.text(
        x + w / 2,
        y + h + 0.38,
        title,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
    )
    ax.text(
        x + w / 2,
        y - 0.3,
        col_axis,
        ha="center",
        va="center",
        fontsize=8,
        color=COLORS["muted"],
    )
    ax.text(
        x - 0.34,
        y + h / 2,
        (
            tr(lang, "Q-head rows", "Q head 行")
            if kind == "k"
            else tr(lang, "W rows (Q-head ownership)", "W 行（沿用 Q head 所有权）")
        ),
        ha="center",
        va="center",
        rotation=90,
        fontsize=8,
        color=COLORS["muted"],
    )
    cell_colors = [
        [COLORS["cta0"], COLORS["cross01"]],
        [COLORS["cross10"], COLORS["cta1"]],
    ]
    q_labels = ["Q0", "Q1"] if kind == "k" else ["W(Q0)", "W(Q1)"]
    b_labels = ["K0", "K1"] if kind == "k" else ["V0", "V1"]
    for row in range(2):
        for col in range(2):
            cx = x + col * w / 2
            cy = y + (1 - row) * h / 2
            plain_rect(ax, cx, cy, w / 2, h / 2, cell_colors[row][col])
            ax.text(
                cx + w / 4,
                cy + h / 4,
                f"{q_labels[row]} × {b_labels[col]}",
                ha="center",
                va="center",
                fontsize=8.5 if kind == "v" else 9,
                weight="bold",
                color=COLORS["ink"],
                zorder=4,
            )
    ax.text(
        x + 0.10,
        y + h - 0.11,
        "heads 0:64",
        ha="left",
        va="top",
        fontsize=6.7,
        color=COLORS["muted"],
        zorder=5,
    )
    ax.text(
        x + 0.10,
        y + h / 2 - 0.11,
        "heads 64:128",
        ha="left",
        va="top",
        fontsize=6.7,
        color=COLORS["muted"],
        zorder=5,
    )


def draw(lang: str, output: str, font_path: str | None = None) -> None:
    configure_style(lang, font_path)
    fig, ax = plt.subplots(figsize=(16.0, 8.8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.8)
    ax.axis("off")

    ax.text(
        8,
        8.5,
        tr(
            lang,
            "Sparse FlashMLA Head128 Regular: 2-CTA Ownership",
            "Sparse FlashMLA Head128 Regular：2-CTA 所有权",
        ),
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
    )
    ax.text(
        8,
        8.12,
        tr(
            lang,
            "one CTA pair per query row · B_H=128 · B_TOPK=128 · D_V=512",
            "每个 query row 使用一个 CTA pair · B_H=128 · B_TOPK=128 · D_V=512",
        ),
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    headings = [
        (0.45, tr(lang, "Q ownership", "Q 所有权")),
        (5.5, tr(lang, "K selected-token ownership", "K 的选中 token 所有权")),
        (10.55, tr(lang, "V feature ownership", "V feature 所有权")),
    ]
    for x, text in headings:
        ax.text(
            x,
            7.66,
            text,
            ha="left",
            va="center",
            fontsize=11,
            weight="bold",
            color=COLORS["ink"],
        )

    _partition_bar(
        ax,
        0.65,
        6.13,
        4.15,
        1.15,
        labels=("CTA 0 · Q heads 0:64", "CTA 1 · Q heads 64:128"),
        colors=(COLORS["cta0"], COLORS["cta1"]),
        split="horizontal",
    )
    ax.text(
        2.72,
        5.88,
        tr(lang, "each CTA: 64 × d_qk", "每个 CTA：64 × d_qk"),
        ha="center",
        fontsize=8,
        color=COLORS["muted"],
    )

    _partition_bar(
        ax,
        5.7,
        6.13,
        4.15,
        1.15,
        labels=("CTA 0\nslots 0:64", "CTA 1\nslots 64:128"),
        colors=(COLORS["cta0"], COLORS["cta1"]),
    )
    ax.text(
        7.77,
        5.88,
        tr(lang, "within the current sparse B_TOPK tile", "当前 sparse B_TOPK tile 内"),
        ha="center",
        fontsize=8,
        color=COLORS["muted"],
    )

    _partition_bar(
        ax,
        10.75,
        6.13,
        4.15,
        1.15,
        labels=("CTA 0\nfeatures 0:256", "CTA 1\nfeatures 256:512"),
        colors=(COLORS["cta0"], COLORS["cta1"]),
    )
    ax.text(
        12.82,
        5.88,
        tr(
            lang,
            "each half covers all 128 selected-token rows",
            "每个 feature half 覆盖全部 128 个选中 token 行",
        ),
        ha="center",
        fontsize=8,
        color=COLORS["muted"],
    )

    _logical_matrix(
        ax,
        0.85,
        1.75,
        5.3,
        2.7,
        title=tr(
            lang,
            "QK · cta_group::2 → logits L  [128 heads × 128 slots]",
            "QK · cta_group::2 → logits L  [128 heads × 128 slots]",
        ),
        col_axis=tr(lang, "selected-token columns", "选中 token 列"),
        lang=lang,
        kind="k",
    )
    _logical_matrix(
        ax,
        9.85,
        1.75,
        5.3,
        2.7,
        title=tr(
            lang,
            "PV · cta_group::2 → output O  [128 heads × 512 features]",
            "PV · cta_group::2 → output O  [128 heads × 512 features]",
        ),
        col_axis=tr(lang, "output-feature columns", "输出 feature 列"),
        lang=lang,
        kind="v",
    )

    rounded_box(
        ax,
        6.70,
        2.48,
        2.6,
        1.25,
        tr(
            lang,
            "L → WG0 softmax → W\nsame head × slot shape",
            "L → WG0 softmax → W\n保持相同的 head × slot 形状",
        ),
        COLORS["softmax"],
        fontsize=8.7,
    )
    arrow(ax, (6.15, 3.1), (6.70, 3.1), label="L")
    arrow(ax, (9.30, 3.1), (9.85, 3.1), label="W")
    # Ownership-to-logical-tile guide arrows.
    arrow(ax, (2.72, 5.76), (2.72, 4.68), color=COLORS["cta0_dark"], linewidth=1.0)
    arrow(
        ax, (7.77, 5.76), (5.4, 4.7), color=COLORS["cta1_dark"], linewidth=1.0, rad=0.08
    )
    arrow(ax, (12.82, 5.76), (12.82, 4.68), color=COLORS["cta1_dark"], linewidth=1.0)

    rounded_box(
        ax,
        1.0,
        0.52,
        14.0,
        0.62,
        tr(
            lang,
            "Every colored quadrant is a required block product. Cross-colored quadrants are not cross-CTA reductions; cta_group::2 combines operand halves without an explicit DSMEM copy.",
            "每个彩色象限都是必需的 block product。交叉配色象限不是跨 CTA reduction；cta_group::2 在没有显式 DSMEM copy 的情况下组合 operand halves。",
        ),
        COLORS["note"],
        fontsize=8.7,
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
