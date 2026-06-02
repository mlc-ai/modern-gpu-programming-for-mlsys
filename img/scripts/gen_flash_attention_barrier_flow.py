#!/usr/bin/env python3
"""Generate Flash Attention 4 barrier diagrams."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "tma": "#f6c7c8",
    "mma": "#c7dbf4",
    "softmax": "#ffe3a6",
    "wg2": "#cfead0",
    "store": "#c7ead2",
    "bar": "#eef6ff",
    "merge": "#eee7fb",
}


def box(ax, x, y, w, h, text, color, fs=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.04",
        linewidth=1.15,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold")


def arrow(ax, x1, y1, x2, y2, color="#4b5563", rad=0.0, lw=1.25):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def label(ax, x, y, text, fs=8.5, color="#374151"):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=color,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#d1d5db"),
    )


def gen_main_handoff():
    fig, ax = plt.subplots(figsize=(16.5, 5.8))
    ax.set_xlim(0, 18.25)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    ax.text(9.1, 5.45, "Flash Attention 4 MMA Input Gates", ha="center", fontsize=17, weight="bold")
    ax.text(
        9.1,
        5.12,
        "score MMA and value MMA have different input gates",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )

    # Score path.
    ax.text(0.45, 4.55, "Score path", fontsize=11, weight="bold", color="#1f2937")
    box(ax, 0.45, 3.75, 1.75, 0.72, "TMA load\nQ -> SMEM", COLORS["tma"], fs=8.6)
    box(ax, 0.45, 2.85, 1.75, 0.72, "TMA load\nK -> SMEM", COLORS["tma"], fs=8.6)
    box(ax, 2.85, 3.22, 2.0, 0.72, "wait\nq_load.full\nkv_load.full", COLORS["bar"], fs=8.0)
    box(ax, 5.45, 3.22, 1.75, 0.72, "score MMA\nQ,K -> S", COLORS["mma"], fs=8.6)
    box(ax, 7.75, 3.22, 1.35, 0.72, "s_ready.full", COLORS["bar"], fs=8.2)
    box(ax, 9.65, 3.22, 1.9, 0.72, "softmax\nS -> P", COLORS["softmax"], fs=8.6)
    arrow(ax, 2.2, 4.11, 2.85, 3.58)
    arrow(ax, 2.2, 3.21, 2.85, 3.58)
    arrow(ax, 4.85, 3.58, 5.45, 3.58)
    arrow(ax, 7.2, 3.58, 7.75, 3.58)
    arrow(ax, 9.1, 3.58, 9.65, 3.58)

    # Value path.
    ax.text(0.45, 2.05, "Value path", fontsize=11, weight="bold", color="#1f2937")
    box(ax, 0.45, 1.22, 1.75, 0.72, "TMA load\nV -> SMEM", COLORS["tma"], fs=8.6)
    box(ax, 2.85, 1.22, 2.05, 0.72, "wait\nkv_load.full", COLORS["bar"], fs=8.2)
    box(ax, 5.45, 1.22, 2.35, 0.72, "value gate\np_o_rescale.full\np_ready_2.full", COLORS["merge"], fs=7.7)
    box(ax, 8.45, 1.22, 1.85, 0.72, "value MMA\nP,V -> O", COLORS["mma"], fs=8.6)
    box(ax, 10.85, 1.22, 1.35, 0.72, "o_ready.full", COLORS["bar"], fs=8.2)
    box(ax, 12.65, 1.22, 1.55, 0.72, "epilogue\nO -> O_smem", COLORS["wg2"], fs=8.0)
    box(ax, 14.65, 1.22, 1.45, 0.72, "bar_corr\n_epi_full", COLORS["bar"], fs=8.0)
    box(ax, 16.55, 1.22, 1.45, 0.72, "TMA store\nO_smem -> GMEM", COLORS["store"], fs=7.8)
    arrow(ax, 2.2, 1.58, 2.85, 1.58)
    arrow(ax, 4.9, 1.58, 5.45, 1.58)
    arrow(ax, 7.8, 1.58, 8.45, 1.58)
    arrow(ax, 10.3, 1.58, 10.85, 1.58)
    arrow(ax, 12.2, 1.58, 12.65, 1.58)
    arrow(ax, 14.2, 1.58, 14.65, 1.58)
    arrow(ax, 16.1, 1.58, 16.55, 1.58)

    # Softmax and WG2 prerequisites for the value gate.
    arrow(ax, 10.6, 3.22, 6.62, 1.94, color="#b45309", rad=-0.08, lw=1.1)
    label(ax, 8.05, 2.38, "P ready:\nfirst 96 cols, then final 32", fs=7.8)
    box(ax, 5.25, 0.22, 2.55, 0.55, "WG2 releases or rescales O", COLORS["wg2"], fs=8.0)
    arrow(ax, 6.62, 0.77, 6.62, 1.22, color="#166534", lw=1.1)

    fig.savefig("../flash_attention_main_handoff.png", dpi=170, bbox_inches="tight", facecolor="white")


def gen_softmax_correction():
    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 5.35)
    ax.axis("off")

    ax.text(6.0, 5.0, "Softmax / WG2 Scale Slot Handshake", ha="center", fontsize=17, weight="bold")
    ax.text(
        6.0,
        4.66,
        "softmax_corr.full and softmax_corr.empty protect one SMEM slot, not the P/O compute path",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )

    # Main full/empty lifecycle.
    box(ax, 0.55, 3.35, 1.75, 0.72, "slot empty\nsoftmax may write", COLORS["bar"], fs=8.8)
    box(ax, 2.8, 3.35, 1.95, 0.72, "softmax writes\nacc_scale / row_sum", COLORS["softmax"], fs=8.8)
    box(ax, 5.25, 3.35, 1.7, 0.72, "bar_softmax\n_corr_full", COLORS["bar"], fs=8.4)
    box(ax, 7.45, 3.35, 1.9, 0.72, "WG2 reads\nthat SMEM slot", COLORS["wg2"], fs=8.8)
    box(ax, 9.85, 3.35, 1.7, 0.72, "bar_softmax\n_corr_empty", COLORS["bar"], fs=8.4)

    arrow(ax, 2.3, 3.71, 2.8, 3.71)
    arrow(ax, 4.75, 3.71, 5.25, 3.71)
    arrow(ax, 6.95, 3.71, 7.45, 3.71)
    arrow(ax, 9.35, 3.71, 9.85, 3.71)
    arrow(ax, 10.7, 3.35, 1.42, 3.35, color="#7c3aed", rad=-0.18, lw=1.35)
    label(ax, 5.95, 2.7, "empty goes back to softmax:\nthis slot can be overwritten next time", fs=8.4)

    ax.text(0.7, 4.25, "producer", fontsize=9, weight="bold", color="#92400e")
    ax.text(7.95, 4.25, "consumer", fontsize=9, weight="bold", color="#166534")
    ax.text(0.6, 1.95, "What the full/empty pair proves", fontsize=11, weight="bold")
    ax.text(
        0.6,
        1.58,
        "full: WG2 may read the scale or final row_sum from SMEM\n"
        "empty: softmax may reuse that same SMEM slot\n"
        "scope: one slot per Q stage, arrived by 128 warpgroup threads",
        fontsize=9.2,
        color="#374151",
        va="top",
    )

    ax.text(7.05, 1.95, "What it does not prove", fontsize=11, weight="bold")
    ax.text(
        7.05,
        1.58,
        "not: P has been written to TMEM\n"
        "not: O has been rescaled\n"
        "not: value MMA may start\n"
        "those are covered by p_o_rescale.full and p_ready_2.full",
        fontsize=9.2,
        color="#374151",
        va="top",
    )

    fig.savefig("../flash_attention_softmax_correction.png", dpi=170, bbox_inches="tight", facecolor="white")


def main():
    gen_main_handoff()
    gen_softmax_correction()


if __name__ == "__main__":
    main()
