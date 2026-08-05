#!/usr/bin/env python3
"""Generate the Flash Attention 4 pipeline diagram."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]


def configure_font(lang, font_path=None):
    if lang != "zh":
        return
    if not font_path:
        raise ValueError("Chinese output requires --font-path pointing to a CJK font")
    path = Path(font_path)
    if not path.exists():
        raise FileNotFoundError("Chinese output requires --font-path pointing to a CJK font")
    font_manager.fontManager.addfont(path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams["svg.fonttype"] = "path"


COLORS = {
    "tma": "#bfdbfe",
    "mma": "#bbf7d0",
    "softmax": "#ddd6fe",
    "corr": "#ccfbf1",
    "label": "#f8fafc",
}


def block(ax, x, y, w, h, text, color, fs=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.05",
        linewidth=1.2,
        edgecolor="#1f2937",
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold", zorder=4)


def arrow(ax, x1, y1, x2, y2, label=None, color="#4b5563", rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.1,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        alpha=0.78,
        zorder=1,
    )
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, label, fontsize=7, color=color, ha="center", zorder=2)


def main(lang="en", font_path=None):
    configure_font(lang, font_path)
    zh = lang == "zh"
    tr = lambda en, cn: cn if zh else en
    fig, ax = plt.subplots(figsize=(18.0, 7.5))
    ax.set_xlim(0, 16.0)
    ax.set_ylim(-0.25, 6.65)
    ax.axis("off")

    ax.text(8.0, 6.45, tr("Flash Attention 4 Pipeline Timeline", "Flash Attention 4 Pipeline 时间线"), ha="center", fontsize=17, weight="bold")
    ax.text(
        8.0,
        6.18,
        tr(
            "representative issue order; the MMA warp interleaves PV MMA for current V with QK$^{\\mathsf{T}}$ MMA for next K",
            "一种典型的发起顺序：MMA warp 交错执行当前 V 的 PV MMA 和下一个 K 的 QK$^{\\mathsf{T}}$ MMA",
        ),
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    arrow(ax, 1.75, 5.95, 15.55, 5.95, color="#9ca3af")
    ax.text(8.65, 6.02, tr("time", "时间"), ha="center", va="bottom", fontsize=8, color="#6b7280", style="italic", zorder=2)

    rows = [
        ("WG3 warp 1", tr("TMA load", "TMA 加载"), 5.0),
        ("WG3 warp 0", tr("MMA issue", "发起 MMA"), 4.0),
        ("WG0", tr("softmax Q stage 0", "softmax Q stage 0"), 3.0),
        ("WG1", tr("softmax Q stage 1", "softmax Q stage 1"), 2.0),
        ("WG2", tr("correction / epilogue", "重缩放 / epilogue"), 1.0),
        ("WG3 warp 2", tr("TMA store", "TMA 写回"), 0.0),
    ]
    for name, role, y in rows:
        block(ax, 0.15, y + 0.12, 1.35, 0.62, f"{name}\n{role}", COLORS["label"], fs=8)
        ax.plot([1.75, 15.8], [y + 0.43, y + 0.43], color="#e5e7eb", lw=1, zorder=0)

    # TMA load order from the source: Q0, K_last, Q1, V_last, then K/V stream.
    for x, text in [
        (2.0, tr("load Q0", "加载 Q0")),
        (3.1, tr("load K[n-1]", "加载 K[n-1]")),
        (4.2, tr("load Q1", "加载 Q1")),
        (5.3, tr("load V[n-1]", "加载 V[n-1]")),
        (6.7, tr("load K[n-2]", "加载 K[n-2]")),
        (7.8, tr("load V[n-2]", "加载 V[n-2]")),
        (9.2, tr("load K[n-3]", "加载 K[n-3]")),
        (10.3, tr("load V[n-3]", "加载 V[n-3]")),
    ]:
        block(ax, x, 5.12, 0.88, 0.62, text, COLORS["tma"], fs=8)
    ax.text(11.55, 5.43, "...", fontsize=13, color="#6b7280")

    # MMA issue order: bootstrap QK^T, then interleave PV for current V with QK^T for next K.
    mma_blocks = [
        (4.0, "QK$^{\\mathsf{T}}$ MMA\nQ0 @ K[n-1]^T", COLORS["mma"]),
        (5.1, "QK$^{\\mathsf{T}}$ MMA\nQ1 @ K[n-1]^T", COLORS["mma"]),
        (6.35, "PV MMA\nP0 @ V[n-1]", COLORS["mma"]),
        (7.45, "QK$^{\\mathsf{T}}$ MMA\nQ0 @ K[n-2]^T", COLORS["mma"]),
        (8.55, "PV MMA\nP1 @ V[n-1]", COLORS["mma"]),
        (9.65, "QK$^{\\mathsf{T}}$ MMA\nQ1 @ K[n-2]^T", COLORS["mma"]),
        (10.75, "PV MMA\nP0 @ V[n-2]", COLORS["mma"]),
    ]
    for x, text, color in mma_blocks:
        block(ax, x, 4.12, 0.98, 0.66, text, color, fs=7.4)
    ax.text(11.82, 4.43, "...", fontsize=13, color="#6b7280")
    block(ax, 12.35, 4.12, 0.92, 0.66, "PV MMA\nP0 @ V[0]", COLORS["mma"], fs=7.2)
    block(ax, 13.37, 4.12, 0.92, 0.66, "PV MMA\nP1 @ V[0]", COLORS["mma"], fs=7.2)

    # Softmax and correction events. Keep one dependency loop per Q stage readable.
    block(ax, 4.75, 3.12, 1.05, 0.66, tr("softmax S0\nwrite P0", "softmax S0\n写入 P0"), COLORS["softmax"], fs=8)
    block(ax, 5.85, 2.12, 1.05, 0.66, tr("softmax S1\nwrite P1", "softmax S1\n写入 P1"), COLORS["softmax"], fs=8)
    block(ax, 1.82, 1.12, 1.12, 0.66, tr("pre-release\nO0 / O1", "预先放行\nO0 / O1"), COLORS["corr"], fs=8)
    block(ax, 8.55, 1.12, 1.02, 0.66, tr("rescale O0\nif needed", "按需重缩放\nO0"), COLORS["corr"], fs=8)
    block(ax, 10.45, 1.12, 1.02, 0.66, tr("rescale O1\nif needed", "按需重缩放\nO1"), COLORS["corr"], fs=8)
    block(ax, 8.35, 3.12, 1.05, 0.66, tr("softmax S0\nwrite P0", "softmax S0\n写入 P0"), COLORS["softmax"], fs=8)
    block(ax, 10.25, 2.12, 1.05, 0.66, tr("softmax S1\nwrite P1", "softmax S1\n写入 P1"), COLORS["softmax"], fs=8)
    ax.text(11.82, 3.43, "...", fontsize=13, color="#6b7280")
    ax.text(11.82, 2.43, "...", fontsize=13, color="#6b7280")
    ax.text(11.82, 1.43, "...", fontsize=13, color="#6b7280")
    block(ax, 14.48, 1.12, 1.08, 0.66, tr("normalize\nO0 / O1", "归一化\nO0 / O1"), COLORS["corr"], fs=8)
    block(ax, 14.78, 0.12, 0.92, 0.62, tr("store O0,\nthen O1", "先写回 O0\n再写回 O1"), COLORS["tma"], fs=8)

    # Keep this as a timeline. Barrier-level dependencies are shown in
    # flash_attention_barrier_flow_v2.png; drawing them again here makes
    # the pipeline view harder to read.

    # Legend.
    legend = [
        (tr("TMA load/store", "TMA 加载/写回"), COLORS["tma"]),
        ("Tensor Core MMA", COLORS["mma"]),
        ("softmax", COLORS["softmax"]),
        (tr("correction/epilogue", "重缩放/epilogue"), COLORS["corr"]),
    ]
    lx = 2.0
    for name, color in legend:
        block(ax, lx, -0.12, 0.22, 0.16, "", color, fs=1)
        ax.text(lx + 0.3, -0.04, name, fontsize=8, va="center", color="#4b5563")
        lx += 1.55

    output = "../flash_attention_pipeline_v2_zh.svg" if zh else "../flash_attention_pipeline_v2.png"
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--font-path")
    args = parser.parse_args()
    main(args.lang, args.font_path)
