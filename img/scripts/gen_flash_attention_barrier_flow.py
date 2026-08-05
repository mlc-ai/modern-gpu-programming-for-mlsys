#!/usr/bin/env python3
"""Generate Flash Attention 4 barrier diagrams."""

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
    "smem": "#e9d5ff",
    "tmem": "#fed7aa",
    "mma": "#bbf7d0",
    "softmax": "#ddd6fe",
    "wg2": "#ccfbf1",
    "bar": "#fde68a",
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


def label(ax, x, y, text, fs=8.5, color="#374151", facecolor=None):
    if facecolor is None:
        facecolor = COLORS["bar"]
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=color,
        bbox=dict(boxstyle="round,pad=0.18", facecolor=facecolor, edgecolor="#d97706"),
    )


def gen_main_handoff(lang="en"):
    zh = lang == "zh"
    tr = lambda en, cn: cn if zh else en
    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    ax.text(7.75, 7.82, tr("Flash Attention 4: MMA Input Gates", "Flash Attention 4：MMA 的启动条件"), ha="center", fontsize=17, weight="bold")
    ax.text(
        7.75,
        7.46,
        tr("inputs that must be ready before each MMA may fire", "每次 MMA 发起前必须准备好的输入"),
        ha="center",
        fontsize=10,
        color="#4b5563",
    )

    # ---- QK^T MMA gate (top): Q and K must be in SMEM. ----
    ax.text(0.5, 7.10, tr(r"QK$^{\mathsf{T}}$ MMA gate", r"QK$^{\mathsf{T}}$ MMA"), fontsize=11.5, weight="bold", color="#1f2937")
    box(ax, 0.6, 6.28, 1.95, 0.62, tr("Q tile\nin SMEM", "SMEM 中的\nQ tile"), COLORS["smem"], fs=8.8)
    box(ax, 0.6, 5.46, 1.95, 0.62, tr("K tile\nin SMEM", "SMEM 中的\nK tile"), COLORS["smem"], fs=8.8)
    box(ax, 7.15, 5.72, 2.4, 0.95, tr("QK$^{\\mathsf{T}}$ MMA\nQ,K -> S", "QK$^{\\mathsf{T}}$ MMA\nQ,K -> S"), COLORS["mma"], fs=10.5)
    arrow(ax, 2.55, 6.59, 7.15, 6.35, rad=-0.04)
    arrow(ax, 2.55, 5.77, 7.15, 6.04, rad=0.04)
    label(ax, 4.70, 6.66, "q_load.full", fs=8.3)
    label(ax, 4.70, 5.70, "kv_load.full", fs=8.3)
    ax.text(8.35, 5.50, tr("fires when both SMEM tiles are ready", "两块 SMEM tile 就绪后发起"), ha="center", fontsize=7.8,
            color="#6b7280", style="italic")

    # ---- PV MMA gate (bottom): two separately issued inner-K slices. ----
    ax.text(0.5, 4.63, tr("PV MMA gates", "PV MMA 的两段计算"), fontsize=11.5, weight="bold", color="#1f2937")

    # First sub-MMA consumes the first 96 positions and may initialize or accumulate O.
    box(ax, 0.6, 3.72, 2.35, 0.62, tr("V rows 0:96\nin SMEM", "SMEM 中 V 的行 0:96"), COLORS["smem"], fs=8.6)
    box(ax, 0.6, 2.76, 3.25, 0.72, tr("P cols 0:96 in TMEM\n+ O slot ready (WG2)", "TMEM 中 P 的列 0:96\n+ O slot 已准备好（WG2）"), COLORS["tmem"], fs=8.2)
    box(ax, 6.10, 3.03, 3.20, 0.98, tr("first PV MMA\ninner-K 0:96\ninitialize or accumulate O", "第一段 PV MMA\ninner-K 0:96\n初始化或累加 O"), COLORS["mma"], fs=8.8)
    arrow(ax, 2.95, 4.03, 6.10, 3.73, rad=-0.04)
    arrow(ax, 3.85, 3.12, 6.10, 3.36, rad=0.02)
    label(ax, 4.55, 4.04, "kv_load.full", fs=8.3)
    label(ax, 5.00, 3.08, "p_o_rescale", fs=8.3)

    # Second sub-MMA waits only for the final P chunk; V was already proved ready.
    box(ax, 0.6, 1.55, 2.35, 0.62, tr("V rows 96:128\nin SMEM", "SMEM 中 V 的行 96:128"), COLORS["smem"], fs=8.6)
    box(ax, 0.6, 0.58, 3.25, 0.72, tr("P cols 96:128\nin TMEM", "TMEM 中 P 的列 96:128"), COLORS["tmem"], fs=8.4)
    box(ax, 6.10, 0.86, 3.20, 0.98, tr("second PV MMA\ninner-K 96:128\naccumulate into the same O", "第二段 PV MMA\ninner-K 96:128\n累加到同一块 O"), COLORS["mma"], fs=8.8)
    arrow(ax, 2.95, 1.86, 6.10, 1.56, rad=-0.04)
    arrow(ax, 3.85, 0.94, 6.10, 1.20, rad=0.02)
    arrow(ax, 7.70, 3.03, 7.70, 1.84, color="#374151")
    label(ax, 5.00, 0.90, "p_ready_2", fs=8.3)
    ax.text(8.85, 2.40, tr("same MMA warp; program order", "同一 MMA warp；按程序顺序发起"), ha="center", fontsize=7.7,
            color="#6b7280", style="italic")
    ax.text(4.65, 2.00, tr("V already covered by kv_load.full", "整块 V 已由 kv_load.full 确认就绪"), ha="center", fontsize=7.5,
            color="#6b7280", style="italic")

    # ---- Legend (right gap). ----
    lx = 11.2
    ax.text(lx, 7.10, tr("Legend", "图例"), fontsize=11.5, weight="bold", color="#1f2937")

    def swatch(y, color, text):
        box(ax, lx, y, 0.42, 0.34, "", color)
        ax.text(lx + 0.6, y + 0.17, text, ha="left", va="center", fontsize=8.7, color="#374151")

    swatch(6.55, COLORS["smem"], tr("SMEM operand (TMA-loaded)", "SMEM operand（由 TMA 加载）"))
    swatch(6.01, COLORS["tmem"], tr("TMEM operand / O slot", "TMEM operand / O slot"))
    swatch(5.47, COLORS["mma"], tr("MMA operation", "MMA 操作"))
    label(ax, lx + 0.5, 4.81, "barrier", fs=8.0)
    ax.text(lx + 1.05, 4.81, tr("gate that must complete before\nthe corresponding MMA phase", "对应 MMA 阶段发起前\n必须完成的 barrier"),
            ha="left", va="center", fontsize=8.7, color="#374151")
    ax.text(lx, 3.60, tr("The same kv_load.full pipeline tracks a K\nstage or a V stage, depending on the iteration.", "同一个 kv_load.full pipeline 会随迭代\n分别追踪 K stage 或 V stage。"),
            ha="left", va="center", fontsize=8.5, color="#6b7280", style="italic")

    output = "../flash_attention_main_handoff_zh.svg" if zh else "../flash_attention_main_handoff.png"
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")


def gen_softmax_correction(lang="en"):
    zh = lang == "zh"
    tr = lambda en, cn: cn if zh else en
    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 5.35)
    ax.axis("off")

    ax.text(6.0, 5.0, tr("Softmax / WG2 Scale Slot Handshake", "Softmax 与 WG2 的 Scale Slot 交接"), ha="center", fontsize=17, weight="bold")
    ax.text(
        6.0,
        4.66,
        tr("softmax_corr.full and softmax_corr.empty protect one SMEM slot, not the P/O compute path", "softmax_corr.full / empty 保护一个 SMEM slot，不表示 P/O 计算已经完成"),
        ha="center",
        fontsize=10,
        color="#4b5563",
    )

    # Main full/empty lifecycle.
    box(ax, 0.55, 3.35, 1.75, 0.72, tr("slot empty\nsoftmax may write", "slot 为空\nsoftmax 可以写入"), COLORS["bar"], fs=8.8)
    box(ax, 2.8, 3.35, 1.95, 0.72, tr("softmax writes\nacc_scale / row_sum", "softmax 写入\nacc_scale / row_sum"), COLORS["softmax"], fs=8.8)
    box(ax, 5.25, 3.35, 1.7, 0.72, "softmax_corr\n.full", COLORS["bar"], fs=8.4)
    box(ax, 7.45, 3.35, 1.9, 0.72, tr("WG2 reads\nthat SMEM slot", "WG2 读取\n该 SMEM slot"), COLORS["wg2"], fs=8.8)
    box(ax, 9.85, 3.35, 1.7, 0.72, "softmax_corr\n.empty", COLORS["bar"], fs=8.4)

    arrow(ax, 2.3, 3.71, 2.8, 3.71)
    arrow(ax, 4.75, 3.71, 5.25, 3.71)
    arrow(ax, 6.95, 3.71, 7.45, 3.71)
    arrow(ax, 9.35, 3.71, 9.85, 3.71)
    arrow(ax, 10.7, 3.35, 1.42, 3.35, color="#7c3aed", rad=-0.18, lw=1.35)
    label(ax, 5.95, 2.7, tr("empty goes back to softmax:\nthis slot can be overwritten next time", "empty 返回 softmax：\n下一轮可以覆盖该 slot"), fs=8.4)

    ax.text(0.7, 4.25, "producer", fontsize=9, weight="bold", color="#92400e")
    ax.text(7.95, 4.25, "consumer", fontsize=9, weight="bold", color="#166534")
    ax.text(0.6, 1.95, tr("What the full/empty pair proves", "这对 full/empty barriers 能证明什么"), fontsize=11, weight="bold")
    ax.text(
        0.6,
        1.58,
        tr(
            "full: WG2 may read the scale or final row_sum from SMEM\n"
            "empty: softmax may reuse that same SMEM slot\n"
            "scope: one slot per Q stage, arrived by 128 warpgroup threads",
            "full：WG2 可以从 SMEM 读取 scale 或最终 row_sum\n"
            "empty：softmax 可以复用同一个 SMEM slot\n"
            "scope：每个 Q stage 一个 slot，由 warpgroup 的 128 个 threads 报告 arrival",
        ),
        fontsize=9.2,
        color="#374151",
        va="top",
    )

    ax.text(7.05, 1.95, tr("What it does not prove", "这对 barriers 不能证明什么"), fontsize=11, weight="bold")
    ax.text(
        7.05,
        1.58,
        tr(
            "not: P has been written to TMEM\n"
            "not: O has been rescaled\n"
            "not: either PV MMA segment may start\n"
            "first segment: p_o_rescale; second: p_ready_2",
            "不能证明：P 已经写入 TMEM\n"
            "不能证明：O 已经完成重缩放\n"
            "不能证明：任一段 PV MMA 可以开始\n"
            "第一段由 p_o_rescale 放行；第二段由 p_ready_2 放行",
        ),
        fontsize=9.2,
        color="#374151",
        va="top",
    )

    output = "../flash_attention_softmax_correction_zh.svg" if zh else "../flash_attention_softmax_correction.png"
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")


def main(lang="en", font_path=None):
    configure_font(lang, font_path)
    gen_main_handoff(lang)
    gen_softmax_correction(lang)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--font-path")
    args = parser.parse_args()
    main(args.lang, args.font_path)
