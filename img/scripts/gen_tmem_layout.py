import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle


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


def add_slot(ax, x0, x1, y, label, color, note, height=0.44):
    ax.add_patch(
        Rectangle(
            (x0, y - height / 2),
            x1 - x0,
            height,
            facecolor=color,
            edgecolor="#222222",
            linewidth=1.4,
            zorder=2,
        )
    )
    ax.text((x0 + x1) / 2, y + 0.07, label, ha="center", va="center", fontsize=13, weight="bold", zorder=3)
    ax.text((x0 + x1) / 2, y - 0.12, note, ha="center", va="center", fontsize=8.2, color="#555555", zorder=3)


def main(lang="en", font_path=None):
    configure_font(lang, font_path)
    zh = lang == "zh"
    tr = lambda en, cn: cn if zh else en
    fig, ax = plt.subplots(figsize=(14, 5.2), dpi=180)
    ax.set_xlim(-30, 535)
    ax.set_ylim(-0.45, 3.2)
    ax.axis("off")

    ax.text(256, 2.95, tr("TMEM Layout: 128 rows x 512 columns", "TMEM 布局：128 行 x 512 列"), ha="center", va="center", fontsize=16, weight="bold")

    # Axis.
    y_axis = 2.15
    ax.plot([0, 512], [y_axis, y_axis], color="#222222", linewidth=1.2)
    for x in [0, 64, 128, 192, 256, 384, 512]:
        ax.plot([x, x], [y_axis - 0.08, y_axis + 0.08], color="#222222", linewidth=1.2)
        ax.text(x, y_axis + 0.2, str(x), ha="center", va="bottom", fontsize=9, color="#333333")
    ax.text(512, y_axis - 0.24, tr("TMEM column coordinate", "TMEM 列坐标"), ha="right", va="top", fontsize=9.5, color="#555555")

    # Q-stage labels.
    ax.plot([256, 256], [-0.2, 2.1], color="#777777", linewidth=1.1, linestyle="--", zorder=0)
    ax.text(96, 1.78, tr("Q stage 0 (WG0)", "Q stage 0（WG0）"), color="#3976c6", ha="center", fontsize=12, weight="bold")
    ax.text(352, 1.78, tr("Q stage 1 (WG1)", "Q stage 1（WG1）"), color="#1f9d55", ha="center", fontsize=12, weight="bold")

    # Row labels.
    ax.text(-20, 1.35, "S (fp32)", color="#cc3333", ha="right", va="center", fontsize=11, weight="bold")
    ax.text(-20, 0.63, "P (fp16)", color="#d98200", ha="right", va="center", fontsize=11, weight="bold")
    ax.text(-20, -0.08, "O (fp32)", color="#2f66cc", ha="right", va="center", fontsize=11, weight="bold")

    # Source constants:
    # tmem_s_base = 0, tmem_p_base = 64, tmem_o_base = 256, tmem_offset = 128.
    add_slot(ax, 0, 128, 1.35, "S0", "#f6b9b9", tr("cols 0-127", "列 0-127"))
    add_slot(ax, 128, 256, 1.35, "S1", "#c5f6c7", tr("cols 128-255", "列 128-255"))

    # P is addressed through the fp16 view. The source uses
    # tmem_as_f16[:, tmem_col_p * 2 + ...], so each 128-column fp16
    # tile occupies 64 physical fp32 TMEM columns.
    add_slot(ax, 64, 128, 0.63, "P0", "#ffdca3", tr("phys 64-127\nf16 view 128-255", "物理列 64-127\nf16 视图 128-255"), height=0.58)
    add_slot(ax, 192, 256, 0.63, "P1", "#d7efd9", tr("phys 192-255\nf16 view 384-511", "物理列 192-255\nf16 视图 384-511"), height=0.58)

    add_slot(ax, 256, 384, -0.08, "O0", "#c2d8f7", tr("cols 256-383", "列 256-383"))
    add_slot(ax, 384, 512, -0.08, "O1", "#c9e6cc", tr("cols 384-511", "列 384-511"))

    ax.text(
        256,
        -0.38,
        tr(
            "S and P slots are reused after the matching barrier handoff; O slots hold the running output accumulator.",
            "完成相应的 barrier 交接后，S 和 P 可以复用同一位置；O slots 保存持续更新的输出 accumulator。",
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )

    fig.tight_layout(pad=0.3)
    output = "../tmem_layout_v3_zh.svg" if zh else "../tmem_layout_v3.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--font-path")
    args = parser.parse_args()
    main(args.lang, args.font_path)
