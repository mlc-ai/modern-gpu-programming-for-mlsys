import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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


def main():
    fig, ax = plt.subplots(figsize=(14, 5.2), dpi=180)
    ax.set_xlim(-30, 535)
    ax.set_ylim(-0.45, 3.2)
    ax.axis("off")

    ax.text(256, 2.95, "TMEM Layout: 128 rows x 512 columns", ha="center", va="center", fontsize=16, weight="bold")

    # Axis.
    y_axis = 2.15
    ax.plot([0, 512], [y_axis, y_axis], color="#222222", linewidth=1.2)
    for x in [0, 64, 128, 192, 256, 384, 512]:
        ax.plot([x, x], [y_axis - 0.08, y_axis + 0.08], color="#222222", linewidth=1.2)
        ax.text(x, y_axis + 0.2, str(x), ha="center", va="bottom", fontsize=9, color="#333333")
    ax.text(512, y_axis - 0.24, "TMEM column coordinate", ha="right", va="top", fontsize=9.5, color="#555555")

    # Q-stage labels.
    ax.plot([256, 256], [-0.2, 2.1], color="#777777", linewidth=1.1, linestyle="--", zorder=0)
    ax.text(96, 1.78, "Q stage 0 (WG0)", color="#3976c6", ha="center", fontsize=12, weight="bold")
    ax.text(352, 1.78, "Q stage 1 (WG1)", color="#1f9d55", ha="center", fontsize=12, weight="bold")

    # Row labels.
    ax.text(-20, 1.35, "S (fp32)", color="#cc3333", ha="right", va="center", fontsize=11, weight="bold")
    ax.text(-20, 0.63, "P (fp16)", color="#d98200", ha="right", va="center", fontsize=11, weight="bold")
    ax.text(-20, -0.08, "O (fp32)", color="#2f66cc", ha="right", va="center", fontsize=11, weight="bold")

    # Source constants:
    # tmem_s_base = 0, tmem_p_base = 64, tmem_o_base = 256, tmem_offset = 128.
    add_slot(ax, 0, 128, 1.35, "S0", "#f6b9b9", "cols 0-127")
    add_slot(ax, 128, 256, 1.35, "S1", "#c5f6c7", "cols 128-255")

    # P is addressed through the fp16 view. The source uses
    # tmem_as_f16[:, tmem_col_p * 2 + ...], so each 128-column fp16
    # tile occupies 64 physical fp32 TMEM columns.
    add_slot(ax, 64, 128, 0.63, "P0", "#ffdca3", "phys 64-127\nf16 view 128-255", height=0.58)
    add_slot(ax, 192, 256, 0.63, "P1", "#d7efd9", "phys 192-255\nf16 view 384-511", height=0.58)

    add_slot(ax, 256, 384, -0.08, "O0", "#c2d8f7", "cols 256-383")
    add_slot(ax, 384, 512, -0.08, "O1", "#c9e6cc", "cols 384-511")

    ax.text(
        256,
        -0.38,
        "S and P slots are reused after the matching barrier handoff; O slots hold the running output accumulator.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )

    fig.tight_layout(pad=0.3)
    fig.savefig("../tmem_layout_v3.png", bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
