"""How tcgen05 scale_vec modes select bytes from a 32-bit TMEM cell.

Each TMEM cell has four scale-factor bytes [0:7][8:15][16:23][24:31].
  1X selects one byte using byte-id 0, 1, 2, or 3.
  2X selects the low or high byte pair using byte-id 0 or 2.
  4X selects all four bytes and requires byte-id 0.
Outputs SVG into ../ (the img/ dir).
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pathlib import Path; OUT = str(Path(__file__).resolve().parent.parent)  # the repo img/ dir
TXT = "#1f2937"
SFC = {"SF0": "#ef4444", "SF1": "#3b82f6", "SF2": "#10b981", "SF3": "#f59e0b"}

fig, ax = plt.subplots(figsize=(9.6, 4.6))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.text(50, 97, "scale_vec modes: bytes selected from one 32-bit TMEM cell", ha="center", va="top",
        fontsize=12, fontweight="bold", color=TXT)

SX, SW = 30, 44          # strip x-origin, width
bw = SW / 4
BYTES = ["[0:7]", "[8:15]", "[16:23]", "[24:31]"]

# byte-position header
for b in range(4):
    ax.text(SX + (b + 0.5) * bw, 86, BYTES[b], ha="center", va="center", fontsize=7.5, color=TXT)
ax.text(SX + SW / 2, 90, "bytes of the 32-bit word", ha="center", fontsize=8, color=TXT)

ROWS = [
    ("1X", "fp8 / mxfp8", 1, "select 1 byte   (byte-id = 0, 1, 2, or 3)"),
    ("2X", "mxfp4", 2, "select low or high pair   (byte-id = 0 or 2)"),
    ("4X", "nvfp4", 4, "select all 4 bytes   (byte-id = 0)"),
]
ys = [66, 46, 26]
sh = 13
for (mode, fmt, selected, cap), y in zip(ROWS, ys):
    ax.text(SX - 3, y + sh / 2, f"{mode}", ha="right", va="center", fontsize=11, fontweight="bold", color=TXT)
    ax.text(SX - 3, y - 2.5, fmt, ha="right", va="center", fontsize=7.5, color=TXT)
    for b, lab in enumerate(("SF0", "SF1", "SF2", "SF3")):
        active = b < selected
        ax.add_patch(Rectangle((SX + b * bw, y), bw, sh,
                               facecolor=SFC[lab] if active else "#e5e7eb",
                               edgecolor="white", linewidth=1.6, alpha=0.92))
        ax.text(SX + (b + 0.5) * bw, y + sh / 2, lab, ha="center", va="center",
                color="white" if active else "#6b7280",
                fontsize=9, fontweight="bold")
    ax.text(SX + SW + 3, y + sh / 2, cap, ha="left", va="center", fontsize=8, color=TXT)

ax.text(50, 9, "Highlighted bytes show the byte-id = 0 selection. 1X and 2X can select other positions.",
        ha="center", fontsize=7.6, color=TXT, style="italic")
ax.text(50, 4, "The modes consume 1 (fp8), 2 (mxfp4), or 4 (nvfp4) K-block scales per MMA.",
        ha="center", fontsize=7.6, color=TXT, style="italic")

fig.savefig(f"{OUT}/sf_scale_vec.svg", facecolor="white", bbox_inches="tight")
plt.close(fig)
print("wrote sf_scale_vec.svg")
