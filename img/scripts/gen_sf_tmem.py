"""Block-scaled MMA scale factors (SFA, SFB) in TMEM. Ground truth: nymph-rust +
tvm sf_tmem_layout (backend/cuda/.../gemm_async/tcgen05.py): rows must be a
multiple of 32; M = rows // 32; epc = 4 (four 8-bit SFs per 32-bit TMEM column);
the atom is one 32-row chunk with R[4 : 32@TLane] (a warpx4 broadcast).

Net: 128 M-rows occupy only 32 TMEM lanes. TLane = m % 32; the m // 32 group runs
along TCol (stride epc); each u32 column packs the per-MMA K-blocks. The warpx4
broadcast feeds the 32 stored rows to all 128 warpgroup lanes. Outputs SVG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = "/home/bohanhou/worktree/tirx-tutorial/tirx-tutorial/img"
TXT = "#1f2937"
PURPLE = ["#7c3aed", "#8b5cf6", "#a78bfa", "#c4b5fd"]

fig, ax = plt.subplots(figsize=(9.8, 5.8))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.text(50, 98.5, "Scale factors in TMEM:  128 M-rows → 32 TMEM rows", ha="center", va="top",
        fontsize=12.5, fontweight="bold", color=TXT)

X0 = 26
CW = 17
YT = 84
RH = 8.5
# columns = the m // 32 row-group (for M = 128 there are 4); within a column the
# u32 packs the per-MMA K-blocks.
GROUPS = ["m//32 = 0\n(rows 0–31)", "= 1\n(rows 32–63)", "= 2\n(rows 64–95)", "= 3\n(rows 96–127)"]
lanes = [(0, "TLane 0"), (1, "TLane 1"), (2, "TLane 2"), (3, "TLane 3"), (None, "⋮"), (31, "TLane 31")]

for gi, glab in enumerate(GROUPS):
    cx = X0 + gi * CW
    ax.text(cx + CW / 2, YT + 2.5, glab, ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=TXT)
    for ri, (lane, _) in enumerate(lanes):
        y = YT - (ri + 1) * RH
        if lane is None:
            ax.text(cx + CW / 2, y + RH / 2, "⋮", ha="center", va="center", fontsize=12, color=TXT)
            continue
        ax.add_patch(Rectangle((cx, y), CW, RH, facecolor=PURPLE[gi], edgecolor="white",
                               linewidth=1.5, alpha=0.92))
        ax.text(cx + CW / 2, y + RH / 2, f"row {gi * 32 + lane}", ha="center", va="center",
                color="white", fontsize=7.5, fontweight="bold")

# lane labels (left) — physical TMEM row
for ri, (lane, llab) in enumerate(lanes):
    y = YT - (ri + 1) * RH
    ax.text(X0 - 2, y + RH / 2, llab, ha="right", va="center", fontsize=8, fontweight="bold", color=TXT)

# axes
gx1 = X0 + 4 * CW
ax.annotate("", xy=(X0 - 11, YT - 6 * RH), xytext=(X0 - 11, YT), arrowprops=dict(arrowstyle="-|>", color=TXT, lw=1.2))
ax.text(X0 - 13, YT - 3 * RH, "TLane ↓\n(only 32)", ha="right", va="center", fontsize=8, color=TXT)
ax.text((X0 + gx1) / 2, YT + 7.5, "TCol →  (m // 32 group, then K)", ha="center", fontsize=8.5, color=TXT)

# byte detail: one M-row's u32 along K (scale_vec mode), per the PTX SF-A diagrams
BYTE = ["#a78bfa", "#c4b5fd", "#8b5cf6", "#6d28d9"]
sx, sw, syy, sh = 35, 23, 23, 6.5
ax.text(sx - 1.5, syy + sh / 2, "row 0 u32:", ha="right", va="center", fontsize=8, fontweight="bold", color=TXT)
for b, lab in enumerate(["SF0", "SF1", "SF2", "SF3"]):
    bw = sw / 4
    ax.add_patch(Rectangle((sx + b * bw, syy), bw, sh, facecolor=BYTE[b], edgecolor="white", linewidth=1.2))
    ax.text(sx + (b + 0.5) * bw, syy + sh / 2, lab, ha="center", va="center", color="white",
            fontsize=7.5, fontweight="bold")
ax.text(sx + sw + 2, syy + sh / 2, "scale_vec::4X (nvfp4).  1X (fp8): SF0 ×4.  2X: SF0,1 ×2.",
        ha="left", va="center", fontsize=7, color=TXT)

# notes
ax.text(50, 13, "TLane = m % 32 → all 128 M-rows live in 32 TMEM lanes; the m // 32 group runs along "
        "TCol, then K.", ha="center", fontsize=8.2, color=TXT, style="italic")
ax.text(50, 7.5, "Loaded SMEM→TMEM via `tcgen05.cp`; the 32 stored rows feed all 128 warpgroup lanes "
        "(`R[4:32@TLane]` warpx4 broadcast).", ha="center", fontsize=8.2, color=TXT, style="italic")

fig.savefig(f"{OUT}/sf_tmem.svg", facecolor="white", bbox_inches="tight")
plt.close(fig)
print("wrote sf_tmem.svg")
