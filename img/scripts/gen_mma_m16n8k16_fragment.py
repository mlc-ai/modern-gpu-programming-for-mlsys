"""Generate the mma.m16n8k16 C/D register-fragment teaching diagram."""

from html import escape
from pathlib import Path


OUT = Path(__file__).resolve().parent.parent
WIDTH = 1320
HEIGHT = 760

GROUP_COLORS = [
    "#dbeafe",
    "#dcfce7",
    "#fef3c7",
    "#fce7f3",
    "#ede9fe",
    "#cffafe",
    "#ffedd5",
    "#e2e8f0",
]
GROUP_STROKES = [
    "#3b82f6",
    "#22c55e",
    "#eab308",
    "#ec4899",
    "#8b5cf6",
    "#06b6d4",
    "#f97316",
    "#64748b",
]


TEXT = {
    "zh": {
        "title": "mma.sync.m16n8k16 的 C/D Register Fragment",
        "subtitle": "32 个 lanes 如何共同持有一个 16 x 8 fp32 accumulator tile",
        "matrix": "逻辑 C/D tile（cell 中为持有该元素的 lane）",
        "upper": "rows 0-7",
        "lower": "rows 8-15",
        "step1": "第一步：group 选择两条 rows",
        "step1_formula": "g = lane // 4",
        "step1_example": "lane 5: g = 1 -> rows 1 和 9",
        "step2": "第二步：组内位置选择一对 columns",
        "step2_formula": "t = lane mod 4",
        "step2_example": "lane 5: t = 1 -> columns 2 和 3",
        "result": "lane 5 持有四个 fp32 累加值",
        "legend": "相同底色表示同一个 4-lane group；黑框标出 lane 5 持有的四个元素。",
        "regs": "每个 cell 对应一个 fp32 register value",
    },
    "en": {
        "title": "C/D Register Fragment for mma.sync.m16n8k16",
        "subtitle": "How 32 lanes collectively hold a 16 x 8 fp32 accumulator tile",
        "matrix": "Logical C/D tile (each cell names its owning lane)",
        "upper": "rows 0-7",
        "lower": "rows 8-15",
        "step1": "Step 1: the group selects two rows",
        "step1_formula": "g = lane // 4",
        "step1_example": "lane 5: g = 1 -> rows 1 and 9",
        "step2": "Step 2: position selects two columns",
        "step2_formula": "t = lane mod 4",
        "step2_example": "lane 5: t = 1 -> columns 2 and 3",
        "result": "lane 5 holds four fp32 accumulator values",
        "legend": "Matching fills identify one four-lane group; black borders mark lane 5's four elements.",
        "regs": "Each cell corresponds to one fp32 register value",
    },
}


def text(x, y, value, size=20, weight=400, anchor="start", fill="#1f2937", family="sans"):
    families = {
        "sans": "Arial, 'Noto Sans CJK SC', sans-serif",
        "mono": "'DejaVu Sans Mono', 'Noto Sans Mono CJK SC', monospace",
    }
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="{families[family]}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escape(value)}</text>'
    )


def rect(x, y, w, h, fill, stroke="#cbd5e1", sw=1.5, radius=0):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def line(x1, y1, x2, y2, stroke="#64748b", sw=2, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
    )


def generate(lang):
    tr = TEXT[lang]
    out = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{escape(tr["title"])}">'
    )
    out.append(rect(0, 0, WIDTH, HEIGHT, "#ffffff", "none", 0))
    out.append(text(WIDTH / 2, 48, tr["title"], 30, 700, "middle", "#111827"))
    out.append(text(WIDTH / 2, 80, tr["subtitle"], 18, 400, "middle", "#64748b"))

    # Matrix panel.
    panel_x, panel_y, panel_w, panel_h = 38, 108, 760, 570
    out.append(rect(panel_x, panel_y, panel_w, panel_h, "#f8fafc", "#cbd5e1", 2, 8))
    out.append(text(panel_x + panel_w / 2, 140, tr["matrix"], 19, 700, "middle"))

    grid_x, grid_y = 186, 190
    cell_w, cell_h = 64, 27
    half_gap = 16

    for c in range(8):
        out.append(text(grid_x + c * cell_w + cell_w / 2, grid_y - 12, str(c), 15, 600, "middle", "#64748b"))
    out.append(text(grid_x + 4 * cell_w, grid_y - 30, "column n", 15, 600, "middle", "#475569"))

    for r in range(16):
        y = grid_y + r * cell_h + (half_gap if r >= 8 else 0)
        g = r % 8
        out.append(text(grid_x - 18, y + 19, str(r), 14, 600, "end", "#64748b"))
        for c in range(8):
            lane = 4 * g + c // 2
            x = grid_x + c * cell_w
            highlight = lane == 5
            out.append(
                rect(
                    x,
                    y,
                    cell_w - 2,
                    cell_h - 2,
                    GROUP_COLORS[g],
                    "#111827" if highlight else GROUP_STROKES[g],
                    3.5 if highlight else 1.1,
                    2,
                )
            )
            out.append(text(x + (cell_w - 2) / 2, y + 19, f"L{lane}", 13, 700, "middle", "#1f2937", "mono"))

    # Row-axis labels and split.
    out.append(text(grid_x - 94, grid_y + 4 * cell_h, tr["upper"], 15, 600, "middle", "#475569"))
    out.append(text(grid_x - 94, grid_y + 12 * cell_h + half_gap, tr["lower"], 15, 600, "middle", "#475569"))
    out.append(line(grid_x - 48, grid_y, grid_x - 48, grid_y + 8 * cell_h - 2, "#94a3b8", 2))
    out.append(line(grid_x - 48, grid_y + 8 * cell_h + half_gap, grid_x - 48, grid_y + 16 * cell_h + half_gap - 2, "#94a3b8", 2))
    out.append(line(grid_x, grid_y + 8 * cell_h + half_gap / 2, grid_x + 8 * cell_w - 2, grid_y + 8 * cell_h + half_gap / 2, "#475569", 2, "5 5"))

    # Explanation panel.
    ex_x, ex_y, ex_w, ex_h = 830, 108, 452, 570
    out.append(rect(ex_x, ex_y, ex_w, ex_h, "#ffffff", "#cbd5e1", 2, 8))

    out.append(text(ex_x + 28, 155, tr["step1"], 20, 700))
    out.append(rect(ex_x + 28, 172, ex_w - 56, 48, "#eff6ff", "#93c5fd", 1.5, 5))
    out.append(text(ex_x + ex_w / 2, 204, tr["step1_formula"], 23, 700, "middle", "#1d4ed8", "mono"))
    out.append(text(ex_x + 28, 247, tr["step1_example"], 17, 500, "start", "#334155"))

    out.append(text(ex_x + 28, 302, tr["step2"], 20, 700))
    out.append(rect(ex_x + 28, 319, ex_w - 56, 48, "#f0fdf4", "#86efac", 1.5, 5))
    out.append(text(ex_x + ex_w / 2, 351, tr["step2_formula"], 23, 700, "middle", "#15803d", "mono"))
    out.append(text(ex_x + 28, 394, tr["step2_example"], 17, 500, "start", "#334155"))

    out.append(line(ex_x + 28, 430, ex_x + ex_w - 28, 430, "#e2e8f0", 2))
    out.append(text(ex_x + 28, 466, tr["result"], 20, 700))

    coords = [("c0", "(1, 2)"), ("c1", "(1, 3)"), ("c2", "(9, 2)"), ("c3", "(9, 3)")]
    for i, (reg, coord) in enumerate(coords):
        x = ex_x + 28 + (i % 2) * 198
        y = 486 + (i // 2) * 66
        out.append(rect(x, y, 180, 50, "#f8fafc", "#111827", 1.8, 5))
        out.append(text(x + 18, y + 31, reg, 16, 700, "start", "#475569", "mono"))
        out.append(text(x + 158, y + 31, coord, 17, 700, "end", "#111827", "mono"))
    out.append(text(ex_x + ex_w / 2, 633, tr["regs"], 15, 500, "middle", "#64748b"))

    out.append(text(WIDTH / 2, 720, tr["legend"], 16, 500, "middle", "#475569"))
    out.append("</svg>")

    suffix = "" if lang == "zh" else "_en"
    path = OUT / f"mma_m16n8k16_fragment{suffix}.svg"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {path}")


for language in ("zh", "en"):
    generate(language)
