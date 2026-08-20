"""Render the measured Nsight Systems timeline used by the benchmarking appendix."""

from html import escape
from pathlib import Path


WIDTH = 1500
HEIGHT = 500
LEFT = 235
RIGHT = 1440
T_MAX = 400.0
BAR_HEIGHT = 34


def x_pos(time_us: float) -> float:
    return LEFT + (RIGHT - LEFT) * time_us / T_MAX


def render(*, chinese: bool, output: Path) -> None:
    title = (
        "B200 上的一次真实 Nsight Systems 采集"
        if chinese
        else "One Real Nsight Systems Capture on B200"
    )
    subtitle = (
        "4096×4096 BF16：GEMM → ReLU"
        if chinese
        else "4096×4096 BF16: GEMM → ReLU"
    )
    rows = ["Outer NVTX", "Child NVTX ranges", "CUDA APIs", "GPU stream 7"]
    note = (
        "时间以外层 NVTX range 的起点为 0；横向长度按真实采集比例绘制。"
        if chinese
        else "Time is relative to the outer NVTX-range start; horizontal lengths come from the measured capture."
    )

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#ffffff" rx="14"/>',
        '<style>text { font-family: Inter, Arial, "Noto Sans CJK SC", sans-serif; }</style>',
        f'<text x="{LEFT}" y="35" font-size="24" font-weight="700">{escape(title)}</text>',
        f'<text x="{LEFT}" y="61" font-size="15" fill="#58677c">{escape(subtitle)}</text>',
    ]

    axis_y = 88
    parts.append(f'<line x1="{LEFT}" y1="{axis_y}" x2="{RIGHT}" y2="{axis_y}" stroke="#9aa7b8" stroke-width="1"/>')
    for tick in range(0, 401, 50):
        x = x_pos(float(tick))
        parts.append(f'<line x1="{x:.2f}" y1="{axis_y - 5}" x2="{x:.2f}" y2="430" stroke="#e5e9ef" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="80" font-size="12" text-anchor="middle" fill="#64748b">{tick} μs</text>')

    row_y = [125, 210, 295, 380]
    for label, y in zip(rows, row_y):
        parts.append(f'<text x="{LEFT - 18}" y="{y + 22}" font-size="15" font-weight="600" text-anchor="end">{escape(label)}</text>')
        parts.append(f'<line x1="{LEFT}" y1="{y + BAR_HEIGHT + 11}" x2="{RIGHT}" y2="{y + BAR_HEIGHT + 11}" stroke="#eef1f5"/>')

    def rect(start: float, end: float, y: float, color: str, label: str = "", text_color: str = "#ffffff") -> None:
        x = x_pos(start)
        width = x_pos(end) - x
        parts.append(
            f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="{BAR_HEIGHT}" rx="6" '
            f'fill="{color}" stroke="#243247" stroke-width="0.8"/>'
        )
        if label:
            parts.append(
                f'<text x="{x + width / 2:.2f}" y="{y + 22}" font-size="13" font-weight="600" '
                f'text-anchor="middle" style="fill:{text_color}">{escape(label)}</text>'
            )

    # Outer and child NVTX ranges, relative to target-operation start.
    rect(0.0, 375.992, row_y[0], "#365f9d", "target operation · host NVTX 376.0 μs")
    rect(5.927, 201.012, row_y[1], "#598bc2", "BF16 GEMM")
    rect(213.209, 272.981, row_y[1], "#86b6df", "ReLU", "#243247")

    # Host CUDA API intervals.
    rect(135.682, 186.399, row_y[2], "#d7832f")
    rect(255.070, 268.544, row_y[2], "#c76d25")
    rect(352.196, 372.905, row_y[2], "#b85f46")
    for time_us, label, anchor in [
        (161.041, "GEMM launch · 50.7 μs", "middle"),
        (261.807, "ReLU launch · 13.5 μs", "middle"),
        (362.551, "sync · 20.7 μs", "end"),
    ]:
        x = x_pos(time_us)
        dx = 12 if anchor == "start" else (-12 if anchor == "end" else 0)
        parts.append(f'<line x1="{x:.2f}" y1="{row_y[2]}" x2="{x + dx:.2f}" y2="{row_y[2] - 16}" stroke="#8a4c23"/>')
        parts.append(
            f'<text x="{x + dx:.2f}" y="{row_y[2] - 21}" font-size="12" text-anchor="{anchor}" fill="#7a4222">{escape(label)}</text>'
        )

    # GPU activity on the default stream.
    rect(180.786, 273.394, row_y[3], "#6f55b5", "GEMM · 92.6 μs")
    rect(273.618, 284.562, row_y[3], "#a574d1")
    relu_x = x_pos(279.090)
    parts.append(f'<line x1="{relu_x:.2f}" y1="{row_y[3]}" x2="{relu_x - 15:.2f}" y2="{row_y[3] - 18}" stroke="#6f4b8d"/>')
    parts.append(f'<text x="{relu_x - 18:.2f}" y="{row_y[3] - 23}" font-size="12" text-anchor="end" fill="#6f4b8d">ReLU · 10.9 μs</text>')

    parts.append(f'<text x="{LEFT}" y="475" font-size="13" fill="#58677c">{escape(note)}</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    image_dir = Path(__file__).resolve().parents[1]
    render(chinese=False, output=image_dir / "nsys_b200_timeline.svg")
    render(chinese=True, output=image_dir / "nsys_b200_timeline_zh_en_tracks.svg")


if __name__ == "__main__":
    main()
