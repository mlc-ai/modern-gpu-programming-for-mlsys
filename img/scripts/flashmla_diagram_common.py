"""Shared deterministic drawing helpers for the FlashMLA tutorial figures."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


COLORS = {
    "ink": "#1f2937",
    "muted": "#4b5563",
    "grid": "#d1d5db",
    "line": "#64748b",
    "cta0": "#bfdbfe",
    "cta0_dark": "#2563eb",
    "cta1": "#fbcfe8",
    "cta1_dark": "#db2777",
    "cross01": "#ddd6fe",
    "cross10": "#ccfbf1",
    "gmem": "#dbeafe",
    "tma": "#bfdbfe",
    "smem": "#e9d5ff",
    "tmem": "#fed7aa",
    "mma": "#bbf7d0",
    "softmax": "#ddd6fe",
    "barrier": "#fde68a",
    "projection": "#fecaca",
    "neutral": "#f8fafc",
    "note": "#fefce8",
}


def configure_style(lang: str, font_path: str | None = None) -> None:
    """Configure a fixed backend, font stack, and deterministic SVG IDs."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "flashmla-tutorial-v1",
        }
    )
    if lang != "zh":
        return

    candidates = [
        font_path,
        os.environ.get("FLASHMLA_CJK_FONT"),
        "/tmp/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    selected = next(
        (Path(item) for item in candidates if item and Path(item).is_file()), None
    )
    if selected is None:
        raise FileNotFoundError(
            "Chinese output requires a CJK font. Pass --font-path or set FLASHMLA_CJK_FONT."
        )
    font_manager.fontManager.addfont(selected)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=selected).get_name()
    # Path Chinese glyphs so the checked-in SVG renders without a host CJK font.
    plt.rcParams["svg.fonttype"] = "path"


def tr(lang: str, en: str, zh: str) -> str:
    return zh if lang == "zh" else en


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    color: str,
    *,
    fontsize: float = 9,
    weight: str = "bold",
    edgecolor: str | None = None,
    linewidth: float = 1.2,
    linestyle: str = "-",
    zorder: int = 3,
    text_color: str | None = None,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=linewidth,
        linestyle=linestyle,
        edgecolor=edgecolor or COLORS["ink"],
        facecolor=color,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color=text_color or COLORS["ink"],
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    color: str | None = None,
    linewidth: float = 1.25,
    linestyle: str = "-",
    rad: float = 0.0,
    label_offset: tuple[float, float] = (0.0, 0.12),
    zorder: int = 2,
) -> FancyArrowPatch:
    line_color = color or COLORS["line"]
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=linewidth,
        linestyle=linestyle,
        color=line_color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=7.4,
            color=line_color,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="none",
                alpha=0.92,
            ),
            zorder=zorder + 1,
        )
    return patch


def plain_rect(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    *,
    edgecolor: str | None = None,
    linewidth: float = 1.1,
    hatch: str | None = None,
    zorder: int = 2,
) -> Rectangle:
    patch = Rectangle(
        (x, y),
        w,
        h,
        facecolor=color,
        edgecolor=edgecolor or COLORS["ink"],
        linewidth=linewidth,
        hatch=hatch,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def save_figure(fig, output: str | Path, *, dpi: int = 160) -> Path:
    path = Path(output)
    if path.suffix.lower() not in {".png", ".svg"}:
        raise ValueError("--output must end in .png or .svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".svg":
        metadata = {"Date": None, "Creator": "Modern GPU Programming for MLSys"}
    else:
        metadata = {"Software": "Modern GPU Programming for MLSys"}
    fig.savefig(
        path,
        dpi=dpi,
        facecolor="white",
        transparent=False,
        bbox_inches=None,
        metadata=metadata,
    )
    plt.close(fig)
    if path.suffix.lower() == ".svg":
        # Matplotlib emits trailing spaces in multiline SVG path data. Normalize
        # generated assets so newly staged files pass Git's whitespace checks.
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(
            "\n".join(line.rstrip() for line in lines) + "\n",
            encoding="utf-8",
        )
    return path
