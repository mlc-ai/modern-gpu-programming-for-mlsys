#!/usr/bin/env python3
"""Generate the introductory MHA-to-MLA cache story for the FlashMLA tutorial."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from flashmla_diagram_common import (
    COLORS,
    arrow,
    configure_style,
    plain_rect,
    rounded_box,
    save_figure,
    tr,
)


def _panel(ax, x, y, w, h, title, subtitle, color) -> None:
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        linewidth=1.35,
        edgecolor=color,
        facecolor="#ffffff",
        linestyle="--",
        zorder=0,
    )
    ax.add_patch(panel)
    ax.text(
        x + 0.25,
        y + h - 0.34,
        title,
        ha="left",
        va="center",
        fontsize=11.5,
        weight="bold",
        color=color,
    )
    ax.text(
        x + 0.25,
        y + h - 0.72,
        subtitle,
        ha="left",
        va="center",
        fontsize=7.8,
        color=COLORS["muted"],
    )


def draw(lang: str, output: str, font_path: str | None = None) -> None:
    configure_style(lang, font_path)
    fig, ax = plt.subplots(figsize=(18.0, 9.0))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        9,
        8.68,
        tr(
            lang,
            "How Can One MLA Cache Entry Serve 128 Query Heads?",
            "一条 MLA Cache Entry 如何服务 128 个 Query Heads？",
        ),
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
    )
    ax.text(
        9,
        8.28,
        tr(
            lang,
            "store one shared compressed state; keep head-specific work on the query and output sides",
            "只保存一份共享压缩状态；各 head 的差异留在 query 与 output 两侧",
        ),
        ha="center",
        va="center",
        fontsize=9.3,
        color=COLORS["muted"],
    )

    _panel(
        ax,
        0.35,
        1.30,
        4.45,
        6.45,
        tr(lang, "1 · Ordinary MHA cache", "1 · 普通 MHA cache"),
        tr(
            lang,
            "store a key and a value for every head",
            "为每个 head 保存一份 key 和 value",
        ),
        COLORS["cta0_dark"],
    )
    _panel(
        ax,
        5.15,
        1.30,
        5.05,
        6.45,
        tr(lang, "2 · MLA shared cache", "2 · MLA 共享 cache"),
        tr(
            lang,
            "compress once and store one shared state",
            "压缩一次，只保存一份共享状态",
        ),
        "#7c3aed",
    )
    _panel(
        ax,
        10.55,
        1.30,
        7.10,
        6.45,
        tr(lang, "3 · Use the shared cache", "3 · 使用共享 cache"),
        tr(
            lang,
            "head-specific work happens before and after attention",
            "各 head 的特定计算放在 attention 前后",
        ),
        COLORS["cta1_dark"],
    )

    # MHA: a visual stack of materialized per-head cache entries.
    ax.text(
        2.58,
        6.53,
        tr(lang, "one cached token", "一个 cached token"),
        ha="center",
        fontsize=9.5,
        weight="bold",
    )
    row_entries = (
        (
            "head 0",
            tr(lang, "cached\nkey", "缓存\nkey"),
            tr(lang, "cached\nvalue", "缓存\nvalue"),
        ),
        (
            "head 1",
            tr(lang, "cached\nkey", "缓存\nkey"),
            tr(lang, "cached\nvalue", "缓存\nvalue"),
        ),
        ("...", "...", "..."),
        (
            "head 127",
            tr(lang, "cached\nkey", "缓存\nkey"),
            tr(lang, "cached\nvalue", "缓存\nvalue"),
        ),
    )
    for idx, (label, k_label, v_label) in enumerate(row_entries):
        y = 5.70 - idx * 0.83
        ax.text(0.72, y + 0.29, label, ha="left", va="center", fontsize=7.7)
        plain_rect(ax, 1.55, y, 1.18, 0.58, COLORS["gmem"])
        plain_rect(ax, 2.73, y, 1.18, 0.58, "#bfdbfe")
        ax.text(
            2.14,
            y + 0.29,
            k_label,
            ha="center",
            va="center",
            fontsize=7.4,
            weight="bold",
        )
        ax.text(
            3.32,
            y + 0.29,
            v_label,
            ha="center",
            va="center",
            fontsize=7.4,
            weight="bold",
        )
    rounded_box(
        ax,
        0.82,
        1.72,
        3.50,
        0.88,
        tr(
            lang,
            "cache stores a separate key/value slice\nfor every head",
            "cache 为每个 head 保存\n独立的 key/value slice",
        ),
        COLORS["note"],
        fontsize=8.5,
        weight="normal",
        edgecolor="#ca8a04",
    )

    # MLA: compress the token's content once and store it with one shared
    # positional channel.  Matrix names are intentionally deferred to the
    # derivation that follows this introductory figure.
    rounded_box(
        ax,
        5.45,
        5.79,
        1.45,
        0.80,
        tr(lang, "token\nrepresentation", "token\n表示"),
        COLORS["neutral"],
        fontsize=8.1,
    )
    rounded_box(
        ax,
        7.12,
        5.79,
        1.42,
        0.80,
        tr(lang, "compress\ncontent once", "只压缩一次\ncontent"),
        COLORS["projection"],
        fontsize=8.0,
    )
    rounded_box(
        ax,
        8.78,
        5.69,
        1.05,
        1.00,
        tr(lang, "shared\ncontent state\n512 numbers", "共享\ncontent 状态\n512 个数"),
        COLORS["smem"],
        fontsize=7.4,
    )
    arrow(ax, (6.90, 6.19), (7.12, 6.19))
    arrow(ax, (8.54, 6.19), (8.78, 6.19))
    rounded_box(
        ax,
        5.55,
        4.47,
        2.05,
        0.78,
        tr(lang, "position information", "位置信息"),
        "#bfdbfe",
        fontsize=8.4,
    )
    rounded_box(
        ax,
        8.02,
        4.47,
        1.81,
        0.78,
        tr(lang, "shared position part\n64 numbers", "共享位置信息\n64 个数"),
        "#bfdbfe",
        fontsize=7.8,
    )
    arrow(ax, (7.60, 4.86), (8.02, 4.86))

    ax.text(
        7.68,
        3.94,
        tr(lang, "one cache entry per token", "每个 token 只有一条 cache entry"),
        ha="center",
        va="center",
        fontsize=9.2,
        weight="bold",
        color="#6d28d9",
    )
    plain_rect(ax, 5.82, 2.90, 2.76, 0.76, COLORS["gmem"])
    plain_rect(ax, 8.58, 2.90, 1.20, 0.76, "#bfdbfe")
    ax.text(
        7.20,
        3.28,
        tr(lang, "shared compressed\ncontent · 512", "共享压缩 content\n· 512"),
        ha="center",
        va="center",
        fontsize=7.8,
        weight="bold",
    )
    ax.text(
        9.18,
        3.28,
        tr(lang, "position\npart · 64", "位置信息\n· 64"),
        ha="center",
        va="center",
        fontsize=7.5,
        weight="bold",
    )
    rounded_box(
        ax,
        5.82,
        1.72,
        3.96,
        0.72,
        tr(
            lang,
            "one shared entry · not 128 key/value pairs",
            "一份共享 cache entry，而不是 128 组 key/value",
        ),
        COLORS["note"],
        fontsize=8.2,
        weight="normal",
        edgecolor="#ca8a04",
    )
    arrow(ax, (9.78, 3.28), (11.15, 4.09), color="#7c3aed", linewidth=1.6, rad=-0.05)

    # Head-specific query/output work surrounds attention over the shared cache.
    rounded_box(
        ax,
        10.92,
        5.89,
        1.46,
        0.70,
        tr(
            lang,
            "content part of query\nfor one head",
            "某个 head query 的\ncontent 部分",
        ),
        COLORS["neutral"],
        fontsize=7.8,
    )
    rounded_box(
        ax,
        12.78,
        5.89,
        1.66,
        0.70,
        tr(lang, "head-specific\nquery transform", "该 head 的\nquery 变换"),
        COLORS["projection"],
        fontsize=7.7,
    )
    rounded_box(
        ax,
        10.92,
        4.93,
        1.46,
        0.70,
        tr(lang, "position part of query\nfor that head", "该 head query 的\n位置信息"),
        "#bfdbfe",
        fontsize=7.7,
    )
    rounded_box(
        ax,
        14.88,
        5.34,
        2.28,
        1.02,
        tr(
            lang,
            "query used by attention\ncontent view + position",
            "attention 使用的 query\ncontent 视图 + 位置信息",
        ),
        COLORS["tmem"],
        fontsize=7.8,
    )
    arrow(ax, (12.38, 6.24), (12.78, 6.24))
    arrow(ax, (14.44, 6.24), (14.88, 6.03))
    arrow(
        ax,
        (12.38, 5.28),
        (14.88, 5.62),
        label=tr(lang, "position stays separate", "位置信息保持独立"),
        color=COLORS["cta0_dark"],
        rad=-0.05,
        label_offset=(0.0, -0.14),
    )

    rounded_box(
        ax,
        11.15,
        3.63,
        2.56,
        0.92,
        tr(
            lang,
            "one shared cache entry\nkey side: content + position\nvalue side: content",
            "一条共享 cache entry\nkey 侧：压缩 content + 位置信息\nvalue 侧：压缩 content",
        ),
        COLORS["gmem"],
        fontsize=7.6,
    )
    rounded_box(
        ax,
        14.15,
        3.48,
        2.12,
        1.10,
        tr(lang, "attention over\nthe shared cache", "在共享 cache 上\n计算 attention"),
        COLORS["mma"],
        fontsize=8.7,
    )
    arrow(ax, (16.02, 5.34), (15.56, 4.58), color=COLORS["line"], rad=0.04)
    arrow(ax, (13.71, 4.09), (14.15, 4.03))
    rounded_box(
        ax,
        13.22,
        2.22,
        1.60,
        0.72,
        tr(lang, "shared-space result\n512 numbers", "共享空间结果\n512 个数"),
        COLORS["tmem"],
        fontsize=7.6,
    )
    rounded_box(
        ax,
        15.22,
        2.22,
        1.90,
        0.72,
        tr(
            lang,
            "head-specific\noutput transform",
            "该 head 的\noutput 变换",
        ),
        COLORS["projection"],
        fontsize=7.6,
    )
    arrow(ax, (15.05, 3.48), (14.10, 2.94), color=COLORS["line"], rad=0.04)
    arrow(ax, (14.82, 2.58), (15.22, 2.58))
    ax.text(
        15.18,
        1.74,
        tr(
            lang,
            "repeat the query/output work for all 128 heads\n→ 128 distinct head outputs",
            "对全部 128 个 heads 重复 query/output 处理\n→ 128 份不同的 head output",
        ),
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
        weight="bold",
    )

    rounded_box(
        ax,
        1.65,
        0.30,
        14.70,
        0.66,
        tr(
            lang,
            "MLA caches one shared compressed state per token. Head-specific transformations happen around attention, so the cache does not store 128 key/value pairs.",
            "MLA 为每个 token 只缓存一份共享压缩状态。各 head 的特定变换放在 attention 两侧，因此 cache 无需保存 128 组 key/value。",
        ),
        COLORS["note"],
        fontsize=8.4,
        weight="normal",
        edgecolor="#ca8a04",
    )

    save_figure(fig, output, dpi=160)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--output", required=True)
    parser.add_argument("--font-path")
    args = parser.parse_args()
    draw(args.lang, args.output, args.font_path)
