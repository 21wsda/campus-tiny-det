"""
结构图：骨干 → 脖子（红圈接入 P2）→ 四层检测头。
用法：
  python scripts/draw_neck_p2.py
  python scripts/draw_neck_p2.py --out docs/figures/neck_p2.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Ellipse


OUT_DEFAULT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "neck_p2.png"

INK = "#1F2933"
MUTED = "#6B7280"
NAVY = "#2C4A6E"
GOLD = "#8A5A22"
GREEN = "#2F6B4F"
RED = "#C0392B"
LINE = "#4B5563"
PAPER = "#FAFBFC"
BB_BG = "#E8EEF6"
NK_BG = "#FFF4E5"
HD_BG = "#E7F3EA"
P2_FILL = "#FDECEC"


def setup_font():
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def rounded(ax, x, y, w, h, fc, ec, lw=1.4, z=3, radius=0.08):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
        mutation_aspect=0.6,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=LINE, lw=1.7, style="-", rad=0.0, z=4):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            linestyle=style,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=z,
            shrinkA=0,
            shrinkB=0,
        )
    )


def text(ax, x, y, s, size=10, color=INK, weight="normal", ha="center", va="center", z=6):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z)


def draw(out: Path):
    setup_font()
    fig, ax = plt.subplots(figsize=(15.6, 8.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 8.4)
    ax.set_axis_off()
    ax.set_facecolor("white")

    # title
    text(ax, 7.8, 8.08, "YOLOv8n-P2 结构示意（只改颈部）", size=16, weight="bold")
    text(
        ax,
        7.8,
        7.68,
        "骨干不换  ·  红圈：把步长 4 的 P2 接到脖子  ·  检测头由 3 层改为 4 层",
        size=10,
        color=MUTED,
    )

    # three stage panels（顶栏留白，避免和红圈抢位置）
    stages = [
        (0.22, 1.22, 4.28, 6.18, BB_BG, NAVY, "①  骨干 Backbone", "不换，照常抽出 P2～P5"),
        (4.64, 1.22, 6.12, 6.18, NK_BG, GOLD, "②  颈部 Neck", "只改这里：接入 P2，融合先 Concat"),
        (10.90, 1.22, 4.46, 6.18, HD_BG, GREEN, "③  检测头 Head", "跟着多看一层，不做新检测器"),
    ]
    for x, y, w, h, fc, ec, title, sub in stages:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.2, alpha=0.55, zorder=0))
        text(ax, x + 0.18, y + h - 0.26, title, size=12, weight="bold", color=ec, ha="left")
        text(ax, x + 0.18, y + h - 0.54, sub, size=8.5, color=MUTED, ha="left")

    # feature-map geometry (bottom-left of each box)
    levels = [
        {"name": "P2", "stride": "步长 4", "hw": "160×160", "cells": "约 5 格", "y": 5.28, "w": 1.70, "h": 0.92},
        {"name": "P3", "stride": "步长 8", "hw": "80×80", "cells": "约 1～2 格", "y": 3.98, "w": 1.48, "h": 0.82},
        {"name": "P4", "stride": "步长 16", "hw": "40×40", "cells": "不到 1 格", "y": 2.80, "w": 1.28, "h": 0.74},
        {"name": "P5", "stride": "步长 32", "hw": "20×20", "cells": "几乎没有", "y": 1.68, "w": 1.10, "h": 0.68},
    ]

    bb_x = 0.55
    nk_x = 6.55
    nk_w = 2.55
    hd_x = 11.35
    hd_w = 3.55

    # backbone maps
    for i, lv in enumerate(levels):
        is_p2 = lv["name"] == "P2"
        w, h, y = lv["w"], lv["h"], lv["y"]
        x = bb_x + (1.70 - w) / 2
        rounded(
            ax,
            x,
            y,
            w,
            h,
            P2_FILL if is_p2 else "white",
            RED if is_p2 else NAVY,
            lw=2.4 if is_p2 else 1.3,
            z=3,
        )
        text(ax, x + w / 2, y + h * 0.66, f"{lv['name']}  / {lv['stride']}", size=10, weight="bold", color=RED if is_p2 else NAVY)
        text(ax, x + w / 2, y + h * 0.28, lv["hw"], size=8, color=MUTED)
        lv["bb"] = (x, y, w, h)

        # downsample within backbone
        if i < 3:
            nxt = levels[i + 1]
            x2 = bb_x + (1.70 - nxt["w"]) / 2
            arrow(ax, x + w / 2, y, x2 + nxt["w"] / 2, nxt["y"] + nxt["h"], color=NAVY, lw=1.2)

    rounded(ax, 0.55, 6.42, 1.70, 0.42, "white", NAVY, lw=1.1, radius=0.06)
    text(ax, 1.40, 6.63, "输入 640×640", size=8.5, color=NAVY)
    arrow(ax, 1.40, 6.42, 1.40, 6.20, color=NAVY, lw=1.2)
    text(ax, 2.95, 1.44, "P2 算出来了，官方后面丢掉", size=8, color=RED, ha="left")

    # neck concat nodes
    for lv in levels:
        is_p2 = lv["name"] == "P2"
        y = lv["y"] + (lv["h"] - 0.86) / 2
        h = 0.86
        rounded(ax, nk_x, y, nk_w, h, P2_FILL if is_p2 else "white", RED if is_p2 else GOLD, lw=2.3 if is_p2 else 1.3)
        if is_p2:
            text(ax, nk_x + nk_w / 2, y + 0.58, "Concat  ← 接入 P2", size=10, weight="bold", color=RED)
            text(ax, nk_x + nk_w / 2, y + 0.26, "浅层位置还在", size=8, color=RED)
        else:
            text(ax, nk_x + nk_w / 2, y + 0.56, f"Concat  {lv['name']}", size=10, weight="bold", color=GOLD)
            text(ax, nk_x + nk_w / 2, y + 0.24, "官方已有", size=8, color=MUTED)
        lv["nk"] = (nk_x, y, nk_w, h)

        # lateral skip backbone → neck
        bx, by, bw, bh = lv["bb"]
        col = RED if is_p2 else LINE
        lw = 2.4 if is_p2 else 1.4
        arrow(ax, bx + bw + 0.02, by + bh / 2, nk_x - 0.04, y + h / 2, color=col, lw=lw)

    # FPN up (left of neck) and PAN down (right of neck)
    for i in range(3, 0, -1):
        a, b = levels[i], levels[i - 1]
        ax1, ay1 = a["nk"][0] + 0.42, a["nk"][1] + a["nk"][3]
        ax2, ay2 = b["nk"][0] + 0.42, b["nk"][1]
        col = RED if b["name"] == "P2" else GOLD
        arrow(ax, ax1, ay1, ax2, ay2, color=col, lw=1.8 if b["name"] == "P2" else 1.3)
    text(ax, nk_x - 0.10, 4.72, "上采样", size=8, color=GOLD, ha="right", va="center")

    for i in range(3):
        a, b = levels[i], levels[i + 1]
        ax1, ay1 = a["nk"][0] + a["nk"][2] - 0.42, a["nk"][1]
        ax2, ay2 = b["nk"][0] + b["nk"][2] - 0.42, b["nk"][1] + b["nk"][3]
        arrow(ax, ax1, ay1, ax2, ay2, color=GOLD, lw=1.3)
    text(ax, nk_x + nk_w + 0.08, 4.55, "下采样", size=8, color=GOLD, ha="left")

    text(ax, 7.82, 1.48, "稍后把已对齐的 Concat 换成 WeightedFuse", size=8, color=MUTED)

    # red circle on P2 neck node + skip
    p2 = levels[0]
    cx = (p2["bb"][0] + p2["bb"][2] + nk_x + nk_w) / 2
    cy = p2["y"] + p2["h"] / 2
    ax.add_patch(
        Ellipse(
            (cx + 0.18, cy),
            width=5.35,
            height=1.48,
            fill=False,
            edgecolor=RED,
            linewidth=2.6,
            linestyle="-",
            zorder=7,
        )
    )
    rounded(ax, 3.05, 6.38, 1.62, 0.36, RED, RED, lw=0, radius=0.12, z=8)
    text(ax, 3.86, 6.56, "红圈：本次接入", size=8.5, color="white", weight="bold", z=9)

    # detect heads
    for lv in levels:
        is_p2 = lv["name"] == "P2"
        y = lv["y"] + (lv["h"] - 0.86) / 2
        h = 0.86
        rounded(ax, hd_x, y, hd_w, h, P2_FILL if is_p2 else "white", RED if is_p2 else GREEN, lw=2.3 if is_p2 else 1.3)
        extra = "  ← 新增" if is_p2 else ""
        text(ax, hd_x + hd_w / 2, y + 0.56, f"Detect  {lv['name']}{extra}", size=10, weight="bold", color=RED if is_p2 else GREEN)
        text(ax, hd_x + hd_w / 2, y + 0.24, f"{lv['cells']}（约 20px 的头）", size=8, color=MUTED)
        lv["hd"] = (hd_x, y, hd_w, h)
        arrow(
            ax,
            nk_x + nk_w + 0.04,
            lv["nk"][1] + lv["nk"][3] / 2,
            hd_x - 0.04,
            y + h / 2,
            color=RED if is_p2 else GREEN,
            lw=2.2 if is_p2 else 1.4,
        )

    text(ax, hd_x + hd_w / 2, 1.48, "官方只有 P3 / P4 / P5 三层", size=8, color=MUTED)

    # bottom caption
    ax.add_patch(Rectangle((0.22, 0.18), 15.14, 0.88, facecolor=PAPER, edgecolor="#D1D5DB", linewidth=1.0, zorder=1))
    text(
        ax,
        0.42,
        0.74,
        "怎么读这张图：教室图进骨干，抽出四张特征图。官方脖子和检测头只用 P3、P4、P5（最小步长 8），后排约 20 像素的头只剩 1～2 格，容易漏。",
        size=8.8,
        ha="left",
        color=INK,
    )
    text(
        ax,
        0.42,
        0.42,
        "我们只改脖子：把更清晰的 P2（步长 4，约占 5 格）接进去，检测头看四层。融合暂时仍是 Concat；加权模块另画，不在这张图里改骨干。",
        size=8.8,
        ha="left",
        color=INK,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    draw(args.out)
