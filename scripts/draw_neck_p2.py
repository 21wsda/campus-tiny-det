"""
结构图：骨干 → 脖子（红圈接入 P2）→ 四层检测头。
用法：
  python scripts/draw_neck_p2.py
  python scripts/draw_neck_p2.py --out docs/figures/neck_p2.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch,
    Ellipse,
    Rectangle,
    Circle,
)


OUT_DEFAULT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "neck_p2.png"

# editorial palette — cool slate + one red accent
INK = "#15202B"
MUTED = "#5B6B7C"
SLATE = "#2F4A63"
GOLD = "#9A6B2F"
GREEN = "#1F6B4A"
RED = "#C62828"
RED_SOFT = "#FBE9E7"
LINE = "#7A8794"
RULE = "#D8DEE6"
PAPER = "#F7F9FC"
BB_BG = "#EEF3F8"
NK_BG = "#FBF6EE"
HD_BG = "#EEF7F2"
WHITE = "#FFFFFF"


def setup_font():
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def rounded(ax, x, y, w, h, fc, ec, lw=1.5, z=3, radius=0.10, alpha=1.0):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
        alpha=alpha,
        mutation_aspect=0.55,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=LINE, lw=1.6, rad=0.0, z=4, ms=12):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=z,
            shrinkA=1,
            shrinkB=1,
        )
    )


def text(ax, x, y, s, size=10, color=INK, weight="normal", ha="center", va="center", z=6):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        zorder=z,
        linespacing=1.25,
    )


def panel(ax, x, y, w, h, fc, accent, title, subtitle, index):
    rounded(ax, x, y, w, h, fc, RULE, lw=1.0, radius=0.14, z=0)
    # left accent bar
    ax.add_patch(Rectangle((x, y), 0.08, h, facecolor=accent, edgecolor="none", zorder=1))
    text(ax, x + 0.28, y + h - 0.32, f"{index}  {title}", size=13, weight="bold", color=accent, ha="left")
    text(ax, x + 0.28, y + h - 0.62, subtitle, size=8.8, color=MUTED, ha="left")


def feature_card(ax, x, y, w, h, name, stride, detail, accent=False):
    ec = RED if accent else SLATE
    fc = RED_SOFT if accent else WHITE
    lw = 2.2 if accent else 1.25
    rounded(ax, x, y, w, h, fc, ec, lw=lw, radius=0.09, z=3)
    text(ax, x + w / 2, y + h * 0.64, f"{name}  ·  {stride}", size=10.5, weight="bold", color=ec)
    text(ax, x + w / 2, y + h * 0.28, detail, size=8.2, color=MUTED)


def draw(out: Path):
    setup_font()
    W, H = 16.0, 9.0
    fig, ax = plt.subplots(figsize=(W, H), dpi=220)
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    ax.set_facecolor(WHITE)

    # header rule
    ax.add_patch(Rectangle((0.35, 8.35), 15.3, 0.015, facecolor=INK, edgecolor="none", zorder=1))
    text(ax, 0.35, 8.62, "YOLOv8n-P2 结构示意", size=20, weight="bold", color=INK, ha="left")
    text(ax, 15.65, 8.62, "只改颈部 · 骨干与检测器不重做", size=11, color=MUTED, ha="right")
    text(
        ax,
        0.35,
        8.12,
        "红圈标出本次改动：把步长 4 的 P2 接到脖子，检测头由三层改为四层",
        size=10,
        color=MUTED,
        ha="left",
    )

    # three panels
    panel(ax, 0.30, 1.55, 4.55, 6.25, BB_BG, SLATE, "骨干 Backbone", "不换，照常抽出 P2～P5", "①")
    panel(ax, 5.10, 1.55, 5.70, 6.25, NK_BG, GOLD, "颈部 Neck", "只改这里：接入 P2，融合先 Concat", "②")
    panel(ax, 11.05, 1.55, 4.65, 6.25, HD_BG, GREEN, "检测头 Head", "跟着多看一层，不做新检测器", "③")

    # vertical positions for P2–P5 (top to bottom)
    rows = [
        {"name": "P2", "stride": "步长 4", "hw": "160×160", "cells": "约 5 格", "y": 6.35, "accent": True},
        {"name": "P3", "stride": "步长 8", "hw": "80×80", "cells": "约 1～2 格", "y": 4.95, "accent": False},
        {"name": "P4", "stride": "步长 16", "hw": "40×40", "cells": "不到 1 格", "y": 3.55, "accent": False},
        {"name": "P5", "stride": "步长 32", "hw": "20×20", "cells": "几乎没有", "y": 2.15, "accent": False},
    ]
    card_h = 0.95

    # backbone cards — decreasing width suggests resolution
    bb_cx = 2.55
    for i, r in enumerate(rows):
        w = 2.55 - i * 0.22
        x = bb_cx - w / 2
        feature_card(ax, x, r["y"], w, card_h, r["name"], r["stride"], r["hw"], accent=r["accent"])
        r["bb"] = (x, r["y"], w, card_h)
        if i < 3:
            nxt = rows[i + 1]
            w2 = 2.55 - (i + 1) * 0.22
            arrow(
                ax,
                bb_cx,
                r["y"],
                bb_cx,
                nxt["y"] + card_h,
                color=SLATE,
                lw=1.2,
                ms=10,
            )

    # input badge
    rounded(ax, 1.55, 7.42, 2.00, 0.38, WHITE, SLATE, lw=1.1, radius=0.08, z=5)
    text(ax, 2.55, 7.61, "输入 640×640", size=9, color=SLATE, weight="bold")
    arrow(ax, 2.55, 7.42, 2.55, 7.30, color=SLATE, lw=1.2, ms=9)
    text(ax, 0.55, 1.78, "P2 算出来了，官方后面丢掉", size=8.5, color=RED, ha="left")

    # neck cards
    nk_x, nk_w = 6.15, 3.55
    for r in rows:
        y = r["y"]
        ec = RED if r["accent"] else GOLD
        fc = RED_SOFT if r["accent"] else WHITE
        lw = 2.2 if r["accent"] else 1.25
        rounded(ax, nk_x, y, nk_w, card_h, fc, ec, lw=lw, radius=0.09)
        if r["accent"]:
            text(ax, nk_x + nk_w / 2, y + 0.62, "Concat  ←  接入 P2", size=11, weight="bold", color=RED)
            text(ax, nk_x + nk_w / 2, y + 0.28, "浅层位置还在", size=8.5, color=RED)
        else:
            text(ax, nk_x + nk_w / 2, y + 0.62, f"Concat  {r['name']}", size=11, weight="bold", color=GOLD)
            text(ax, nk_x + nk_w / 2, y + 0.28, "官方已有", size=8.5, color=MUTED)
        r["nk"] = (nk_x, y, nk_w, card_h)

        # lateral: backbone → neck
        bx, by, bw, bh = r["bb"]
        arrow(
            ax,
            bx + bw + 0.04,
            by + bh / 2,
            nk_x - 0.04,
            y + card_h / 2,
            color=RED if r["accent"] else LINE,
            lw=2.4 if r["accent"] else 1.35,
            ms=13 if r["accent"] else 11,
        )

    # FPN up / PAN down
    for i in range(3, 0, -1):
        a, b = rows[i], rows[i - 1]
        col = RED if b["accent"] else GOLD
        arrow(
            ax,
            nk_x + 0.55,
            a["nk"][1] + card_h,
            nk_x + 0.55,
            b["nk"][1],
            color=col,
            lw=1.7 if b["accent"] else 1.25,
            ms=10,
        )
    text(ax, nk_x - 0.12, 5.55, "上采样", size=8.2, color=GOLD, ha="right")

    for i in range(3):
        a, b = rows[i], rows[i + 1]
        arrow(
            ax,
            nk_x + nk_w - 0.55,
            a["nk"][1],
            nk_x + nk_w - 0.55,
            b["nk"][1] + card_h,
            color=GOLD,
            lw=1.25,
            ms=10,
        )
    text(ax, nk_x + nk_w + 0.10, 5.55, "下采样", size=8.2, color=GOLD, ha="left")
    text(ax, 7.92, 1.78, "稍后把已对齐的 Concat 换成 WeightedFuse", size=8.5, color=MUTED)

    # red ellipse highlight on P2 path
    p2 = rows[0]
    cx = (p2["bb"][0] + p2["bb"][2] + nk_x + nk_w) / 2 + 0.05
    cy = p2["y"] + card_h / 2
    ax.add_patch(
        Ellipse(
            (cx, cy),
            width=5.9,
            height=1.55,
            fill=False,
            edgecolor=RED,
            linewidth=2.8,
            zorder=8,
        )
    )
    # badge
    rounded(ax, 3.55, 7.40, 1.85, 0.40, RED, RED, lw=0, radius=0.12, z=9)
    text(ax, 4.48, 7.60, "红圈 · 本次接入", size=9, color=WHITE, weight="bold", z=10)
    # small pointer dot
    ax.add_patch(Circle((4.48, 7.40), 0.05, facecolor=RED, edgecolor="none", zorder=10))

    # head cards
    hd_x, hd_w = 11.45, 3.85
    for r in rows:
        y = r["y"]
        ec = RED if r["accent"] else GREEN
        fc = RED_SOFT if r["accent"] else WHITE
        lw = 2.2 if r["accent"] else 1.25
        rounded(ax, hd_x, y, hd_w, card_h, fc, ec, lw=lw, radius=0.09)
        tag = "  ← 新增" if r["accent"] else ""
        text(ax, hd_x + hd_w / 2, y + 0.62, f"Detect  {r['name']}{tag}", size=11, weight="bold", color=ec)
        text(ax, hd_x + hd_w / 2, y + 0.28, f"{r['cells']}（约 20px 的头）", size=8.5, color=MUTED)
        arrow(
            ax,
            nk_x + nk_w + 0.06,
            r["nk"][1] + card_h / 2,
            hd_x - 0.04,
            y + card_h / 2,
            color=RED if r["accent"] else GREEN,
            lw=2.3 if r["accent"] else 1.35,
            ms=13 if r["accent"] else 11,
        )

    text(ax, hd_x + hd_w / 2, 1.78, "官方只有 P3 / P4 / P5 三层", size=8.5, color=MUTED)

    # footer caption card
    rounded(ax, 0.30, 0.18, 15.40, 1.18, PAPER, RULE, lw=1.0, radius=0.10, z=1)
    text(ax, 0.55, 0.98, "怎么读", size=10, weight="bold", color=INK, ha="left")
    text(
        ax,
        0.55,
        0.68,
        "教室图进骨干，抽出四张特征图。官方脖子与检测头只用 P3、P4、P5（最小步长 8），后排约 20 像素的头只剩 1～2 格，容易漏。",
        size=9,
        color=INK,
        ha="left",
    )
    text(
        ax,
        0.55,
        0.38,
        "我们只改脖子：把更清晰的 P2（步长 4，约占 5 格）接进去，检测头看四层。融合暂时仍是 Concat；加权模块另写，不画进本图。",
        size=9,
        color=INK,
        ha="left",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=WHITE, pad_inches=0.15)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    draw(args.out)
