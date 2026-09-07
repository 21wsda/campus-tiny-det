"""
框尺寸直方图脚本（工单 T07，乙主做，甲搭手）
统计 YOLO 标签中所有框的像素宽高，出直方图，确认大量框确实很小。
用法：
  python scripts/plot_box_size.py --imgdir ~/datasets/scut_head_yolo/images/train \
      --lbldir ~/datasets/scut_head_yolo/labels/train --out docs/figures/box_size_hist.png
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def collect_sizes(imgdir, lbldir):
    widths, heights, areas = [], [], []
    imgs = sorted(imgdir.glob("*.jpg")) + sorted(imgdir.glob("*.png"))
    for img_path in imgs:
        lbl_path = lbldir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = map(float, parts)
            pw = bw * w
            ph = bh * h
            widths.append(pw)
            heights.append(ph)
            areas.append(pw * ph)
    return np.array(widths), np.array(heights), np.array(areas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--lbldir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    imgdir = Path(args.imgdir).expanduser()
    lbldir = Path(args.lbldir).expanduser()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    widths, heights, areas = collect_sizes(imgdir, lbldir)
    if len(widths) == 0:
        raise SystemExit("没有读到任何标签框")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(widths, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title(f"Box Width (px)\nmedian={np.median(widths):.1f}, mean={np.mean(widths):.1f}")
    axes[0].set_xlabel("width (px)")
    axes[0].set_ylabel("count")

    axes[1].hist(heights, bins=50, color="seagreen", edgecolor="white")
    axes[1].set_title(f"Box Height (px)\nmedian={np.median(heights):.1f}, mean={np.mean(heights):.1f}")
    axes[1].set_xlabel("height (px)")

    axes[2].hist(np.sqrt(areas), bins=50, color="coral", edgecolor="white")
    axes[2].set_title(f"Box sqrt(Area) (px)\nmedian={np.median(np.sqrt(areas)):.1f}")
    axes[2].set_xlabel("sqrt(area) (px)")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"直方图已保存：{out}")
    print(f"总框数：{len(widths)}")
    print(f"宽度 median={np.median(widths):.1f}px, mean={np.mean(widths):.1f}px")
    print(f"高度 median={np.median(heights):.1f}px, mean={np.mean(heights):.1f}px")
    small = np.sum(widths < 32)
    print(f"宽度 < 32px 的框：{small} ({small/len(widths)*100:.1f}%)")


if __name__ == "__main__":
    main()
