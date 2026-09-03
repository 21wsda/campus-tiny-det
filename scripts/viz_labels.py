"""按 YOLO txt 在图上画框，用来检查 VOC→YOLO 有没有转错。"""
import argparse
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def yolo_to_xyxy(xc, yc, bw, bh, width, height):
    x1 = int((xc - bw / 2) * width)
    y1 = int((yc - bh / 2) * height)
    x2 = int((xc + bw / 2) * width)
    y2 = int((yc + bh / 2) * height)
    return x1, y1, x2, y2


def draw_one(img_path: Path, lbl_path: Path, out_path: Path) -> int:
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"读不了图：{img_path}")
    height, width = img.shape[:2]
    n_box = 0
    if lbl_path.is_file():
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            xc, yc, bw, bh = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, width, height)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            n_box += 1
    else:
        cv2.putText(
            img, "NO LABEL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), img):
        raise SystemExit(f"写不出图：{out_path}")
    return n_box


def collect_images(imgdir: Path):
    imgs = [p for p in sorted(imgdir.iterdir()) if p.is_file() and p.suffix in IMG_EXTS]
    if not imgs:
        raise SystemExit(f"文件夹里没有图：{imgdir}")
    return imgs


def main():
    ap = argparse.ArgumentParser(description="按 YOLO txt 画框，检查标签有没有飞掉")
    ap.add_argument("--imgdir", required=True, help="图片文件夹，例如 images/train")
    ap.add_argument("--lbldir", required=True, help="标签文件夹，例如 labels/train")
    ap.add_argument("--n", type=int, default=5, help="抽几张画")
    ap.add_argument("--outdir", required=True, help="检查图输出目录")
    args = ap.parse_args()

    imgdir = Path(args.imgdir)
    lbldir = Path(args.lbldir)
    outdir = Path(args.outdir)
    if not imgdir.is_dir():
        raise SystemExit(f"找不到图片目录：{imgdir}")
    if not lbldir.is_dir():
        raise SystemExit(f"找不到标签目录：{lbldir}")

    imgs = collect_images(imgdir)
    picked = imgs[: max(args.n, 0)]
    print(f"共 {len(imgs)} 张，画 {len(picked)} 张 → {outdir}")
    for img_path in picked:
        lbl_path = lbldir / f"{img_path.stem}.txt"
        out_path = outdir / img_path.name
        n_box = draw_one(img_path, lbl_path, out_path)
        print(f"{img_path.name}: {n_box} boxes → {out_path}")


if __name__ == "__main__":
    main()
