"""
VOC xml -> YOLO txt 转换脚本（工单 T05/T07，甲主做）
用法：
  python scripts/convert_voc_to_yolo.py --root ~/datasets/scut_head --out ~/datasets/scut_head_yolo
  python scripts/convert_voc_to_yolo.py --root ~/datasets/scut_head --out ~/datasets/scut_head_yolo --seed 42

输入目录期望 VOC 布局：
  <root>/JPEGImages/   （或 images/）
  <root>/Annotations/  （或 annotations/、labels/）
  <root>/ImageSets/Main/train.txt、val.txt  （可选，有则按官方划分）

输出目录：
  <out>/images/train/  <out>/images/val/
  <out>/labels/train/  <out>/labels/val/
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_xml(xml_path):
    """解析单个 VOC xml，返回 YOLO 格式行列表（类别 0=head）。"""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)
    lines = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        xc = ((xmin + xmax) / 2) / w
        yc = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        xc, yc, bw, bh = [min(max(v, 0.0), 1.0) for v in (xc, yc, bw, bh)]
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def collect_pairs(root: Path):
    """在 root 下找 Annotations 和 JPEGImages 目录，配对 (jpg, xml)。"""
    ann = None
    for name in ["Annotations", "annotations", "labels"]:
        d = root / name
        if d.is_dir():
            ann = d
            break
    img = None
    for name in ["JPEGImages", "images", "JPEG"]:
        d = root / name
        if d.is_dir():
            img = d
            break
    if ann is None or img is None:
        raise SystemExit(
            f"找不到 Annotations/JPEGImages：{root}\n"
            f"实际有：{[p.name for p in root.iterdir()]}"
        )
    pairs = []
    for xml in sorted(ann.glob("*.xml")):
        stem = xml.stem
        jpg = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            cand = img / f"{stem}{ext}"
            if cand.exists():
                jpg = cand
                break
        if jpg:
            pairs.append((jpg, xml))
    return pairs


def root_image_sets(root: Path, pairs):
    """如果有 ImageSets/Main/train.txt、val.txt，按官方划分返回 dict。"""
    main_dir = root / "ImageSets" / "Main"
    if not main_dir.is_dir():
        return None
    stem_to_pair = {p[0].stem: p for p in pairs}
    splits = {}
    for split_name in ["train", "val", "test"]:
        txt = main_dir / f"{split_name}.txt"
        if txt.exists():
            items = []
            for line in txt.read_text(encoding="utf-8").splitlines():
                stem = line.strip().split()[0] if line.strip() else ""
                if stem and stem in stem_to_pair:
                    items.append(stem_to_pair[stem])
            if items:
                splits[split_name] = items
    return splits if splits else None


def split_pairs(root: Path, pairs, seed=42):
    """优先用官方 ImageSets 划分；没有则随机留 20% 做 val。"""
    sets = root_image_sets(root, pairs)
    if sets:
        return sets
    rnd = random.Random(seed)
    pairs = pairs[:]
    rnd.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.2))
    return {"train": pairs[n_val:], "val": pairs[:n_val]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="VOC 格式数据集根目录")
    ap.add_argument("--out", required=True, help="YOLO 格式输出目录")
    ap.add_argument("--seed", type=int, default=42, help="随机划分种子（无 ImageSets 时生效）")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser()

    pairs = collect_pairs(root)
    if not pairs:
        raise SystemExit("0 对图和 xml，检查 --root")

    splits = split_pairs(root, pairs, args.seed)

    total_labels = 0
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        split_labels = 0
        for jpg, xml in items:
            shutil.copy2(jpg, out / "images" / split / jpg.name)
            lines = parse_xml(xml)
            split_labels += len(lines)
            txt = ("\n".join(lines) + "\n") if lines else ""
            (out / "labels" / split / (jpg.stem + ".txt")).write_text(txt, encoding="utf-8")
        print(f"{split}: {len(items)} 张图, {split_labels} 个标注框")
        total_labels += split_labels

    print(f"\n完成：共 {len(pairs)} 张图, {total_labels} 个标注框")
    print(f"输出目录：{out}")
    print(f"划分种子：{args.seed}（若使用 ImageSets 则忽略 seed）")


if __name__ == "__main__":
    main()
