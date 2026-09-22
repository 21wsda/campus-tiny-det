"""
完整方法训练入口: P2 + WeightedFuse
先注册模块，再用 YOLO 类加载 fuse yaml 开始训练

用法（在仓库根目录，GPU机器上执行）：
  python scripts/train_fuse.py

如果没有 P2 预训练权重，改成从官方 yolov8n.pt 开始：
  python scripts/train_fuse.py --from-scratch
"""
import argparse
import os
import sys

# 把仓库根目录加到 Python 路径，确保能 import models
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import models.register  # noqa: F401  必须先注册

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yolov8n-p2-fuse.yaml",
                    help="模型配置文件路径")
    ap.add_argument("--data", default="configs/scut_head.yaml",
                    help="数据集配置文件路径")
    ap.add_argument("--pretrained", default="runs/detect/train-7/weights/best.pt",
                    help="预训练权重路径（默认从P2权重微调）")
    ap.add_argument("--from-scratch", action="store_true",
                    help="从官方 yolov8n.pt 开始，不从P2权重微调")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", default="train-fuse")
    args = ap.parse_args()

    if args.from_scratch:
        args.pretrained = "yolov8n.pt"
        print(f"[train_fuse] 从官方权重 {args.pretrained} 开始训练")
    else:
        print(f"[train_fuse] 从 P2 预训练权重 {args.pretrained} 微调")

    # 加载自定义模型配置
    model = YOLO(args.config)

    # 加载预训练权重（如果有）
    if args.pretrained and args.pretrained != "yolov8n.pt":
        model = model.load(args.pretrained)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
    )

    print(f"[train_fuse] 训练完成。结果在 runs/detect/{args.name}/")
    print("[train_fuse] 记得把结果填进 docs/实验记录.csv，method=yolov8n-p2-fuse")


if __name__ == "__main__":
    main()
