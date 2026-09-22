"""
完整方法训练入口: P2 + WeightedFuse
1. 用标准 Concat 的 yaml 构建模型（让 parse_model 正常工作）
2. 遍历模型，把所有 Concat 层替换成 WeightedFuse
3. 加载预训练权重，开始训练

用法（在仓库根目录，GPU机器上执行）：
  python scripts/train_fuse.py

如果没有 P2 预训练权重，改成从官方 yolov8n.pt 开始：
  python scripts/train_fuse.py --from-scratch
"""
import argparse
import os
import sys

# 把仓库根目录加到 Python 路径
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Concat
from models.weighted_fuse import WeightedFuse


def replace_concat_with_fuse(model):
    """遍历模型，把所有 Concat 层替换成 WeightedFuse"""
    count = 0
    # model.model 是 nn.Sequential
    for i, layer in enumerate(model.model):
        if isinstance(layer, Concat):
            # 创建 WeightedFuse，参数会在第一次 forward 时自动初始化
            fuse = WeightedFuse(dimension=layer.d)
            # 把新模块移到和原层相同的 device
            device = next(layer.parameters()).device if len(list(layer.parameters())) > 0 else next(model.parameters()).device
            fuse = fuse.to(device)
            model.model[i] = fuse
            count += 1
            print(f"  [replace] 层 {i}: Concat → WeightedFuse")
    print(f"[replace] 共替换 {count} 个 Concat 层为 WeightedFuse")
    return model


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

    # 1. 用标准 Concat 的 yaml 构建模型
    print("[train_fuse] 构建模型（标准Concat）...")
    model = YOLO(args.config)

    # 2. 把所有 Concat 替换成 WeightedFuse
    print("[train_fuse] 替换 Concat → WeightedFuse...")
    model = replace_concat_with_fuse(model)

    # 3. 加载预训练权重（Concat没有可学习参数，替换不影响其他层权重加载）
    if args.pretrained and args.pretrained != "yolov8n.pt":
        if os.path.exists(args.pretrained):
            print(f"[train_fuse] 加载预训练权重 {args.pretrained}...")
            model = model.load(args.pretrained)
        else:
            print(f"[train_fuse] 警告：预训练权重 {args.pretrained} 不存在，从随机初始化开始")

    # 4. 开始训练
    print("[train_fuse] 开始训练...")
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
