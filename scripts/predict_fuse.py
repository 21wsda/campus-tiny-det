"""
fuse 模型推理脚本
用和训练一样的方式：先用标准 Concat 构建模型，再替换成 WeightedFuse，然后加载权重

用法：
  python scripts/predict_fuse.py --weights fuse_best.pt --source 图片或文件夹 --conf 0.25

丙做三例图时用这个脚本，不要直接 yolo predict，因为直接加载会变回 Concat。
"""
import argparse
import os
import sys

# 把仓库根目录加到 Python 路径
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ultralytics import YOLO
from ultralytics.nn.modules import Concat
from models.weighted_fuse import WeightedFuse


def replace_concat_with_fuse(model):
    """遍历模型，把所有 Concat 层替换成 WeightedFuse"""
    count = 0
    layers = model.model.model
    for i, layer in enumerate(layers):
        if isinstance(layer, Concat):
            fuse = WeightedFuse(dimension=layer.d)
            device = next(model.parameters()).device
            fuse = fuse.to(device)
            layers[i] = fuse
            count += 1
    # 同步修改 yaml 配置
    if hasattr(model.model, 'yaml') and model.model.yaml is not None:
        yaml_cfg = model.model.yaml
        if 'head' in yaml_cfg:
            for layer_def in yaml_cfg['head']:
                if len(layer_def) >= 3 and layer_def[2] == 'Concat':
                    layer_def[2] = 'WeightedFuse'
    print(f"[predict_fuse] 替换 {count} 个 Concat → WeightedFuse")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="fuse 权重路径")
    ap.add_argument("--source", required=True, help="图片或文件夹路径")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--config", default="configs/yolov8n-p2-fuse.yaml",
                    help="模型配置（用标准 Concat 的那个）")
    args = ap.parse_args()

    # 1. 用标准 Concat 的 yaml 构建模型
    print("[predict_fuse] 构建模型...")
    model = YOLO(args.config)

    # 2. 替换 Concat → WeightedFuse
    model = replace_concat_with_fuse(model)

    # 3. 加载权重
    print(f"[predict_fuse] 加载权重 {args.weights}...")
    model = model.load(args.weights)

    # 4. 推理
    print("[predict_fuse] 开始推理...")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
    )

    print(f"[predict_fuse] 完成。结果在 runs/detect/predict/")


if __name__ == "__main__":
    main()
