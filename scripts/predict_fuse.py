"""
fuse 模型推理脚本

两种加载方式，优先用更可靠的直接加载：
  A. YOLO(weights) 直接加载训练时保存的模型对象（里面就是 WeightedFuse 层）
  B. 万一图里变回了普通 Concat，就用标准 Concat 的 yaml 建图 →
     把 Concat 换成 WeightedFuse → 先空跑一次建出参数 → 再加载权重

用法：
  python scripts/predict_fuse.py --weights fuse_best.pt --source 图片或文件夹 --conf 0.25

丙做三例图时用这个脚本，不要直接 yolo predict，因为直接 yolo 命令会变回 Concat。
"""
import argparse
import os
import sys

# 把仓库根目录加到 Python 路径
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from ultralytics import YOLO
from ultralytics.nn.modules import Concat
from models.weighted_fuse import WeightedFuse


def count_fuse(model):
    """数一数图里真·加权融合层有几个（0 就说明权重没加载对）"""
    return sum(1 for layer in model.model.model if isinstance(layer, WeightedFuse))


def fix_fuse_attributes(model):
    """直接加载的权重里，WeightedFuse层可能缺f/i等记账属性，补上"""
    count = 0
    layers = model.model.model
    for i, layer in enumerate(layers):
        if isinstance(layer, WeightedFuse):
            if not hasattr(layer, 'f'):
                layer.f = -1
            if not hasattr(layer, 'i'):
                layer.i = i
            if not hasattr(layer, 'type'):
                layer.type = type(layer)
            count += 1
    print(f"[predict_fuse] 修复 {count} 个 WeightedFuse 层的记账属性")
    return model


def replace_concat_with_fuse(model):
    """遍历模型，把所有 Concat 层替换成 WeightedFuse"""
    count = 0
    layers = model.model.model
    for i, layer in enumerate(layers):
        if isinstance(layer, Concat):
            fuse = WeightedFuse(dimension=layer.d)
            device = next(model.parameters()).device
            fuse = fuse.to(device)
            # 关键：把 ultralytics 建图时挂在 Concat 上的记账属性搬过来。
            # 少了这一步，前向会报 AttributeError: 'WeightedFuse' object has no attribute 'f'
            fuse.f = layer.f
            fuse.i = layer.i
            fuse.type = layer.type
            fuse.np = layer.np
            layers[i] = fuse
            count += 1
            print(f"  [replace] 层 {i}: Concat → WeightedFuse")
    # 同步修改 yaml 配置
    if hasattr(model.model, 'yaml') and model.model.yaml is not None:
        yaml_cfg = model.model.yaml
        if 'head' in yaml_cfg:
            for layer_def in yaml_cfg['head']:
                if len(layer_def) >= 3 and layer_def[2] == 'Concat':
                    layer_def[2] = 'WeightedFuse'
    print(f"[predict_fuse] 替换 {count} 个 Concat → WeightedFuse")
    return model


def load_fuse_model(weights, config, imgsz):
    """优先直接加载；只有图里确实没有加权层时，才手工重建"""
    print(f"[predict_fuse] 直接加载 {weights} ...")
    try:
        model = YOLO(weights)
        n = count_fuse(model)
        if n > 0:
            print(f"[predict_fuse] 直接加载成功，加权层 {n} 个")
            model = fix_fuse_attributes(model)
            return model
        print("[predict_fuse] 警告：直接加载后图里没有加权层，改用重建方式")
    except Exception as err:
        print(f"[predict_fuse] 直接加载失败（{err}），改用重建方式")

    print(f"[predict_fuse] 用 {config} 重建模型...")
    model = YOLO(config)
    model = replace_concat_with_fuse(model)

    # WeightedFuse 的投影层是「第一次前向」时才建出来的。
    # 必须先空跑一次把参数建出来，否则加载权重时这些参数对不上，会被静默丢掉。
    print("[predict_fuse] 先空跑一次，初始化加权层参数...")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=imgsz, verbose=False)

    print(f"[predict_fuse] 加载权重 {weights}...")
    model = model.load(weights)
    print(f"[predict_fuse] 加载完成，加权层 {count_fuse(model)} 个")
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

    model = load_fuse_model(args.weights, args.config, args.imgsz)

    print("[predict_fuse] 开始推理...")
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
    )

    print("[predict_fuse] 完成。结果在 runs/detect/predict/")


if __name__ == "__main__":
    main()
