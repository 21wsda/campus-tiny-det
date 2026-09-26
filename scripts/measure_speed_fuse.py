"""
fuse 模型测速脚本（T11 / T28 补测用）

为什么不能直接用 measure_speed.py：
  fuse 的加权融合层是训练时用 models.register 把 Concat 顶替掉的。
  直接 YOLO(best.pt) 有可能把图当成普通 Concat，那就不是完整方法了，
  测出来的 FPS 也不作数。这个脚本先确认「真·加权层」在不在，再测速。

测速方法和 scripts/measure_speed.py 完全一致（预热 20 张 + 正测 100 张），
这样和 yolov8s / yolov5nu / P2 的数字才可比。

用法：
  python scripts/measure_speed_fuse.py --weights runs/detect/train-fuse-v2-4/weights/best.pt --imgsz 640
"""
import argparse
import os
import sys
import time

# 把仓库根目录加到 Python 路径
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from ultralytics import YOLO
from ultralytics.nn.modules import Concat
from models.weighted_fuse import WeightedFuse


def count_fuse(model):
    return sum(1 for layer in model.model.model if isinstance(layer, WeightedFuse))


def build_fuse_model(weights, config, imgsz):
    """优先直接加载；图里没有加权层时才用 yaml 重建"""
    try:
        model = YOLO(weights)
        n = count_fuse(model)
        if n > 0:
            print(f"[fuse-speed] 直接加载成功，加权层 {n} 个")
            return model
        print("[fuse-speed] 直接加载后没有加权层，改用 yaml 重建")
    except Exception as err:
        print(f"[fuse-speed] 直接加载失败（{err}），改用 yaml 重建")

    model = YOLO(config)
    layers = model.model.model
    for i, layer in enumerate(layers):
        if isinstance(layer, Concat):
            fuse = WeightedFuse(dimension=layer.d)
            fuse.f = layer.f      # 这三行不能省，否则前向会报没有 f 属性
            fuse.i = layer.i
            fuse.type = layer.type
            fuse.np = layer.np
            layers[i] = fuse
    print(f"[fuse-speed] 替换 Concat → WeightedFuse：{count_fuse(model)} 个")

    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=imgsz, verbose=False)   # 先建出加权层参数
    model = model.load(weights)                        # 再加载训练好的权重
    print(f"[fuse-speed] 加载完成，加权层 {count_fuse(model)} 个")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="fuse 权重路径")
    ap.add_argument("--config", default="configs/yolov8n-p2-fuse.yaml",
                    help="标准 Concat 的模型配置（重建时才用）")
    ap.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    ap.add_argument("--warmup", type=int, default=20, help="预热张数")
    ap.add_argument("--num", type=int, default=100, help="测速张数")
    args = ap.parse_args()

    model = build_fuse_model(args.weights, args.config, args.imgsz)

    # 模型信息：参数量、FLOPs
    model.info(verbose=True)

    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    print(f"\n预热 {args.warmup} 张...")
    for _ in range(args.warmup):
        model.predict(dummy, imgsz=args.imgsz, verbose=False)

    print(f"测速 {args.num} 张...")
    t0 = time.time()
    for _ in range(args.num):
        model.predict(dummy, imgsz=args.imgsz, verbose=False)
    elapsed = time.time() - t0

    fps = args.num / elapsed
    print(f"\n{'='*40}")
    print(f"权重: {args.weights}")
    print(f"输入尺寸: {args.imgsz}")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"FPS: {fps:.2f}")
    print(f"单张耗时: {1000/fps:.2f}ms")
    print(f"{'='*40}")
    print("\n把上面的 params(M)、FLOPs(G)、FPS 填进实验记录csv对应行。")


if __name__ == "__main__":
    main()
