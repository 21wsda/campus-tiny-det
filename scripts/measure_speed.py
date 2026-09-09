"""
测速脚本（工单 T11，丙主做，甲提供权重路径）
测量模型的 FPS、参数量、FLOPs。
用法：
  python scripts/measure_speed.py --weights runs/detect/train/weights/best.pt --imgsz 640
"""
import argparse
import time

import numpy as np
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="模型权重路径")
    ap.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    ap.add_argument("--warmup", type=int, default=20, help="预热张数")
    ap.add_argument("--num", type=int, default=100, help="测速张数")
    args = ap.parse_args()

    model = YOLO(args.weights)

    # 模型信息：参数量、FLOPs
    model.info(verbose=True)

    # 构造随机输入图
    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    # 预热
    print(f"\n预热 {args.warmup} 张...")
    for _ in range(args.warmup):
        model.predict(dummy, imgsz=args.imgsz, verbose=False)

    # 正式测速
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
