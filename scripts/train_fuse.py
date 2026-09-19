"""
完整方法训练入口: P2 + WeightedFuse
先注册模块，再用 YOLO 类加载 fuse yaml 开始训练
"""
import models.register  # noqa: F401  必须先注册

from ultralytics import YOLO


def main():
    # 配置文件路径（在仓库根目录运行）
    model_cfg = "configs/yolov8n-p2-fuse.yaml"

    # 从 P2 的预训练权重微调（接 P2 那一步训好的）
    # 如果没有 P2 权重，改成 pretrained="yolov8n.pt" 从官方权重开始
    pretrained = "runs/detect/train-7/weights/best.pt"

    model = YOLO(model_cfg).load(pretrained)

    model.train(
        data="configs/scut_head.yaml",
        imgsz=640,
        epochs=100,
        batch=8,
        name="train-fuse",
        pretrained=True,
    )

    print("[train_fuse] 训练完成。查看 runs/detect/train-fuse/")


if __name__ == "__main__":
    main()
