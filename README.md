# campus-tiny-det

面向校园监控的轻量小目标检测。以 YOLOv8n 为底座，在颈部接 P2 并改加权融合，不换骨干。

## 环境配置

Python 3.10 或 3.11（不要 3.13）。

```bash
pip install ultralytics opencv-python matplotlib
python --version
```

GPU 机器另外装带 CUDA 的 PyTorch。CPU 只能验证命令，正式训练必须 GPU。

## 数据放哪

数据放在仓库外面，例如 `C:/Users/75804/datasets/scut_head/`（VOC 布局：`JPEGImages/`、`Annotations/`）。

- 教室数据 SCUT-HEAD PartA：https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release
- 远处人 TinyPerson：https://github.com/ucas-vg/PointTinyBenchmark/tree/master/dataset

`.gitignore` 已过滤 `datasets/`、`runs/`、`*.pt`、`*.jpg`，数据和权重不进 Git。

## 怎么转数据（VOC → YOLO）

```bash
python scripts/convert_voc_to_yolo.py --root C:/Users/75804/datasets/scut_head --out C:/Users/75804/datasets/scut_head_yolo
```

输出目录结构（ultralytics 要求）：

```
scut_head_yolo/
  images/train/  images/val/
  labels/train/  labels/val/
```

转换后抽查标签是否画对：

```bash
python scripts/viz_labels.py --imgdir C:/Users/75804/datasets/scut_head_yolo/images/train \
    --lbldir C:/Users/75804/datasets/scut_head_yolo/labels/train --n 5 --outdir docs/figures/label_check
```

## 怎么 train

先改 `configs/scut_head.yaml` 里的 `path` 为你本机的 YOLO 格式数据目录绝对路径。

冒烟训练（20 张，3 epoch，只验证流程）：

```bash
yolo train model=yolov8n.pt data=configs/scut_head.yaml imgsz=640 epochs=3 batch=8
```

正式训练（统一设置，不许改 imgsz 和 epochs）：

```bash
yolo train model=yolov8n.pt data=configs/scut_head.yaml imgsz=640 epochs=100 batch=16
```

显存不够只降 batch。断了用 `yolo train resume model=runs/detect/train/weights/last.pt`。

## 预测

```bash
yolo predict model=yolov8n.pt source=./test.jpg conf=0.25 save=True
```

## 仓库结构

```
campus-tiny-det/
  README.md
  configs/
    scut_head.yaml          # 数据集配置
  scripts/
    convert_voc_to_yolo.py  # VOC → YOLO 转换（甲）
    viz_labels.py           # 标签画框检查（乙）
    plot_box_size.py        # 框尺寸直方图（乙）
  docs/
    实验记录.csv             # 一行实验一行，谁训谁填
    figures/                 # 截图、直方图、对比图
  .gitignore
```

## 实验记录

`docs/实验记录.csv` 表头固定，自己跑的训练自己填一行，不许代填：

```
date,method,data,epochs,imgsz,batch,map50,map50-95,params_m,flops_g,fps,weights,note
```
