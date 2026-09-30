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

数据放在仓库外面，例如 `C:/Users/你的用户名/datasets/scut_head/`（VOC 布局：`JPEGImages/`、`Annotations/`）。

- 教室数据 SCUT-HEAD PartA：https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release
- 远处人 TinyPerson：https://github.com/ucas-vg/PointTinyBenchmark/tree/master/dataset

`.gitignore` 已过滤 `datasets/`、`runs/`、`*.pt`、`*.jpg`，数据和权重不进 Git。

## 怎么转数据（VOC → YOLO）

以下路径仅为示例，请根据自己电脑上的实际数据路径修改。

```bash
python scripts/convert_voc_to_yolo.py --root C:/Users/你的用户名/datasets/scut_head --out C:/Users/你的用户名/datasets/scut_head_yolo
```

输出目录结构（ultralytics 要求）：

```
scut_head_yolo/
  images/train/  images/val/
  labels/train/  labels/val/
```

转换后抽查标签是否画对：

```bash
python scripts/viz_labels.py --imgdir C:/Users/你的用户名/datasets/scut_head_yolo/images/train \
    --lbldir C:/Users/你的用户名/datasets/scut_head_yolo/labels/train --n 5 --outdir docs/figures/label_check
```

## 怎么 train

先改 `configs/scut_head.yaml` 里的 `path` 为你本机的 YOLO 格式数据目录绝对路径。

冒烟训练（3 epoch，只验证流程）：

```bash
yolo train model=yolov8n.pt data=configs/scut_head.yaml imgsz=640 epochs=3 batch=8
```

正式训练（统一设置，不许改 imgsz 和 epochs）：

```bash
yolo train model=yolov8n.pt data=configs/scut_head.yaml imgsz=640 epochs=100 batch=16
```

显存不够只降 batch。断了用 `yolo train resume model=runs/detect/train/weights/last.pt`。

### 训练 P2（接浅层特征）

```bash
yolo train model=configs/yolov8n-p2.yaml data=configs/scut_head.yaml pretrained=yolov8n.pt imgsz=640 epochs=100 batch=8
```

### 训练完整方法（P2 + 加权融合）

先注册自定义模块（必须在训练前执行）：

```bash
python -c "import models.register"
```

看到无报错说明注册成功。然后用脚本启动训练：

```bash
python scripts/train_fuse.py
```

> 报 `WeightedFuse not defined`：检查 `models/register.py` 是否在训练前被 import。不要改 site-packages。

## 怎么 val（评估）

在 val 集上跑 mAP：

```bash
yolo val model=runs/detect/train-3/weights/best.pt data=configs/scut_head.yaml imgsz=640
```

把 `model=` 换成你要评估的权重路径。

### fuse 模型 val

fuse 权重不能直接 `yolo val`（会变回 Concat），用脚本：

```bash
python scripts/predict_fuse.py --weights runs/detect/train-fuse-v2-4/weights/best.pt --source 数据val图片目录 --conf 0.001 --save-txt
```

然后用 ultralytics 的 val 逻辑评估，或直接看训练时保存的 results.csv。

## 怎么测速

所有 FPS 必须在同一台机器测（默认 AutoDL RTX 5090）：

```bash
python scripts/measure_speed.py --weights runs/detect/train-3/weights/best.pt --imgsz 640
```

## 预测

单张图片：

```bash
yolo predict model=runs/detect/train-3/weights/best.pt source=./test.jpg conf=0.25 save=True
```

预测整个文件夹：

```bash
yolo predict model=runs/detect/train-3/weights/best.pt source=./val图片文件夹 conf=0.25 save=True
```

### fuse 模型预测

**必须用 `scripts/predict_fuse.py`，不要直接 `yolo predict`**（直接加载会变回普通 Concat，没有加权融合）：

```bash
python scripts/predict_fuse.py --weights runs/detect/train-fuse-v2-4/weights/best.pt --source 图片或文件夹 --conf 0.25
```

结果保存在 `runs/detect/predict/`。

## 仓库结构

```
campus-tiny-det/
  README.md
  configs/
    scut_head.yaml          # 教室数据集配置
    tinyperson.yaml         # TinyPerson数据集配置
    yolov8n-p2.yaml         # P2 检测头配置
    yolov8n-p2-fuse.yaml    # P2 + 加权融合配置
  scripts/
    convert_voc_to_yolo.py  # VOC → YOLO 转换（甲）
    viz_labels.py           # 标签画框检查（乙）
    plot_box_size.py        # 框尺寸直方图（乙）
    measure_speed.py        # FPS 测速（丙）
    train_fuse.py           # 完整方法训练入口（甲）
    predict_fuse.py         # fuse模型推理脚本（甲，必须用这个）
  models/
    weighted_fuse.py        # 加权融合模块（乙）
    register.py              # 模块注册（乙）
  docs/
    实验记录.csv             # 一行实验一行，谁训谁填
    实验验证.md              # 实验设置、数据、对比表（甲）
    风险.md                  # 缺口与风险清单（甲）
    figures/                 # 截图、直方图、对比图
  .gitignore
```

## 实验记录

`docs/实验记录.csv` 表头固定，自己跑的训练自己填一行，不许代填：

```
date,method,data,epochs,imgsz,batch,map50,map50-95,params_m,flops_g,fps,weights,note
```

## 复现 val（9/29 验收）

换一个干净目录，按本文档走一遍：

1. `git clone` 仓库
2. `pip install ultralytics opencv-python matplotlib`
3. 下载 SCUT-HEAD PartA，跑 `convert_voc_to_yolo.py`
4. 改 `configs/scut_head.yaml` 的 path
5. 下载权重 `best.pt`，跑 `yolo val model=best.pt data=configs/scut_head.yaml imgsz=640`
6. 跑 `python scripts/measure_speed.py --weights best.pt --imgsz 640`

数字应与 `docs/实验记录.csv` 一致。
