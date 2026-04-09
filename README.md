# E2E-AUD 

E2E-AUD 是一个基于 Ultralytics 框架的目标检测项目，用于 DUO 等数据集的训练与实验复现。

## Features

- 基于 `ultralytics/` 主干，兼容 YOLO 训练/推理流程
- 提供训练入口脚本 `train.py`，支持断点续训与日志输出
- 提供模型配置 `E2E-AUD.yaml`
- 包含可视化与辅助脚本（`scripts/`）

## Project Structure

```text
.
├─ ultralytics/        # 核心框架与模型实现
├─ scripts/            # 可视化与辅助脚本
├─ assets/             # 图片资源
├─ examples/           # 推理与部署示例
├─ train.py            # 训练入口
├─ E2E-AUD.yaml        # 模型配置
└─ requirements.txt    # 依赖列表
```

## Installation

### 1. Clone

```bash
git clone <your-repo-url>
cd E2E-AUD
```

### 2. Create environment

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset Preparation

`train.py` 默认读取数据集配置：

- `/root/autodl-tmp/_DUO/dataset.yaml`

建议显式传入 `--data`，并确保 `dataset.yaml` 包含 `train/val/test` 路径与类别名定义。

## Train

### Quick start

```bash
python train.py --data /path/to/dataset.yaml --model E2E-AUD.yaml --device 0
```

### Common example

```bash
python train.py \
  --data /path/to/dataset.yaml \
  --model E2E-AUD.yaml \
  --weights /path/to/pretrained.pt \
  --epochs 200 \
  --batch 14 \
  --imgsz 640 \
  --workers 6 \
  --device 0 \
  --project runs/train_duo \
  --name e2e_aud_exp
```

### Resume training

```bash
python train.py --resume runs/train_duo/e2e_aud_exp/weights/last.pt --device 0
```

## Validation / Inference

你可以直接使用 Ultralytics CLI：

```bash
yolo detect predict model=runs/train_duo/e2e_aud_exp/weights/best.pt source=path/to/image.jpg
```

## Outputs

训练结果默认保存在：

- `runs/train_duo/<experiment_name>/`

常见文件包括 `weights/best.pt`、`weights/last.pt`、训练日志与可视化图表。

## Notes

- Windows 建议将 `--workers` 设为 `0-4` 以提高稳定性
- 若显存不足，优先降低 `--batch` 或 `--imgsz`
- `requirements.txt` 包含特定 CUDA/PyTorch 版本，请按你的环境调整

## License

本项目基于 Ultralytics 生态，仓库内许可证见 `LICENSE`。
