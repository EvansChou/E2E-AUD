"""DUO dataset one-click training script for YOLOv13.

Examples (default config is runnable once dataset exists):
    python train_duo.py

Custom arguments:
    python train_duo.py --data /data/DUO/dataset.yaml \
        --model yolov13s.yaml --weights demo/yolov13s.pt --batch 8 --epochs 200 --device 0
"""

import argparse
import os
import time
from pathlib import Path

from ultralytics import YOLO

# 默认数据集路径（云端路径 /root/autodl-tmp/_DUO，可按需用 --data 覆盖）
DEFAULT_DATA = Path("/root/autodl-tmp/_DUO/dataset.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv13 DUO training entrypoint")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Dataset YAML path (train/val/test and names)",
    )
    parser.add_argument(
        "--model",
        default="yolov13_ldconv.yaml",
        help="Model config or weights (.yaml/.pt), default Nano config",
    )
    parser.add_argument(
        "--weights",
        # default="demo/yolov13n.pt",
        default="",
        help="Optional pretrained weights path; empty to train from scratch",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Resume from a checkpoint path (e.g., runs/train_duo/.../last.pt)",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch", type=int, default=14, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    parser.add_argument("--workers", type=int, default=6, help="DataLoader workers (0-4 on Windows recommended)")
    parser.add_argument(
        "--device",
        default="0",
        help="Device selection, e.g. '0', '0,1', or 'cpu'. Defaults to GPU 0",
    )
    parser.add_argument("--project", default="runs/train_duo", help="Output directory")
    parser.add_argument("--name", default="yolov13n_duo", help="Experiment name")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP)")
    parser.add_argument("--cos-lr", action="store_true", help="Enable cosine LR schedule")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Log epoch summary every N epochs (>=1) to stdout (方便在 PyCharm 远程查看)",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional log file to append epoch summaries for remote viewing (空则不写文件)",
    )
    parser.add_argument(
        "--batch-log-interval",
        type=int,
        default=0,
        help="Log intra-epoch progress every N batches (0=off, e.g. 50 -> roughly 50 batches一条)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable TQDM progress bar (PyCharm 远程控制台更清爽)",
    )
    return parser.parse_args()


def main() -> None:
    # PyCharm 远程控制台更稳定的输出
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    args = parse_args()

    # 需要时关闭 tqdm 进度条，避免远程控制台回车覆盖
    if args.no_progress:
        os.environ["YOLO_VERBOSE"] = "1"  # 仍保留日志
        os.environ["ULTRALYTICS_TQDM"] = "0"

    # Resume takes priority
    if args.resume:
        ckpt = Path(args.resume).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found for resume: {ckpt}")
        YOLO(ckpt).train(resume=True, device=args.device, workers=args.workers)
        return

    data_yaml = Path(args.data).expanduser().resolve()
    # 兼容传入目录的情况：自动拼接 dataset.yaml
    if data_yaml.is_dir():
        data_yaml = data_yaml / "dataset.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}. "
            "请确保路径指向 dataset.yaml，或传入数据集目录/文件的正确位置。"
        )

    weights_path = None
    if args.weights:
        cand = [
            Path(args.weights).expanduser(),
            Path(__file__).parent / args.weights,
            Path(__file__).parent / "demo" / Path(args.weights).name,
            Path(__file__).parent / Path(args.weights).name,  # same-name file in repo root
            Path("yolov13n.pt"),  # compatibility check default filename
        ]
        for p in cand:
            if p.is_file():
                weights_path = p
                break
        if weights_path is None:
            print(f"Warning: weights {args.weights} not found, training from scratch.")

    model = YOLO(args.model)
    if weights_path:
        model = model.load(str(weights_path))

    # 训练阶段增加 epoch 级别日志，方便远程查看
    log_fh = None
    if args.log_file:
        log_fh = Path(args.log_file).expanduser().resolve().open("a", encoding="utf-8")

    epoch_start_time = {"t": None}

    def log_msg(msg: str) -> None:
        print(msg, flush=True)
        if log_fh:
            log_fh.write(msg + "\n")
            log_fh.flush()

    def _extract_metrics(mobj):
        if hasattr(mobj, "results_dict"):
            rd = mobj.results_dict
            if callable(rd):
                return rd()
            return rd if isinstance(rd, dict) else {}
        if isinstance(mobj, dict):
            return mobj
        return {}

    def on_train_epoch_start(trainer) -> None:
        epoch_start_time["t"] = time.time()

    def on_train_epoch_end(trainer) -> None:
        # 每 N 个 epoch 打印一次摘要
        if args.log_interval <= 0:
            return
        if (trainer.epoch + 1) % args.log_interval != 0:
            return
        m = getattr(getattr(trainer, "validator", None), "metrics", None)
        mdict = _extract_metrics(m)
        p = mdict.get("metrics/precision(B)", 0.0)
        r = mdict.get("metrics/recall(B)", 0.0)
        map50 = mdict.get("metrics/mAP50(B)", 0.0)
        map5095 = mdict.get("metrics/mAP50-95(B)", 0.0)
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        loss_items = getattr(trainer, "loss_items", None)
        loss_str = (
            f"loss={loss_items.tolist()}" if loss_items is not None else f"loss={getattr(trainer, 'loss', 'N/A')}"
        )
        progress = (trainer.epoch + 1) / trainer.epochs * 100
        epoch_dur = None
        if epoch_start_time["t"]:
            epoch_dur = time.time() - epoch_start_time["t"]
        dur_str = f", epoch_time={epoch_dur:.1f}s" if epoch_dur is not None else ""

        # 计算 FPS（基于验证速度统计，ms/张）
        fps = None
        try:
            vmetrics = getattr(getattr(trainer, "validator", None), "metrics", None)
            if vmetrics and getattr(vmetrics, "speed", None):
                sd = vmetrics.speed
                latency_ms = sd.get("preprocess", 0.0) + sd.get("inference", 0.0) + sd.get("postprocess", 0.0)
                fps = 1000.0 / latency_ms if latency_ms > 0 else None
        except Exception:
            fps = None
        fps_str = f", FPS={fps:.2f}" if fps else ""

        log_msg(
            f"[epoch {trainer.epoch + 1}/{trainer.epochs} | {progress:5.1f}%] "
            f"P={p:.6f} R={r:.6f} F1={f1:.6f} mAP50={map50:.6f} mAP50-95={map5095:.6f} "
            f"{loss_str}{dur_str}{fps_str}"
        )

    def on_train_batch_end(trainer) -> None:
        if args.batch_log_interval <= 0:
            return
        nb = len(getattr(trainer, "dataloader", [])) or 0
        if nb == 0:
            return
        if (trainer.batch_i + 1) % args.batch_log_interval != 0:
            return
        progress = (trainer.batch_i + 1) / nb * 100
        log_msg(
            f"[epoch {trainer.epoch + 1}/{trainer.epochs}] "
            f"batch {trainer.batch_i + 1}/{nb} ({progress:5.1f}%)"
        )

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_batch_end", on_train_batch_end)

    metrics = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        cos_lr=args.cos_lr,
        amp=not args.no_amp,
    )
    # ---- 训练完成后打印核心指标 ----
    metrics = metrics or getattr(model, "metrics", {}) or {}
    p = float(metrics.get("metrics/precision(B)", 0.0))
    r = float(metrics.get("metrics/recall(B)", 0.0))
    map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.0))
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    # 参数量（M）
    params_m = sum(p_.numel() for p_ in model.model.parameters()) / 1e6 if hasattr(model, "model") else 0.0

    # FPS（基于验证阶段的速度统计，毫秒/图像）
    fps = None
    try:
        speed = getattr(getattr(model, "trainer", None), "validator", None)
        if speed and getattr(speed, "metrics", None) and getattr(speed.metrics, "speed", None):
            sd = speed.metrics.speed
            latency_ms = sd.get("preprocess", 0.0) + sd.get("inference", 0.0) + sd.get("postprocess", 0.0)
            fps = 1000.0 / latency_ms if latency_ms > 0 else None
    except Exception:
        fps = None

    print("\n==== 训练完成指标 ====")
    print(f"Precision (%): {p * 100:.3f}")
    print(f"Recall    (%): {r * 100:.3f}")
    print(f"F1-Score (%): {f1 * 100:.3f}")
    print(f"mAP50    (%): {map50 * 100:.3f}")
    print(f"mAP50-95 (%): {map50_95 * 100:.3f}")
    print(f"Params (M): {params_m:.3f}")
    if fps:
        print(f"FPS: {fps:.2f}")
    else:
        print("FPS: 未获取到验证阶段速度统计")
    if log_fh:
        log_fh.close()


if __name__ == "__main__":
    main()
