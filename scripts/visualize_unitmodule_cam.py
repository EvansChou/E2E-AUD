import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm

from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.nn.modules import A2C2f, C2f, C3, DSC3k2, UnitModule, UnitEnhance


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def letterbox_image(img_rgb, imgsz, color=114):
    h, w = img_rgb.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = Image.fromarray(img_rgb).resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), (color, color, color))
    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2
    canvas.paste(resized, (pad_w, pad_h))
    return np.array(canvas)


def load_model(weights: Path, device):
    model, _ = attempt_load_one_weight(weights, device=device, inplace=True, fuse=False)
    model.eval()
    return model


def find_target_layer(model):
    preferred = (A2C2f, C2f, C3, DSC3k2)
    for m in reversed(list(model.model)):
        if isinstance(m, preferred):
            return m
    # fallback: last module before Detect
    return model.model[-2]


def preprocess(img_rgb, imgsz, device):
    img = letterbox_image(img_rgb, imgsz)
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0).to(device), img


def save_side_by_side(img_rgb, enh_rgb, out_path: Path):
    h = max(img_rgb.shape[0], enh_rgb.shape[0])
    w = img_rgb.shape[1] + enh_rgb.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: img_rgb.shape[0], : img_rgb.shape[1]] = img_rgb
    canvas[: enh_rgb.shape[0], img_rgb.shape[1] : img_rgb.shape[1] + enh_rgb.shape[1]] = enh_rgb
    Image.fromarray(canvas).save(out_path)


def find_enhancer(model):
    for m in model.model:
        if isinstance(m, (UnitModule, UnitEnhance)):
            return m
    return None


def unitmodule_enhance(model, x):
    enhancer = find_enhancer(model)
    if enhancer is None:
        raise RuntimeError("UnitModule/UnitEnhance not found in model. Please use weights trained with UnitModule.")
    with torch.no_grad():
        y = enhancer(x)
    if y.shape[1] != 3:
        raise RuntimeError(f"Enhancer output has {y.shape[1]} channels, expected 3.")
    return y


def _select_score(pred):
    if torch.is_tensor(pred):
        # try objectness + classes (last dimension)
        if pred.ndim >= 3 and pred.shape[-1] >= 6:
            return pred[..., 4:].max()
        return pred.max()
    if isinstance(pred, (list, tuple)):
        scores = []
        for p in pred:
            if torch.is_tensor(p):
                if p.ndim >= 3 and p.shape[-1] >= 6:
                    scores.append(p[..., 4:].max())
                else:
                    scores.append(p.max())
        if scores:
            return torch.stack(scores).max()
    raise RuntimeError("Unsupported prediction type for CAM.")


def grad_cam(model, x, target_layer):
    activations = []
    gradients = []

    def fwd_hook(_, __, output):
        activations.append(output)

    def bwd_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

    x = x.requires_grad_(True)
    with torch.enable_grad():
        pred = model(x)
        score = _select_score(pred)

        model.zero_grad(set_to_none=True)
        score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    act = activations[-1]  # (B, C, H, W)
    grad = gradients[-1]   # (B, C, H, W)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze(0).squeeze(0)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam ** 0.5  # boost contrasts
    return cam.detach().cpu().numpy()


def save_cam_images(img_rgb, cam, heatmap_path: Path, overlay_path: Path):
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((img_rgb.shape[1], img_rgb.shape[0]), Image.BILINEAR)
    ) / 255.0
    heatmap = (cm.get_cmap("jet")(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = (img_rgb.astype(np.float32) * 0.55 + heatmap.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
    Image.fromarray(heatmap).save(heatmap_path)
    Image.fromarray(overlay).save(overlay_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duo-root", type=str, required=True, help="Path to DUO dataset root")
    parser.add_argument("--images-dir", type=str, default="", help="Optional directory of images to visualize")
    parser.add_argument("--weights-unit", type=str, required=True, help="Path to UnitModule model weights")
    parser.add_argument("--weights-compare", type=str, required=True, nargs="+", help="Other model weights to compare")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for visualizations")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--num-images", type=int, default=6, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    seed_everything(args.seed)

    duo_root = Path(args.duo_root)
    images_dir = Path(args.images_dir) if args.images_dir else (duo_root / "images" / "test")
    imgs = list_images(images_dir)
    if not imgs:
        raise FileNotFoundError(f"No images found in {images_dir}")
    random.shuffle(imgs)
    imgs = imgs[: args.num_images]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = out_dir / "unitmodule_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    cam_dir = out_dir / "heatmaps"
    cam_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load UnitModule model
    unit_model = load_model(Path(args.weights_unit), device)

    # Save UnitModule enhanced comparisons
    for p in imgs:
        img_rgb = np.array(Image.open(p).convert("RGB"))
        x, img_lb = preprocess(img_rgb, args.imgsz, device)
        y = unitmodule_enhance(unit_model, x)
        y = y.squeeze(0).permute(1, 2, 0).cpu().numpy()
        y = (y * 255).clip(0, 255).astype(np.uint8)
        out_path = compare_dir / f"{p.stem}_orig_vs_unit.jpg"
        save_side_by_side(img_lb, y, out_path)

    # Heatmaps for UnitModule model and comparison models
    weight_list = [Path(args.weights_unit)] + [Path(w) for w in args.weights_compare]
    for w in weight_list:
        model = load_model(w, device)
        target_layer = find_target_layer(model)
        model_name = w.parent.parent.name + "_" + w.stem  # e.g., yolov8n200epoch_best
        model_out = cam_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        for p in imgs:
            img_rgb = np.array(Image.open(p).convert("RGB"))
            x, img_lb = preprocess(img_rgb, args.imgsz, device)
            cam = grad_cam(model, x, target_layer)
            heatmap_path = model_out / f"{p.stem}_heatmap.jpg"
            overlay_path = model_out / f"{p.stem}_cam.jpg"
            save_cam_images(img_lb, cam, heatmap_path, overlay_path)


if __name__ == "__main__":
    main()
