import argparse
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image, ImageDraw

from ultralytics.nn.modules import A2C2f, C2f, C3, DSC3k2
from ultralytics.nn.tasks import attempt_load_one_weight

# ===== User-configurable defaults =====
DEFAULT_WEIGHTS = [
    r"C:\Users\Administrator\Desktop\study\yolov13-main\runs\train_duo\yolov13l\weights\best.pt",
    r"C:\Users\Administrator\Desktop\study\yolov8,11\runs\detect\yolov8n\yolov8n200epoch\weights\best.pt",
]
DEFAULT_IMAGES = r"C:\Users\Administrator\Desktop\study\yolov13_Unitmodule_HWD_LDConv\testpic"
DEFAULT_OUT_DIR = r"C:\Users\Administrator\Desktop\study\yolov13_Unitmodule_HWD_LDConv\runs\channel_avg_heatmaps"
DEFAULT_IMGSZ = 640
DEFAULT_NUM_IMAGES = 4
# ======================================


def list_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


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


def preprocess(img_rgb, imgsz, device):
    img = letterbox_image(img_rgb, imgsz)
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0).to(device), img


def load_model(weights: Path, device):
    model, _ = attempt_load_one_weight(weights, device=device, inplace=True, fuse=False)
    model.eval()
    return model


def find_target_layer(model):
    preferred = (A2C2f, C2f, C3, DSC3k2)
    for m in reversed(list(model.model)):
        if isinstance(m, preferred):
            return m
    return model.model[-2]


def normalize_map(cam):
    cam = cam.astype(np.float32)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = np.power(cam, 0.7)
    return cam


def channel_average_heatmap(model, x, target_layer):
    activations = []

    def fwd_hook(_, __, output):
        activations.append(output.detach())

    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    with torch.no_grad():
        _ = model(x)
    handle_fwd.remove()

    if not activations:
        raise RuntimeError("Failed to capture target layer activations.")

    act = activations[-1].squeeze(0).mean(dim=0).cpu().numpy()
    return normalize_map(act)


def heatmap_to_rgb(cam):
    return (cm.get_cmap("jet")(cam)[..., :3] * 255).astype(np.uint8)


def overlay_heatmap(img_rgb, cam, alpha=0.45):
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((img_rgb.shape[1], img_rgb.shape[0]), Image.BILINEAR)
    ) / 255.0
    heatmap = heatmap_to_rgb(cam_resized)
    overlay = (img_rgb.astype(np.float32) * (1.0 - alpha) + heatmap.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return heatmap, overlay


def add_label(img_rgb, text):
    img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 28), fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255))
    return np.array(img)


def save_triptych(orig, heatmap, overlay, out_path):
    items = [
        add_label(orig, "Original"),
        add_label(heatmap, "Channel-Average Heatmap"),
        add_label(overlay, "Overlay"),
    ]
    canvas = np.concatenate(items, axis=1)
    Image.fromarray(canvas).save(out_path)


def model_name_from_weights(weight_path: Path):
    parent = weight_path.parent.parent.name
    return parent.replace(" ", "_").replace(",", "_")


def main():
    parser = argparse.ArgumentParser(description="Generate channel-average heatmaps for one or more models.")
    parser.add_argument(
        "--weights",
        nargs="+",
        default=DEFAULT_WEIGHTS,
        help="One or more .pt weight paths. Defaults to YOLOv13-L and YOLOv8-N best.pt.",
    )
    parser.add_argument("--images", type=str, default=DEFAULT_IMAGES, help="Directory of images or a single image")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Inference image size")
    parser.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES, help="Max number of images to process")
    args = parser.parse_args()

    images_path = Path(args.images)
    if images_path.is_dir():
        imgs = list_images(images_path)[: args.num_images]
    else:
        imgs = [images_path]

    if not imgs:
        raise FileNotFoundError(f"No images found in {images_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for weight_str in args.weights:
        weight_path = Path(weight_str)
        model = load_model(weight_path, device)
        target_layer = find_target_layer(model)
        model_out_dir = out_dir / model_name_from_weights(weight_path)
        model_out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            x, img_lb = preprocess(img_rgb, args.imgsz, device)
            cam = channel_average_heatmap(model, x, target_layer)
            heatmap, overlay = overlay_heatmap(img_lb, cam)
            save_triptych(img_lb, heatmap, overlay, model_out_dir / f"{img_path.stem}_channel_avg.jpg")


if __name__ == "__main__":
    main()
