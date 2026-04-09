import argparse
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image, ImageDraw

from ultralytics.nn.modules import UnitEnhance, UnitModule
from ultralytics.nn.tasks import attempt_load_one_weight


def list_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.is_file():
        return [path]
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in exts])


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


def find_enhancer(model):
    for m in model.model:
        if isinstance(m, (UnitModule, UnitEnhance)):
            return m
    return None


def normalize_map(arr):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    denom = arr.max() + 1e-8
    return arr / denom


def heatmap_rgb(arr_2d):
    arr_2d = normalize_map(arr_2d)
    return (cm.get_cmap("jet")(arr_2d)[..., :3] * 255).astype(np.uint8)


def edge_strength_map(img_rgb):
    gray = np.asarray(Image.fromarray(img_rgb).convert("L"), dtype=np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def capture_unitmodule_outputs(enhancer, x):
    captured = {}

    def hook_lk2(_, __, output):
        captured["lk2"] = output.detach()

    def hook_tconv2(_, __, output):
        captured["t_logits"] = output.detach()

    h1 = enhancer.lk2.register_forward_hook(hook_lk2)
    h2 = enhancer.t_conv2.register_forward_hook(hook_tconv2)

    with torch.no_grad():
        enhanced = enhancer(x)

    h1.remove()
    h2.remove()

    if "t_logits" in captured:
        captured["t_map"] = torch.sigmoid(captured["t_logits"]).detach()
    captured["enhanced"] = enhanced.detach()
    return captured


def tensor_to_uint8_image(tensor):
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def resize_like(img_rgb, target_wh):
    return np.array(Image.fromarray(img_rgb).resize(target_wh, Image.BILINEAR))


def add_label(img_rgb, text):
    img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 26), fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255))
    return np.array(img)


def save_strip(images, out_path):
    height = max(im.shape[0] for im in images)
    width = sum(im.shape[1] for im in images)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for im in images:
        canvas[: im.shape[0], x : x + im.shape[1]] = im
        x += im.shape[1]
    Image.fromarray(canvas).save(out_path)


def save_channel_grid(feature_tensor, out_path, prefix, num_channels=6, tile_size=256):
    feat = feature_tensor.squeeze(0).cpu().numpy()
    num_channels = min(num_channels, feat.shape[0])
    tiles = []
    for i in range(num_channels):
        tile = heatmap_rgb(feat[i])
        tile = resize_like(tile, (tile_size, tile_size))
        tile = add_label(tile, f"{prefix} ch{i}")
        tiles.append(tile)

    if not tiles:
        return

    rows = []
    for i in range(0, len(tiles), 3):
        row_tiles = tiles[i : i + 3]
        while len(row_tiles) < 3:
            row_tiles.append(np.zeros_like(tiles[0]))
        row = np.concatenate(row_tiles, axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize UnitModule enhancement effect and internal maps.")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights trained with UnitModule/UnitEnhance.")
    parser.add_argument("--images", type=str, required=True, help="Image file or directory.")
    parser.add_argument("--out-dir", type=str, default="runs/unitmodule_effect", help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--num-images", type=int, default=4, help="Maximum number of images to process.")
    parser.add_argument("--feature-channels", type=int, default=6, help="How many feature channels to visualize.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Path(args.weights)
    images = list_images(Path(args.images))[: args.num_images]
    if not images:
        raise FileNotFoundError(f"No images found in {args.images}")

    out_dir = Path(args.out_dir)
    compare_dir = out_dir / "comparisons"
    trans_dir = out_dir / "transmission_maps"
    feat_dir = out_dir / "feature_maps"
    compare_dir.mkdir(parents=True, exist_ok=True)
    trans_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(weights, device)
    enhancer = find_enhancer(model)
    if enhancer is None:
        raise RuntimeError("Current weights do not contain UnitModule/UnitEnhance.")

    for img_path in images:
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        x, img_lb = preprocess(img_rgb, args.imgsz, device)
        outputs = capture_unitmodule_outputs(enhancer, x)

        enhanced_rgb = tensor_to_uint8_image(outputs["enhanced"])
        diff_map = np.mean(np.abs(enhanced_rgb.astype(np.float32) - img_lb.astype(np.float32)), axis=2)
        diff_rgb = heatmap_rgb(diff_map)

        orig_edge = edge_strength_map(img_lb)
        enh_edge = edge_strength_map(enhanced_rgb)
        edge_gain = heatmap_rgb(enh_edge - orig_edge)

        strip = [
            add_label(img_lb, "Original"),
            add_label(enhanced_rgb, "UnitModule Enhanced"),
            add_label(diff_rgb, "Enhancement Difference"),
            add_label(edge_gain, "Edge Gain Heatmap"),
        ]
        save_strip(strip, compare_dir / f"{img_path.stem}_comparison.jpg")

        if "t_map" in outputs:
            save_channel_grid(
                outputs["t_map"],
                trans_dir / f"{img_path.stem}_transmission.jpg",
                prefix="t",
                num_channels=3,
                tile_size=256,
            )

        if "lk2" in outputs:
            save_channel_grid(
                outputs["lk2"],
                feat_dir / f"{img_path.stem}_lk2_features.jpg",
                prefix="lk2",
                num_channels=args.feature_channels,
                tile_size=256,
            )


if __name__ == "__main__":
    main()
