"""WideDrive3 full-image LPIPS, matching metrics/widedrive_photometric.py.

Alex spatial LPIPS, inputs in [-1, 1]. The map is resized to the image and
averaged over the sparse GT mask. No histogram matching. WideDrive3 GT masks
are fully valid, so this is the Full score.
"""
import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
GT_ROOT = os.path.join(ROOT, "datasets/WideDrive_processed/sparseWideFOVImages3_1554x294")
JOBS = {
    "ropstitch": {
        "pred": os.path.join(ROOT, "outputs/metric_eval/widedrive3/pairwise"),
        "layout": "scene",
        "output": os.path.join(ROOT, "outputs/metric_eval/widedrive3/pairwise_lpips_full.txt"),
    },
    "dsfn": {
        "pred": "/home/test/LIVA/XZP/ImageStitch/DSFN_infer/outputs/metrics_wide/widedrive3",
        "layout": "crop",
        "output": "/home/test/LIVA/XZP/ImageStitch/DSFN_infer/outputs/metrics_wide/widedrive3/reports/LPIPS_full_pairwise.txt",
    },
    "unistitch": {
        "pred": "/home/test/LIVA/XZP/ImageStitch/UniStitch_infer/outputs/widedrive3",
        "layout": "crop",
        "output": "/home/test/LIVA/XZP/ImageStitch/UniStitch_infer/outputs/widedrive3_pairwise_lpips_full.txt",
    },
}


def scene_dirs(pred_root, layout):
    scenes = []
    for name in sorted(os.listdir(pred_root)):
        if layout == "scene":
            folder = os.path.join(pred_root, name)
        else:
            folder = os.path.join(pred_root, name, "pairwise", "crop")
        if os.path.isdir(folder):
            scenes.append((name, folder))
    return scenes


def load_rgb(path, size_wh=None):
    image = Image.open(path).convert("RGB")
    if size_wh is not None and image.size != tuple(size_wh):
        image = image.resize(tuple(size_wh), Image.BICUBIC)
    return np.asarray(image).astype(np.float32) / 255.0


def load_mask(path, shape_hw):
    mask = Image.open(path).convert("L")
    if mask.size != (shape_hw[1], shape_hw[0]):
        mask = mask.resize((shape_hw[1], shape_hw[0]), Image.NEAREST)
    return np.asarray(mask) > 127


def gt_paths(scene, frame):
    name = f"{frame}_2_sparse_wide.png"
    return (
        os.path.join(GT_ROOT, scene, "rgb", name),
        os.path.join(GT_ROOT, scene, "mask", name),
    )


def collect(pred_root, layout, limit):
    items = []
    for scene, folder in scene_dirs(pred_root, layout):
        names = sorted(name for name in os.listdir(folder) if name.lower().endswith((".png", ".jpg", ".jpeg")))
        if limit:
            names = names[:limit]
        for name in names:
            frame = os.path.splitext(name)[0]
            gt_path, mask_path = gt_paths(scene, frame)
            if os.path.exists(gt_path) and os.path.exists(mask_path):
                items.append((scene, frame, os.path.join(folder, name), gt_path, mask_path))
    return items


def score_batch(model, batch, device):
    preds = []
    gts = []
    masks = []
    for _scene, _frame, pred_path, gt_path, mask_path in batch:
        gt = load_rgb(gt_path)
        pred = load_rgb(pred_path, (gt.shape[1], gt.shape[0]))
        mask = load_mask(mask_path, gt.shape[:2])
        preds.append(pred)
        gts.append(gt)
        masks.append(mask)
    pred_t = torch.from_numpy(np.stack(preds)).permute(0, 3, 1, 2).to(device) * 2.0 - 1.0
    gt_t = torch.from_numpy(np.stack(gts)).permute(0, 3, 1, 2).to(device) * 2.0 - 1.0
    with torch.no_grad():
        lpips_map = model(pred_t, gt_t)
    if lpips_map.ndim == 3:
        lpips_map = lpips_map.unsqueeze(1)
    if tuple(lpips_map.shape[-2:]) != tuple(masks[0].shape):
        lpips_map = F.interpolate(lpips_map, size=masks[0].shape, mode="bilinear", align_corners=False)
    values = []
    for index, mask in enumerate(masks):
        score_map = lpips_map[index, 0].detach().float().cpu().numpy()
        valid = score_map[mask]
        values.append(float(valid.mean()) if valid.size else float("nan"))
    return values


def write_report(path, name, rows):
    by_scene = {}
    for scene, value in rows:
        by_scene.setdefault(scene, []).append(value)
    finite = [value for _scene, value in rows if np.isfinite(value)]
    lines = [
        f"=== {name} widedrive3 pairwise LPIPS-Full ===",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "LPIPS: Alex spatial map, resized to the image, mean over the sparse GT mask.",
        "No histogram matching. Same definition as widedrive_metrics Full.",
        "WideDrive3 GT masks are fully valid, so Full covers the whole canvas.",
        "Lower is better.",
        "",
        f"LPIPS-Full: {float(np.mean(finite)):.4f}",
        f"Frames: {len(finite)}",
        "",
        "Per-scene",
    ]
    for scene in sorted(by_scene):
        values = [value for value in by_scene[scene] if np.isfinite(value)]
        lines.append(f"{scene}: LPIPS-Full={float(np.mean(values)):.4f}, Frames={len(values)}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = "\n".join(lines) + "\n"
    with open(path, "w") as handle:
        handle.write(text)
    print(text, end="", flush=True)
    print(f"Wrote {path}", flush=True)
    return float(np.mean(finite)), len(finite)


def main():
    parser = argparse.ArgumentParser(description="WideDrive3 pairwise LPIPS-Full")
    parser.add_argument("--methods", nargs="+", default=["ropstitch", "dsfn", "unistitch"], choices=sorted(JOBS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="frames per scene; 0 means all")
    args = parser.parse_args()

    import lpips

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex", spatial=True).to(device)
    model.eval()
    for name in args.methods:
        job = JOBS[name]
        items = collect(job["pred"], job["layout"], args.limit)
        if not items:
            raise SystemExit(f"no frames for {name} under {job['pred']}")
        print(f"{name}: {len(items)} frames", flush=True)
        rows = []
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            scores = score_batch(model, batch, device)
            rows.extend((item[0], score) for item, score in zip(batch, scores))
            done = start + len(batch)
            if done == len(batch) or done % 200 == 0 or done == len(items):
                print(f"  {done}/{len(items)}", flush=True)
        write_report(job["output"], name, rows)


if __name__ == "__main__":
    main()
