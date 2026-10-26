"""Full-image histogram-matched scores on the sparse GT mask.

Same definition as 0923_synGithub/metrics/full_hm.py. No L/R/C split and no LPIPS.
Valid pixels come only from the sparse GT mask. HM-SSIM is the per-pixel
luminance term.
"""
import argparse
import os
from datetime import datetime

import numpy as np
from PIL import Image
from skimage.exposure import match_histograms


def frame_hm(render, gt, mask, match=True):
    if mask.sum() == 0:
        return None
    pred = render[mask]
    target = gt[mask]
    if match:
        pred = match_histograms(pred[np.newaxis, ...], target[np.newaxis, ...], channel_axis=-1)[0]
    matched = pred
    diff = matched - target
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    c1 = 0.01 ** 2
    ssim = float(np.mean((2 * matched * target + c1) / (matched ** 2 + target ** 2 + c1)))
    return {
        "hm_mae": mae,
        "hm_rmse": float(np.sqrt(mse)),
        "hm_psnr": float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf"),
        "hm_ssim": ssim,
    }


def find_gt(gt_dir, frame, cam):
    suffix = f"_{cam}_sparse_wide.png"
    rgb = os.path.join(gt_dir, "rgb", f"{frame}{suffix}")
    mask = os.path.join(gt_dir, "mask", f"{frame}{suffix}")
    if os.path.exists(rgb) and os.path.exists(mask):
        return rgb, mask
    return None, None


def match_gt(render, gt, mask):
    if gt.shape[:2] == render.shape[:2] and mask.shape == render.shape[:2]:
        return gt, mask
    size = (render.shape[1], render.shape[0])
    gt_img = Image.fromarray(np.clip(gt * 255.0, 0, 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255)).resize(size, Image.NEAREST)
    return np.array(gt_img).astype(np.float64) / 255.0, np.array(mask_img) > 127


def blank_side(render, fraction=0.90, level=0.98):
    white = render.min(axis=-1) >= level
    third = white.shape[1] // 3
    if third < 1:
        return True
    for start in (0, white.shape[1] - third):
        if float(white[:, start:start + third].mean()) >= fraction:
            return True
    return False


def score_dir(pred_dir, gt_dir, cam):
    names = sorted(name for name in os.listdir(pred_dir) if name.lower().endswith(".png"))
    rows = []
    skipped = 0
    blank = 0
    for name in names:
        frame = os.path.splitext(name)[0]
        gt_path, mask_path = find_gt(gt_dir, frame, cam)
        if gt_path is None:
            skipped += 1
            continue
        render = np.array(Image.open(os.path.join(pred_dir, name)).convert("RGB")).astype(np.float64) / 255.0
        if blank_side(render):
            blank += 1
            print(f"reject {frame}: blank side", flush=True)
            continue
        gt = np.array(Image.open(gt_path).convert("RGB")).astype(np.float64) / 255.0
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        gt, mask = match_gt(render, gt, mask)
        row = frame_hm(render, gt, mask)
        if row is None:
            skipped += 1
            continue
        row["frame"] = frame
        rows.append(row)
    judged = len(rows) + blank
    print(
        f"Success: {len(rows)}/{judged} ({(len(rows) / judged if judged else 0):.1%}); "
        f"blank-side rejected: {blank}",
        flush=True,
    )
    return rows, skipped


def mean_rows(rows):
    return {
        "hm_psnr": float(np.mean([row["hm_psnr"] for row in rows])),
        "hm_ssim": float(np.mean([row["hm_ssim"] for row in rows])),
        "hm_mae": float(np.mean([row["hm_mae"] for row in rows])),
        "hm_rmse": float(np.mean([row["hm_rmse"] for row in rows])),
        "n": len(rows),
    }


def format_full(stats):
    return "\n".join([
        f"HM-PSNR-Full: {stats['hm_psnr']:.2f}",
        f"HM-SSIM-Full: {stats['hm_ssim']:.4f}",
        f"HM-MAE-Full: {stats['hm_mae'] * 255:.2f}",
        f"HM-RMSE-Full: {stats['hm_rmse'] * 255:.2f}",
        f"Frames: {stats['n']}",
    ])


def main():
    parser = argparse.ArgumentParser(description="Full-image HM metrics on the sparse GT mask")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--gt-cam", type=int, default=5)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows, skipped = score_dir(args.pred_dir, args.gt_dir, args.gt_cam)
    if not rows:
        raise SystemExit(f"no scored frames in {args.pred_dir}")
    stats = mean_rows(rows)
    scene = args.scene or os.path.basename(os.path.normpath(args.pred_dir))
    lines = [
        "=== sparse wide-FOV full-image HM metrics ===",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Scene: {scene}",
        f"Pred: {args.pred_dir}",
        f"GT: {args.gt_dir}",
        "Mask: sparse GT mask only. Regions are not split. LPIPS is not computed.",
        "HM-SSIM is the per-pixel luminance term used by the regional HM reports.",
        "MAE and RMSE are on the 0-255 scale. PSNR uses [0, 1] images.",
        f"Skipped frames: {skipped}.",
        "",
        format_full(stats),
    ]
    text = "\n".join(lines) + "\n"
    out_path = args.output or os.path.join(os.path.dirname(args.pred_dir), f"full_hm_{scene}.txt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write(text)
    print(text, end="")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
