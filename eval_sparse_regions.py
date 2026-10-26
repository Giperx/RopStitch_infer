"""L/C/R histogram-matched scores on the sparse GT mask.

Regions are the three equal widths of the wide canvas, matching nuScenes
ORIG_W = 518 inside 1554. Each region is histogram-matched on the sparse GT
mask only. L+R and L+R+C are means of those region means, not a second
pixel-weighted pool. No LPIPS.
"""
import argparse
import os
from datetime import datetime

import numpy as np

from eval_sparse_full import find_gt, frame_hm, match_gt
from PIL import Image


REGION_ORDER = ("Left", "Center", "Right")


def region_slices(width):
    if width % 3 != 0:
        raise SystemExit(f"width {width} is not divisible by 3")
    orig_w = width // 3
    return {
        "Left": (0, orig_w),
        "Center": (orig_w, orig_w * 2),
        "Right": (orig_w * 2, orig_w * 3),
    }


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
    buckets = {name: [] for name in REGION_ORDER}
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
        slices = region_slices(render.shape[1])
        scored = False
        for region, (x0, x1) in slices.items():
            row = frame_hm(render[:, x0:x1], gt[:, x0:x1], mask[:, x0:x1])
            if row is None:
                continue
            row["frame"] = frame
            buckets[region].append(row)
            scored = True
        if not scored:
            skipped += 1
    print(f"blank-side rejected: {blank}", flush=True)
    return buckets, skipped


def mean_rows(rows):
    if not rows:
        return None
    return {
        "hm_psnr": float(np.mean([row["hm_psnr"] for row in rows])),
        "hm_ssim": float(np.mean([row["hm_ssim"] for row in rows])),
        "hm_mae": float(np.mean([row["hm_mae"] for row in rows])),
        "hm_rmse": float(np.mean([row["hm_rmse"] for row in rows])),
        "n": len(rows),
    }


def mean_regions(stats_by_name):
    keys = ("hm_psnr", "hm_ssim", "hm_mae", "hm_rmse")
    return {key: float(np.mean([stats_by_name[name][key] for name in stats_by_name])) for key in keys}


def format_stats(stats):
    return (
        f"HM-PSNR: {stats['hm_psnr']:.2f}, "
        f"HM-SSIM: {stats['hm_ssim']:.4f}, "
        f"HM-MAE: {stats['hm_mae'] * 255:.2f}, "
        f"HM-RMSE: {stats['hm_rmse'] * 255:.2f}, "
        f"Frames: {stats['n']}"
    )


def main():
    parser = argparse.ArgumentParser(description="L/C/R HM metrics on the sparse GT mask")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--gt-cam", type=int, default=5)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    buckets, skipped = score_dir(args.pred_dir, args.gt_dir, args.gt_cam)
    stats = {name: mean_rows(buckets[name]) for name in REGION_ORDER}
    if any(item is None for item in stats.values()):
        missing = [name for name, item in stats.items() if item is None]
        raise SystemExit(f"no scores for {missing}")
    lr = mean_regions({name: stats[name] for name in ("Left", "Right")})
    lrc = mean_regions(stats)
    scene = args.scene or os.path.basename(os.path.normpath(args.pred_dir))
    lines = [
        "=== sparse wide-FOV regional HM metrics ===",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Scene: {scene}",
        f"Pred: {args.pred_dir}",
        f"GT: {args.gt_dir}",
        "Mask: sparse GT mask only. LPIPS is not computed.",
        "HM-SSIM is the per-pixel luminance term.",
        "L+R and L+R+C are unweighted means of the region means.",
        "MAE and RMSE are on the 0-255 scale. PSNR uses [0, 1] images.",
        f"Skipped frames: {skipped}.",
        "",
    ]
    for name in REGION_ORDER:
        lines.append(f"{name:8s}: {format_stats(stats[name])}")
    lines.append(f"L+R    : {format_stats({**lr, 'n': stats['Left']['n']})}".replace(
        f"Frames: {stats['Left']['n']}", "mean of region means"
    ))
    lines.append(f"L+R+C  : {format_stats({**lrc, 'n': stats['Center']['n']})}".replace(
        f"Frames: {stats['Center']['n']}", "mean of region means"
    ))
    text = "\n".join(lines) + "\n"
    out_path = args.output or os.path.join(os.path.dirname(args.pred_dir), f"region_hm_{scene}.txt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write(text)
    print(text, end="")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
