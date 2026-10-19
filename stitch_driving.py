"""Stitch driving cameras 5, 4, and 3 with a white background.

chain:    cam5 with cam4, then that panorama with cam3
pairwise: cam5+cam4 and cam5+cam3 separately, then translation-blend on cam5

Metric crop, not a registration:
  vertical span is camera 5's axis-aligned bounding box;
  left and right extend to the maximum stitched image;
  that crop is then resized to the sparse wide-FOV GT size.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

# torch 1.7 cusolver cannot create a handle on this GPU. Inverse is only used
# for the small homography / TPS systems, so compute it on CPU.
_torch_inverse = torch.inverse


def _inverse_compat(tensor, *args, **kwargs):
    if torch.is_tensor(tensor) and tensor.is_cuda:
        return _torch_inverse(tensor.cpu(), *args, **kwargs).to(device=tensor.device)
    return _torch_inverse(tensor, *args, **kwargs)


torch.inverse = _inverse_compat

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(ROOT, "woCoefNet", "Codes")
sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)

import grid_res  # noqa: E402
import utils.torch_DLT as torch_DLT  # noqa: E402
import utils.torch_tps_transform as torch_tps_transform  # noqa: E402
from network import H2Mesh, Network, get_norm_mesh, get_rigid_mesh, resize_512  # noqa: E402

DATASETS = {
    "nuscenes": os.path.join(ROOT, "datasets/nuscenes/processed_10Hz_v2"),
    "lyft1920": os.path.join(ROOT, "datasets/lyft/lyft_val1920_3cams"),
    "lyft1224": os.path.join(ROOT, "datasets/lyft/lyft_val1224_3cams"),
    "ddad": os.path.join(ROOT, "datasets/ddad_process/valid"),
    "widedrive": os.path.join(ROOT, "datasets/WideDrive_processed/WideDriveVal"),
    "widedrive3": os.path.join(ROOT, "datasets/WideDrive3_processed/WideDriveVal"),
    "widedrive_exposure": os.path.join(ROOT, "datasets/WideDrive_exposure_processed/WideDriveVal"),
}
GT_SPECS = {
    "nuscenes": {
        "root": os.path.join(ROOT, "datasets/nuscenes/sparseWideFOVImages3_1554x294"),
        "size": (1554, 294),
        "suffix": "_5_sparse_wide.png",
        "val_list": os.path.join(ROOT, "datasets/nuscenes/processed_10Hz_v2/nuScenes_Val.txt"),
    },
    "lyft1920": {
        "root": os.path.join(ROOT, "datasets/lyft/1920_sparseWideFOVImages3_1554x294"),
        "size": (1554, 294),
        "suffix": "_5_sparse_wide.png",
        "val_list": os.path.join(ROOT, "datasets/lyft/lyft_val1920_3cams/lyft_val1920.txt"),
    },
    "lyft1224": {
        "root": os.path.join(ROOT, "datasets/lyft/1224_sparseWideFOVImages3_1554x434"),
        "size": (1554, 434),
        "suffix": "_5_sparse_wide.png",
        "val_list": os.path.join(ROOT, "datasets/lyft/lyft_val1224_3cams/lyft_val1224.txt"),
    },
    "widedrive3": {
        "root": os.path.join(ROOT, "datasets/WideDrive_processed/sparseWideFOVImages3_1554x294"),
        "size": (1554, 294),
        "suffix": "_2_sparse_wide.png",
        "val_list": os.path.join(ROOT, "datasets/WideDrive3_processed/WideDriveVal/val.txt"),
    },
}
WEIGHT = os.path.join(ROOT, "pretrained", "epoch100_model.pth")
GRID_H = grid_res.GRID_H + 1
GRID_W = grid_res.GRID_W + 1
WHITE = (255, 255, 255)


def find_image(scene_dir, frame, cam):
    image_dir = os.path.join(scene_dir, "images")
    text = str(frame)
    names = []
    if text.isdigit():
        number = int(text)
        for width in (len(text), 3, 4):
            names.append(f"{number:0{width}d}_{cam}.jpg")
        names.append(f"{number}_{cam}.jpg")
    else:
        names.append(f"{text}_{cam}.jpg")
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
            path = os.path.join(image_dir, name)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f"no camera {cam} frame {frame} under {image_dir}")


def load_bgr(path, max_side):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {path}")
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def to_tensor(image, device):
    array = image.astype(np.float32)
    array = (array / 127.5) - 1.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).to(device)


def blend_masked(ref, tgt, mask_ref, mask_tgt):
    mask_ref = mask_ref.astype(np.uint8)
    mask_tgt = mask_tgt.astype(np.uint8)
    weight_ref = cv2.distanceTransform(mask_ref, cv2.DIST_L2, 5).astype(np.float32) ** 2
    weight_tgt = cv2.distanceTransform(mask_tgt, cv2.DIST_L2, 5).astype(np.float32) ** 2
    weight = weight_ref + weight_tgt + 1e-6
    fused = ref * (weight_ref / weight)[..., None] + tgt * (weight_tgt / weight)[..., None]
    valid = (mask_ref | mask_tgt).astype(bool)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    fused[~valid] = 255
    return fused, valid


def crop_valid_bundle(image, valid, cam5, margin=2):
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return image, valid, cam5
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(image.shape[0], int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(image.shape[1], int(xs.max()) + margin + 1)
    return image[y0:y1, x0:x1], valid[y0:y1, x0:x1], cam5[y0:y1, x0:x1]


def crop_cam5_lr_max(image, valid, cam5):
    """Vertical: cam5 bounding box. Horizontal: maximum stitched image."""
    ys, xs = np.where(cam5)
    if len(xs) == 0:
        raise RuntimeError("camera 5 mask is empty")
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    content_ys, content_xs = np.where(valid)
    if len(content_xs) == 0:
        raise RuntimeError("stitched image is empty")
    x0 = int(content_xs.min())
    x1 = int(content_xs.max()) + 1
    y0 = max(0, min(y0, image.shape[0] - 1))
    y1 = max(y0 + 1, min(image.shape[0], y1))
    x0 = max(0, min(x0, image.shape[1] - 1))
    x1 = max(x0 + 1, min(image.shape[1], x1))
    return image[y0:y1, x0:x1]


def resize_to(image, width, height):
    if image.shape[1] == width and image.shape[0] == height:
        return image
    shrinking = image.shape[1] > width or image.shape[0] > height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def pair_meshes(net, ref_t, side_t):
    batch_size, _, img_h, img_w = ref_t.size()
    coef = torch.zeros((batch_size, 4, 1), device=ref_t.device) + 0.5
    with torch.no_grad():
        motion, motion_ref, motion_tgt = net(resize_512(ref_t), resize_512(side_t), coef, 0.5)
    motion = motion.reshape(batch_size, 4, 2)
    motion_ref = motion_ref.reshape(batch_size, GRID_H, GRID_W, 2)
    motion_tgt = motion_tgt.reshape(batch_size, GRID_H, GRID_W, 2)
    motion = torch.stack([motion[..., 0] * img_w / 512, motion[..., 1] * img_h / 512], 2)
    motion_ref = torch.stack([motion_ref[..., 0] * img_w / 512, motion_ref[..., 1] * img_h / 512], 3)
    motion_tgt = torch.stack([motion_tgt[..., 0] * img_w / 512, motion_tgt[..., 1] * img_h / 512], 3)
    src = torch.tensor(
        [[0.0, 0.0], [float(img_w), 0.0], [0.0, float(img_h)], [float(img_w), float(img_h)]],
        device=ref_t.device,
    ).unsqueeze(0).expand(batch_size, -1, -1)
    dst = src + motion
    dst_tgt = src + motion * coef
    homo = torch_DLT.Tensor_ACA_rect(src, dst)
    homo_tgt = torch_DLT.Tensor_ACA_rect(src, dst_tgt)
    homo_ref = torch.matmul(torch.inverse(homo), homo_tgt)
    rigid = get_rigid_mesh(batch_size, img_h, img_w)
    mesh_ref = H2Mesh(homo_ref, rigid) + motion_ref
    mesh_tgt = H2Mesh(homo_tgt, rigid) + motion_tgt
    return mesh_ref, mesh_tgt, rigid


def split_warp(warped):
    color = np.clip(warped[0, 0:3].detach().float().cpu().numpy().transpose(1, 2, 0) * 127.5, 0, 255)
    mask = warped[0, 3].detach().float().cpu().numpy() > 0.5
    return color, mask


def render_pair(ref_t, side_t, mesh_ref, mesh_tgt, rigid, cam5_t, max_out_height):
    xs = torch.cat([mesh_ref[..., 0].reshape(-1), mesh_tgt[..., 0].reshape(-1)])
    ys = torch.cat([mesh_ref[..., 1].reshape(-1), mesh_tgt[..., 1].reshape(-1)])
    width_min, width_max = torch.min(xs), torch.max(xs)
    height_min, height_max = torch.min(ys), torch.max(ys)
    out_width = width_max - width_min
    out_height = height_max - height_min
    out_w = int(out_width.item())
    out_h = int(out_height.item())
    if out_w < 2 or out_h < 2 or max(out_w, out_h) >= max_out_height:
        raise RuntimeError(f"warp is larger than {max_out_height}: {out_w}x{out_h}")
    mesh_trans_ref = torch.stack([mesh_ref[..., 0] - width_min, mesh_ref[..., 1] - height_min], 3)
    mesh_trans_tgt = torch.stack([mesh_tgt[..., 0] - width_min, mesh_tgt[..., 1] - height_min], 3)
    img_h, img_w = ref_t.shape[-2], ref_t.shape[-1]
    norm_rigid = get_norm_mesh(rigid, img_h, img_w)
    norm_ref = get_norm_mesh(mesh_trans_ref, out_height, out_width)
    norm_tgt = get_norm_mesh(mesh_trans_tgt, out_height, out_width)
    ones = torch.ones_like(ref_t[:, :1])
    with torch.no_grad():
        output_ref = torch_tps_transform.transformer(
            torch.cat((ref_t + 1, ones), 1), norm_ref, norm_rigid, (out_h, out_w)
        )
        output_tgt = torch_tps_transform.transformer(
            torch.cat((side_t + 1, ones), 1), norm_tgt, norm_rigid, (out_h, out_w)
        )
        warped_cam5 = torch_tps_transform.transformer(cam5_t, norm_ref, norm_rigid, (out_h, out_w))
    ref_img, ref_mask = split_warp(output_ref)
    tgt_img, tgt_mask = split_warp(output_tgt)
    fused, valid = blend_masked(ref_img, tgt_img, ref_mask, tgt_mask)
    cam5 = warped_cam5[0, 0].detach().float().cpu().numpy() > 0.5
    return fused, valid, cam5


def stitch_pair(net, reference, side, device, max_out_height, cam5_mask=None):
    if side.shape[:2] != reference.shape[:2]:
        side = cv2.resize(side, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    ref_t = to_tensor(reference, device)
    side_t = to_tensor(side, device)
    if cam5_mask is None:
        cam5_t = torch.ones((1, 1, reference.shape[0], reference.shape[1]), device=device)
    else:
        cam5_t = torch.from_numpy(np.ascontiguousarray(cam5_mask.astype(np.float32))).view(
            1, 1, cam5_mask.shape[0], cam5_mask.shape[1]
        ).to(device)
    mesh_ref, mesh_tgt, rigid = pair_meshes(net, ref_t, side_t)
    return render_pair(ref_t, side_t, mesh_ref, mesh_tgt, rigid, cam5_t, max_out_height)


def median_xy(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("empty camera-5 mask")
    return np.array([np.median(xs), np.median(ys)], np.float32)


def compose_translation(base, base_valid, base_cam5, other, other_valid, other_cam5):
    """Shift `other` so its camera 5 lines up with `base`. Translation only."""
    delta = median_xy(base_cam5) - median_xy(other_cam5)
    affine = np.array([[1.0, 0.0, float(delta[0])], [0.0, 1.0, float(delta[1])]], np.float32)
    height, width = other.shape[:2]
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], np.float32)
    warped_corners = cv2.transform(corners[None, :, :], affine)[0]
    base_corners = np.array(
        [[0, 0], [base.shape[1], 0], [0, base.shape[0]], [base.shape[1], base.shape[0]]],
        np.float32,
    )
    all_pts = np.vstack([warped_corners, base_corners])
    min_xy = np.floor(all_pts.min(axis=0)).astype(np.int32)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(np.int32)
    shift = -np.minimum(min_xy, 0)
    affine_shift = affine.copy()
    affine_shift[:, 2] += shift.astype(np.float32)
    canvas_w = int(max_xy[0] + shift[0])
    canvas_h = int(max_xy[1] + shift[1])

    def warp_mask(mask):
        painted = cv2.warpAffine(
            mask.astype(np.uint8) * 255,
            affine_shift,
            (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        return painted > 127

    other_img = cv2.warpAffine(other, affine_shift, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderValue=0)
    other_valid_w = warp_mask(other_valid)
    other_cam5_w = warp_mask(other_cam5)
    base_img = np.zeros((canvas_h, canvas_w, 3), np.float32)
    base_valid_c = np.zeros((canvas_h, canvas_w), np.uint8)
    base_cam5_c = np.zeros((canvas_h, canvas_w), np.uint8)
    x, y = int(shift[0]), int(shift[1])
    base_img[y:y + base.shape[0], x:x + base.shape[1]] = base
    base_valid_c[y:y + base.shape[0], x:x + base.shape[1]] = base_valid.astype(np.uint8)
    base_cam5_c[y:y + base.shape[0], x:x + base.shape[1]] = base_cam5.astype(np.uint8)
    fused, valid = blend_masked(base_img, other_img.astype(np.float32), base_valid_c, other_valid_w)
    cam5 = (base_cam5_c > 0) | other_cam5_w
    return fused, valid, cam5


def stitch_chain(net, images, device, max_out_height):
    fused, valid, cam5 = stitch_pair(net, images[5], images[4], device, max_out_height)
    fused, valid, cam5 = crop_valid_bundle(fused, valid, cam5)
    fused, valid, cam5 = stitch_pair(net, fused, images[3], device, max_out_height, cam5_mask=cam5)
    return crop_valid_bundle(fused, valid, cam5)


def stitch_pairwise(net, images, device, max_out_height):
    left, left_valid, left_cam5 = stitch_pair(net, images[5], images[4], device, max_out_height)
    left, left_valid, left_cam5 = crop_valid_bundle(left, left_valid, left_cam5)
    right, right_valid, right_cam5 = stitch_pair(net, images[5], images[3], device, max_out_height)
    right, right_valid, right_cam5 = crop_valid_bundle(right, right_valid, right_cam5)
    return compose_translation(left, left_valid, left_cam5, right, right_valid, right_cam5)


def load_net(weight, device):
    net = Network()
    checkpoint = torch.load(weight, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    net.load_state_dict(state)
    net.to(device).eval()
    return net


def list_frames(scene_dir, gt_rgb_dir, suffix):
    image_dir = os.path.join(scene_dir, "images")
    frames = []
    for name in sorted(os.listdir(image_dir)):
        if not name.endswith("_5.jpg"):
            continue
        frame = name[: -len("_5.jpg")]
        paths = [os.path.join(image_dir, f"{frame}_{cam}.jpg") for cam in (5, 4, 3)]
        gt_path = os.path.join(gt_rgb_dir, f"{frame}{suffix}")
        if all(os.path.exists(path) for path in paths) and os.path.exists(gt_path):
            frames.append(frame)
    return frames


def save_metric(image, valid, cam5, gt_w, gt_h, preview_dir, metric_dir, frame):
    """Write the white stitch and the GT-sized crop. Masks are not written."""
    if max(image.shape[:2]) > 65500:
        raise RuntimeError(f"canvas {image.shape[1]}x{image.shape[0]} exceeds the image limit")
    stitch = image.copy()
    stitch[~valid] = 255
    cropped = crop_cam5_lr_max(stitch, valid, cam5)
    resized = resize_to(cropped, gt_w, gt_h)
    stitch_dir = os.path.join(preview_dir, "stitch")
    os.makedirs(stitch_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    cv2.imwrite(os.path.join(stitch_dir, f"{frame}.jpg"), stitch)
    cv2.imwrite(os.path.join(metric_dir, f"{frame}.png"), resized)
    return stitch.shape, resized.shape


def read_list(path):
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def outputs_ready(preview_dir, metric_dir, frame):
    return all(
        os.path.exists(path)
        for path in (
            os.path.join(preview_dir, "stitch", f"{frame}.jpg"),
            os.path.join(metric_dir, f"{frame}.png"),
        )
    )


def main():
    parser = argparse.ArgumentParser(description="White-background cam5/4/3 stitch, resized to sparse GT")
    parser.add_argument("--dataset", required=True, choices=sorted(GT_SPECS))
    parser.add_argument("--scene", default=None)
    parser.add_argument("--all", action="store_true", help="every scene in the dataset val list")
    parser.add_argument("--frame", default=None)
    parser.add_argument("--methods", nargs="+", default=["chain", "pairwise"], choices=["chain", "pairwise"])
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--max-out-height", type=int, default=4000)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "outputs"))
    parser.add_argument("--metric-dir", default=os.path.join(ROOT, "outputs", "metric_eval"))
    parser.add_argument("--weight", default=WEIGHT)
    parser.add_argument("--gt-width", type=int, default=None)
    parser.add_argument("--gt-height", type=int, default=None)
    args = parser.parse_args()

    spec = GT_SPECS[args.dataset]
    gt_w, gt_h = spec["size"]
    if args.gt_width is not None:
        gt_w = args.gt_width
    if args.gt_height is not None:
        gt_h = args.gt_height
    if args.all:
        scenes = read_list(spec["val_list"])
    elif args.scene:
        scenes = [args.scene]
    else:
        raise SystemExit("pass --scene or --all")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_net(args.weight, device)
    print(f"{args.dataset}: {len(scenes)} scenes -> {gt_w}x{gt_h}, methods {args.methods}", flush=True)
    totals = {method: {"saved": 0, "skipped": 0, "failed": 0} for method in args.methods}
    for scene_index, scene in enumerate(scenes):
        scene_dir = os.path.join(DATASETS[args.dataset], scene)
        gt_rgb_dir = os.path.join(spec["root"], scene, "rgb")
        frames = [args.frame] if args.frame is not None else list_frames(scene_dir, gt_rgb_dir, spec["suffix"])
        if args.max_frames > 0:
            frames = frames[: args.max_frames]
        if not frames:
            print(f"  scene {scene}: no frames", flush=True)
            continue
        print(f"scene {scene_index + 1}/{len(scenes)} {scene}: {len(frames)} frames", flush=True)
        for index, frame in enumerate(frames):
            pending = []
            for method in args.methods:
                preview_dir = os.path.join(args.output_dir, args.dataset, scene, method)
                metric_dir = os.path.join(args.metric_dir, args.dataset, method, scene)
                if outputs_ready(preview_dir, metric_dir, frame):
                    totals[method]["skipped"] += 1
                else:
                    pending.append((method, preview_dir, metric_dir))
            if not pending:
                continue
            images = {cam: load_bgr(find_image(scene_dir, frame, cam), args.max_side) for cam in (5, 4, 3)}
            for method, preview_dir, metric_dir in pending:
                try:
                    if method == "chain":
                        image, valid, cam5 = stitch_chain(net, images, device, args.max_out_height)
                    else:
                        image, valid, cam5 = stitch_pairwise(net, images, device, args.max_out_height)
                    stitch_shape, metric_shape = save_metric(
                        image, valid, cam5, gt_w, gt_h, preview_dir, metric_dir, frame
                    )
                except (RuntimeError, OSError, ValueError) as exc:
                    totals[method]["failed"] += 1
                    print(f"  skip {method} {scene} {frame}: {exc}", flush=True)
                    continue
                totals[method]["saved"] += 1
                if index == 0:
                    print(
                        f"  {method} {frame}: stitch {stitch_shape[1]}x{stitch_shape[0]} "
                        f"crop {metric_shape[1]}x{metric_shape[0]}",
                        flush=True,
                    )
            if index == 0 or (index + 1) % 20 == 0 or index + 1 == len(frames):
                print(f"  {index + 1}/{len(frames)} frame {frame}", flush=True)
    for method in args.methods:
        counts = totals[method]
        print(
            f"{method}: saved {counts['saved']}, skipped {counts['skipped']}, failed {counts['failed']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
