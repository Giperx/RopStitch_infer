#!/usr/bin/env python3
"""Finetune RopStitch's coefficient network on driving pairs.

The alignment network stays frozen. Only CoefNetwork is optimized, with the
original mid-plane perceptual loss. Validation is not run.
"""
import argparse
import os
import random
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(ROOT, "wCoefNet", "Codes")
sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)

from loss import cal_ddmperception_loss  # noqa: E402
from network import CoefNetwork, Network, build_model  # noqa: E402
from train_drive import repair_buffers, stable_step  # noqa: E402
from torchvision import models  # noqa: E402

DATA = os.path.join(ROOT, "datasets")
ALIGN_WEIGHT = os.path.join(ROOT, "pretrained", "epoch100_model.pth")


def source(root, train_list):
    return {"root": os.path.join(DATA, root), "list": os.path.join(DATA, train_list)}


SPECS = {
    "nuscenes": {
        "epochs": 1,
        "sources": [source("nuscenes/processed_10Hz_v2", "nuscenes/processed_10Hz_v2/nuScenes_Train.txt")],
    },
    "lyft1920": {
        "epochs": 1,
        "sources": [source("lyft/lyft_train1920_3cams", "lyft/lyft_train1920_3cams/train1920.txt")],
    },
    "lyft1224": {
        "epochs": 1,
        "sources": [source("lyft/lyft_train1224_3cams", "lyft/lyft_train1224_3cams/train1224.txt")],
    },
    "lyft": {
        "epochs": 1,
        "sources": [
            source("lyft/lyft_train1920_3cams", "lyft/lyft_train1920_3cams/train1920.txt"),
            source("lyft/lyft_train1224_3cams", "lyft/lyft_train1224_3cams/train1224.txt"),
        ],
    },
    "widedrive3": {
        "epochs": 1,
        "sources": [source("WideDrive3_processed/WideDriveTrain", "WideDrive3_processed/WideDriveTrain/train.txt")],
    },
}


def read_list(path):
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def index_pairs(image_root, scenes):
    pairs = []
    missing = 0
    for scene in scenes:
        image_dir = os.path.join(image_root, scene, "images")
        if not os.path.isdir(image_dir):
            missing += 1
            continue
        cams = {}
        for name in os.listdir(image_dir):
            if not name.endswith(".jpg") or "_" not in name[:-4]:
                continue
            frame, cam = name[:-4].rsplit("_", 1)
            if cam in ("3", "4", "5"):
                cams.setdefault(frame, set()).add(cam)
        for frame, present in cams.items():
            if {"3", "4", "5"} <= present:
                size = input_size(image_root)
                pairs.append((image_root, scene, frame, 4, size))
                pairs.append((image_root, scene, frame, 3, size))
    return pairs, missing


def collect_pairs(spec):
    pairs = []
    scenes = 0
    missing = 0
    for item in spec["sources"]:
        if not os.path.isfile(item["list"]):
            raise SystemExit(f"missing train list {item['list']}")
        listed = read_list(item["list"])
        found, absent = index_pairs(item["root"], listed)
        pairs.extend(found)
        scenes += len(listed) - absent
        missing += absent
    return pairs, scenes, missing


def input_size(dataset):
    return (518, 434) if "1224" in dataset else (518, 294)


class DrivingPairs(Dataset):
    def __init__(self, pairs, dataset):
        self.pairs = pairs
        self.dataset = dataset

    def __len__(self):
        return len(self.pairs)

    def _load(self, image_root, scene, frame, cam):
        path = os.path.join(image_root, scene, "images", f"{frame}_{cam}.jpg")
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {path}")
        image = cv2.resize(image, self._size, interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        array = image.astype(np.float32)
        array = (array / 127.5) - 1.0
        return torch.from_numpy(np.transpose(array, (2, 0, 1)))

    def __getitem__(self, index):
        image_root, scene, frame, side, size = self.pairs[index]
        self._size = size
        left = self._load(image_root, scene, frame, 5)
        right = self._load(image_root, scene, frame, side)
        if random.randint(0, 1) == 1:
            left, right = right, left
        return left, right


def load_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def train_step(net, coef_net, vgg, optimizer, left, right):
    optimizer.zero_grad(set_to_none=True)
    fixed = repair_buffers(coef_net)
    if fixed:
        print(f"reset {fixed} nonfinite coef batch-norm buffers", flush=True)
    batch_out = build_model(net, coef_net, left, right)
    loss = 10 * cal_ddmperception_loss(
        vgg, left, right, batch_out["DDM_ref_list"], batch_out["DDM_tgt_list"]
    )
    if (not torch.isfinite(loss)) or float(loss) > 100:
        print(f"skip coef step: loss {float(loss)}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        return None
    loss.backward()
    torch.nn.utils.clip_grad_norm_(coef_net.parameters(), max_norm=3, norm_type=2)
    reason = stable_step(optimizer, coef_net.parameters())
    if reason:
        print(f"skip coef step: {reason}", flush=True)
        return None
    return float(loss.item())


class SameSizeBatches(Sampler):
    def __init__(self, sizes, batch_size, seed=0):
        self.sizes = list(sizes)
        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self):
        groups = {}
        for index, size in enumerate(self.sizes):
            groups.setdefault(size, []).append(index)
        rng = random.Random(self.seed)
        batches = []
        for indexes in groups.values():
            rng.shuffle(indexes)
            for start in range(0, len(indexes) - self.batch_size + 1, self.batch_size):
                batches.append(indexes[start:start + self.batch_size])
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        groups = {}
        for size in self.sizes:
            groups[size] = groups.get(size, 0) + 1
        return sum(count // self.batch_size for count in groups.values())


def main():
    parser = argparse.ArgumentParser(description="Finetune RopStitch CoefNetwork with a frozen alignment net")
    parser.add_argument("--dataset", required=True, choices=sorted(SPECS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--align", default=ALIGN_WEIGHT)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--benchmark", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=114514)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    epochs = SPECS[args.dataset]["epochs"] if args.epochs is None else args.epochs
    pairs, scenes, missing = collect_pairs(SPECS[args.dataset])
    if not pairs:
        raise SystemExit(f"no pairs for {args.dataset}")
    print(
        f"coef {args.dataset}: scenes {scenes}, missing {missing}, pairs {len(pairs)}, "
        f"frozen align {args.align}",
        flush=True,
    )
    sizes = [pair[4] for pair in pairs]
    dataset = DrivingPairs(pairs, args.dataset)
    if len(set(sizes)) > 1:
        loader = DataLoader(
            dataset,
            batch_sampler=SameSizeBatches(sizes, args.batch_size, args.seed),
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
            pin_memory=True,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network().to(device)
    net.load_state_dict(load_state(args.align))
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    coef_net = CoefNetwork().to(device).train()
    try:
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    except Exception:
        vgg = models.vgg19(pretrained=True)
    vgg = vgg.to(device).eval()
    for parameter in vgg.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(coef_net.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8)
    save_dir = args.save_dir or os.path.join(ROOT, "outputs", "finetune", args.dataset, "coef")
    os.makedirs(save_dir, exist_ok=True)

    def save(epoch, step):
        repair_buffers(coef_net)
        path = os.path.join(save_dir, f"epoch{epoch:03d}.pth")
        torch.save({"model": coef_net.state_dict(), "epoch": epoch, "glob_iter": step}, path)
        print(f"saved {path}", flush=True)

    step = 0
    limit = args.max_steps or args.benchmark or None
    for epoch in range(epochs):
        print(f"coef epoch {epoch + 1}/{epochs}", flush=True)
        for left, right in loader:
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)
            loss = train_step(net, coef_net, vgg, optimizer, left, right)
            step += 1
            if loss is None:
                if step == 1 or step % 50 == 0:
                    print(f"coef step {step} skipped", flush=True)
            elif step == 1 or step % 50 == 0:
                print(f"coef step {step} loss {loss:.4f}", flush=True)
            if limit is not None and step >= limit:
                save(epoch + 1, step)
                print(f"benchmark {step} steps", flush=True)
                return
        save(epoch + 1, step)
    print(f"coef done {args.dataset} steps {step}", flush=True)


if __name__ == "__main__":
    main()
