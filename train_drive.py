"""Finetune RopStitch on driving pairs from the released epoch100 weights.

The original loop is unchanged: two images, resized to 512, AdamW at 4e-5,
overlap photometric loss plus the mesh regularizer. Each sample is one pair,
cam5 with cam4 or cam5 with cam3. Scenes and frames are shuffled every epoch,
and the two images are swapped at random, matching TrainDataset.

Defaults: nuscenes 1 epoch, lyft1920 mixed with lyft1224 for 3 epochs,
widedrive3 3 epochs.
"""
import argparse
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(ROOT, "woCoefNet", "Codes")
sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)

from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss  # noqa: E402
from network import Network, build_model  # noqa: E402

DATA = os.path.join(ROOT, "datasets")
WEIGHT = os.path.join(ROOT, "pretrained", "epoch100_model.pth")
def source(root, train_list):
    return {"root": os.path.join(DATA, root), "list": os.path.join(DATA, train_list)}


# Default is finetuning epoch100_model.pth. Lyft mixes both resolutions.
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
        print(f"  {item['root']}: scenes {len(listed) - absent}, pairs {len(found)}", flush=True)
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


def load_init(net, path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    net.load_state_dict(state)
    print(f"init {path}", flush=True)


def stable_step(optimizer, parameters, max_growth=3.0):
    """Apply the update, then undo it if the weights blow up or become non-finite."""
    params = [p for p in parameters if p.requires_grad]
    for p in params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            optimizer.zero_grad(set_to_none=True)
            return "nonfinite grad"
    with torch.no_grad():
        before = torch.zeros((), device=params[0].device)
        saved = []
        for p in params:
            before = before + p.detach().float().pow(2).sum()
            saved.append(p.detach().clone())
        before = torch.sqrt(before)
    optimizer.step()
    with torch.no_grad():
        after = torch.zeros((), device=params[0].device)
        bad = False
        for p in params:
            if not torch.isfinite(p).all():
                bad = True
                break
            after = after + p.detach().float().pow(2).sum()
        after = torch.sqrt(after) if not bad else after
        if not bad and float(before) > 0 and float(after) > max_growth * float(before):
            bad = True
        if bad:
            for p, old in zip(params, saved):
                p.copy_(old)
            optimizer.zero_grad(set_to_none=True)
            return "weight x{:.1f}".format(float(after) / max(float(before), 1e-12))
    return ""


def repair_buffers(net):
    """Batch-norm population stats update in forward and are not rolled back with the weights."""
    fixed = 0
    with torch.no_grad():
        for module in net.modules():
            mean = getattr(module, "running_mean", None)
            var = getattr(module, "running_var", None)
            if mean is None or var is None:
                continue
            if (not torch.isfinite(mean).all()) or (not torch.isfinite(var).all()):
                mean.zero_()
                var.fill_(1)
                fixed += 1
    return fixed


def train_step(net, optimizer, left, right):
    optimizer.zero_grad(set_to_none=True)
    fixed = repair_buffers(net)
    if fixed:
        print(f"reset {fixed} nonfinite batch-norm buffers", flush=True)
    batch_out = build_model(net, left, right)
    overlap_loss = cal_lp_loss(
        left,
        batch_out["output_H"],
        batch_out["output_H_ref"],
        batch_out["output_H_tgt"],
        batch_out["output_tps_ref"],
        batch_out["output_tps_tgt"],
    )
    nonoverlap_loss = (
        inter_grid_loss(batch_out["mesh_ref"])
        + intra_grid_loss(batch_out["mesh_ref"])
        + inter_grid_loss(batch_out["mesh_tgt"])
        + intra_grid_loss(batch_out["mesh_tgt"])
    )
    total = overlap_loss + 4 * nonoverlap_loss
    if (not torch.isfinite(total)) or float(total) > 10:
        print(f"skip step: loss {float(total)}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        return None
    total.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
    reason = stable_step(optimizer, net.parameters())
    if reason:
        print(f"skip step: {reason}", flush=True)
        return None
    return total.item(), overlap_loss.item(), nonoverlap_loss.item()


def save_checkpoint(path, net, optimizer, epoch, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"model": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "glob_iter": step},
        path,
    )
    print(f"saved {path}", flush=True)


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
    parser = argparse.ArgumentParser(description="Finetune RopStitch on cam5/cam4 and cam5/cam3 pairs")
    parser.add_argument("--dataset", required=True, choices=sorted(SPECS))
    parser.add_argument("--epochs", type=int, default=None, help="default: nuscenes 1, lyft 3, widedrive3 3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--init", default=WEIGHT)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--benchmark", type=int, default=0, help="run this many steps and exit")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many samples")
    parser.add_argument("--seed", type=int, default=114514)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    spec = SPECS[args.dataset]
    epochs = spec["epochs"] if args.epochs is None else args.epochs
    if not os.path.isfile(args.init) and not args.scratch:
        raise SystemExit(f"missing init checkpoint {args.init}")

    print(f"{args.dataset}: finetune {epochs} epochs from {args.init}", flush=True)
    pairs, scenes, missing = collect_pairs(spec)
    if not pairs:
        raise SystemExit(f"no cam5/cam4/cam3 frames for {args.dataset}")
    steps_per_epoch = len(pairs) // args.batch_size
    print(
        f"{args.dataset}: scenes {scenes}, missing {missing}, pairs {len(pairs)}, "
        f"steps/epoch {steps_per_epoch} at batch {args.batch_size}",
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
    if not args.scratch:
        load_init(net, args.init)
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    save_dir = args.save_dir or os.path.join(ROOT, "outputs", "finetune", args.dataset)

    def run_steps(limit):
        losses = []
        started = time.time()
        step = 0
        while step < limit:
            for left, right in loader:
                left = left.to(device, non_blocking=True)
                right = right.to(device, non_blocking=True)
                result = train_step(net, optimizer, left, right)
                if result is None:
                    losses.append(float("nan"))
                    total = overlap = regular = float("nan")
                else:
                    total, overlap, regular = result
                    losses.append(total)
                step += 1
                if step == 1 or step % args.log_every == 0 or step == limit:
                    elapsed = time.time() - started
                    print(
                        f"step {step} loss {total:.4f} overlap {overlap:.4f} "
                        f"grid {regular:.4f} {elapsed / step:.3f}s/step",
                        flush=True,
                    )
                if step >= limit:
                    break
        elapsed = time.time() - started
        return elapsed / max(step, 1), float(np.mean(losses))

    if args.benchmark > 0:
        seconds, mean_loss = run_steps(args.benchmark)
        save_checkpoint(os.path.join(save_dir, "epoch000.pth"), net, optimizer, 0, args.benchmark)
        print(
            f"benchmark {args.benchmark} steps: {seconds:.3f}s/step, mean loss {mean_loss:.4f}",
            flush=True,
        )
        print(
            f"one epoch ~ {steps_per_epoch * seconds / 3600:.2f} h; "
            f"{epochs} epochs ~ {epochs * steps_per_epoch * seconds / 3600:.1f} h; "
            f"original 100 epochs ~ {100 * steps_per_epoch * seconds / 3600:.1f} h",
            flush=True,
        )
        return

    step = 0
    for epoch in range(epochs):
        print(f"epoch {epoch + 1}/{epochs} lr {optimizer.param_groups[0]['lr']:.8f}", flush=True)
        loss_sum = 0.0
        seen = 0
        for left, right in loader:
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)
            result = train_step(net, optimizer, left, right)
            step += 1
            if result is None:
                if step % args.log_every == 0:
                    print(f"epoch {epoch + 1} step {step} skipped", flush=True)
                if args.max_steps and step >= args.max_steps:
                    break
                continue
            total, overlap, regular = result
            seen += 1
            loss_sum += total
            if step % args.log_every == 0:
                print(
                    f"epoch {epoch + 1} step {step} loss {loss_sum / seen:.4f} "
                    f"overlap {overlap:.4f} grid {regular:.4f}",
                    flush=True,
                )
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(os.path.join(save_dir, f"step{step:07d}.pth"), net, optimizer, epoch + 1, step)
            if args.max_steps and step >= args.max_steps:
                break
        scheduler.step()
        save_checkpoint(os.path.join(save_dir, f"epoch{epoch + 1:03d}.pth"), net, optimizer, epoch + 1, step)
        if args.max_steps and step >= args.max_steps:
            break
    print(f"done {args.dataset} steps {step}", flush=True)


if __name__ == "__main__":
    main()
