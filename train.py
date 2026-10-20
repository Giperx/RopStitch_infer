#!/usr/bin/env python3
"""Run RopStitch driving finetuning as alignment, then the coefficient net.

The alignment stage is woCoefNet. The coefficient stage freezes that checkpoint
and trains CoefNetwork only. Each stage uses the driving epoch count.

    python train.py --dataset widedrive3 --gpu 6 --epochs 3
"""
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELEASED = ROOT / "pretrained" / "epoch100_model.pth"


def latest_epoch(directory):
    paths = sorted(glob.glob(str(directory / "epoch*.pth")))
    return paths[-1] if paths else None


def run(command):
    print("\n########## " + Path(command[2]).name + " ##########", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description="RopStitch alignment then coefficient finetuning")
    parser.add_argument("--dataset", required=True, choices=("nuscenes", "lyft", "lyft1920", "lyft1224", "widedrive3"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--stage", choices=("both", "align", "coef"), default="both")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--align-batch", type=int, default=4)
    parser.add_argument("--coef-batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--benchmark", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    names = ("lyft1920", "lyft1224") if args.dataset == "lyft" else (args.dataset,)
    for name in names:
        args.dataset = name
        align_dir = ROOT / "outputs" / "finetune" / name
        run_dataset(args, align_dir)


def run_dataset(args, align_dir):
    if args.stage in ("both", "align"):
        command = [
            sys.executable, "-u", str(ROOT / "train_drive.py"),
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.align_batch),
            "--workers", str(args.workers),
        ]
        if args.benchmark:
            command.extend(["--benchmark", str(args.benchmark)])
        run(command)
    if args.stage in ("both", "coef"):
        align = latest_epoch(align_dir) or str(RELEASED)
        if not latest_epoch(align_dir):
            print(f"no alignment checkpoint, coefficient stage starts from {align}", flush=True)
        command = [
            sys.executable, "-u", str(ROOT / "train_coef_drive.py"),
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.coef_batch),
            "--workers", str(args.workers),
            "--align", align,
        ]
        if args.benchmark:
            command.extend(["--benchmark", str(args.benchmark)])
        run(command)


if __name__ == "__main__":
    main()
