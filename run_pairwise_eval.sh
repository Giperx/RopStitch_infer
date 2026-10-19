#!/usr/bin/env bash
# Pairwise stitch, then full and regional HM metrics. One dataset per process.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
DATASET="${1:?dataset name}"
GPU="${2:?gpu id}"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/${DATASET}_pairwise.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== ${DATASET} gpu ${GPU} start $(date '+%F %T') ==="
source /home/test/anaconda3/etc/profile.d/conda.sh
conda activate RopStitch
export CUDA_VISIBLE_DEVICES="$GPU"
python -u stitch_driving.py --dataset "$DATASET" --all --methods pairwise
conda activate dwsplat
python -u eval_hm_dataset.py --dataset "$DATASET"
echo "=== ${DATASET} done $(date '+%F %T') ==="
