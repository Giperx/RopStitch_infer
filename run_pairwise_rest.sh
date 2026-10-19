#!/usr/bin/env bash
# Lyft 1920, Lyft 1224, WideDrive3, then the pooled Lyft HM report.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$ROOT/run_pairwise_eval.sh" lyft1920 1
"$ROOT/run_pairwise_eval.sh" lyft1224 1
"$ROOT/run_pairwise_eval.sh" widedrive3 1
source /home/test/anaconda3/etc/profile.d/conda.sh
conda activate dwsplat
cd "$ROOT"
python -u eval_hm_dataset.py --dataset lyft
echo "=== lyft pool done $(date '+%F %T') ==="
