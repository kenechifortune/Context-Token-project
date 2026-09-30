#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL=${1:-"$ROOT/runs/main/model.pt"}
if [ ! -f "$MODEL" ]; then
  echo "Missing model checkpoint: $MODEL" >&2
  echo "Run code/reproduce_main.sh first." >&2
  exit 1
fi
python -u "$ROOT/code/context_sr_explore_torch.py" \
  --Lx 3 --Ly 2 --t-steps 17 --t-max 0.4 \
  --d-model 12 --num-heads 3 --num-layers 1 --ctx-tokens 2 \
  --train-protocols 1 --test-protocols 6 --steps 0 \
  --dtype float32 --sr-reg 0.03 --test-family stress \
  --load-model "$MODEL" --out "$ROOT/runs/stress"
