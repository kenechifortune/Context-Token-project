#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/runs"
python -u "$ROOT/code/context_sr_explore_torch.py" \
  --Lx 3 --Ly 2 --t-steps 17 --t-max 0.4 \
  --d-model 12 --num-heads 3 --num-layers 1 --ctx-tokens 2 \
  --train-protocols 1 --test-protocols 4 \
  --steps 400 --smooth-weight 1e-3 --dtype float32 \
  --sr-reg 0.03 --test-family ood --seed 42 \
  --out "$ROOT/runs/main" --save-model
