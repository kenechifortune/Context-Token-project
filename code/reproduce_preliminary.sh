#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/code/context_sr_explore_torch.py"
RUNS="$ROOT/runs"
mkdir -p "$RUNS"

# Main: one training trajectory, four qualitatively OOD protocols.
python -u "$PY" \
  --Lx 3 --Ly 2 --t-steps 17 --t-max 10 \
  --d-model 12 --num-heads 3 --num-layers 1 --ctx-tokens 2 \
  --train-protocols 1 --test-protocols 4 \
  --steps 400 --smooth-weight 1e-3 --dtype float32 \
  --sr-reg 0.03 --test-family ood --seed 42 \
  --out "$RUNS/main" --save-model

# Anchor-only baseline.
python -u "$PY" \
  --Lx 3 --Ly 2 --t-steps 17 --t-max 0.4 \
  --d-model 12 --num-heads 3 --num-layers 1 --ctx-tokens 2 \
  --train-protocols 1 --test-protocols 4 \
  --steps 400 --fit-mode anchor --dtype float32 \
  --sr-reg 0.03 --test-family ood --seed 42 \
  --out "$RUNS/anchor"

# Stress family using the trained main model.
python -u "$PY" \
  --Lx 3 --Ly 2 --t-steps 17 --t-max 0.4 \
  --d-model 12 --num-heads 3 --num-layers 1 --ctx-tokens 2 \
  --train-protocols 1 --test-protocols 6 \
  --steps 0 --dtype float32 --sr-reg 0.03 --test-family stress \
  --load-model "$RUNS/main/model.pt" \
  --out "$RUNS/stress"

echo "Preliminary reproduction runs are in: $RUNS"
