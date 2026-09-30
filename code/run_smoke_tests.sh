#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=${PYTHON:-python}

echo "[1/3] Independent SR math tests"
$PYTHON "$ROOT/code/test_learnMdot_sr_math.py"

echo "[2/3] Tiny end-to-end PyTorch run"
rm -rf "$ROOT/runs/smoke"
$PYTHON -u "$ROOT/code/context_sr_explore_torch.py" \
  --Lx 3 --Ly 2 --t-steps 17 --t-max 0.4 \
  --d-model 12 --num-heads 1 --num-layers 1 --ctx-tokens 2 \
  --train-protocols 1 --test-protocols 1 --steps 3 \
  --dtype float64 --sr-reg 0.03 --test-family ood \
  --out "$ROOT/runs/smoke" --save-model --print-every 1

echo "[3/3] Checkpoint-level validation"
$PYTHON "$ROOT/code/validate_torch_sr.py" --model "$ROOT/runs/smoke/model.pt"

echo "ALL SMOKE TESTS PASSED"
