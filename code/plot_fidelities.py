"""Plot frozen-transformer SR rollout fidelity for three training methods."""

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("anchor", "Anchor", "tab:blue"),
    ("trajectory", "Trajectory", "tab:orange"),
    ("neighborhood_walk", "Neighborhood walk", "tab:green"),
]


def load_rollout(method_dir: Path):
    result_path = method_dir / "results.npz"
    summary_path = method_dir / "summary.json"

    if not result_path.exists():
        raise FileNotFoundError(f"Missing {result_path}")

    with np.load(result_path) as data:
        keys = sorted(
            (key for key in data.files if key.startswith("gfid_")),
            key=lambda key: int(key.rsplit("_", 1)[-1]),
        )

        if not keys:
            raise ValueError(
                f"{result_path} has no gfid_* arrays. "
                "Run the SR evaluator for this method first."
            )

        fidelities = np.stack([np.asarray(data[key]) for key in keys])

    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        t_max = float(summary["config"]["t_max"])
    else:
        t_max = 1.0

    times = np.linspace(0.0, t_max, fidelities.shape[1])
    mean_fidelity = fidelities.mean(axis=0)
    return times, mean_fidelity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Root containing the d_model_XX folders",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        required=True,
        help="Model width to plot, for example 12",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path",
    )
    args = parser.parse_args()

    width_dir = args.runs_root / f"d_model_{args.d_model:02d}"
    output_path = args.output or (
        args.runs_root / f"frozen_transformer_rollout_d{args.d_model:02d}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for folder, label, color in METHODS:
        times, mean_fidelity = load_rollout(width_dir / folder)
        ax.plot(times, mean_fidelity, color=color, linewidth=2, label=label)

    ax.set_xlabel(r"Time $t$", fontsize=13)
    ax.set_ylabel("SR rollout fidelity", fontsize=13)
    ax.set_xlim(0.0, times[-1])
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
