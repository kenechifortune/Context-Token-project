"""Compare initial fidelity for anchor and context-walk training."""

from pathlib import Path
import argparse
import multiprocessing as mp
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "code" / "context_sr_explore_torch.py"
RUNS = ROOT / "runs" / "initial_fidelity_vs_Lx"
RUNS.mkdir(parents=True, exist_ok=True)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--lx-values",
    nargs="+",
    type=int,
    default=[1, 2, 3, 4],
)

parser.add_argument(
    "--ly",
    type=int,
    default=1,
)

args = parser.parse_args()

LX_VALUES = args.lx_values
LY = args.ly


COMMON_TRAIN = [
    "--t-steps", "20",
    "--t-max", "1",
    "--d-model", "16",
    "--num-heads", "4",
    "--num-layers", "2",
    "--ctx-tokens", "4",
    "--train-protocols", "1",
    "--steps", "300",
    "--dtype", "float32",
    "--print-every", "100",
]


def run(command):
    print("\nRUNNING:", " ".join(map(str, command)))
    subprocess.run(command, check=True)


def run_experiment(job):
    lx, method, fit_mode = job

    run_name = f"{method}_Lx_{lx}_Ly_{LY}"
    output_dir = RUNS / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_command = [
        sys.executable,
        "-u",
        str(TRAIN),
        "--Lx", str(lx),
        "--Ly", str(LY),
        *COMMON_TRAIN,
        "--fit-mode", fit_mode,
        "--out", str(output_dir),
        "--save-model",
        "--device", "cpu",
    ]

    run(train_command)

    # The training script must save this file.
    fidelity_at_zero = float(
        np.load(output_dir / "initial_fidelity.npy")
    )

    print(
        f"{method}, Lx={lx}, Ly={LY}: "
        f"initial fidelity={fidelity_at_zero:.8f}"
    )

    return method, lx, fidelity_at_zero


def main():
    jobs = [
        (lx, method, fit_mode)
        for lx in LX_VALUES
        for method, fit_mode in [
            ("anchor", "anchor"),
            ("context_walk", "trajectory"),
        ]
    ]

    with mp.Pool(processes=4) as pool:
        results = pool.map(run_experiment, jobs)

    results.sort(key=lambda item: (item[1], item[0]))

    anchor_fidelities = np.array([
        fidelity
        for method, lx, fidelity in results
        if method == "anchor"
    ])

    walk_fidelities = np.array([
        fidelity
        for method, lx, fidelity in results
        if method == "context_walk"
    ])

    np.savez(
        RUNS / "initial_fidelity_data.npz",
        Lx=np.array(LX_VALUES),
        anchor_fidelity=anchor_fidelities,
        context_walk_fidelity=walk_fidelities,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        LX_VALUES,
        anchor_fidelities,
        "o-",
        linewidth=2,
        label="Anchor training",
    )

    plt.plot(
        LX_VALUES,
        walk_fidelities,
        "s--",
        linewidth=2,
        label="Context-walk training",
    )

    plt.xlabel(r"Lattice length $L_x$")
    plt.ylabel("Initial fidelity")
    plt.title(
        f"Initial fidelity versus $L_x$ for $L_y={LY}$"
    )
    plt.ylim(0, 1.02)
    plt.xticks(LX_VALUES)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RUNS / "initial_fidelity_vs_Lx.png",
        dpi=180,
    )

    plt.close()

    print(
        "\nSaved plot:",
        RUNS / "initial_fidelity_vs_Lx.png",
    )


if __name__ == "__main__":
    main()