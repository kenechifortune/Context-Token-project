"""Compare initial fidelity for anchor and trajectory/context-walk training."""

from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "code" / "context_sr_explore_torch.py"
EVALUATE = ROOT / "code" / "evaluate_context_sr_torch.py"
RUNS = ROOT / "runs" / "initial_fidelity_vs_Lx"
RUNS.mkdir(parents=True, exist_ok=True)


# Change these values to choose the lattice sizes.
LX_VALUES = np.arange(1, 7)
LY = 2


# Use modest values for the first test. Increase steps afterward.
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
    "--device", "auto",
    "--print-every", "100",
]


def run(command):
    print("\nRUNNING:", " ".join(map(str, command)))
    subprocess.run(command, check=True)


anchor_fidelities = []
walk_fidelities = []

for lx in LX_VALUES:
    for method, fit_mode in [
        ("anchor", "anchor"),
        ("context_walk", "trajectory"),
    ]:
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
        ]
        run(train_command)

        evaluate_command = [
            sys.executable,
            "-u",
            str(EVALUATE),
            "--Lx", str(lx),
            "--Ly", str(LY),
            "--t-steps", "20",
            "--t-max", "1",
            "--test-protocols", "1",
            "--test-family", "ood",
            "--integrator", "heun",
            "--sr-reg", "0.03",
            "--load-model", str(output_dir / "model.pt"),
            "--out", str(output_dir),
        ]
        run(evaluate_command)

        data = np.load(output_dir / "results.npz")
        fidelity_at_zero = float(np.asarray(data["gfid_0"])[0])

        print(
            f"{method}, Lx={lx}, Ly={LY}: "
            f"initial fidelity={fidelity_at_zero:.8f}"
        )

        if method == "anchor":
            anchor_fidelities.append(fidelity_at_zero)
        else:
            walk_fidelities.append(fidelity_at_zero)


anchor_fidelities = np.array(anchor_fidelities)
walk_fidelities = np.array(walk_fidelities)

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
plt.ylabel(r"Initial fidelity $|⟨\psi(0)|\psi_\theta(M_0)⟩|^2$")
plt.title(f"Initial fidelity versus $L_x$ for $L_y={LY}$")
plt.ylim(0, 1.02)
plt.xticks(LX_VALUES)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RUNS / "initial_fidelity_vs_Lx.png", dpi=180)
plt.close()

print("\nSaved data:", RUNS / "initial_fidelity_data.npz")
print("Saved plot:", RUNS / "initial_fidelity_vs_Lx.png")
