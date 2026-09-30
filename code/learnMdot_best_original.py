import os

# Honor an existing CUDA selection from the shell; default to GPU 1.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import functools
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from numpy.polynomial.legendre import Legendre
from jax import lax, random
from jax.scipy.special import logsumexp
from tqdm import tqdm


Array = jax.Array
PRNGKey = Array


# ============================================================
# 0) CLI
# ============================================================
CLI_PARSER = argparse.ArgumentParser()
FIX_M0_GROUP = CLI_PARSER.add_mutually_exclusive_group()
FIX_M0_GROUP.add_argument(
    "--fix-M0",
    dest="fix_M0",
    action="store_true",
    help="Use the default fixed random unit-norm M0 tokens as the initial context for integrating Mdot(t).",
)
FIX_M0_GROUP.add_argument(
    "--canonical-M0",
    dest="canonical_M0",
    action="store_true",
    help="Use fixed canonical-basis M0 tokens as the initial context for integrating Mdot(t).",
)
FIX_M0_GROUP.add_argument(
    "--diff-M",
    dest="diff_M",
    action="store_true",
    help="Use a learnable shared M0 instead of the default fixed M0.",
)
CLI_PARSER.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Resume from a checkpoint written by this script. Params-only pickles are also accepted.",
)
CLI_PARSER.add_argument(
    "--checkpoint-path",
    type=str,
    default="zihaoequinox_latest_checkpoint.pkl",
    help="Path where the latest resumable training checkpoint is written.",
)
CLI_PARSER.add_argument("--Lx", type=int, default=4, help="Number of lattice sites along x.")
CLI_PARSER.add_argument("--Ly", type=int, default=4, help="Number of lattice sites along y.")
CLI_PARSER.add_argument(
    "--site-order",
    type=str,
    choices=("row_major", "col_major", "snake"),
    default="snake",
    help="1D site ordering for tokenization and lattice edges.",
)
CLI_PARSER.add_argument("--t-steps", type=int, default=100, help="Number of time steps in each protocol.")
CLI_PARSER.add_argument("--t-max", type=float, default=0.5, help="Physical time horizon of each protocol.")
CLI_PARSER.add_argument("--batch-size", type=int, default=6, help="Number of protocols per batch.")
CLI_PARSER.add_argument("--num-time-points", type=int, default=4, help="Number of sampled time points per protocol.")
CLI_PARSER.add_argument("--num-mc-samples", type=int, default=128, help="Number of Monte Carlo spin samples per selected time point.")
CLI_PARSER.add_argument("--d-model", type=int, default=96, help="Model embedding dimension.")
CLI_PARSER.add_argument("--fno-width", type=int, default=128, help="Width of the FNO backbone.")
CLI_PARSER.add_argument("--num-layers", type=int, default=3, help="Number of transformer decoder layers.")
CLI_PARSER.add_argument("--ctx-tokens", type=int, default=4, help="Number of context tokens output by the FNO.")
CLI_PARSER.add_argument(
    "--integration-rule",
    type=str,
    choices=("trap", "simpson"),
    default="simpson",
    help="Numerical rule used to reconstruct M(t) from sampled Mdot(t).",
)
CLI_PARSER.add_argument("--anchor-weight", type=float, default=1.0, help="Weight of the anchor term in the physics loss.")
CLI_PARSER.add_argument(
    "--score-function-weight",
    type=float,
    default=1.0,
    help="Optional score-function surrogate term added to the optimization objective.",
)
CLI_PARSER.add_argument(
    "--phase-mode",
    type=str,
    choices=("raw", "softsign"),
    default="softsign",
    help="Phase parameterization for the decoder head.",
)
CLI_PARSER.add_argument(
    "--gauge-centering",
    type=str,
    choices=("global", "per_time"),
    default="global",
    help="Center the Schrödinger residual globally over all sampled points or separately for each sampled time.",
)
CLI_PARSER.add_argument(
    "--self-kv-cache",
    action="store_true",
    help="Use exact incremental self-attention KV caching during autoregressive sampling.",
)
CLI_PARSER.add_argument("--steps", type=int, default=40000, help="Number of TDVP training steps.")
CLI_PARSER.add_argument("--anchor-pretrain-steps", type=int, default=500, help="Number of anchor-only warmup steps.")
CLI_PARSER.add_argument("--benchmark-every", type=int, default=5000, help="Run the exact benchmark every this many TDVP steps. Set to 0 to disable.")
CLI_PARSER.add_argument("--benchmark-key", type=int, default=14850, help="PRNG seed used to generate the fixed benchmark driving protocol.")
CLI_PARSER.add_argument("--print-every", type=int, default=50, help="Print running loss every this many TDVP steps.")
CLI_ARGS = CLI_PARSER.parse_args()

print(f"JAX is using: {jax.devices()}")


# ============================================================
# 1) Hyperparameters
# ============================================================
LX = CLI_ARGS.Lx
LY = CLI_ARGS.Ly
SITE_ORDER = CLI_ARGS.site_order
N_SPINS = LX * LY

START_LR = 8e-4
MIN_LR = 5e-6
DECAY_RATE = 0.95
DECAY_STEPS = 2000

J_zz = 1.0
omega_factor = 10

T_MAX = float(CLI_ARGS.t_max)
T_STEPS = CLI_ARGS.t_steps
DT = T_MAX / (T_STEPS - 1)

BATCH_SIZE = CLI_ARGS.batch_size
K = CLI_ARGS.num_time_points
M_SAMPLES = CLI_ARGS.num_mc_samples

EMBED_DIM = CLI_ARGS.d_model
FNO_WIDTH = CLI_ARGS.fno_width
FNO_MODES = 64
NUM_HEADS = 8
NUM_LAYERS = CLI_ARGS.num_layers
CTX_TOKENS = CLI_ARGS.ctx_tokens
SEED = 42

n_modes = 10
INTEGRATION_RULE = CLI_ARGS.integration_rule
anchor_weight = CLI_ARGS.anchor_weight
SCORE_FUNCTION_WEIGHT = float(CLI_ARGS.score_function_weight)
init_state = "plus"
PHASE_MODE = CLI_ARGS.phase_mode
GAUGE_CENTERING = CLI_ARGS.gauge_centering
ATTENTION_KERNEL = "matmul"
IN_FIELDS = 2
FC_LEGENDRE_D = 4
FC_ADDITIONAL_PTS = 30
SELF_KV_CACHE = CLI_ARGS.self_kv_cache

if CLI_ARGS.canonical_M0:
    FIX_M0_MODE = "canonical_basis"
elif CLI_ARGS.diff_M:
    FIX_M0_MODE = "learnable_diff"
else:
    FIX_M0_MODE = "random_unit_rows"
FIX_M0_SEED = SEED
RESUME_PATH = Path(CLI_ARGS.resume).expanduser().resolve() if CLI_ARGS.resume is not None else None
CHECKPOINT_PATH = Path(CLI_ARGS.checkpoint_path).expanduser().resolve()


def lattice_visit_order(Lx: int, Ly: int, site_order: str = "row_major"):
    coords = []
    if site_order == "row_major":
        for y in range(Ly):
            for x in range(Lx):
                coords.append((x, y))
        return coords

    if site_order == "col_major":
        for x in range(Lx):
            for y in range(Ly):
                coords.append((x, y))
        return coords

    if site_order == "snake":
        if Lx >= Ly:
            for y in range(Ly):
                xs = range(Lx) if y % 2 == 0 else range(Lx - 1, -1, -1)
                for x in xs:
                    coords.append((x, y))
        else:
            for x in range(Lx):
                ys = range(Ly) if x % 2 == 0 else range(Ly - 1, -1, -1)
                for y in ys:
                    coords.append((x, y))
        return coords

    raise ValueError(f"Unknown site_order: {site_order}")


def make_lattice_bonds(Lx: int, Ly: int, site_order: str = "row_major"):
    coord_to_idx = {
        coord: idx for idx, coord in enumerate(lattice_visit_order(Lx, Ly, site_order))
    }
    bonds_i = []
    bonds_j = []
    for y_c in range(Ly):
        for x_c in range(Lx):
            i = coord_to_idx[(x_c, y_c)]
            if x_c + 1 < Lx:
                bonds_i.append(i)
                bonds_j.append(coord_to_idx[(x_c + 1, y_c)])
            if y_c + 1 < Ly:
                bonds_i.append(i)
                bonds_j.append(coord_to_idx[(x_c, y_c + 1)])
    return bonds_i, bonds_j


# ============================================================
# 2) Neighbor indices
# ============================================================
bonds_i_np, bonds_j_np = make_lattice_bonds(LX, LY, SITE_ORDER)
bonds_i_np = np.array(bonds_i_np, dtype=np.int32)
bonds_j_np = np.array(bonds_j_np, dtype=np.int32)
N_BONDS = len(bonds_i_np)

BOND_I = jnp.array(bonds_i_np, dtype=jnp.int32)
BOND_J = jnp.array(bonds_j_np, dtype=jnp.int32)


# ============================================================
# 3) Data & Physics Utils
# ============================================================
def generate_grf_trajectories(key, batch_size, t_steps, t_max, n_modes=n_modes):
    kx1, kx2, kx3, kz1, kz2, kz3 = random.split(key, 6)

    t = jnp.linspace(0.0, t_max, t_steps, dtype=jnp.float32)
    k = jnp.arange(1, n_modes + 1, dtype=jnp.float32)
    omega = k * (jnp.pi / t_max) * omega_factor
    amp_decay = 1.0 / (k**1.5)

    amps_x = random.uniform(kx1, (batch_size, n_modes), minval=-0.50, maxval=0.50) * amp_decay[None, :]
    phases_x = random.uniform(kx2, (batch_size, n_modes), minval=0.0, maxval=2 * jnp.pi)
    hx0 = random.uniform(kx3, (batch_size, 1), minval=-0.05, maxval=0.05) + 1.0
    args_x = omega[None, :, None] * t[None, None, :] + phases_x[:, :, None]
    sig_x = jnp.sum(amps_x[:, :, None] * jnp.sin(args_x), axis=1)
    hx = hx0 + sig_x

    amps_z = random.uniform(kz1, (batch_size, n_modes), minval=-0.05, maxval=0.05) * amp_decay[None, :]
    phases_z = random.uniform(kz2, (batch_size, n_modes), minval=0.0, maxval=2 * jnp.pi)
    hz0 = random.uniform(kz3, (batch_size, 1), minval=-0.05, maxval=0.05) + 0.00
    args_z = omega[None, :, None] * t[None, None, :] + phases_z[:, :, None]
    sig_z = jnp.sum(amps_z[:, :, None] * jnp.sin(args_z), axis=1)
    hz = hz0 + sig_z

    return jnp.stack([hx, hz], axis=-1)


def make_random_initial_ctx(seed: int, ctx_tokens: int, embed_dim: int) -> Array:
    key = random.PRNGKey(seed)
    initial_ctx = random.normal(key, (1, ctx_tokens, embed_dim), dtype=jnp.float32)
    norms = jnp.linalg.norm(initial_ctx, axis=-1, keepdims=True)
    return initial_ctx / jnp.clip(norms, a_min=1e-8)


def make_canonical_initial_ctx(ctx_tokens: int, embed_dim: int) -> Array:
    if ctx_tokens > embed_dim:
        raise ValueError(
            f"Cannot build canonical M0 with ctx_tokens={ctx_tokens} "
            f"and embed_dim={embed_dim}: need ctx_tokens <= embed_dim."
        )
    initial_ctx = jnp.zeros((1, ctx_tokens, embed_dim), dtype=jnp.float32)
    token_ids = jnp.arange(ctx_tokens, dtype=jnp.int32)
    return initial_ctx.at[0, token_ids, token_ids].set(1.0)


def integrate_context_velocity(initial_ctx: Array, mdot: Array, dt: float) -> Array:
    if mdot.ndim != 4:
        raise ValueError(f"Expected mdot with shape (B, T, C, D), got {mdot.shape}")

    if mdot.shape[1] == 0:
        raise ValueError("Cannot integrate an empty time axis for mdot.")

    m0 = initial_ctx[:, None, :, :]
    if mdot.shape[1] == 1:
        return m0

    if INTEGRATION_RULE == "trap":
        # Reconstruct the context trajectory from the predicted velocity with a stable trapezoidal rule.
        increments = 0.5 * dt * (mdot[:, 1:, :, :] + mdot[:, :-1, :, :])
        cumulative = jnp.cumsum(increments, axis=1)
        return jnp.concatenate([m0, m0 + cumulative], axis=1)

    if INTEGRATION_RULE != "simpson":
        raise ValueError(f"Unknown integration rule: {INTEGRATION_RULE}")

    flat_mdot = mdot.reshape(mdot.shape[0], mdot.shape[1], -1)
    cumulative_flat = jnp.zeros_like(flat_mdot)

    if mdot.shape[1] == 2:
        cumulative_flat = cumulative_flat.at[:, 1].set(0.5 * dt * (flat_mdot[:, 0] + flat_mdot[:, 1]))
    else:
        # First step from quadratic interpolation through the first three samples.
        first_step = dt * (5.0 * flat_mdot[:, 0] + 8.0 * flat_mdot[:, 1] - flat_mdot[:, 2]) / 12.0
        cumulative_flat = cumulative_flat.at[:, 1].set(first_step)

        for i in range(2, mdot.shape[1]):
            if i % 2 == 0:
                # Standard Simpson contribution over the last two intervals.
                step_val = (
                    cumulative_flat[:, i - 2]
                    + dt * (flat_mdot[:, i - 2] + 4.0 * flat_mdot[:, i - 1] + flat_mdot[:, i]) / 3.0
                )
            else:
                # One-step quadratic correction for odd grid points.
                step_val = (
                    cumulative_flat[:, i - 1]
                    + dt * (-flat_mdot[:, i - 2] + 8.0 * flat_mdot[:, i - 1] + 5.0 * flat_mdot[:, i]) / 12.0
                )
            cumulative_flat = cumulative_flat.at[:, i].set(step_val)

    cumulative = cumulative_flat.reshape(mdot.shape)
    return m0 + cumulative


def arrays_only(model):
    return eqx.filter(model, eqx.is_array)


def combine_model(arrays, template):
    static = eqx.partition(template, eqx.is_array)[1]
    return eqx.combine(arrays, static)


def save_training_checkpoint(
    path: Path,
    *,
    step: int,
    model,
    opt_state,
    rng,
    best_loss: float,
    best_model,
    loss_hist,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "zihaoequinox_checkpoint_v1",
        "step": int(step),
        "model": jax.device_get(arrays_only(model)),
        "opt_state": jax.device_get(opt_state),
        "rng": np.asarray(jax.device_get(rng)),
        "best_loss": float(best_loss),
        "best_model": jax.device_get(arrays_only(best_model)),
        "loss_hist": [float(x) for x in loss_hist],
        "config": {
            "fixed_M0_mode": FIX_M0_MODE,
            "init_state": init_state,
            "score_function_weight": SCORE_FUNCTION_WEIGHT,
            "phase_mode": PHASE_MODE,
            "gauge_centering": GAUGE_CENTERING,
            "attention_kernel": ATTENTION_KERNEL,
            "site_order": SITE_ORDER,
            "self_kv_cache": SELF_KV_CACHE,
            "context_dynamics": "direct_fno_mdot",
            "integration_rule": INTEGRATION_RULE,
            "T_STEPS": T_STEPS,
            "BATCH_SIZE": BATCH_SIZE,
            "K": K,
            "M_SAMPLES": M_SAMPLES,
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_resume_payload(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_benchmark_series(output_dir: Path, step: int, result: dict) -> None:
    series = result["series"]
    np.savez(
        output_dir / f"benchmark_step_{step:05d}.npz",
        fids=np.asarray(series["fids"], dtype=np.float32),
        mx_nqs=np.asarray(series["mx_nqs"], dtype=np.float32),
        mx_exact=np.asarray(series["mx_exact"], dtype=np.float32),
        zz_nqs=np.asarray(series["zz_nqs"], dtype=np.float32),
        zz_exact=np.asarray(series["zz_exact"], dtype=np.float32),
    )


@jax.jit
def compute_observables_exact(psi: Array):
    dim = psi.shape[0]
    n_spins = int(np.log2(dim))
    idx = jnp.arange(dim, dtype=jnp.int32)
    pop = jax.lax.population_count(idx.astype(jnp.uint32)).astype(jnp.float32)
    mz = (n_spins - 2.0 * pop) / float(n_spins)
    ez = jnp.real(jnp.vdot(psi, mz.astype(psi.dtype) * psi))
    psi_tensor = psi.reshape((2,) * n_spins)
    sx_psi = jnp.zeros_like(psi)
    for i in range(n_spins):
        sx_psi = sx_psi + jnp.flip(psi_tensor, axis=n_spins - 1 - i).reshape(-1)
    ex = jnp.real(jnp.vdot(psi, sx_psi)) / float(n_spins)
    return ez, ex


@jax.jit
def apply_H_exact(psi: Array, h_x_val: Array, h_z_val: Array):
    dim = psi.shape[0]
    idx = jnp.arange(dim, dtype=jnp.int32)

    e_diag = jnp.zeros((dim,), dtype=jnp.float32)
    z_sum = jnp.zeros((dim,), dtype=jnp.float32)

    for bi, bj in zip(bonds_i_np.tolist(), bonds_j_np.tolist()):
        sbi = (idx >> bi) & 1
        sbj = (idx >> bj) & 1
        zi = (1 - 2 * sbi).astype(jnp.float32)
        zj = (1 - 2 * sbj).astype(jnp.float32)
        e_diag = e_diag + (-J_zz) * (zi * zj)

    for i in range(N_SPINS):
        bi = (idx >> i) & 1
        zi = (1 - 2 * bi).astype(jnp.float32)
        z_sum = z_sum + zi

    e_diag = e_diag + (-h_z_val) * z_sum
    res = e_diag.astype(psi.real.dtype) * psi

    psi_tensor = psi.reshape((2,) * N_SPINS)
    for i in range(N_SPINS):
        flipped = jnp.flip(psi_tensor, axis=N_SPINS - 1 - i).reshape(-1)
        res = res + (-h_x_val) * flipped

    return res


def run_benchmark(model, h_fields_1d, show_plot=False, plot_path: Optional[str] = None):
    if show_plot:
        print("\n--- Running Benchmark (Exact vs NOQS) ---")

    h_input = h_fields_1d[None, :, :]
    m_traj, _ = model.encode_field(h_input)
    m_traj = m_traj[0]

    dim = 2**N_SPINS
    idx = jnp.arange(dim, dtype=jnp.uint32)
    all_spins = ((idx[:, None] & (jnp.uint32(1) << jnp.arange(N_SPINS, dtype=jnp.uint32))) > 0).astype(jnp.int32)

    zz_diag = jnp.zeros((dim,), dtype=jnp.float32)
    for bi, bj in zip(bonds_i_np.tolist(), bonds_j_np.tolist()):
        sbi = (idx >> bi) & 1
        sbj = (idx >> bj) & 1
        zi = 1.0 - 2.0 * sbi.astype(jnp.float32)
        zj = 1.0 - 2.0 * sbj.astype(jnp.float32)
        zz_diag = zz_diag + (zi * zj)
    avg_zz_op = zz_diag / float(N_BONDS)

    if init_state == "plus":
        psi_exact = jnp.ones((dim,), dtype=jnp.complex64) / jnp.sqrt(dim)
    elif init_state == "up":
        psi_exact = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
    else:
        raise ValueError(f"Unknown init_state: {init_state}")

    fids, mx_nqs, zz_nqs = [], [], []
    mx_exact, zz_exact = [], []

    for i in range(T_STEPS):
        h_x_curr = h_fields_1d[i, 0]
        h_z_curr = h_fields_1d[i, 1]

        _, x_ex = compute_observables_exact(psi_exact)
        mx_exact.append(float(x_ex))
        zz_exact.append(float(jnp.real(jnp.vdot(psi_exact, avg_zz_op * psi_exact))))

        if i < T_STEPS - 1:
            dt = DT
            h_x_next = h_fields_1d[i + 1, 0]
            h_z_next = h_fields_1d[i + 1, 1]
            h_x_mid = 0.5 * (h_x_curr + h_x_next)
            h_z_mid = 0.5 * (h_z_curr + h_z_next)
            k1 = -1j * apply_H_exact(psi_exact, h_x_curr, h_z_curr)
            k2 = -1j * apply_H_exact(psi_exact + 0.5 * dt * k1, h_x_mid, h_z_mid)
            k3 = -1j * apply_H_exact(psi_exact + 0.5 * dt * k2, h_x_mid, h_z_mid)
            k4 = -1j * apply_H_exact(psi_exact + dt * k3, h_x_next, h_z_next)
            psi_exact = psi_exact + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            psi_exact = psi_exact / jnp.linalg.norm(psi_exact)

        m_i = m_traj[i]
        m_batch = jnp.broadcast_to(m_i[None, :, :], (dim, m_i.shape[0], m_i.shape[1]))
        log_psi = model.log_psi_from_tokens(all_spins, m_batch)
        log_norm = 0.5 * logsumexp(2.0 * jnp.real(log_psi))
        psi_nqs = jnp.exp(log_psi - log_norm)

        ov = jnp.vdot(psi_exact, psi_nqs)
        fids.append(float(jnp.abs(ov) ** 2))
        _, x_nqs = compute_observables_exact(psi_nqs)
        mx_nqs.append(float(x_nqs))
        zz_nqs.append(float(jnp.real(jnp.vdot(psi_nqs, avg_zz_op * psi_nqs))))

    summary = {
        "avg_fidelity": float(np.mean(fids)),
        "final_fidelity": float(fids[-1]),
        "best_fidelity": float(np.max(fids)),
        "mx_mae": float(np.mean(np.abs(np.asarray(mx_nqs) - np.asarray(mx_exact)))),
        "zz_mae": float(np.mean(np.abs(np.asarray(zz_nqs) - np.asarray(zz_exact)))),
    }

    if show_plot:
        t_axis = np.linspace(0.0, T_MAX, len(mx_nqs))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        ax1.plot(t_axis, fids, color="black")
        ax1.set_title(f"Fidelity (Avg: {summary['avg_fidelity']:.4f})")

        ax2.plot(t_axis, mx_nqs, label="NOQS")
        ax2.plot(t_axis, mx_exact, "--", color="black", label="GT")
        ax2.set_title("<X> Mag")

        ax3.plot(t_axis, zz_nqs, label="NOQS")
        ax3.plot(t_axis, zz_exact, "--", color="black", label="GT")
        ax3.set_title("<ZZ> Corr")

        plt.tight_layout()
        if plot_path is None:
            plot_path = "benchmark.png"
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved benchmark plot to: {plot_path.resolve()}")

    return {
        "summary": summary,
        "series": {
            "fids": fids,
            "mx_nqs": mx_nqs,
            "mx_exact": mx_exact,
            "zz_nqs": zz_nqs,
            "zz_exact": zz_exact,
        },
    }


# ============================================================
# 4) Pure FNO Architecture
# ============================================================
def _fc_legendre_boundary_matrix(fc_degree: int, n_additional_pts: int) -> np.ndarray:
    if fc_degree <= 0:
        raise ValueError(f"fc_degree must be positive, got {fc_degree}")
    if n_additional_pts < 0:
        raise ValueError(f"n_additional_pts must be non-negative, got {n_additional_pts}")
    if n_additional_pts % 2 != 0:
        raise ValueError(
            f"Legendre Fourier continuation expects an even n_additional_pts, got {n_additional_pts}"
        )

    total_points = 2 * fc_degree + n_additional_pts
    full_grid = np.linspace(-1.0, 1.0, total_points, dtype=np.float64)
    fit_grid = np.concatenate((full_grid[:fc_degree], full_grid[-fc_degree:]), axis=0)
    extension_grid = full_grid[fc_degree:-fc_degree]

    eye = np.eye(2 * fc_degree, dtype=np.float64)
    polynomials = [
        np.sqrt((2 * j + 1) / 2.0) * Legendre(eye[j, :])
        for j in range(2 * fc_degree)
    ]

    x_fit = np.stack([poly(fit_grid) for poly in polynomials], axis=1)
    q_ext = np.stack([poly(extension_grid) for poly in polynomials], axis=1)
    return q_ext @ np.linalg.pinv(x_fit, rcond=1e-15)


@functools.lru_cache(maxsize=None)
def _fc_legendre_extension_matrix(axis_size: int, fc_degree: int, n_additional_pts: int) -> np.ndarray:
    if axis_size < 2 * fc_degree:
        raise ValueError(
            f"Need at least 2 * fc_degree points for Fourier continuation, "
            f"got axis_size={axis_size}, fc_degree={fc_degree}"
        )

    c = n_additional_pts // 2
    ext_mat = np.zeros((axis_size + n_additional_pts, axis_size), dtype=np.float64)
    ext_mat[c : c + axis_size, :] = np.eye(axis_size, dtype=np.float64)

    if c > 0:
        ext_mat_boundary = _fc_legendre_boundary_matrix(fc_degree, n_additional_pts)
        ext_mat[:c, :fc_degree] = ext_mat_boundary[-c:, fc_degree:]
        ext_mat[:c, axis_size - fc_degree :] = ext_mat_boundary[-c:, :fc_degree]
        ext_mat[-c:, :fc_degree] = ext_mat_boundary[:c, fc_degree:]
        ext_mat[-c:, axis_size - fc_degree :] = ext_mat_boundary[:c, :fc_degree]

    return ext_mat


def fourier_diff(
    x,
    dt,
    order=1,
    use_fc="legendre",
    fc_degree=4,
    fc_n_additional_pts=50,
    low_pass_filter_ratio=None,
):
    if x.ndim != 3:
        raise ValueError(f"fourier_diff expects shape (B, T, D), got {x.shape}")
    if order < 0:
        raise ValueError(f"Derivative order must be non-negative, got {order}")
    if use_fc not in (None, False, "legendre"):
        raise ValueError(f"Unsupported use_fc mode: {use_fc}")

    _, T, _ = x.shape
    x_work = x
    c = 0

    if use_fc == "legendre":
        ext_mat_np = _fc_legendre_extension_matrix(T, fc_degree, fc_n_additional_pts)
        ext_mat = jnp.asarray(ext_mat_np, dtype=x.dtype)
        x_work = jnp.matmul(ext_mat, x_work)
        c = fc_n_additional_pts // 2

    n_t = x_work.shape[1]
    x_ft = jnp.fft.rfft(x_work, axis=1)
    k_t = 2.0 * jnp.pi * jnp.fft.rfftfreq(n_t, d=dt)

    if low_pass_filter_ratio is not None:
        cutoff = int(x_ft.shape[1] * low_pass_filter_ratio)
        mask = (jnp.arange(x_ft.shape[1]) < cutoff).astype(x_ft.dtype)
        x_ft = x_ft * mask[None, :, None]

    derivative_ft = ((1j * k_t) ** order)[None, :, None] * x_ft
    derivative = jnp.fft.irfft(derivative_ft, axis=1, n=n_t)

    if c > 0:
        derivative = derivative[:, c:-c, :]

    return derivative


def spectral_derivative(x, dt, d=5, n_additional_pts=50):
    return fourier_diff(
        x,
        dt,
        order=1,
        use_fc="legendre",
        fc_degree=d,
        fc_n_additional_pts=n_additional_pts,
    )


flax_dense_kernel_init = jax.nn.initializers.lecun_normal()
flax_embed_init = jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)


def zeros_init(shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype=dtype)


class Linear(eqx.Module):
    weight: Array
    bias: Optional[Array]
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: PRNGKey,
        use_bias: bool = True,
        kernel_init=flax_dense_kernel_init,
        bias_init=zeros_init,
        dtype=jnp.float32,
    ):
        wkey, bkey = random.split(key)
        self.weight = kernel_init(wkey, (in_features, out_features), dtype).T
        self.use_bias = use_bias
        self.bias = bias_init((out_features,), dtype) if use_bias else None

    def __call__(self, x: Array) -> Array:
        y = jnp.matmul(x, self.weight.T)
        if self.use_bias and self.bias is not None:
            y = y + self.bias
        return y


class Embedding(eqx.Module):
    weight: Array

    def __init__(self, num_embeddings: int, embedding_size: int, *, key: PRNGKey, dtype=jnp.float32):
        self.weight = flax_embed_init(key, (num_embeddings, embedding_size), dtype)

    def __call__(self, idx: Array) -> Array:
        return self.weight[idx]


class LayerNorm(eqx.Module):
    weight: Array
    bias: Array
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-6, *, key: Optional[PRNGKey] = None, dtype=jnp.float32):
        del key
        self.weight = jnp.ones((dim,), dtype=dtype)
        self.bias = jnp.zeros((dim,), dtype=dtype)
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        orig_dtype = x.dtype
        dtype = jnp.result_type(x.dtype, jnp.float32)
        y = x.astype(dtype)
        mean = jnp.mean(y, axis=-1, keepdims=True)
        var = jnp.maximum(jnp.mean(y * y, axis=-1, keepdims=True) - mean * mean, 0.0)
        y = (y - mean) * lax.rsqrt(var + self.eps)
        y = y * self.weight.astype(dtype) + self.bias.astype(dtype)
        return y.astype(orig_dtype)


class MultiHeadDotProductAttention(eqx.Module):
    query_proj: Linear
    key_proj: Linear
    value_proj: Linear
    out_proj: Linear
    num_heads: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        key: PRNGKey,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        kq, kk, kv, ko = random.split(key, 4)
        self.query_proj = Linear(embed_dim, embed_dim, key=kq)
        self.key_proj = Linear(embed_dim, embed_dim, key=kk)
        self.value_proj = Linear(embed_dim, embed_dim, key=kv)
        self.out_proj = Linear(embed_dim, embed_dim, key=ko)

    def __call__(self, query: Array, key: Array, value: Array, mask: Optional[Array] = None) -> Array:
        q = self.project_query(query)
        k, v = self.project_kv(key, value)
        return self.attend_projected(q, k, v, mask=mask)

    def project_query(self, query: Array) -> Array:
        B, Lq, _ = query.shape
        q = self.query_proj(query).reshape(B, Lq, self.num_heads, self.head_dim)
        return jnp.transpose(q, (0, 2, 1, 3))

    def project_kv(self, key: Array, value: Array) -> Tuple[Array, Array]:
        B, Lk, _ = key.shape
        k = self.key_proj(key).reshape(B, Lk, self.num_heads, self.head_dim)
        v = self.value_proj(value).reshape(B, Lk, self.num_heads, self.head_dim)
        return jnp.transpose(k, (0, 2, 1, 3)), jnp.transpose(v, (0, 2, 1, 3))

    def attend_projected(self, q: Array, k: Array, v: Array, mask: Optional[Array] = None) -> Array:
        B, _, Lq, _ = q.shape

        logits = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * (self.head_dim ** -0.5)
        if mask is not None:
            if mask.ndim == 2:
                logits = jnp.where(mask[None, None, :, :], logits, jnp.finfo(logits.dtype).min)
            elif mask.ndim == 3:
                logits = jnp.where(mask[:, None, :, :], logits, jnp.finfo(logits.dtype).min)
            else:
                raise ValueError(f"Unsupported mask rank: {mask.ndim}")
        attn = jax.nn.softmax(logits, axis=-1)
        out = jnp.matmul(attn, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, Lq, self.embed_dim)
        return self.out_proj(out)


class SpectralConv1D(eqx.Module):
    weight_re: Array
    weight_im: Array
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    modes: int = eqx.field(static=True)

    def __init__(self, in_channels: int, out_channels: int, modes: int, *, key: PRNGKey):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        k1, k2 = random.split(key)
        scale = 0.02
        self.weight_re = scale * random.normal(k1, (modes, in_channels, out_channels), dtype=jnp.float32)
        self.weight_im = scale * random.normal(k2, (modes, in_channels, out_channels), dtype=jnp.float32)

    def __call__(self, x: Array) -> Array:
        B, T, _ = x.shape
        x_ft = jnp.fft.rfft(x, axis=1)
        t_freq = x_ft.shape[1]
        m = min(self.modes, t_freq)

        w = self.weight_re[:m] + 1j * self.weight_im[:m]
        out_ft_low = jnp.swapaxes(jnp.matmul(jnp.swapaxes(x_ft[:, :m, :], 0, 1), w), 0, 1)
        pad = jnp.zeros((B, t_freq - m, self.out_channels), dtype=out_ft_low.dtype)
        out_ft = jnp.concatenate([out_ft_low, pad], axis=1)
        return jnp.fft.irfft(out_ft, n=T, axis=1)


class FNO1D(eqx.Module):
    lift: Linear
    spectral_layers: Tuple[SpectralConv1D, ...]
    linear_layers: Tuple[Linear, ...]
    norm: LayerNorm
    proj: Linear
    out_tokens: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    in_fields: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        modes: int,
        width: int,
        out_tokens: int,
        out_dim: int,
        in_fields: int,
        depth: int = 4,
        *,
        key: PRNGKey,
    ):
        self.out_tokens = out_tokens
        self.out_dim = out_dim
        self.in_fields = in_fields
        self.depth = depth

        keys = random.split(key, 2 * depth + 3)
        self.lift = Linear(in_fields + 1, width, key=keys[0])
        self.spectral_layers = tuple(
            SpectralConv1D(width, width, modes, key=keys[1 + i]) for i in range(depth)
        )
        self.linear_layers = tuple(
            Linear(width, width, key=keys[1 + depth + i]) for i in range(depth)
        )
        self.norm = LayerNorm(width)
        self.proj = Linear(width, out_tokens * out_dim, key=keys[-1])

    def __call__(self, h_fields: Array) -> Array:
        B, T, _ = h_fields.shape
        t = jnp.linspace(0, 1, T)[:, jnp.newaxis]
        t_grid = jnp.broadcast_to(t[None, :, :], (B, T, 1))
        x = jnp.concatenate([h_fields, t_grid], axis=-1)
        x = self.lift(x)

        for spectral_layer, linear_layer in zip(self.spectral_layers, self.linear_layers):
            x = jax.nn.gelu(spectral_layer(x) + linear_layer(x))

        x = self.norm(x)
        x = self.proj(x)
        return x.reshape(B, T, self.out_tokens, self.out_dim)


def causal_mask(seq_len: int) -> Array:
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))


class SpinDecoderBlock(eqx.Module):
    num_heads: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    mlp_mult: int = eqx.field(static=True)
    ln1: LayerNorm
    ln2: LayerNorm
    ln3: LayerNorm
    self_attn: MultiHeadDotProductAttention
    cross_attn: MultiHeadDotProductAttention
    mlp_in: Linear
    mlp_out: Linear

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        mlp_mult: int = 4,
        *,
        key: PRNGKey,
    ):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.mlp_mult = mlp_mult
        k1, k2, k3, k4 = random.split(key, 4)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.ln3 = LayerNorm(embed_dim)
        self.self_attn = MultiHeadDotProductAttention(
            embed_dim,
            num_heads,
            key=k1,
        )
        self.cross_attn = MultiHeadDotProductAttention(
            embed_dim,
            num_heads,
            key=k2,
        )
        self.mlp_in = Linear(embed_dim, embed_dim * mlp_mult, key=k3)
        self.mlp_out = Linear(embed_dim * mlp_mult, embed_dim, key=k4)

    def __call__(self, x: Array, ctx: Array, mask: Array) -> Array:
        y = self.ln1(x)
        y = self.self_attn(y, y, y, mask=mask)
        x = x + y

        y = self.ln2(x)
        y = self.cross_attn(y, ctx, ctx)
        x = x + y

        y = self.ln3(x)
        y = jax.nn.gelu(self.mlp_in(y))
        y = self.mlp_out(y)
        x = x + y
        return x


class NeuralOperatorQuantumState(eqx.Module):
    embed_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    fno_modes: int = eqx.field(static=True)
    fno_width: int = eqx.field(static=True)
    ctx_tokens: int = eqx.field(static=True)
    in_fields: int = eqx.field(static=True)
    phase_mode: str = eqx.field(static=True)
    self_kv_cache: bool = eqx.field(static=True)
    fixed_M0_mode: Optional[str] = eqx.field(static=True)
    fixed_M0_seed: Optional[int] = eqx.field(static=True)
    fno: FNO1D
    spin_embed: Embedding
    pos_embed: Embedding
    start_token: Array
    initial_ctx: Optional[Array]
    blocks: Tuple[SpinDecoderBlock, ...]
    ln_final: LayerNorm
    amp_head: Linear
    phase_head: Linear

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        fno_modes: int,
        fno_width: int,
        ctx_tokens: int,
        in_fields: int,
        *,
        phase_mode: str = "raw",
        self_kv_cache: bool = False,
        fixed_M0_mode: Optional[str] = None,
        fixed_M0_seed: Optional[int] = None,
        key: PRNGKey,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.fno_modes = fno_modes
        self.fno_width = fno_width
        self.ctx_tokens = ctx_tokens
        self.in_fields = in_fields
        self.phase_mode = phase_mode
        self.self_kv_cache = self_kv_cache
        self.fixed_M0_mode = fixed_M0_mode
        self.fixed_M0_seed = fixed_M0_seed

        keys = random.split(key, num_layers + 7)
        self.fno = FNO1D(fno_modes, fno_width, ctx_tokens, embed_dim, in_fields, key=keys[0])
        self.spin_embed = Embedding(2, embed_dim, key=keys[1])
        self.pos_embed = Embedding(N_SPINS, embed_dim, key=keys[2])
        self.start_token = 0.02 * random.normal(keys[3], (1, 1, embed_dim), dtype=jnp.float32)

        if fixed_M0_mode is None or fixed_M0_mode == "learnable_diff":
            self.initial_ctx = 0.02 * random.normal(keys[4], (1, ctx_tokens, embed_dim), dtype=jnp.float32)
        elif fixed_M0_mode == "random_unit_rows":
            self.initial_ctx = make_random_initial_ctx(
                fixed_M0_seed if fixed_M0_seed is not None else 0,
                ctx_tokens,
                embed_dim,
            )
        elif fixed_M0_mode == "canonical_basis":
            self.initial_ctx = make_canonical_initial_ctx(ctx_tokens, embed_dim)
        else:
            raise ValueError(f"Unknown fixed_M0_mode: {fixed_M0_mode}")

        self.blocks = tuple(
            SpinDecoderBlock(
                num_heads,
                embed_dim,
                key=keys[5 + i],
            )
            for i in range(num_layers)
        )
        self.ln_final = LayerNorm(embed_dim)
        self.amp_head = Linear(embed_dim, 2, key=keys[-2], kernel_init=jax.nn.initializers.orthogonal())
        self.phase_head = Linear(embed_dim, 2, key=keys[-1], kernel_init=jax.nn.initializers.normal(0.02))

    def _runtime_initial_ctx(self) -> Array:
        if self.fixed_M0_mode is None or self.fixed_M0_mode == "learnable_diff":
            return self.initial_ctx
        return lax.stop_gradient(self.initial_ctx)

    def encode_field(self, h_fields: Array) -> Tuple[Array, Array]:
        B, T, _ = h_fields.shape
        Mdot = self.fno(h_fields)
        M0 = jnp.broadcast_to(self._runtime_initial_ctx(), (B, self.ctx_tokens, self.embed_dim))
        M = integrate_context_velocity(M0, Mdot, DT)
        return M, Mdot

    def get_initial_ctx(self) -> Array:
        return self._runtime_initial_ctx()

    def _decode(self, spins_idx: Array, ctx_tokens: Array) -> Array:
        B, N = spins_idx.shape
        pos = self.pos_embed(jnp.arange(N, dtype=jnp.int32))[None, :, :]
        start = jnp.tile(self.start_token, (B, 1, 1)) + pos[:, 0:1, :]
        prev = self.spin_embed(spins_idx[:, :-1]) + pos[:, 1:, :]
        x = jnp.concatenate([start, prev], axis=1)
        mask = causal_mask(N)
        for blk in self.blocks:
            x = blk(x, ctx_tokens, mask)
        return self.ln_final(x)

    def build_cross_kv_cache(self, ctx_tokens: Array) -> Tuple[Tuple[Array, Array], ...]:
        return tuple(
            blk.cross_attn.project_kv(ctx_tokens, ctx_tokens) for blk in self.blocks
        )

    def empty_self_kv_cache(self, batch_size: int, n_spins: int) -> Tuple[Tuple[Array, Array], ...]:
        head_dim = self.embed_dim // self.num_heads
        shape = (batch_size, self.num_heads, n_spins, head_dim)
        dtype = self.start_token.dtype
        return tuple(
            (
                jnp.zeros(shape, dtype=dtype),
                jnp.zeros(shape, dtype=dtype),
            )
            for _ in self.blocks
        )

    def _decode_token_step(
        self,
        token_x: Array,
        self_kv_cache: Tuple[Tuple[Array, Array], ...],
        cross_kv_cache: Tuple[Tuple[Array, Array], ...],
        step_idx: Array,
        n_spins: int,
    ) -> Tuple[Array, Tuple[Tuple[Array, Array], ...]]:
        valid_mask = (jnp.arange(n_spins, dtype=jnp.int32) <= step_idx)[None, :]
        x = token_x
        new_self_kv_cache = []
        update_idx = (0, 0, step_idx, 0)

        for blk, cross_kv, self_kv in zip(self.blocks, cross_kv_cache, self_kv_cache):
            k_cache, v_cache = self_kv

            y = blk.ln1(x)
            q = blk.self_attn.project_query(y)
            k_new, v_new = blk.self_attn.project_kv(y, y)
            k_cache = lax.dynamic_update_slice(k_cache, k_new, update_idx)
            v_cache = lax.dynamic_update_slice(v_cache, v_new, update_idx)
            y = blk.self_attn.attend_projected(q, k_cache, v_cache, mask=valid_mask)
            x = x + y

            y = blk.ln2(x)
            q = blk.cross_attn.project_query(y)
            y = blk.cross_attn.attend_projected(q, cross_kv[0], cross_kv[1])
            x = x + y

            y = blk.ln3(x)
            y = jax.nn.gelu(blk.mlp_in(y))
            y = blk.mlp_out(y)
            x = x + y
            new_self_kv_cache.append((k_cache, v_cache))

        return x, tuple(new_self_kv_cache)

    def logits_from_tokens(self, spins_idx: Array, ctx_tokens: Array) -> Array:
        return self.amp_head(self._decode(spins_idx, ctx_tokens))

    def log_psi_from_tokens(self, spins_idx: Array, ctx_tokens: Array) -> Array:
        h = self._decode(spins_idx, ctx_tokens)
        logits_amp = self.amp_head(h)
        logits_phase = self.phase_head(h)
        if self.phase_mode == "softsign":
            logits_phase = jax.nn.soft_sign(logits_phase) * jnp.pi
        elif self.phase_mode != "raw":
            raise ValueError(f"Unknown phase_mode: {self.phase_mode}")
        logp = jax.nn.log_softmax(logits_amp, axis=-1)
        one_hot = jax.nn.one_hot(spins_idx, 2)
        log_amp = 0.5 * jnp.sum(jnp.sum(logp * one_hot, axis=-1), axis=-1)
        phase = jnp.sum(jnp.sum(logits_phase * one_hot, axis=-1), axis=-1)
        return log_amp + 1j * phase

    def __call__(self, h_fields: Array, t_idx: Array, spins_idx: Array) -> Array:
        B = h_fields.shape[0]
        t_idx = jnp.asarray(t_idx, dtype=jnp.int32)
        if t_idx.ndim == 0:
            t_idx = jnp.full((B,), t_idx, dtype=jnp.int32)
        M_traj, _ = self.encode_field(h_fields)
        M_t = M_traj[jnp.arange(B), t_idx]
        return self.log_psi_from_tokens(spins_idx, M_t)


# ============================================================
# 5) Loss & Training (Global)
# ============================================================
@eqx.filter_jit
def sample_autoregressive(key, model, ctx, n_spins):
    B = ctx.shape[0]
    spins = jnp.zeros((B, n_spins), dtype=jnp.int32)

    if model.self_kv_cache:
        pos = model.pos_embed(jnp.arange(n_spins, dtype=jnp.int32))
        cross_kv_cache = model.build_cross_kv_cache(ctx)
        self_kv_cache = model.empty_self_kv_cache(B, n_spins)
        token_x0 = jnp.tile(model.start_token, (B, 1, 1)) + pos[None, 0:1, :]

        def body(carry, i):
            s, token_x, self_kv_cache, rng = carry
            x, self_kv_cache = model._decode_token_step(token_x, self_kv_cache, cross_kv_cache, i, n_spins)
            h = model.ln_final(x)
            logits = model.amp_head(h)[:, 0, :]
            rng, sub = random.split(rng)
            s_i = random.categorical(sub, logits).astype(jnp.int32)
            s = s.at[:, i].set(s_i)
            next_pos = jnp.take(pos, jnp.minimum(i + 1, n_spins - 1), axis=0)[None, None, :]
            next_token_x = model.spin_embed(s_i)[:, None, :] + next_pos
            return (s, next_token_x, self_kv_cache, rng), None

        (spins, _, _, _), _ = lax.scan(
            body,
            (spins, token_x0, self_kv_cache, key),
            jnp.arange(n_spins, dtype=jnp.int32),
        )
        return spins * 2 - 1

    def body(carry, i):
        s, rng = carry
        logits = model.logits_from_tokens(s, ctx)[:, i]
        rng, sub = random.split(rng)
        s_i = random.categorical(sub, logits).astype(jnp.int32)
        s = s.at[:, i].set(s_i)
        return (s, rng), None

    (spins, _), _ = lax.scan(body, (spins, key), jnp.arange(n_spins))
    return spins * 2 - 1


@eqx.filter_jit
def compute_local_energy(spins_pm, ctx, h_x_t, h_z_t, model):
    spins_idx = ((spins_pm + 1) // 2).astype(jnp.int32)

    E_zz = -J_zz * jnp.sum(spins_pm[:, BOND_I] * spins_pm[:, BOND_J], axis=1)
    E_z = -h_z_t * jnp.sum(spins_pm, axis=1)

    log_psi = model.log_psi_from_tokens(spins_idx, ctx)

    def ratio_fn(i):
        s_flip = spins_pm.at[:, i].set(-spins_pm[:, i])
        s_flip_idx = ((s_flip + 1) // 2).astype(jnp.int32)
        lp_flip = model.log_psi_from_tokens(s_flip_idx, ctx)
        return jnp.exp(lp_flip - log_psi)

    ratios = jax.vmap(ratio_fn)(jnp.arange(N_SPINS, dtype=jnp.int32))
    E_x = -h_x_t * jnp.sum(ratios, axis=0)
    return E_zz + E_z + E_x


@eqx.filter_jit
def anchor_loss(model, key):
    k2 = key
    initial_ctx = model.get_initial_ctx()
    M0 = jnp.broadcast_to(initial_ctx, (BATCH_SIZE, CTX_TOKENS, EMBED_DIM))

    def loss_plus(k):
        spins_idx = random.bernoulli(k, 0.5, (BATCH_SIZE, N_SPINS)).astype(jnp.int32)
        log_psi = model.log_psi_from_tokens(spins_idx, M0)
        log_p = 2.0 * jnp.real(log_psi)
        target = -float(N_SPINS) * jnp.log(2.0)
        l_amp = jnp.mean((log_p - target) ** 2)
        l_phase = jnp.mean(jnp.imag(log_psi) ** 2)
        return l_amp + l_phase

    def loss_z_up(k):
        up = jnp.zeros((BATCH_SIZE, N_SPINS), dtype=jnp.int32)
        log_psi_up = model.log_psi_from_tokens(up, M0)
        log_p_up = 2.0 * jnp.real(log_psi_up)
        l_amp = jnp.mean(log_p_up**2)
        l_phase = jnp.mean(jnp.imag(log_psi_up) ** 2)

        spins_idx = random.bernoulli(k, 0.5, (BATCH_SIZE, N_SPINS)).astype(jnp.int32)
        spins_idx = jnp.where(
            jnp.sum(spins_idx, axis=1, keepdims=True) == 0,
            spins_idx.at[:, 0].set(1),
            spins_idx,
        )
        log_psi_other = model.log_psi_from_tokens(spins_idx, M0)
        log_p_other = 2.0 * jnp.real(log_psi_other)
        margin = 10.0
        l_other = jnp.mean(jnp.maximum(log_p_other + margin, 0.0) ** 2)
        return l_amp + l_phase + l_other

    if init_state == "plus":
        return loss_plus(k2)
    elif init_state == "up":
        return loss_z_up(k2)
    else:
        raise ValueError(f"Unknown mode: {init_state}")


@eqx.filter_jit
def noqs_loss(model, key, h_fields, t_idx, anchor_w):
    B = h_fields.shape[0]

    M_traj, Mdot_traj = model.encode_field(h_fields)
    batch_indices = jnp.arange(B)[:, None]
    M_t = M_traj[batch_indices, t_idx]
    Mdot_t = Mdot_traj[batch_indices, t_idx]

    h_t = h_fields[batch_indices, t_idx, :]
    h_x_t = h_t[..., 0]
    h_z_t = h_t[..., 1]

    num_groups = B * K
    M_group = M_t.reshape(num_groups, CTX_TOKENS, EMBED_DIM)
    Mdot_group = Mdot_t.reshape(num_groups, CTX_TOKENS, EMBED_DIM)
    h_x_group = h_x_t.reshape(num_groups)
    h_z_group = h_z_t.reshape(num_groups)

    M_flat = jnp.broadcast_to(M_group[:, None, :, :], (num_groups, M_SAMPLES, CTX_TOKENS, EMBED_DIM)).reshape(
        -1, CTX_TOKENS, EMBED_DIM
    )
    Mdot_flat = jnp.broadcast_to(
        Mdot_group[:, None, :, :], (num_groups, M_SAMPLES, CTX_TOKENS, EMBED_DIM)
    ).reshape(-1, CTX_TOKENS, EMBED_DIM)
    h_x_flat = jnp.broadcast_to(h_x_group[:, None], (num_groups, M_SAMPLES)).reshape(-1)
    h_z_flat = jnp.broadcast_to(h_z_group[:, None], (num_groups, M_SAMPLES)).reshape(-1)

    key_s, key_a = random.split(key)
    spins_pm = sample_autoregressive(key_s, model, M_flat, N_SPINS)
    spins_idx = ((spins_pm + 1) // 2).astype(jnp.int32)

    def logpsi_and_dt_logpsi(m, mdot, s):
        def _f(x):
            return model.log_psi_from_tokens(s[None], x[None])[0]

        return jax.jvp(_f, (m,), (mdot,))

    if SCORE_FUNCTION_WEIGHT == 0.0:
        def dt_only(m, mdot, s):
            return logpsi_and_dt_logpsi(m, mdot, s)[1]

        dt_lp = jax.vmap(dt_only)(M_flat, Mdot_flat, spins_idx)
        log_psi_base = None
    else:
        log_psi_base, dt_lp = jax.vmap(logpsi_and_dt_logpsi)(M_flat, Mdot_flat, spins_idx)

    E_loc = compute_local_energy(spins_pm, M_flat, h_x_flat, h_z_flat, model)
    diff = 1j * dt_lp - E_loc
    if GAUGE_CENTERING == "global":
        diff_centered = diff - lax.stop_gradient(jnp.mean(diff))
        sample_loss = jnp.abs(diff_centered) ** 2
        loss_phys = jnp.mean(sample_loss)
        score_weight = sample_loss - loss_phys
    elif GAUGE_CENTERING == "per_time":
        num_groups = B * K
        diff_grouped = diff.reshape(num_groups, M_SAMPLES)
        diff_centered = diff_grouped - lax.stop_gradient(jnp.mean(diff_grouped, axis=1, keepdims=True))
        sample_loss = jnp.abs(diff_centered) ** 2
        loss_phys = jnp.mean(sample_loss)
        score_weight = sample_loss.reshape(-1) - loss_phys
    else:
        raise ValueError(f"Unknown gauge centering: {GAUGE_CENTERING}")

    l_anchor = anchor_loss(model, key_a)
    physical_loss = loss_phys + anchor_w * l_anchor

    if SCORE_FUNCTION_WEIGHT == 0.0:
        opt_loss = physical_loss
    else:
        log_p = 2.0 * jnp.real(log_psi_base)
        loss_score = jnp.mean(lax.stop_gradient(score_weight) * log_p)
        opt_loss = physical_loss + SCORE_FUNCTION_WEIGHT * loss_score

    return opt_loss, physical_loss


# ============================================================
# 6) Training Driver
# ============================================================
print("--- Training Pure FNO Global Model (TFIM + longitudinal field, OBC) ---")
if FIX_M0_MODE is None:
    print("M0 mode: shared learnable initial context")
else:
    print(f"M0 mode: {FIX_M0_MODE}")
print(f"Phase mode: {PHASE_MODE}")
print(f"Gauge centering: {GAUGE_CENTERING}")
print(f"Score function weight: {SCORE_FUNCTION_WEIGHT}")
print(f"Attention kernel: {ATTENTION_KERNEL}")
print(f"Self-KV cache: {SELF_KV_CACHE}")
print(f"Context dynamics: direct FNO Mdot with {INTEGRATION_RULE} integration")
print(f"Site order: {SITE_ORDER}")

rng = random.PRNGKey(SEED)
model_key, rng = random.split(rng)
model = NeuralOperatorQuantumState(
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    FNO_MODES,
    FNO_WIDTH,
    CTX_TOKENS,
    IN_FIELDS,
    phase_mode=PHASE_MODE,
    self_kv_cache=SELF_KV_CACHE,
    fixed_M0_mode=FIX_M0_MODE,
    fixed_M0_seed=FIX_M0_SEED if FIX_M0_MODE == "random_unit_rows" else None,
    key=model_key,
)
model_template = model

schedule = optax.exponential_decay(START_LR, DECAY_STEPS, DECAY_RATE, end_value=MIN_LR)
optimizer = optax.chain(optax.clip_by_global_norm(0.1), optax.adam(learning_rate=schedule))
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
start_step = 0
loss_hist = []
best_loss = float("inf")
best_model = model

if RESUME_PATH is not None:
    if not RESUME_PATH.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {RESUME_PATH}")
    resume_payload = load_resume_payload(RESUME_PATH)
    if isinstance(resume_payload, dict) and resume_payload.get("format") == "zihaoequinox_checkpoint_v1":
        model = combine_model(resume_payload["model"], model_template)
        best_model = combine_model(resume_payload.get("best_model", resume_payload["model"]), model_template)
        opt_state = resume_payload["opt_state"]
        rng = jnp.asarray(resume_payload["rng"], dtype=jnp.uint32)
        best_loss = float(resume_payload.get("best_loss", float("inf")))
        loss_hist = [float(x) for x in resume_payload.get("loss_hist", [])]
        start_step = int(resume_payload.get("step", -1)) + 1
        print(f"Loaded resumable checkpoint from: {RESUME_PATH}")
        print(f"Resuming TDVP training from step {start_step}")
    else:
        model = combine_model(resume_payload, model_template)
        best_model = model
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        print(f"Loaded params-only checkpoint from: {RESUME_PATH}")
        print("Optimizer state was reinitialized; starting TDVP training from step 0")

if RESUME_PATH is None:
    print("Pre-training Anchor Condition...")

    @eqx.filter_jit
    def anchor_step(model, opt_state, key):
        loss, grads = eqx.filter_value_and_grad(anchor_loss)(model, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for i in tqdm(range(CLI_ARGS.anchor_pretrain_steps)):
        rng, k = random.split(rng)
        model, opt_state, _ = anchor_step(model, opt_state, k)
else:
    print("Skipping anchor pre-training because --resume was provided.")

print("TDVP Training...")
benchmark_history = []
pbar = tqdm(range(start_step, CLI_ARGS.steps), initial=start_step, total=CLI_ARGS.steps)


@eqx.filter_jit
def train_step(model, opt_state, key):
    k1, k2, k3 = random.split(key, 3)
    h_batch = generate_grf_trajectories(k1, BATCH_SIZE, T_STEPS, T_MAX)
    t_idx = random.randint(k2, (BATCH_SIZE, K), 0, T_STEPS)
    (loss, physical_loss), grads = eqx.filter_value_and_grad(noqs_loss, has_aux=True)(
        model, k3, h_batch, t_idx, anchor_weight
    )
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, physical_loss


last_completed_step = start_step - 1

for step in pbar:
    rng, k_step = random.split(rng)
    model, opt_state, loss = train_step(model, opt_state, k_step)
    last_completed_step = step

    current_loss = float(loss)
    loss_hist.append(current_loss)

    if current_loss < best_loss:
        best_loss = current_loss
        best_model = model

    if step % CLI_ARGS.print_every == 0:
        pbar.set_description(f"Loss: {np.mean(loss_hist[-30:]):.5f}")

    if step > 0 and step % 500 == 0:
        save_training_checkpoint(
            CHECKPOINT_PATH,
            step=step,
            model=model,
            opt_state=opt_state,
            rng=rng,
            best_loss=best_loss,
            best_model=best_model,
            loss_hist=loss_hist,
        )

    if CLI_ARGS.benchmark_every > 0 and (step + 1) % CLI_ARGS.benchmark_every == 0:
        k_val = random.PRNGKey(CLI_ARGS.benchmark_key)
        h_val = generate_grf_trajectories(k_val, 1, T_STEPS, T_MAX)[0]
        plot_path = Path("benchmark_plots") / f"benchmark_step_{step:05d}.png"
        benchmark_result = run_benchmark(model, h_val, show_plot=True, plot_path=str(plot_path))
        benchmark_record = {"step": int(step), **benchmark_result["summary"]}
        benchmark_history.append(benchmark_record)
        tqdm.write(
            f"[benchmark step {step}] "
            f"avg_fid={benchmark_record['avg_fidelity']:.6f}, "
            f"final_fid={benchmark_record['final_fidelity']:.6f}, "
            f"best_fid={benchmark_record['best_fidelity']:.6f}, "
            f"mx_mae={benchmark_record['mx_mae']:.6e}, "
            f"zz_mae={benchmark_record['zz_mae']:.6e}"
        )
        save_benchmark_series(Path("."), step, benchmark_result)
        save_json(Path("benchmark_history.json"), {"benchmarks": benchmark_history})
        save_training_checkpoint(
            CHECKPOINT_PATH,
            step=step,
            model=model,
            opt_state=opt_state,
            rng=rng,
            best_loss=best_loss,
            best_model=best_model,
            loss_hist=loss_hist,
        )

print(f"Training Complete. Best loss achieved: {best_loss:.6f}")

save_training_checkpoint(
    CHECKPOINT_PATH,
    step=last_completed_step,
    model=model,
    opt_state=opt_state,
    rng=rng,
    best_loss=best_loss,
    best_model=best_model,
    loss_hist=loss_hist,
)
print(f"Saved resumable checkpoint to: {CHECKPOINT_PATH}")
save_json(Path("benchmark_history.json"), {"benchmarks": benchmark_history})

with open("zihaoequinox_best.pkl", "wb") as f:
    pickle.dump(jax.device_get(arrays_only(best_model)), f)
