import os

# Honor an existing CUDA selection from the shell; default to GPU 1.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
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
import scipy.sparse as sp
from jax import lax, random
from scipy.sparse.linalg import expm_multiply
from tqdm import tqdm


Array = jax.Array
PRNGKey = Array


# ============================================================
# 0) CLI
# ============================================================
CLI_PARSER = argparse.ArgumentParser(
    description=(
        "Context-only NQS experiment. There is no neural operator/FNO. "
        "A shared transformer defines psi_theta(sigma; M). Optionally fit "
        "that manifold using free context tokens, then freeze theta and "
        "obtain Mdot(t) either from an M-only TDVP/SR solve or by a finite-step "
        "projection baseline."
    )
)

CLI_PARSER.add_argument("--Lx", type=int, default=3, help="Number of lattice sites along x.")
CLI_PARSER.add_argument("--Ly", type=int, default=3, help="Number of lattice sites along y.")
CLI_PARSER.add_argument(
    "--site-order",
    type=str,
    choices=("row_major", "col_major", "snake"),
    default="snake",
    help="1D site ordering for tokenization and lattice edges.",
)
CLI_PARSER.add_argument("--t-steps", type=int, default=41, help="Number of time points.")
CLI_PARSER.add_argument("--t-max", type=float, default=0.5, help="Physical time horizon.")
CLI_PARSER.add_argument("--d-model", type=int, default=64, help="Transformer embedding dimension.")
CLI_PARSER.add_argument("--num-layers", type=int, default=3, help="Number of transformer decoder layers.")
CLI_PARSER.add_argument("--num-heads", type=int, default=8, help="Number of attention heads.")
CLI_PARSER.add_argument("--ctx-tokens", type=int, default=4, help="Number of context tokens.")
CLI_PARSER.add_argument(
    "--phase-mode",
    type=str,
    choices=("raw", "softsign"),
    default="softsign",
    help="Phase parameterization for the decoder head.",
)
CLI_PARSER.add_argument(
    "--init-state",
    type=str,
    choices=("plus", "up"),
    default="plus",
    help="Physical initial state shared by all trajectories.",
)
CLI_PARSER.add_argument(
    "--M0-mode",
    type=str,
    choices=("random_sqrt_d", "random_unit", "canonical"),
    default="random_sqrt_d",
    help=(
        "Fixed initial context. random_sqrt_d gives each token RMS=1, "
        "i.e. ||M_a||=sqrt(d); random_unit gives ||M_a||=1."
    ),
)
CLI_PARSER.add_argument(
    "--ctx-norm",
    type=str,
    choices=("none", "unit", "sqrt_d"),
    default="none",
    help="Optional hard row-wise context normalization after an update.",
)
CLI_PARSER.add_argument("--seed", type=int, default=42)
CLI_PARSER.add_argument("--train-protocols", type=int, default=2)
CLI_PARSER.add_argument("--train-protocol-key", type=int, default=1234)
CLI_PARSER.add_argument("--test-protocol-key", type=int, default=14850)
CLI_PARSER.add_argument("--n-modes", type=int, default=10)
CLI_PARSER.add_argument("--omega-factor", type=float, default=10.0)

# Optional oracle/free-context manifold fitting. This is deliberately NOT an FNO.
CLI_PARSER.add_argument(
    "--fit-steps",
    type=int,
    default=3000,
    help=(
        "Supervised free-context manifold fitting steps before the local-Mdot test. "
        "Set to 0 to skip fitting and only test the initialized/loaded decoder."
    ),
)
CLI_PARSER.add_argument("--fit-batch", type=int, default=2, help="Number of (protocol,time) states per fit step.")
CLI_PARSER.add_argument("--fit-lr", type=float, default=5e-4)
CLI_PARSER.add_argument("--fit-min-lr", type=float, default=5e-6)
CLI_PARSER.add_argument("--fit-decay-steps", type=int, default=1000)
CLI_PARSER.add_argument("--fit-decay-rate", type=float, default=0.95)
CLI_PARSER.add_argument("--anchor-weight", type=float, default=1.0)
CLI_PARSER.add_argument(
    "--context-smooth-weight",
    type=float,
    default=0.0,
    help="Optional ||M(t)-M(t-dt)||^2 regularizer during free-context fitting.",
)
CLI_PARSER.add_argument("--print-every", type=int, default=50)

# Context-velocity solver.  The default is SR/TDVP directly in M-space.
CLI_PARSER.add_argument(
    "--mdot-method",
    type=str,
    choices=("sr", "finite_projection"),
    default="sr",
    help=(
        "How to obtain Mdot. 'sr' solves the TDVP/SR linear system directly; "
        "'finite_projection' retains the previous inner Adam projection as a validation baseline."
    ),
)
CLI_PARSER.add_argument(
    "--sr-backend",
    type=str,
    choices=("auto", "parameter", "sample"),
    default="auto",
    help=(
        "Linear solve for SR. parameter solves the P x P metric; sample uses the "
        "Rende et al. push-through identity and solves a 2Ns x 2Ns system; auto chooses the smaller."
    ),
)
CLI_PARSER.add_argument(
    "--sr-data",
    type=str,
    choices=("exact", "mc"),
    default="exact",
    help=(
        "How SR expectation values are estimated. exact enumerates the Hilbert space; "
        "mc uses autoregressive samples. The free-context fitting/reference trajectory remains exact."
    ),
)
CLI_PARSER.add_argument("--sr-samples", type=int, default=128, help="Number of autoregressive samples for --sr-data=mc.")
CLI_PARSER.add_argument(
    "--sr-reg",
    type=float,
    default=1e-4,
    help="Dimensionless diagonal-shift strength used in the SR solve.",
)
CLI_PARSER.add_argument(
    "--sr-reg-mode",
    type=str,
    choices=("trace_scaled", "absolute"),
    default="trace_scaled",
    help=(
        "trace_scaled uses lambda = sr_reg * Tr(S)/P, making the shift less sensitive to the scale of M; "
        "absolute uses lambda = sr_reg directly."
    ),
)
CLI_PARSER.add_argument(
    "--sr-rcond",
    type=float,
    default=1e-10,
    help="Relative singular-value threshold used only for rank diagnostics and pseudoinverse diagnostics.",
)
CLI_PARSER.add_argument(
    "--sr-jac-chunk-size",
    type=int,
    default=128,
    help="Configurations per batch when evaluating context log-derivatives O_a = d log(psi)/d M_a.",
)
CLI_PARSER.add_argument(
    "--compare-sr-backends",
    action="store_true",
    help="At each rollout step, also solve the alternative SR backend and report relative disagreement.",
)
CLI_PARSER.add_argument(
    "--self-test",
    action="store_true",
    help="Run algebraic and model-level SR correctness tests and exit before training.",
)

# Optional finite-step projection baseline.
CLI_PARSER.add_argument(
    "--projection-steps",
    type=int,
    default=100,
    help="Adam iterations used to solve each local context projection.",
)
CLI_PARSER.add_argument(
    "--projection-lr",
    type=float,
    default=5e-2,
    help=(
        "Learning rate for Delta M = dt*Mdot. Optimizing Delta M rather than Mdot "
        "avoids a 1/dt conditioning problem; Mdot is returned as Delta M/dt."
    ),
)
CLI_PARSER.add_argument("--projection-tol", type=float, default=1e-8)
CLI_PARSER.add_argument(
    "--projection-reg",
    type=float,
    default=0.0,
    help="Optional L2 penalty on Delta M during the local projection.",
)
CLI_PARSER.add_argument(
    "--projection-metric",
    type=str,
    choices=("phase_l2", "infidelity"),
    default="phase_l2",
    help=(
        "Local state-space distance. phase_l2 is min_phi ||target-exp(i phi) candidate||^2 "
        "= 2-2|<target|candidate>| for normalized states; infidelity is 1-|overlap|^2."
    ),
)
CLI_PARSER.add_argument(
    "--warm-start-mdot",
    action="store_true",
    help="Initialize the next Delta M with dt times the previous Mdot.",
)
CLI_PARSER.add_argument(
    "--local-H",
    type=str,
    choices=("left", "midpoint"),
    default="left",
    help=(
        "Hamiltonian used in the local projection target. 'left' implements "
        "exp[-i H(t) dt] exactly as written; 'midpoint' uses H(t+dt/2)."
    ),
)

# Exact Hilbert-space evaluation is the cleanest version of this experiment.
CLI_PARSER.add_argument(
    "--exact-chunk-size",
    type=int,
    default=1024,
    help="Basis configurations per transformer call when computing exact overlaps.",
)
CLI_PARSER.add_argument(
    "--max-exact-spins",
    type=int,
    default=16,
    help="Safety guard for exact Hilbert-space calculations.",
)

CLI_PARSER.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Resume a checkpoint produced by THIS context-only script.",
)
CLI_PARSER.add_argument(
    "--checkpoint-path",
    type=str,
    default="context_only_checkpoint.pkl",
)
CLI_PARSER.add_argument("--output-dir", type=str, default="context_only_results")
CLI_ARGS = CLI_PARSER.parse_args()

print(f"JAX is using: {jax.devices()}")


# ============================================================
# 1) Hyperparameters
# ============================================================
LX = CLI_ARGS.Lx
LY = CLI_ARGS.Ly
N_SPINS = LX * LY
SITE_ORDER = CLI_ARGS.site_order
T_STEPS = CLI_ARGS.t_steps
T_MAX = float(CLI_ARGS.t_max)
if T_STEPS < 2:
    raise ValueError("Need at least two time points.")
DT = T_MAX / (T_STEPS - 1)

EMBED_DIM = CLI_ARGS.d_model
NUM_LAYERS = CLI_ARGS.num_layers
NUM_HEADS = CLI_ARGS.num_heads
CTX_TOKENS = CLI_ARGS.ctx_tokens
PHASE_MODE = CLI_ARGS.phase_mode
INIT_STATE = CLI_ARGS.init_state
CTX_NORM = CLI_ARGS.ctx_norm
SEED = CLI_ARGS.seed
J_ZZ = 1.0
N_MODES = CLI_ARGS.n_modes
OMEGA_FACTOR = float(CLI_ARGS.omega_factor)
FIT_BATCH = CLI_ARGS.fit_batch
EXACT_CHUNK_SIZE = CLI_ARGS.exact_chunk_size
SR_JAC_CHUNK_SIZE = max(1, int(CLI_ARGS.sr_jac_chunk_size))
SR_P = CTX_TOKENS * EMBED_DIM
OUTPUT_DIR = Path(CLI_ARGS.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = Path(CLI_ARGS.checkpoint_path)
RESUME_PATH = Path(CLI_ARGS.resume) if CLI_ARGS.resume else None

if EMBED_DIM % NUM_HEADS != 0:
    raise ValueError("d-model must be divisible by num-heads.")
if CLI_ARGS.sr_data == "mc" and CLI_ARGS.sr_samples < 2:
    raise ValueError("--sr-samples must be at least 2 for a covariance-based SR estimate.")
if N_SPINS > CLI_ARGS.max_exact_spins:
    raise ValueError(
        f"This script uses exact Hilbert-space overlaps. N={N_SPINS} exceeds "
        f"--max-exact-spins={CLI_ARGS.max_exact_spins}. Start with a smaller lattice "
        "or explicitly raise the guard if you have enough memory."
    )


# ============================================================
# 2) Geometry and Hamiltonian utilities
# ============================================================
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
    for y in range(Ly):
        for x in range(Lx):
            i = coord_to_idx[(x, y)]
            if x + 1 < Lx:
                bonds_i.append(i)
                bonds_j.append(coord_to_idx[(x + 1, y)])
            if y + 1 < Ly:
                bonds_i.append(i)
                bonds_j.append(coord_to_idx[(x, y + 1)])
    return np.asarray(bonds_i, np.int32), np.asarray(bonds_j, np.int32)


BOND_I_NP, BOND_J_NP = make_lattice_bonds(LX, LY, SITE_ORDER)
N_BONDS = len(BOND_I_NP)
DIM = 2**N_SPINS


def generate_grf_trajectories(key, batch_size, t_steps, t_max):
    kx1, kx2, kx3, kz1, kz2, kz3 = random.split(key, 6)
    t = jnp.linspace(0.0, t_max, t_steps, dtype=jnp.float32)
    k = jnp.arange(1, N_MODES + 1, dtype=jnp.float32)
    omega = k * (jnp.pi / t_max) * OMEGA_FACTOR
    amp_decay = 1.0 / (k**1.5)

    amps_x = random.uniform(kx1, (batch_size, N_MODES), minval=-0.50, maxval=0.50) * amp_decay[None, :]
    phases_x = random.uniform(kx2, (batch_size, N_MODES), minval=0.0, maxval=2 * jnp.pi)
    hx0 = random.uniform(kx3, (batch_size, 1), minval=-0.05, maxval=0.05) + 1.0
    args_x = omega[None, :, None] * t[None, None, :] + phases_x[:, :, None]
    hx = hx0 + jnp.sum(amps_x[:, :, None] * jnp.sin(args_x), axis=1)

    amps_z = random.uniform(kz1, (batch_size, N_MODES), minval=-0.05, maxval=0.05) * amp_decay[None, :]
    phases_z = random.uniform(kz2, (batch_size, N_MODES), minval=0.0, maxval=2 * jnp.pi)
    hz0 = random.uniform(kz3, (batch_size, 1), minval=-0.05, maxval=0.05)
    args_z = omega[None, :, None] * t[None, None, :] + phases_z[:, :, None]
    hz = hz0 + jnp.sum(amps_z[:, :, None] * jnp.sin(args_z), axis=1)
    return jnp.stack([hx, hz], axis=-1)


def build_sparse_terms():
    idx = np.arange(DIM, dtype=np.int64)
    zz_sum = np.zeros(DIM, dtype=np.float64)
    z_sum = np.zeros(DIM, dtype=np.float64)

    for bi, bj in zip(BOND_I_NP.tolist(), BOND_J_NP.tolist()):
        zi = 1.0 - 2.0 * ((idx >> bi) & 1)
        zj = 1.0 - 2.0 * ((idx >> bj) & 1)
        zz_sum += zi * zj

    for i in range(N_SPINS):
        zi = 1.0 - 2.0 * ((idx >> i) & 1)
        z_sum += zi

    rows = []
    cols = []
    for i in range(N_SPINS):
        rows.append(idx)
        cols.append(idx ^ (1 << i))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(rows.shape[0], dtype=np.float64)
    x_sum = sp.coo_matrix((data, (rows, cols)), shape=(DIM, DIM)).tocsr()
    return zz_sum, z_sum, x_sum


ZZ_SUM_DIAG, Z_SUM_DIAG, X_SUM_SPARSE = build_sparse_terms()
AVG_ZZ_DIAG = ZZ_SUM_DIAG / max(N_BONDS, 1)


def sparse_hamiltonian(hx: float, hz: float):
    diag = -J_ZZ * ZZ_SUM_DIAG - float(hz) * Z_SUM_DIAG
    return sp.diags(diag, format="csr") - float(hx) * X_SUM_SPARSE


def exp_h_step(psi: np.ndarray, hx: float, hz: float, dt: float) -> np.ndarray:
    """Apply exp[-i H(hx,hz) dt] using sparse Krylov expm_multiply."""
    H = sparse_hamiltonian(hx, hz)
    out = expm_multiply((-1j * dt) * H, np.asarray(psi, dtype=np.complex128))
    out = out / np.linalg.norm(out)
    return out.astype(np.complex64)


def initial_physical_state() -> np.ndarray:
    if INIT_STATE == "plus":
        return (np.ones(DIM, dtype=np.complex64) / np.sqrt(DIM)).astype(np.complex64)
    if INIT_STATE == "up":
        out = np.zeros(DIM, dtype=np.complex64)
        out[0] = 1.0 + 0.0j
        return out
    raise ValueError(INIT_STATE)


def exact_time_dependent_trajectory(fields: np.ndarray) -> np.ndarray:
    """
    Reference physical trajectory. A midpoint Hamiltonian is used on each
    interval, giving a second-order approximation to the time-ordered evolution.
    """
    states = [initial_physical_state()]
    psi = states[0]
    for t in range(T_STEPS - 1):
        hx = 0.5 * (float(fields[t, 0]) + float(fields[t + 1, 0]))
        hz = 0.5 * (float(fields[t, 1]) + float(fields[t + 1, 1]))
        psi = exp_h_step(psi, hx, hz, DT)
        states.append(psi)
    return np.stack(states, axis=0)


def compute_observables(psi: np.ndarray) -> Tuple[float, float, float]:
    prob = np.abs(psi) ** 2
    mz = float(np.sum(prob * (Z_SUM_DIAG / N_SPINS)))
    zz = float(np.sum(prob * AVG_ZZ_DIAG))

    idx = np.arange(DIM, dtype=np.int64)
    x_val = 0.0 + 0.0j
    for i in range(N_SPINS):
        flipped = idx ^ (1 << i)
        x_val += np.vdot(psi, psi[flipped])
    mx = float(np.real(x_val) / N_SPINS)
    return mx, mz, zz


# ============================================================
# 3) Context utilities
# ============================================================
def make_initial_ctx(key: PRNGKey) -> Array:
    if CLI_ARGS.M0_mode == "canonical":
        if CTX_TOKENS > EMBED_DIM:
            raise ValueError("canonical M0 requires ctx-tokens <= d-model")
        out = jnp.zeros((1, CTX_TOKENS, EMBED_DIM), dtype=jnp.float32)
        ids = jnp.arange(CTX_TOKENS, dtype=jnp.int32)
        return out.at[0, ids, ids].set(jnp.sqrt(float(EMBED_DIM)))

    out = random.normal(key, (1, CTX_TOKENS, EMBED_DIM), dtype=jnp.float32)
    norm = jnp.linalg.norm(out, axis=-1, keepdims=True)
    out = out / jnp.maximum(norm, 1e-8)
    if CLI_ARGS.M0_mode == "random_sqrt_d":
        out = out * jnp.sqrt(float(EMBED_DIM))
    elif CLI_ARGS.M0_mode != "random_unit":
        raise ValueError(CLI_ARGS.M0_mode)
    return out


def project_context(ctx: Array) -> Array:
    if CTX_NORM == "none":
        return ctx
    norm = jnp.linalg.norm(ctx, axis=-1, keepdims=True)
    unit = ctx / jnp.maximum(norm, 1e-8)
    if CTX_NORM == "unit":
        return unit
    if CTX_NORM == "sqrt_d":
        return unit * jnp.sqrt(float(EMBED_DIM))
    raise ValueError(CTX_NORM)


# ============================================================
# 4) Transformer NQS with cross-attention ONLY -- no FNO
# ============================================================
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
        wkey, _ = random.split(key)
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

    def __init__(self, num_embeddings: int, embedding_size: int, *, key: PRNGKey):
        self.weight = flax_embed_init(key, (num_embeddings, embedding_size), jnp.float32)

    def __call__(self, idx: Array) -> Array:
        return self.weight[idx]


class LayerNorm(eqx.Module):
    weight: Array
    bias: Array
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = jnp.ones((dim,), dtype=jnp.float32)
        self.bias = jnp.zeros((dim,), dtype=jnp.float32)
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

    def __init__(self, embed_dim: int, num_heads: int, *, key: PRNGKey):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        kq, kk, kv, ko = random.split(key, 4)
        self.query_proj = Linear(embed_dim, embed_dim, key=kq)
        self.key_proj = Linear(embed_dim, embed_dim, key=kk)
        self.value_proj = Linear(embed_dim, embed_dim, key=kv)
        self.out_proj = Linear(embed_dim, embed_dim, key=ko)

    def __call__(self, query: Array, key: Array, value: Array, mask: Optional[Array] = None) -> Array:
        B, Lq, _ = query.shape
        _, Lk, _ = key.shape
        q = self.query_proj(query).reshape(B, Lq, self.num_heads, self.head_dim)
        k = self.key_proj(key).reshape(B, Lk, self.num_heads, self.head_dim)
        v = self.value_proj(value).reshape(B, Lk, self.num_heads, self.head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        logits = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * (self.head_dim ** -0.5)
        if mask is not None:
            logits = jnp.where(mask[None, None, :, :], logits, jnp.finfo(logits.dtype).min)
        attn = jax.nn.softmax(logits, axis=-1)
        out = jnp.matmul(attn, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, Lq, self.embed_dim)
        return self.out_proj(out)


def causal_mask(seq_len: int) -> Array:
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))


class SpinDecoderBlock(eqx.Module):
    ln1: LayerNorm
    ln2: LayerNorm
    ln3: LayerNorm
    self_attn: MultiHeadDotProductAttention
    cross_attn: MultiHeadDotProductAttention
    mlp_in: Linear
    mlp_out: Linear

    def __init__(self, embed_dim: int, num_heads: int, *, key: PRNGKey):
        k1, k2, k3, k4 = random.split(key, 4)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.ln3 = LayerNorm(embed_dim)
        self.self_attn = MultiHeadDotProductAttention(embed_dim, num_heads, key=k1)
        self.cross_attn = MultiHeadDotProductAttention(embed_dim, num_heads, key=k2)
        self.mlp_in = Linear(embed_dim, 4 * embed_dim, key=k3)
        self.mlp_out = Linear(4 * embed_dim, embed_dim, key=k4)

    def __call__(self, x: Array, ctx: Array, mask: Array) -> Array:
        y = self.ln1(x)
        x = x + self.self_attn(y, y, y, mask=mask)
        y = self.ln2(x)
        x = x + self.cross_attn(y, ctx, ctx)
        y = self.ln3(x)
        x = x + self.mlp_out(jax.nn.gelu(self.mlp_in(y)))
        return x


class ContextQuantumState(eqx.Module):
    embed_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    ctx_tokens: int = eqx.field(static=True)
    phase_mode: str = eqx.field(static=True)
    spin_embed: Embedding
    pos_embed: Embedding
    start_token: Array
    initial_ctx: Array
    blocks: Tuple[SpinDecoderBlock, ...]
    ln_final: LayerNorm
    amp_head: Linear
    phase_head: Linear

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ctx_tokens: int,
        *,
        phase_mode: str,
        key: PRNGKey,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ctx_tokens = ctx_tokens
        self.phase_mode = phase_mode
        keys = random.split(key, num_layers + 6)
        self.spin_embed = Embedding(2, embed_dim, key=keys[0])
        self.pos_embed = Embedding(N_SPINS, embed_dim, key=keys[1])
        self.start_token = 0.02 * random.normal(keys[2], (1, 1, embed_dim), dtype=jnp.float32)
        self.initial_ctx = make_initial_ctx(keys[3])
        self.blocks = tuple(
            SpinDecoderBlock(embed_dim, num_heads, key=keys[4 + i])
            for i in range(num_layers)
        )
        self.ln_final = LayerNorm(embed_dim)
        self.amp_head = Linear(
            embed_dim,
            2,
            key=keys[-2],
            kernel_init=jax.nn.initializers.orthogonal(),
        )
        self.phase_head = Linear(
            embed_dim,
            2,
            key=keys[-1],
            kernel_init=jax.nn.initializers.normal(0.02),
        )

    def get_initial_ctx(self) -> Array:
        return lax.stop_gradient(self.initial_ctx)

    def _decode(self, spins_idx: Array, ctx_tokens: Array) -> Array:
        B, N = spins_idx.shape
        pos = self.pos_embed(jnp.arange(N, dtype=jnp.int32))[None, :, :]
        start = jnp.tile(self.start_token, (B, 1, 1)) + pos[:, 0:1, :]
        prev = self.spin_embed(spins_idx[:, :-1]) + pos[:, 1:, :]
        x = jnp.concatenate([start, prev], axis=1)
        mask = causal_mask(N)
        for block in self.blocks:
            x = block(x, ctx_tokens, mask)
        return self.ln_final(x)

    def log_psi_from_tokens(self, spins_idx: Array, ctx_tokens: Array) -> Array:
        h = self._decode(spins_idx, ctx_tokens)
        logits_amp = self.amp_head(h)
        logits_phase = self.phase_head(h)
        if self.phase_mode == "softsign":
            logits_phase = jax.nn.soft_sign(logits_phase) * jnp.pi
        elif self.phase_mode != "raw":
            raise ValueError(self.phase_mode)
        logp = jax.nn.log_softmax(logits_amp, axis=-1)
        one_hot = jax.nn.one_hot(spins_idx, 2)
        log_amp = 0.5 * jnp.sum(jnp.sum(logp * one_hot, axis=-1), axis=-1)
        phase = jnp.sum(jnp.sum(logits_phase * one_hot, axis=-1), axis=-1)
        return log_amp + 1j * phase


# ============================================================
# 5) Exact overlap machinery, chunked to control memory
# ============================================================
def make_basis_chunks(chunk_size: int):
    chunk_size = min(int(chunk_size), DIM)
    n_chunks = (DIM + chunk_size - 1) // chunk_size
    padded_dim = n_chunks * chunk_size
    idx = np.arange(padded_dim, dtype=np.uint64)
    valid = idx < DIM
    idx_safe = np.where(valid, idx, 0)
    bits = ((idx_safe[:, None] >> np.arange(N_SPINS, dtype=np.uint64)) & 1).astype(np.int32)
    return (
        jnp.asarray(bits.reshape(n_chunks, chunk_size, N_SPINS)),
        jnp.asarray(valid.reshape(n_chunks, chunk_size), dtype=jnp.float32),
    )


SPIN_CHUNKS, VALID_CHUNKS = make_basis_chunks(EXACT_CHUNK_SIZE)
N_CHUNKS = SPIN_CHUNKS.shape[0]
PADDED_DIM = N_CHUNKS * SPIN_CHUNKS.shape[1]


def pad_target(target: Array) -> Array:
    pad = PADDED_DIM - DIM
    if pad == 0:
        return target.reshape(N_CHUNKS, -1)
    target = jnp.pad(target, (0, pad))
    return target.reshape(N_CHUNKS, -1)


def fidelity_to_target(model: ContextQuantumState, ctx: Array, target: Array) -> Array:
    """Exact fidelity |<target|psi_theta(M)>|^2, with chunked basis evaluation."""
    target_chunks = pad_target(target)

    def body(carry, xs):
        overlap, cand_norm = carry
        spins, valid, target_chunk = xs
        ctx_batch = jnp.broadcast_to(ctx[None, :, :], (spins.shape[0], CTX_TOKENS, EMBED_DIM))
        # remat trades compute for memory in the exact-Hilbert-space calculation.
        logpsi = jax.checkpoint(model.log_psi_from_tokens)(spins, ctx_batch)
        cand = jnp.exp(logpsi)
        valid_c = valid.astype(cand.dtype)
        overlap = overlap + jnp.sum(jnp.conj(target_chunk) * cand * valid_c)
        cand_norm = cand_norm + jnp.sum(jnp.abs(cand) ** 2 * valid)
        return (overlap, cand_norm), None

    (overlap, cand_norm), _ = lax.scan(
        body,
        (jnp.asarray(0.0 + 0.0j, dtype=jnp.complex64), jnp.asarray(0.0, dtype=jnp.float32)),
        (SPIN_CHUNKS, VALID_CHUNKS, target_chunks),
    )
    target_norm = jnp.sum(jnp.abs(target) ** 2)
    denom = jnp.maximum(cand_norm * target_norm, 1e-20)
    return jnp.clip(jnp.abs(overlap) ** 2 / denom, 0.0, 1.0)


def full_model_state(model: ContextQuantumState, ctx: Array) -> np.ndarray:
    pieces = []
    for c in range(N_CHUNKS):
        spins = SPIN_CHUNKS[c]
        valid = np.asarray(VALID_CHUNKS[c], dtype=bool)
        ctx_batch = jnp.broadcast_to(ctx[None, :, :], (spins.shape[0], CTX_TOKENS, EMBED_DIM))
        logpsi = model.log_psi_from_tokens(spins, ctx_batch)
        cand = np.asarray(jax.device_get(jnp.exp(logpsi)))
        pieces.append(cand[valid])
    psi = np.concatenate(pieces, axis=0)[:DIM].astype(np.complex64)
    psi = psi / np.linalg.norm(psi)
    return psi


ALL_SPINS_NP = (
    (np.arange(DIM, dtype=np.uint64)[:, None] >> np.arange(N_SPINS, dtype=np.uint64)[None, :]) & 1
).astype(np.int32)


@eqx.filter_jit
def _context_log_derivative_chunk(
    model: ContextQuantumState,
    m_flat: Array,
    spins_idx: Array,
):
    """JIT-compiled chunk of logpsi and d logpsi / d vec(M)."""
    def one_sample(mf, spin):
        m = mf.reshape(CTX_TOKENS, EMBED_DIM)
        z = model.log_psi_from_tokens(spin[None, :], m[None, :, :])[0]
        return jnp.stack((jnp.real(z), jnp.imag(z)))

    def one(spin):
        vals = one_sample(m_flat, spin)
        jac = jax.jacrev(one_sample, argnums=0)(m_flat, spin)
        return vals, jac

    return jax.vmap(one)(spins_idx)


def context_log_derivatives(
    model: ContextQuantumState,
    ctx: Array,
    spins_idx_np: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return log psi(sigma) and O_a(sigma)=d log psi(sigma)/d m_a for real
    context coordinates m=vec(M).  The Jacobian is evaluated in chunks.

    Output shapes:
      logpsi: (Ns,)
      O:      (Ns, P), complex, P = ctx_tokens * embed_dim
    """
    spins_idx_np = np.asarray(spins_idx_np, dtype=np.int32)
    m0 = jnp.asarray(ctx).reshape(-1)
    log_parts = []
    jac_parts = []

    for start in range(0, len(spins_idx_np), SR_JAC_CHUNK_SIZE):
        stop = min(start + SR_JAC_CHUNK_SIZE, len(spins_idx_np))
        vals, jac = _context_log_derivative_chunk(
            model, m0, jnp.asarray(spins_idx_np[start:stop])
        )
        vals = np.asarray(jax.device_get(vals), dtype=np.float64)
        jac = np.asarray(jax.device_get(jac), dtype=np.float64)
        log_parts.append(vals[:, 0] + 1j * vals[:, 1])
        jac_parts.append(jac[:, 0, :] + 1j * jac[:, 1, :])

    return np.concatenate(log_parts, axis=0), np.concatenate(jac_parts, axis=0)


@eqx.filter_jit
def sample_autoregressive_context(key, model: ContextQuantumState, ctx: Array, n_samples: int) -> Array:
    """Independent autoregressive samples, returned as 0/1 spin indices."""
    ctx_batch = jnp.broadcast_to(ctx[None, :, :], (n_samples, CTX_TOKENS, EMBED_DIM))
    spins = jnp.zeros((n_samples, N_SPINS), dtype=jnp.int32)

    def body(carry, i):
        s, rng = carry
        h = model._decode(s, ctx_batch)
        logits = model.amp_head(h)[:, i, :]
        rng, sub = random.split(rng)
        s_i = random.categorical(sub, logits).astype(jnp.int32)
        s = s.at[:, i].set(s_i)
        return (s, rng), None

    (spins, _), _ = lax.scan(body, (spins, key), jnp.arange(N_SPINS, dtype=jnp.int32))
    return spins


@eqx.filter_jit
def local_energy_for_samples(
    model: ContextQuantumState,
    ctx: Array,
    spins_idx: Array,
    hx: Array,
    hz: Array,
) -> Array:
    """TFIM local energy for 0/1 configurations; bit 0 corresponds to Z=+1."""
    ns = spins_idx.shape[0]
    ctx_batch = jnp.broadcast_to(ctx[None, :, :], (ns, CTX_TOKENS, EMBED_DIM))
    z = 1.0 - 2.0 * spins_idx.astype(jnp.float32)
    e_diag = -J_ZZ * jnp.sum(z[:, jnp.asarray(BOND_I_NP)] * z[:, jnp.asarray(BOND_J_NP)], axis=1)
    e_diag = e_diag - hz * jnp.sum(z, axis=1)
    logpsi = model.log_psi_from_tokens(spins_idx, ctx_batch)

    def ratio_for_site(i):
        flipped = spins_idx.at[:, i].set(1 - spins_idx[:, i])
        lp_flip = model.log_psi_from_tokens(flipped, ctx_batch)
        return jnp.exp(lp_flip - logpsi)

    ratios = jax.vmap(ratio_for_site)(jnp.arange(N_SPINS, dtype=jnp.int32))
    return e_diag.astype(ratios.dtype) - hx * jnp.sum(ratios, axis=0)


# ============================================================
# 6) Optional free-context manifold fitting
# ============================================================
class ManifoldBundle(eqx.Module):
    model: ContextQuantumState
    contexts: Array  # (P, T, C, D)


def make_initial_context_table(model: ContextQuantumState, key: PRNGKey, n_protocols: int) -> Array:
    M0 = model.get_initial_ctx()[0]
    table = jnp.broadcast_to(M0[None, None, :, :], (n_protocols, T_STEPS, CTX_TOKENS, EMBED_DIM))
    noise = 0.01 * random.normal(key, table.shape, dtype=table.dtype)
    time_mask = (jnp.arange(T_STEPS) > 0)[None, :, None, None]
    table = table + noise * time_mask
    table = project_context(table)
    return table.at[:, 0].set(M0)


def sampled_fit_loss(
    bundle: ManifoldBundle,
    p_idx: Array,
    t_idx: Array,
    targets: Array,
) -> Array:
    losses = []
    for b in range(FIT_BATCH):
        ctx = bundle.contexts[p_idx[b], t_idx[b]]
        fid = fidelity_to_target(bundle.model, ctx, targets[b])
        losses.append(1.0 - fid)

    fit_loss = jnp.mean(jnp.stack(losses))

    # Always keep the shared t=0 anchor in the objective.
    M0 = bundle.model.get_initial_ctx()[0]
    psi0 = jnp.asarray(initial_physical_state())
    anchor = 1.0 - fidelity_to_target(bundle.model, M0, psi0)

    smooth = jnp.asarray(0.0, dtype=jnp.float32)
    if CLI_ARGS.context_smooth_weight != 0.0:
        prev_t = jnp.maximum(t_idx - 1, 0)
        curr = bundle.contexts[p_idx, t_idx]
        prev = bundle.contexts[p_idx, prev_t]
        smooth = jnp.mean((curr - prev) ** 2)

    return fit_loss + CLI_ARGS.anchor_weight * anchor + CLI_ARGS.context_smooth_weight * smooth


@eqx.filter_jit
def fit_step(bundle, opt_state, optimizer, p_idx, t_idx, targets):
    loss, grads = eqx.filter_value_and_grad(sampled_fit_loss)(bundle, p_idx, t_idx, targets)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(bundle, eqx.is_inexact_array))
    bundle = eqx.apply_updates(bundle, updates)

    # M(0) is a fixed shared coordinate for the fixed physical initial state.
    M0 = bundle.model.get_initial_ctx()[0]
    contexts = project_context(bundle.contexts)
    contexts = contexts.at[:, 0].set(M0)
    bundle = eqx.tree_at(lambda b: b.contexts, bundle, contexts)
    return bundle, opt_state, loss


def save_checkpoint(path: Path, bundle: ManifoldBundle, step: int, opt_state, loss_hist):
    payload = {
        "format": "context_only_local_projection_v1",
        "step": int(step),
        "bundle": jax.device_get(eqx.filter(bundle, eqx.is_array)),
        "opt_state": jax.device_get(opt_state),
        "loss_hist": [float(x) for x in loss_hist],
        "config": vars(CLI_ARGS),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: Path, template: ManifoldBundle):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("format") != "context_only_local_projection_v1":
        raise ValueError(
            "Checkpoint is not from this context-only script. Legacy FNO checkpoints "
            "are intentionally not loaded here because the model structure is different."
        )
    static = eqx.partition(template, eqx.is_array)[1]
    bundle = eqx.combine(payload["bundle"], static)
    return bundle, payload


# ============================================================
# 7a) Context-only SR / real-time TDVP
# Sample-space backend adapts Rende et al., Commun. Phys. 7, 260 (2024), Eq. (10).
# ============================================================
def _sr_centered_arrays(
    O: np.ndarray,
    E_loc: np.ndarray,
    weights: np.ndarray,
):
    """
    Center logarithmic derivatives and local energies with normalized weights.

    O has shape (Ns, P), E_loc has shape (Ns,), weights has shape (Ns,).
    """
    O = np.asarray(O, dtype=np.complex128)
    E_loc = np.asarray(E_loc, dtype=np.complex128)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    mean_O = np.sum(weights[:, None] * O, axis=0)
    mean_E = np.sum(weights * E_loc)
    O_tilde = O - mean_O[None, :]
    E_tilde = E_loc - mean_E
    return O_tilde, E_tilde, weights, mean_E


def _sr_sample_space_factors(
    O_tilde: np.ndarray,
    E_tilde: np.ndarray,
    weights: np.ndarray,
):
    """
    Build the real matrix X and real force vector f such that

        S = X X^T,
        F = X f,

    for real context coordinates and real-time TDVP

        S Mdot = F,
        F_a = Im < O_tilde_a^* E_tilde >.

    For equal Monte-Carlo weights this is the Rende et al. sample-space
    construction.  The sqrt(weights) form also covers exact enumeration.
    """
    sqrt_w = np.sqrt(weights)
    # Y is P x Ns, with Y_ai = sqrt(w_i) O_tilde_i,a.
    Y = O_tilde.T * sqrt_w[None, :]
    e = E_tilde * sqrt_w
    X = np.concatenate((Y.real, Y.imag), axis=1).astype(np.float64, copy=False)
    # Im(Y^* e) = Y_R e_I - Y_I e_R = X [e_I, -e_R].
    f = np.concatenate((e.imag, -e.real), axis=0).astype(np.float64, copy=False)
    return X, f


def _sr_lambda(X: np.ndarray) -> float:
    if CLI_ARGS.sr_reg_mode == "absolute":
        return float(CLI_ARGS.sr_reg)
    trace_s = float(np.sum(X * X))
    scale = trace_s / max(X.shape[0], 1)
    return float(CLI_ARGS.sr_reg) * max(scale, 1e-16)


def _stable_linear_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=CLI_ARGS.sr_rcond)[0]


def _solve_sr_from_X(
    X: np.ndarray,
    f: np.ndarray,
    backend: str,
    lam: float,
):
    """Solve (X X^T + lam I) v = X f in parameter or sample space."""
    P, two_ns = X.shape
    if backend == "auto":
        backend = "parameter" if P <= two_ns else "sample"
    if backend == "parameter":
        S = X @ X.T
        F = X @ f
        v = _stable_linear_solve(S + lam * np.eye(P, dtype=np.float64), F)
    elif backend == "sample":
        G = X.T @ X
        u = _stable_linear_solve(G + lam * np.eye(two_ns, dtype=np.float64), f)
        v = X @ u
        F = X @ f
        S = None
    else:
        raise ValueError(f"Unknown SR backend: {backend}")
    return v, backend, S, F


def _sr_rank_diagnostics(X: np.ndarray):
    min_dim = min(X.shape)
    if min_dim > 2048:
        return -1, np.nan, np.nan
    svals = np.linalg.svd(X, compute_uv=False)
    if svals.size == 0 or svals[0] == 0.0:
        return 0, 0.0, np.inf
    cutoff = float(CLI_ARGS.sr_rcond) * svals[0]
    active = svals[svals > cutoff]
    rank = int(active.size)
    smallest = float(active[-1]) if active.size else 0.0
    cond = float(svals[0] / smallest) if smallest > 0.0 else np.inf
    return rank, float(svals[0]), cond


def sr_statistics(
    model: ContextQuantumState,
    M_t: Array,
    hx: float,
    hz: float,
    key: Optional[PRNGKey],
):
    """Build centered O/E arrays and sample-space factors for exact or MC SR."""
    if CLI_ARGS.sr_data == "exact":
        spins_np = ALL_SPINS_NP
        logpsi, O = context_log_derivatives(model, M_t, spins_np)
        psi = np.exp(logpsi).astype(np.complex128)
        psi /= np.linalg.norm(psi)
        H = sparse_hamiltonian(hx, hz)
        Hpsi = H @ psi
        # Autoregressive probabilities are strictly positive in exact arithmetic.
        # Clip only for numerical underflow in extremely small amplitudes.
        denom = np.where(np.abs(psi) > 1e-300, psi, 1e-300 + 0.0j)
        E_loc = Hpsi / denom
        weights = np.abs(psi) ** 2
    elif CLI_ARGS.sr_data == "mc":
        if key is None:
            raise ValueError("MC SR requires a PRNG key")
        spins = sample_autoregressive_context(key, model, M_t, int(CLI_ARGS.sr_samples))
        spins_np = np.asarray(jax.device_get(spins), dtype=np.int32)
        _, O = context_log_derivatives(model, M_t, spins_np)
        E_loc = np.asarray(
            jax.device_get(
                local_energy_for_samples(
                    model,
                    M_t,
                    spins,
                    jnp.asarray(hx, dtype=jnp.float32),
                    jnp.asarray(hz, dtype=jnp.float32),
                )
            ),
            dtype=np.complex128,
        )
        weights = np.full(len(spins_np), 1.0 / len(spins_np), dtype=np.float64)
    else:
        raise ValueError(CLI_ARGS.sr_data)

    O_tilde, E_tilde, weights, mean_E = _sr_centered_arrays(O, E_loc, weights)
    X, f = _sr_sample_space_factors(O_tilde, E_tilde, weights)
    var_h = float(np.sum(weights * np.abs(E_tilde) ** 2).real)
    return {
        "O_tilde": O_tilde,
        "E_tilde": E_tilde,
        "weights": weights,
        "mean_E": mean_E,
        "X": X,
        "f": f,
        "var_h": var_h,
        "n_data": len(weights),
    }


def solve_sr_mdot(
    model: ContextQuantumState,
    M_t: Array,
    hx: float,
    hz: float,
    key: Optional[PRNGKey] = None,
):
    """Compute the optimal infinitesimal Mdot from context-only real-time TDVP/SR."""
    stats = sr_statistics(model, M_t, hx, hz, key)
    X = stats["X"]
    f = stats["f"]
    lam = _sr_lambda(X)
    mdot_flat, backend, S, F = _solve_sr_from_X(X, f, CLI_ARGS.sr_backend, lam)

    compare_rel = np.nan
    if CLI_ARGS.compare_sr_backends:
        alt = "sample" if backend == "parameter" else "parameter"
        mdot_alt, _, _, _ = _solve_sr_from_X(X, f, alt, lam)
        compare_rel = float(
            np.linalg.norm(mdot_flat - mdot_alt) / max(np.linalg.norm(mdot_flat), 1e-14)
        )

    # Unregularized physical residual evaluated for the regularized solution:
    # ||sum_a v_a T_a + i(H-E)|psi>||^2
    # = Var(H) - 2 F^T v + v^T S v, with S=X X^T.
    Xt_v = X.T @ mdot_flat
    residual_sq = stats["var_h"] - 2.0 * float(F @ mdot_flat) + float(Xt_v @ Xt_v)
    residual_sq = max(residual_sq, 0.0)
    residual_fraction = residual_sq / max(stats["var_h"], 1e-30)
    tangent_coverage = 1.0 - residual_fraction
    rank, sigma_max, cond = _sr_rank_diagnostics(X)

    mdot = mdot_flat.reshape(CTX_TOKENS, EMBED_DIM).astype(np.float32)
    diagnostics = {
        "backend": backend,
        "n_data": int(stats["n_data"]),
        "P": int(SR_P),
        "lambda": float(lam),
        "energy_real": float(np.real(stats["mean_E"])),
        "energy_imag": float(np.imag(stats["mean_E"])),
        "var_h": float(stats["var_h"]),
        "tangent_residual_sq": float(residual_sq),
        "tangent_residual_fraction": float(residual_fraction),
        "tangent_coverage": float(tangent_coverage),
        "effective_rank": int(rank),
        "sigma_max_X": float(sigma_max),
        "condition_X_active": float(cond),
        "backend_relative_disagreement": float(compare_rel),
        "mdot_rms": float(np.sqrt(np.mean(mdot**2))),
    }
    return jnp.asarray(mdot), diagnostics


def sr_euler_step(
    model: ContextQuantumState,
    M_t: Array,
    hx: float,
    hz: float,
    key: Optional[PRNGKey] = None,
):
    if CTX_NORM != "none":
        raise ValueError(
            "Hard --ctx-norm is intentionally disabled for SR: it makes the variational "
            "coordinates constrained. Use --ctx-norm none for the unconstrained TDVP/SR test."
        )
    mdot, diagnostics = solve_sr_mdot(model, M_t, hx, hz, key)
    M_next = M_t + DT * mdot
    return M_next, mdot, diagnostics


# ============================================================
# 7b) Finite-step local projection baseline
# ============================================================
def local_projection_loss(
    delta_M: Array,
    model: ContextQuantumState,
    M_t: Array,
    target_next: Array,
) -> Array:
    M_next = project_context(M_t + delta_M)
    fid = fidelity_to_target(model, M_next, target_next)
    if CLI_ARGS.projection_metric == "phase_l2":
        # For normalized states this is exactly
        # min_phi || |target> - exp(i phi)|candidate> ||^2.
        distance = 2.0 - 2.0 * jnp.sqrt(jnp.clip(fid, 0.0, 1.0))
    elif CLI_ARGS.projection_metric == "infidelity":
        distance = 1.0 - fid
    else:
        raise ValueError(CLI_ARGS.projection_metric)
    reg = CLI_ARGS.projection_reg * jnp.mean(delta_M**2)
    return distance + reg


@eqx.filter_jit
def local_projection_step(delta_M, opt_state, optimizer, model, M_t, target_next):
    loss, grad = jax.value_and_grad(local_projection_loss)(delta_M, model, M_t, target_next)
    updates, opt_state = optimizer.update(grad, opt_state, delta_M)
    delta_M = optax.apply_updates(delta_M, updates)
    return delta_M, opt_state, loss


def solve_local_mdot_finite_projection(
    model: ContextQuantumState,
    M_t: Array,
    target_next_np: np.ndarray,
    delta_init: Optional[Array] = None,
):
    if delta_init is None:
        delta_M = jnp.zeros_like(M_t)
    else:
        delta_M = jnp.asarray(delta_init, dtype=M_t.dtype)

    optimizer = optax.adam(CLI_ARGS.projection_lr)
    opt_state = optimizer.init(delta_M)
    target_next = jnp.asarray(target_next_np)
    best_delta = delta_M
    best_loss = float("inf")

    for _ in range(CLI_ARGS.projection_steps):
        delta_M, opt_state, loss = local_projection_step(
            delta_M, opt_state, optimizer, model, M_t, target_next
        )
        loss_f = float(loss)
        if loss_f < best_loss:
            best_loss = loss_f
            best_delta = delta_M
        if loss_f < CLI_ARGS.projection_tol:
            break

    M_next = project_context(M_t + best_delta)
    effective_delta = M_next - M_t
    Mdot = effective_delta / DT
    fid = 1.0 - float(local_projection_loss(best_delta, model, M_t, target_next))
    # If a regularizer was used, recompute the true fidelity without subtracting the regularizer.
    fid = float(fidelity_to_target(model, M_next, target_next))
    return M_next, Mdot, fid, best_loss


def run_sr_self_tests(model: ContextQuantumState):
    print("--- Running SR correctness self-tests ---")

    # 1) Pure linear-algebra push-through identity.
    rg = np.random.default_rng(123)
    P0, M0 = 11, 7
    X0 = rg.normal(size=(P0, 2 * M0))
    f0 = rg.normal(size=(2 * M0,))
    lam0 = 0.173
    vp = np.linalg.solve(X0 @ X0.T + lam0 * np.eye(P0), X0 @ f0)
    vs = X0 @ np.linalg.solve(X0.T @ X0 + lam0 * np.eye(2 * M0), f0)
    rel_push = np.linalg.norm(vp - vs) / max(np.linalg.norm(vp), 1e-15)
    print(f"[test] push-through identity relative error = {rel_push:.3e}")
    if rel_push > 1e-10:
        raise AssertionError("Push-through identity implementation failed")

    # 2) Model-level exact SR construction on the current tiny test model.
    if DIM > 4096:
        print("[test] skipping model-level exact tests because DIM > 4096")
        print("--- SR self-tests passed ---")
        return

    ctx = model.get_initial_ctx()[0]
    hx, hz = 0.73, -0.041
    logpsi, O = context_log_derivatives(model, ctx, ALL_SPINS_NP)
    psi = np.exp(logpsi).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    H = sparse_hamiltonian(hx, hz)
    Hpsi = H @ psi
    E_loc = Hpsi / np.where(np.abs(psi) > 1e-300, psi, 1e-300 + 0.0j)
    weights = np.abs(psi) ** 2
    O_tilde, E_tilde, weights, mean_E = _sr_centered_arrays(O, E_loc, weights)
    X, f = _sr_sample_space_factors(O_tilde, E_tilde, weights)

    S_x = X @ X.T
    F_x = X @ f
    S_direct = np.real(O_tilde.conj().T @ (weights[:, None] * O_tilde))
    F_direct = np.imag(O_tilde.conj().T @ (weights * E_tilde))
    rel_S = np.linalg.norm(S_x - S_direct) / max(np.linalg.norm(S_direct), 1e-15)
    rel_F = np.linalg.norm(F_x - F_direct) / max(np.linalg.norm(F_direct), 1e-15)
    print(f"[test] S factorization relative error = {rel_S:.3e}")
    print(f"[test] real-time force factorization relative error = {rel_F:.3e}")
    if rel_S > 5e-7 or rel_F > 5e-7:
        raise AssertionError("SR sample-space factors do not reproduce S/F")

    # 3) Check the real-time force sign against a direct projective tangent least squares.
    T = psi[:, None] * O_tilde
    b = -1j * (Hpsi - mean_E * psi)
    S_tan = np.real(T.conj().T @ T)
    F_tan = np.real(T.conj().T @ b)
    rel_St = np.linalg.norm(S_tan - S_direct) / max(np.linalg.norm(S_direct), 1e-15)
    rel_Ft = np.linalg.norm(F_tan - F_direct) / max(np.linalg.norm(F_direct), 1e-15)
    print(f"[test] tangent-metric relative error = {rel_St:.3e}")
    print(f"[test] tangent-force/sign relative error = {rel_Ft:.3e}")
    if rel_St > 5e-7 or rel_Ft > 5e-7:
        raise AssertionError("Real-time TDVP sign or metric is inconsistent with direct tangent least squares")

    # 4) Parameter-space and Rende/sample-space solves must agree for the same lambda.
    trace_scale = np.trace(S_direct) / max(S_direct.shape[0], 1)
    lam = 1e-3 * max(float(trace_scale), 1e-12)
    v_param, _, _, _ = _solve_sr_from_X(X, f, "parameter", lam)
    v_sample, _, _, _ = _solve_sr_from_X(X, f, "sample", lam)
    rel_backend = np.linalg.norm(v_param - v_sample) / max(np.linalg.norm(v_param), 1e-15)
    print(f"[test] parameter vs sample-space SR relative error = {rel_backend:.3e}")
    if rel_backend > 5e-6:
        raise AssertionError("Parameter- and sample-space SR backends disagree")

    # 5) Check one context log-derivative against a central finite difference.
    pidx = min(3, SR_P - 1)
    spin0 = jnp.asarray(ALL_SPINS_NP[min(1, DIM - 1)])
    mflat = np.asarray(jax.device_get(ctx)).reshape(-1).astype(np.float64)
    eps = 2e-4
    mp = mflat.copy(); mp[pidx] += eps
    mm = mflat.copy(); mm[pidx] -= eps
    zp = model.log_psi_from_tokens(
        spin0[None, :], jnp.asarray(mp.reshape(CTX_TOKENS, EMBED_DIM), dtype=ctx.dtype)[None, :, :]
    )[0]
    zm = model.log_psi_from_tokens(
        spin0[None, :], jnp.asarray(mm.reshape(CTX_TOKENS, EMBED_DIM), dtype=ctx.dtype)[None, :, :]
    )[0]
    fd = complex(np.asarray(jax.device_get((zp - zm) / (2.0 * eps))))
    ad = O[min(1, DIM - 1), pidx]
    rel_jac = abs(fd - ad) / max(abs(ad), 1e-8)
    print(f"[test] context log-derivative finite-difference relative error = {rel_jac:.3e}")
    if rel_jac > 5e-2:
        raise AssertionError("Context log-derivative autodiff check failed")

    # 6) A sufficiently small Euler step along +Mdot should be at least as good
    # as the opposite direction for the finite-step Schrödinger target.
    dt_test = min(1e-3, max(DT * 1e-2, 1e-5))
    target = exp_h_step(psi, hx, hz, dt_test)
    vj = jnp.asarray(v_param.reshape(CTX_TOKENS, EMBED_DIM), dtype=ctx.dtype)
    fid_plus = float(fidelity_to_target(model, ctx + dt_test * vj, jnp.asarray(target)))
    fid_minus = float(fidelity_to_target(model, ctx - dt_test * vj, jnp.asarray(target)))
    print(f"[test] finite-dt directional fidelity: plus={fid_plus:.9f}, minus={fid_minus:.9f}")
    if fid_plus + 5e-7 < fid_minus:
        raise AssertionError("Real-time SR direction appears to have the wrong sign")

    print("--- SR self-tests passed ---")


# ============================================================
# 8) Fit the manifold, then freeze theta and test M-only dynamics
# ============================================================
print("--- Context-only NQS: NO FNO / NO neural operator ---")
print(f"Lattice: {LX} x {LY} (N={N_SPINS}, Hilbert dim={DIM})")
print(f"Context: {CTX_TOKENS} tokens x {EMBED_DIM} dimensions (P={SR_P})")
print(f"M0 mode: {CLI_ARGS.M0_mode}; hard context norm: {CTX_NORM}")
print(f"Mdot method: {CLI_ARGS.mdot_method}")
if CLI_ARGS.mdot_method == "sr":
    print(
        f"SR: data={CLI_ARGS.sr_data}, backend={CLI_ARGS.sr_backend}, "
        f"samples={CLI_ARGS.sr_samples}, reg={CLI_ARGS.sr_reg} ({CLI_ARGS.sr_reg_mode})"
    )
else:
    print(
        f"Finite projection: {CLI_ARGS.projection_steps} Adam steps, "
        f"lr={CLI_ARGS.projection_lr}, metric={CLI_ARGS.projection_metric}"
    )
print(f"dt={DT:.6g}; local H convention={CLI_ARGS.local_H}")

if CLI_ARGS.mdot_method == "sr" and CTX_NORM != "none":
    raise ValueError(
        "For the SR/TDVP experiment use --ctx-norm none. Hard normalization imposes "
        "constraints on M and requires a constrained TDVP solve."
    )

rng = random.PRNGKey(SEED)
model_key, ctx_key, rng = random.split(rng, 3)
model = ContextQuantumState(
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    CTX_TOKENS,
    phase_mode=PHASE_MODE,
    key=model_key,
)

if CLI_ARGS.self_test:
    run_sr_self_tests(model)
    raise SystemExit(0)

contexts = make_initial_context_table(model, ctx_key, CLI_ARGS.train_protocols)
bundle = ManifoldBundle(model=model, contexts=contexts)
bundle_template = bundle

train_key = random.PRNGKey(CLI_ARGS.train_protocol_key)
train_fields = np.asarray(
    generate_grf_trajectories(train_key, CLI_ARGS.train_protocols, T_STEPS, T_MAX),
    dtype=np.float32,
)

print("Generating exact training trajectories...")
train_states_np = np.stack(
    [exact_time_dependent_trajectory(train_fields[p]) for p in tqdm(range(CLI_ARGS.train_protocols))],
    axis=0,
)

schedule = optax.exponential_decay(
    CLI_ARGS.fit_lr,
    CLI_ARGS.fit_decay_steps,
    CLI_ARGS.fit_decay_rate,
    end_value=CLI_ARGS.fit_min_lr,
)
fit_optimizer = optax.chain(optax.clip_by_global_norm(0.1), optax.adam(schedule))
fit_opt_state = fit_optimizer.init(eqx.filter(bundle, eqx.is_inexact_array))
loss_hist = []
start_step = 0

if RESUME_PATH is not None:
    bundle, payload = load_checkpoint(RESUME_PATH, bundle_template)
    fit_opt_state = payload["opt_state"]
    loss_hist = [float(x) for x in payload.get("loss_hist", [])]
    start_step = int(payload.get("step", -1)) + 1
    print(f"Resumed context-only checkpoint at fit step {start_step}.")

if CLI_ARGS.fit_steps > start_step:
    print("Fitting shared transformer + FREE context table (still no neural operator)...")
    pbar = tqdm(range(start_step, CLI_ARGS.fit_steps), initial=start_step, total=CLI_ARGS.fit_steps)
    for step in pbar:
        rng, kp, kt = random.split(rng, 3)
        p_idx = random.randint(kp, (FIT_BATCH,), 0, CLI_ARGS.train_protocols)
        t_idx = random.randint(kt, (FIT_BATCH,), 1, T_STEPS)
        p_np = np.asarray(jax.device_get(p_idx), dtype=np.int32)
        t_np = np.asarray(jax.device_get(t_idx), dtype=np.int32)
        targets = jnp.asarray(train_states_np[p_np, t_np])
        bundle, fit_opt_state, loss = fit_step(
            bundle, fit_opt_state, fit_optimizer, p_idx, t_idx, targets
        )
        loss_f = float(loss)
        loss_hist.append(loss_f)
        if step % CLI_ARGS.print_every == 0:
            pbar.set_description(f"fit loss={np.mean(loss_hist[-30:]):.4e}")
        if step > 0 and step % 500 == 0:
            save_checkpoint(CHECKPOINT_PATH, bundle, step, fit_opt_state, loss_hist)
else:
    print("Skipping free-context manifold fitting.")

save_checkpoint(
    CHECKPOINT_PATH,
    bundle,
    max(CLI_ARGS.fit_steps - 1, start_step - 1),
    fit_opt_state,
    loss_hist,
)
print(f"Saved context-only checkpoint to {CHECKPOINT_PATH.resolve()}")

print("Evaluating free-context/oracle training fidelities...")
oracle_fids = np.zeros((CLI_ARGS.train_protocols, T_STEPS), dtype=np.float32)
for pidx in range(CLI_ARGS.train_protocols):
    for tidx in range(T_STEPS):
        oracle_fids[pidx, tidx] = float(
            fidelity_to_target(
                bundle.model,
                bundle.contexts[pidx, tidx],
                jnp.asarray(train_states_np[pidx, tidx]),
            )
        )
print(
    f"Oracle/free-context fit: mean fidelity={oracle_fids.mean():.6f}, "
    f"worst fidelity={oracle_fids.min():.6f}"
)

# Freeze theta here. From this point onward only M is changed.
model = bundle.model

test_key = random.PRNGKey(CLI_ARGS.test_protocol_key)
test_fields = np.asarray(
    generate_grf_trajectories(test_key, 1, T_STEPS, T_MAX)[0], dtype=np.float32
)
print("Generating held-out exact reference trajectory...")
exact_test = exact_time_dependent_trajectory(test_fields)

M_t = model.get_initial_ctx()[0]
M_traj = [np.asarray(jax.device_get(M_t), dtype=np.float32)]
Mdot_traj = []
local_step_fids = []
local_step_losses = []
sr_diagnostics = []
global_fids = []
mx_nqs, mz_nqs, zz_nqs = [], [], []
mx_exact, mz_exact, zz_exact = [], [], []

psi_model = full_model_state(model, M_t)
global_fids.append(float(np.abs(np.vdot(exact_test[0], psi_model)) ** 2))
o = compute_observables(psi_model)
mx_nqs.append(o[0]); mz_nqs.append(o[1]); zz_nqs.append(o[2])
o = compute_observables(exact_test[0])
mx_exact.append(o[0]); mz_exact.append(o[1]); zz_exact.append(o[2])

prev_mdot = None
print("Rollout: transformer weights are frozen; evolving only M(t)...")
for t in tqdm(range(T_STEPS - 1)):
    psi_model_t = full_model_state(model, M_t)
    if CLI_ARGS.local_H == "left":
        hx_loc = float(test_fields[t, 0])
        hz_loc = float(test_fields[t, 1])
    else:
        hx_loc = 0.5 * (float(test_fields[t, 0]) + float(test_fields[t + 1, 0]))
        hz_loc = 0.5 * (float(test_fields[t, 1]) + float(test_fields[t + 1, 1]))

    # This finite-step target is used as a common diagnostic for both methods.
    target_local = exp_h_step(psi_model_t, hx_loc, hz_loc, DT)

    if CLI_ARGS.mdot_method == "sr":
        rng, k_sr = random.split(rng)
        M_next, Mdot, diag = sr_euler_step(
            model,
            M_t,
            hx_loc,
            hz_loc,
            key=k_sr if CLI_ARGS.sr_data == "mc" else None,
        )
        proj_fid = float(fidelity_to_target(model, M_next, jnp.asarray(target_local)))
        proj_loss = 1.0 - proj_fid
        diag["local_finite_step_fidelity"] = proj_fid
        sr_diagnostics.append(diag)
    else:
        delta_init = None
        if CLI_ARGS.warm_start_mdot and prev_mdot is not None:
            delta_init = DT * prev_mdot
        M_next, Mdot, proj_fid, proj_loss = solve_local_mdot_finite_projection(
            model, M_t, target_local, delta_init=delta_init
        )
        sr_diagnostics.append({})

    Mdot_traj.append(np.asarray(jax.device_get(Mdot), dtype=np.float32))
    local_step_fids.append(proj_fid)
    local_step_losses.append(proj_loss)
    M_t = M_next
    prev_mdot = Mdot
    M_traj.append(np.asarray(jax.device_get(M_t), dtype=np.float32))

    psi_model = full_model_state(model, M_t)
    global_fids.append(float(np.abs(np.vdot(exact_test[t + 1], psi_model)) ** 2))
    o = compute_observables(psi_model)
    mx_nqs.append(o[0]); mz_nqs.append(o[1]); zz_nqs.append(o[2])
    o = compute_observables(exact_test[t + 1])
    mx_exact.append(o[0]); mz_exact.append(o[1]); zz_exact.append(o[2])

M_traj = np.asarray(M_traj, dtype=np.float32)
Mdot_traj = np.asarray(Mdot_traj, dtype=np.float32)
local_step_fids = np.asarray(local_step_fids, dtype=np.float32)
global_fids = np.asarray(global_fids, dtype=np.float32)
mdot_rms = np.sqrt(np.mean(Mdot_traj**2, axis=(1, 2))) if len(Mdot_traj) else np.zeros(0)

if CLI_ARGS.mdot_method == "sr":
    tangent_coverage = np.asarray([d["tangent_coverage"] for d in sr_diagnostics], dtype=np.float64)
    tangent_residual_fraction = np.asarray(
        [d["tangent_residual_fraction"] for d in sr_diagnostics], dtype=np.float64
    )
    sr_lambda = np.asarray([d["lambda"] for d in sr_diagnostics], dtype=np.float64)
    sr_rank = np.asarray([d["effective_rank"] for d in sr_diagnostics], dtype=np.int32)
    sr_condition = np.asarray([d["condition_X_active"] for d in sr_diagnostics], dtype=np.float64)
    backend_disagreement = np.asarray(
        [d["backend_relative_disagreement"] for d in sr_diagnostics], dtype=np.float64
    )
    sr_backend_used = [d["backend"] for d in sr_diagnostics]
else:
    tangent_coverage = np.full(T_STEPS - 1, np.nan)
    tangent_residual_fraction = np.full(T_STEPS - 1, np.nan)
    sr_lambda = np.full(T_STEPS - 1, np.nan)
    sr_rank = np.full(T_STEPS - 1, -1, dtype=np.int32)
    sr_condition = np.full(T_STEPS - 1, np.nan)
    backend_disagreement = np.full(T_STEPS - 1, np.nan)
    sr_backend_used = []

summary = {
    "mdot_method": CLI_ARGS.mdot_method,
    "sr_data": CLI_ARGS.sr_data if CLI_ARGS.mdot_method == "sr" else None,
    "sr_backend_requested": CLI_ARGS.sr_backend if CLI_ARGS.mdot_method == "sr" else None,
    "sr_backend_used": sorted(set(sr_backend_used)) if sr_backend_used else [],
    "oracle_train_mean_fidelity": float(oracle_fids.mean()),
    "oracle_train_worst_fidelity": float(oracle_fids.min()),
    "local_step_mean_fidelity": float(local_step_fids.mean()),
    "local_step_worst_fidelity": float(local_step_fids.min()),
    "global_rollout_mean_fidelity": float(global_fids.mean()),
    "global_rollout_final_fidelity": float(global_fids[-1]),
    "mx_mae": float(np.mean(np.abs(np.asarray(mx_nqs) - np.asarray(mx_exact)))),
    "zz_mae": float(np.mean(np.abs(np.asarray(zz_nqs) - np.asarray(zz_exact)))),
}
if CLI_ARGS.mdot_method == "sr":
    summary.update(
        {
            "mean_tangent_coverage": float(np.nanmean(tangent_coverage)),
            "worst_tangent_coverage": float(np.nanmin(tangent_coverage)),
            "mean_tangent_residual_fraction": float(np.nanmean(tangent_residual_fraction)),
            "median_effective_rank": float(np.median(sr_rank[sr_rank >= 0])) if np.any(sr_rank >= 0) else None,
            "max_backend_relative_disagreement": (
                float(np.nanmax(backend_disagreement))
                if np.any(np.isfinite(backend_disagreement))
                else None
            ),
        }
    )
print(json.dumps(summary, indent=2))

np.savez(
    OUTPUT_DIR / "mdot_results.npz",
    fields=test_fields,
    M=M_traj,
    Mdot=Mdot_traj,
    local_step_fids=local_step_fids,
    local_step_losses=np.asarray(local_step_losses, dtype=np.float32),
    global_fids=global_fids,
    mdot_rms=mdot_rms,
    tangent_coverage=tangent_coverage,
    tangent_residual_fraction=tangent_residual_fraction,
    sr_lambda=sr_lambda,
    sr_rank=sr_rank,
    sr_condition=sr_condition,
    backend_disagreement=backend_disagreement,
    mx_nqs=np.asarray(mx_nqs, dtype=np.float32),
    mx_exact=np.asarray(mx_exact, dtype=np.float32),
    mz_nqs=np.asarray(mz_nqs, dtype=np.float32),
    mz_exact=np.asarray(mz_exact, dtype=np.float32),
    zz_nqs=np.asarray(zz_nqs, dtype=np.float32),
    zz_exact=np.asarray(zz_exact, dtype=np.float32),
    oracle_fids=oracle_fids,
)
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
with open(OUTPUT_DIR / "sr_diagnostics.json", "w") as f:
    json.dump(sr_diagnostics, f, indent=2)

# Diagnostics plot.
t_axis = np.linspace(0.0, T_MAX, T_STEPS)
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes[0, 0].plot(t_axis, global_fids)
axes[0, 0].set_ylim(0.0, 1.02)
axes[0, 0].set_title("Global fidelity vs exact trajectory")
axes[0, 0].set_xlabel("t")

axes[0, 1].semilogy(t_axis[1:], np.maximum(1.0 - local_step_fids, 1e-12))
axes[0, 1].set_title("Finite-step local infidelity")
axes[0, 1].set_xlabel("t")

axes[0, 2].plot(t_axis[1:], mdot_rms)
axes[0, 2].set_title("RMS context speed")
axes[0, 2].set_xlabel("t")

axes[1, 0].plot(t_axis, mx_nqs, label="context NQS")
axes[1, 0].plot(t_axis, mx_exact, "--", label="exact")
axes[1, 0].set_title("<X>")
axes[1, 0].legend()

axes[1, 1].plot(t_axis, zz_nqs, label="context NQS")
axes[1, 1].plot(t_axis, zz_exact, "--", label="exact")
axes[1, 1].set_title("<ZZ>")
axes[1, 1].legend()

if CLI_ARGS.mdot_method == "sr":
    axes[1, 2].plot(t_axis[1:], tangent_coverage)
    axes[1, 2].set_ylim(-0.05, 1.05)
    axes[1, 2].set_title("TDVP tangent-space coverage")
else:
    axes[1, 2].axis("off")
axes[1, 2].set_xlabel("t")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "mdot_diagnostics.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved results to {OUTPUT_DIR.resolve()}")
