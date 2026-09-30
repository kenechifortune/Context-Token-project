# Environment Setup

## Recommended: begin with the PyTorch exact small-system code

Create a virtual environment and install:

```
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
pip install -r requirements_core.txt
```

Then run:

```
bash code/run_smoke_tests.sh
```

The exploratory PyTorch implementation uses exact Hilbert-space enumeration and is intended for small systems where `2^N` is manageable.

## Optional scalable JAX path

Install the packages in `requirements_jax_optional.txt`. For GPU JAX, follow the current official JAX installation instructions appropriate to the machine/CUDA version rather than blindly installing a CPU wheel.

The JAX script was inherited from the original project and is intended for larger stochastic/sample-space SR experiments.

## Known environment note from the packaging session

At packaging time, the available environment had PyTorch, JAX, NumPy, SciPy, and Matplotlib, but did not have Equinox or Optax. Therefore the independent SR math tests and PyTorch smoke tests were executable here, while the full JAX/Equinox run was only syntax-checked. See `TEST_STATUS.md`.
