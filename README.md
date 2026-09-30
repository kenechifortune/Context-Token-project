# Context-Space SR/TDVP for Neural Quantum States

## Start here

This folder is a self-contained research handoff for a student project on **context-only quantum dynamics**.

The central research question is:

> **Can a fixed neural-network quantum-state decoder, conditioned only by a small set of context tokens `M`, form a reusable low-dimensional variational manifold for quantum time evolution, with `Mdot` determined directly by physics through TDVP / stochastic reconfiguration rather than predicted by a neural operator?**

The project grew out of the NOQS and UNP architectures in the included reference papers. In those works a neural operator maps a time-dependent Hamiltonian to context tokens `M(t)` or `Mdot(t)`. Here we deliberately remove the neural operator and ask a more basic representation question:

1. Fix a transformer NQS decoder `psi_theta(sigma; M)`.
2. Treat only `M` as dynamical coordinates.
3. Compute the best local velocity `Mdot` by projecting the Schrodinger vector field onto the context tangent space.
4. Evolve `M(t)` and measure how faithfully the resulting state tracks exact quantum dynamics.

This separation is scientifically useful because it distinguishes two questions that were mixed together in the original neural-operator construction:

- **Representation:** Is the context-conditioned NQS manifold expressive enough to contain the relevant quantum trajectories?
- **Amortization:** Can a neural operator efficiently predict the correct path on that manifold from `H(t)`?

If context-only TDVP works, the neural operator is not fundamentally needed to *define* the dynamics. It can instead be interpreted as an amortized solver for a lower-dimensional physics-derived latent flow.

## What is currently promising

For an exploratory exact-Hilbert-space `3 x 2` TFIM (`N=6`), a small transformer with only 24 real context coordinates was trained on one smooth Fourier-series trajectory. The transformer was then frozen and only `M(t)` was evolved by context-space SR/TDVP on four qualitatively different out-of-distribution drives. Preliminary mean OOD fidelity was about `0.983`, final-time OOD fidelity about `0.956`, and mean optimal tangent-space coverage about `0.856`. An anchor-only model that learned only the initial state performed much worse. See `results/preliminary/PRELIMINARY_RESULTS.md`.

These small-system observations are **not yet publication-quality evidence**. Their value is that they identify sharp hypotheses that can be tested at larger size and with stronger controls.

## Suggested order for a new student

1. Read `RESEARCH_GOAL.md`.
2. Read `notes/PEDAGOGICAL_NOTES.md`.
3. Run `bash code/run_smoke_tests.sh`.
4. Reproduce the small `N=6` experiment with `bash code/reproduce_preliminary.sh`.
5. Read `EXPERIMENT_ROADMAP.md` and choose one focused extension.
6. Keep a research log with exact commands, git commit hashes, random seeds, and output folders.

## Code map

- `code/context_sr_explore_torch.py` — simplest exact-Hilbert-space exploratory implementation; best place to learn the method and modify small-system experiments.
- `code/learnMdot_sr_jax.py` — scalable JAX/Equinox implementation with SR/sample-space backends; intended for larger Monte Carlo work.
- `code/learnMdot_local_projection.py` — earlier finite-step local-projection formulation; useful as an independent comparison against infinitesimal SR/TDVP.
- `code/learnMdot_best_original.py` — original FNO-based baseline supplied before the context-only modifications.
- `code/test_learnMdot_sr_math.py` — independent mathematical tests of the SR equations and sample-space identity.
- `code/validate_torch_sr.py` — portable checkpoint-level checks of the SR direction and control-affine identity.
- `code/reproduce_preliminary.sh` — commands for the main small-system preliminary experiments.
- `code/run_smoke_tests.sh` — fast sanity checks.

## Reference papers

The included PDFs are the two project-origin papers and the Rende et al. scalable SR paper. They were supplied in the research conversation and are included here for the student's convenience.

## Important scientific discipline

Do not optimize for attractive plots. Try to falsify the central hypothesis. Always include baselines, report failures, and distinguish:

- decoder/manifold error,
- local tangent projection error,
- numerical integration error,
- Monte Carlo/statistical error,
- generalization error.

A negative result about where the context manifold fails can be scientifically valuable if it is systematic and well diagnosed.
