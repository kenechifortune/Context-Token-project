# Experiment Roadmap Toward a Publishable Result

## Phase 0: Verify the implementation

Required before scientific interpretation:

- Run the independent SR math tests.
- Check `+Mdot` versus `-Mdot` for an infinitesimal exact Schrodinger step.
- Compare finite-step local projection with SR as `dt -> 0`.
- Compare parameter-space and sample-space SR where both are feasible.
- Verify control-affine decomposition numerically.
- Check convergence versus SR regularization, SVD cutoff, and time step.

## Phase 1: Reproduce the preliminary N=6 result

Use the supplied reproduction script.

Questions:

- Does one-trajectory pretraining consistently outperform anchor-only training?
- Does OOD context-SR rollout stay near the free-context oracle?
- Does tangent coverage correlate with global fidelity under stress?

Repeat with at least 5 network seeds and several field seeds.

## Phase 2: Scale system size

Target sequence:

- `N=8`
- `N=10`
- `N=12`
- then `N=14-16` if feasible.

For exact enumeration, stop when Jacobian/Hilbert-space cost becomes prohibitive. Then use autoregressive sampling and the sample-space Rende SR backend.

At each size record:

- context dimension `P = Nc * d`;
- number of samples;
- SR effective rank;
- tangent coverage;
- global fidelity when exact reference is available;
- local observables when exact full fidelity is unavailable;
- wall time and memory.

## Phase 3: Context-capacity and geometry scaling

Sweep both:

- number of context tokens `Nc`;
- context/embedding dimension `d`.

Do not report only raw `P`. Also report:

- numerical rank of `S`;
- effective rank from eigenvalue participation ratio;
- condition number;
- performance versus active rank.

Goal: determine whether the physically active context dimension grows slowly or rapidly with system size/time/entanglement.

## Phase 4: Reliability diagnostic

Build diverse stress families:

- increasing pulse amplitude;
- increasing drive frequency;
- narrowing pulses;
- chirps;
- disorder;
- longer time horizons;
- protocols near finite-size critical regions.

At every time point store tangent coverage and future fidelity degradation. Test whether coverage predicts failure with:

- Pearson/Spearman correlation;
- threshold detection curves;
- calibration plots;
- lead time before a fidelity threshold is crossed.

A successful a-posteriori error indicator would be a strong result.

## Phase 5: Matched baselines

Important baselines:

1. Anchor-only decoder + context SR.
2. One-trajectory decoder + context SR.
3. Multiple-trajectory decoder + context SR.
4. Fixed-context decoder (no evolution).
5. Full-parameter NQS TDVP/SR.
6. Explicit time-conditioned NQS with similar parameter count.
7. Original FNO/NOQS architecture at matched transformer capacity.
8. Free-context oracle: optimize `M` separately for each exact target state.
9. Finite-step local projection onto the context manifold.

The free-context oracle separates manifold expressivity from dynamical integration error.

## Phase 6: Test the control-affine structure

For Hamiltonians affine in controls, numerically verify

`Mdot(M,h) = f0(M) + sum_mu h_mu f_mu(M)`

across many contexts, not just one point.

Then investigate whether the vector fields `f_mu(M)` can themselves be approximated with a small local model. This may yield a simpler causal alternative to the full neural operator.

## Phase 7: Longer-time stability

Compare Euler, Heun, midpoint/RK methods, and adaptive step size. If a larger context gives better tangent coverage but worse rollout fidelity, test whether the problem is stiffness rather than expressivity.

Record step-size convergence to separate numerical integration error from manifold error.

## Minimum standard before writing a paper

- Multiple independent seeds.
- At least one size beyond the preliminary exact toy regime.
- Robust baselines.
- Clear uncertainty estimates.
- A failure regime.
- Code and commands sufficient for reproduction.
- No claim of universality unless the state/protocol class is precisely scoped.
