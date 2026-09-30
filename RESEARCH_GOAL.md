# Research Goal and Motivation

## Main goal

Determine whether a transformer neural quantum state conditioned by a small number of context tokens can serve as a **compact dynamical manifold for many-body quantum states**.

Write the state as

`|psi(M)> = |psi_theta(M)>`,

where `theta` denotes all shared transformer weights and `M` denotes a comparatively small set of real context coordinates. During the dynamical rollout, `theta` is frozen and only `M` changes.

The core test is whether the exact Schrodinger velocity

`-i H(t) |psi(M)>`

can be accurately represented by motion tangent to the context manifold,

`sum_a Mdot_a |partial_a psi(M)>`.

The optimal infinitesimal `Mdot` follows from TDVP / stochastic reconfiguration.

## Why this question matters

In the earlier NOQS/UNP construction, a neural operator maps a whole Hamiltonian protocol `H(.)` to `M(t)` or `Mdot(t)`. That works computationally, but leaves several conceptual questions:

- What exactly is `M(t)`?
- Is a neural operator fundamentally needed?
- How expressive is the context-conditioned NQS before learning any map from `H` to `M`?
- How many context coordinates are physically active?
- Can the context manifold generalize across driving protocols once it has learned useful geometry?

This project isolates those questions.

## Context-space TDVP/SR equation

Flatten all context tokens into real coordinates `m_a`. Define

`O_a(sigma) = partial_{m_a} log psi(sigma; M)`

and the centered quantities

`Otilde_a = O_a - <O_a>`

and

`Etilde_loc = E_loc - <E_loc>`.

For real-time evolution, the context-space quantum geometric matrix and force are

`S_ab = Re < Otilde_a^* Otilde_b >`

and

`F_a = Im < Otilde_a^* Etilde_loc >`.

Then

`S Mdot = F`.

A regularized/pseudoinverse solution gives the best tangent-space approximation to Schrodinger evolution while all transformer parameters remain fixed.

## Central hypotheses

### H1. Low-dimensional dynamical manifold
A context dimension much smaller than the full neural-network parameter count can track physically relevant trajectories with high fidelity.

### H2. Geometry learned from few trajectories transfers
Training the shared decoder plus free contexts on one or a few trajectories can produce a tangent geometry that supports context-only TDVP on previously unseen driving protocols.

### H3. Tangent coverage predicts reliability
Define the normalized residual of the optimal tangent projection. A useful form is

`coverage = 1 - ||R||^2 / Var(H)`.

If this quantity decreases before the global state fidelity fails, it becomes an internal physics-based reliability diagnostic that does not require exact target states.

### H4. The latent dynamics is control-affine
For

`H(t) = H0 + sum_mu h_mu(t) H_mu`,

`S(M)` is independent of the instantaneous control coefficients while the TDVP force is linear in `H`. Therefore

`Mdot = f0(M) + sum_mu h_mu(t) f_mu(M)`.

This gives a strong structural interpretation: the neural operator in the original architecture can be viewed as amortizing a physics-derived controlled latent ODE.

### H5. More context is not automatically better
Increasing context dimension should enlarge the tangent space, but may worsen conditioning or integration stiffness. The important tradeoff is among expressivity, effective rank, conditioning, and stable dynamics.

## What would make the project publishable

A convincing paper should go beyond `N=6` and establish several of the following:

1. Scaling to `N ~ 10-16` or beyond using stochastic/sample-space SR.
2. Multiple training and test seeds with uncertainty/error bars.
3. Robust OOD protocol families and longer-time rollouts.
4. A quantitative relation between tangent coverage and future/global fidelity.
5. Context-dimension/effective-rank scaling with system size and trajectory complexity.
6. Comparison against full-parameter NQS TDVP, ordinary time-conditioned NQS, and the original FNO/NOQS construction.
7. Separation of manifold error from integration error using direct free-context or finite-step projection oracles.
8. Regularization and integrator studies demonstrating that conclusions are not numerical artifacts.

## Possible paper-level message if the hypotheses survive

A concise conceptual message could be:

> A pretrained context-conditioned neural quantum state defines a reusable variational manifold. Quantum dynamics on this manifold can be generated directly by a low-dimensional TDVP/SR equation for the context tokens. Neural operators are therefore optional amortized solvers, while tangent-space coverage gives a geometric measure of dynamical representability and reliability.

This is a hypothesis to test, not a conclusion to assume.
