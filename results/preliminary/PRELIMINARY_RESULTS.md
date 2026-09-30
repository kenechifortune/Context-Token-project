# Preliminary context-only SR results

These are exploratory exact-Hilbert-space results for a 3x2 (N=6) driven transverse-field Ising model. The transformer weights are frozen during rollout; only the context tokens M(t) move according to the exact context-space SR/TDVP equation. The training stage contains **no FNO/neural operator**.

## Strongest current result

A transformer/context manifold trained on **one Fourier-series trajectory** generalized, via physics-derived context SR, to four qualitatively different OOD drives (Gaussian pulse, tanh ramp, chirp, and a two-pulse protocol):

- OOD mean fidelity: 0.982622
- OOD final-time fidelity: 0.955837
- mean optimal tangent coverage: 0.855805
- training-oracle fidelity: 0.989263
- held-out OOD oracle fidelity after directly optimizing M for each target state: 0.989106
- held-out OOD oracle final-time fidelity: 0.960852

Thus context SR reaches most of the fidelity available from the frozen decoder manifold: the OOD mean-fidelity gap to the held-out context oracle is only 0.006484.

By contrast, training the decoder only to reproduce the initial state at M0 gives:

- OOD mean fidelity: 0.821628
- OOD final-time fidelity: 0.555709
- mean optimal tangent coverage: 0.582076

The fixed-initial-state baseline has OOD mean fidelity 0.752656 and final-time fidelity 0.438852.

This indicates that trajectory training is primarily learning a useful **tangent geometry / variational manifold**, not merely anchoring the initial state.

## Seed robustness

For three independent transformer initializations, one-trajectory pretraining followed by OOD context-SR rollout gives:

- OOD mean fidelity = 0.982829 +/- 0.002167
- OOD final fidelity = 0.959422 +/- 0.004998
- mean tangent coverage = 0.860823 +/- 0.017712

## Stress test

For Gaussian pulses with amplitudes A=[0.2, 0.5, 0.8, 1.1, 1.4, 1.8], final fidelity decreases monotonically from 0.960711 to 0.898941, while tangent coverage decreases from 0.859553 to 0.748047. The Pearson correlation between final fidelity and mean tangent coverage is r=0.991248.

This suggests tangent coverage can serve as a physics-based reliability diagnostic for when the learned context manifold is leaving its domain of validity.

## Context capacity

At fixed embedding d=12, increasing the number of context tokens gives P=12,24,48 real context coordinates. Mean OOD tangent coverage is respectively 0.8512, 0.8558, 0.9168. The rollout fidelity does not improve monotonically with P, indicating a second issue—conditioning/integration stiffness—once the tangent space becomes large. This is a useful distinction between **expressivity** and **stable dynamics**.

## Analytic implication: control-affine latent dynamics

For H(t)=H0+sum_mu h_mu(t) H_mu, the context-space SR force is linear in H while S(M) is independent of H. Therefore, for fixed regularization depending only on M,

    Mdot = f0(M) + sum_mu h_mu(t) f_mu(M).

The numerical implementation verifies this identity at a representative context to relative error ~3.7e-8. This gives a direct physics-derived alternative to a neural operator H(.) -> Mdot(.): the neural operator can be interpreted as amortizing a control-affine TDVP flow rather than being required to define it.

## Implementation checks

At a representative learned context:

- the SR direction improves the infinitesimal Schrödinger-step fidelity relative to the opposite direction;
- finite-difference projective tangent coverage converges to the analytic regularized SR coverage (~0.8980);
- control-affine decomposition error is 3.7e-8.

## Caveats before publication

These are preliminary N=6 exact-enumeration results. The strongest next checks are: N=8-12 with Monte Carlo/sample-space SR; more training/test seeds; longer-time rollouts; a matched standard full-parameter TDVP baseline; and comparison with the original FNO model at matched decoder capacity.
