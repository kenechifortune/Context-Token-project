# Derivation of Context-Space Real-Time SR/TDVP

Let the normalized state be `|psi(m)>`, where `m` is the flattened real context vector. The large decoder parameters are fixed.

## 1. Projective tangent vectors

Define

`|D_a> = (1 - |psi><psi|) |partial_a psi>`.

The exact physical Schrodinger tangent, with the global-phase component removed, is

`|b> = -i (H - <H>) |psi>`.

We seek the real velocity `v_a = mdot_a` minimizing

`|| sum_a v_a |D_a> - |b> ||^2`.

Taking derivatives with respect to each real `v_a` gives

`sum_b S_ab v_b = F_a`,

with

`S_ab = Re <D_a|D_b>`

and

`F_a = Re <D_a|b> = Im <D_a|(H-<H>)|psi>`.

## 2. Log-derivative representation

For configuration `sigma`, let

`O_a(sigma) = partial_a log psi(sigma)`.

For normalized `psi`, the projective metric becomes

`S_ab = Re < Otilde_a^* Otilde_b >`,

where

`Otilde_a = O_a - <O_a>`.

Using the local energy

`E_loc(sigma) = <sigma|H|psi> / psi(sigma)`,

the force becomes

`F_a = Im < Otilde_a^* Etilde_loc >`,

where

`Etilde_loc = E_loc - <E_loc>`.

Thus the context-only TDVP equation is

`S mdot = F`.

## 3. Tangent residual and coverage

For any candidate velocity `v`, the projective residual norm is

`r(v) = Var(H) - 2 F^T v + v^T S v`.

The unregularized pseudoinverse solution minimizes this residual. Define

`coverage = 1 - r(v_opt)/Var(H)`.

Equivalently, with the Moore-Penrose pseudoinverse,

`coverage = F^T S^+ F / Var(H)`

when numerical null modes are handled consistently.

## 4. Regularization

In practice one solves either

`(S + lambda I) v = F`

or uses an eigen/SVD cutoff. For scientific interpretation, it is useful to distinguish:

- the **regularized velocity** actually used in integration;
- the **unregularized optimal coverage** used as a measure of manifold expressivity.

The code reports both where practical.

## 5. Sample-space / Rende formulation

For `Ns` Monte Carlo samples, define centered log derivatives and arrange them as a complex matrix `Y` with parameter index along rows. For equal-weight samples,

`Y_ai = Otilde_a(sigma_i) / sqrt(Ns)`.

Write

`Y = Y_R + i Y_I`

and construct the real matrix

`X = [Y_R, Y_I]`.

Then

`S = X X^T`.

Let

`e_i = Etilde_loc(sigma_i)/sqrt(Ns) = e_R + i e_I`.

The real-time force can be written

`F = X f`,

with

`f = [e_I, -e_R]^T`.

Hence

`(X X^T + lambda I) mdot = X f`.

Using

`(X X^T + lambda I)^(-1) X = X (X^T X + lambda I)^(-1)`,

we obtain

`mdot = X (X^T X + lambda I)^(-1) f`.

This is useful when the number of context parameters is much larger than twice the sample count.

## 6. Control-affine structure

If

`H = H0 + sum_mu h_mu H_mu`,

then `E_loc` and therefore `F` are linear in the coefficients `h_mu`, while `S` depends only on the current state/context. Thus, for a fixed regularization rule depending only on `M`,

`mdot = f0(M) + sum_mu h_mu f_mu(M)`.

This is an exact structural consequence of context-space TDVP, not an empirical neural-network approximation.
