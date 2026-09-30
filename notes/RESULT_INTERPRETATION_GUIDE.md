# How to Interpret Common Outcomes

## High tangent coverage + high global fidelity
The context manifold is locally expressive and the integrator is tracking it successfully.

## High tangent coverage + declining global fidelity
Likely causes include accumulated integration error, curvature, stiffness, regularization bias, or drift away from the region on which the decoder was trained. Reduce `dt`, compare integrators, and compare with finite-step projection.

## Low tangent coverage + low fidelity
The fixed decoder/context manifold is locally insufficient for that Hamiltonian/state region. Increasing context capacity or manifold training diversity may help.

## Free-context oracle high, SR rollout low
The manifold can represent the target states but the context dynamics/integration cannot find or follow the correct path.

## Free-context oracle low
The bottleneck is the decoder/manifold itself. Improving the SR solver alone cannot fix it.

## Larger context gives higher coverage but worse rollout
Investigate conditioning and stiffness. Plot the SR spectrum, active rank, and velocity components in weak metric directions. Use stronger pseudoinverse truncation and smaller/adaptive time steps.

## Anchor-only model works almost as well as trajectory-trained model
This would weaken the claim that trajectory training learns dynamical geometry. Repeat with harder protocols and controls; check whether the decoder architecture alone already spans the relevant tangent directions.

## Tangent coverage does not correlate with failure at larger N
This would weaken the reliability-diagnostic story. Examine whether Monte Carlo noise biases coverage, whether regularized versus unregularized coverage matters, and whether a time-integrated residual is more predictive.
