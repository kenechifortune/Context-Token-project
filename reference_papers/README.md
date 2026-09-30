# Reference Papers and Why They Matter

## NOQS (`NOQS_2603.25066v1.pdf`)
Introduces the neural-operator quantum-state architecture in which time-dependent driving protocols are encoded into context tokens `M(t)` that condition a transformer NQS through cross-attention. This project removes the neural operator and studies the context-conditioned decoder/manifold directly.

## UNP (`UNP_2605.05299v1.pdf`)
Extends the idea to a learned propagator and uses an FNO to produce `Mdot(t)` before integration. This makes the present project especially natural: instead of predicting `Mdot` with an FNO, derive it directly from TDVP/SR while freezing the decoder.

## Rende et al. 2024 (`Rende_2024_scalable_SR.pdf`)
Shows how SR can be rewritten using a push-through linear-algebra identity so the solve can occur in sample space rather than the full parameter space. This is directly useful when the context dimension becomes larger than the number of stochastic samples.
