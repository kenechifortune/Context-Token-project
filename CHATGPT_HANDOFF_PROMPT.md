# Prompt to use when handing this project to another ChatGPT account

Upload this ZIP (or the extracted folder) and send a message similar to the following:

---

I am a student continuing a research project on context-space TDVP/SR for neural quantum states. Please first read `README.md`, `RESEARCH_GOAL.md`, `notes/PEDAGOGICAL_NOTES.md`, `EXPERIMENT_ROADMAP.md`, and `results/preliminary/PRELIMINARY_RESULTS.md`. Then inspect the relevant code before recommending changes.

The core question is whether a fixed transformer NQS conditioned by low-dimensional context tokens `M` forms a reusable dynamical variational manifold. During rollout the transformer weights are frozen and `Mdot` is computed from context-space TDVP / stochastic reconfiguration, rather than predicted by a Fourier neural operator.

Important rules for our collaboration:

1. Treat all current N=6 numerical findings as preliminary, not established results.
2. Separate decoder/manifold error, tangent-projection error, integration error, Monte Carlo error, and generalization error.
3. Before modifying the equations, independently verify signs, conjugations, centering, normalization, and dimensions.
4. Prefer controlled ablations and reproducible scripts over architecture changes.
5. When proposing a result as publishable, identify the strongest alternative explanation and a test that could falsify it.
6. Keep exact commands, seeds, and output files for every important run.
7. If using web literature, distinguish established facts from our new interpretation.

First, summarize the project in your own words and tell me which implementation checks should be run before starting new experiments. Do not assume the preliminary claims are correct until the tests are inspected.

---

After the first exchange, give ChatGPT the output of `code/run_smoke_tests.sh` and the first reproduction run. Ask it to diagnose discrepancies before proceeding.
