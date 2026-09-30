# Student Guide: One-Semester Research Workflow

## Week 1-2: Understand and reproduce

1. Read `README.md`, `RESEARCH_GOAL.md`, and `notes/PEDAGOGICAL_NOTES.md`.
2. Read the SR section of the Rende paper and the architecture sections of NOQS/UNP.
3. Create a clean Python environment.
4. Run `bash code/run_smoke_tests.sh`.
5. Run the preliminary reproduction script on a small CPU/GPU job.
6. Write a one-page summary in your own words explaining `S`, `F`, tangent coverage, and the difference between the neural operator and context TDVP.

## Week 3-5: Reproduce robustly

Repeat the N=6 experiment over several seeds. Do not change the architecture until the baseline is stable. Build scripts that aggregate results automatically into CSV/JSON files.

Required plots:

- global fidelity vs time;
- local projection fidelity vs time;
- tangent coverage vs time;
- spectrum/effective rank of `S` vs time;
- anchor-only vs trajectory-trained comparison;
- OOD stress parameter vs final fidelity and coverage.

## Week 6-9: Choose one main extension

Good one-semester choices:

A. **Reliability:** establish tangent coverage as an error indicator.

B. **Capacity:** determine minimal/effective context dimension and how it scales.

C. **Scaling:** implement stochastic/sample-space SR and push to larger systems.

D. **Dynamics structure:** study the control-affine vector fields `f_mu(M)` and whether a simpler causal local model can replace the FNO.

Pick one as the primary story and treat others as supporting tests.

## Week 10-12: Stress test the claim

Try to break the method. Increase time, drive strength, frequency, disorder, or entanglement. A paper is stronger when it identifies a clear domain of validity and a diagnostic of failure.

## Week 13-15: Consolidate

- Freeze the code version.
- Re-run final experiments from scripts.
- Save all random seeds and configs.
- Make publication-quality plots.
- Write a concise methods note and results summary.
- Discuss whether the evidence supports a paper, a workshop/poster result, or a follow-up project.

## Reproducibility habits

Every run should have an output directory containing:

- `summary.json`;
- raw arrays (`npz` or equivalent);
- the exact command line;
- git commit hash;
- random seeds;
- environment/package versions.

Never overwrite an old run directory without deliberate reason.

## How to use ChatGPT effectively

Use ChatGPT as a coding/reasoning collaborator, not as the source of truth. Ask it to:

- derive equations independently;
- inspect code for sign/conjugation/normalization errors;
- design controls and ablations;
- write analysis scripts;
- interpret unexpected results;
- search recent literature;
- help draft notes after results are verified.

Always verify numerical claims with independent checks and retain the exact scripts used to produce them.
