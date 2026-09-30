# Preliminary Results Folder

These artifacts summarize exploratory `N=6` exact-enumeration experiments. They are included to give the student a concrete starting point and reproduction target, not as final publication claims.

Files:

- `PRELIMINARY_RESULTS.md`: narrative summary and caveats.
- `summary_table.csv`: main anchor-only versus one-trajectory comparison.
- `stress_test_summary.csv`: stress-drive summary transcribed from the preliminary exploration.
- `context_capacity_summary.csv`: context-token sweep summary transcribed from the preliminary exploration.
- `seed_robustness_summary.csv`: three-seed aggregate statistics.
- `fig1_pretraining_vs_anchor.png`: main training comparison.
- `fig2_stress_coverage.png`: stress/reliability figure.
- `fig3_context_capacity.png`: context-capacity figure.
- `test_math_output.txt`: independent SR equation/backend test output from the packaging session.

The raw run directories that originally generated every plotted point were not all retained in the conversation workspace. Therefore the student should treat the supplied reproduction script as the authoritative route to regenerate new raw results and should preserve all future raw run outputs.
