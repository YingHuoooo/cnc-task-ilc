# Supplementary constrained basis-function NOILC baseline

This directory contains a standalone, prospectively specified supplementary
comparison between the frozen Proposed configuration and a
**parameter-matched constrained basis-function NOILC** baseline.

The baseline is a literature-grounded norm-optimal ILC method adapted to the
same B-spline command basis and hard motion constraints as the Proposed method.
It uses full-trajectory identity weighting, the unaligned nominal sensitivity,
the same regularization and smoothness weights, and a unit relaxation factor
(`alpha = 1`). It does not use task-zone selection, residual-delay alignment,
the adaptive trust radius, or score-based rollback.

The experiment uses eight new seeded plants (26001--26008), all 15 tasks, and
four operating conditions. Proposed and baseline are strictly paired by plant,
task, scenario, initial command, and noise seed. The demand-conflict subset is
the primary scope; all 15 tasks form a broader scope result. These results are
kept separate from the original four-plant Table 3 benchmark.

Files:

- `protocol_pre_execution.json`: frozen design and interpretation rules.
- `scripts/`: runner, analysis, and validation code.
- `results/`: raw run, trial, update, and statistical outputs.
- `figures/`: supplementary diagnostic figures.
- `analysis_report.md`: concise evidence interpretation.
- `MANIFEST.json`: final artifact hashes.

Run sequence:

```bash
python scripts/run_experiment.py --smoke --workers 1
python scripts/run_experiment.py --workers 4
python scripts/analyze_results.py
python scripts/validate_results.py
```

The comparison is configuration-level and must not be used to attribute an
observed effect solely to selection, alignment, the learning-rate difference,
or rollback.
