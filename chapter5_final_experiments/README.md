# Final Chapter 5 package

This directory contains the rewritten `Experiments and Results` chapter and its publication figures.

## Main files

- `chapter5_experiments_and_results.md`: readable manuscript source.
- `chapter5_experiments_and_results.tex`: LaTeX fragment for integration after Chapter 4.
- `chapter5_standalone.tex`: compilation wrapper for chapter-level QA.
- `figures/`: SVG, PDF, 600 dpi TIFF, and 300 dpi PNG exports.
- `scripts/generate_figures.py`: Python/Matplotlib figure generator.
- `figure_contracts.md`: claim, evidence, comparator, and review-risk contract for each figure.
- `terminology_ledger.md`: canonical method and metric names.
- `source_map.md`: figure-to-source-data traceability.

## Regeneration

From the workspace root:

```bash
MPLCONFIGDIR=${TMPDIR:-/tmp}/chapter5-final-mpl \
python3 chapter5_final_experiments/scripts/generate_figures.py
```

The chapter deliberately separates configuration-level reference comparisons from strictly matched component attribution. Held-out plant effects in Section 5.6 are always stated as Proposed versus No residual alignment.
