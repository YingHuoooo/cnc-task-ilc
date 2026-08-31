# Externally Specified Machining-Task Gate

## Protocol

- Critical zones are fixed from normalized programmed-path progress and process labels.
- Zone tolerances are fixed before any virtual-machine feedback is observed.
- Six task zones compete for a three-zone learning budget.
- The primary method enumerates zone combinations with a mismatched nominal model and reselects after every ILC trial.
- Confirmation machine seeds: 431, 449, 461, 479, 491, 509
- Paired cases: 30
- ILC updates per run: 4

## Result

- Decision: REVISE
- Strongest baseline: violation_dynamic
- Median task-AUC improvement: -0.13%
- Paired win rate: 46.67%
- Constrained-run success: 100.00%
- Median global trade-off multiplier: 1.150x
- Median final violation-rate reduction: 0.00%
- Primary median task AUC: 0.4089
- Primary median final task-error ratio: 0.1764
- Primary median final tolerance-violation rate: 21.21%

## Interpretation

This gate tests task-aware allocation rather than rediscovering measured error
peaks. Evaluation labels and tolerances are external to the virtual-machine
tracking error. A GO decision therefore supports the revised research question;
a REVISE decision means the allocation method still needs improvement.
