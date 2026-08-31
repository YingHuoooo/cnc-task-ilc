# Small Effectiveness Gate

## Scope

- Trajectory families: 5
- Virtual machine domains: 6
- Paired trajectory–machine cases: 30
- ILC updates per run: 4
- Methods: full trajectory, curvature, jerk, error peak, random, and automatic windows with three global-protection weights
- Evaluation windows: fixed per trajectory from separate calibration domains

## Predefined gate

- median critical-error AUC improvement over the strongest aggregate baseline ≥ 10%
- paired win rate ≥ 60%
- successful constrained runs ≥ 95%
- median global-error ratio multiplier versus the strongest baseline ≤ 1.5

## Result

- Decision: REVISE
- Strongest baseline: error_peak_window
- Median critical AUC improvement: -4.02%
- Paired win rate: 26.67%
- Primary-method success rate: 100.00%
- Median global trade-off multiplier: 0.996×
- Primary median normalized critical AUC: 0.3736
- Primary median final critical-error ratio: 0.1730
- Primary median final global-error ratio: 0.3421
- Versus full-trajectory ILC: -2.78% median AUC improvement; 43.33% paired win rate
- Versus error-peak windows: -4.02% median AUC improvement; 26.67% paired win rate
- Median automatic-window/evaluation-window IoU: 0.500

## Interpretation

The gate evaluates whether the automatic critical-window contribution is
already strong enough to justify adding LinuxCNC and more advanced learning
modules. A REVISE decision does not invalidate ILC feasibility; it means the
automatic-window method has not yet passed the predefined comparative standard.
