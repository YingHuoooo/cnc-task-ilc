# Small Effectiveness Gate V2

## Scope

- Trajectory families: 5
- Virtual machine domains: 6
- Paired trajectory–machine cases: 30
- ILC updates per run: 4
- Methods: full trajectory, curvature, jerk, error peak, random, fixed-score automatic windows, and a calibrated automatic selector
- Evaluation windows: fixed per trajectory from separate calibration domains
- Learned selector protocol: leave-one-trajectory-family-out calibration; test-domain labels are never used for fitting
- Primary method: learned automatic windows with global-protection weight eta=1.0

## Predefined gate

- median critical-error AUC improvement over the strongest aggregate baseline ≥ 10%
- paired win rate ≥ 60%
- successful constrained runs ≥ 95%
- median global-error ratio multiplier versus the strongest baseline ≤ 1.5

## Result

- Decision: REVISE
- Strongest baseline: error_peak_window
- Median critical AUC improvement: -1.57%
- Paired win rate: 30.00%
- Primary-method success rate: 100.00%
- Median global trade-off multiplier: 0.871×
- Primary median normalized critical AUC: 0.3718
- Primary median final critical-error ratio: 0.1400
- Primary median final global-error ratio: 0.3070
- Versus full-trajectory ILC: 3.92% median AUC improvement; 73.33% paired win rate
- Versus error-peak windows: -1.57% median AUC improvement; 30.00% paired win rate
- Median learned-window/evaluation-window IoU: 0.784

## Interpretation

The gate evaluates whether the revised automatic critical-window contribution is
already strong enough to justify adding LinuxCNC and more advanced learning
modules. A REVISE decision does not invalidate ILC feasibility; it means the
automatic-window method has not yet passed the predefined comparative standard.
