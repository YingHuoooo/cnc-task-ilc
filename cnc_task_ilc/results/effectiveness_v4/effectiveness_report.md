# Small Effectiveness Gate

## Scope

- Trajectory families: 5
- Virtual machine domains: 6
- Paired trajectory–machine cases: 30
- ILC updates per run: 4
- Methods: full trajectory, curvature, jerk, error peak, random, fixed-score automatic windows, a calibrated point classifier, and a risk-controlled three-window combination selector
- Evaluation windows: fixed per trajectory from separate calibration domains
- Learned selector protocol: leave-one-trajectory-family-out calibration; test-domain labels are never used for fitting
- Safe-combination protocol: two error-peak anchors plus one risk-controlled exploration window; labels are three-window, full-horizon ILC AUC rewards; ensemble risk and conservative fallback are calibrated; confirmation uses previously unseen machine seeds
- Primary method: safe_combo_eta_0.30

## Predefined gate

- median critical-error AUC improvement over the strongest aggregate baseline ≥ 10%
- paired win rate ≥ 60%
- successful constrained runs ≥ 95%
- median global-error ratio multiplier versus the strongest baseline ≤ 1.5

## Result

- Decision: REVISE
- Strongest baseline: error_peak_window
- Median critical AUC improvement: 0.00%
- Paired win rate: 10.00%
- Primary-method success rate: 100.00%
- Median global trade-off multiplier: 1.000×
- Primary median normalized critical AUC: 0.3264
- Primary median final critical-error ratio: 0.0840
- Primary median final global-error ratio: 0.2638
- Versus full-trajectory ILC: 5.89% median AUC improvement; 70.00% paired win rate
- Versus error-peak windows: 0.00% median AUC improvement; 10.00% paired win rate
- Median safe-combination-window/evaluation-window IoU: 0.536

## Interpretation

The gate evaluates whether the revised automatic critical-window contribution is
already strong enough to justify adding LinuxCNC and more advanced learning
modules. A REVISE decision does not invalidate ILC feasibility; it means the
automatic-window method has not yet passed the predefined comparative standard.
