# Small Effectiveness Gate

## Scope

- Trajectory families: 5
- Virtual machine domains: 6
- Paired trajectory–machine cases: 30
- ILC updates per run: 4
- Methods: full trajectory, curvature, jerk, error peak, random, fixed-score automatic windows, a calibrated point classifier, and a simulation-reward window ranker
- Evaluation windows: fixed per trajectory from separate calibration domains
- Learned selector protocol: leave-one-trajectory-family-out calibration; test-domain labels are never used for fitting
- Reward-ranker protocol: candidate-window one-step ILC rewards are generated only on calibration domains; nested trajectory-family validation selects ridge regularization and error-peak blending; the held-out trajectory family and test domains remain frozen
- Primary method: reward_rank_eta_1.00

## Predefined gate

- median critical-error AUC improvement over the strongest aggregate baseline ≥ 10%
- paired win rate ≥ 60%
- successful constrained runs ≥ 95%
- median global-error ratio multiplier versus the strongest baseline ≤ 1.5

## Result

- Decision: REVISE
- Strongest baseline: error_peak_window
- Median critical AUC improvement: -0.68%
- Paired win rate: 33.33%
- Primary-method success rate: 100.00%
- Median global trade-off multiplier: 0.871×
- Primary median normalized critical AUC: 0.4093
- Primary median final critical-error ratio: 0.2117
- Primary median final global-error ratio: 0.2724
- Versus full-trajectory ILC: 2.42% median AUC improvement; 86.67% paired win rate
- Versus error-peak windows: -0.68% median AUC improvement; 33.33% paired win rate
- Median reward-ranked-window/evaluation-window IoU: 0.500

## Interpretation

The gate evaluates whether the revised automatic critical-window contribution is
already strong enough to justify adding LinuxCNC and more advanced learning
modules. A REVISE decision does not invalidate ILC feasibility; it means the
automatic-window method has not yet passed the predefined comparative standard.
