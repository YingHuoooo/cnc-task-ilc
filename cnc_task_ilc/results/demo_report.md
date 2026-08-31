# Feasibility Demo Report

## Scope

This is an offline two-axis simulation of the core Task-Level ILC route. The
learner uses a low-order, delay-free and decoupled model. The evaluator contains
axis mismatch, delay, friction, saturation, cross-axis coupling and a
repeatable disturbance.

## Result

- Initial critical-window maximum error: 2.276518 mm
- Final critical-window maximum error: 0.029345 mm
- Critical-window peak reduction: 98.71%
- Full-trajectory ILC peak reduction on the same windows: 98.67%
- Final critical RMSE advantage over full-trajectory ILC: 36.03%
- Final global RMSE for critical-window ILC: 0.054051 mm
- All constrained updates solved: True
- Configured constraints violated: 0
- Feasibility checks passed: True

## Interpretation

The result demonstrates that measured task error from a structurally mismatched
virtual machine can be mapped through an approximate nominal-model sensitivity
and a constrained low-dimensional update to improve the next repeated trial.
This is evidence for algorithmic feasibility, not evidence of physical machine
or cutting-process performance.
