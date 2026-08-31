# CNC Task-Level ILC Feasibility Demo

This project is the first executable stage of the research route documented in
../cnc_task_ilc_project/. It demonstrates the algorithmic core before
LinuxCNC integration:

1. generate a two-axis reference contour;
2. execute it on a nonlinear virtual physical machine;
3. measure task-space contour error;
4. use a deliberately mismatched low-order model to compute local sensitivity;
5. parameterize command corrections with cubic B-splines;
6. solve a constrained quadratic update;
7. repeat the trial and compare full-trajectory ILC with critical-window ILC.

The demo is simulation-only. It proves that the proposed trial-to-trial
optimization chain is executable; it does not prove performance on a physical
machine tool.

## Environment

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

No LinuxCNC installation is required for this first demo. LinuxCNC is the next
integration layer after the numerical core is verified.

## Run

~~~bash
cd cnc_task_ilc
python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/run_demo.py
~~~

Generated outputs:

- results/metrics.json
- results/learning_curves.csv
- results/demo_summary.png
- results/trial_data.npz
- results/demo_report.md

Run the revised small multi-trajectory effectiveness gate:

~~~bash
make gate
~~~

The V2 gate fits nonnegative window-ranking weights with a
leave-one-trajectory-family-out protocol, freezes them for each held-out family,
and writes raw paired results, aggregate statistics, a decision report and plots
under results/effectiveness_v2/. The original V1 results remain under
results/effectiveness/ and can be reproduced with `make gate-v1`.

The current executed V2 gate and Chinese diagnosis are available at:

- results/effectiveness_v2/effectiveness_report.md
- results/effectiveness_v2/effectiveness_diagnosis_zh.md

Run the V3 simulation-reward window ranker with:

~~~bash
make gate-v3
~~~

V3 learns candidate-window utility from real one-update calibration rollouts,
uses nested trajectory-family validation, and keeps the original 30 test cases
frozen. Its outputs and Chinese diagnosis are under results/effectiveness_v3/.

Run the V4 risk-controlled confirmation experiment with:

~~~bash
make gate-v4
~~~

V4 keeps two error-peak anchors, learns only one exploration window from
three-window four-update AUC rewards, calibrates conservative fallback, and
confirms performance on six previously unseen virtual-machine seeds. Outputs
are under results/effectiveness_v4/.

Run the V5 externally specified machining-task experiment with:

~~~bash
make gate-v5
~~~

V5 replaces error-defined evaluation windows with fixed programmed-path task
zones and tolerances, limits each ILC update to two of six zones, and compares
online task-budget schedulers on new virtual-machine domains. Outputs are under
results/effectiveness_v5_task/.

Run the V6 semantic task-level confirmation with:

~~~bash
make gate-v6
~~~

V6 replaces the shared fixed path fractions with trajectory-specific task zones
generated from programmed entry/exit, geometric extrema, curvature features,
blend transitions and motion transitions before any feedback is observed.  The
task state combines zone RMS, peak and local-ripple tolerance ratios, with an
equal emphasis on aggregate quality and the worst functional zone.  The frozen
primary method is tolerance-violation scheduling with output-space step limiting
and rollback after an observed degrading trial.  Confirmation outputs are under
results/effectiveness_v6_semantic/.

Build the feedback-independent tolerance-conflict taskset with:

~~~bash
make build-taskset
~~~

The generated `data/tolerance_conflict_v1/` dataset contains 15 fixed task
manifests: five trajectory families under neutral, program-demand-aligned and
program-demand-conflict tolerance assignments.  Program demand uses only
reference curvature, reference jerk and nominal-model sensitivity; neither
virtual-machine feedback nor measured tracking error is used to place zones or
assign tolerances.  Audit seeds are recorded separately and must not be reused
for the formal method comparison.

Run the frozen V7 formal comparison with:

~~~bash
make gate-v7
~~~

V7 verifies the SHA-256 digest of the 15 manifests before running.  It uses six
new virtual-machine domains and compares seven methods over all neutral,
program-demand-aligned and program-demand-conflict tasks.  The preregistered
primary endpoint is safe tolerance scheduling versus dynamic error-peak
selection on the conflict regime; the audit domains are explicitly excluded.
Outputs are written to `results/effectiveness_v7_conflict/`.

V8 addresses the V7 conflict-regime failure with a dual-anchor scheduler.  One
of the two active zones is reserved for the largest tolerance-normalized task
urgency; the other is reserved for the largest raw error peak outside that
zone.  Both receive the same optimization boost, followed by the existing
output-space trust limit and rollback.  This avoids counting a tight tolerance
twice in both selection and weight magnitude.

The V8 workflow deliberately separates development and confirmation:

~~~bash
make development-v8
make freeze-v8
make gate-v8
~~~

The first command screens the fixed method on three development-only machine
domains.  The second freezes the taskset, scheduler source and development
result hashes together with all settings, methods, thresholds and six unseen
formal domains.  The final command verifies that frozen protocol, executes 720
method runs, and applies the preregistered paired statistics.  Existing frozen
protocols are never overwritten when their content differs.  Outputs are under
`results/development_v8_dual_anchor/` and
`results/effectiveness_v8_dual_anchor/`.

The completed formal V8 run passed all five gates.  On 30 conflict-task pairs,
dual-anchor scheduling improved normalized task AUC by a median 4.27% over
dynamic error-peak scheduling, won all 30 pairs, and had a bootstrap 95%
interval of [2.44%, 6.10%].  Across all 90 task-domain pairs it improved the
median task AUC by 6.72% over full-trajectory learning while keeping the paired
global-error multiplier at 1.252x and the solver/constraint success rate at
100%.  These are virtual-machine results, not a substitute for physical-machine
validation.

V9 keeps the V8 scheduler frozen and maps its virtual-machine robustness
boundary with a one-factor-at-a-time design.  Learning and rollback use the
measured position, including deterministic paired measurement noise, while all
reported task metrics use the hidden noise-free plant position.  The seven
fixed scenarios are the shared baseline, 0.02/0.05 mm position noise, two/four
additional delay steps, and 1.35x/1.70x dynamic mismatch.  Run the isolated
maximum-level audit and frozen confirmation with:

~~~bash
make audit-v9
make freeze-v9
make gate-v9
~~~

The completed V9 experiment contains 1,680 method runs over four new formal
machine domains.  It was classified `BROADLY_ROBUST`: dual-anchor scheduling
retained a positive bootstrap lower bound and at least 60% strict paired wins
against error-peak scheduling in five of six nonbaseline stress scenarios.  It
remained confirmed through 0.02 mm measurement noise, four added delay steps
and 1.70x dynamic mismatch.  At 0.05 mm noise the median advantage was still
1.66% with 70% wins, but the 95% interval crossed zero, so that level is not
claimed as statistically confirmed.  All V9 dual-anchor runs were finite,
solver-successful and command-constraint compliant.  Outputs are under
`results/development_v9_robustness/` and
`results/effectiveness_v9_robustness/`.

V10 tests simultaneous stress with a frozen full 2x2x2 factorial design.  The
high levels are 0.05 mm measurement noise, four additional delay steps and
1.70x dynamic mismatch.  All eight low/high combinations are required so the
main effects, three pairwise interactions and three-factor interaction can be
estimated without changing the V8 scheduler or V9 stress runner.  Run with:

~~~bash
make audit-v10
make freeze-v10
make gate-v10
~~~

The formal experiment contains 1,920 method runs on four additional isolated
machine domains.  Its frozen classification is
`COMBINED_ADVANTAGE_UNRESOLVED`: under all three high stresses the dual-anchor
method retained a 0.27% median AUC advantage and 65% paired wins over dynamic
error-peak scheduling, but the 95% interval [-0.54%, 1.79%] crossed zero.  The
algorithm still learned effectively in absolute terms (median normalized AUC
0.618 and final task ratio 0.463), remained 2.02% better than full-trajectory
learning across all tasks, and had 100% solver/constraint success.  Factorial
analysis identified added delay as the dominant normalized-AUC effect (41.44%);
noise and all interaction terms were statistically unresolved.  Outputs are in
`results/development_v10_factorial/` and
`results/effectiveness_v10_factorial/`.

V11 follows the V10 diagnosis that added delay is the dominant failure mode.
It estimates the effective x/y lag after every completed trial by correlating
command and measured-feedback velocities, subtracts the lag already represented
by the nominal model, and fractionally time-aligns the two axis-sensitivity
blocks.  An isolated gain sweep showed that full delay insertion overcompensates
under structural mismatch; the preregistered robust rule selected a conservative
0.25 gain before any formal-domain run.  Reproduce the stages with:

~~~bash
make development-v11
make freeze-v11
make gate-v11
~~~

The frozen V11 confirmation contains 1,200 method runs over four new machine
domains, 15 task manifests, and baseline, +2-delay, +4-delay and simultaneous
noise/delay/mismatch scenarios.  It was classified
`DELAY_COMPENSATION_CONFIRMED`: relative to the original dual-anchor method,
the conflict-task normalized AUC improved by a median 4.62% under +4 delay
(95% bootstrap CI [1.84%, 8.19%], 90% wins) and 6.77% under simultaneous
stress ([4.71%, 9.56%], 90% wins).  The online estimator's median axis-delay
absolute error was 0.5 samples, and all solver/constraint success rates were
100%.  Fixed +4-sample compensation was not statistically equivalent under
the simultaneous stress, supporting online residual estimation plus uncertainty
shrinkage rather than blind lookahead.  Outputs are under
`results/development_v11_delay_compensation/` and
`results/effectiveness_v11_delay_compensation/`.

V12 tests the unresolved question of whether online estimation is necessary
when the physical x/y delays are unknown or drift slowly between trials.  Three
new development-only machine domains compare the original dual-anchor method,
fixed +2/+4/+6 sensitivity shifts, V11 online 0.25 compensation, and a
true-delay 0.25 oracle.  Run the stopped development experiment with:

~~~bash
make development-v12
~~~

The preregistered development gate did not pass, so no V12 formal protocol was
frozen and the four reserved formal seeds were not used.  Fixed +2 was selected
as the strongest fixed baseline.  Online compensation improved over the
original dual-anchor method by a median 5.39% for unknown static delays and
4.12% for slow drift, but improved over fixed +2 by only 0.37% and -0.26%,
respectively; both exploratory bootstrap intervals crossed zero.  The
true-delay oracle also failed to separate from fixed +2, while online lag MAE
was 1.0 sample and solver/constraint success was 100%.  The result indicates
that 0.25 shrinkage plus the broad 12-control-point spline collapses diverse
physical delays into an effective shift near two samples, so the current online
identifier adds no confirmed task benefit beyond a tuned fixed phase advance.
Outputs are under `results/development_v12_delay_generalization/`.

V13 is the single targeted follow-up allowed after V12.  It replaces the V11
all-history lag median with a two-trial rolling tracker and uses correlation
confidence to adapt a nonlinear compensation gain between 0.15 and 0.38.  The
development design balances low/high static delays, slow rising/falling drift,
and abrupt/asymmetric switches.  Reproduce the stopped screen with:

~~~bash
make development-v13
~~~

The method did not pass its declared development gate and therefore did not
use the four reserved formal domains.  Relative to fixed +2, its median AUC
improvements were 3.60%, 0.77%, and 0.73% for static, slow-drift, and switch
scenarios, but the slow-drift win rate was only 55%.  Relative to V11, the
slow-drift and switch medians were -0.32% and -0.19%, with only 45% and 30%
wins.  Lag MAE remained 0.5 samples and safety success was 100%, indicating a
control-law failure rather than an estimator or numerical failure.  The final
method therefore remains the formally confirmed V11 cumulative-median 0.25
compensation.  V12/V13 are retained as limitation and ablation studies under
`results/development_v12_delay_generalization/` and
`results/development_v13_adaptive_delay/`.

## What counts as feasibility

The demo is considered feasible when:

- the virtual plant response differs from the command and remains stable;
- the nominal learner model is structurally different from the plant;
- the constrained optimizer returns valid updates;
- the learned command reduces the selected task error over repeated trials;
- the final command remains within the configured correction, velocity and
  acceleration limits.

## Project structure

~~~text
src/cnc_task_ilc/
    trajectory.py          reference contour and geometry
    plant.py               nominal model and nonlinear virtual plant
    basis.py               cubic B-spline command correction
    metrics.py             task-space errors and constraints
    critical_windows.py    automatic window scoring and selection
    ilc.py                 sensitivity, constrained QP and learning loop
    delay_compensation_runner.py  online lag estimation and fractional alignment
    delay_compensation_benchmark.py  V11 development, protocol and confirmation
    delay_generalization_runner.py  V12 unknown/drifting delay schedules and oracle
    delay_generalization_benchmark.py  stopped V12 development and diagnosis
    adaptive_delay_runner.py  stopped V13 rolling confidence-adaptive design
    adaptive_delay_benchmark.py  V13 development gate and negative-result analysis
    demo.py                experiment, plots and reports
scripts/run_demo.py        command-line entry point
tests/test_core.py         numerical and structural tests
~~~

## Next step

Replace the initial command generator with LinuxCNC trajectory output and place
the same virtual plant between position command and feedback. The ILC optimizer
can remain an external trial-to-trial Python process.
