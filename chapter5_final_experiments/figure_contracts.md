# Chapter 5 figure contracts

Backend: Python/Matplotlib only. All figures use frozen numerical results already present in the workspace. Final width is 183 mm, text remains editable in SVG/PDF, and each figure is exported as SVG, PDF, 600 dpi TIFF, and 300 dpi PNG.

## Figure 2 — Configuration-level comparison

- Core conclusion: the proposed configuration outperformed uniform full-trajectory and raw-error-peak learning across the evaluated conditions, while a well-chosen fixed shift remained a strong reference strategy under added delay.
- Evidence: four operating conditions and three reference strategies on a common demand-conflict subset, plus a separately displayed all-15-task full-trajectory scope result from the formal benchmark.
- Archetype: two-level quantitative forest plot separating common-subset and broader-scope evidence.
- Attribution boundary: the figure compares complete configurations and does not isolate task weighting, selector, residual alignment, or rollback.

## Figure 3 — Matched ablation and dependence-aware inference

- Core conclusion: residual effective-lag alignment is the component that provides the stable independent improvement in the matched evaluation.
- Evidence: strictly matched ablations under three scenarios, followed by paired, plant-level, and hierarchical bootstrap intervals.
- Archetype: quantitative grid with a full-width inference panel.
- Attribution boundary: the task/raw complementary selector is retained for interpretability; its independent synergistic advantage was not established.

## Figure 4 — Temporal-mismatch diagnosis

- Core conclusion: additional delay was the dominant tested source of finite-trial degradation.
- Evidence: factorial main effects and absolute AUC degradation relative to the corresponding baseline.
- Archetype: two-panel quantitative diagnosis.
- Review risk: artificial mismatch scaling is not presented as a monotone severity axis.

## Figure 5 — Held-out virtual-plant generalization

- Core conclusion: the benefit of residual effective-lag alignment generalized across 24 previously unseen numerical plants in the predefined LHS family.
- Evidence: plant-level effects for Proposed versus No residual alignment, plant-level confidence intervals, and a separate challenge-plant boundary panel.
- Archetype: asymmetric validation figure.
- Comparator boundary: every effect in this figure is Proposed versus No residual alignment.

## Figure 6 — Residual-alignment gain sensitivity

- Core conclusion: conservative fractional alignment is stable over a moderate gain range, whereas full compensation can over-correct.
- Evidence: one-factor-at-a-time development scans of gamma under added-delay and triple-stress conditions.
- Archetype: two-panel sensitivity plot.
- Decision boundary: the scan does not replace the prespecified gamma=0.25 used in confirmatory evaluations.

## Figure 7 — Representative trial-wise replay

- Core conclusion: in a prespecified median-effect replay, nominal-sensitivity alignment improved the finite-trial trajectory without explicitly translating the reference or command signal in time.
- Evidence: final contour, trial-wise task score, pointwise contour error, applied fractional lag, and active-zone history.
- Archetype: asymmetric mixed-modality figure.
- Review risk: the case is selected automatically by closeness to the median paired effect and is used for interpretation rather than statistical proof.
