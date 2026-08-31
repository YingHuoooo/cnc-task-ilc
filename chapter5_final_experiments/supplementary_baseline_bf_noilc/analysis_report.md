# Supplementary constrained BF-NOILC comparison

This is a prospectively specified supplementary comparison on eight new plants. It is separate from the original V11 formal protocol and Table 3.

## Primary demand-conflict result

| Condition | Task-AUC effect | Hierarchical 95% CI | Win rate | Interpretation |
|---|---:|---:|---:|---|
| Baseline | -23.269% | [-46.026%, -5.666%] | 10.0% | supports BF-NOILC superiority |
| Added delay +2 | -5.352% | [-16.770%, 1.429%] | 32.5% | direction favors BF-NOILC; interval crosses zero |
| Added delay +4 | -2.215% | [-4.363%, 2.497%] | 37.5% | direction favors BF-NOILC; interval crosses zero |
| Triple stress | -1.818% | [-4.290%, 1.659%] | 35.0% | direction favors BF-NOILC; interval crosses zero |

## Finite-budget AUC versus final-trial performance

| Condition | Task-AUC effect | Final-task-ratio effect | Final worst-zone effect |
|---|---:|---:|---:|
| Baseline | -23.269% [-46.026%, -5.666%] | -39.570% [-88.620%, 6.486%] | -33.530% [-83.525%, 8.980%] |
| Added delay +2 | -5.352% [-16.770%, 1.429%] | 4.664% [-26.148%, 17.987%] | 3.562% [-19.400%, 24.047%] |
| Added delay +4 | -2.215% [-4.363%, 2.497%] | 6.550% [0.270%, 16.695%] | 7.319% [0.919%, 21.824%] |
| Triple stress | -1.818% [-4.290%, 1.659%] | 5.374% [-1.105%, 14.254%] | 6.475% [0.334%, 15.134%] |

Positive task-AUC and global-RMSE effects favor Proposed. Positive effort differences mean Proposed used more update effort than BF-NOILC.

## Objective and effort trade-off

| Condition | Task-AUC median effect | Global-RMSE-AUC median effect | Learned command-update effort difference |
|---|---:|---:|---:|
| Baseline | -23.269% | -42.196% | -19.161% |
| Added delay +2 | -5.352% | -18.428% | -24.868% |
| Added delay +4 | -2.215% | -9.354% | -28.730% |
| Triple stress | -1.818% | -8.095% | -27.952% |

## Numerical validation

- Finite results: PASS
- Solver success for every update: PASS
- Implemented motion constraints: PASS

The comparison is configuration-level. It does not attribute any observed difference solely to selection, temporal alignment, the relaxation factor, or rollback.

## Evidence interpretation

- Proposed superiority on the primary finite-budget task AUC is not established in any of the four conditions.
- BF-NOILC superiority on task AUC is supported at baseline. Under added delay, the BF-NOILC direction remains favorable for AUC, but the demand-conflict hierarchical intervals cross zero.
- Proposed has a later-trial advantage under temporal mismatch: the final task ratio is significantly better at added delay +4, and the final worst-zone ratio is significantly better at added delay +4 and triple stress.
- BF-NOILC is significantly better on global RMSE AUC in all four demand-conflict conditions, consistent with its full-trajectory quadratic objective.
- Proposed uses significantly less coefficient and learned command-update effort in all four conditions. The result is therefore a speed/aggressiveness versus delayed-condition endpoint trade-off, not across-the-board superiority of either configuration.
