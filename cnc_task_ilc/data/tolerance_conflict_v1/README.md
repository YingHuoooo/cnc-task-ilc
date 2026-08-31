# Tolerance-conflict taskset audit

## Construction

- Taskset: `tolerance-conflict-v1`
- Manifests: 15 (5 trajectories x 3 regimes)
- Zones per manifest: 6
- Machine feedback used to construct manifests: no
- Measured tracking error used to assign tolerances: no
- Audit-only machine seeds: 733, 751, 769

## Regimes

- `neutral`: all zones use 0.24 mm.
- `demand_aligned`: high programmed demand receives tighter tolerances.
- `demand_conflict`: high programmed demand receives looser tolerances and low-demand zones receive tighter tolerances.

## Audit result

| Regime | Selection disagreement | Mean top-2 Jaccard | Median rank correlation |
|---|---:|---:|---:|
| Neutral | 0.00% | 1.000 | 1.000 |
| Demand aligned | 26.67% | 0.822 | 0.943 |
| Demand conflict | 80.00% | 0.444 | 0.714 |

Ready for a new formal experiment: **True**

The audit measures whether raw absolute-error priority and tolerance-normalized
priority differ. Audit feedback never changes a manifest. These seeds must not
be reused for the formal algorithm comparison.
