# Terminology ledger for Chapter 5

| Canonical term | Definition and use | Rejected variants |
|---|---|---|
| Proposed method | Complete frozen method evaluated in Chapter 5 | V11, V11 Full |
| No residual alignment | Matched ablation with residual-alignment gain set to zero | uncompensated version, task-aware selector without alignment |
| Uniform full trajectory | Configuration-level baseline with a full-contour objective | full trajectory, uniform baseline |
| Raw-error peak | Configuration-level sparse baseline driven by raw contour-error peaks | error peak, geometric peak |
| Fixed sensitivity shift | Configuration-level baseline using a predetermined four-sample sensitivity shift | fixed delay compensation |
| Task-top2 | Matched selector that chooses both zones by tolerance-normalized urgency | task-only selector |
| Raw-top2 | Matched selector that chooses both zones by raw contour-error magnitude | raw-only selector |
| Residual effective-lag alignment | Nominal-lag-subtracted, cumulative-median, fractional alignment of the nominal sensitivity | delay identification, physical-delay recovery |
| Nominal contour sensitivity | The model-based matrix shifted during alignment | shifted command, shifted reference |
| Tolerance-normalized task AUC | Primary finite-trial task-quality metric; lower is better | generic AUC, convergence score |
| Hidden numerical plant | Plant realization unavailable to the learner | real plant, physical machine population |
| Held-out LHS plants | The 24 previously unseen plants used for main plant-level evaluation | challenge plants |
| Challenge plants | Six separately designed numerical boundary cases | held-out population |

