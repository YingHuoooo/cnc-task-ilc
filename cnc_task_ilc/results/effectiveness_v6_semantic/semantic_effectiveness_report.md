# V6 Semantic Task-Level ILC Confirmation

## Scope

- Trajectories: 5
- Virtual-machine domains: 6
- Methods: 8
- Total method runs: 240
- ILC updates per case: 4
- Active-zone budget: 2 of 6

## Frozen decision

- Primary method: `violation_safe`
- Decision: **REVISE**
- Median AUC improvement vs full trajectory: 6.72%
- Win rate vs dynamic error peak: 46.67%
- Median AUC improvement vs plain violation scheduling: 0.00%
- Primary success rate: 100.00%
- Median rejected trials: 0.00

Task zones are generated from programmed-path geometry before any virtual-machine
feedback is observed.  A REVISE decision means either the semantic scheduler or
the safety layer did not meet every frozen criterion.
