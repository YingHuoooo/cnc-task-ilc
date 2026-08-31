.PHONY: all test demo build-taskset gate gate-v1 gate-v2 gate-v3 gate-v4 gate-v5 gate-v6 gate-v7 development-v8 freeze-v8 gate-v8 audit-v9 freeze-v9 gate-v9 audit-v10 freeze-v10 gate-v10 development-v11 freeze-v11 gate-v11 development-v12 compare-v12 development-v13 compare-v13 compare-v2 compare-v3 compare-v4 compare-v5 compare-v6 compare-v7 compare-v8 compare-v9 compare-v10 compare-v11

all: test demo

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python3 scripts/run_demo.py

build-taskset:
	PYTHONPATH=src python3 scripts/build_conflict_taskset.py

gate:
	PYTHONPATH=src python3 scripts/run_effectiveness_v2.py
	PYTHONPATH=src python3 scripts/compare_effectiveness_v1_v2.py

gate-v1:
	PYTHONPATH=src python3 scripts/run_effectiveness_gate.py

gate-v2:
	PYTHONPATH=src python3 scripts/run_effectiveness_v2.py
	PYTHONPATH=src python3 scripts/compare_effectiveness_v1_v2.py

gate-v3:
	PYTHONPATH=src python3 scripts/run_effectiveness_v3.py
	PYTHONPATH=src python3 scripts/compare_effectiveness_v2_v3.py

gate-v4:
	PYTHONPATH=src python3 scripts/run_effectiveness_v4.py
	PYTHONPATH=src python3 scripts/compare_effectiveness_v4.py

gate-v5:
	PYTHONPATH=src python3 scripts/run_task_definition_v5.py
	PYTHONPATH=src python3 scripts/compare_task_definition_v5.py

gate-v6:
	PYTHONPATH=src python3 scripts/run_semantic_v6.py
	PYTHONPATH=src python3 scripts/compare_semantic_v6.py

gate-v7:
	PYTHONPATH=src python3 scripts/run_conflict_v7.py
	PYTHONPATH=src python3 scripts/compare_conflict_v7.py

development-v8:
	PYTHONPATH=src python3 scripts/run_dual_anchor_development.py

freeze-v8:
	PYTHONPATH=src python3 scripts/freeze_dual_anchor_v8.py

gate-v8:
	PYTHONPATH=src python3 scripts/freeze_dual_anchor_v8.py
	PYTHONPATH=src python3 scripts/run_dual_anchor_v8.py
	PYTHONPATH=src python3 scripts/compare_dual_anchor_v8.py

audit-v9:
	PYTHONPATH=src python3 scripts/run_robustness_audit_v9.py

freeze-v9:
	PYTHONPATH=src python3 scripts/freeze_robustness_v9.py

gate-v9:
	PYTHONPATH=src python3 scripts/freeze_robustness_v9.py
	PYTHONPATH=src python3 scripts/run_robustness_v9.py
	PYTHONPATH=src python3 scripts/compare_robustness_v9.py

audit-v10:
	PYTHONPATH=src python3 scripts/run_factorial_audit_v10.py

freeze-v10:
	PYTHONPATH=src python3 scripts/freeze_factorial_v10.py

gate-v10:
	PYTHONPATH=src python3 scripts/freeze_factorial_v10.py
	PYTHONPATH=src python3 scripts/run_factorial_v10.py
	PYTHONPATH=src python3 scripts/compare_factorial_v10.py

development-v11:
	PYTHONPATH=src python3 scripts/run_delay_compensation_development_v11.py

freeze-v11:
	PYTHONPATH=src python3 scripts/freeze_delay_compensation_v11.py

gate-v11:
	PYTHONPATH=src python3 scripts/freeze_delay_compensation_v11.py
	PYTHONPATH=src python3 scripts/run_delay_compensation_v11.py
	PYTHONPATH=src python3 scripts/compare_delay_compensation_v11.py

development-v12:
	PYTHONPATH=src python3 scripts/run_delay_generalization_development_v12.py
	PYTHONPATH=src python3 scripts/compare_delay_generalization_development_v12.py

compare-v12:
	PYTHONPATH=src python3 scripts/compare_delay_generalization_development_v12.py

development-v13:
	PYTHONPATH=src python3 scripts/run_adaptive_delay_development_v13.py
	PYTHONPATH=src python3 scripts/compare_adaptive_delay_development_v13.py

compare-v13:
	PYTHONPATH=src python3 scripts/compare_adaptive_delay_development_v13.py

compare-v2:
	PYTHONPATH=src python3 scripts/compare_effectiveness_v1_v2.py

compare-v3:
	PYTHONPATH=src python3 scripts/compare_effectiveness_v2_v3.py

compare-v4:
	PYTHONPATH=src python3 scripts/compare_effectiveness_v4.py

compare-v5:
	PYTHONPATH=src python3 scripts/compare_task_definition_v5.py

compare-v6:
	PYTHONPATH=src python3 scripts/compare_semantic_v6.py

compare-v7:
	PYTHONPATH=src python3 scripts/compare_conflict_v7.py

compare-v8:
	PYTHONPATH=src python3 scripts/compare_dual_anchor_v8.py

compare-v9:
	PYTHONPATH=src python3 scripts/compare_robustness_v9.py

compare-v10:
	PYTHONPATH=src python3 scripts/compare_factorial_v10.py

compare-v11:
	PYTHONPATH=src python3 scripts/compare_delay_compensation_v11.py
