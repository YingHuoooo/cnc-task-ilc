PYTHON ?= python3
CORE := cnc_task_ilc
ADDITIONAL := v11_additional_experiments
CHAPTER := chapter5_final_experiments
BF_NOILC := $(CHAPTER)/supplementary_baseline_bf_noilc

.PHONY: install test demo main-experiment additional-experiments additional-analysis additional-validate bf-noilc-smoke bf-noilc bf-noilc-analysis bf-noilc-validate figures

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e ./$(CORE)

test:
	$(MAKE) -C $(CORE) test

demo:
	$(MAKE) -C $(CORE) demo

main-experiment:
	$(MAKE) -C $(CORE) gate-v11

additional-experiments:
	PYTHONPATH=$(CORE)/src:$(ADDITIONAL)/scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHON) $(ADDITIONAL)/scripts/run_experiments.py

additional-analysis:
	PYTHONPATH=$(CORE)/src:$(ADDITIONAL)/scripts $(PYTHON) $(ADDITIONAL)/scripts/analyze.py

additional-validate:
	$(PYTHON) $(ADDITIONAL)/scripts/validate.py

bf-noilc-smoke:
	$(PYTHON) $(BF_NOILC)/scripts/run_experiment.py --smoke --workers 1

bf-noilc:
	$(PYTHON) $(BF_NOILC)/scripts/run_experiment.py --workers $${V11_WORKERS:-4}

bf-noilc-analysis:
	$(PYTHON) $(BF_NOILC)/scripts/analyze_results.py

bf-noilc-validate:
	$(PYTHON) $(BF_NOILC)/scripts/validate_results.py

figures:
	MPLCONFIGDIR=$${TMPDIR:-/tmp}/chapter5-final-mpl $(PYTHON) $(CHAPTER)/scripts/generate_figures.py

