# Finite-Trial Task-Aware Contour Learning under Hidden Plant Mismatch

This repository contains the complete numerical experiment package for a
finite-trial, task-aware contour-learning method. The learner improves repeated
two-axis contour execution over **five trials (four command updates)** using a
low-dimensional B-spline correction, semantic task zones, tolerance-normalized
quality, a nominal contour-sensitivity model, residual effective-lag alignment,
constrained updates, and rollback.

The objective is task-level contour improvement under a deliberately mismatched
nominal model. The method does **not** attempt to identify the hidden plant or
recover a physical transport delay. All experiments in this repository use
numerical virtual plants; no LinuxCNC controller or physical machine was used.

## What is included

- Installable Python source for the virtual plant, trajectory generation,
  contour metrics, semantic task construction, B-spline correction, constrained
  ILC update, residual-lag estimator, and benchmark runners.
- Frozen task manifests and machine-readable experiment protocols.
- Main benchmark, robustness, factorial-stress, and residual-alignment results.
- Strictly matched component ablations and dependence-aware bootstrap analyses.
- Parameter sensitivity, representative replay, and held-out virtual-plant
  generalization experiments.
- A parameter-matched constrained basis-function NOILC comparison.
- Raw run-level outputs, summary tables, publication figures, and validation
  reports required to audit the reported results.

Development labels retained in filenames are archival identifiers only. They
record the order in which experiment protocols were frozen and are not part of
the scientific method definition.

## Repository layout

```text
.
├── cnc_task_ilc/                       # Core package, tests, protocols, runners, results
│   ├── src/cnc_task_ilc/               # Numerical model and learning algorithms
│   ├── scripts/                        # Experiment and comparison entry points
│   ├── data/tolerance_conflict_v1/     # Frozen task manifests and protocols
│   ├── tests/                          # Core unit tests
│   └── results/                        # Executed main and development benchmarks
├── v11_additional_experiments/         # Matched ablation and post-freeze analyses
│   ├── scripts/                        # Resumable runner, analysis, validation
│   ├── results/                        # Raw and summarized experiment outputs
│   ├── figures/                        # Publication exports
│   └── qa/                             # Integrity and figure checks
├── chapter5_final_experiments/         # Final experiment chapter assets
│   ├── scripts/generate_figures.py     # Chapter-level figure regeneration
│   ├── figures/                        # Final SVG/PDF/TIFF/PNG figures
│   └── supplementary_baseline_bf_noilc/
│       ├── scripts/                    # Constrained BF-NOILC comparison
│       ├── results/                    # Paired formal outputs
│       └── figures/                    # Supplementary comparison figures
├── cnc_v11_paper_package/              # Frozen evidence subset used by the figures
│   ├── source_data/                    # Figure-ready numerical tables
│   └── source_snapshot/                # Code, protocols, and raw-result snapshots
├── docs/                               # Experiment protocol and reproducibility notes
├── requirements.txt
└── Makefile                            # Repository-level convenience commands
```

## Experimental design at a glance

| Suite | Scientific question | Experimental unit | Main entry point |
|---|---|---|---|
| Core tests and demo | Is the numerical learning chain executable and constraint compliant? | Trajectory/trial | `make test`, `make demo` |
| Frozen main benchmark | Does the complete configuration improve finite-trial task quality under nominal and stressed dynamics? | Paired plant-task-scenario case | `make main-experiment` |
| Matched ablation | Which matched component changes produce independent performance differences? | Paired plant-task-scenario case | `make additional-experiments` |
| Dependence-aware inference | Are paired effects stable after plant/domain clustering is respected? | Paired case and plant/domain | `make additional-analysis` |
| Parameter sensitivity | How does performance vary with alignment gain and secondary settings? | Configuration-task-plant case | Included in the additional experiment runner |
| Held-out plant family | Does the residual-alignment benefit persist across unseen numerical plants? | Held-out plant | Included in the additional experiment runner |
| Constrained BF-NOILC comparison | How does the final configuration compare with a parameter-matched full-trajectory basis-function baseline? | Paired plant-task-scenario case | `make bf-noilc-smoke` / `make bf-noilc` |
| Figure regeneration | Can every experiment figure be rebuilt from frozen numerical files? | Figure/data contract | `make figures` |

The additional package contains 4,380 method runs: 1,800 matched-ablation
runs, 780 parameter-sensitivity runs, and 1,800 held-out/challenge-plant runs.
The supplementary constrained BF-NOILC experiment contains 960 formal method
runs. Runners use append-only JSONL checkpoints so interrupted grids can resume
without repeating completed jobs.

## Installation

Python 3.8 or newer is required. The recorded environment used Python 3.8.8,
NumPy 1.22.3, SciPy 1.10.1, Matplotlib 3.3.4, and Pillow 10.4.0.

```bash
git clone <repository-url>
cd cnc_contour_learning_experiments_github

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ./cnc_task_ilc
```

For Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`. The multiprocessing experiment runners were
validated on a Unix-like environment.

## Quick verification

Run the unit tests and the small numerical demo before starting any large grid:

```bash
make test
make demo
```

The unit tests exercise the numerical core, including trajectory generation,
plant simulation, contour-error computation, constrained updates, semantic
task metrics, protocol integrity, and delay-alignment utilities. The demo writes
its outputs to `cnc_task_ilc/results/`.

For a one-case verification of the supplementary baseline pipeline:

```bash
make bf-noilc-smoke
```

## Reproducing the experiments

### 1. Frozen main residual-alignment benchmark

```bash
make main-experiment
```

This command verifies the frozen protocol, executes the formal benchmark, and
rebuilds the comparison summary. The primary outputs are written to:

```text
cnc_task_ilc/results/effectiveness_v11_delay_compensation/
```

Earlier effectiveness, task-definition, robustness, and factorial benchmarks
remain available through the targets documented in
[`cnc_task_ilc/Makefile`](cnc_task_ilc/Makefile). They are included for complete
provenance, not as a version-evolution narrative.

### 2. Matched ablation, sensitivity, replay, and virtual-plant family

```bash
make additional-experiments
make additional-analysis
make additional-validate
```

The numerical runner executes priorities 1, 3, and 5. The analysis stage derives
the dependence-aware bootstrap results and selects the representative replay.
The experiment is resumable through the `raw_results.jsonl` checkpoints.

To control parallelism:

```bash
V11_WORKERS=4 make additional-experiments
```

The detailed design, seeds, pairing rules, statistics, and interpretation
boundaries are recorded in
[`v11_additional_experiments/experiment_design.md`](v11_additional_experiments/experiment_design.md)
and
[`v11_additional_experiments/protocol_pre_execution.json`](v11_additional_experiments/protocol_pre_execution.json).

### 3. Parameter-matched constrained BF-NOILC comparison

Run the smoke test first, then the formal paired grid:

```bash
make bf-noilc-smoke
make bf-noilc
make bf-noilc-analysis
make bf-noilc-validate
```

The formal comparison uses eight new virtual plants, 15 task manifests, four
operating conditions, and two methods. Its evidence is configuration-level:
the comparison does not attribute the total difference to any one component.

### 4. Publication figures

```bash
make figures
```

This rebuilds the final Chapter 5 figures from the frozen CSV, JSON, and NPZ
files. Each figure is exported as editable SVG, vector PDF, 600 dpi TIFF, and
300 dpi PNG. Figure-to-source traceability is documented in
[`chapter5_final_experiments/source_map.md`](chapter5_final_experiments/source_map.md).

## Metrics and pairing

The main task endpoint is finite-trial area under the tolerance-normalized task
quality curve. Each trajectory contains semantic zones defined before measured
feedback is observed. Zone error is evaluated in the contour-normal direction
and normalized by the zone-specific tolerance. Experiments use paired plant,
task, scenario, initial-command, and noise-seed designs whenever methods are
compared.

The reported analyses distinguish three types of evidence:

1. **Reference-strategy comparisons** evaluate complete configurations.
2. **Matched ablations** isolate a specified component while holding the rest of
   the numerical protocol fixed.
3. **Held-out plant experiments** evaluate one stated comparator across a
   predefined numerical uncertainty family; challenge plants are reported
   separately as boundary cases.

## Included representative results

- In the strictly matched added-delay evaluation, residual sensitivity
  alignment improved normalized task AUC by a median **5.952%** relative to the
  otherwise matched no-alignment configuration.
- Paired, plant-level, and hierarchical uncertainty intervals for this matched
  contrast remained positive in the recorded evaluation.
- Across the predefined held-out Latin-hypercube plant family, the minimum
  plant-level median effect for **Proposed vs No residual alignment** remained
  positive.

These are task-level results from numerical virtual plants. The raw records and
aggregation inputs are included so the summaries can be independently audited.

## Reproducibility and audit trail

- Experiment protocols are written before formal execution and include source
  hashes, seeds, settings, methods, sample counts, and claim boundaries.
- Formal task manifests are hash-checked before use.
- Paired noise seeds and plant-task keys are retained in raw output tables.
- Large grids write one JSON object per completed job before assembling CSVs.
- Figure scripts read frozen numerical files; no plotted value is entered by
  manual graphic editing.
- Validation reports are provided under each experiment package's `qa/`
  directory.

The original package-level manifest copied into
`cnc_v11_paper_package/ARCHIVED_PACKAGE_MANIFEST.json` describes the larger
paper archive from which the evidence subset was extracted. It is retained for
provenance and is not a manifest of this GitHub directory.

## Large figure files

High-resolution TIFF exports are included because they are manuscript assets.
The largest file is approximately 64 MiB: below GitHub's 100 MiB per-file
limit, but large enough that Git LFS is strongly recommended. Git LFS is not
assumed by the default `.gitattributes`, so the repository can still be added
with an ordinary Git installation.

```bash
# Recommended before the first git add, after installing Git LFS:
git lfs install
git lfs track "*.tif" "*.tiff"

git add .
git commit -m "Add reproducible contour-learning experiments"
```

If high-resolution TIFFs are not required in the public repository, they can be
distributed as a release archive while retaining the SVG/PDF/PNG versions in
Git. Do not remove raw CSV, JSON, JSONL, or NPZ files if full numerical
traceability is required.

## Scope

This repository evaluates a repeated-contour learning method in a controlled
numerical environment. It does not contain a real-time controller interface,
G-code execution path, cutting experiment, physical metrology data, or an
empirically identified population model of machine tools. The Latin-hypercube
family is a predefined numerical uncertainty set.

## Citation, authorship, and license

Author names, the final paper title, DOI, and software license were not inferred
during repository packaging. Before making the repository public, add:

- a `CITATION.cff` file containing the final bibliographic metadata;
- an explicit open-source license approved by all rights holders; and
- the final paper or preprint link.

Without an explicit license, GitHub publication does not grant reuse rights by
default.
