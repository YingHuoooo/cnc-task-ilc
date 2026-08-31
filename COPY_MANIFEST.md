# Packaging manifest

This directory was assembled from the experiment-related contents of the parent
workspace. The copy preserves the original sibling-directory relationships used
by the experiment scripts.

## Included components

| Repository path | Source workspace path | Inclusion rule |
|---|---|---|
| `cnc_task_ilc/` | `cnc_task_ilc/` | Complete core package, scripts, task data, tests, and results; Python and Matplotlib caches excluded |
| `v11_additional_experiments/` | `v11_additional_experiments/` | Complete additional experiment package; Python caches excluded |
| `chapter5_final_experiments/` | `chapter5_final_experiments/` | Experiment narrative, source maps, figures, figure generator, validation report, and supplementary BF-NOILC package |
| `cnc_v11_paper_package/source_data/` | Same path | Complete frozen figure-ready data |
| `cnc_v11_paper_package/source_snapshot/{code,protocols,results,development}/` | Same paths | Frozen experimental evidence and source snapshot |
| `cnc_v11_paper_package/{figures,tables,qa}/` | Same paths | Experiment figures, summary tables, and validation records |
| `docs/` | Selected files from `cnc_task_ilc_project/` and `V11增加实验.md` | Method I/O, experiment protocol, reproducibility checklist, and additional-experiment plan |

## Deliberately excluded

- Manuscript-only LaTeX sources outside the experiment chapter.
- Literature-reader PDFs, extracted paper images, translations, and reference
  management files.
- LaTeX auxiliary files (`.aux`, `.log`, `.out`, `.blg`, `.bbl`).
- Python bytecode, `__pycache__`, local Matplotlib caches, and operating-system
  metadata.
- The standalone Chapter 5 review PDF and its LaTeX compilation wrapper; the
  readable experiment chapter Markdown and all underlying figures/data remain.

The exclusions remove presentation and local build artifacts, not numerical
evidence required by the included experiments.
