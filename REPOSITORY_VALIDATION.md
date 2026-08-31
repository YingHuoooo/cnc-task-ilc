# Repository validation

Validation date: 2026-08-31

Status: **PASS**

## Package integrity

- Experiment-related files: 390 before this report was added.
- Package size: approximately 734.4 MiB.
- Files larger than 100 MiB: 0.
- Largest file: `chapter5_final_experiments/figures/fig7_representative_trial_replay.tiff`, approximately 63.7 MiB.
- Python bytecode and cache files remaining in the package: 0.
- Broken relative links in the root `README.md`: 0.

## Executed checks

- Core numerical unit tests: **40 passed**.
- Additional-experiment validator: **35 passed, 0 failed**.
- Additional numerical method-run counts: 1,800 matched ablation, 780 parameter sensitivity, and 1,800 virtual-plant family runs.
- Supplementary constrained BF-NOILC validator: **PASS**.
- Supplementary output counts: 960 runs, 4,800 trial rows, 3,840 update rows, and 480 paired cases.
- Main experiment and figure entry-point scripts: Python syntax check passed.
- Additional-experiment manifest: 50/50 file hashes matched.
- Supplementary BF-NOILC manifest: 33/33 file hashes matched.

## Packaging checks

- Existing source results and protocols were copied without rewriting numerical values.
- Original workspace directories were not modified by the packaging operation.
- The new repository preserves the sibling-directory relationships required by experiment and figure scripts.
- Manuscript-only sources, literature-reader assets, LaTeX build files, and local caches were excluded as documented in `COPY_MANIFEST.md`.
- Git LFS is not installed on the packaging host. The default Git attributes therefore mark large raster files as binary without requiring an LFS filter. All files remain below GitHub's hard per-file limit; LFS setup is recommended before the first public commit.
