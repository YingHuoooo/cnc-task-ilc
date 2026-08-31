# Chapter 5 Validation Report

Status: **PASS**

## Deliverables

- Journal-style Chapter 5 manuscript in Markdown and LaTeX.
- A compiled nine-page standalone PDF for review.
- Six experiment figures, each exported as SVG, PDF, TIFF, and PNG (24 files in total).
- Figure-generation source code, figure contracts, terminology ledger, and source-data map.

## Manuscript checks

- The chapter separates reference-strategy comparisons from matched component ablations.
- Section 5.3 reports configuration-level evidence only; independent component attribution is restricted to Section 5.4.
- Section 5.3 now compares all three reference strategies on the same demand-conflict subset (`n=20` per comparison and condition); the all-15-task full-trajectory result is retained in a separate broader-scope panel and table block.
- The conflict-only full-trajectory values are derived from the existing frozen V11 raw runs with deterministic, comparison-specific paired-bootstrap seeds; no numerical experiment was rerun.
- Table 4 reports exact tie rates for every matched comparator. The Raw-top2 explanation was checked against the paired run histories: all exact AUC ties coincide with identical complete active-zone histories.
- The Triple-stress Raw-top2 hierarchical interval was synchronized with the frozen statistical source as `[0.000%, 0.045%]`.
- Held-out plant results state the comparator explicitly as **Proposed vs No residual alignment**.
- The representative replay states that temporal alignment is applied to the nominal sensitivity model and that neither the reference nor command signal receives an explicit temporal shift.
- The manuscript contains no development-version labels (V8--V13) and makes no synergistic-selector claim.
- The primary matched residual-alignment estimate (5.952%) and the broader formal benchmark estimate (4.616%) are assigned distinct evidential roles.

## Figure checks

- PNG previews are approximately 300 dpi.
- TIFF exports are 600 dpi.
- SVG exports retain editable text objects.
- Visual inspection passed for all six figures: no clipped labels, overlapping legends, broken panels, or misleading connections between unordered plant identifiers were observed.
- Figure 2 was visually rechecked after its two-panel scope separation; both the common-subset forest plot and broader all-task panel remain legible at manuscript width.
- Held-out Latin-hypercube plants and challenge cases are displayed as separate evidence sets.

## Typesetting checks

- The standalone manuscript compiles successfully with Tectonic.
- The compilation log contains no error, overfull-box, underfull-box, or warning entries.
- The final PDF contains nine A4 pages and was visually inspected page by page.
- Tables 3 and 4, equations, captions, and figure panels are readable without clipping or overflow after the added scope and tie-rate columns.

## Reproducibility and traceability

- `scripts/generate_figures.py` regenerates the complete figure set from the frozen experiment outputs.
- `source_map.md` records the source file used for every reported result and figure.
- `figure_contracts.md` records the intended claim, comparator, evidence unit, and interpretation boundary for each figure.
- `terminology_ledger.md` fixes method and comparator names used throughout the chapter.

## Scope

All work was written to the new `chapter5_final_experiments` directory. Existing experiment outputs and source manuscripts outside this directory were not modified.
