"""Validate completeness, pairing, configuration, and artifact integrity."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
EXPECTED_RUNS = 960
EXPECTED_TRIALS = 4800
EXPECTED_UPDATES = 3840


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    runs = read_csv(RESULTS / "raw_runs.csv")
    trials = read_csv(RESULTS / "trial_history.csv")
    updates = read_csv(RESULTS / "update_history.csv")
    summary = read_csv(RESULTS / "comparison_summary.csv")
    indexed = {
        (row["plant_id"], row["manifest_id"], row["scenario_id"], row["method"]): row
        for row in runs
    }
    base_keys = {
        (row["plant_id"], row["manifest_id"], row["scenario_id"])
        for row in runs
    }
    paired = all(
        key + ("proposed",) in indexed
        and key + ("parameter_matched_constrained_bf_noilc",) in indexed
        for key in base_keys
    )
    paired_noise = all(
        indexed[key + ("proposed",)]["noise_seed"]
        == indexed[key + ("parameter_matched_constrained_bf_noilc",)]["noise_seed"]
        for key in base_keys
    )
    checks = {
        "run_row_count": len(runs) == EXPECTED_RUNS,
        "trial_row_count": len(trials) == EXPECTED_TRIALS,
        "update_row_count": len(updates) == EXPECTED_UPDATES,
        "complete_method_pairing": paired,
        "paired_noise_seeds": paired_noise,
        "eight_new_plants": sorted({int(row["plant_seed"]) for row in runs})
        == list(range(26001, 26009)),
        "fifteen_tasks": len({row["manifest_id"] for row in runs}) == 15,
        "four_scenarios": len({row["scenario_id"] for row in runs}) == 4,
        "baseline_alpha_one": all(
            float(row["learning_rate"]) == 1.0
            for row in runs
            if row["method"] == "parameter_matched_constrained_bf_noilc"
        ),
        "proposed_alpha_frozen": all(
            float(row["learning_rate"]) == 0.65
            for row in runs
            if row["method"] == "proposed"
        ),
        "parameter_matched_regularization": all(
            float(row["regularization"]) == 3.0e-3
            and float(row["smoothness"]) == 2.0e-8
            for row in runs
        ),
        "finite_results": all(int(row["finite_result"]) == 1 for row in runs),
        "solver_success": all(int(row["all_updates_succeeded"]) == 1 for row in runs),
        "implemented_constraints": all(
            int(row["all_update_constraints_satisfied"]) == 1
            and int(row["final_constraint_violation"]) == 0
            for row in runs
        ),
        "analysis_summary_present": len(summary) == 4 * 2 * 6,
        "analysis_report_present": (ROOT / "analysis_report.md").is_file(),
        "figures_present": all(
            (ROOT / "figures" / f"{name}.{suffix}").is_file()
            for name in (
                "paired_task_auc_effects",
                "control_effort_comparison",
                "demand_conflict_learning_curves",
            )
            for suffix in ("pdf", "svg", "png")
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "runs": len(runs),
            "trials": len(trials),
            "updates": len(updates),
            "paired_cases": len(base_keys),
        },
    }
    (RESULTS / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json" or "__pycache__" in path.parts:
            continue
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (ROOT / "MANIFEST.json").write_text(
        json.dumps({"files": files}, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
