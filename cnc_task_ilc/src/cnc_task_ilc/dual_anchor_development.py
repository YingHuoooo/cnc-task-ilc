"""Independent development check for the V8 dual-anchor scheduler."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .basis import cubic_bspline_basis
from .benchmark import BenchmarkSettings
from .conflict_benchmark import validate_frozen_taskset
from .conflict_taskset import TASK_REGIMES, specification_from_manifest
from .semantic_task_benchmark import run_semantic_task_method
from .trajectory import make_trajectory_family


DEVELOPMENT_DOMAIN_SEEDS = (1031, 1049, 1061)
DEVELOPMENT_METHODS = (
    "full_trajectory",
    "error_peak_dynamic",
    "violation_dynamic",
    "violation_safe",
    "dual_anchor_dynamic",
)
PRIMARY_METHOD = "dual_anchor_dynamic"

# Written before the development run.  This is a screening rule, not the
# formal V8 confirmation gate.
DEVELOPMENT_CRITERIA = {
    "conflict_vs_error_peak_median_improvement_above_zero": 0.0,
    "conflict_vs_error_peak_win_rate_at_least": 0.60,
    "all_tasks_vs_full_median_improvement_above_zero": 0.0,
    "solver_and_constraint_success_rate_at_least": 0.95,
}


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _paired_improvement(
    rows: Sequence[Dict[str, object]],
    comparator: str,
    regime: Optional[str],
) -> Dict[str, object]:
    scoped = (
        list(rows)
        if regime is None
        else [row for row in rows if row["regime"] == regime]
    )
    indexed = {
        (str(row["manifest_id"]), int(row["domain_seed"]), str(row["method"])): row
        for row in scoped
    }
    keys = sorted({(key[0], key[1]) for key in indexed})
    proposed = np.asarray(
        [float(indexed[key + (PRIMARY_METHOD,)]["task_auc_normalized"]) for key in keys]
    )
    baseline = np.asarray(
        [float(indexed[key + (comparator,)]["task_auc_normalized"]) for key in keys]
    )
    improvement = 100.0 * (baseline - proposed) / baseline
    return {
        "scope": "all" if regime is None else regime,
        "comparator": comparator,
        "paired_cases": int(improvement.size),
        "median_task_auc_improvement_percent": float(np.median(improvement)),
        "mean_task_auc_improvement_percent": float(np.mean(improvement)),
        "strict_win_rate": float(np.mean(proposed < baseline)),
        "tie_rate": float(np.mean(np.isclose(proposed, baseline))),
    }


def run_dual_anchor_development(
    taskset_path: Path,
    output_directory: Path,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    """Run the pre-formal development screen on three isolated domains."""

    if tuple(settings.domain_seeds) != DEVELOPMENT_DOMAIN_SEEDS:
        raise RuntimeError("development seeds differ from the declared protocol")
    if settings.number_of_windows != 2:
        raise RuntimeError("dual-anchor development requires a two-zone budget")
    taskset = validate_frozen_taskset(taskset_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for manifest_index, manifest in enumerate(taskset["manifests"]):
        family = str(manifest["trajectory_family"])
        reference = make_trajectory_family(
            family,
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(
            samples=settings.samples,
            control_points=settings.control_points,
        )
        specification = specification_from_manifest(reference, manifest)
        for domain_seed in settings.domain_seeds:
            for method in DEVELOPMENT_METHODS:
                result = run_semantic_task_method(
                    method,
                    reference,
                    basis,
                    specification,
                    int(domain_seed),
                    settings,
                    random_seed=70000 + 100 * manifest_index + int(domain_seed),
                )
                result["manifest_id"] = str(manifest["manifest_id"])
                result["trajectory"] = family
                result["regime"] = str(manifest["regime"])
                rows.append(result)

    column_order = [
        "manifest_id",
        "trajectory",
        "regime",
        "domain_seed",
        "method",
        "initial_task_score",
        "final_task_score",
        "last_observed_task_score",
        "task_auc_normalized",
        "final_task_ratio",
        "initial_violation_rate",
        "final_violation_rate",
        "initial_task_max_ratio",
        "final_task_max_ratio",
        "initial_global_rmse",
        "final_global_rmse",
        "final_global_ratio",
        "selection_switches",
        "selection_history",
        "accepted_history",
        "rejected_trials",
        "final_trust_radius_mm",
        "constraint_violation",
        "all_updates_succeeded",
        "elapsed_s",
    ]
    ordered = [{column: row[column] for column in column_order} for row in rows]
    conflict = _paired_improvement(ordered, "error_peak_dynamic", "demand_conflict")
    full = _paired_improvement(ordered, "full_trajectory", None)
    plain = _paired_improvement(ordered, "violation_dynamic", None)
    safe = _paired_improvement(ordered, "violation_safe", None)
    primary_rows = [row for row in ordered if row["method"] == PRIMARY_METHOD]
    success_rate = float(
        np.mean(
            [
                row["all_updates_succeeded"] == 1
                and row["constraint_violation"] == 0
                for row in primary_rows
            ]
        )
    )
    criteria = {
        "conflict_vs_error_peak_median_improvement_above_zero": (
            conflict["median_task_auc_improvement_percent"]
            > DEVELOPMENT_CRITERIA[
                "conflict_vs_error_peak_median_improvement_above_zero"
            ]
        ),
        "conflict_vs_error_peak_win_rate_at_least_60_percent": (
            conflict["strict_win_rate"]
            >= DEVELOPMENT_CRITERIA[
                "conflict_vs_error_peak_win_rate_at_least"
            ]
        ),
        "all_tasks_vs_full_median_improvement_above_zero": (
            full["median_task_auc_improvement_percent"]
            > DEVELOPMENT_CRITERIA[
                "all_tasks_vs_full_median_improvement_above_zero"
            ]
        ),
        "success_rate_at_least_95_percent": (
            success_rate
            >= DEVELOPMENT_CRITERIA[
                "solver_and_constraint_success_rate_at_least"
            ]
        ),
    }
    payload = {
        "stage": "development_only",
        "development_domain_seeds": list(DEVELOPMENT_DOMAIN_SEEDS),
        "methods": list(DEVELOPMENT_METHODS),
        "task_regimes": list(TASK_REGIMES),
        "screening_thresholds": DEVELOPMENT_CRITERIA,
        "comparisons": {
            "conflict_vs_error_peak": conflict,
            "all_vs_full": full,
            "all_vs_plain_violation": plain,
            "all_vs_safe_violation": safe,
        },
        "primary_success_rate": success_rate,
        "criteria": criteria,
        "passed": bool(all(criteria.values())),
        "decision": "FREEZE_AND_CONFIRM" if all(criteria.values()) else "REVISE",
        "total_method_runs": len(ordered),
    }
    _write_csv(output_directory / "dual_anchor_development_raw.csv", ordered)
    (output_directory / "dual_anchor_development_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload
