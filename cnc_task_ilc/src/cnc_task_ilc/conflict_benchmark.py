"""Frozen V7 confirmation on tolerance-conflict task manifests."""

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from .basis import cubic_bspline_basis
from .benchmark import BenchmarkSettings
from .conflict_taskset import (
    TASK_REGIMES,
    load_conflict_taskset,
    specification_from_manifest,
)
from .semantic_task_benchmark import run_semantic_task_method
from .trajectory import make_trajectory_family


EXPECTED_TASKSET_SHA256 = (
    "f2838bb56df7382ab7821c2140dbfd203108469ec529049d0b8263c6d05a51b9"
)
FORMAL_DOMAIN_SEEDS = (811, 827, 853, 877, 907, 929)
AUDIT_DOMAIN_SEEDS = (733, 751, 769)
PRIMARY_METHOD = "violation_safe"
CONFLICT_METHODS = (
    "full_trajectory",
    "static_tolerance",
    "curvature_zones",
    "error_peak_dynamic",
    "violation_dynamic",
    "violation_safe",
    "random_zones",
)
BOOTSTRAP_SEED = 20260719


def taskset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_taskset(path: Path) -> Dict[str, object]:
    digest = taskset_sha256(path)
    if digest != EXPECTED_TASKSET_SHA256:
        raise RuntimeError(
            "task manifest hash changed after preregistration: " + digest
        )
    if set(FORMAL_DOMAIN_SEEDS) & set(AUDIT_DOMAIN_SEEDS):
        raise RuntimeError("formal and audit machine seeds overlap")
    payload = load_conflict_taskset(path)
    if int(payload["manifest_count"]) != 15:
        raise RuntimeError("frozen taskset must contain 15 manifests")
    return payload


def preregistered_protocol() -> Dict[str, object]:
    return {
        "primary_method": PRIMARY_METHOD,
        "primary_scope": "demand_conflict",
        "primary_comparator": "error_peak_dynamic",
        "formal_domain_seeds": list(FORMAL_DOMAIN_SEEDS),
        "excluded_audit_seeds": list(AUDIT_DOMAIN_SEEDS),
        "methods": list(CONFLICT_METHODS),
        "criteria": {
            "conflict_vs_error_peak_bootstrap_ci_lower_above_zero": True,
            "conflict_vs_error_peak_strict_win_rate_at_least_60_percent": 0.60,
            "all_tasks_vs_full_bootstrap_ci_lower_above_zero": True,
            "paired_global_ratio_multiplier_vs_full_no_more_than_1_5": 1.50,
            "solver_and_constraint_success_rate_at_least_95_percent": 0.95,
        },
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    summary = []
    for scope in ("all",) + TASK_REGIMES:
        scoped = (
            list(rows)
            if scope == "all"
            else [row for row in rows if row["regime"] == scope]
        )
        for method in CONFLICT_METHODS:
            selected = [row for row in scoped if row["method"] == method]
            summary.append(
                {
                    "scope": scope,
                    "method": method,
                    "runs": len(selected),
                    "median_task_auc": float(
                        np.median(
                            [row["task_auc_normalized"] for row in selected]
                        )
                    ),
                    "median_final_task_ratio": float(
                        np.median([row["final_task_ratio"] for row in selected])
                    ),
                    "median_final_violation_rate": float(
                        np.median(
                            [row["final_violation_rate"] for row in selected]
                        )
                    ),
                    "median_final_global_ratio": float(
                        np.median(
                            [row["final_global_ratio"] for row in selected]
                        )
                    ),
                    "median_rejected_trials": float(
                        np.median([row["rejected_trials"] for row in selected])
                    ),
                    "success_rate": float(
                        np.mean(
                            [
                                row["all_updates_succeeded"] == 1
                                and row["constraint_violation"] == 0
                                for row in selected
                            ]
                        )
                    ),
                    "median_elapsed_s": float(
                        np.median([row["elapsed_s"] for row in selected])
                    ),
                }
            )
    return summary


def run_conflict_confirmation(
    taskset_path: Path,
    output_directory: Path,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    """Execute the frozen 15-task, 6-domain, 7-method confirmation."""

    if tuple(settings.domain_seeds) != FORMAL_DOMAIN_SEEDS:
        raise RuntimeError("formal domain seeds differ from preregistration")
    if settings.number_of_windows != 2:
        raise RuntimeError("formal active-zone budget must remain two")
    taskset = validate_frozen_taskset(taskset_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for manifest_index, manifest in enumerate(taskset["manifests"]):
        family = str(manifest["trajectory_family"])
        regime = str(manifest["regime"])
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
            for method in CONFLICT_METHODS:
                row = run_semantic_task_method(
                    method,
                    reference,
                    basis,
                    specification,
                    int(domain_seed),
                    settings,
                    random_seed=(
                        50000 + 100 * manifest_index + int(domain_seed)
                    ),
                )
                row["manifest_id"] = str(manifest["manifest_id"])
                row["trajectory"] = family
                row["regime"] = regime
                rows.append(row)

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
    ordered_rows = [
        {column: row[column] for column in column_order} for row in rows
    ]
    summary = _aggregate(ordered_rows)
    metadata = {
        "taskset_path": str(taskset_path),
        "taskset_sha256": taskset_sha256(taskset_path),
        "taskset_manifest_count": int(taskset["manifest_count"]),
        "settings": {
            "samples": settings.samples,
            "duration_s": settings.duration,
            "control_points": settings.control_points,
            "iterations": settings.iterations,
            "zone_budget": settings.number_of_windows,
            "formal_domain_seeds": list(settings.domain_seeds),
            "total_paired_cases": 15 * len(settings.domain_seeds),
            "total_method_runs": len(ordered_rows),
        },
        "preregistered_protocol": preregistered_protocol(),
        "summary": summary,
    }
    _write_csv(output_directory / "conflict_effectiveness_raw.csv", ordered_rows)
    _write_csv(output_directory / "conflict_effectiveness_summary.csv", summary)
    (output_directory / "conflict_effectiveness_metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def _numeric_rows(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    integer_fields = {
        "domain_seed",
        "selection_switches",
        "rejected_trials",
        "constraint_violation",
        "all_updates_succeeded",
    }
    text_fields = {
        "manifest_id",
        "trajectory",
        "regime",
        "method",
        "selection_history",
        "accepted_history",
    }
    rows = []
    for raw_row in raw:
        row = {}
        for key, value in raw_row.items():
            if key in text_fields:
                row[key] = value
            elif key in integer_fields:
                row[key] = int(value)
            else:
                row[key] = float(value)
        rows.append(row)
    return rows


def _paired_statistics(
    rows: Sequence[Dict[str, object]],
    comparator: str,
    regime: Optional[str],
    rng: np.random.Generator,
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
    proposed_global = np.asarray(
        [float(indexed[key + (PRIMARY_METHOD,)]["final_global_ratio"]) for key in keys]
    )
    baseline_global = np.asarray(
        [float(indexed[key + (comparator,)]["final_global_ratio"]) for key in keys]
    )
    proposed_violation = np.asarray(
        [float(indexed[key + (PRIMARY_METHOD,)]["final_violation_rate"]) for key in keys]
    )
    baseline_violation = np.asarray(
        [float(indexed[key + (comparator,)]["final_violation_rate"]) for key in keys]
    )
    improvement = 100.0 * (baseline - proposed) / baseline
    samples = rng.integers(0, improvement.size, size=(20000, improvement.size))
    bootstrap = np.median(improvement[samples], axis=1)
    if np.allclose(proposed, baseline):
        p_value = 1.0
    else:
        p_value = float(
            wilcoxon(proposed, baseline, alternative="less").pvalue
        )
    return {
        "scope": "all" if regime is None else regime,
        "comparator": comparator,
        "paired_cases": len(keys),
        "median_task_auc_improvement_percent": float(np.median(improvement)),
        "mean_task_auc_improvement_percent": float(np.mean(improvement)),
        "paired_win_rate": float(np.mean(proposed < baseline)),
        "paired_tie_rate": float(np.mean(np.isclose(proposed, baseline))),
        "bootstrap_median_improvement_95ci_percent": [
            float(value) for value in np.percentile(bootstrap, (2.5, 97.5))
        ],
        "one_sided_wilcoxon_p_proposed_lower": p_value,
        "median_global_ratio_multiplier": float(
            np.median(proposed_global / np.maximum(baseline_global, 1.0e-12))
        ),
        "median_final_violation_rate_reduction": float(
            np.median(baseline_violation - proposed_violation)
        ),
    }


def _plot_confirmation(
    output_path: Path,
    rows: Sequence[Dict[str, object]],
    comparisons: Dict[str, Dict[str, object]],
    decision: str,
) -> None:
    labels = {
        "full_trajectory": "Full",
        "static_tolerance": "Static tol.",
        "curvature_zones": "Curvature",
        "error_peak_dynamic": "Error peak",
        "violation_dynamic": "Violation",
        "violation_safe": "Safe violation",
        "random_zones": "Random",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))

    regime_labels = ["Neutral", "Aligned", "Conflict"]
    medians = []
    lower = []
    upper = []
    for regime in TASK_REGIMES:
        item = comparisons[regime]["error_peak_dynamic"]
        median = item["median_task_auc_improvement_percent"]
        ci = item["bootstrap_median_improvement_95ci_percent"]
        medians.append(median)
        lower.append(median - ci[0])
        upper.append(ci[1] - median)
    axes[0, 0].bar(
        np.arange(3), medians, color=["#a9b6bd", "#6d8f5f", "#b46b5a"]
    )
    axes[0, 0].errorbar(
        np.arange(3), medians, yerr=[lower, upper], fmt="none", color="#333333", capsize=4
    )
    axes[0, 0].axhline(0.0, color="#666666", linewidth=1.0)
    axes[0, 0].set_xticks(np.arange(3), regime_labels)
    axes[0, 0].set_ylabel("Median AUC improvement vs error peak (%)")
    axes[0, 0].set_title("Primary effect by task regime")

    conflict_rows = [row for row in rows if row["regime"] == "demand_conflict"]
    methods = list(CONFLICT_METHODS)
    auc = [
        np.median(
            [
                float(row["task_auc_normalized"])
                for row in conflict_rows
                if row["method"] == method
            ]
        )
        for method in methods
    ]
    colors = ["#2f6b8a" if method == PRIMARY_METHOD else "#a9b6bd" for method in methods]
    axes[0, 1].bar(np.arange(len(methods)), auc, color=colors)
    axes[0, 1].set_xticks(
        np.arange(len(methods)),
        [labels[method] for method in methods],
        rotation=32,
        ha="right",
    )
    axes[0, 1].set_ylabel("Median normalized task AUC")
    axes[0, 1].set_title("Conflict-task method comparison")

    indexed = {
        (str(row["manifest_id"]), int(row["domain_seed"]), str(row["method"])): row
        for row in conflict_rows
    }
    keys = sorted({(key[0], key[1]) for key in indexed})
    proposed = np.asarray(
        [float(indexed[key + (PRIMARY_METHOD,)]["task_auc_normalized"]) for key in keys]
    )
    baseline = np.asarray(
        [float(indexed[key + ("error_peak_dynamic",)]["task_auc_normalized"]) for key in keys]
    )
    axes[1, 0].scatter(baseline, proposed, color="#2f6b8a", alpha=0.82)
    limits = [min(proposed.min(), baseline.min()), max(proposed.max(), baseline.max())]
    axes[1, 0].plot(limits, limits, color="#666666", linewidth=1.0)
    axes[1, 0].set_xlabel("Error-peak AUC")
    axes[1, 0].set_ylabel("Safe tolerance AUC")
    axes[1, 0].set_title("Paired conflict-task cases")

    task_ratio = []
    global_ratio = []
    for method in methods:
        selected = [row for row in conflict_rows if row["method"] == method]
        task_ratio.append(np.median([row["final_task_ratio"] for row in selected]))
        global_ratio.append(np.median([row["final_global_ratio"] for row in selected]))
    axes[1, 1].scatter(global_ratio, task_ratio, c=colors, s=55)
    annotation_offsets = {
        "full_trajectory": (5, 4),
        "static_tolerance": (5, 5),
        "curvature_zones": (5, 5),
        "error_peak_dynamic": (5, -10),
        "violation_dynamic": (-70, 8),
        "violation_safe": (5, 9),
        "random_zones": (5, 5),
    }
    for method, x_value, y_value in zip(methods, global_ratio, task_ratio):
        axes[1, 1].annotate(
            labels[method],
            (x_value, y_value),
            xytext=annotation_offsets[method],
            textcoords="offset points",
            fontsize=8,
        )
    axes[1, 1].set_xlabel("Median final global-error ratio")
    axes[1, 1].set_ylabel("Median final task ratio")
    axes[1, 1].set_title("Conflict-task/global trade-off")

    fig.suptitle("V7 frozen tolerance-conflict confirmation: " + decision)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_conflict_confirmation(
    result_directory: Path,
) -> Dict[str, object]:
    rows = _numeric_rows(result_directory / "conflict_effectiveness_raw.csv")
    metadata = json.loads(
        (result_directory / "conflict_effectiveness_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if metadata["taskset_sha256"] != EXPECTED_TASKSET_SHA256:
        raise RuntimeError("result metadata does not match frozen taskset hash")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = {}
    for regime in (None,) + TASK_REGIMES:
        scope = "all" if regime is None else regime
        comparisons[scope] = {
            method: _paired_statistics(rows, method, regime, rng)
            for method in CONFLICT_METHODS
            if method != PRIMARY_METHOD
        }

    primary = comparisons["demand_conflict"]["error_peak_dynamic"]
    full = comparisons["all"]["full_trajectory"]
    primary_rows = [row for row in rows if row["method"] == PRIMARY_METHOD]
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
        "conflict_vs_error_peak_ci_lower_above_zero": (
            primary["bootstrap_median_improvement_95ci_percent"][0] > 0.0
        ),
        "conflict_vs_error_peak_win_rate_at_least_60_percent": (
            primary["paired_win_rate"] >= 0.60
        ),
        "all_tasks_vs_full_ci_lower_above_zero": (
            full["bootstrap_median_improvement_95ci_percent"][0] > 0.0
        ),
        "global_tradeoff_vs_full_no_more_than_1_5x": (
            full["median_global_ratio_multiplier"] <= 1.50
        ),
        "success_rate_at_least_95_percent": success_rate >= 0.95,
    }
    decision = "PASS" if all(criteria.values()) else "REVISE"
    payload = {
        "primary_method": PRIMARY_METHOD,
        "taskset_sha256": EXPECTED_TASKSET_SHA256,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "comparisons": comparisons,
        "gate": {
            "decision": decision,
            "passed": bool(all(criteria.values())),
            "criteria": criteria,
            "primary_success_rate": success_rate,
            "primary_conflict_vs_error_peak": primary,
            "secondary_all_tasks_vs_full": full,
        },
    }
    (result_directory / "comparison_v7.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _plot_confirmation(
        result_directory / "conflict_effectiveness_summary.png",
        rows,
        comparisons,
        decision,
    )
    report = f"""# V7 Frozen Tolerance-Conflict Confirmation

## Preregistered primary result

- Decision: **{decision}**
- Primary scope: `demand_conflict`
- Primary comparator: `error_peak_dynamic`
- Paired cases: {primary['paired_cases']}
- Median task-AUC improvement: {primary['median_task_auc_improvement_percent']:.2f}%
- Strict paired win rate: {100.0 * primary['paired_win_rate']:.2f}%
- Bootstrap 95% interval: [{primary['bootstrap_median_improvement_95ci_percent'][0]:.2f}%, {primary['bootstrap_median_improvement_95ci_percent'][1]:.2f}%]
- One-sided Wilcoxon p: {primary['one_sided_wilcoxon_p_proposed_lower']:.6g}

## Secondary all-task result versus full trajectory

- Median task-AUC improvement: {full['median_task_auc_improvement_percent']:.2f}%
- Strict paired win rate: {100.0 * full['paired_win_rate']:.2f}%
- Bootstrap 95% interval: [{full['bootstrap_median_improvement_95ci_percent'][0]:.2f}%, {full['bootstrap_median_improvement_95ci_percent'][1]:.2f}%]
- Paired global-error multiplier: {full['median_global_ratio_multiplier']:.3f}x
- Primary success rate: {100.0 * success_rate:.2f}%

The task-manifest hash, methods, formal seeds and gate were frozen before the
formal run. Audit seeds were excluded.
"""
    (result_directory / "conflict_effectiveness_report.md").write_text(
        report, encoding="utf-8"
    )
    return payload
