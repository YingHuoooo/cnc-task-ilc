"""Frozen V8 confirmation for the dual-anchor semantic ILC scheduler."""

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
from .conflict_benchmark import (
    AUDIT_DOMAIN_SEEDS,
    EXPECTED_TASKSET_SHA256,
    FORMAL_DOMAIN_SEEDS as V7_FORMAL_DOMAIN_SEEDS,
    taskset_sha256,
    validate_frozen_taskset,
)
from .conflict_taskset import TASK_REGIMES, specification_from_manifest
from .dual_anchor_development import DEVELOPMENT_DOMAIN_SEEDS
from .semantic_task_benchmark import run_semantic_task_method
from .trajectory import make_trajectory_family


FORMAL_DOMAIN_SEEDS = (1103, 1129, 1151, 1171, 1193, 1217)
PRIMARY_METHOD = "dual_anchor_dynamic"
FORMAL_METHODS = (
    "full_trajectory",
    "static_tolerance",
    "curvature_zones",
    "error_peak_dynamic",
    "violation_dynamic",
    "violation_safe",
    "dual_anchor_dynamic",
    "random_zones",
)
BOOTSTRAP_SEED = 20260723
FORMAL_SETTINGS = {
    "samples": 161,
    "duration_s": 6.0,
    "control_points": 12,
    "iterations": 4,
    "active_zone_budget": 2,
    "half_width": 5,
}
FORMAL_CRITERIA = {
    "conflict_vs_error_peak_bootstrap_ci_lower_above_zero": True,
    "conflict_vs_error_peak_strict_win_rate_at_least": 0.60,
    "all_tasks_vs_full_bootstrap_ci_lower_above_zero": True,
    "paired_global_ratio_multiplier_vs_full_no_more_than": 1.50,
    "solver_and_constraint_success_rate_at_least": 0.95,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_seed_isolation() -> None:
    groups = {
        "v7_audit": set(AUDIT_DOMAIN_SEEDS),
        "v7_formal": set(V7_FORMAL_DOMAIN_SEEDS),
        "v8_development": set(DEVELOPMENT_DOMAIN_SEEDS),
        "v8_formal": set(FORMAL_DOMAIN_SEEDS),
    }
    names = list(groups)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            if groups[first] & groups[second]:
                raise RuntimeError(first + " and " + second + " seeds overlap")


def preregistered_protocol(
    taskset_path: Path,
    development_summary_path: Path,
    scheduler_source_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Construct the exact protocol that must be frozen before formal runs."""

    _assert_seed_isolation()
    validate_frozen_taskset(taskset_path)
    development = json.loads(development_summary_path.read_text(encoding="utf-8"))
    if not bool(development.get("passed")):
        raise RuntimeError("development screen did not authorize formal confirmation")
    if scheduler_source_path is None:
        scheduler_source_path = Path(__file__).with_name(
            "semantic_task_benchmark.py"
        )
    return {
        "protocol_id": "v8-dual-anchor-frozen-confirmation",
        "frozen_before_formal_execution": True,
        "primary_method": PRIMARY_METHOD,
        "mechanism": {
            "task_anchor": "largest tolerance-normalized semantic urgency",
            "error_anchor": "largest raw zone peak excluding task anchor",
            "anchor_weighting": "equal critical boost after selection",
            "safety": "output-space trust limit and rollback",
        },
        "primary_scope": "demand_conflict",
        "primary_comparator": "error_peak_dynamic",
        "secondary_scope": "all",
        "secondary_comparator": "full_trajectory",
        "taskset_sha256": taskset_sha256(taskset_path),
        "scheduler_source_sha256": file_sha256(scheduler_source_path),
        "development_summary_sha256": file_sha256(development_summary_path),
        "development_domain_seeds_excluded": list(DEVELOPMENT_DOMAIN_SEEDS),
        "v7_audit_domain_seeds_excluded": list(AUDIT_DOMAIN_SEEDS),
        "v7_formal_domain_seeds_excluded": list(V7_FORMAL_DOMAIN_SEEDS),
        "formal_domain_seeds": list(FORMAL_DOMAIN_SEEDS),
        "methods": list(FORMAL_METHODS),
        "settings": FORMAL_SETTINGS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "criteria": FORMAL_CRITERIA,
    }


def freeze_formal_protocol(
    taskset_path: Path,
    development_summary_path: Path,
    protocol_path: Path,
) -> Dict[str, object]:
    """Write once, or verify byte-equivalent semantic content thereafter."""

    protocol = preregistered_protocol(taskset_path, development_summary_path)
    if protocol["taskset_sha256"] != EXPECTED_TASKSET_SHA256:
        raise RuntimeError("unexpected taskset hash while freezing V8")
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("existing V8 protocol differs from current code or inputs")
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return protocol


def validate_frozen_protocol(
    taskset_path: Path,
    development_summary_path: Path,
    protocol_path: Path,
) -> Dict[str, object]:
    if not protocol_path.exists():
        raise RuntimeError("V8 protocol must be frozen before formal execution")
    existing = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = preregistered_protocol(taskset_path, development_summary_path)
    if existing != expected:
        raise RuntimeError("frozen V8 protocol no longer matches code or inputs")
    return existing


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for scope in ("all",) + TASK_REGIMES:
        scoped = (
            list(rows)
            if scope == "all"
            else [row for row in rows if row["regime"] == scope]
        )
        for method in FORMAL_METHODS:
            selected = [row for row in scoped if row["method"] == method]
            summary.append(
                {
                    "scope": scope,
                    "method": method,
                    "runs": len(selected),
                    "median_task_auc": float(
                        np.median([row["task_auc_normalized"] for row in selected])
                    ),
                    "median_final_task_ratio": float(
                        np.median([row["final_task_ratio"] for row in selected])
                    ),
                    "median_final_violation_rate": float(
                        np.median([row["final_violation_rate"] for row in selected])
                    ),
                    "median_final_global_ratio": float(
                        np.median([row["final_global_ratio"] for row in selected])
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
                }
            )
    return summary


def _validate_settings(settings: BenchmarkSettings) -> None:
    if tuple(settings.domain_seeds) != FORMAL_DOMAIN_SEEDS:
        raise RuntimeError("V8 formal seeds differ from the frozen protocol")
    observed = {
        "samples": settings.samples,
        "duration_s": settings.duration,
        "control_points": settings.control_points,
        "iterations": settings.iterations,
        "active_zone_budget": settings.number_of_windows,
        "half_width": settings.half_width,
    }
    if observed != FORMAL_SETTINGS:
        raise RuntimeError("V8 settings differ from the frozen protocol")


def run_dual_anchor_confirmation(
    taskset_path: Path,
    development_summary_path: Path,
    protocol_path: Path,
    output_directory: Path,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    """Execute 15 tasks x 6 new domains x 8 frozen methods."""

    _validate_settings(settings)
    protocol = validate_frozen_protocol(
        taskset_path,
        development_summary_path,
        protocol_path,
    )
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
            for method in FORMAL_METHODS:
                result = run_semantic_task_method(
                    method,
                    reference,
                    basis,
                    specification,
                    int(domain_seed),
                    settings,
                    random_seed=90000 + 100 * manifest_index + int(domain_seed),
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
    summary = _aggregate(ordered)
    metadata = {
        "protocol_sha256": file_sha256(protocol_path),
        "taskset_sha256": taskset_sha256(taskset_path),
        "protocol": protocol,
        "total_paired_cases": 15 * len(FORMAL_DOMAIN_SEEDS),
        "total_method_runs": len(ordered),
        "summary": summary,
    }
    _write_csv(output_directory / "dual_anchor_effectiveness_raw.csv", ordered)
    _write_csv(output_directory / "dual_anchor_effectiveness_summary.csv", summary)
    (output_directory / "dual_anchor_effectiveness_metrics.json").write_text(
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
    rows: List[Dict[str, object]] = []
    for raw_row in raw:
        row: Dict[str, object] = {}
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
    sample_indices = rng.integers(0, improvement.size, size=(20000, improvement.size))
    bootstrap = np.median(improvement[sample_indices], axis=1)
    p_value = (
        1.0
        if np.allclose(proposed, baseline)
        else float(wilcoxon(proposed, baseline, alternative="less").pvalue)
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


def _selection_diagnostics(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    conflict = [row for row in rows if row["regime"] == "demand_conflict"]
    indexed = {
        (str(row["manifest_id"]), int(row["domain_seed"]), str(row["method"])): row
        for row in conflict
    }
    keys = sorted({(key[0], key[1]) for key in indexed})
    identical_histories = []
    jaccards = []
    for key in keys:
        dual = json.loads(str(indexed[key + (PRIMARY_METHOD,)]["selection_history"]))
        peak = json.loads(str(indexed[key + ("error_peak_dynamic",)]["selection_history"]))
        identical_histories.append(dual == peak)
        for dual_trial, peak_trial in zip(dual, peak):
            dual_set = set(dual_trial)
            peak_set = set(peak_trial)
            jaccards.append(len(dual_set & peak_set) / len(dual_set | peak_set))
    return {
        "conflict_identical_selection_history_rate": float(np.mean(identical_histories)),
        "conflict_mean_per_trial_selection_jaccard": float(np.mean(jaccards)),
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
        "dual_anchor_dynamic": "Dual anchor",
        "random_zones": "Random",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.7))

    medians = []
    lower = []
    upper = []
    for regime in TASK_REGIMES:
        item = comparisons[regime]["error_peak_dynamic"]
        median = item["median_task_auc_improvement_percent"]
        interval = item["bootstrap_median_improvement_95ci_percent"]
        medians.append(median)
        lower.append(median - interval[0])
        upper.append(interval[1] - median)
    axes[0, 0].bar(
        np.arange(3), medians, color=["#a9b6bd", "#6d8f5f", "#b46b5a"]
    )
    axes[0, 0].errorbar(
        np.arange(3), medians, yerr=[lower, upper], fmt="none", color="#333333", capsize=4
    )
    axes[0, 0].axhline(0.0, color="#666666", linewidth=1.0)
    axes[0, 0].set_xticks(np.arange(3), ["Neutral", "Aligned", "Conflict"])
    axes[0, 0].set_ylabel("Median AUC improvement vs error peak (%)")
    axes[0, 0].set_title("Effect by tolerance regime")

    conflict_rows = [row for row in rows if row["regime"] == "demand_conflict"]
    auc = [
        np.median(
            [
                float(row["task_auc_normalized"])
                for row in conflict_rows
                if row["method"] == method
            ]
        )
        for method in FORMAL_METHODS
    ]
    colors = [
        "#2f6b8a" if method == PRIMARY_METHOD else "#a9b6bd"
        for method in FORMAL_METHODS
    ]
    axes[0, 1].bar(np.arange(len(FORMAL_METHODS)), auc, color=colors)
    axes[0, 1].set_xticks(
        np.arange(len(FORMAL_METHODS)),
        [labels[method] for method in FORMAL_METHODS],
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
    dual = np.asarray(
        [float(indexed[key + (PRIMARY_METHOD,)]["task_auc_normalized"]) for key in keys]
    )
    peak = np.asarray(
        [float(indexed[key + ("error_peak_dynamic",)]["task_auc_normalized"]) for key in keys]
    )
    axes[1, 0].scatter(peak, dual, color="#2f6b8a", alpha=0.82)
    limits = [min(dual.min(), peak.min()), max(dual.max(), peak.max())]
    axes[1, 0].plot(limits, limits, color="#666666", linewidth=1.0)
    axes[1, 0].set_xlabel("Error-peak AUC")
    axes[1, 0].set_ylabel("Dual-anchor AUC")
    axes[1, 0].set_title("Paired conflict-task cases")

    task_ratio = []
    global_ratio = []
    for method in FORMAL_METHODS:
        selected = [row for row in conflict_rows if row["method"] == method]
        task_ratio.append(np.median([row["final_task_ratio"] for row in selected]))
        global_ratio.append(np.median([row["final_global_ratio"] for row in selected]))
    tradeoff_colors = [
        "#2f6b8a"
        if method == PRIMARY_METHOD
        else "#6d8f5f"
        if method == "full_trajectory"
        else "#b46b5a"
        if method == "error_peak_dynamic"
        else "#a9b6bd"
        for method in FORMAL_METHODS
    ]
    axes[1, 1].scatter(global_ratio, task_ratio, c=tradeoff_colors, s=55)
    annotation_offsets = {
        "full_trajectory": (5, 5),
        "static_tolerance": (-58, 4),
        "error_peak_dynamic": (8, 9),
        "dual_anchor_dynamic": (5, 6),
        "random_zones": (5, 6),
    }
    for method, x_value, y_value in zip(FORMAL_METHODS, global_ratio, task_ratio):
        if method not in annotation_offsets:
            continue
        axes[1, 1].annotate(
            labels[method],
            (x_value, y_value),
            xytext=annotation_offsets[method],
            textcoords="offset points",
            fontsize=7.5,
        )
    axes[1, 1].set_xlabel("Median final global-error ratio")
    axes[1, 1].set_ylabel("Median final task ratio")
    axes[1, 1].set_title("Conflict task/global trade-off")

    fig.suptitle("V8 frozen dual-anchor confirmation: " + decision)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_dual_anchor_confirmation(result_directory: Path) -> Dict[str, object]:
    rows = _numeric_rows(result_directory / "dual_anchor_effectiveness_raw.csv")
    metadata = json.loads(
        (result_directory / "dual_anchor_effectiveness_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    protocol = metadata["protocol"]
    if protocol["taskset_sha256"] != EXPECTED_TASKSET_SHA256:
        raise RuntimeError("V8 results do not match the frozen taskset")
    if protocol["formal_domain_seeds"] != list(FORMAL_DOMAIN_SEEDS):
        raise RuntimeError("V8 results do not match frozen formal seeds")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons: Dict[str, Dict[str, object]] = {}
    for regime in (None,) + TASK_REGIMES:
        scope = "all" if regime is None else regime
        comparisons[scope] = {
            method: _paired_statistics(rows, method, regime, rng)
            for method in FORMAL_METHODS
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
            primary["paired_win_rate"]
            >= FORMAL_CRITERIA["conflict_vs_error_peak_strict_win_rate_at_least"]
        ),
        "all_tasks_vs_full_ci_lower_above_zero": (
            full["bootstrap_median_improvement_95ci_percent"][0] > 0.0
        ),
        "global_tradeoff_vs_full_no_more_than_1_5x": (
            full["median_global_ratio_multiplier"]
            <= FORMAL_CRITERIA[
                "paired_global_ratio_multiplier_vs_full_no_more_than"
            ]
        ),
        "success_rate_at_least_95_percent": (
            success_rate
            >= FORMAL_CRITERIA["solver_and_constraint_success_rate_at_least"]
        ),
    }
    decision = "PASS" if all(criteria.values()) else "REVISE"
    diagnostics = _selection_diagnostics(rows)
    payload = {
        "primary_method": PRIMARY_METHOD,
        "protocol_sha256": metadata["protocol_sha256"],
        "taskset_sha256": EXPECTED_TASKSET_SHA256,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "comparisons": comparisons,
        "selection_diagnostics": diagnostics,
        "gate": {
            "decision": decision,
            "passed": bool(all(criteria.values())),
            "criteria": criteria,
            "primary_success_rate": success_rate,
            "primary_conflict_vs_error_peak": primary,
            "secondary_all_tasks_vs_full": full,
        },
    }
    (result_directory / "comparison_v8.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _plot_confirmation(
        result_directory / "dual_anchor_effectiveness_summary.png",
        rows,
        comparisons,
        decision,
    )
    english_report = f"""# V8 Frozen Dual-Anchor Confirmation

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

The taskset, implementation hash, methods, settings, formal seeds, bootstrap
seed and decision thresholds were frozen before any V8 formal run.
"""
    (result_directory / "dual_anchor_effectiveness_report.md").write_text(
        english_report,
        encoding="utf-8",
    )
    chinese_report = f"""# V8 双锚点调度有效性诊断

## 正式结论

- 冻结判定：**{decision}**
- 冲突公差任务相对误差峰值法：中位任务 AUC 提升 {primary['median_task_auc_improvement_percent']:.2f}%，严格胜率 {100.0 * primary['paired_win_rate']:.2f}%。
- 上述中位提升的 bootstrap 95% 区间：[{primary['bootstrap_median_improvement_95ci_percent'][0]:.2f}%，{primary['bootstrap_median_improvement_95ci_percent'][1]:.2f}%]。
- 单侧 Wilcoxon 检验 p={primary['one_sided_wilcoxon_p_proposed_lower']:.6g}。
- 全部任务相对全轨迹法：中位任务 AUC 提升 {full['median_task_auc_improvement_percent']:.2f}%，95% 区间 [{full['bootstrap_median_improvement_95ci_percent'][0]:.2f}%，{full['bootstrap_median_improvement_95ci_percent'][1]:.2f}%]。
- 相对全轨迹法的配对全局误差倍率为 {full['median_global_ratio_multiplier']:.3f}x，求解与约束成功率为 {100.0 * success_rate:.2f}%。

## 机制检查

冲突任务中，双锚点法与误差峰值法的完整选区历史完全一致率为 {100.0 * diagnostics['conflict_identical_selection_history_rate']:.2f}%，逐次选区平均 Jaccard 相似度为 {diagnostics['conflict_mean_per_trial_selection_jaccard']:.3f}。这用于确认性能差异确实伴随调度行为变化，而不是方法标签变化但实际选择相同。

## 可解释边界

该结论只证明在当前虚拟物理机床、冻结的 15 个任务清单、每次两个活动区和四次 ILC 更新条件下有效；它不能替代真实机床实验，也不能证明对任意动力学、噪声、时延和刀具切削载荷均有效。
"""
    (result_directory / "effectiveness_diagnosis_zh.md").write_text(
        chinese_report,
        encoding="utf-8",
    )
    return payload
