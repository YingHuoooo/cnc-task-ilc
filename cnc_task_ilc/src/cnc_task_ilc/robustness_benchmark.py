"""V9 one-factor-at-a-time robustness audit and frozen confirmation."""

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
from .conflict_benchmark import EXPECTED_TASKSET_SHA256, taskset_sha256, validate_frozen_taskset
from .conflict_taskset import specification_from_manifest
from .robustness_runner import (
    ROBUSTNESS_METHODS,
    STRESS_SCENARIOS,
    StressScenario,
    run_robustness_method,
    scenario_to_dict,
)
from .trajectory import make_trajectory_family


AUDIT_DOMAIN_SEEDS = (1231, 1249)
FORMAL_DOMAIN_SEEDS = (1301, 1321, 1361, 1381)
AUDIT_SCENARIO_IDS = (
    "baseline",
    "noise_0p05",
    "delay_4",
    "mismatch_1p70",
)
PRIMARY_METHOD = "dual_anchor_dynamic"
PRIMARY_COMPARATOR = "error_peak_dynamic"
BOOTSTRAP_SEED = 20260729
FORMAL_SETTINGS = {
    "samples": 161,
    "duration_s": 6.0,
    "control_points": 12,
    "iterations": 4,
    "active_zone_budget": 2,
    "half_width": 5,
}
ROBUST_EFFECT_RULE = {
    "bootstrap_median_improvement_ci_lower_above_zero": True,
    "strict_paired_win_rate_at_least": 0.60,
}
CLASSIFICATION_RULE = {
    "broadly_robust_minimum_stress_scenarios": 5,
    "conditionally_robust_minimum_stress_scenarios": 3,
    "each_factor_must_retain_at_least_one_nonbaseline_level": True,
    "per_scenario_solver_and_constraint_success_at_least": 0.95,
}

ROW_COLUMNS = [
    "manifest_id",
    "trajectory",
    "regime",
    "scenario_id",
    "stress_factor",
    "stress_level",
    "measurement_noise_std_mm",
    "extra_delay_steps",
    "mismatch_scale",
    "domain_seed",
    "method",
    "initial_task_score",
    "final_task_score",
    "last_observed_task_score",
    "task_auc_normalized",
    "final_task_ratio",
    "initial_violation_rate",
    "final_violation_rate",
    "initial_global_rmse",
    "final_global_rmse",
    "final_global_ratio",
    "measured_score_auc_normalized",
    "selection_switches",
    "selection_history",
    "accepted_history",
    "rejected_trials",
    "final_trust_radius_mm",
    "constraint_violation",
    "all_updates_succeeded",
    "finite_result",
    "elapsed_s",
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario(scenario_id: str) -> StressScenario:
    return next(item for item in STRESS_SCENARIOS if item.scenario_id == scenario_id)


def _assert_seed_isolation() -> None:
    from .conflict_benchmark import (
        AUDIT_DOMAIN_SEEDS as V7_AUDIT,
        FORMAL_DOMAIN_SEEDS as V7_FORMAL,
    )
    from .dual_anchor_confirmation import FORMAL_DOMAIN_SEEDS as V8_FORMAL
    from .dual_anchor_development import DEVELOPMENT_DOMAIN_SEEDS as V8_DEVELOPMENT

    groups = (
        set(V7_AUDIT),
        set(V7_FORMAL),
        set(V8_DEVELOPMENT),
        set(V8_FORMAL),
        set(AUDIT_DOMAIN_SEEDS),
        set(FORMAL_DOMAIN_SEEDS),
    )
    for first_index, first in enumerate(groups):
        for second in groups[first_index + 1 :]:
            if first & second:
                raise RuntimeError("V9 machine-domain seeds overlap earlier stages")


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _validate_settings(settings: BenchmarkSettings, seeds: Tuple[int, ...]) -> None:
    if tuple(settings.domain_seeds) != seeds:
        raise RuntimeError("machine-domain seeds differ from the declared V9 stage")
    observed = {
        "samples": settings.samples,
        "duration_s": settings.duration,
        "control_points": settings.control_points,
        "iterations": settings.iterations,
        "active_zone_budget": settings.number_of_windows,
        "half_width": settings.half_width,
    }
    if observed != FORMAL_SETTINGS:
        raise RuntimeError("V9 settings differ from the declared protocol")


def _run_grid(
    manifests: Sequence[Dict[str, object]],
    scenarios: Sequence[StressScenario],
    settings: BenchmarkSettings,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scenario_index, scenario in enumerate(scenarios):
        for manifest_index, manifest in enumerate(manifests):
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
                shared_noise_seed = (
                    120000
                    + 10000 * scenario_index
                    + 100 * manifest_index
                    + int(domain_seed)
                )
                for method in ROBUSTNESS_METHODS:
                    result = run_robustness_method(
                        method,
                        reference,
                        basis,
                        specification,
                        int(domain_seed),
                        settings,
                        scenario,
                        shared_noise_seed,
                    )
                    result["manifest_id"] = str(manifest["manifest_id"])
                    result["trajectory"] = family
                    result["regime"] = str(manifest["regime"])
                    rows.append({column: result[column] for column in ROW_COLUMNS})
        print(
            "[V9] completed scenario "
            + scenario.scenario_id
            + " ("
            + str(len(rows))
            + " method runs)",
            flush=True,
        )
    return rows


def run_robustness_audit(
    taskset_path: Path,
    output_directory: Path,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    """Numerically audit the maximum level of every factor without tuning."""

    _assert_seed_isolation()
    _validate_settings(settings, AUDIT_DOMAIN_SEEDS)
    taskset = validate_frozen_taskset(taskset_path)
    manifests = [
        item for item in taskset["manifests"] if item["regime"] == "demand_conflict"
    ]
    scenarios = [_scenario(item) for item in AUDIT_SCENARIO_IDS]
    rows = _run_grid(manifests, scenarios, settings)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "robustness_audit_raw.csv", rows)
    primary_rows = [row for row in rows if row["method"] == PRIMARY_METHOD]
    finite_rate = float(np.mean([row["finite_result"] == 1 for row in rows]))
    success_rate = float(
        np.mean(
            [
                row["finite_result"] == 1
                and row["all_updates_succeeded"] == 1
                and row["constraint_violation"] == 0
                for row in primary_rows
            ]
        )
    )
    criteria = {
        "all_audit_results_finite": finite_rate == 1.0,
        "primary_success_rate_at_least_95_percent": success_rate >= 0.95,
        "all_four_declared_methods_executed": {
            str(row["method"]) for row in rows
        }
        == set(ROBUSTNESS_METHODS),
    }
    payload = {
        "stage": "v9_numerical_audit_only",
        "taskset_sha256": taskset_sha256(taskset_path),
        "audit_domain_seeds": list(AUDIT_DOMAIN_SEEDS),
        "audit_scenario_ids": list(AUDIT_SCENARIO_IDS),
        "methods": list(ROBUSTNESS_METHODS),
        "total_method_runs": len(rows),
        "finite_rate": finite_rate,
        "primary_success_rate": success_rate,
        "criteria": criteria,
        "passed": bool(all(criteria.values())),
        "decision": "FREEZE_FORMAL_PROTOCOL" if all(criteria.values()) else "REVISE",
    }
    (output_directory / "robustness_audit_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def preregistered_protocol(
    taskset_path: Path,
    audit_summary_path: Path,
    v8_protocol_path: Path,
) -> Dict[str, object]:
    _assert_seed_isolation()
    audit = json.loads(audit_summary_path.read_text(encoding="utf-8"))
    if not bool(audit.get("passed")):
        raise RuntimeError("V9 audit did not authorize formal execution")
    v8_protocol = json.loads(v8_protocol_path.read_text(encoding="utf-8"))
    scheduler_path = Path(__file__).with_name("semantic_task_benchmark.py")
    runner_path = Path(__file__).with_name("robustness_runner.py")
    if v8_protocol["taskset_sha256"] != taskset_sha256(taskset_path):
        raise RuntimeError("V8 protocol and V9 taskset hashes differ")
    if v8_protocol["scheduler_source_sha256"] != file_sha256(scheduler_path):
        raise RuntimeError("frozen V8 scheduler changed before V9")
    return {
        "protocol_id": "v9-oat-robustness-boundary",
        "frozen_before_formal_execution": True,
        "design": "one-factor-at-a-time with a shared baseline",
        "true_error_used_for_evaluation": True,
        "noisy_error_used_for_learning_and_rollback": True,
        "primary_method": PRIMARY_METHOD,
        "primary_scope": "demand_conflict",
        "primary_comparator": PRIMARY_COMPARATOR,
        "methods": list(ROBUSTNESS_METHODS),
        "scenarios": [scenario_to_dict(item) for item in STRESS_SCENARIOS],
        "settings": FORMAL_SETTINGS,
        "formal_domain_seeds": list(FORMAL_DOMAIN_SEEDS),
        "audit_domain_seeds_excluded": list(AUDIT_DOMAIN_SEEDS),
        "taskset_sha256": taskset_sha256(taskset_path),
        "v8_protocol_sha256": file_sha256(v8_protocol_path),
        "audit_summary_sha256": file_sha256(audit_summary_path),
        "robustness_runner_sha256": file_sha256(runner_path),
        "v8_scheduler_sha256": file_sha256(scheduler_path),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "robust_effect_rule": ROBUST_EFFECT_RULE,
        "classification_rule": CLASSIFICATION_RULE,
        "formal_method_runs": (
            15
            * len(FORMAL_DOMAIN_SEEDS)
            * len(STRESS_SCENARIOS)
            * len(ROBUSTNESS_METHODS)
        ),
    }


def freeze_robustness_protocol(
    taskset_path: Path,
    audit_summary_path: Path,
    v8_protocol_path: Path,
    protocol_path: Path,
) -> Dict[str, object]:
    protocol = preregistered_protocol(
        taskset_path,
        audit_summary_path,
        v8_protocol_path,
    )
    if protocol["taskset_sha256"] != EXPECTED_TASKSET_SHA256:
        raise RuntimeError("unexpected taskset hash while freezing V9")
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("existing V9 protocol differs from current inputs")
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return protocol


def validate_robustness_protocol(
    taskset_path: Path,
    audit_summary_path: Path,
    v8_protocol_path: Path,
    protocol_path: Path,
) -> Dict[str, object]:
    if not protocol_path.exists():
        raise RuntimeError("V9 protocol must be frozen before formal execution")
    existing = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = preregistered_protocol(
        taskset_path,
        audit_summary_path,
        v8_protocol_path,
    )
    if existing != expected:
        raise RuntimeError("frozen V9 protocol no longer matches code or inputs")
    return existing


def run_robustness_confirmation(
    taskset_path: Path,
    audit_summary_path: Path,
    v8_protocol_path: Path,
    protocol_path: Path,
    output_directory: Path,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    _validate_settings(settings, FORMAL_DOMAIN_SEEDS)
    protocol = validate_robustness_protocol(
        taskset_path,
        audit_summary_path,
        v8_protocol_path,
        protocol_path,
    )
    taskset = validate_frozen_taskset(taskset_path)
    rows = _run_grid(taskset["manifests"], STRESS_SCENARIOS, settings)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "robustness_effectiveness_raw.csv", rows)
    metadata = {
        "protocol_sha256": file_sha256(protocol_path),
        "protocol": protocol,
        "total_method_runs": len(rows),
        "paired_cases_per_scenario_all_tasks": 15 * len(FORMAL_DOMAIN_SEEDS),
        "paired_cases_per_scenario_conflict": 5 * len(FORMAL_DOMAIN_SEEDS),
    }
    (output_directory / "robustness_effectiveness_metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def _numeric_rows(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    text_fields = {
        "manifest_id",
        "trajectory",
        "regime",
        "scenario_id",
        "stress_factor",
        "method",
        "selection_history",
        "accepted_history",
    }
    integer_fields = {
        "domain_seed",
        "extra_delay_steps",
        "selection_switches",
        "rejected_trials",
        "constraint_violation",
        "all_updates_succeeded",
        "finite_result",
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
    scenario_id: str,
    comparator: str,
    regime: Optional[str],
    rng: np.random.Generator,
) -> Dict[str, object]:
    scoped = [
        row
        for row in rows
        if row["scenario_id"] == scenario_id
        and (regime is None or row["regime"] == regime)
    ]
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
    sample_indices = rng.integers(0, improvement.size, size=(20000, improvement.size))
    bootstrap = np.median(improvement[sample_indices], axis=1)
    p_value = (
        1.0
        if np.allclose(proposed, baseline)
        else float(wilcoxon(proposed, baseline, alternative="less").pvalue)
    )
    return {
        "scenario_id": scenario_id,
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
    }


def _absolute_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for scenario in STRESS_SCENARIOS:
        for scope in ("all", "demand_conflict"):
            scoped = [
                row
                for row in rows
                if row["scenario_id"] == scenario.scenario_id
                and (scope == "all" or row["regime"] == scope)
            ]
            for method in ROBUSTNESS_METHODS:
                selected = [row for row in scoped if row["method"] == method]
                summary.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "stress_factor": scenario.factor,
                        "stress_level": scenario.factor_level,
                        "scope": scope,
                        "method": method,
                        "runs": len(selected),
                        "median_task_auc": float(
                            np.median([row["task_auc_normalized"] for row in selected])
                        ),
                        "median_final_task_ratio": float(
                            np.median([row["final_task_ratio"] for row in selected])
                        ),
                        "median_final_global_ratio": float(
                            np.median([row["final_global_ratio"] for row in selected])
                        ),
                        "success_rate": float(
                            np.mean(
                                [
                                    row["finite_result"] == 1
                                    and row["all_updates_succeeded"] == 1
                                    and row["constraint_violation"] == 0
                                    for row in selected
                                ]
                            )
                        ),
                    }
                )
    return summary


def _scenario_degradation(
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    indexed = {
        (
            str(row["manifest_id"]),
            int(row["domain_seed"]),
            str(row["scenario_id"]),
            str(row["method"]),
        ): row
        for row in rows
    }
    keys = sorted(
        {
            (str(row["manifest_id"]), int(row["domain_seed"]))
            for row in rows
        }
    )
    output: List[Dict[str, object]] = []
    for scenario in STRESS_SCENARIOS:
        for method in ROBUSTNESS_METHODS:
            changes = []
            for key in keys:
                current = float(
                    indexed[key + (scenario.scenario_id, method)][
                        "task_auc_normalized"
                    ]
                )
                baseline = float(
                    indexed[key + ("baseline", method)]["task_auc_normalized"]
                )
                changes.append(100.0 * (current - baseline) / baseline)
            output.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "method": method,
                    "median_auc_degradation_percent": float(np.median(changes)),
                    "mean_auc_degradation_percent": float(np.mean(changes)),
                }
            )
    return output


def _plot_robustness(
    output_path: Path,
    summary: Sequence[Dict[str, object]],
    conflict_vs_peak: Dict[str, Dict[str, object]],
    classification: str,
) -> None:
    labels = {
        "full_trajectory": "Full",
        "error_peak_dynamic": "Error peak",
        "violation_safe": "Safe tolerance",
        "dual_anchor_dynamic": "Dual anchor",
    }
    colors = {
        "full_trajectory": "#6d8f5f",
        "error_peak_dynamic": "#b46b5a",
        "violation_safe": "#a9b6bd",
        "dual_anchor_dynamic": "#2f6b8a",
    }
    factor_panels = (
        ("measurement_noise", ["baseline", "noise_0p02", "noise_0p05"], "Noise std. (mm)", [0.0, 0.02, 0.05]),
        ("added_delay", ["baseline", "delay_2", "delay_4"], "Extra delay (steps)", [0.0, 2.0, 4.0]),
        ("dynamic_mismatch", ["baseline", "mismatch_1p35", "mismatch_1p70"], "Mismatch scale", [1.0, 1.35, 1.70]),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8))
    for axis, (_, scenario_ids, x_label, levels) in zip(axes.flat[:3], factor_panels):
        for method in ROBUSTNESS_METHODS:
            values = []
            for scenario_id in scenario_ids:
                item = next(
                    row
                    for row in summary
                    if row["scenario_id"] == scenario_id
                    and row["scope"] == "demand_conflict"
                    and row["method"] == method
                )
                values.append(float(item["median_task_auc"]))
            axis.plot(
                levels,
                values,
                marker="o",
                linewidth=1.8,
                color=colors[method],
                label=labels[method],
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel("Median normalized task AUC")
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axes[0, 0].set_title("Measurement-noise robustness")
    axes[0, 1].set_title("Additional-delay robustness")
    axes[1, 0].set_title("Dynamic-mismatch robustness")
    axes[0, 0].legend(frameon=False, fontsize=8)

    scenario_ids = [item.scenario_id for item in STRESS_SCENARIOS]
    medians = [
        conflict_vs_peak[item]["median_task_auc_improvement_percent"]
        for item in scenario_ids
    ]
    intervals = [
        conflict_vs_peak[item]["bootstrap_median_improvement_95ci_percent"]
        for item in scenario_ids
    ]
    lower = [median - interval[0] for median, interval in zip(medians, intervals)]
    upper = [interval[1] - median for median, interval in zip(medians, intervals)]
    bar_colors = [
        "#2f6b8a"
        if interval[0] > 0.0
        and conflict_vs_peak[scenario_id]["paired_win_rate"] >= 0.60
        else "#a9b6bd"
        for scenario_id, interval in zip(scenario_ids, intervals)
    ]
    axes[1, 1].bar(np.arange(len(scenario_ids)), medians, color=bar_colors)
    axes[1, 1].errorbar(
        np.arange(len(scenario_ids)),
        medians,
        yerr=[lower, upper],
        fmt="none",
        color="#333333",
        capsize=3,
    )
    axes[1, 1].axhline(0.0, color="#666666", linewidth=1.0)
    axes[1, 1].set_xticks(
        np.arange(len(scenario_ids)),
        ["Base", "N.02", "N.05", "D+2", "D+4", "M1.35", "M1.70"],
        rotation=30,
        ha="right",
    )
    axes[1, 1].set_ylabel("Dual-anchor AUC improvement vs error peak (%)")
    axes[1, 1].set_title("Retained paired advantage")
    fig.suptitle("V9 virtual-machine robustness: " + classification)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_robustness_confirmation(result_directory: Path) -> Dict[str, object]:
    rows = _numeric_rows(result_directory / "robustness_effectiveness_raw.csv")
    metadata = json.loads(
        (result_directory / "robustness_effectiveness_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if metadata["protocol"]["taskset_sha256"] != EXPECTED_TASKSET_SHA256:
        raise RuntimeError("V9 result metadata has the wrong taskset hash")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons: Dict[str, Dict[str, Dict[str, object]]] = {}
    for scenario in STRESS_SCENARIOS:
        comparisons[scenario.scenario_id] = {}
        for scope, regime in (("all", None), ("demand_conflict", "demand_conflict")):
            comparisons[scenario.scenario_id][scope] = {
                method: _paired_statistics(
                    rows,
                    scenario.scenario_id,
                    method,
                    regime,
                    rng,
                )
                for method in ROBUSTNESS_METHODS
                if method != PRIMARY_METHOD
            }

    conflict_vs_peak = {
        scenario.scenario_id: comparisons[scenario.scenario_id][
            "demand_conflict"
        ][PRIMARY_COMPARATOR]
        for scenario in STRESS_SCENARIOS
    }
    robust_flags = {
        scenario_id: bool(
            item["bootstrap_median_improvement_95ci_percent"][0] > 0.0
            and item["paired_win_rate"]
            >= ROBUST_EFFECT_RULE["strict_paired_win_rate_at_least"]
        )
        for scenario_id, item in conflict_vs_peak.items()
    }
    nonbaseline = [item.scenario_id for item in STRESS_SCENARIOS[1:]]
    robust_stress_count = sum(robust_flags[item] for item in nonbaseline)
    factor_ids = {
        "measurement_noise": ("noise_0p02", "noise_0p05"),
        "added_delay": ("delay_2", "delay_4"),
        "dynamic_mismatch": ("mismatch_1p35", "mismatch_1p70"),
    }
    factor_retention = {
        factor: any(robust_flags[item] for item in scenario_ids)
        for factor, scenario_ids in factor_ids.items()
    }
    primary_rows = [row for row in rows if row["method"] == PRIMARY_METHOD]
    per_scenario_success = {
        scenario.scenario_id: float(
            np.mean(
                [
                    row["finite_result"] == 1
                    and row["all_updates_succeeded"] == 1
                    and row["constraint_violation"] == 0
                    for row in primary_rows
                    if row["scenario_id"] == scenario.scenario_id
                ]
            )
        )
        for scenario in STRESS_SCENARIOS
    }
    success_ok = all(
        value
        >= CLASSIFICATION_RULE[
            "per_scenario_solver_and_constraint_success_at_least"
        ]
        for value in per_scenario_success.values()
    )
    factor_ok = all(factor_retention.values())
    if (
        robust_stress_count
        >= CLASSIFICATION_RULE["broadly_robust_minimum_stress_scenarios"]
        and factor_ok
        and success_ok
    ):
        classification = "BROADLY_ROBUST"
    elif (
        robust_stress_count
        >= CLASSIFICATION_RULE[
            "conditionally_robust_minimum_stress_scenarios"
        ]
        and factor_ok
        and success_ok
    ):
        classification = "CONDITIONALLY_ROBUST"
    else:
        classification = "FRAGILE_OR_UNRESOLVED"

    factor_boundaries = {}
    for factor, scenario_ids in factor_ids.items():
        retained = [item for item in scenario_ids if robust_flags[item]]
        factor_boundaries[factor] = retained[-1] if retained else "baseline_only"

    summary = _absolute_summary(rows)
    degradation = _scenario_degradation(rows)
    payload = {
        "classification": classification,
        "taskset_sha256": EXPECTED_TASKSET_SHA256,
        "protocol_sha256": metadata["protocol_sha256"],
        "comparisons": comparisons,
        "robust_effect_flags": robust_flags,
        "robust_nonbaseline_scenario_count": robust_stress_count,
        "factor_retention": factor_retention,
        "factor_boundaries": factor_boundaries,
        "primary_success_rate_by_scenario": per_scenario_success,
        "absolute_summary": summary,
        "degradation_from_baseline": degradation,
    }
    (result_directory / "comparison_v9_robustness.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(result_directory / "robustness_summary.csv", summary)
    _write_csv(result_directory / "robustness_degradation.csv", degradation)
    _plot_robustness(
        result_directory / "robustness_summary.png",
        summary,
        conflict_vs_peak,
        classification,
    )

    lines = [
        "# V9 虚拟物理机床鲁棒性实验",
        "",
        "## 结论",
        "",
        "- 冻结分类：**" + classification + "**",
        "- 六个非基线压力场景中保留严格统计优势："
        + str(robust_stress_count)
        + "/6。",
        "- 三类因素至少一个压力等级保持优势："
        + ("是。" if factor_ok else "否。"),
        "- 所有场景的求解与约束成功率均达到 95%："
        + ("是。" if success_ok else "否。"),
        "",
        "## 冲突任务：双锚点相对误差峰值法",
        "",
        "| 场景 | 中位 AUC 提升 | 严格胜率 | Bootstrap 95% CI | 保留优势 |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario in STRESS_SCENARIOS:
        item = conflict_vs_peak[scenario.scenario_id]
        interval = item["bootstrap_median_improvement_95ci_percent"]
        lines.append(
            "| "
            + scenario.label
            + " | "
            + f"{item['median_task_auc_improvement_percent']:.2f}%"
            + " | "
            + f"{100.0 * item['paired_win_rate']:.2f}%"
            + " | ["
            + f"{interval[0]:.2f}%, {interval[1]:.2f}%"
            + "] | "
            + ("是" if robust_flags[scenario.scenario_id] else "否")
            + " |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "这是单因素压力实验：噪声、附加时延和动力学失配分别改变，未覆盖三者同时恶化的交互效应。学习器使用带噪反馈，评价使用虚拟机床的无噪真实位置，因此噪声结果不是用带噪指标自我证明。该实验仍属于仿真证据，不能代替真实机床验证。",
            "",
        ]
    )
    (result_directory / "robustness_diagnosis_zh.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return payload
