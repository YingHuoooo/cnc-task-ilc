"""V12 unknown-delay development, preregistration and confirmation."""

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
    EXPECTED_TASKSET_SHA256,
    taskset_sha256,
    validate_frozen_taskset,
)
from .conflict_taskset import specification_from_manifest
from .delay_generalization_runner import (
    FIXED_DELAY_CANDIDATES,
    GENERALIZATION_METHODS,
    GENERALIZATION_SCENARIOS,
    ONLINE_COMPENSATION_GAIN,
    DelayGeneralizationScenario,
    run_delay_generalization_method,
    scenario_to_dict,
)
from .trajectory import make_trajectory_family


DEVELOPMENT_DOMAIN_SEEDS = (1709, 1723, 1741)
FORMAL_DOMAIN_SEEDS = (1801, 1823, 1847, 1861)
PRIMARY_METHOD = "delay_aware_dual_anchor"
ORACLE_METHOD = "oracle_true_delay_0p25"
BASELINE_METHOD = "dual_anchor_dynamic"
BOOTSTRAP_SEED = 20260819
FORMAL_SETTINGS = {
    "samples": 161,
    "duration_s": 6.0,
    "control_points": 12,
    "iterations": 4,
    "active_zone_budget": 2,
    "half_width": 5,
}
DEVELOPMENT_CRITERIA = {
    "each_scenario_median_improvement_vs_best_fixed_above_percent": 0.0,
    "each_scenario_win_rate_vs_best_fixed_at_least": 0.60,
    "median_axis_lag_absolute_error_at_most_steps": 1.0,
    "median_excess_auc_over_oracle_at_most_percent": 10.0,
    "solver_and_constraint_success_rate_at_least": 0.95,
}
FORMAL_CRITERIA = {
    "each_scenario_ci_lower_vs_best_fixed_above_zero": True,
    "each_scenario_win_rate_vs_best_fixed_at_least": 0.60,
    "each_scenario_ci_lower_vs_original_above_zero": True,
    "median_axis_lag_absolute_error_at_most_steps": 1.0,
    "median_excess_auc_over_oracle_at_most_percent": 10.0,
    "each_scenario_median_auc_below": 1.0,
    "each_scenario_median_final_ratio_below": 1.0,
    "each_scenario_success_rate_at_least": 0.95,
}

ROW_COLUMNS = [
    "manifest_id",
    "trajectory",
    "regime",
    "scenario_id",
    "scenario_mode",
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
    "base_axis_delay_steps",
    "extra_delay_schedule",
    "actual_axis_delay_history",
    "raw_estimated_lag_history",
    "total_estimated_lag_history",
    "nominal_estimated_lag_history",
    "applied_lag_history",
    "lag_absolute_error_steps",
    "median_peak_correlation",
    "elapsed_s",
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_settings(
    settings: BenchmarkSettings,
    expected_seeds: Tuple[int, ...],
) -> None:
    observed = {
        "samples": settings.samples,
        "duration_s": settings.duration,
        "control_points": settings.control_points,
        "iterations": settings.iterations,
        "active_zone_budget": settings.number_of_windows,
        "half_width": settings.half_width,
    }
    if tuple(settings.domain_seeds) != expected_seeds:
        raise RuntimeError("V12 seeds differ from the declared stage")
    if observed != FORMAL_SETTINGS:
        raise RuntimeError("V12 settings differ from the declared design")


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _numeric_rows(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    text_fields = {
        "manifest_id",
        "trajectory",
        "regime",
        "scenario_id",
        "scenario_mode",
        "method",
        "selection_history",
        "accepted_history",
        "base_axis_delay_steps",
        "extra_delay_schedule",
        "actual_axis_delay_history",
        "raw_estimated_lag_history",
        "total_estimated_lag_history",
        "nominal_estimated_lag_history",
        "applied_lag_history",
    }
    integer_fields = {
        "domain_seed",
        "selection_switches",
        "rejected_trials",
        "constraint_violation",
        "all_updates_succeeded",
        "finite_result",
    }
    rows: List[Dict[str, object]] = []
    for raw in raw_rows:
        row: Dict[str, object] = {}
        for key, value in raw.items():
            if key in text_fields:
                row[key] = value
            elif key in integer_fields:
                row[key] = int(value)
            else:
                row[key] = float(value)
        rows.append(row)
    return rows


def _run_grid(
    manifests: Sequence[Dict[str, object]],
    settings: BenchmarkSettings,
    stage_label: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scenario in GENERALIZATION_SCENARIOS:
        for manifest in manifests:
            family = str(manifest["trajectory_family"])
            reference = make_trajectory_family(
                family,
                samples=settings.samples,
                duration=settings.duration,
            )
            basis = cubic_bspline_basis(
                settings.samples,
                settings.control_points,
            )
            specification = specification_from_manifest(reference, manifest)
            for domain_seed in settings.domain_seeds:
                for method in GENERALIZATION_METHODS:
                    result = run_delay_generalization_method(
                        method,
                        reference,
                        basis,
                        specification,
                        int(domain_seed),
                        settings,
                        scenario,
                    )
                    result["manifest_id"] = str(manifest["manifest_id"])
                    result["trajectory"] = family
                    result["regime"] = str(manifest["regime"])
                    rows.append(
                        {column: result[column] for column in ROW_COLUMNS}
                    )
        print(
            "[V12 "
            + stage_label
            + "] completed "
            + scenario.scenario_id
            + " ("
            + str(len(rows))
            + " method runs)",
            flush=True,
        )
    return rows


def _paired_values(
    rows: Sequence[Dict[str, object]],
    scenario_id: str,
    proposed: str,
    comparator: str,
    regime: Optional[str] = "demand_conflict",
) -> Tuple[np.ndarray, np.ndarray]:
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
    proposed_values = np.asarray(
        [float(indexed[key + (proposed,)]["task_auc_normalized"]) for key in keys]
    )
    comparator_values = np.asarray(
        [float(indexed[key + (comparator,)]["task_auc_normalized"]) for key in keys]
    )
    return proposed_values, comparator_values


def _simple_paired_summary(
    rows: Sequence[Dict[str, object]],
    scenario_id: str,
    proposed: str,
    comparator: str,
) -> Dict[str, object]:
    proposed_values, comparator_values = _paired_values(
        rows,
        scenario_id,
        proposed,
        comparator,
    )
    improvement = 100.0 * (comparator_values - proposed_values) / comparator_values
    return {
        "scenario_id": scenario_id,
        "proposed": proposed,
        "comparator": comparator,
        "paired_cases": int(improvement.size),
        "median_task_auc_improvement_percent": float(np.median(improvement)),
        "mean_task_auc_improvement_percent": float(np.mean(improvement)),
        "strict_win_rate": float(np.mean(proposed_values < comparator_values)),
        "tie_rate": float(np.mean(np.isclose(proposed_values, comparator_values))),
    }


def _select_best_fixed(rows: Sequence[Dict[str, object]]) -> str:
    candidates = ["fixed_delay_" + str(item) for item in FIXED_DELAY_CANDIDATES]
    medians = {
        method: float(
            np.median(
                [
                    row["task_auc_normalized"]
                    for row in rows
                    if row["method"] == method
                    and row["regime"] == "demand_conflict"
                ]
            )
        )
        for method in candidates
    }
    return min(candidates, key=lambda method: (medians[method], method))


def _oracle_excess_summary(
    rows: Sequence[Dict[str, object]],
    scenario_id: str,
) -> Dict[str, object]:
    online, oracle = _paired_values(
        rows,
        scenario_id,
        PRIMARY_METHOD,
        ORACLE_METHOD,
    )
    excess = 100.0 * (online - oracle) / oracle
    return {
        "scenario_id": scenario_id,
        "paired_cases": int(excess.size),
        "median_online_excess_auc_over_oracle_percent": float(np.median(excess)),
        "mean_online_excess_auc_over_oracle_percent": float(np.mean(excess)),
        "online_no_worse_than_oracle_rate": float(np.mean(online <= oracle)),
    }


def _development_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    best_fixed = _select_best_fixed(rows)
    comparisons = {
        scenario.scenario_id: _simple_paired_summary(
            rows,
            scenario.scenario_id,
            PRIMARY_METHOD,
            best_fixed,
        )
        for scenario in GENERALIZATION_SCENARIOS
    }
    oracle_gaps = {
        scenario.scenario_id: _oracle_excess_summary(rows, scenario.scenario_id)
        for scenario in GENERALIZATION_SCENARIOS
    }
    primary_rows = [row for row in rows if row["method"] == PRIMARY_METHOD]
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
    lag_mae = float(
        np.median([row["lag_absolute_error_steps"] for row in primary_rows])
    )
    worst_oracle_excess = float(
        max(
            item["median_online_excess_auc_over_oracle_percent"]
            for item in oracle_gaps.values()
        )
    )
    criteria = {
        "each_scenario_median_improvement_vs_best_fixed_above_zero": all(
            item["median_task_auc_improvement_percent"]
            > DEVELOPMENT_CRITERIA[
                "each_scenario_median_improvement_vs_best_fixed_above_percent"
            ]
            for item in comparisons.values()
        ),
        "each_scenario_win_rate_vs_best_fixed_at_least_60_percent": all(
            item["strict_win_rate"]
            >= DEVELOPMENT_CRITERIA[
                "each_scenario_win_rate_vs_best_fixed_at_least"
            ]
            for item in comparisons.values()
        ),
        "median_axis_lag_error_at_most_one_step": lag_mae
        <= DEVELOPMENT_CRITERIA[
            "median_axis_lag_absolute_error_at_most_steps"
        ],
        "worst_median_oracle_excess_at_most_ten_percent": worst_oracle_excess
        <= DEVELOPMENT_CRITERIA[
            "median_excess_auc_over_oracle_at_most_percent"
        ],
        "success_rate_at_least_95_percent": success_rate
        >= DEVELOPMENT_CRITERIA[
            "solver_and_constraint_success_rate_at_least"
        ],
    }
    fixed_medians = {
        "fixed_delay_" + str(item): float(
            np.median(
                [
                    row["task_auc_normalized"]
                    for row in rows
                    if row["method"] == "fixed_delay_" + str(item)
                    and row["regime"] == "demand_conflict"
                ]
            )
        )
        for item in FIXED_DELAY_CANDIDATES
    }
    return {
        "stage": "v12_unknown_delay_development_only",
        "development_domain_seeds": list(DEVELOPMENT_DOMAIN_SEEDS),
        "scenarios": [scenario_to_dict(item) for item in GENERALIZATION_SCENARIOS],
        "methods": list(GENERALIZATION_METHODS),
        "fixed_candidate_median_auc": fixed_medians,
        "fixed_selection_rule": (
            "lowest pooled median conflict-task AUC across both development scenarios"
        ),
        "selected_best_fixed_method": best_fixed,
        "comparisons": comparisons,
        "oracle_gap_diagnostics": oracle_gaps,
        "primary_success_rate": success_rate,
        "median_axis_lag_absolute_error_steps": lag_mae,
        "screening_thresholds": DEVELOPMENT_CRITERIA,
        "criteria": criteria,
        "passed": bool(all(criteria.values())),
        "decision": "FREEZE_AND_CONFIRM" if all(criteria.values()) else "STOP_AND_REPORT",
        "total_method_runs": len(rows),
    }


def run_delay_generalization_development(
    taskset_path: Path,
    output_directory: Path,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    _validate_settings(settings, DEVELOPMENT_DOMAIN_SEEDS)
    taskset = validate_frozen_taskset(taskset_path)
    manifests = [
        item for item in taskset["manifests"] if item["regime"] == "demand_conflict"
    ]
    rows = _run_grid(manifests, settings, "development")
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "delay_generalization_development_raw.csv", rows)
    payload = _development_summary(rows)
    (output_directory / "delay_generalization_development_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
) -> List[float]:
    indices = rng.integers(0, values.size, size=(20000, values.size))
    samples = np.median(values[indices], axis=1)
    return [float(item) for item in np.percentile(samples, (2.5, 97.5))]


def _paired_statistics(
    rows: Sequence[Dict[str, object]],
    scenario_id: str,
    proposed: str,
    comparator: str,
    rng: np.random.Generator,
) -> Dict[str, object]:
    proposed_values, comparator_values = _paired_values(
        rows,
        scenario_id,
        proposed,
        comparator,
    )
    improvement = 100.0 * (comparator_values - proposed_values) / comparator_values
    p_value = (
        1.0
        if np.allclose(proposed_values, comparator_values)
        else float(
            wilcoxon(
                proposed_values,
                comparator_values,
                alternative="less",
            ).pvalue
        )
    )
    return {
        "scenario_id": scenario_id,
        "scope": "demand_conflict",
        "proposed": proposed,
        "comparator": comparator,
        "paired_cases": int(improvement.size),
        "median_task_auc_improvement_percent": float(np.median(improvement)),
        "mean_task_auc_improvement_percent": float(np.mean(improvement)),
        "paired_win_rate": float(np.mean(proposed_values < comparator_values)),
        "bootstrap_median_improvement_95ci_percent": _bootstrap_interval(
            improvement,
            rng,
        ),
        "one_sided_wilcoxon_p_proposed_lower": p_value,
    }


def _absolute_summary(
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for scenario in GENERALIZATION_SCENARIOS:
        scoped = [
            row
            for row in rows
            if row["scenario_id"] == scenario.scenario_id
            and row["regime"] == "demand_conflict"
        ]
        for method in GENERALIZATION_METHODS:
            selected = [row for row in scoped if row["method"] == method]
            output.append(
                {
                    "scenario_id": scenario.scenario_id,
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
                    "median_rejected_trials": float(
                        np.median([row["rejected_trials"] for row in selected])
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
    return output


def _delay_diagnostics(
    rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    online_rows = [row for row in rows if row["method"] == PRIMARY_METHOD]
    unique: Dict[Tuple[str, int], Dict[str, object]] = {}
    for row in online_rows:
        key = (str(row["scenario_id"]), int(row["domain_seed"]))
        unique.setdefault(key, row)
    by_scenario: Dict[str, Dict[str, object]] = {}
    for scenario in GENERALIZATION_SCENARIOS:
        selected = [
            row
            for (scenario_id, _), row in unique.items()
            if scenario_id == scenario.scenario_id
        ]
        actual_flat: List[float] = []
        estimated_flat: List[float] = []
        applied_flat: List[float] = []
        actual_by_trial: List[List[float]] = [list() for _ in range(5)]
        estimated_by_trial: List[List[float]] = [list() for _ in range(5)]
        applied_by_trial: List[List[float]] = [list() for _ in range(5)]
        for row in selected:
            actual = json.loads(str(row["actual_axis_delay_history"]))
            estimated = json.loads(str(row["raw_estimated_lag_history"]))
            applied = json.loads(str(row["applied_lag_history"]))
            for trial, (actual_axis, estimated_axis, applied_axis) in enumerate(
                zip(actual, estimated, applied)
            ):
                actual_flat.extend(float(item) for item in actual_axis)
                estimated_flat.extend(float(item) for item in estimated_axis)
                applied_flat.extend(float(item) for item in applied_axis)
                actual_by_trial[trial].extend(float(item) for item in actual_axis)
                estimated_by_trial[trial].extend(float(item) for item in estimated_axis)
                applied_by_trial[trial].extend(float(item) for item in applied_axis)
        by_scenario[scenario.scenario_id] = {
            "unique_domain_schedules": len(selected),
            "actual_axis_delay_range_steps": [
                float(np.min(actual_flat)),
                float(np.max(actual_flat)),
            ],
            "estimated_residual_lag_range_steps": [
                float(np.min(estimated_flat)),
                float(np.max(estimated_flat)),
            ],
            "online_applied_lag_range_steps": [
                float(np.min(applied_flat)),
                float(np.max(applied_flat)),
            ],
            "median_actual_axis_delay_steps": float(np.median(actual_flat)),
            "median_estimated_residual_lag_steps": float(
                np.median(estimated_flat)
            ),
            "median_online_applied_lag_steps": float(np.median(applied_flat)),
            "median_by_trial": {
                "actual_axis_delay_steps": [
                    float(np.median(item)) for item in actual_by_trial
                ],
                "estimated_residual_lag_steps": [
                    float(np.median(item)) for item in estimated_by_trial
                ],
                "online_applied_lag_steps": [
                    float(np.median(item)) for item in applied_by_trial
                ],
            },
        }
    return by_scenario


def _plot_development_diagnosis(
    output_path: Path,
    comparisons: Dict[str, Dict[str, Dict[str, object]]],
    absolute: Sequence[Dict[str, object]],
    delay_diagnostics: Dict[str, Dict[str, object]],
    best_fixed: str,
) -> None:
    scenario_ids = [item.scenario_id for item in GENERALIZATION_SCENARIOS]
    scenario_labels = ["Unknown static", "Slow drift"]
    method_labels = {
        BASELINE_METHOD: "Original",
        "fixed_delay_2": "Fixed +2",
        "fixed_delay_4": "Fixed +4",
        "fixed_delay_6": "Fixed +6",
        PRIMARY_METHOD: "Online 0.25",
        ORACLE_METHOD: "True-delay oracle",
    }
    colors = {
        BASELINE_METHOD: "#8da0aa",
        "fixed_delay_2": "#d7a640",
        "fixed_delay_4": "#c68f3b",
        "fixed_delay_6": "#ad7732",
        PRIMARY_METHOD: "#276c8e",
        ORACLE_METHOD: "#6f8f63",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.9))

    x = np.arange(len(scenario_ids))
    width = 0.34
    for offset, method, label in (
        (-0.17, PRIMARY_METHOD, "Online vs best fixed"),
        (0.17, ORACLE_METHOD, "Oracle vs best fixed"),
    ):
        stats = [comparisons[item][method] for item in scenario_ids]
        medians = [item["median_task_auc_improvement_percent"] for item in stats]
        intervals = [item["bootstrap_median_improvement_95ci_percent"] for item in stats]
        lower = [median - interval[0] for median, interval in zip(medians, intervals)]
        upper = [interval[1] - median for median, interval in zip(medians, intervals)]
        axes[0, 0].bar(
            x + offset,
            medians,
            width,
            label=label,
            color=colors[method],
        )
        axes[0, 0].errorbar(
            x + offset,
            medians,
            yerr=[lower, upper],
            fmt="none",
            color="#333333",
            capsize=3,
        )
    axes[0, 0].axhline(0.0, color="#666666", linewidth=1)
    axes[0, 0].set_xticks(x, scenario_labels)
    axes[0, 0].set_ylabel("Median paired AUC improvement (%)")
    axes[0, 0].set_title("Development comparison with " + method_labels[best_fixed])
    axes[0, 0].legend(frameon=False, fontsize=8)

    bar_width = 0.13
    for index, method in enumerate(GENERALIZATION_METHODS):
        values = [
            next(
                row["median_task_auc"]
                for row in absolute
                if row["scenario_id"] == scenario_id and row["method"] == method
            )
            for scenario_id in scenario_ids
        ]
        axes[0, 1].bar(
            x + (index - 2.5) * bar_width,
            values,
            bar_width,
            label=method_labels[method],
            color=colors[method],
        )
    axes[0, 1].set_xticks(x, scenario_labels)
    axes[0, 1].set_ylabel("Median normalized task AUC")
    axes[0, 1].set_title("Absolute conflict-task performance")
    axes[0, 1].legend(frameon=False, fontsize=7, ncol=2)

    for scenario_id, label, marker, color in (
        (scenario_ids[0], "Unknown static", "o", "#276c8e"),
        (scenario_ids[1], "Slow drift", "s", "#b96d5b"),
    ):
        diagnostic = delay_diagnostics[scenario_id]
        actual = diagnostic["median_by_trial"]["actual_axis_delay_steps"]
        estimated = diagnostic["median_by_trial"]["estimated_residual_lag_steps"]
        axes[1, 0].plot(
            range(5),
            actual,
            marker=marker,
            color=color,
            label=label + " actual",
        )
        axes[1, 0].plot(
            range(5),
            estimated,
            marker=marker,
            linestyle="--",
            color=color,
            alpha=0.75,
            label=label + " estimated",
        )
    axes[1, 0].set_xticks(range(5))
    axes[1, 0].set_xlabel("Trial")
    axes[1, 0].set_ylabel("Axis delay / estimated lag (steps)")
    axes[1, 0].set_title("Estimator follows physical delay")
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)

    for scenario_id, label, marker, color in (
        (scenario_ids[0], "Unknown static", "o", "#276c8e"),
        (scenario_ids[1], "Slow drift", "s", "#b96d5b"),
    ):
        diagnostic = delay_diagnostics[scenario_id]
        actual = diagnostic["median_by_trial"]["actual_axis_delay_steps"]
        applied = diagnostic["median_by_trial"]["online_applied_lag_steps"]
        axes[1, 1].plot(
            range(5),
            applied,
            marker=marker,
            color=color,
            label=label + " online",
        )
        axes[1, 1].plot(
            range(5),
            [ONLINE_COMPENSATION_GAIN * item for item in actual],
            linestyle=":",
            color=color,
            label=label + " oracle target",
        )
    axes[1, 1].axhline(2.0, color="#d7a640", linewidth=1.5, label="Fixed +2")
    axes[1, 1].set_xticks(range(5))
    axes[1, 1].set_xlabel("Trial")
    axes[1, 1].set_ylabel("Applied sensitivity shift (steps)")
    axes[1, 1].set_title("0.25 shrinkage collapses delays near +2")
    axes[1, 1].legend(frameon=False, fontsize=7, ncol=2)

    fig.suptitle("V12 development gate failed — formal seeds remain untouched", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def analyze_delay_generalization_development(
    output_directory: Path,
) -> Dict[str, object]:
    rows = _numeric_rows(
        output_directory / "delay_generalization_development_raw.csv"
    )
    development = json.loads(
        (output_directory / "delay_generalization_development_summary.json").read_text(
            encoding="utf-8"
        )
    )
    best_fixed = str(development["selected_best_fixed_method"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons: Dict[str, Dict[str, Dict[str, object]]] = {}
    for scenario in GENERALIZATION_SCENARIOS:
        comparisons[scenario.scenario_id] = {
            PRIMARY_METHOD: _paired_statistics(
                rows,
                scenario.scenario_id,
                PRIMARY_METHOD,
                best_fixed,
                rng,
            ),
            ORACLE_METHOD: _paired_statistics(
                rows,
                scenario.scenario_id,
                ORACLE_METHOD,
                best_fixed,
                rng,
            ),
            BASELINE_METHOD: _paired_statistics(
                rows,
                scenario.scenario_id,
                PRIMARY_METHOD,
                BASELINE_METHOD,
                rng,
            ),
        }
    absolute = _absolute_summary(rows)
    delay_diagnostics = _delay_diagnostics(rows)
    payload = {
        "classification": "DEVELOPMENT_GATE_FAILED_NO_FORMAL_RUN",
        "formal_domain_seeds_used": False,
        "selected_best_fixed_method": best_fixed,
        "comparisons": comparisons,
        "absolute_summary": absolute,
        "delay_diagnostics": delay_diagnostics,
        "development_criteria": development["criteria"],
        "development_passed": False,
        "total_method_runs": len(rows),
    }
    (output_directory / "comparison_v12_development.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _plot_development_diagnosis(
        output_directory / "delay_generalization_development_summary.png",
        comparisons,
        absolute,
        delay_diagnostics,
        best_fixed,
    )

    static_id = GENERALIZATION_SCENARIOS[0].scenario_id
    drift_id = GENERALIZATION_SCENARIOS[1].scenario_id
    static_online = comparisons[static_id][PRIMARY_METHOD]
    drift_online = comparisons[drift_id][PRIMARY_METHOD]
    static_oracle = comparisons[static_id][ORACLE_METHOD]
    drift_oracle = comparisons[drift_id][ORACLE_METHOD]
    static_absolute = next(
        item
        for item in absolute
        if item["scenario_id"] == static_id and item["method"] == PRIMARY_METHOD
    )
    drift_absolute = next(
        item
        for item in absolute
        if item["scenario_id"] == drift_id and item["method"] == PRIMARY_METHOD
    )
    report = f"""# V12 未知与漂移时延开发实验诊断

## 决策

分类：**DEVELOPMENT_GATE_FAILED_NO_FORMAL_RUN**。隔离开发门槛未通过，因此没有冻结 V12 协议，也没有使用预留的正式种子 {list(FORMAL_DOMAIN_SEEDS)}。

## 主要结果

开发集从固定 +2、+4、+6 步补偿中选择了 **{best_fixed}**，选择规则是在两个场景的冲突任务上取得最低合并中位归一化 AUC。

| 场景 | 在线方法相对最佳固定补偿的中位 AUC 改善 | 探索性 95% bootstrap CI | 胜率 | 在线绝对 AUC | 最终任务比 |
|---|---:|---:|---:|---:|---:|
| 未知静态 0–8 步 | {static_online['median_task_auc_improvement_percent']:.3f}% | [{static_online['bootstrap_median_improvement_95ci_percent'][0]:.3f}, {static_online['bootstrap_median_improvement_95ci_percent'][1]:.3f}] | {static_online['paired_win_rate']:.1%} | {static_absolute['median_task_auc']:.3f} | {static_absolute['median_final_task_ratio']:.3f} |
| 跨试次缓慢漂移 ±2 步 | {drift_online['median_task_auc_improvement_percent']:.3f}% | [{drift_online['bootstrap_median_improvement_95ci_percent'][0]:.3f}, {drift_online['bootstrap_median_improvement_95ci_percent'][1]:.3f}] | {drift_online['paired_win_rate']:.1%} | {drift_absolute['median_task_auc']:.3f} | {drift_absolute['median_final_task_ratio']:.3f} |

真实时延 Oracle 相对最佳固定补偿在静态与漂移场景的中位改善分别为 {static_oracle['median_task_auc_improvement_percent']:.3f}% 和 {drift_oracle['median_task_auc_improvement_percent']:.3f}%。在线估计的轴向时延中位绝对误差为 {development['median_axis_lag_absolute_error_steps']:.3f} 步，全部在线运行的求解与约束成功率为 {development['primary_success_rate']:.1%}。

## 机理解释

失败不是由明显的时延识别误差或数值不稳定造成。在线方法与真实时延 Oracle 的 AUC 差距不足 1%，但两者都没有稳定超过固定 +2。V11 的 0.25 收缩把开发域中较宽的物理轴时延压缩为接近 2 个采样点的灵敏度平移；在仅 12 个控制点的宽 B 样条参数化下，固定 +2 已覆盖大部分有效相位修正。漂移只有五个试次，累计中位数估计又会抑制变化，因此在线适应性没有转化为显著任务收益。

## 可以与不可以得出的结论

- 可以确认：估计器数值稳定、接近真实时延 Oracle，并保持 100% 约束安全。
- 不能确认：在当前收缩率、基函数宽度和漂移速度下，在线估计优于开发集选出的固定补偿。
- 这项负结果说明 V11 的收益可能主要来自保守相位提前，而不是在线适应本身；论文若强调在线辨识的必要性，还需要新的机制设计和独立确认。

## 后续建议

不要使用 V12 正式种子继续调参。若继续该方向，应建立由相关峰置信度或跨试次预测误差控制的自适应增益，并使用更窄基函数或更长试次序列，使不同真实时延产生可分辨的控制作用；修改完成后重新分配开发与正式种子。
"""
    (output_directory / "delay_generalization_diagnosis_zh.md").write_text(
        report,
        encoding="utf-8",
    )
    return payload
