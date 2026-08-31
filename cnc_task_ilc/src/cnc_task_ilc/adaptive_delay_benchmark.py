"""V13 one-shot rolling confidence-adaptive delay experiment."""

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

from .adaptive_delay_runner import (
    ADAPTIVE_METHODS,
    BALANCED_DELAY_SCENARIOS,
    MAX_ADAPTIVE_GAIN,
    MAX_APPLIED_LAG_STEPS,
    MIN_ADAPTIVE_GAIN,
    ROLLING_WINDOW,
    BalancedDelayScenario,
    run_adaptive_delay_method,
    scenario_to_dict,
)
from .basis import cubic_bspline_basis
from .benchmark import BenchmarkSettings
from .conflict_benchmark import (
    EXPECTED_TASKSET_SHA256,
    taskset_sha256,
    validate_frozen_taskset,
)
from .conflict_taskset import specification_from_manifest
from .trajectory import make_trajectory_family


DEVELOPMENT_DOMAIN_SEEDS = (1901, 1931, 1951, 1973)
FORMAL_DOMAIN_SEEDS = (2027, 2053, 2081, 2099)
PRIMARY_METHOD = "adaptive_rolling_delay"
FIXED_COMPARATOR = "fixed_delay_2"
V11_COMPARATOR = "delay_aware_dual_anchor"
ORACLE_METHOD = "oracle_adaptive_delay"
BASELINE_METHOD = "dual_anchor_dynamic"
BOOTSTRAP_SEED = 20260827
FORMAL_SETTINGS = {
    "samples": 161,
    "duration_s": 6.0,
    "control_points": 12,
    "iterations": 4,
    "active_zone_budget": 2,
    "half_width": 5,
}
DEVELOPMENT_CRITERIA = {
    "each_scenario_median_improvement_vs_fixed2_above_percent": 0.0,
    "each_scenario_win_rate_vs_fixed2_at_least": 0.60,
    "slow_and_switch_median_improvement_vs_v11_above_percent": 0.0,
    "slow_and_switch_win_rate_vs_v11_at_least": 0.60,
    "static_median_improvement_vs_v11_above_percent": -1.0,
    "median_axis_lag_absolute_error_at_most_steps": 1.5,
    "median_excess_auc_over_oracle_at_most_percent": 10.0,
    "solver_and_constraint_success_rate_at_least": 0.95,
}

ROW_COLUMNS = [
    "manifest_id",
    "trajectory",
    "regime",
    "scenario_id",
    "scenario_mode",
    "domain_seed",
    "schedule_slot",
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
    "actual_axis_delay_history",
    "raw_estimated_lag_history",
    "total_estimated_lag_history",
    "nominal_estimated_lag_history",
    "applied_lag_history",
    "confidence_history",
    "gain_history",
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
        raise RuntimeError("V13 seeds differ from the declared stage")
    if observed != FORMAL_SETTINGS:
        raise RuntimeError("V13 settings differ from the declared design")


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
        "actual_axis_delay_history",
        "raw_estimated_lag_history",
        "total_estimated_lag_history",
        "nominal_estimated_lag_history",
        "applied_lag_history",
        "confidence_history",
        "gain_history",
    }
    integer_fields = {
        "domain_seed",
        "schedule_slot",
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
    for scenario in BALANCED_DELAY_SCENARIOS:
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
            for schedule_slot, domain_seed in enumerate(settings.domain_seeds):
                for method in ADAPTIVE_METHODS:
                    result = run_adaptive_delay_method(
                        method,
                        reference,
                        basis,
                        specification,
                        int(domain_seed),
                        schedule_slot,
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
            "[V13 "
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


def _paired_summary(
    rows: Sequence[Dict[str, object]],
    scenario_id: str,
    comparator: str,
) -> Dict[str, object]:
    proposed, baseline = _paired_values(
        rows,
        scenario_id,
        PRIMARY_METHOD,
        comparator,
    )
    improvement = 100.0 * (baseline - proposed) / baseline
    return {
        "scenario_id": scenario_id,
        "proposed": PRIMARY_METHOD,
        "comparator": comparator,
        "paired_cases": int(improvement.size),
        "median_task_auc_improvement_percent": float(np.median(improvement)),
        "mean_task_auc_improvement_percent": float(np.mean(improvement)),
        "strict_win_rate": float(np.mean(proposed < baseline)),
        "tie_rate": float(np.mean(np.isclose(proposed, baseline))),
    }


def _oracle_excess(
    rows: Sequence[Dict[str, object]],
    scenario_id: str,
) -> Dict[str, object]:
    proposed, oracle = _paired_values(
        rows,
        scenario_id,
        PRIMARY_METHOD,
        ORACLE_METHOD,
    )
    excess = 100.0 * (proposed - oracle) / oracle
    return {
        "scenario_id": scenario_id,
        "paired_cases": int(excess.size),
        "median_online_excess_auc_over_oracle_percent": float(np.median(excess)),
        "mean_online_excess_auc_over_oracle_percent": float(np.mean(excess)),
        "online_no_worse_than_oracle_rate": float(np.mean(proposed <= oracle)),
    }


def _development_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    vs_fixed = {
        scenario.scenario_id: _paired_summary(
            rows,
            scenario.scenario_id,
            FIXED_COMPARATOR,
        )
        for scenario in BALANCED_DELAY_SCENARIOS
    }
    vs_v11 = {
        scenario.scenario_id: _paired_summary(
            rows,
            scenario.scenario_id,
            V11_COMPARATOR,
        )
        for scenario in BALANCED_DELAY_SCENARIOS
    }
    oracle_gaps = {
        scenario.scenario_id: _oracle_excess(rows, scenario.scenario_id)
        for scenario in BALANCED_DELAY_SCENARIOS
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
    drift_ids = ("balanced_slow_drift", "balanced_switch")
    criteria = {
        "each_scenario_median_improvement_vs_fixed2_above_zero": all(
            item["median_task_auc_improvement_percent"]
            > DEVELOPMENT_CRITERIA[
                "each_scenario_median_improvement_vs_fixed2_above_percent"
            ]
            for item in vs_fixed.values()
        ),
        "each_scenario_win_rate_vs_fixed2_at_least_60_percent": all(
            item["strict_win_rate"]
            >= DEVELOPMENT_CRITERIA[
                "each_scenario_win_rate_vs_fixed2_at_least"
            ]
            for item in vs_fixed.values()
        ),
        "slow_and_switch_median_improvement_vs_v11_above_zero": all(
            vs_v11[item]["median_task_auc_improvement_percent"]
            > DEVELOPMENT_CRITERIA[
                "slow_and_switch_median_improvement_vs_v11_above_percent"
            ]
            for item in drift_ids
        ),
        "slow_and_switch_win_rate_vs_v11_at_least_60_percent": all(
            vs_v11[item]["strict_win_rate"]
            >= DEVELOPMENT_CRITERIA[
                "slow_and_switch_win_rate_vs_v11_at_least"
            ]
            for item in drift_ids
        ),
        "static_noninferior_to_v11_within_one_percent": (
            vs_v11["balanced_static"]["median_task_auc_improvement_percent"]
            > DEVELOPMENT_CRITERIA[
                "static_median_improvement_vs_v11_above_percent"
            ]
        ),
        "median_axis_lag_error_at_most_1p5_steps": lag_mae
        <= DEVELOPMENT_CRITERIA[
            "median_axis_lag_absolute_error_at_most_steps"
        ],
        "each_median_oracle_excess_at_most_ten_percent": all(
            item["median_online_excess_auc_over_oracle_percent"]
            <= DEVELOPMENT_CRITERIA[
                "median_excess_auc_over_oracle_at_most_percent"
            ]
            for item in oracle_gaps.values()
        ),
        "success_rate_at_least_95_percent": success_rate
        >= DEVELOPMENT_CRITERIA[
            "solver_and_constraint_success_rate_at_least"
        ],
    }
    return {
        "stage": "v13_one_shot_adaptive_delay_development",
        "development_domain_seeds": list(DEVELOPMENT_DOMAIN_SEEDS),
        "formal_domain_seeds_reserved": list(FORMAL_DOMAIN_SEEDS),
        "scenarios": [scenario_to_dict(item) for item in BALANCED_DELAY_SCENARIOS],
        "methods": list(ADAPTIVE_METHODS),
        "adaptive_design": {
            "rolling_window_trials": ROLLING_WINDOW,
            "minimum_gain": MIN_ADAPTIVE_GAIN,
            "maximum_gain": MAX_ADAPTIVE_GAIN,
            "maximum_applied_lag_steps": MAX_APPLIED_LAG_STEPS,
            "confidence_sources": [
                "axis velocity-correlation peak",
                "axis correlation peak margin",
            ],
        },
        "comparisons_vs_fixed2": vs_fixed,
        "comparisons_vs_v11": vs_v11,
        "oracle_gap_diagnostics": oracle_gaps,
        "primary_success_rate": success_rate,
        "median_axis_lag_absolute_error_steps": lag_mae,
        "screening_thresholds": DEVELOPMENT_CRITERIA,
        "criteria": criteria,
        "passed": bool(all(criteria.values())),
        "decision": "FREEZE_AND_CONFIRM" if all(criteria.values()) else "STOP_USE_V11",
        "total_method_runs": len(rows),
    }


def run_adaptive_delay_development(
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
    _write_csv(output_directory / "adaptive_delay_development_raw.csv", rows)
    payload = _development_summary(rows)
    (output_directory / "adaptive_delay_development_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
) -> List[float]:
    """Return an exploratory percentile interval for the paired median."""

    indices = rng.integers(0, values.size, size=(20000, values.size))
    medians = np.median(values[indices], axis=1)
    return [float(item) for item in np.percentile(medians, (2.5, 97.5))]


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
    improvement = 100.0 * (
        comparator_values - proposed_values
    ) / comparator_values
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
        "scope": "demand_conflict_development_only",
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
        "inference_status": "exploratory_development_statistics",
    }


def _absolute_summary(
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for scenario in BALANCED_DELAY_SCENARIOS:
        scoped = [
            row
            for row in rows
            if row["scenario_id"] == scenario.scenario_id
            and row["regime"] == "demand_conflict"
        ]
        for method in ADAPTIVE_METHODS:
            selected = [row for row in scoped if row["method"] == method]
            output.append(
                {
                    "scenario_id": scenario.scenario_id,
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


def _adaptive_diagnostics(
    rows: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    by_scenario: Dict[str, Dict[str, object]] = {}
    for scenario in BALANCED_DELAY_SCENARIOS:
        selected = [
            row
            for row in rows
            if row["scenario_id"] == scenario.scenario_id
            and row["method"] == PRIMARY_METHOD
            and row["regime"] == "demand_conflict"
        ]
        actual_by_trial: List[List[float]] = [list() for _ in range(5)]
        estimated_by_trial: List[List[float]] = [list() for _ in range(5)]
        applied_by_trial: List[List[float]] = [list() for _ in range(5)]
        confidence_by_trial: List[List[float]] = [list() for _ in range(5)]
        gain_by_trial: List[List[float]] = [list() for _ in range(5)]
        for row in selected:
            histories = (
                json.loads(str(row["actual_axis_delay_history"])),
                json.loads(str(row["raw_estimated_lag_history"])),
                json.loads(str(row["applied_lag_history"])),
                json.loads(str(row["confidence_history"])),
                json.loads(str(row["gain_history"])),
            )
            for trial, trial_values in enumerate(zip(*histories)):
                containers = (
                    actual_by_trial,
                    estimated_by_trial,
                    applied_by_trial,
                    confidence_by_trial,
                    gain_by_trial,
                )
                for container, axis_values in zip(containers, trial_values):
                    container[trial].extend(float(item) for item in axis_values)

        def median_series(values: Sequence[Sequence[float]]) -> List[float]:
            return [float(np.median(item)) for item in values]

        applied_flat = [item for trial in applied_by_trial for item in trial]
        confidence_flat = [
            item for trial in confidence_by_trial for item in trial
        ]
        gain_flat = [item for trial in gain_by_trial for item in trial]
        by_scenario[scenario.scenario_id] = {
            "runs": len(selected),
            "median_axis_lag_absolute_error_steps": float(
                np.median([row["lag_absolute_error_steps"] for row in selected])
            ),
            "median_peak_correlation": float(
                np.median([row["median_peak_correlation"] for row in selected])
            ),
            "applied_lag_range_steps": [
                float(np.min(applied_flat)),
                float(np.max(applied_flat)),
            ],
            "confidence_range": [
                float(np.min(confidence_flat)),
                float(np.max(confidence_flat)),
            ],
            "gain_range": [
                float(np.min(gain_flat)),
                float(np.max(gain_flat)),
            ],
            "median_by_trial": {
                "actual_axis_delay_steps": median_series(actual_by_trial),
                "estimated_residual_lag_steps": median_series(
                    estimated_by_trial
                ),
                "applied_lag_steps": median_series(applied_by_trial),
                "confidence": median_series(confidence_by_trial),
                "adaptive_gain": median_series(gain_by_trial),
            },
        }
    return by_scenario


def _plot_development_diagnosis(
    output_path: Path,
    comparisons: Dict[str, Dict[str, Dict[str, object]]],
    absolute: Sequence[Dict[str, object]],
    diagnostics: Dict[str, Dict[str, object]],
) -> None:
    scenario_ids = [item.scenario_id for item in BALANCED_DELAY_SCENARIOS]
    scenario_labels = ["Static", "Slow drift", "Switch"]
    method_labels = {
        BASELINE_METHOD: "Original",
        FIXED_COMPARATOR: "Fixed +2",
        V11_COMPARATOR: "V11 cumulative",
        PRIMARY_METHOD: "V13 rolling",
        ORACLE_METHOD: "V13 oracle",
    }
    colors = {
        BASELINE_METHOD: "#9aa6ad",
        FIXED_COMPARATOR: "#d19a32",
        V11_COMPARATOR: "#2d6f8e",
        PRIMARY_METHOD: "#b45345",
        ORACLE_METHOD: "#6c8a58",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0))

    y = np.arange(6)
    forest_rows = []
    for scenario_id, scenario_label in zip(scenario_ids, scenario_labels):
        forest_rows.append(
            (scenario_label + " vs fixed +2", comparisons[scenario_id]["vs_fixed2"])
        )
        forest_rows.append(
            (scenario_label + " vs V11", comparisons[scenario_id]["vs_v11"])
        )
    medians = [item[1]["median_task_auc_improvement_percent"] for item in forest_rows]
    intervals = [
        item[1]["bootstrap_median_improvement_95ci_percent"]
        for item in forest_rows
    ]
    lower = [median - interval[0] for median, interval in zip(medians, intervals)]
    upper = [interval[1] - median for median, interval in zip(medians, intervals)]
    axes[0, 0].errorbar(
        medians,
        y,
        xerr=[lower, upper],
        fmt="o",
        color="#8f3f35",
        ecolor="#5d6670",
        capsize=3,
    )
    axes[0, 0].axvline(0.0, color="#666666", linewidth=1)
    axes[0, 0].set_yticks(y, [item[0] for item in forest_rows])
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Median paired AUC improvement (%)")
    axes[0, 0].set_title("V13 exploratory paired comparisons")

    x = np.arange(len(scenario_ids))
    width = 0.15
    for index, method in enumerate(ADAPTIVE_METHODS):
        values = [
            next(
                item["median_task_auc"]
                for item in absolute
                if item["scenario_id"] == scenario_id
                and item["method"] == method
            )
            for scenario_id in scenario_ids
        ]
        axes[0, 1].bar(
            x + (index - 2) * width,
            values,
            width,
            label=method_labels[method],
            color=colors[method],
        )
    axes[0, 1].set_xticks(x, scenario_labels)
    axes[0, 1].set_ylabel("Median normalized task AUC")
    axes[0, 1].set_title("Absolute development performance")
    axes[0, 1].legend(frameon=False, fontsize=7, ncol=2)

    for scenario_id, label, marker, color in (
        (scenario_ids[0], "Static", "o", "#2d6f8e"),
        (scenario_ids[1], "Slow drift", "s", "#b45345"),
        (scenario_ids[2], "Switch", "^", "#6c8a58"),
    ):
        diagnostic = diagnostics[scenario_id]
        axes[1, 0].plot(
            range(5),
            diagnostic["median_by_trial"]["actual_axis_delay_steps"],
            marker=marker,
            color=color,
            label=label + " actual",
        )
        axes[1, 0].plot(
            range(5),
            diagnostic["median_by_trial"]["estimated_residual_lag_steps"],
            marker=marker,
            linestyle="--",
            alpha=0.75,
            color=color,
            label=label + " estimate",
        )
    axes[1, 0].set_xticks(range(5))
    axes[1, 0].set_xlabel("Trial")
    axes[1, 0].set_ylabel("Delay / residual lag (steps)")
    axes[1, 0].set_title("Estimator remains accurate")
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)

    for scenario_id, label, marker, color in (
        (scenario_ids[0], "Static", "o", "#2d6f8e"),
        (scenario_ids[1], "Slow drift", "s", "#b45345"),
        (scenario_ids[2], "Switch", "^", "#6c8a58"),
    ):
        diagnostic = diagnostics[scenario_id]
        axes[1, 1].plot(
            range(5),
            diagnostic["median_by_trial"]["applied_lag_steps"],
            marker=marker,
            color=color,
            label=label,
        )
    axes[1, 1].axhline(2.0, color="#d19a32", linewidth=1.4, label="Fixed +2")
    axes[1, 1].set_xticks(range(5))
    axes[1, 1].set_xlabel("Trial")
    axes[1, 1].set_ylabel("Applied sensitivity shift (steps)")
    axes[1, 1].set_title("Fast adaptation does not improve ILC repeatability")
    axes[1, 1].legend(frameon=False, fontsize=7, ncol=2)

    fig.suptitle(
        "V13 development failed — final method remains V11; formal seeds unused",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def analyze_adaptive_delay_development(
    output_directory: Path,
) -> Dict[str, object]:
    """Archive the failed V13 screen without touching formal domains."""

    rows = _numeric_rows(
        output_directory / "adaptive_delay_development_raw.csv"
    )
    development = json.loads(
        (
            output_directory / "adaptive_delay_development_summary.json"
        ).read_text(encoding="utf-8")
    )
    if development["decision"] != "STOP_USE_V11":
        raise RuntimeError("V13 analysis is only valid for the stopped screen")
    if set(int(row["domain_seed"]) for row in rows) != set(
        DEVELOPMENT_DOMAIN_SEEDS
    ):
        raise RuntimeError("V13 raw data contain undeclared domain seeds")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons: Dict[str, Dict[str, Dict[str, object]]] = {}
    for scenario in BALANCED_DELAY_SCENARIOS:
        scenario_id = scenario.scenario_id
        comparisons[scenario_id] = {
            "vs_fixed2": _paired_statistics(
                rows, scenario_id, PRIMARY_METHOD, FIXED_COMPARATOR, rng
            ),
            "vs_v11": _paired_statistics(
                rows, scenario_id, PRIMARY_METHOD, V11_COMPARATOR, rng
            ),
            "vs_original": _paired_statistics(
                rows, scenario_id, PRIMARY_METHOD, BASELINE_METHOD, rng
            ),
            "oracle_vs_fixed2": _paired_statistics(
                rows, scenario_id, ORACLE_METHOD, FIXED_COMPARATOR, rng
            ),
            "oracle_vs_v11": _paired_statistics(
                rows, scenario_id, ORACLE_METHOD, V11_COMPARATOR, rng
            ),
        }
    absolute = _absolute_summary(rows)
    diagnostics = _adaptive_diagnostics(rows)
    payload = {
        "classification": "V13_DEVELOPMENT_FAILED_FINAL_METHOD_V11",
        "final_selected_method": V11_COMPARATOR,
        "formal_domain_seeds_reserved": list(FORMAL_DOMAIN_SEEDS),
        "formal_domain_seeds_used": False,
        "statistical_scope": (
            "exploratory development statistics; not formal confirmation"
        ),
        "comparisons": comparisons,
        "absolute_summary": absolute,
        "adaptive_diagnostics": diagnostics,
        "development_criteria": development["criteria"],
        "development_passed": False,
        "total_method_runs": len(rows),
    }
    (output_directory / "comparison_v13_development.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _plot_development_diagnosis(
        output_directory / "adaptive_delay_development_summary.png",
        comparisons,
        absolute,
        diagnostics,
    )

    labels = {
        "balanced_static": "平衡静态",
        "balanced_slow_drift": "缓慢漂移",
        "balanced_switch": "突变切换",
    }
    table_rows = []
    for scenario in BALANCED_DELAY_SCENARIOS:
        scenario_id = scenario.scenario_id
        fixed = comparisons[scenario_id]["vs_fixed2"]
        v11 = comparisons[scenario_id]["vs_v11"]
        table_rows.append(
            "| "
            + labels[scenario_id]
            + f" | {fixed['median_task_auc_improvement_percent']:.3f}% "
            + f"| [{fixed['bootstrap_median_improvement_95ci_percent'][0]:.3f}, "
            + f"{fixed['bootstrap_median_improvement_95ci_percent'][1]:.3f}] "
            + f"| {fixed['paired_win_rate']:.1%} "
            + f"| {v11['median_task_auc_improvement_percent']:.3f}% "
            + f"| [{v11['bootstrap_median_improvement_95ci_percent'][0]:.3f}, "
            + f"{v11['bootstrap_median_improvement_95ci_percent'][1]:.3f}] "
            + f"| {v11['paired_win_rate']:.1%} |"
        )
    report = f"""# V13 滚动置信度自适应时延：一次性针对性改进结论

## 最终决策

分类：**V13_DEVELOPMENT_FAILED_FINAL_METHOD_V11**。V13 没有通过预先声明的开发门槛，按一次性改进约定停止，不冻结、不运行正式确认，最终技术方法保持 **V11 累计中位数在线时延补偿**。预留正式种子 {list(FORMAL_DOMAIN_SEEDS)} 未使用。

## 开发结果

以下 bootstrap 区间和 Wilcoxon 检验均为开发集上的探索性统计，不是正式确认结论。每个场景包含 20 个成对冲突任务。

| 场景 | V13 相对 fixed+2 中位改善 | 探索性 95% CI | 胜率 | V13 相对 V11 中位改善 | 探索性 95% CI | 胜率 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

V13 的轴向时延中位绝对误差为 {development['median_axis_lag_absolute_error_steps']:.3f} 步，求解、有限性和约束成功率为 {development['primary_success_rate']:.1%}。它在静态、慢漂移和突变场景相对 Oracle 的中位 AUC 差距分别为 {development['oracle_gap_diagnostics']['balanced_static']['median_online_excess_auc_over_oracle_percent']:.3f}%、{development['oracle_gap_diagnostics']['balanced_slow_drift']['median_online_excess_auc_over_oracle_percent']:.3f}% 和 {development['oracle_gap_diagnostics']['balanced_switch']['median_online_excess_auc_over_oracle_percent']:.3f}%。

## 为什么改进无效

失败不是估计器失真、求解失败或约束违规导致的：时延 MAE、Oracle 差距与安全性均满足门槛。问题在控制机制。ILC 依赖跨试次可重复性；两试次滚动跟踪与置信度自适应增益会快速响应本次时延，但下一试次的物理时延已经改变，灵敏度平移与已经累积的命令修正发生错配。V11 使用全历史中位数和固定 0.25 收缩，虽然响应更慢，却能把时变时延当作不确定性进行稳健平滑，因此在慢漂移和突变条件下更适合当前四次更新的 ILC。

## 可以得出的论文结论

- V13 证明“估得更快、更接近当前时延”并不自动带来更低的跨试次任务 AUC。
- V11 的优势来自保守、稳健的跨试次相位校正，而不是追踪每一次时延变化。
- V12/V13 应作为负结果、消融实验和适用边界；主方法与正式结论仍使用已通过独立确认的 V11。
- 这里不应继续使用同一开发集调参，也不应消耗 V13 正式种子。下一步应转向论文组织和 LinuxCNC/虚拟物理机床接口验证。
"""
    (output_directory / "adaptive_delay_diagnosis_zh.md").write_text(
        report,
        encoding="utf-8",
    )
    return payload
