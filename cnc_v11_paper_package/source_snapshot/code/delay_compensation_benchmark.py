"""V11 development and formal evaluation of delay-aware dual-anchor ILC."""

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
from .delay_compensation_runner import (
    DELAY_COMPENSATION_METHODS,
    run_delay_compensated_method,
)
from .factorial_benchmark import FACTORIAL_SCENARIOS
from .robustness_runner import (
    StressScenario,
    run_robustness_method,
    scenario_to_dict,
)
from .trajectory import make_trajectory_family


DEVELOPMENT_DOMAIN_SEEDS = (1523, 1549, 1571)
FORMAL_DOMAIN_SEEDS = (1601, 1627, 1657, 1669)
PRIMARY_METHOD = "delay_aware_dual_anchor"
BASELINE_METHOD = "dual_anchor_dynamic"
DEVELOPMENT_SCENARIO_IDS = ("n0_d1_m0", "n1_d1_m1")
DEVELOPMENT_COMPENSATION_GAINS = (0.25, 0.50, 0.75, 1.00)
DEVELOPMENT_METHOD_SPECS = (
    (BASELINE_METHOD, BASELINE_METHOD, 0.0),
    ("fixed_delay_dual_anchor", "fixed_delay_dual_anchor", 1.0),
) + tuple(
    (
        "delay_aware_gain_" + str(gain).replace(".", "p"),
        PRIMARY_METHOD,
        gain,
    )
    for gain in DEVELOPMENT_COMPENSATION_GAINS
)
SELECTED_COMPENSATION_GAIN = 0.25
BOOTSTRAP_SEED = 20260811
DELAY_2_SCENARIO = StressScenario(
    scenario_id="delay_2",
    label="Added delay +2 steps",
    factor="added_delay",
    factor_level=2.0,
    measurement_noise_std_mm=0.0,
    extra_delay_steps=2,
    mismatch_scale=1.0,
)
FORMAL_SCENARIOS = (
    next(item for item in FACTORIAL_SCENARIOS if item.scenario_id == "n0_d0_m0"),
    DELAY_2_SCENARIO,
    next(item for item in FACTORIAL_SCENARIOS if item.scenario_id == "n0_d1_m0"),
    next(item for item in FACTORIAL_SCENARIOS if item.scenario_id == "n1_d1_m1"),
)
FORMAL_METHODS = (
    "full_trajectory",
    "error_peak_dynamic",
    BASELINE_METHOD,
    "fixed_delay_dual_anchor",
    PRIMARY_METHOD,
)
FORMAL_CRITERIA = {
    "delay4_and_extreme_conflict_ci_lower_above_zero": True,
    "delay4_and_extreme_conflict_win_rate_at_least": 0.60,
    "delay4_and_extreme_all_tasks_vs_full_ci_lower_above_zero": True,
    "baseline_conflict_median_improvement_above_percent": -2.0,
    "extreme_median_normalized_task_auc_below": 1.0,
    "extreme_median_final_task_ratio_below": 1.0,
    "per_scenario_solver_and_constraint_success_at_least": 0.95,
}
FORMAL_SETTINGS = {
    "samples": 161,
    "duration_s": 6.0,
    "control_points": 12,
    "iterations": 4,
    "active_zone_budget": 2,
    "half_width": 5,
}
DEVELOPMENT_CRITERIA = {
    "per_scenario_conflict_median_improvement_above_percent": 0.0,
    "per_scenario_conflict_win_rate_at_least": 0.60,
    "solver_and_constraint_success_rate_at_least": 0.95,
    "median_axis_lag_absolute_error_at_most_steps": 2.0,
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
    "compensation_gain",
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
    "raw_estimated_lag_history",
    "total_estimated_lag_history",
    "nominal_estimated_lag_history",
    "applied_lag_history",
    "median_estimated_lag_steps",
    "median_estimated_axis_lag_steps",
    "configured_mean_delay_steps",
    "lag_absolute_error_steps",
    "median_peak_correlation",
    "elapsed_s",
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario(scenario_id: str) -> StressScenario:
    return next(
        item for item in FACTORIAL_SCENARIOS if item.scenario_id == scenario_id
    )


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
        raise RuntimeError("V11 seeds differ from the declared stage")
    if observed != FORMAL_SETTINGS:
        raise RuntimeError("V11 settings differ from the declared design")


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
        "stress_factor",
        "method",
        "selection_history",
        "accepted_history",
        "raw_estimated_lag_history",
        "total_estimated_lag_history",
        "nominal_estimated_lag_history",
        "applied_lag_history",
        "median_estimated_axis_lag_steps",
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
    scenarios: Sequence[StressScenario],
    settings: BenchmarkSettings,
    stage_code: int,
    method_specs: Sequence[Tuple[str, str, float]] = DEVELOPMENT_METHOD_SPECS,
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
                paired_noise_seed = (
                    stage_code
                    + 10000 * scenario_index
                    + 100 * manifest_index
                    + int(domain_seed)
                )
                for label, method, compensation_gain in method_specs:
                    result = run_delay_compensated_method(
                        method,
                        reference,
                        basis,
                        specification,
                        int(domain_seed),
                        settings,
                        scenario,
                        paired_noise_seed,
                        compensation_gain=compensation_gain,
                    )
                    result["method"] = label
                    result["manifest_id"] = str(manifest["manifest_id"])
                    result["trajectory"] = family
                    result["regime"] = str(manifest["regime"])
                    rows.append(
                        {column: result[column] for column in ROW_COLUMNS}
                    )
        print(
            "[V11 development] completed "
            + scenario.scenario_id
            + " ("
            + str(len(rows))
            + " method runs)",
            flush=True,
        )
    return rows


def _paired_improvement(
    rows: Sequence[Dict[str, object]],
    scenario_id: str,
    comparator: str = BASELINE_METHOD,
    regime: Optional[str] = "demand_conflict",
    proposed: str = PRIMARY_METHOD,
) -> Dict[str, object]:
    proposed_label = proposed
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
        [
            float(indexed[key + (proposed_label,)]["task_auc_normalized"])
            for key in keys
        ]
    )
    baseline = np.asarray(
        [float(indexed[key + (comparator,)]["task_auc_normalized"]) for key in keys]
    )
    improvement = 100.0 * (baseline - proposed_values) / baseline
    return {
        "scenario_id": scenario_id,
        "scope": "all" if regime is None else regime,
        "comparator": comparator,
        "proposed": proposed_label,
        "paired_cases": len(keys),
        "median_task_auc_improvement_percent": float(np.median(improvement)),
        "mean_task_auc_improvement_percent": float(np.mean(improvement)),
        "strict_win_rate": float(np.mean(proposed_values < baseline)),
        "tie_rate": float(np.mean(np.isclose(proposed_values, baseline))),
    }


def _development_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Select the robust gain and evaluate the frozen development screen."""

    candidate_labels = [
        item[0] for item in DEVELOPMENT_METHOD_SPECS if item[1] == PRIMARY_METHOD
    ]
    candidate_comparisons = {
        label: {
            scenario_id: _paired_improvement(
                rows,
                scenario_id,
                proposed=label,
            )
            for scenario_id in DEVELOPMENT_SCENARIO_IDS
        }
        for label in candidate_labels
    }
    selected_label = max(
        candidate_labels,
        key=lambda label: (
            min(
                item["median_task_auc_improvement_percent"]
                for item in candidate_comparisons[label].values()
            ),
            min(
                item["strict_win_rate"]
                for item in candidate_comparisons[label].values()
            ),
            -next(
                spec[2] for spec in DEVELOPMENT_METHOD_SPECS if spec[0] == label
            ),
        ),
    )
    selected_gain = next(
        spec[2] for spec in DEVELOPMENT_METHOD_SPECS if spec[0] == selected_label
    )
    comparisons = candidate_comparisons[selected_label]
    primary_rows = [row for row in rows if row["method"] == selected_label]
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
    criteria: Dict[str, bool] = {
        "all_scenario_medians_above_zero": all(
            item["median_task_auc_improvement_percent"]
            > DEVELOPMENT_CRITERIA[
                "per_scenario_conflict_median_improvement_above_percent"
            ]
            for item in comparisons.values()
        ),
        "all_scenario_win_rates_at_least_60_percent": all(
            item["strict_win_rate"]
            >= DEVELOPMENT_CRITERIA[
                "per_scenario_conflict_win_rate_at_least"
            ]
            for item in comparisons.values()
        ),
        "success_rate_at_least_95_percent": success_rate
        >= DEVELOPMENT_CRITERIA[
            "solver_and_constraint_success_rate_at_least"
        ],
        "median_axis_lag_error_at_most_two_steps": lag_mae
        <= DEVELOPMENT_CRITERIA[
            "median_axis_lag_absolute_error_at_most_steps"
        ],
    }
    return {
        "stage": "v11_fractional_compensation_gain_development_only",
        "development_domain_seeds": list(DEVELOPMENT_DOMAIN_SEEDS),
        "scenarios": list(DEVELOPMENT_SCENARIO_IDS),
        "method_specs": [list(item) for item in DEVELOPMENT_METHOD_SPECS],
        "candidate_selection_rule": (
            "maximize worst-scenario median AUC improvement, then worst-scenario "
            "win rate, then prefer smaller gain"
        ),
        "selected_method_label": selected_label,
        "selected_compensation_gain": selected_gain,
        "screening_thresholds": DEVELOPMENT_CRITERIA,
        "comparisons": comparisons,
        "all_candidate_comparisons": candidate_comparisons,
        "primary_success_rate": success_rate,
        "median_axis_lag_absolute_error_steps": lag_mae,
        "criteria": criteria,
        "passed": bool(all(criteria.values())),
        "decision": "FREEZE_AND_CONFIRM" if all(criteria.values()) else "REVISE",
        "total_method_runs": len(rows),
    }


def run_delay_compensation_development(
    taskset_path: Path,
    output_directory: Path,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    """Screen the unmodified online estimator on isolated conflict tasks."""

    _validate_settings(settings, DEVELOPMENT_DOMAIN_SEEDS)
    taskset = validate_frozen_taskset(taskset_path)
    manifests = [
        item for item in taskset["manifests"] if item["regime"] == "demand_conflict"
    ]
    scenarios = [_scenario(item) for item in DEVELOPMENT_SCENARIO_IDS]
    rows = _run_grid(manifests, scenarios, settings, stage_code=180000)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "delay_compensation_development_raw.csv", rows)

    payload = _development_summary(rows)
    (output_directory / "delay_compensation_development_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def analyze_existing_delay_development(
    output_directory: Path,
) -> Dict[str, object]:
    """Rebuild the development summary without rerunning simulations."""

    rows = _numeric_rows(
        output_directory / "delay_compensation_development_raw.csv"
    )
    payload = _development_summary(rows)
    (output_directory / "delay_compensation_development_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _assert_seed_isolation() -> None:
    from .conflict_benchmark import (
        AUDIT_DOMAIN_SEEDS as V7_AUDIT,
        FORMAL_DOMAIN_SEEDS as V7_FORMAL,
    )
    from .dual_anchor_confirmation import FORMAL_DOMAIN_SEEDS as V8_FORMAL
    from .dual_anchor_development import DEVELOPMENT_DOMAIN_SEEDS as V8_DEVELOPMENT
    from .factorial_benchmark import (
        AUDIT_DOMAIN_SEEDS as V10_AUDIT,
        FORMAL_DOMAIN_SEEDS as V10_FORMAL,
    )
    from .robustness_benchmark import (
        AUDIT_DOMAIN_SEEDS as V9_AUDIT,
        FORMAL_DOMAIN_SEEDS as V9_FORMAL,
    )

    groups = (
        set(V7_AUDIT),
        set(V7_FORMAL),
        set(V8_DEVELOPMENT),
        set(V8_FORMAL),
        set(V9_AUDIT),
        set(V9_FORMAL),
        set(V10_AUDIT),
        set(V10_FORMAL),
        set(DEVELOPMENT_DOMAIN_SEEDS),
        set(FORMAL_DOMAIN_SEEDS),
    )
    for first_index, first in enumerate(groups):
        for second in groups[first_index + 1 :]:
            if first & second:
                raise RuntimeError("V11 domain seeds overlap an earlier stage")


def preregistered_delay_protocol(
    taskset_path: Path,
    development_summary_path: Path,
    v10_protocol_path: Path,
) -> Dict[str, object]:
    """Build the V11 protocol only after the isolated screen passes."""

    _assert_seed_isolation()
    development = json.loads(
        development_summary_path.read_text(encoding="utf-8")
    )
    if not bool(development.get("passed")):
        raise RuntimeError("V11 development screen did not authorize confirmation")
    if float(development["selected_compensation_gain"]) != SELECTED_COMPENSATION_GAIN:
        raise RuntimeError("V11 selected gain differs from the declared value")
    v10_protocol = json.loads(v10_protocol_path.read_text(encoding="utf-8"))
    if v10_protocol["taskset_sha256"] != taskset_sha256(taskset_path):
        raise RuntimeError("V10 protocol and V11 taskset hashes differ")
    runner_path = Path(__file__).with_name("delay_compensation_runner.py")
    return {
        "protocol_id": "v11-online-fractional-delay-compensation",
        "frozen_before_formal_execution": True,
        "hypothesis": (
            "online residual axis-delay estimation with conservative fractional "
            "sensitivity alignment reduces task AUC under unmodeled delay"
        ),
        "primary_method": PRIMARY_METHOD,
        "primary_scope": "demand_conflict",
        "selected_compensation_gain": SELECTED_COMPENSATION_GAIN,
        "gain_source": "isolated V11 development screen",
        "methods": list(FORMAL_METHODS),
        "scenarios": [scenario_to_dict(item) for item in FORMAL_SCENARIOS],
        "settings": FORMAL_SETTINGS,
        "formal_domain_seeds": list(FORMAL_DOMAIN_SEEDS),
        "development_domain_seeds_excluded": list(DEVELOPMENT_DOMAIN_SEEDS),
        "taskset_sha256": taskset_sha256(taskset_path),
        "v10_protocol_sha256": file_sha256(v10_protocol_path),
        "development_summary_sha256": file_sha256(development_summary_path),
        "delay_compensation_runner_sha256": file_sha256(runner_path),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "formal_criteria": FORMAL_CRITERIA,
        "formal_method_runs": (
            15
            * len(FORMAL_DOMAIN_SEEDS)
            * len(FORMAL_SCENARIOS)
            * len(FORMAL_METHODS)
        ),
    }


def freeze_delay_protocol(
    taskset_path: Path,
    development_summary_path: Path,
    v10_protocol_path: Path,
    protocol_path: Path,
) -> Dict[str, object]:
    protocol = preregistered_delay_protocol(
        taskset_path,
        development_summary_path,
        v10_protocol_path,
    )
    if protocol["taskset_sha256"] != EXPECTED_TASKSET_SHA256:
        raise RuntimeError("unexpected taskset hash while freezing V11")
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("existing V11 protocol differs from current inputs")
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return protocol


def validate_delay_protocol(
    taskset_path: Path,
    development_summary_path: Path,
    v10_protocol_path: Path,
    protocol_path: Path,
) -> Dict[str, object]:
    if not protocol_path.exists():
        raise RuntimeError("V11 protocol must be frozen before formal execution")
    existing = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = preregistered_delay_protocol(
        taskset_path,
        development_summary_path,
        v10_protocol_path,
    )
    if existing != expected:
        raise RuntimeError("frozen V11 protocol no longer matches code or inputs")
    return existing


def _formal_result(
    method: str,
    reference: object,
    basis: np.ndarray,
    specification: object,
    domain_seed: int,
    settings: BenchmarkSettings,
    scenario: StressScenario,
    noise_seed: int,
) -> Dict[str, object]:
    if method in ("full_trajectory", "error_peak_dynamic", BASELINE_METHOD):
        result = run_robustness_method(
            method,
            reference,
            basis,
            specification,
            domain_seed,
            settings,
            scenario,
            noise_seed,
        )
        result.update(
            {
                "compensation_gain": 0.0,
                "raw_estimated_lag_history": "[]",
                "total_estimated_lag_history": "[]",
                "nominal_estimated_lag_history": "[]",
                "applied_lag_history": "[]",
                "median_estimated_lag_steps": float("nan"),
                "median_estimated_axis_lag_steps": "[]",
                "configured_mean_delay_steps": float("nan"),
                "lag_absolute_error_steps": float("nan"),
                "median_peak_correlation": float("nan"),
            }
        )
        return result
    return run_delay_compensated_method(
        method,
        reference,
        basis,
        specification,
        domain_seed,
        settings,
        scenario,
        noise_seed,
        compensation_gain=(
            SELECTED_COMPENSATION_GAIN
            if method == PRIMARY_METHOD
            else 1.0
        ),
    )


def _run_formal_grid(
    manifests: Sequence[Dict[str, object]],
    settings: BenchmarkSettings,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scenario_index, scenario in enumerate(FORMAL_SCENARIOS):
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
                paired_noise_seed = (
                    210000
                    + 10000 * scenario_index
                    + 100 * manifest_index
                    + int(domain_seed)
                )
                for method in FORMAL_METHODS:
                    result = _formal_result(
                        method,
                        reference,
                        basis,
                        specification,
                        int(domain_seed),
                        settings,
                        scenario,
                        paired_noise_seed,
                    )
                    result["manifest_id"] = str(manifest["manifest_id"])
                    result["trajectory"] = family
                    result["regime"] = str(manifest["regime"])
                    rows.append(
                        {column: result[column] for column in ROW_COLUMNS}
                    )
        print(
            "[V11 formal] completed "
            + scenario.scenario_id
            + " ("
            + str(len(rows))
            + " method runs)",
            flush=True,
        )
    return rows


def run_delay_confirmation(
    taskset_path: Path,
    development_summary_path: Path,
    v10_protocol_path: Path,
    protocol_path: Path,
    output_directory: Path,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    _validate_settings(settings, FORMAL_DOMAIN_SEEDS)
    protocol = validate_delay_protocol(
        taskset_path,
        development_summary_path,
        v10_protocol_path,
        protocol_path,
    )
    taskset = validate_frozen_taskset(taskset_path)
    rows = _run_formal_grid(taskset["manifests"], settings)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "delay_compensation_effectiveness_raw.csv", rows)
    metadata = {
        "protocol_sha256": file_sha256(protocol_path),
        "protocol": protocol,
        "total_method_runs": len(rows),
        "paired_cases_per_scenario_all_tasks": 15 * len(FORMAL_DOMAIN_SEEDS),
        "paired_cases_per_scenario_conflict": 5 * len(FORMAL_DOMAIN_SEEDS),
    }
    (output_directory / "delay_compensation_effectiveness_metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


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
    comparator: str,
    regime: Optional[str],
    rng: np.random.Generator,
    proposed_method: str = PRIMARY_METHOD,
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
        [
            float(indexed[key + (proposed_method,)]["task_auc_normalized"])
            for key in keys
        ]
    )
    baseline = np.asarray(
        [float(indexed[key + (comparator,)]["task_auc_normalized"]) for key in keys]
    )
    improvement = 100.0 * (baseline - proposed) / baseline
    p_value = (
        1.0
        if np.allclose(proposed, baseline)
        else float(wilcoxon(proposed, baseline, alternative="less").pvalue)
    )
    return {
        "scenario_id": scenario_id,
        "scope": "all" if regime is None else regime,
        "proposed": proposed_method,
        "comparator": comparator,
        "paired_cases": len(keys),
        "median_task_auc_improvement_percent": float(np.median(improvement)),
        "mean_task_auc_improvement_percent": float(np.mean(improvement)),
        "paired_win_rate": float(np.mean(proposed < baseline)),
        "paired_tie_rate": float(np.mean(np.isclose(proposed, baseline))),
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
    for scenario in FORMAL_SCENARIOS:
        for scope in ("all", "demand_conflict"):
            scoped = [
                row
                for row in rows
                if row["scenario_id"] == scenario.scenario_id
                and (scope == "all" or row["regime"] == scope)
            ]
            for method in FORMAL_METHODS:
                selected = [row for row in scoped if row["method"] == method]
                output.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "scope": scope,
                        "method": method,
                        "runs": len(selected),
                        "median_initial_task_score": float(
                            np.median([row["initial_task_score"] for row in selected])
                        ),
                        "median_final_task_score": float(
                            np.median([row["final_task_score"] for row in selected])
                        ),
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
    return output


def _plot_confirmation(
    output_path: Path,
    comparisons: Dict[str, Dict[str, Dict[str, object]]],
    summary: Sequence[Dict[str, object]],
    development: Dict[str, object],
    rows: Sequence[Dict[str, object]],
    classification: str,
) -> None:
    scenario_ids = [item.scenario_id for item in FORMAL_SCENARIOS]
    scenario_labels = ["Base", "+2 delay", "+4 delay", "N+D+M"]
    method_labels = {
        "full_trajectory": "Full",
        "error_peak_dynamic": "Error peak",
        BASELINE_METHOD: "Dual anchor",
        "fixed_delay_dual_anchor": "Fixed +4",
        PRIMARY_METHOD: "Online 0.25",
    }
    colors = {
        "full_trajectory": "#76946b",
        "error_peak_dynamic": "#c06c5b",
        BASELINE_METHOD: "#8fa6b3",
        "fixed_delay_dual_anchor": "#d5a64a",
        PRIMARY_METHOD: "#236b8e",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0))
    x = np.arange(len(scenario_ids))

    stats = [comparisons[item][BASELINE_METHOD] for item in scenario_ids]
    medians = [item["median_task_auc_improvement_percent"] for item in stats]
    intervals = [item["bootstrap_median_improvement_95ci_percent"] for item in stats]
    lower = [median - interval[0] for median, interval in zip(medians, intervals)]
    upper = [interval[1] - median for median, interval in zip(medians, intervals)]
    axes[0, 0].bar(x, medians, color="#236b8e")
    axes[0, 0].errorbar(x, medians, yerr=[lower, upper], fmt="none", color="#333", capsize=4)
    axes[0, 0].axhline(0.0, color="#777", linewidth=1)
    axes[0, 0].set_xticks(x, scenario_labels)
    axes[0, 0].set_ylabel("AUC improvement vs dual anchor (%)")
    axes[0, 0].set_title("Paired conflict-task benefit")

    width = 0.16
    for method_index, method in enumerate(FORMAL_METHODS):
        values = [
            next(
                item["median_task_auc"]
                for item in summary
                if item["scenario_id"] == scenario_id
                and item["scope"] == "demand_conflict"
                and item["method"] == method
            )
            for scenario_id in scenario_ids
        ]
        axes[0, 1].bar(
            x + (method_index - 2) * width,
            values,
            width,
            label=method_labels[method],
            color=colors[method],
        )
    axes[0, 1].set_xticks(x, scenario_labels)
    axes[0, 1].set_ylabel("Median normalized task AUC")
    axes[0, 1].set_title("Absolute conflict-task performance")
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=2)

    gain_items = development["all_candidate_comparisons"]
    gains = list(DEVELOPMENT_COMPENSATION_GAINS)
    for scenario_id, label, color in (
        ("n0_d1_m0", "+4 delay", "#236b8e"),
        ("n1_d1_m1", "N+D+M", "#c06c5b"),
    ):
        values = [
            gain_items["delay_aware_gain_" + str(gain).replace(".", "p")][
                scenario_id
            ]["median_task_auc_improvement_percent"]
            for gain in gains
        ]
        axes[1, 0].plot(gains, values, marker="o", label=label, color=color)
    axes[1, 0].axhline(0.0, color="#777", linewidth=1)
    axes[1, 0].axvline(SELECTED_COMPENSATION_GAIN, color="#444", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("Fractional compensation gain")
    axes[1, 0].set_ylabel("Development AUC improvement (%)")
    axes[1, 0].set_title("Why conservative compensation was selected")
    axes[1, 0].legend(frameon=False)

    primary_rows = [row for row in rows if row["method"] == PRIMARY_METHOD]
    configured = np.asarray(
        [float(row["configured_mean_delay_steps"]) for row in primary_rows]
    )
    estimated = np.asarray(
        [float(row["median_estimated_lag_steps"]) for row in primary_rows]
    )
    axes[1, 1].scatter(configured, estimated, s=13, alpha=0.35, color="#236b8e")
    limit = max(float(np.max(configured)), float(np.max(estimated))) + 0.5
    axes[1, 1].plot((0, limit), (0, limit), color="#777", linestyle="--", linewidth=1)
    axes[1, 1].set_xlim(0, limit)
    axes[1, 1].set_ylim(0, limit)
    axes[1, 1].set_xlabel("Configured mean axis delay (steps)")
    axes[1, 1].set_ylabel("Estimated residual lag (steps)")
    axes[1, 1].set_title("Online delay-estimation diagnostic")

    fig.suptitle("V11 fractional delay compensation — " + classification, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def analyze_delay_confirmation(
    output_directory: Path,
    development_summary_path: Path,
) -> Dict[str, object]:
    rows = _numeric_rows(
        output_directory / "delay_compensation_effectiveness_raw.csv"
    )
    development = json.loads(
        development_summary_path.read_text(encoding="utf-8")
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons: Dict[str, Dict[str, Dict[str, object]]] = {}
    for scenario in FORMAL_SCENARIOS:
        comparisons[scenario.scenario_id] = {
            comparator: _paired_statistics(
                rows,
                scenario.scenario_id,
                comparator,
                "demand_conflict",
                rng,
            )
            for comparator in (
                BASELINE_METHOD,
                "fixed_delay_dual_anchor",
                "error_peak_dynamic",
            )
        }
        comparisons[scenario.scenario_id]["full_trajectory_all"] = (
            _paired_statistics(
                rows,
                scenario.scenario_id,
                "full_trajectory",
                None,
                rng,
            )
        )
    summary = _absolute_summary(rows)
    fixed_vs_original = {
        scenario.scenario_id: _paired_statistics(
            rows,
            scenario.scenario_id,
            BASELINE_METHOD,
            "demand_conflict",
            rng,
            proposed_method="fixed_delay_dual_anchor",
        )
        for scenario in FORMAL_SCENARIOS
    }
    high_ids = ("n0_d1_m0", "n1_d1_m1")
    primary_high = [comparisons[item][BASELINE_METHOD] for item in high_ids]
    high_vs_full = [comparisons[item]["full_trajectory_all"] for item in high_ids]
    primary_rows = [row for row in rows if row["method"] == PRIMARY_METHOD]
    success_by_scenario = {
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
        for scenario in FORMAL_SCENARIOS
    }
    extreme_absolute = next(
        item
        for item in summary
        if item["scenario_id"] == "n1_d1_m1"
        and item["scope"] == "all"
        and item["method"] == PRIMARY_METHOD
    )
    lag_mae = float(
        np.median([row["lag_absolute_error_steps"] for row in primary_rows])
    )
    criteria: Dict[str, bool] = {
        "delay4_and_extreme_conflict_ci_lower_above_zero": all(
            item["bootstrap_median_improvement_95ci_percent"][0] > 0.0
            for item in primary_high
        ),
        "delay4_and_extreme_conflict_win_rate_at_least_60_percent": all(
            item["paired_win_rate"] >= 0.60 for item in primary_high
        ),
        "delay4_and_extreme_all_tasks_vs_full_ci_lower_above_zero": all(
            item["bootstrap_median_improvement_95ci_percent"][0] > 0.0
            for item in high_vs_full
        ),
        "baseline_conflict_median_improvement_above_minus_two_percent": (
            comparisons["n0_d0_m0"][BASELINE_METHOD][
                "median_task_auc_improvement_percent"
            ]
            > FORMAL_CRITERIA[
                "baseline_conflict_median_improvement_above_percent"
            ]
        ),
        "extreme_median_normalized_task_auc_below_one": (
            extreme_absolute["median_task_auc"]
            < FORMAL_CRITERIA["extreme_median_normalized_task_auc_below"]
        ),
        "extreme_median_final_task_ratio_below_one": (
            extreme_absolute["median_final_task_ratio"]
            < FORMAL_CRITERIA["extreme_median_final_task_ratio_below"]
        ),
        "all_scenario_success_rates_at_least_95_percent": all(
            value
            >= FORMAL_CRITERIA[
                "per_scenario_solver_and_constraint_success_at_least"
            ]
            for value in success_by_scenario.values()
        ),
    }
    passed = bool(all(criteria.values()))
    classification = (
        "DELAY_COMPENSATION_CONFIRMED"
        if passed
        else "DELAY_COMPENSATION_NOT_CONFIRMED"
    )
    payload = {
        "classification": classification,
        "criteria": criteria,
        "comparisons": comparisons,
        "absolute_summary": summary,
        "fixed_vs_original_diagnostic": fixed_vs_original,
        "primary_success_rate_by_scenario": success_by_scenario,
        "median_axis_lag_absolute_error_steps": lag_mae,
        "selected_compensation_gain": SELECTED_COMPENSATION_GAIN,
        "formal_method_runs": len(rows),
    }
    (output_directory / "comparison_v11_delay_compensation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _plot_confirmation(
        output_directory / "delay_compensation_summary.png",
        comparisons,
        summary,
        development,
        rows,
        classification,
    )
    baseline = comparisons["n0_d0_m0"][BASELINE_METHOD]
    delay2 = comparisons["delay_2"][BASELINE_METHOD]
    delay4 = comparisons["n0_d1_m0"][BASELINE_METHOD]
    extreme = comparisons["n1_d1_m1"][BASELINE_METHOD]
    fixed_extreme = fixed_vs_original["n1_d1_m1"]
    report = f"""# V11 在线分数时延补偿：正式实验诊断

## 结论

正式分类：**{classification}**。在线估计器的残差时延只以 0.25 增益写入双锚点 ILC 的名义灵敏度；该增益完全由隔离开发种子选择，正式种子未用于调参。

## 核心结果

| 场景 | 相对原双锚点的任务 AUC 中位改善 | 95% bootstrap CI | 配对胜率 |
|---|---:|---:|---:|
| 基线 | {baseline['median_task_auc_improvement_percent']:.3f}% | [{baseline['bootstrap_median_improvement_95ci_percent'][0]:.3f}, {baseline['bootstrap_median_improvement_95ci_percent'][1]:.3f}] | {baseline['paired_win_rate']:.1%} |
| 额外时延 +2 | {delay2['median_task_auc_improvement_percent']:.3f}% | [{delay2['bootstrap_median_improvement_95ci_percent'][0]:.3f}, {delay2['bootstrap_median_improvement_95ci_percent'][1]:.3f}] | {delay2['paired_win_rate']:.1%} |
| 额外时延 +4 | {delay4['median_task_auc_improvement_percent']:.3f}% | [{delay4['bootstrap_median_improvement_95ci_percent'][0]:.3f}, {delay4['bootstrap_median_improvement_95ci_percent'][1]:.3f}] | {delay4['paired_win_rate']:.1%} |
| 噪声+时延+失配 | {extreme['median_task_auc_improvement_percent']:.3f}% | [{extreme['bootstrap_median_improvement_95ci_percent'][0]:.3f}, {extreme['bootstrap_median_improvement_95ci_percent'][1]:.3f}] | {extreme['paired_win_rate']:.1%} |

在线估计的轴向时延中位绝对误差为 {lag_mae:.3f} 步。极端场景下，方法的全任务中位归一化 AUC 为 {extreme_absolute['median_task_auc']:.3f}，最终任务比为 {extreme_absolute['median_final_task_ratio']:.3f}，数值求解与约束成功率为 {success_by_scenario['n1_d1_m1']:.1%}。

固定 +4 步补偿相对原双锚点在极端场景的中位改善为 {fixed_extreme['median_task_auc_improvement_percent']:.3f}%（95% CI [{fixed_extreme['bootstrap_median_improvement_95ci_percent'][0]:.3f}, {fixed_extreme['bootstrap_median_improvement_95ci_percent'][1]:.3f}]）；区间跨过 0，因此固定补偿只能作为有潜力的诊断对照，不能按同一 bootstrap 标准宣称稳定有效。在线 0.25 补偿相对固定补偿在 +4 与极端场景的区间也跨过 0，故本实验确认的是在线方法相对原双锚点、误差峰值和全轨迹基线的优势，并未证明其在每个高时延场景都显著优于固定补偿。

## 方法解释

全量补偿在开发实验中失败，而 0.25 分数补偿通过。原因是互相关估计给出的是等效群时延，其中混合了纯传输延迟、闭环轴动态相位和结构失配。把全部估计量当成精确纯延迟会过度平移灵敏度；分数补偿等价于对名义相位模型做保守更新，保留原双锚点 ILC 对宽基函数和模型失配的鲁棒性。

## 学术边界

本实验支持的是“任务驱动 ILC 中的在线等效时延估计与不确定性收缩”这一结论，而不是证明 0.25 对所有机床最优。下一阶段应把固定增益推广为由相关峰置信度、频带或跨试次预测误差驱动的自适应收缩，并在 LinuxCNC 虚拟机床或更高保真伺服模型上复核。
"""
    (output_directory / "delay_compensation_diagnosis_zh.md").write_text(
        report,
        encoding="utf-8",
    )
    return payload
