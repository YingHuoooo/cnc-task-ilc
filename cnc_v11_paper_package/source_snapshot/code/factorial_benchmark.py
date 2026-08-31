"""V10 frozen 2^3 combined-stress factorial experiment."""

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
from .robustness_benchmark import ROW_COLUMNS
from .robustness_runner import (
    ROBUSTNESS_METHODS,
    StressScenario,
    run_robustness_method,
    scenario_to_dict,
)
from .trajectory import make_trajectory_family


AUDIT_DOMAIN_SEEDS = (1409, 1417)
FORMAL_DOMAIN_SEEDS = (1423, 1451, 1471, 1499)
PRIMARY_METHOD = "dual_anchor_dynamic"
PRIMARY_COMPARATOR = "error_peak_dynamic"
BOOTSTRAP_SEED = 20260803
FORMAL_SETTINGS = {
    "samples": 161,
    "duration_s": 6.0,
    "control_points": 12,
    "iterations": 4,
    "active_zone_budget": 2,
    "half_width": 5,
}
FACTOR_LEVELS = {
    "measurement_noise_std_mm": [0.0, 0.05],
    "extra_delay_steps": [0, 4],
    "mismatch_scale": [1.0, 1.70],
}
EXTREME_SCENARIO_ID = "n1_d1_m1"
FORMAL_CRITERIA = {
    "extreme_conflict_vs_error_peak_ci_lower_above_zero": True,
    "extreme_conflict_vs_error_peak_win_rate_at_least": 0.60,
    "extreme_all_tasks_vs_full_ci_lower_above_zero": True,
    "extreme_median_normalized_task_auc_below": 1.0,
    "extreme_median_final_task_ratio_below": 1.0,
    "per_scenario_solver_and_constraint_success_at_least": 0.95,
}


def _factorial_scenario(noise: int, delay: int, mismatch: int) -> StressScenario:
    scenario_id = f"n{noise}_d{delay}_m{mismatch}"
    labels = []
    if noise:
        labels.append("N=0.05 mm")
    if delay:
        labels.append("D=+4")
    if mismatch:
        labels.append("M=1.70x")
    return StressScenario(
        scenario_id=scenario_id,
        label=" + ".join(labels) if labels else "Baseline",
        factor="factorial_combination",
        factor_level=float(noise + delay + mismatch),
        measurement_noise_std_mm=0.05 if noise else 0.0,
        extra_delay_steps=4 if delay else 0,
        mismatch_scale=1.70 if mismatch else 1.0,
    )


FACTORIAL_SCENARIOS = tuple(
    _factorial_scenario(noise, delay, mismatch)
    for noise, delay, mismatch in (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    )
)


def factor_codes(scenario_id: str) -> Tuple[int, int, int]:
    parts = scenario_id.split("_")
    if len(parts) != 3:
        raise ValueError("invalid factorial scenario id: " + scenario_id)
    return tuple(int(part[1:]) for part in parts)  # type: ignore[return-value]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _assert_seed_isolation() -> None:
    from .conflict_benchmark import (
        AUDIT_DOMAIN_SEEDS as V7_AUDIT,
        FORMAL_DOMAIN_SEEDS as V7_FORMAL,
    )
    from .dual_anchor_confirmation import FORMAL_DOMAIN_SEEDS as V8_FORMAL
    from .dual_anchor_development import DEVELOPMENT_DOMAIN_SEEDS as V8_DEVELOPMENT
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
        set(AUDIT_DOMAIN_SEEDS),
        set(FORMAL_DOMAIN_SEEDS),
    )
    for first_index, first in enumerate(groups):
        for second in groups[first_index + 1 :]:
            if first & second:
                raise RuntimeError("V10 domain seeds overlap an earlier stage")


def _validate_settings(settings: BenchmarkSettings, seeds: Tuple[int, ...]) -> None:
    if tuple(settings.domain_seeds) != seeds:
        raise RuntimeError("V10 seeds differ from the declared stage")
    observed = {
        "samples": settings.samples,
        "duration_s": settings.duration,
        "control_points": settings.control_points,
        "iterations": settings.iterations,
        "active_zone_budget": settings.number_of_windows,
        "half_width": settings.half_width,
    }
    if observed != FORMAL_SETTINGS:
        raise RuntimeError("V10 settings differ from the frozen design")


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
                paired_noise_seed = (
                    160000
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
                        paired_noise_seed,
                    )
                    result["manifest_id"] = str(manifest["manifest_id"])
                    result["trajectory"] = family
                    result["regime"] = str(manifest["regime"])
                    rows.append({column: result[column] for column in ROW_COLUMNS})
        print(
            "[V10] completed cell "
            + scenario.scenario_id
            + " ("
            + str(len(rows))
            + " method runs)",
            flush=True,
        )
    return rows


def run_factorial_audit(
    taskset_path: Path,
    output_directory: Path,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    _assert_seed_isolation()
    _validate_settings(settings, AUDIT_DOMAIN_SEEDS)
    taskset = validate_frozen_taskset(taskset_path)
    manifests = [
        item for item in taskset["manifests"] if item["regime"] == "demand_conflict"
    ]
    scenarios = (FACTORIAL_SCENARIOS[0], FACTORIAL_SCENARIOS[-1])
    rows = _run_grid(manifests, scenarios, settings)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "factorial_audit_raw.csv", rows)
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
        "baseline_and_extreme_cells_both_executed": {
            str(row["scenario_id"]) for row in rows
        }
        == {"n0_d0_m0", EXTREME_SCENARIO_ID},
    }
    payload = {
        "stage": "v10_numerical_audit_only",
        "audit_domain_seeds": list(AUDIT_DOMAIN_SEEDS),
        "taskset_sha256": taskset_sha256(taskset_path),
        "scenarios": [scenario_to_dict(item) for item in scenarios],
        "methods": list(ROBUSTNESS_METHODS),
        "total_method_runs": len(rows),
        "finite_rate": finite_rate,
        "primary_success_rate": success_rate,
        "criteria": criteria,
        "passed": bool(all(criteria.values())),
        "decision": "FREEZE_FORMAL_PROTOCOL" if all(criteria.values()) else "REVISE",
    }
    (output_directory / "factorial_audit_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def preregistered_protocol(
    taskset_path: Path,
    audit_summary_path: Path,
    v9_protocol_path: Path,
) -> Dict[str, object]:
    _assert_seed_isolation()
    audit = json.loads(audit_summary_path.read_text(encoding="utf-8"))
    if not bool(audit.get("passed")):
        raise RuntimeError("V10 audit did not authorize formal execution")
    v9_protocol = json.loads(v9_protocol_path.read_text(encoding="utf-8"))
    runner_path = Path(__file__).with_name("robustness_runner.py")
    if v9_protocol["taskset_sha256"] != taskset_sha256(taskset_path):
        raise RuntimeError("V9 protocol and V10 taskset hashes differ")
    if v9_protocol["robustness_runner_sha256"] != file_sha256(runner_path):
        raise RuntimeError("frozen V9 stress runner changed before V10")
    return {
        "protocol_id": "v10-2x2x2-combined-stress-factorial",
        "frozen_before_formal_execution": True,
        "design": "full 2^3 factorial on high stress levels",
        "factor_order": ["measurement_noise", "added_delay", "dynamic_mismatch"],
        "factor_levels": FACTOR_LEVELS,
        "scenarios": [scenario_to_dict(item) for item in FACTORIAL_SCENARIOS],
        "primary_method": PRIMARY_METHOD,
        "primary_scope": "demand_conflict",
        "primary_comparator": PRIMARY_COMPARATOR,
        "extreme_scenario_id": EXTREME_SCENARIO_ID,
        "methods": list(ROBUSTNESS_METHODS),
        "settings": FORMAL_SETTINGS,
        "formal_domain_seeds": list(FORMAL_DOMAIN_SEEDS),
        "audit_domain_seeds_excluded": list(AUDIT_DOMAIN_SEEDS),
        "taskset_sha256": taskset_sha256(taskset_path),
        "v9_protocol_sha256": file_sha256(v9_protocol_path),
        "audit_summary_sha256": file_sha256(audit_summary_path),
        "robustness_runner_sha256": file_sha256(runner_path),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "formal_criteria": FORMAL_CRITERIA,
        "factorial_effect_convention": (
            "contrast effect divided by 2^(k-1), reported relative to case grand mean"
        ),
        "formal_method_runs": (
            15
            * len(FORMAL_DOMAIN_SEEDS)
            * len(FACTORIAL_SCENARIOS)
            * len(ROBUSTNESS_METHODS)
        ),
    }


def freeze_factorial_protocol(
    taskset_path: Path,
    audit_summary_path: Path,
    v9_protocol_path: Path,
    protocol_path: Path,
) -> Dict[str, object]:
    protocol = preregistered_protocol(
        taskset_path,
        audit_summary_path,
        v9_protocol_path,
    )
    if protocol["taskset_sha256"] != EXPECTED_TASKSET_SHA256:
        raise RuntimeError("unexpected taskset hash while freezing V10")
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("existing V10 protocol differs from current inputs")
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return protocol


def validate_factorial_protocol(
    taskset_path: Path,
    audit_summary_path: Path,
    v9_protocol_path: Path,
    protocol_path: Path,
) -> Dict[str, object]:
    if not protocol_path.exists():
        raise RuntimeError("V10 protocol must be frozen before formal execution")
    existing = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = preregistered_protocol(
        taskset_path,
        audit_summary_path,
        v9_protocol_path,
    )
    if existing != expected:
        raise RuntimeError("frozen V10 protocol no longer matches code or inputs")
    return existing


def run_factorial_confirmation(
    taskset_path: Path,
    audit_summary_path: Path,
    v9_protocol_path: Path,
    protocol_path: Path,
    output_directory: Path,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    _validate_settings(settings, FORMAL_DOMAIN_SEEDS)
    protocol = validate_factorial_protocol(
        taskset_path,
        audit_summary_path,
        v9_protocol_path,
        protocol_path,
    )
    taskset = validate_frozen_taskset(taskset_path)
    rows = _run_grid(taskset["manifests"], FACTORIAL_SCENARIOS, settings)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "factorial_effectiveness_raw.csv", rows)
    metadata = {
        "protocol_sha256": file_sha256(protocol_path),
        "protocol": protocol,
        "total_method_runs": len(rows),
        "paired_cases_per_cell_all_tasks": 15 * len(FORMAL_DOMAIN_SEEDS),
        "paired_cases_per_cell_conflict": 5 * len(FORMAL_DOMAIN_SEEDS),
    }
    (output_directory / "factorial_effectiveness_metrics.json").write_text(
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
        "bootstrap_median_improvement_95ci_percent": _bootstrap_interval(
            improvement,
            rng,
        ),
        "one_sided_wilcoxon_p_proposed_lower": p_value,
    }


def _absolute_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for scenario in FACTORIAL_SCENARIOS:
        for scope in ("all", "demand_conflict"):
            scoped = [
                row
                for row in rows
                if row["scenario_id"] == scenario.scenario_id
                and (scope == "all" or row["regime"] == scope)
            ]
            for method in ROBUSTNESS_METHODS:
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


def _factorial_effects(
    rows: Sequence[Dict[str, object]],
    method: str,
    regime: str,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, object]]:
    selected = [
        row for row in rows if row["method"] == method and row["regime"] == regime
    ]
    indexed = {
        (str(row["manifest_id"]), int(row["domain_seed"]), str(row["scenario_id"])): row
        for row in selected
    }
    keys = sorted({(key[0], key[1]) for key in indexed})
    terms = {
        "noise": (0,),
        "delay": (1,),
        "mismatch": (2,),
        "noise_x_delay": (0, 1),
        "noise_x_mismatch": (0, 2),
        "delay_x_mismatch": (1, 2),
        "noise_x_delay_x_mismatch": (0, 1, 2),
    }
    output: Dict[str, Dict[str, object]] = {}
    for term_name, axes in terms.items():
        relative_effects = []
        for key in keys:
            cell_values = {
                scenario.scenario_id: float(
                    indexed[key + (scenario.scenario_id,)]["task_auc_normalized"]
                )
                for scenario in FACTORIAL_SCENARIOS
            }
            grand_mean = float(np.mean(list(cell_values.values())))
            contrast = 0.0
            for scenario_id, value in cell_values.items():
                codes = factor_codes(scenario_id)
                sign = float(np.prod([1.0 if codes[axis] else -1.0 for axis in axes]))
                contrast += sign * value
            effect = contrast / 4.0
            relative_effects.append(100.0 * effect / max(grand_mean, 1.0e-12))
        values = np.asarray(relative_effects, dtype=float)
        output[term_name] = {
            "method": method,
            "scope": regime,
            "paired_cases": len(keys),
            "median_relative_effect_percent": float(np.median(values)),
            "mean_relative_effect_percent": float(np.mean(values)),
            "bootstrap_median_95ci_percent": _bootstrap_interval(values, rng),
        }
    return output


def _extreme_degradation(
    rows: Sequence[Dict[str, object]],
    method: str,
    regime: str,
    rng: np.random.Generator,
) -> Dict[str, object]:
    selected = [
        row for row in rows if row["method"] == method and row["regime"] == regime
    ]
    indexed = {
        (str(row["manifest_id"]), int(row["domain_seed"]), str(row["scenario_id"])): row
        for row in selected
    }
    keys = sorted({(key[0], key[1]) for key in indexed})
    degradation = []
    excess = []
    for key in keys:
        y000 = float(indexed[key + ("n0_d0_m0",)]["task_auc_normalized"])
        y100 = float(indexed[key + ("n1_d0_m0",)]["task_auc_normalized"])
        y010 = float(indexed[key + ("n0_d1_m0",)]["task_auc_normalized"])
        y001 = float(indexed[key + ("n0_d0_m1",)]["task_auc_normalized"])
        y111 = float(indexed[key + ("n1_d1_m1",)]["task_auc_normalized"])
        degradation.append(100.0 * (y111 - y000) / max(y000, 1.0e-12))
        additive_prediction = y000 + (y100 - y000) + (y010 - y000) + (y001 - y000)
        excess.append(100.0 * (y111 - additive_prediction) / max(y000, 1.0e-12))
    degradation_values = np.asarray(degradation, dtype=float)
    excess_values = np.asarray(excess, dtype=float)
    return {
        "method": method,
        "scope": regime,
        "paired_cases": len(keys),
        "median_extreme_auc_degradation_percent": float(
            np.median(degradation_values)
        ),
        "bootstrap_extreme_degradation_95ci_percent": _bootstrap_interval(
            degradation_values,
            rng,
        ),
        "median_combined_excess_over_additive_singles_percent": float(
            np.median(excess_values)
        ),
        "bootstrap_combined_excess_95ci_percent": _bootstrap_interval(
            excess_values,
            rng,
        ),
    }


def _plot_factorial(
    output_path: Path,
    summary: Sequence[Dict[str, object]],
    cell_vs_peak: Dict[str, Dict[str, object]],
    effects: Dict[str, Dict[str, object]],
    classification: str,
) -> None:
    labels = {
        "n0_d0_m0": "Base",
        "n1_d0_m0": "N",
        "n0_d1_m0": "D",
        "n0_d0_m1": "M",
        "n1_d1_m0": "N+D",
        "n1_d0_m1": "N+M",
        "n0_d1_m1": "D+M",
        "n1_d1_m1": "N+D+M",
    }
    method_labels = {
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
    scenario_ids = [item.scenario_id for item in FACTORIAL_SCENARIOS]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.9))

    x = np.arange(len(scenario_ids))
    width = 0.19
    for method_index, method in enumerate(ROBUSTNESS_METHODS):
        values = [
            next(
                row["median_task_auc"]
                for row in summary
                if row["scenario_id"] == scenario_id
                and row["scope"] == "demand_conflict"
                and row["method"] == method
            )
            for scenario_id in scenario_ids
        ]
        axes[0, 0].bar(
            x + (method_index - 1.5) * width,
            values,
            width,
            label=method_labels[method],
            color=colors[method],
        )
    axes[0, 0].set_xticks(x, [labels[item] for item in scenario_ids], rotation=30, ha="right")
    axes[0, 0].set_ylabel("Median normalized task AUC")
    axes[0, 0].set_title("Absolute performance across factorial cells")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)

    medians = [cell_vs_peak[item]["median_task_auc_improvement_percent"] for item in scenario_ids]
    intervals = [cell_vs_peak[item]["bootstrap_median_improvement_95ci_percent"] for item in scenario_ids]
    lower = [median - interval[0] for median, interval in zip(medians, intervals)]
    upper = [interval[1] - median for median, interval in zip(medians, intervals)]
    axes[0, 1].bar(x, medians, color="#2f6b8a")
    axes[0, 1].errorbar(x, medians, yerr=[lower, upper], fmt="none", color="#333333", capsize=3)
    axes[0, 1].axhline(0.0, color="#666666", linewidth=1.0)
    axes[0, 1].set_xticks(x, [labels[item] for item in scenario_ids], rotation=30, ha="right")
    axes[0, 1].set_ylabel("Dual-anchor improvement vs error peak (%)")
    axes[0, 1].set_title("Paired advantage by combined-stress cell")

    term_labels = {
        "noise": "Noise",
        "delay": "Delay",
        "mismatch": "Mismatch",
        "noise_x_delay": "Noise x delay",
        "noise_x_mismatch": "Noise x mismatch",
        "delay_x_mismatch": "Delay x mismatch",
        "noise_x_delay_x_mismatch": "Three-way",
    }
    terms = list(term_labels)
    effect_values = [effects[item]["median_relative_effect_percent"] for item in terms]
    effect_intervals = [effects[item]["bootstrap_median_95ci_percent"] for item in terms]
    error_lower = [value - interval[0] for value, interval in zip(effect_values, effect_intervals)]
    error_upper = [interval[1] - value for value, interval in zip(effect_values, effect_intervals)]
    y = np.arange(len(terms))
    axes[1, 0].errorbar(
        effect_values,
        y,
        xerr=[error_lower, error_upper],
        fmt="o",
        color="#2f6b8a",
        capsize=3,
    )
    axes[1, 0].axvline(0.0, color="#666666", linewidth=1.0)
    axes[1, 0].set_yticks(y, [term_labels[item] for item in terms])
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Relative factorial effect on task AUC (%)")
    axes[1, 0].set_title("Dual-anchor main and interaction effects")

    for noise, mismatch, style, marker in (
        (0, 0, "-", "o"),
        (1, 0, "--", "s"),
        (0, 1, "-.", "^"),
        (1, 1, ":", "D"),
    ):
        values = []
        for delay in (0, 1):
            scenario_id = f"n{noise}_d{delay}_m{mismatch}"
            values.append(
                next(
                    row["median_task_auc"]
                    for row in summary
                    if row["scenario_id"] == scenario_id
                    and row["scope"] == "demand_conflict"
                    and row["method"] == PRIMARY_METHOD
                )
            )
        axes[1, 1].plot(
            (0, 4),
            values,
            linestyle=style,
            marker=marker,
            color="#2f6b8a" if noise == 0 else "#b46b5a",
            label=f"N={noise}, M={mismatch}",
        )
    axes[1, 1].set_xticks((0, 4))
    axes[1, 1].set_xlabel("Additional delay (steps)")
    axes[1, 1].set_ylabel("Median normalized task AUC")
    axes[1, 1].set_title("Dual-anchor interaction profiles")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle("V10 combined-stress factorial: " + classification)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_factorial_confirmation(result_directory: Path) -> Dict[str, object]:
    rows = _numeric_rows(result_directory / "factorial_effectiveness_raw.csv")
    metadata = json.loads(
        (result_directory / "factorial_effectiveness_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if metadata["protocol"]["taskset_sha256"] != EXPECTED_TASKSET_SHA256:
        raise RuntimeError("V10 result metadata has the wrong taskset hash")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons: Dict[str, Dict[str, Dict[str, object]]] = {}
    for scenario in FACTORIAL_SCENARIOS:
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

    summary = _absolute_summary(rows)
    factorial_effects = {
        method: _factorial_effects(
            rows,
            method,
            "demand_conflict",
            rng,
        )
        for method in ROBUSTNESS_METHODS
    }
    degradation = {
        method: _extreme_degradation(
            rows,
            method,
            "demand_conflict",
            rng,
        )
        for method in ROBUSTNESS_METHODS
    }
    extreme_vs_peak = comparisons[EXTREME_SCENARIO_ID]["demand_conflict"][
        PRIMARY_COMPARATOR
    ]
    extreme_vs_full = comparisons[EXTREME_SCENARIO_ID]["all"]["full_trajectory"]
    extreme_primary = next(
        item
        for item in summary
        if item["scenario_id"] == EXTREME_SCENARIO_ID
        and item["scope"] == "all"
        and item["method"] == PRIMARY_METHOD
    )
    per_scenario_success = {
        scenario.scenario_id: float(
            next(
                item["success_rate"]
                for item in summary
                if item["scenario_id"] == scenario.scenario_id
                and item["scope"] == "all"
                and item["method"] == PRIMARY_METHOD
            )
        )
        for scenario in FACTORIAL_SCENARIOS
    }
    criteria = {
        "extreme_conflict_vs_error_peak_ci_lower_above_zero": (
            extreme_vs_peak["bootstrap_median_improvement_95ci_percent"][0] > 0.0
        ),
        "extreme_conflict_vs_error_peak_win_rate_at_least_60_percent": (
            extreme_vs_peak["paired_win_rate"]
            >= FORMAL_CRITERIA[
                "extreme_conflict_vs_error_peak_win_rate_at_least"
            ]
        ),
        "extreme_all_tasks_vs_full_ci_lower_above_zero": (
            extreme_vs_full["bootstrap_median_improvement_95ci_percent"][0] > 0.0
        ),
        "extreme_median_normalized_task_auc_below_one": (
            extreme_primary["median_task_auc"]
            < FORMAL_CRITERIA["extreme_median_normalized_task_auc_below"]
        ),
        "extreme_median_final_task_ratio_below_one": (
            extreme_primary["median_final_task_ratio"]
            < FORMAL_CRITERIA["extreme_median_final_task_ratio_below"]
        ),
        "all_scenarios_success_rate_at_least_95_percent": all(
            value
            >= FORMAL_CRITERIA[
                "per_scenario_solver_and_constraint_success_at_least"
            ]
            for value in per_scenario_success.values()
        ),
    }
    relative_ok = bool(
        criteria["extreme_conflict_vs_error_peak_ci_lower_above_zero"]
        and criteria[
            "extreme_conflict_vs_error_peak_win_rate_at_least_60_percent"
        ]
        and criteria["extreme_all_tasks_vs_full_ci_lower_above_zero"]
    )
    absolute_ok = bool(
        criteria["extreme_median_normalized_task_auc_below_one"]
        and criteria["extreme_median_final_task_ratio_below_one"]
        and criteria["all_scenarios_success_rate_at_least_95_percent"]
    )
    interval = extreme_vs_peak["bootstrap_median_improvement_95ci_percent"]
    if relative_ok and absolute_ok:
        classification = "COMBINED_STRESS_ROBUST"
    elif relative_ok:
        classification = "RELATIVE_ROBUST_ABSOLUTE_DEGRADED"
    elif interval[0] <= 0.0 <= interval[1]:
        classification = "COMBINED_ADVANTAGE_UNRESOLVED"
    else:
        classification = "NOT_COMBINED_STRESS_ROBUST"

    cell_vs_peak = {
        scenario.scenario_id: comparisons[scenario.scenario_id][
            "demand_conflict"
        ][PRIMARY_COMPARATOR]
        for scenario in FACTORIAL_SCENARIOS
    }
    payload = {
        "classification": classification,
        "protocol_sha256": metadata["protocol_sha256"],
        "taskset_sha256": EXPECTED_TASKSET_SHA256,
        "criteria": criteria,
        "extreme_primary_vs_error_peak": extreme_vs_peak,
        "extreme_secondary_vs_full": extreme_vs_full,
        "extreme_primary_absolute_summary": extreme_primary,
        "primary_success_rate_by_cell": per_scenario_success,
        "comparisons": comparisons,
        "absolute_summary": summary,
        "factorial_effects": factorial_effects,
        "extreme_degradation": degradation,
    }
    (result_directory / "comparison_v10_factorial.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(result_directory / "factorial_summary.csv", summary)
    _plot_factorial(
        result_directory / "factorial_summary.png",
        summary,
        cell_vs_peak,
        factorial_effects[PRIMARY_METHOD],
        classification,
    )

    lines = [
        "# V10 三因素组合压力全因子实验",
        "",
        "## 冻结结论",
        "",
        "- 分类：**" + classification + "**",
        "- 极端组合：0.05 mm 测量噪声 + 4 步附加时延 + 1.70× 动力学失配。",
        "- 极端组合相对误差峰值法：中位 AUC 提升 "
        + f"{extreme_vs_peak['median_task_auc_improvement_percent']:.2f}%"
        + "，胜率 "
        + f"{100.0 * extreme_vs_peak['paired_win_rate']:.2f}%"
        + "，95% CI ["
        + f"{interval[0]:.2f}%, {interval[1]:.2f}%"
        + "]。",
        "- 极端组合相对全轨迹法（全部任务）：中位 AUC 提升 "
        + f"{extreme_vs_full['median_task_auc_improvement_percent']:.2f}%"
        + "。",
        "- 极端组合双锚点绝对中位 AUC："
        + f"{extreme_primary['median_task_auc']:.3f}"
        + "；最终任务比："
        + f"{extreme_primary['median_final_task_ratio']:.3f}"
        + "。",
        "",
        "## 双锚点的主效应与交互效应",
        "",
        "正值表示该因素或交互使任务 AUC 变差，负值表示抵消。",
        "",
        "| 效应 | 中位相对效应 | Bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    effect_labels = {
        "noise": "噪声主效应",
        "delay": "时延主效应",
        "mismatch": "失配主效应",
        "noise_x_delay": "噪声×时延",
        "noise_x_mismatch": "噪声×失配",
        "delay_x_mismatch": "时延×失配",
        "noise_x_delay_x_mismatch": "三因素交互",
    }
    for key, label in effect_labels.items():
        item = factorial_effects[PRIMARY_METHOD][key]
        effect_interval = item["bootstrap_median_95ci_percent"]
        lines.append(
            "| "
            + label
            + " | "
            + f"{item['median_relative_effect_percent']:.2f}%"
            + " | ["
            + f"{effect_interval[0]:.2f}%, {effect_interval[1]:.2f}%"
            + "] |"
        )
    dual_degradation = degradation[PRIMARY_METHOD]
    lines.extend(
        [
            "",
            "## 双锚点各格点的绝对任务量",
            "",
            "| 格点 | 初始任务分数 | 最终任务分数 | 归一化 AUC | 最终任务比 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scenario in FACTORIAL_SCENARIOS:
        item = next(
            row
            for row in summary
            if row["scenario_id"] == scenario.scenario_id
            and row["scope"] == "demand_conflict"
            and row["method"] == PRIMARY_METHOD
        )
        lines.append(
            "| "
            + scenario.scenario_id
            + " | "
            + f"{item['median_initial_task_score']:.3f}"
            + " | "
            + f"{item['median_final_task_score']:.3f}"
            + " | "
            + f"{item['median_task_auc']:.3f}"
            + " | "
            + f"{item['median_final_task_ratio']:.3f}"
            + " |"
        )
    lines.extend(
        [
            "",
            "## 极端组合退化",
            "",
            "- 极端组合相对基线的双锚点 AUC 中位退化："
            + f"{dual_degradation['median_extreme_auc_degradation_percent']:.2f}%"
            + "。",
            "- 超出三个单因素退化线性相加预测的中位部分："
            + f"{dual_degradation['median_combined_excess_over_additive_singles_percent']:.2f}%"
            + "。",
            "",
            "## 解释边界",
            "",
            "该实验使用高/低两级全因子设计，可估计交互方向，但不能给出连续压力空间中的精确拐点。失配倍率同时外推固有频率、阻尼、摩擦、耦合和重复扰动；这些变化对不同轴可能相互抵消，因此负的归一化失配效应只能解释为当前虚拟域中的相对收敛现象，不能解释成失配改善了加工质量。全部结论仍来自虚拟物理机床，不替代真实机床验证。",
            "",
        ]
    )
    (result_directory / "factorial_diagnosis_zh.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return payload
