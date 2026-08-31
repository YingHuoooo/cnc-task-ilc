"""Externally specified machining-task zones and budgeted adaptive ILC."""

import csv
import itertools
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(_PROJECT_ROOT / ".matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from .basis import apply_axis_coefficients, cubic_bspline_basis
from .benchmark import BenchmarkSettings, TRAJECTORY_FAMILIES
from .ilc import (
    ILCConfig,
    build_contour_sensitivity,
    solve_constrained_update,
)
from .metrics import command_kinematics, constraint_report, task_errors
from .plant import make_virtual_machine_domain, nominal_config, simulate_machine
from .trajectory import ReferenceTrajectory, make_trajectory_family


TASK_ZONE_NAMES = (
    "entry_datum",
    "roughing_transition",
    "fit_surface",
    "flow_transition",
    "seal_surface",
    "exit_datum",
)
TASK_ZONE_FRACTIONS = (0.10, 0.26, 0.42, 0.58, 0.74, 0.90)
TASK_ZONE_TOLERANCES_MM = (0.22, 0.32, 0.24, 0.28, 0.20, 0.30)

TASK_METHODS = (
    "full_trajectory",
    "static_tolerance",
    "curvature_zones",
    "jerk_zones",
    "error_peak_dynamic",
    "violation_dynamic",
    "random_zones",
    "nominal_lookahead_dynamic",
)


@dataclass(frozen=True)
class TaskSpecification:
    names: Tuple[str, ...]
    centers: Tuple[int, ...]
    windows: Tuple[Tuple[int, int], ...]
    tolerances: Tuple[float, ...]
    evaluation_mask: np.ndarray
    tolerance_profile: np.ndarray


def make_task_specification(
    reference: ReferenceTrajectory,
    half_width: int,
) -> TaskSpecification:
    """Create CAD/process zones without using any measured tracking error."""

    centers = tuple(
        int(round(fraction * (reference.time.size - 1)))
        for fraction in TASK_ZONE_FRACTIONS
    )
    if any(
        second - first <= 2 * half_width
        for first, second in zip(centers[:-1], centers[1:])
    ):
        raise RuntimeError("task zones overlap at the configured resolution")
    mask = np.zeros(reference.time.size, dtype=bool)
    tolerance_profile = np.full(reference.time.size, np.nan, dtype=float)
    windows = []
    for center, tolerance in zip(centers, TASK_ZONE_TOLERANCES_MM):
        start = center - half_width
        stop = center + half_width + 1
        if start < 0 or stop > reference.time.size:
            raise RuntimeError("task zone exceeds the trajectory boundary")
        mask[start:stop] = True
        tolerance_profile[start:stop] = tolerance
        windows.append((int(start), int(stop - 1)))
    return TaskSpecification(
        names=TASK_ZONE_NAMES,
        centers=centers,
        windows=tuple(windows),
        tolerances=TASK_ZONE_TOLERANCES_MM,
        evaluation_mask=mask,
        tolerance_profile=tolerance_profile,
    )


def _zone_mask(
    specification: TaskSpecification,
    selected_zones: Sequence[int],
) -> np.ndarray:
    mask = np.zeros_like(specification.evaluation_mask)
    for index in selected_zones:
        start, stop = specification.windows[int(index)]
        mask[start : stop + 1] = True
    return mask


def _task_metrics(
    contour_error: np.ndarray,
    specification: TaskSpecification,
) -> Dict[str, float]:
    selected = specification.evaluation_mask
    normalized_error = (
        np.abs(contour_error[selected])
        / specification.tolerance_profile[selected]
    )
    return {
        "task_nrmse": float(np.sqrt(np.mean(normalized_error**2))),
        "task_violation_rate": float(np.mean(normalized_error > 1.0)),
        "task_max_ratio": float(np.max(normalized_error)),
        "global_rmse": float(np.sqrt(np.mean(contour_error**2))),
    }


def _task_config(
    reference: ReferenceTrajectory,
    settings: BenchmarkSettings,
) -> ILCConfig:
    kinematics = command_kinematics(reference.position, reference.dt)
    return ILCConfig(
        iterations=settings.iterations,
        correction_limit=4.0,
        velocity_limit=1.42
        * float(np.max(np.abs(kinematics["velocity"]))),
        acceleration_limit=1.80
        * float(np.max(np.abs(kinematics["acceleration"]))),
        regularization=3.0e-3,
        smoothness=2.0e-8,
        learning_rate=0.65,
        global_protection_weight=0.30,
        critical_boost=5.0,
        solver_max_iterations=240,
    )


def _reference_jerk(reference: ReferenceTrajectory) -> np.ndarray:
    velocity = np.gradient(reference.position, reference.dt, axis=0)
    acceleration = np.gradient(velocity, reference.dt, axis=0)
    jerk = np.gradient(acceleration, reference.dt, axis=0)
    return np.linalg.norm(jerk, axis=1)


def _zone_statistic(
    values: np.ndarray,
    specification: TaskSpecification,
    statistic: str,
) -> np.ndarray:
    scores = []
    for start, stop in specification.windows:
        local = np.asarray(values[start : stop + 1], dtype=float)
        if statistic == "max":
            scores.append(float(np.max(np.abs(local))))
        elif statistic == "rms":
            scores.append(float(np.sqrt(np.mean(local**2))))
        elif statistic == "mean":
            scores.append(float(np.mean(np.abs(local))))
        else:
            raise ValueError("unknown zone statistic")
    return np.asarray(scores, dtype=float)


def _fixed_zone_selection(
    method: str,
    reference: ReferenceTrajectory,
    specification: TaskSpecification,
    budget: int,
    random_seed: int,
) -> Tuple[int, ...]:
    if method == "static_tolerance":
        score = -np.asarray(specification.tolerances)
    elif method == "curvature_zones":
        score = _zone_statistic(
            reference.curvature,
            specification,
            "mean",
        )
    elif method == "jerk_zones":
        score = _zone_statistic(
            _reference_jerk(reference),
            specification,
            "mean",
        )
    elif method == "random_zones":
        score = np.random.RandomState(random_seed).uniform(
            size=len(specification.names)
        )
    else:
        raise ValueError("method does not have a fixed zone selection")
    return tuple(sorted(int(index) for index in np.argsort(score)[-budget:]))


def _dynamic_zone_selection(
    method: str,
    contour_error: np.ndarray,
    specification: TaskSpecification,
    budget: int,
) -> Tuple[int, ...]:
    if method == "error_peak_dynamic":
        score = _zone_statistic(
            contour_error,
            specification,
            "max",
        )
    elif method == "violation_dynamic":
        score = []
        for (start, stop), tolerance in zip(
            specification.windows,
            specification.tolerances,
        ):
            ratio = np.abs(contour_error[start : stop + 1]) / tolerance
            exceedance = np.maximum(ratio - 1.0, 0.0)
            score.append(
                float(
                    np.sqrt(np.mean(ratio**2))
                    + 0.5 * np.mean(exceedance)
                )
            )
        score = np.asarray(score)
    else:
        raise ValueError("method does not have a dynamic zone selection")
    return tuple(sorted(int(index) for index in np.argsort(score)[-budget:]))


def _weights_for_mask(
    contour_error: np.ndarray,
    optimization_mask: np.ndarray,
    config: ILCConfig,
) -> np.ndarray:
    weights = np.full(
        contour_error.shape,
        config.global_protection_weight,
        dtype=float,
    )
    weights[optimization_mask] += config.critical_boost
    weights /= np.mean(weights)
    return weights


def _task_aware_weights(
    contour_error: np.ndarray,
    specification: TaskSpecification,
    selected_zones: Sequence[int],
    config: ILCConfig,
) -> np.ndarray:
    """Weight selected zones by their externally specified tolerances."""

    weights = np.full(
        contour_error.shape,
        config.global_protection_weight,
        dtype=float,
    )
    tolerance_reference = max(specification.tolerances)
    for index in selected_zones:
        start, stop = specification.windows[int(index)]
        tolerance = specification.tolerances[int(index)]
        severity = (tolerance_reference / tolerance) ** 2
        weights[start : stop + 1] += config.critical_boost * severity
    weights /= np.mean(weights)
    return weights


def _lookahead_objective(
    predicted_error: np.ndarray,
    current_error: np.ndarray,
    specification: TaskSpecification,
) -> float:
    task = _task_metrics(predicted_error, specification)
    current_global = float(np.sqrt(np.mean(current_error**2)))
    return float(
        task["task_nrmse"]
        + 0.20 * task["task_violation_rate"]
        + 0.05 * task["global_rmse"] / max(current_global, 1.0e-12)
    )


def _choose_nominal_lookahead(
    contour_error: np.ndarray,
    sensitivity: np.ndarray,
    basis: np.ndarray,
    reference: ReferenceTrajectory,
    initial_command: np.ndarray,
    current_command: np.ndarray,
    specification: TaskSpecification,
    budget: int,
    config: ILCConfig,
) -> Tuple[Tuple[int, ...], Dict[str, object]]:
    best_selection = None
    best_update = None
    best_objective = float("inf")
    for combination in itertools.combinations(
        range(len(specification.names)),
        budget,
    ):
        update = solve_constrained_update(
            contour_error=contour_error,
            sensitivity=sensitivity,
            weights=_task_aware_weights(
                contour_error,
                specification,
                combination,
                config,
            ),
            basis=basis,
            initial_command=initial_command,
            current_command=current_command,
            dt=reference.dt,
            config=config,
        )
        predicted_error = (
            contour_error
            + sensitivity @ (config.learning_rate * update["delta"])
        )
        objective = _lookahead_objective(
            predicted_error,
            contour_error,
            specification,
        )
        if objective < best_objective:
            best_objective = objective
            best_selection = tuple(int(index) for index in combination)
            best_update = update
    if best_selection is None or best_update is None:
        raise RuntimeError("nominal lookahead did not choose a zone set")
    return best_selection, best_update


def run_task_method(
    method: str,
    reference: ReferenceTrajectory,
    basis: np.ndarray,
    specification: TaskSpecification,
    plant_seed: int,
    settings: BenchmarkSettings,
    random_seed: int,
) -> Dict[str, object]:
    """Run one adaptive task-budget method on one virtual machine."""

    config = _task_config(reference, settings)
    sensitivity = build_contour_sensitivity(
        reference,
        basis,
        nominal_config(),
    )
    plant = make_virtual_machine_domain(plant_seed)
    current_command = reference.position.copy()
    metrics = []
    selections: List[Tuple[int, ...]] = []
    solver_status = []
    fixed_selection = None
    if method in (
        "static_tolerance",
        "curvature_zones",
        "jerk_zones",
        "random_zones",
    ):
        fixed_selection = _fixed_zone_selection(
            method,
            reference,
            specification,
            budget=settings.number_of_windows,
            random_seed=random_seed,
        )

    start_time = time.perf_counter()
    for trial in range(config.iterations + 1):
        feedback = simulate_machine(
            current_command,
            reference.dt,
            plant,
        )
        contour_error = task_errors(reference, feedback)["contour"]
        summary = _task_metrics(contour_error, specification)
        summary["trial"] = float(trial)
        metrics.append(summary)
        if trial == config.iterations:
            break

        if method == "full_trajectory":
            selection = tuple(range(len(specification.names)))
            update = solve_constrained_update(
                contour_error=contour_error,
                sensitivity=sensitivity,
                weights=np.ones_like(contour_error),
                basis=basis,
                initial_command=reference.position,
                current_command=current_command,
                dt=reference.dt,
                config=config,
            )
        elif method == "nominal_lookahead_dynamic":
            selection, update = _choose_nominal_lookahead(
                contour_error,
                sensitivity,
                basis,
                reference,
                reference.position,
                current_command,
                specification,
                settings.number_of_windows,
                config,
            )
        else:
            if fixed_selection is not None:
                selection = fixed_selection
            else:
                selection = _dynamic_zone_selection(
                    method,
                    contour_error,
                    specification,
                    settings.number_of_windows,
                )
            mask = _zone_mask(specification, selection)
            update = solve_constrained_update(
                contour_error=contour_error,
                sensitivity=sensitivity,
                weights=_weights_for_mask(contour_error, mask, config),
                basis=basis,
                initial_command=reference.position,
                current_command=current_command,
                dt=reference.dt,
                config=config,
            )
        selections.append(tuple(selection))
        solver_status.append(bool(update["success"]))
        current_command = apply_axis_coefficients(
            current_command,
            basis,
            config.learning_rate * update["delta"],
        )
    elapsed = time.perf_counter() - start_time

    task_values = np.asarray(
        [metric["task_nrmse"] for metric in metrics],
        dtype=float,
    )
    normalized = task_values / max(task_values[0], 1.0e-12)
    auc = float(trapezoid(normalized, dx=1.0) / config.iterations)
    constraints = constraint_report(
        initial_command=reference.position,
        learned_command=current_command,
        dt=reference.dt,
        max_correction=config.correction_limit,
        velocity_limit=config.velocity_limit,
        acceleration_limit=config.acceleration_limit,
    )
    selection_switches = sum(
        first != second
        for first, second in zip(selections[:-1], selections[1:])
    )
    return {
        "method": method,
        "domain_seed": int(plant_seed),
        "initial_task_nrmse": float(metrics[0]["task_nrmse"]),
        "final_task_nrmse": float(metrics[-1]["task_nrmse"]),
        "task_auc_normalized": auc,
        "final_task_ratio": float(
            metrics[-1]["task_nrmse"]
            / max(metrics[0]["task_nrmse"], 1.0e-12)
        ),
        "initial_violation_rate": float(
            metrics[0]["task_violation_rate"]
        ),
        "final_violation_rate": float(
            metrics[-1]["task_violation_rate"]
        ),
        "initial_task_max_ratio": float(metrics[0]["task_max_ratio"]),
        "final_task_max_ratio": float(metrics[-1]["task_max_ratio"]),
        "initial_global_rmse": float(metrics[0]["global_rmse"]),
        "final_global_rmse": float(metrics[-1]["global_rmse"]),
        "final_global_ratio": float(
            metrics[-1]["global_rmse"]
            / max(metrics[0]["global_rmse"], 1.0e-12)
        ),
        "selection_switches": int(selection_switches),
        "selection_history": json.dumps(selections),
        "constraint_violation": int(
            constraints["constraint_violation"]
        ),
        "all_updates_succeeded": int(all(solver_status)),
        "elapsed_s": float(elapsed),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summary = []
    for method in TASK_METHODS:
        selected = [row for row in rows if row["method"] == method]
        auc = np.asarray([row["task_auc_normalized"] for row in selected])
        summary.append(
            {
                "method": method,
                "runs": len(selected),
                "median_task_auc": float(np.median(auc)),
                "q25_task_auc": float(np.percentile(auc, 25.0)),
                "q75_task_auc": float(np.percentile(auc, 75.0)),
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
                "median_selection_switches": float(
                    np.median(
                        [row["selection_switches"] for row in selected]
                    )
                ),
                "success_rate": float(
                    np.mean(
                        [
                            row["all_updates_succeeded"]
                            and not row["constraint_violation"]
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


def _evaluate_gate(
    rows: Sequence[Dict[str, object]],
    summary: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    primary = "nominal_lookahead_dynamic"
    baselines = set(TASK_METHODS) - {primary}
    summary_by_method = {row["method"]: row for row in summary}
    strongest = min(
        baselines,
        key=lambda method: summary_by_method[method]["median_task_auc"],
    )
    indexed = {
        (row["trajectory"], int(row["domain_seed"]), row["method"]): row
        for row in rows
    }
    case_keys = sorted(
        {(row["trajectory"], int(row["domain_seed"])) for row in rows}
    )
    pairwise = {}
    for baseline in sorted(baselines):
        improvements = []
        wins = []
        global_multipliers = []
        violation_improvements = []
        for key in case_keys:
            proposed = indexed[key + (primary,)]
            control = indexed[key + (baseline,)]
            proposed_auc = float(proposed["task_auc_normalized"])
            control_auc = float(control["task_auc_normalized"])
            improvements.append(
                100.0 * (control_auc - proposed_auc)
                / max(control_auc, 1.0e-12)
            )
            wins.append(proposed_auc < control_auc)
            global_multipliers.append(
                float(proposed["final_global_ratio"])
                / max(float(control["final_global_ratio"]), 1.0e-12)
            )
            violation_improvements.append(
                float(control["final_violation_rate"])
                - float(proposed["final_violation_rate"])
            )
        pairwise[baseline] = {
            "paired_cases": len(case_keys),
            "median_auc_improvement_percent": float(
                np.median(improvements)
            ),
            "win_rate": float(np.mean(wins)),
            "median_global_ratio_multiplier": float(
                np.median(global_multipliers)
            ),
            "median_final_violation_rate_reduction": float(
                np.median(violation_improvements)
            ),
        }
    comparison = pairwise[strongest]
    success_rate = float(summary_by_method[primary]["success_rate"])
    criteria = {
        "median_auc_improvement_at_least_10_percent": (
            comparison["median_auc_improvement_percent"] >= 10.0
        ),
        "paired_win_rate_at_least_60_percent": (
            comparison["win_rate"] >= 0.60
        ),
        "success_rate_at_least_95_percent": success_rate >= 0.95,
        "median_global_tradeoff_no_more_than_1_5x": (
            comparison["median_global_ratio_multiplier"] <= 1.50
        ),
    }
    passed = bool(all(criteria.values()))
    return {
        "primary_method": primary,
        "strongest_baseline": strongest,
        "paired_cases": len(case_keys),
        "median_task_auc_improvement_percent": float(
            comparison["median_auc_improvement_percent"]
        ),
        "paired_win_rate": float(comparison["win_rate"]),
        "primary_success_rate": success_rate,
        "median_global_ratio_multiplier_vs_baseline": float(
            comparison["median_global_ratio_multiplier"]
        ),
        "median_final_violation_rate_reduction": float(
            comparison["median_final_violation_rate_reduction"]
        ),
        "criteria": criteria,
        "pairwise_vs_baselines": pairwise,
        "passed": passed,
        "decision": "GO" if passed else "REVISE",
    }


def _plot(
    output_path: Path,
    rows: Sequence[Dict[str, object]],
    summary: Sequence[Dict[str, object]],
    gate: Dict[str, object],
) -> None:
    labels = {
        "full_trajectory": "Full",
        "static_tolerance": "Static tolerance",
        "curvature_zones": "Curvature",
        "jerk_zones": "Jerk",
        "error_peak_dynamic": "Error peak",
        "violation_dynamic": "Violation",
        "random_zones": "Random",
        "nominal_lookahead_dynamic": "Lookahead",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))
    grouped = [
        [
            row["task_auc_normalized"]
            for row in rows
            if row["method"] == method
        ]
        for method in TASK_METHODS
    ]
    box = axes[0, 0].boxplot(grouped, patch_artist=True, showfliers=False)
    for method, patch in zip(TASK_METHODS, box["boxes"]):
        patch.set_facecolor(
            "tab:orange"
            if method == "nominal_lookahead_dynamic"
            else "tab:blue"
        )
        patch.set_alpha(0.50)
    axes[0, 0].set_xticklabels(
        [labels[method] for method in TASK_METHODS],
        rotation=28,
        ha="right",
    )
    axes[0, 0].set_ylabel("Normalized task-error AUC ↓")
    axes[0, 0].set_title("Externally specified machining-task zones")
    axes[0, 0].grid(axis="y", alpha=0.25)

    summary_by_method = {row["method"]: row for row in summary}
    for method in TASK_METHODS:
        item = summary_by_method[method]
        axes[0, 1].scatter(
            item["median_final_global_ratio"],
            item["median_final_task_ratio"],
            s=80,
            marker="s" if method == "nominal_lookahead_dynamic" else "o",
        )
        axes[0, 1].annotate(
            labels[method],
            (
                item["median_final_global_ratio"],
                item["median_final_task_ratio"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0, 1].set_xlabel("Median final global-error ratio ↓")
    axes[0, 1].set_ylabel("Median final task-error ratio ↓")
    axes[0, 1].set_title("Task/global trade-off")
    axes[0, 1].grid(alpha=0.25)

    baseline = gate["strongest_baseline"]
    indexed = {
        (row["trajectory"], row["domain_seed"], row["method"]): row
        for row in rows
    }
    keys = sorted(
        {(row["trajectory"], row["domain_seed"]) for row in rows}
    )
    improvements = []
    for key in keys:
        proposed = indexed[key + ("nominal_lookahead_dynamic",)]
        control = indexed[key + (baseline,)]
        improvements.append(
            100.0
            * (
                control["task_auc_normalized"]
                - proposed["task_auc_normalized"]
            )
            / control["task_auc_normalized"]
        )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].bar(
        np.arange(len(improvements)),
        improvements,
        color=["tab:green" if value > 0 else "tab:red" for value in improvements],
    )
    axes[1, 0].set_xlabel("Paired trajectory–machine case")
    axes[1, 0].set_ylabel("AUC improvement vs strongest baseline [%]")
    axes[1, 0].set_title("Nominal lookahead paired improvements")
    axes[1, 0].grid(axis="y", alpha=0.25)

    values = [
        gate["median_task_auc_improvement_percent"] / 10.0,
        gate["paired_win_rate"] / 0.60,
        gate["primary_success_rate"] / 0.95,
        1.50 / max(gate["median_global_ratio_multiplier_vs_baseline"], 1.0e-12),
    ]
    axes[1, 1].bar(
        np.arange(4),
        values,
        color=["tab:green" if value >= 1.0 else "tab:red" for value in values],
    )
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 1].set_xticks(np.arange(4))
    axes[1, 1].set_xticklabels(
        ["Median AUC", "Win rate", "Success", "Global trade-off"]
    )
    axes[1, 1].set_ylabel("Criterion ratio; pass ≥ 1")
    axes[1, 1].set_title("Task-definition gate: " + gate["decision"])
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    output_path: Path,
    settings: BenchmarkSettings,
    gate: Dict[str, object],
    summary: Sequence[Dict[str, object]],
) -> None:
    primary = {
        row["method"]: row for row in summary
    }["nominal_lookahead_dynamic"]
    text = """# Externally Specified Machining-Task Gate

## Protocol

- Critical zones are fixed from normalized programmed-path progress and process labels.
- Zone tolerances are fixed before any virtual-machine feedback is observed.
- Six task zones compete for a three-zone learning budget.
- The primary method enumerates zone combinations with a mismatched nominal model and reselects after every ILC trial.
- Confirmation machine seeds: {seeds}
- Paired cases: {cases}
- ILC updates per run: {iterations}

## Result

- Decision: {decision}
- Strongest baseline: {baseline}
- Median task-AUC improvement: {improvement:.2f}%
- Paired win rate: {win_rate:.2%}
- Constrained-run success: {success:.2%}
- Median global trade-off multiplier: {global_multiplier:.3f}x
- Median final violation-rate reduction: {violation_reduction:.2%}
- Primary median task AUC: {auc:.4f}
- Primary median final task-error ratio: {task_ratio:.4f}
- Primary median final tolerance-violation rate: {violation_rate:.2%}

## Interpretation

This gate tests task-aware allocation rather than rediscovering measured error
peaks. Evaluation labels and tolerances are external to the virtual-machine
tracking error. A GO decision therefore supports the revised research question;
a REVISE decision means the allocation method still needs improvement.
""".format(
        seeds=", ".join(str(seed) for seed in settings.domain_seeds),
        cases=len(TRAJECTORY_FAMILIES) * len(settings.domain_seeds),
        iterations=settings.iterations,
        decision=gate["decision"],
        baseline=gate["strongest_baseline"],
        improvement=gate["median_task_auc_improvement_percent"],
        win_rate=gate["paired_win_rate"],
        success=gate["primary_success_rate"],
        global_multiplier=gate["median_global_ratio_multiplier_vs_baseline"],
        violation_reduction=gate["median_final_violation_rate_reduction"],
        auc=primary["median_task_auc"],
        task_ratio=primary["median_final_task_ratio"],
        violation_rate=primary["median_final_violation_rate"],
    )
    output_path.write_text(text, encoding="utf-8")


def run_task_definition_gate(
    output_directory: Path,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    """Run the frozen task-definition confirmation experiment."""

    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    task_specs = {}
    for trajectory_index, family in enumerate(TRAJECTORY_FAMILIES):
        reference = make_trajectory_family(
            family,
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(
            samples=settings.samples,
            control_points=settings.control_points,
        )
        specification = make_task_specification(
            reference,
            settings.half_width,
        )
        task_specs[family] = {
            "zone_names": list(specification.names),
            "zone_centers": list(specification.centers),
            "zone_windows": [list(window) for window in specification.windows],
            "zone_tolerances_mm": list(specification.tolerances),
        }
        for domain_seed in settings.domain_seeds:
            for method in TASK_METHODS:
                row = run_task_method(
                    method,
                    reference,
                    basis,
                    specification,
                    int(domain_seed),
                    settings,
                    random_seed=(
                        30000
                        + 100 * trajectory_index
                        + int(domain_seed)
                    ),
                )
                row["trajectory"] = family
                rows.append(row)

    column_order = [
        "trajectory",
        "domain_seed",
        "method",
        "initial_task_nrmse",
        "final_task_nrmse",
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
        "constraint_violation",
        "all_updates_succeeded",
        "elapsed_s",
    ]
    ordered_rows = [
        {column: row[column] for column in column_order} for row in rows
    ]
    summary = _aggregate(ordered_rows)
    gate = _evaluate_gate(ordered_rows, summary)
    payload = {
        "settings": {
            "trajectories": list(TRAJECTORY_FAMILIES),
            "domain_seeds": list(settings.domain_seeds),
            "samples": settings.samples,
            "iterations": settings.iterations,
            "zone_budget": settings.number_of_windows,
            "task_zones_per_trajectory": len(TASK_ZONE_NAMES),
            "total_method_runs": len(ordered_rows),
        },
        "task_specifications": task_specs,
        "gate": gate,
        "summary": summary,
    }
    _write_csv(output_directory / "task_effectiveness_raw.csv", ordered_rows)
    _write_csv(output_directory / "task_effectiveness_summary.csv", summary)
    (output_directory / "task_effectiveness_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _plot(
        output_directory / "task_effectiveness_summary.png",
        ordered_rows,
        summary,
        gate,
    )
    _write_report(
        output_directory / "task_effectiveness_report.md",
        settings,
        gate,
        summary,
    )
    return payload
