"""Semantic machining-task benchmark with tolerance-aware safe ILC.

This V6 benchmark replaces the shared fixed path fractions used in V5 with
trajectory-specific zones generated only from programmed-path geometry.  The
task state is a small vector of zone-level RMS, peak and local ripple ratios.
No measured machine error is used to define the zones or their tolerances.
"""

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from .basis import apply_axis_coefficients, cubic_bspline_basis
from .benchmark import BenchmarkSettings, TRAJECTORY_FAMILIES
from .ilc import ILCConfig, build_contour_sensitivity, solve_constrained_update
from .metrics import command_kinematics, constraint_report, task_errors
from .plant import make_virtual_machine_domain, nominal_config, simulate_machine
from .task_benchmark import _reference_jerk, _zone_statistic
from .trajectory import ReferenceTrajectory, make_trajectory_family


SEMANTIC_METHODS = (
    "full_trajectory",
    "static_tolerance",
    "curvature_zones",
    "jerk_zones",
    "error_peak_dynamic",
    "violation_dynamic",
    "violation_safe",
    "random_zones",
)

# Kept separate so the frozen V6 method list and its published aggregation do
# not change when the V8 scheduler is added.
ADDITIONAL_SEMANTIC_METHODS = ("dual_anchor_dynamic",)

ROLE_TOLERANCES_MM = {
    "entry_datum": 0.28,
    "functional_extreme": 0.18,
    "curvature_feature": 0.20,
    "blend_transition": 0.23,
    "motion_transition": 0.25,
    "exit_datum": 0.30,
}


@dataclass(frozen=True)
class SemanticTaskSpecification:
    names: Tuple[str, ...]
    roles: Tuple[str, ...]
    centers: Tuple[int, ...]
    windows: Tuple[Tuple[int, int], ...]
    tolerances: Tuple[float, ...]
    evaluation_mask: np.ndarray
    generation_rules: Tuple[str, ...]


def _robust_unit_scale(values: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(values, dtype=float))
    lower, upper = np.percentile(values, (5.0, 95.0))
    if upper - lower < 1.0e-12:
        return np.zeros_like(values)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.5)


def _choose_separated_peak(
    score: np.ndarray,
    eligible: np.ndarray,
    selected: Sequence[int],
    minimum_separation: int,
) -> int:
    order = np.argsort(np.asarray(score, dtype=float))[::-1]
    for candidate in order:
        candidate = int(candidate)
        if not eligible[candidate]:
            continue
        if all(abs(candidate - prior) >= minimum_separation for prior in selected):
            return candidate
    raise RuntimeError("unable to place a separated semantic task zone")


def make_semantic_task_specification(
    reference: ReferenceTrajectory,
    family: str,
    half_width: int,
) -> SemanticTaskSpecification:
    """Generate task zones from the programmed path, never machine feedback."""

    if family not in TRAJECTORY_FAMILIES:
        raise ValueError("unknown trajectory family: " + family)
    samples = reference.time.size
    if samples < 12 * half_width + 20:
        raise ValueError("trajectory resolution is too low for six task zones")

    entry = int(round(0.07 * (samples - 1)))
    exit_point = int(round(0.93 * (samples - 1)))
    selected = [entry, exit_point]
    separation = 2 * half_width + 3
    eligible = np.zeros(samples, dtype=bool)
    eligible[
        int(round(0.14 * (samples - 1))) :
        int(round(0.86 * (samples - 1))) + 1
    ] = True

    centered_position = reference.position - np.mean(reference.position, axis=0)
    feature_scores = (
        (
            "functional_extreme",
            _robust_unit_scale(np.linalg.norm(centered_position, axis=1)),
            "maximum programmed radial extent",
        ),
        (
            "curvature_feature",
            _robust_unit_scale(reference.curvature),
            "maximum programmed curvature",
        ),
        (
            "blend_transition",
            _robust_unit_scale(np.gradient(reference.curvature)),
            "maximum programmed curvature change",
        ),
        (
            "motion_transition",
            _robust_unit_scale(_reference_jerk(reference)),
            "maximum programmed jerk",
        ),
    )

    zone_items = [
        (entry, "entry_datum", "program entry datum"),
        (exit_point, "exit_datum", "program exit datum"),
    ]
    for role, score, rule in feature_scores:
        center = _choose_separated_peak(
            score,
            eligible,
            selected,
            separation,
        )
        selected.append(center)
        zone_items.append((center, role, rule))

    zone_items.sort(key=lambda item: item[0])
    mask = np.zeros(samples, dtype=bool)
    names = []
    roles = []
    centers = []
    windows = []
    tolerances = []
    rules = []
    role_counts: Dict[str, int] = {}
    for center, role, rule in zone_items:
        start = int(center - half_width)
        stop = int(center + half_width)
        if start < 0 or stop >= samples:
            raise RuntimeError("semantic task zone exceeds path boundary")
        mask[start : stop + 1] = True
        role_counts[role] = role_counts.get(role, 0) + 1
        suffix = role_counts[role]
        names.append(role if suffix == 1 else role + "_" + str(suffix))
        roles.append(role)
        centers.append(int(center))
        windows.append((start, stop))
        tolerances.append(float(ROLE_TOLERANCES_MM[role]))
        rules.append(rule)

    return SemanticTaskSpecification(
        names=tuple(names),
        roles=tuple(roles),
        centers=tuple(centers),
        windows=tuple(windows),
        tolerances=tuple(tolerances),
        evaluation_mask=mask,
        generation_rules=tuple(rules),
    )


def zone_quality_ratios(
    contour_error: np.ndarray,
    specification: SemanticTaskSpecification,
) -> np.ndarray:
    """Return a tolerance-normalized task-state magnitude for every zone."""

    ratios = []
    for (start, stop), tolerance in zip(
        specification.windows,
        specification.tolerances,
    ):
        local = np.asarray(contour_error[start : stop + 1], dtype=float)
        rms_ratio = float(np.sqrt(np.mean(local**2)) / tolerance)
        peak_ratio = float(np.max(np.abs(local)) / (1.8 * tolerance))
        phase = np.linspace(-1.0, 1.0, local.size)
        trend = np.polyval(np.polyfit(phase, local, deg=1), phase)
        ripple = local - trend
        ripple_ratio = float(
            np.sqrt(np.mean(ripple**2)) / (0.45 * tolerance)
        )
        ratios.append(
            float(
                np.sqrt(
                    0.55 * rms_ratio**2
                    + 0.30 * peak_ratio**2
                    + 0.15 * ripple_ratio**2
                )
            )
        )
    return np.asarray(ratios, dtype=float)


def _semantic_metrics(
    contour_error: np.ndarray,
    specification: SemanticTaskSpecification,
) -> Dict[str, object]:
    ratios = zone_quality_ratios(contour_error, specification)
    zone_rms = float(np.sqrt(np.mean(ratios**2)))
    worst_zone = float(np.max(ratios))
    return {
        # A functional part fails when even one specified surface remains out
        # of tolerance, so the primary task score cannot be a pure average.
        "task_score": float(0.50 * zone_rms + 0.50 * worst_zone),
        "task_zone_rms": zone_rms,
        "task_worst_zone_ratio": worst_zone,
        "task_violation_rate": float(np.mean(ratios > 1.0)),
        "task_max_ratio": worst_zone,
        "global_rmse": float(np.sqrt(np.mean(contour_error**2))),
        "zone_quality_ratios": ratios,
    }


def _semantic_config(
    reference: ReferenceTrajectory,
    settings: BenchmarkSettings,
) -> ILCConfig:
    kinematics = command_kinematics(reference.position, reference.dt)
    return ILCConfig(
        iterations=settings.iterations,
        correction_limit=4.0,
        velocity_limit=1.42 * float(np.max(np.abs(kinematics["velocity"]))),
        acceleration_limit=1.80
        * float(np.max(np.abs(kinematics["acceleration"]))),
        regularization=3.0e-3,
        smoothness=2.0e-8,
        learning_rate=0.65,
        global_protection_weight=0.30,
        critical_boost=5.0,
        solver_max_iterations=240,
    )


def _zone_mask(
    specification: SemanticTaskSpecification,
    selected_zones: Sequence[int],
) -> np.ndarray:
    mask = np.zeros_like(specification.evaluation_mask)
    for index in selected_zones:
        start, stop = specification.windows[int(index)]
        mask[start : stop + 1] = True
    return mask


def _fixed_selection(
    method: str,
    reference: ReferenceTrajectory,
    specification: SemanticTaskSpecification,
    budget: int,
    random_seed: int,
) -> Tuple[int, ...]:
    if method == "static_tolerance":
        score = -np.asarray(specification.tolerances)
    elif method == "curvature_zones":
        score = _zone_statistic(reference.curvature, specification, "mean")
    elif method == "jerk_zones":
        score = _zone_statistic(
            _reference_jerk(reference), specification, "mean"
        )
    elif method == "random_zones":
        score = np.random.RandomState(random_seed).uniform(
            size=len(specification.names)
        )
    else:
        raise ValueError("method does not have a fixed selection")
    return tuple(sorted(int(index) for index in np.argsort(score)[-budget:]))


def _dynamic_selection(
    method: str,
    contour_error: np.ndarray,
    specification: SemanticTaskSpecification,
    budget: int,
) -> Tuple[int, ...]:
    if method == "error_peak_dynamic":
        score = _zone_statistic(contour_error, specification, "max")
    elif method in ("violation_dynamic", "violation_safe"):
        ratios = zone_quality_ratios(contour_error, specification)
        score = ratios + 0.50 * np.maximum(ratios - 1.0, 0.0)
    else:
        raise ValueError("method does not have a dynamic selection")
    return tuple(sorted(int(index) for index in np.argsort(score)[-budget:]))


def _dual_anchor_selection(
    contour_error: np.ndarray,
    specification: SemanticTaskSpecification,
    budget: int,
) -> Tuple[int, ...]:
    """Reserve one zone for task urgency and one for raw error magnitude.

    The tolerance-normalized anchor represents the functional specification.
    The raw-error anchor is a feedback-only proxy for where the virtual plant
    currently offers a large reducible error.  They must be distinct so a
    strict-tolerance zone cannot consume the whole two-zone budget twice.
    """

    if budget != 2:
        raise ValueError("dual-anchor scheduling requires a two-zone budget")
    ratios = zone_quality_ratios(contour_error, specification)
    urgency = ratios + 0.50 * np.maximum(ratios - 1.0, 0.0)
    task_anchor = int(np.argmax(urgency))

    raw_error = _zone_statistic(contour_error, specification, "max")
    error_anchor = next(
        int(index)
        for index in np.argsort(raw_error)[::-1]
        if int(index) != task_anchor
    )
    return tuple(sorted((task_anchor, error_anchor)))


def _task_weights(
    contour_error: np.ndarray,
    specification: SemanticTaskSpecification,
    selected_zones: Sequence[int],
    config: ILCConfig,
) -> np.ndarray:
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


def _balanced_anchor_weights(
    contour_error: np.ndarray,
    specification: SemanticTaskSpecification,
    selected_zones: Sequence[int],
    config: ILCConfig,
) -> np.ndarray:
    """Weight both V8 anchors equally after they have been selected."""

    weights = np.full(
        contour_error.shape,
        config.global_protection_weight,
        dtype=float,
    )
    for index in selected_zones:
        start, stop = specification.windows[int(index)]
        weights[start : stop + 1] += config.critical_boost
    weights /= np.mean(weights)
    return weights


def _trust_limited_candidate(
    current_command: np.ndarray,
    basis: np.ndarray,
    scaled_delta: np.ndarray,
    trust_radius: float,
) -> Tuple[np.ndarray, float]:
    candidate = apply_axis_coefficients(current_command, basis, scaled_delta)
    maximum_step = float(np.max(np.abs(candidate - current_command)))
    if maximum_step > trust_radius:
        scaled_delta = scaled_delta * (trust_radius / maximum_step)
        candidate = apply_axis_coefficients(current_command, basis, scaled_delta)
        maximum_step = float(np.max(np.abs(candidate - current_command)))
    return candidate, maximum_step


def run_semantic_task_method(
    method: str,
    reference: ReferenceTrajectory,
    basis: np.ndarray,
    specification: SemanticTaskSpecification,
    plant_seed: int,
    settings: BenchmarkSettings,
    random_seed: int,
) -> Dict[str, object]:
    """Run one method with identical task, update and trial budgets."""

    if method not in SEMANTIC_METHODS + ADDITIONAL_SEMANTIC_METHODS:
        raise ValueError("unknown semantic task method: " + method)
    safe = method in ("violation_safe", "dual_anchor_dynamic")
    config = _semantic_config(reference, settings)
    sensitivity = build_contour_sensitivity(reference, basis, nominal_config())
    plant = make_virtual_machine_domain(plant_seed)
    current_command = reference.position.copy()
    metrics: List[Dict[str, object]] = []
    selections: List[Tuple[int, ...]] = []
    solver_status: List[bool] = []
    acceptance_history: List[bool] = []
    # The initial radius does not clip ordinary updates.  It becomes
    # restrictive only after an observed trial degrades the accepted task.
    trust_radius = 4.00
    accepted_command = None
    accepted_error = None
    accepted_metric = None
    rejected_trials = 0

    fixed_selection = None
    if method in (
        "static_tolerance",
        "curvature_zones",
        "jerk_zones",
        "random_zones",
    ):
        fixed_selection = _fixed_selection(
            method,
            reference,
            specification,
            settings.number_of_windows,
            random_seed,
        )

    start_time = time.perf_counter()
    for trial in range(config.iterations + 1):
        feedback = simulate_machine(current_command, reference.dt, plant)
        contour_error = task_errors(reference, feedback)["contour"]
        metric = _semantic_metrics(contour_error, specification)
        metric["trial"] = float(trial)
        metrics.append(metric)

        base_command = current_command
        base_error = contour_error
        accepted = True
        if safe:
            if accepted_metric is None or float(metric["task_score"]) <= float(
                accepted_metric["task_score"]
            ):
                accepted_command = current_command.copy()
                accepted_error = contour_error.copy()
                accepted_metric = dict(metric)
                if trial > 0:
                    trust_radius = min(4.00, 1.15 * trust_radius)
            else:
                accepted = False
                rejected_trials += 1
                trust_radius = max(0.15, 0.50 * trust_radius)
                base_command = np.asarray(accepted_command).copy()
                base_error = np.asarray(accepted_error).copy()
                current_command = base_command.copy()
        else:
            accepted_command = current_command.copy()
            accepted_error = contour_error.copy()
            accepted_metric = dict(metric)
        acceptance_history.append(accepted)

        if trial == config.iterations:
            break

        if method == "full_trajectory":
            selection = tuple(range(len(specification.names)))
            weights = np.ones_like(base_error)
        else:
            if fixed_selection is not None:
                selection = fixed_selection
            elif method == "dual_anchor_dynamic":
                selection = _dual_anchor_selection(
                    base_error,
                    specification,
                    settings.number_of_windows,
                )
            else:
                selection = _dynamic_selection(
                    method,
                    base_error,
                    specification,
                    settings.number_of_windows,
                )
            if method == "dual_anchor_dynamic":
                weights = _balanced_anchor_weights(
                    base_error,
                    specification,
                    selection,
                    config,
                )
            else:
                weights = _task_weights(
                    base_error,
                    specification,
                    selection,
                    config,
                )
        update = solve_constrained_update(
            contour_error=base_error,
            sensitivity=sensitivity,
            weights=weights,
            basis=basis,
            initial_command=reference.position,
            current_command=base_command,
            dt=reference.dt,
            config=config,
        )
        selections.append(tuple(selection))
        solver_status.append(bool(update["success"]))
        scaled_delta = config.learning_rate * update["delta"]
        if safe:
            current_command, _ = _trust_limited_candidate(
                base_command,
                basis,
                scaled_delta,
                trust_radius,
            )
        else:
            current_command = apply_axis_coefficients(
                base_command,
                basis,
                scaled_delta,
            )
    elapsed = time.perf_counter() - start_time

    task_values = np.asarray(
        [float(metric["task_score"]) for metric in metrics], dtype=float
    )
    normalized = task_values / max(task_values[0], 1.0e-12)
    auc = float(trapezoid(normalized, dx=1.0) / config.iterations)
    final_metric = dict(accepted_metric)
    final_command = np.asarray(accepted_command)
    constraints = constraint_report(
        initial_command=reference.position,
        learned_command=final_command,
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
        "initial_task_score": float(metrics[0]["task_score"]),
        "final_task_score": float(final_metric["task_score"]),
        "last_observed_task_score": float(metrics[-1]["task_score"]),
        "task_auc_normalized": auc,
        "final_task_ratio": float(
            float(final_metric["task_score"])
            / max(float(metrics[0]["task_score"]), 1.0e-12)
        ),
        "initial_violation_rate": float(metrics[0]["task_violation_rate"]),
        "final_violation_rate": float(final_metric["task_violation_rate"]),
        "initial_task_max_ratio": float(metrics[0]["task_max_ratio"]),
        "final_task_max_ratio": float(final_metric["task_max_ratio"]),
        "initial_global_rmse": float(metrics[0]["global_rmse"]),
        "final_global_rmse": float(final_metric["global_rmse"]),
        "final_global_ratio": float(
            float(final_metric["global_rmse"])
            / max(float(metrics[0]["global_rmse"]), 1.0e-12)
        ),
        "selection_switches": int(selection_switches),
        "selection_history": json.dumps(selections),
        "accepted_history": json.dumps(acceptance_history),
        "rejected_trials": int(rejected_trials),
        "final_trust_radius_mm": float(trust_radius if safe else 0.0),
        "constraint_violation": int(constraints["constraint_violation"]),
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
    for method in SEMANTIC_METHODS:
        selected = [row for row in rows if row["method"] == method]
        summary.append(
            {
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
                "median_elapsed_s": float(
                    np.median([row["elapsed_s"] for row in selected])
                ),
            }
        )
    return summary


def _paired_improvement(
    rows: Sequence[Dict[str, object]],
    primary: str,
    comparator: str,
) -> Dict[str, float]:
    indexed = {
        (str(row["trajectory"]), int(row["domain_seed"]), str(row["method"])): row
        for row in rows
    }
    keys = sorted({(key[0], key[1]) for key in indexed})
    proposed = np.asarray(
        [float(indexed[key + (primary,)]["task_auc_normalized"]) for key in keys]
    )
    baseline = np.asarray(
        [float(indexed[key + (comparator,)]["task_auc_normalized"]) for key in keys]
    )
    improvement = 100.0 * (baseline - proposed) / baseline
    return {
        "paired_cases": int(improvement.size),
        "median_auc_improvement_percent": float(np.median(improvement)),
        "mean_auc_improvement_percent": float(np.mean(improvement)),
        "win_rate": float(np.mean(proposed < baseline)),
        "tie_rate": float(np.mean(np.isclose(proposed, baseline))),
    }


def _evaluate_gate(
    rows: Sequence[Dict[str, object]],
    summary: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    primary = "violation_safe"
    comparisons = {
        method: _paired_improvement(rows, primary, method)
        for method in SEMANTIC_METHODS
        if method != primary
    }
    primary_summary = next(item for item in summary if item["method"] == primary)
    criteria = {
        "median_improvement_vs_full_at_least_10_percent": (
            comparisons["full_trajectory"]["median_auc_improvement_percent"]
            >= 10.0
        ),
        "win_rate_vs_error_peak_at_least_60_percent": (
            comparisons["error_peak_dynamic"]["win_rate"] >= 0.60
        ),
        "not_worse_than_plain_violation_scheduler": (
            comparisons["violation_dynamic"]["median_auc_improvement_percent"]
            >= 0.0
        ),
        "success_rate_at_least_95_percent": (
            primary_summary["success_rate"] >= 0.95
        ),
    }
    return {
        "primary_method": primary,
        "comparisons": comparisons,
        "primary_success_rate": float(primary_summary["success_rate"]),
        "criteria": criteria,
        "passed": bool(all(criteria.values())),
        "decision": "PASS" if all(criteria.values()) else "REVISE",
    }


def _plot(
    output_path: Path,
    rows: Sequence[Dict[str, object]],
    summary: Sequence[Dict[str, object]],
    gate: Dict[str, object],
) -> None:
    labels = {
        "full_trajectory": "Full",
        "static_tolerance": "Static tol.",
        "curvature_zones": "Curvature",
        "jerk_zones": "Jerk",
        "error_peak_dynamic": "Error peak",
        "violation_dynamic": "Violation",
        "violation_safe": "Safe violation",
        "random_zones": "Random",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    x = np.arange(len(summary))
    auc = [item["median_task_auc"] for item in summary]
    colors = ["#2f6b8a" if item["method"] == "violation_safe" else "#a9b6bd" for item in summary]
    axes[0, 0].bar(x, auc, color=colors)
    axes[0, 0].set_xticks(x, [labels[item["method"]] for item in summary], rotation=35, ha="right")
    axes[0, 0].set_ylabel("Median normalized task AUC")
    axes[0, 0].set_title("Trial-efficient semantic task convergence")

    axes[0, 1].scatter(
        [item["median_final_global_ratio"] for item in summary],
        [item["median_final_task_ratio"] for item in summary],
        c=colors,
        s=55,
    )
    annotation_offsets = {
        "full_trajectory": (5, 4),
        "static_tolerance": (5, 6),
        "curvature_zones": (5, 14),
        "jerk_zones": (5, 7),
        "error_peak_dynamic": (5, 3),
        "violation_dynamic": (-82, -4),
        "violation_safe": (5, 9),
        "random_zones": (5, 7),
    }
    for item in summary:
        axes[0, 1].annotate(
            labels[item["method"]],
            (item["median_final_global_ratio"], item["median_final_task_ratio"]),
            xytext=annotation_offsets[item["method"]],
            textcoords="offset points",
            fontsize=8,
        )
    axes[0, 1].set_xlabel("Median final global-error ratio")
    axes[0, 1].set_ylabel("Median final semantic-task ratio")
    axes[0, 1].set_title("Task/global trade-off")

    indexed = {
        (str(row["trajectory"]), int(row["domain_seed"]), str(row["method"])): row
        for row in rows
    }
    keys = sorted({(key[0], key[1]) for key in indexed})
    plain = np.asarray(
        [float(indexed[key + ("violation_dynamic",)]["task_auc_normalized"]) for key in keys]
    )
    safe = np.asarray(
        [float(indexed[key + ("violation_safe",)]["task_auc_normalized"]) for key in keys]
    )
    axes[1, 0].scatter(plain, safe, color="#2f6b8a", alpha=0.8)
    limits = [min(plain.min(), safe.min()), max(plain.max(), safe.max())]
    axes[1, 0].plot(limits, limits, color="#666666", linewidth=1.0)
    axes[1, 0].set_xlabel("Plain violation AUC")
    axes[1, 0].set_ylabel("Safe violation AUC")
    axes[1, 0].set_title("Effect of trust limit and rollback")

    primary_rows = [row for row in rows if row["method"] == "violation_safe"]
    families = list(TRAJECTORY_FAMILIES)
    rejected = [
        np.mean(
            [
                float(row["rejected_trials"])
                for row in primary_rows
                if row["trajectory"] == family
            ]
        )
        for family in families
    ]
    axes[1, 1].bar(np.arange(len(families)), rejected, color="#6d8f5f")
    axes[1, 1].set_xticks(np.arange(len(families)), families, rotation=30, ha="right")
    axes[1, 1].set_ylabel("Mean rejected trials")
    axes[1, 1].set_title("Rollback activity by trajectory")

    fig.suptitle("V6 semantic task-level ILC: " + str(gate["decision"]))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output_path: Path,
    settings: BenchmarkSettings,
    gate: Dict[str, object],
    summary: Sequence[Dict[str, object]],
) -> None:
    primary = next(item for item in summary if item["method"] == "violation_safe")
    comparisons = gate["comparisons"]
    text = f"""# V6 Semantic Task-Level ILC Confirmation

## Scope

- Trajectories: {len(TRAJECTORY_FAMILIES)}
- Virtual-machine domains: {len(settings.domain_seeds)}
- Methods: {len(SEMANTIC_METHODS)}
- Total method runs: {len(TRAJECTORY_FAMILIES) * len(settings.domain_seeds) * len(SEMANTIC_METHODS)}
- ILC updates per case: {settings.iterations}
- Active-zone budget: {settings.number_of_windows} of 6

## Frozen decision

- Primary method: `violation_safe`
- Decision: **{gate['decision']}**
- Median AUC improvement vs full trajectory: {comparisons['full_trajectory']['median_auc_improvement_percent']:.2f}%
- Win rate vs dynamic error peak: {100.0 * comparisons['error_peak_dynamic']['win_rate']:.2f}%
- Median AUC improvement vs plain violation scheduling: {comparisons['violation_dynamic']['median_auc_improvement_percent']:.2f}%
- Primary success rate: {100.0 * primary['success_rate']:.2f}%
- Median rejected trials: {primary['median_rejected_trials']:.2f}

Task zones are generated from programmed-path geometry before any virtual-machine
feedback is observed.  A REVISE decision means either the semantic scheduler or
the safety layer did not meet every frozen criterion.
"""
    output_path.write_text(text, encoding="utf-8")


def run_semantic_task_gate(
    output_directory: Path,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    """Run the V6 confirmation on frozen, previously unseen machine domains."""

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
        specification = make_semantic_task_specification(
            reference,
            family,
            settings.half_width,
        )
        task_specs[family] = {
            "zone_names": list(specification.names),
            "zone_roles": list(specification.roles),
            "zone_centers": list(specification.centers),
            "zone_windows": [list(window) for window in specification.windows],
            "zone_tolerances_mm": list(specification.tolerances),
            "generation_rules": list(specification.generation_rules),
        }
        for domain_seed in settings.domain_seeds:
            for method in SEMANTIC_METHODS:
                row = run_semantic_task_method(
                    method,
                    reference,
                    basis,
                    specification,
                    int(domain_seed),
                    settings,
                    random_seed=40000 + 100 * trajectory_index + int(domain_seed),
                )
                row["trajectory"] = family
                rows.append(row)

    column_order = [
        "trajectory",
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
    gate = _evaluate_gate(ordered_rows, summary)
    payload = {
        "settings": {
            "trajectories": list(TRAJECTORY_FAMILIES),
            "domain_seeds": list(settings.domain_seeds),
            "samples": settings.samples,
            "iterations": settings.iterations,
            "zone_budget": settings.number_of_windows,
            "task_zones_per_trajectory": 6,
            "total_method_runs": len(ordered_rows),
            "task_state": (
                "zone RMS + peak + local ripple normalized by tolerance; "
                "primary score is 50% zone RMS and 50% worst-zone ratio"
            ),
        },
        "task_specifications": task_specs,
        "gate": gate,
        "summary": summary,
    }
    _write_csv(output_directory / "semantic_effectiveness_raw.csv", ordered_rows)
    _write_csv(output_directory / "semantic_effectiveness_summary.csv", summary)
    (output_directory / "semantic_effectiveness_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot(
        output_directory / "semantic_effectiveness_summary.png",
        ordered_rows,
        summary,
        gate,
    )
    _write_report(
        output_directory / "semantic_effectiveness_report.md",
        settings,
        gate,
        summary,
    )
    return payload
