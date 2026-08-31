"""Shared runner for the five post-V11 numerical experiments.

This module intentionally lives outside ``cnc_task_ilc``.  It reuses the
frozen numerical model and optimizer but adds only the instrumentation and
matched switches required by the new experiments.  No LinuxCNC or physical
machine interface is imported.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import trapezoid

from cnc_task_ilc.basis import apply_axis_coefficients
from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.delay_compensation_runner import (
    axis_delay_aligned_sensitivity,
    estimate_effective_delay_steps,
)
from cnc_task_ilc.ilc import build_contour_sensitivity, solve_constrained_update
from cnc_task_ilc.metrics import constraint_report, task_errors
from cnc_task_ilc.plant import (
    AxisDynamics,
    VirtualPlantConfig,
    make_virtual_machine_domain,
    nominal_config,
    simulate_machine,
)
from cnc_task_ilc.robustness_runner import StressScenario, _measurement_noise
from cnc_task_ilc.semantic_task_benchmark import (
    SemanticTaskSpecification,
    _balanced_anchor_weights,
    _dual_anchor_selection,
    _semantic_config,
    _semantic_metrics,
    _trust_limited_candidate,
    _zone_statistic,
    zone_quality_ratios,
)
from cnc_task_ilc.trajectory import ReferenceTrajectory


MATCHED_METHODS = (
    "v11_full",
    "no_residual_alignment",
    "task_top2",
    "raw_top2",
    "uniform_full_trajectory",
)


def scenario_from_dict(payload: Mapping[str, object]) -> StressScenario:
    return StressScenario(
        scenario_id=str(payload["scenario_id"]),
        label=str(payload["label"]),
        factor=str(payload["factor"]),
        factor_level=float(payload["factor_level"]),
        measurement_noise_std_mm=float(payload["measurement_noise_std_mm"]),
        extra_delay_steps=int(payload["extra_delay_steps"]),
        mismatch_scale=float(payload["mismatch_scale"]),
    )


def scenario_dict(scenario: StressScenario) -> Dict[str, object]:
    return asdict(scenario)


def plant_to_dict(plant: VirtualPlantConfig) -> Dict[str, float]:
    return {
        "x_natural_frequency": float(plant.x_axis.natural_frequency),
        "x_damping_ratio": float(plant.x_axis.damping_ratio),
        "x_delay_steps": int(plant.x_axis.delay_steps),
        "x_friction": float(plant.x_axis.friction),
        "x_velocity_scale": float(plant.x_axis.velocity_scale),
        "y_natural_frequency": float(plant.y_axis.natural_frequency),
        "y_damping_ratio": float(plant.y_axis.damping_ratio),
        "y_delay_steps": int(plant.y_axis.delay_steps),
        "y_friction": float(plant.y_axis.friction),
        "y_velocity_scale": float(plant.y_axis.velocity_scale),
        "acceleration_limit": float(plant.acceleration_limit),
        "velocity_limit": float(plant.velocity_limit),
        "cross_coupling": float(plant.cross_coupling),
        "repeatable_disturbance": float(plant.repeatable_disturbance),
    }


def plant_from_dict(payload: Mapping[str, object]) -> VirtualPlantConfig:
    return VirtualPlantConfig(
        x_axis=AxisDynamics(
            natural_frequency=float(payload["x_natural_frequency"]),
            damping_ratio=float(payload["x_damping_ratio"]),
            delay_steps=int(payload["x_delay_steps"]),
            friction=float(payload["x_friction"]),
            velocity_scale=float(payload["x_velocity_scale"]),
        ),
        y_axis=AxisDynamics(
            natural_frequency=float(payload["y_natural_frequency"]),
            damping_ratio=float(payload["y_damping_ratio"]),
            delay_steps=int(payload["y_delay_steps"]),
            friction=float(payload["y_friction"]),
            velocity_scale=float(payload["y_velocity_scale"]),
        ),
        acceleration_limit=float(payload["acceleration_limit"]),
        velocity_limit=float(payload["velocity_limit"]),
        cross_coupling=float(payload["cross_coupling"]),
        repeatable_disturbance=float(payload["repeatable_disturbance"]),
    )


def stress_plant(base: VirtualPlantConfig, scenario: StressScenario) -> VirtualPlantConfig:
    """Apply the frozen V9/V10 stress transformation to an explicit base plant."""

    nominal = nominal_config()

    def axis_stress(axis: AxisDynamics, nominal_axis: AxisDynamics) -> AxisDynamics:
        scale = scenario.mismatch_scale
        return AxisDynamics(
            natural_frequency=float(
                max(
                    5.0,
                    nominal_axis.natural_frequency
                    + scale * (axis.natural_frequency - nominal_axis.natural_frequency),
                )
            ),
            damping_ratio=float(
                max(
                    0.25,
                    nominal_axis.damping_ratio
                    + scale * (axis.damping_ratio - nominal_axis.damping_ratio),
                )
            ),
            delay_steps=int(axis.delay_steps + scenario.extra_delay_steps),
            friction=float(axis.friction * scale),
            velocity_scale=float(axis.velocity_scale),
        )

    return VirtualPlantConfig(
        x_axis=axis_stress(base.x_axis, nominal.x_axis),
        y_axis=axis_stress(base.y_axis, nominal.y_axis),
        acceleration_limit=float(base.acceleration_limit),
        velocity_limit=float(base.velocity_limit),
        cross_coupling=float(base.cross_coupling * scenario.mismatch_scale),
        repeatable_disturbance=float(
            base.repeatable_disturbance * scenario.mismatch_scale
        ),
    )


def base_plant_from_job(job: Mapping[str, object]) -> VirtualPlantConfig:
    if job["plant_kind"] == "seed":
        return make_virtual_machine_domain(int(job["plant_seed"]))
    if job["plant_kind"] == "explicit":
        return plant_from_dict(job["plant_parameters"])
    raise ValueError("unknown plant kind")


def matched_selection(
    method: str,
    contour_error: np.ndarray,
    specification: SemanticTaskSpecification,
) -> Tuple[int, ...]:
    """Select two zones while changing only the information used for ranking."""

    if method in ("v11_full", "no_residual_alignment"):
        return _dual_anchor_selection(contour_error, specification, 2)
    if method == "task_top2":
        ratios = zone_quality_ratios(contour_error, specification)
        urgency = ratios + 0.50 * np.maximum(ratios - 1.0, 0.0)
        return tuple(sorted(int(index) for index in np.argsort(urgency)[-2:]))
    if method == "raw_top2":
        raw_peak = _zone_statistic(contour_error, specification, "max")
        return tuple(sorted(int(index) for index in np.argsort(raw_peak)[-2:]))
    if method == "uniform_full_trajectory":
        return tuple(range(len(specification.names)))
    raise ValueError("unknown matched method: " + method)


def run_matched_method(
    method: str,
    reference: ReferenceTrajectory,
    basis: np.ndarray,
    specification: SemanticTaskSpecification,
    base_plant: VirtualPlantConfig,
    settings: BenchmarkSettings,
    scenario: StressScenario,
    noise_seed: int,
    compensation_gain: float = 0.25,
    smoothing_window: int = 5,
    learning_rate: float = 0.65,
    nominal_sensitivity: Optional[np.ndarray] = None,
    return_trace: bool = False,
) -> Tuple[Dict[str, object], Optional[Dict[str, object]]]:
    """Run a matched V11-centered method with optional full pointwise trace."""

    if method not in MATCHED_METHODS:
        raise ValueError("unknown method: " + method)
    if not 0.0 <= compensation_gain <= 1.0:
        raise ValueError("compensation gain must be in [0, 1]")
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd integer")

    config = replace(_semantic_config(reference, settings), learning_rate=learning_rate)
    if nominal_sensitivity is None:
        nominal_sensitivity = build_contour_sensitivity(reference, basis, nominal_config())
    plant = stress_plant(base_plant, scenario)
    current_command = reference.position.copy()
    true_metrics: List[Dict[str, object]] = []
    measured_scores: List[float] = []
    selections: List[Tuple[int, ...]] = []
    solver_status: List[bool] = []
    acceptance_history: List[bool] = []
    total_lag_history: List[Tuple[int, int]] = []
    nominal_lag_history: List[Tuple[int, int]] = []
    residual_lag_history: List[Tuple[int, int]] = []
    applied_lag_history: List[Tuple[float, float]] = []
    correlation_history: List[float] = []
    trust_radius = 4.0
    accepted_command = None
    accepted_feedback = None
    accepted_true_error = None
    accepted_measured_error = None
    accepted_measured_score = None
    accepted_true_metric = None
    rejected_trials = 0
    trace_commands: List[np.ndarray] = []
    trace_feedbacks: List[np.ndarray] = []
    trace_errors: List[np.ndarray] = []

    start_time = time.perf_counter()
    for trial in range(config.iterations + 1):
        true_feedback = simulate_machine(current_command, reference.dt, plant)
        measured_feedback = true_feedback + _measurement_noise(
            true_feedback.shape,
            scenario.measurement_noise_std_mm,
            noise_seed,
            trial,
        )
        true_error = task_errors(reference, true_feedback)["contour"]
        measured_error = task_errors(reference, measured_feedback)["contour"]
        true_metric = _semantic_metrics(true_error, specification)
        measured_metric = _semantic_metrics(measured_error, specification)
        true_metric["trial"] = float(trial)
        true_metrics.append(true_metric)
        measured_scores.append(float(measured_metric["task_score"]))
        if return_trace:
            trace_commands.append(current_command.copy())
            trace_feedbacks.append(true_feedback.copy())
            trace_errors.append(true_error.copy())

        measured_estimate = estimate_effective_delay_steps(
            current_command,
            measured_feedback,
            smoothing_window=smoothing_window,
        )
        nominal_feedback = simulate_machine(current_command, reference.dt, nominal_config())
        nominal_estimate = estimate_effective_delay_steps(
            current_command,
            nominal_feedback,
            smoothing_window=smoothing_window,
        )
        total_lag = tuple(int(item) for item in measured_estimate["axis_lag_steps"])
        nominal_lag = tuple(int(item) for item in nominal_estimate["axis_lag_steps"])
        residual_lag = tuple(
            max(0, total - nominal) for total, nominal in zip(total_lag, nominal_lag)
        )
        total_lag_history.append(total_lag)
        nominal_lag_history.append(nominal_lag)
        residual_lag_history.append(residual_lag)
        correlation_history.append(float(measured_estimate["peak_correlation"]))

        effective_gain = 0.0 if method == "no_residual_alignment" else compensation_gain
        applied_lag = tuple(
            effective_gain
            * float(np.median([item[axis] for item in residual_lag_history]))
            for axis in range(2)
        )
        applied_lag_history.append(applied_lag)

        base_command = current_command
        base_error = measured_error
        accepted = True
        if (
            accepted_measured_score is None
            or float(measured_metric["task_score"]) <= float(accepted_measured_score)
        ):
            accepted_command = current_command.copy()
            accepted_feedback = true_feedback.copy()
            accepted_true_error = true_error.copy()
            accepted_measured_error = measured_error.copy()
            accepted_measured_score = float(measured_metric["task_score"])
            accepted_true_metric = dict(true_metric)
            if trial > 0:
                trust_radius = min(4.0, 1.15 * trust_radius)
        else:
            accepted = False
            rejected_trials += 1
            trust_radius = max(0.15, 0.50 * trust_radius)
            base_command = np.asarray(accepted_command).copy()
            base_error = np.asarray(accepted_measured_error).copy()
            current_command = base_command.copy()
        acceptance_history.append(accepted)

        if trial == config.iterations:
            break

        selection = matched_selection(method, base_error, specification)
        if method == "uniform_full_trajectory":
            weights = np.ones_like(base_error)
        else:
            weights = _balanced_anchor_weights(
                base_error,
                specification,
                selection,
                config,
            )
        sensitivity = axis_delay_aligned_sensitivity(
            nominal_sensitivity,
            applied_lag,
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
        current_command, _ = _trust_limited_candidate(
            base_command,
            basis,
            scaled_delta,
            trust_radius,
        )

    elapsed = time.perf_counter() - start_time
    task_values = np.asarray(
        [float(metric["task_score"]) for metric in true_metrics], dtype=float
    )
    normalized = task_values / max(task_values[0], 1.0e-12)
    auc = float(trapezoid(normalized, dx=1.0) / config.iterations)
    final_metric = dict(accepted_true_metric)
    final_command = np.asarray(accepted_command)
    constraints = constraint_report(
        initial_command=reference.position,
        learned_command=final_command,
        dt=reference.dt,
        max_correction=config.correction_limit,
        velocity_limit=config.velocity_limit,
        acceleration_limit=config.acceleration_limit,
    )
    median_axis_lag = tuple(
        float(np.median([item[axis] for item in residual_lag_history]))
        for axis in range(2)
    )
    configured_axis_delay = (
        float(plant.x_axis.delay_steps),
        float(plant.y_axis.delay_steps),
    )
    summary: Dict[str, object] = {
        "method": method,
        "compensation_gain": float(compensation_gain),
        "smoothing_window": int(smoothing_window),
        "learning_rate": float(learning_rate),
        "scenario_id": scenario.scenario_id,
        "measurement_noise_std_mm": float(scenario.measurement_noise_std_mm),
        "extra_delay_steps": int(scenario.extra_delay_steps),
        "mismatch_scale": float(scenario.mismatch_scale),
        "initial_task_score": float(true_metrics[0]["task_score"]),
        "final_task_score": float(final_metric["task_score"]),
        "task_auc_normalized": auc,
        "final_task_ratio": float(
            float(final_metric["task_score"])
            / max(float(true_metrics[0]["task_score"]), 1.0e-12)
        ),
        "initial_violation_rate": float(true_metrics[0]["task_violation_rate"]),
        "final_violation_rate": float(final_metric["task_violation_rate"]),
        "initial_global_rmse": float(true_metrics[0]["global_rmse"]),
        "final_global_rmse": float(final_metric["global_rmse"]),
        "final_global_ratio": float(
            float(final_metric["global_rmse"])
            / max(float(true_metrics[0]["global_rmse"]), 1.0e-12)
        ),
        "trial_task_scores": json.dumps([float(item) for item in task_values]),
        "selection_history": json.dumps(selections),
        "accepted_history": json.dumps(acceptance_history),
        "rejected_trials": int(rejected_trials),
        "final_trust_radius_mm": float(trust_radius),
        "constraint_violation": int(constraints["constraint_violation"]),
        "all_updates_succeeded": int(all(solver_status)),
        "finite_result": int(
            np.all(np.isfinite(task_values))
            and np.isfinite(auc)
            and np.all(np.isfinite(final_command))
        ),
        "raw_estimated_lag_history": json.dumps(residual_lag_history),
        "total_estimated_lag_history": json.dumps(total_lag_history),
        "nominal_estimated_lag_history": json.dumps(nominal_lag_history),
        "applied_lag_history": json.dumps(applied_lag_history),
        "median_estimated_axis_lag_steps": json.dumps(median_axis_lag),
        "lag_absolute_error_steps": float(
            np.mean(
                np.abs(
                    np.asarray(median_axis_lag) - np.asarray(configured_axis_delay)
                )
            )
        ),
        "median_peak_correlation": float(np.median(correlation_history)),
        "elapsed_s": float(elapsed),
    }
    trace: Optional[Dict[str, object]] = None
    if return_trace:
        trace = {
            "reference": reference.position.copy(),
            "time": reference.time.copy(),
            "commands": np.asarray(trace_commands),
            "feedbacks": np.asarray(trace_feedbacks),
            "contour_errors": np.asarray(trace_errors),
            "accepted_command": np.asarray(accepted_command).copy(),
            "accepted_feedback": np.asarray(accepted_feedback).copy(),
            "accepted_contour_error": np.asarray(accepted_true_error).copy(),
            "selection_history": selections,
            "acceptance_history": acceptance_history,
            "total_lag_history": total_lag_history,
            "nominal_lag_history": nominal_lag_history,
            "residual_lag_history": residual_lag_history,
            "applied_lag_history": applied_lag_history,
        }
    return summary, trace

