"""Online delay estimation and sensitivity alignment for dual-anchor ILC."""

import json
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import trapezoid

from .basis import apply_axis_coefficients
from .benchmark import BenchmarkSettings
from .ilc import build_contour_sensitivity, solve_constrained_update
from .metrics import constraint_report, task_errors
from .plant import nominal_config, simulate_machine
from .robustness_runner import (
    StressScenario,
    _measurement_noise,
    make_stressed_virtual_machine,
)
from .semantic_task_benchmark import (
    SemanticTaskSpecification,
    _balanced_anchor_weights,
    _dual_anchor_selection,
    _semantic_config,
    _semantic_metrics,
    _trust_limited_candidate,
)
from .trajectory import ReferenceTrajectory


DELAY_COMPENSATION_METHODS = (
    "dual_anchor_dynamic",
    "fixed_delay_dual_anchor",
    "delay_aware_dual_anchor",
)
MAX_ESTIMATED_LAG_STEPS = 14
VELOCITY_SMOOTHING_WINDOW = 5
FIXED_DELAY_STEPS = 4


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    left_padding = (window - 1) // 2
    right_padding = window - 1 - left_padding
    return np.column_stack(
        [
            np.convolve(
                np.pad(
                    values[:, axis],
                    (left_padding, right_padding),
                    mode="edge",
                ),
                kernel,
                mode="valid",
            )
            for axis in range(2)
        ]
    )


def estimate_effective_delay_steps(
    command: np.ndarray,
    measured_feedback: np.ndarray,
    max_lag_steps: int = MAX_ESTIMATED_LAG_STEPS,
    smoothing_window: int = VELOCITY_SMOOTHING_WINDOW,
) -> Dict[str, object]:
    """Estimate batch-trial lag from command/feedback velocity correlation."""

    command = np.asarray(command, dtype=float)
    measured_feedback = np.asarray(measured_feedback, dtype=float)
    if command.shape != measured_feedback.shape or command.ndim != 2:
        raise ValueError("command and feedback must have equal (samples, 2) shape")
    if max_lag_steps < 0 or max_lag_steps >= command.shape[0] - 4:
        raise ValueError("invalid maximum delay search range")
    smoothed_command = _moving_average(command, smoothing_window)
    smoothed_feedback = _moving_average(measured_feedback, smoothing_window)
    command_velocity = np.diff(smoothed_command, axis=0)
    feedback_velocity = np.diff(smoothed_feedback, axis=0)
    axis_scores = [[], []]
    for lag in range(max_lag_steps + 1):
        if lag:
            command_slice = command_velocity[:-lag]
            feedback_slice = feedback_velocity[lag:]
        else:
            command_slice = command_velocity
            feedback_slice = feedback_velocity
        command_slice = command_slice - np.mean(command_slice, axis=0)
        feedback_slice = feedback_slice - np.mean(feedback_slice, axis=0)
        for axis in range(2):
            denominator = float(
                np.linalg.norm(command_slice[:, axis])
                * np.linalg.norm(feedback_slice[:, axis])
            )
            score = (
                float(
                    np.sum(
                        command_slice[:, axis]
                        * feedback_slice[:, axis]
                    )
                    / denominator
                )
                if denominator > 1.0e-12
                else -1.0
            )
            axis_scores[axis].append(score)
    scores = np.mean(np.asarray(axis_scores, dtype=float), axis=0)
    best_lag = int(np.argmax(scores))
    ordered = np.sort(np.asarray(scores, dtype=float))
    margin = float(ordered[-1] - ordered[-2]) if ordered.size > 1 else 0.0
    axis_lags = [int(np.argmax(item)) for item in axis_scores]
    return {
        "lag_steps": best_lag,
        "axis_lag_steps": axis_lags,
        "peak_correlation": float(scores[best_lag]),
        "axis_peak_correlations": [
            float(axis_scores[axis][axis_lags[axis]])
            for axis in range(2)
        ],
        "peak_margin": margin,
        "correlation_scores": [float(item) for item in scores],
        "axis_correlation_scores": [
            [float(value) for value in item] for item in axis_scores
        ],
    }


def delay_aligned_sensitivity(
    nominal_sensitivity: np.ndarray,
    lag_steps: int,
) -> np.ndarray:
    """Map earlier command corrections to their later delayed output rows."""

    nominal_sensitivity = np.asarray(nominal_sensitivity, dtype=float)
    if lag_steps < 0 or lag_steps >= nominal_sensitivity.shape[0]:
        raise ValueError("lag is outside sensitivity horizon")
    if lag_steps == 0:
        return nominal_sensitivity.copy()
    aligned = np.zeros_like(nominal_sensitivity)
    aligned[lag_steps:] = nominal_sensitivity[:-lag_steps]
    return aligned


def axis_delay_aligned_sensitivity(
    nominal_sensitivity: np.ndarray,
    axis_lag_steps: Tuple[float, float],
) -> np.ndarray:
    """Fractionally shift the x/y command sensitivity blocks independently."""

    nominal_sensitivity = np.asarray(nominal_sensitivity, dtype=float)
    if nominal_sensitivity.ndim != 2 or nominal_sensitivity.shape[1] % 2:
        raise ValueError("sensitivity must contain equal x/y column blocks")
    columns_per_axis = nominal_sensitivity.shape[1] // 2
    aligned = np.zeros_like(nominal_sensitivity)
    sample_index = np.arange(nominal_sensitivity.shape[0], dtype=float)
    for axis, lag_steps in enumerate(axis_lag_steps):
        if lag_steps < 0 or lag_steps >= nominal_sensitivity.shape[0]:
            raise ValueError("axis lag is outside sensitivity horizon")
        block = slice(axis * columns_per_axis, (axis + 1) * columns_per_axis)
        if lag_steps == 0:
            aligned[:, block] = nominal_sensitivity[:, block]
        else:
            source_index = sample_index - float(lag_steps)
            for column in range(block.start, block.stop):
                aligned[:, column] = np.interp(
                    source_index,
                    sample_index,
                    nominal_sensitivity[:, column],
                    left=0.0,
                )
    return aligned


def run_delay_compensated_method(
    method: str,
    reference: ReferenceTrajectory,
    basis: np.ndarray,
    specification: SemanticTaskSpecification,
    plant_seed: int,
    settings: BenchmarkSettings,
    scenario: StressScenario,
    noise_seed: int,
    compensation_gain: float = 1.0,
) -> Dict[str, object]:
    """Run dual-anchor ILC with no, fixed or online delay alignment."""

    if method not in DELAY_COMPENSATION_METHODS:
        raise ValueError("unknown delay-compensation method: " + method)
    if compensation_gain < 0.0 or compensation_gain > 1.0:
        raise ValueError("compensation gain must be in [0, 1]")
    config = _semantic_config(reference, settings)
    nominal_sensitivity = build_contour_sensitivity(
        reference,
        basis,
        nominal_config(),
    )
    plant = make_stressed_virtual_machine(plant_seed, scenario)
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
    accepted_measured_error = None
    accepted_measured_score = None
    accepted_true_metric = None
    rejected_trials = 0

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

        measured_estimate = estimate_effective_delay_steps(
            current_command,
            measured_feedback,
        )
        nominal_feedback = simulate_machine(
            current_command,
            reference.dt,
            nominal_config(),
        )
        nominal_estimate = estimate_effective_delay_steps(
            current_command,
            nominal_feedback,
        )
        total_lag = tuple(
            int(item) for item in measured_estimate["axis_lag_steps"]
        )
        nominal_lag = tuple(
            int(item) for item in nominal_estimate["axis_lag_steps"]
        )
        residual_lag = tuple(
            max(0, total - nominal)
            for total, nominal in zip(total_lag, nominal_lag)
        )
        total_lag_history.append(total_lag)
        nominal_lag_history.append(nominal_lag)
        residual_lag_history.append(residual_lag)
        correlation_history.append(
            float(measured_estimate["peak_correlation"])
        )
        if method == "delay_aware_dual_anchor":
            applied_lag = tuple(
                compensation_gain
                * float(
                    np.median(
                        [item[axis] for item in residual_lag_history]
                    )
                )
                for axis in range(2)
            )
        elif method == "fixed_delay_dual_anchor":
            applied_lag = (
                float(FIXED_DELAY_STEPS),
                float(FIXED_DELAY_STEPS),
            )
        else:
            applied_lag = (0.0, 0.0)
        applied_lag_history.append(applied_lag)

        base_command = current_command
        base_error = measured_error
        accepted = True
        if (
            accepted_measured_score is None
            or float(measured_metric["task_score"])
            <= float(accepted_measured_score)
        ):
            accepted_command = current_command.copy()
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

        selection = _dual_anchor_selection(
            base_error,
            specification,
            settings.number_of_windows,
        )
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
        [float(metric["task_score"]) for metric in true_metrics],
        dtype=float,
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
    configured_mean_delay = float(
        0.5 * (plant.x_axis.delay_steps + plant.y_axis.delay_steps)
    )
    median_axis_lag = tuple(
        float(np.median([item[axis] for item in residual_lag_history]))
        for axis in range(2)
    )
    median_estimated_lag = float(np.mean(median_axis_lag))
    configured_axis_delay = (
        float(plant.x_axis.delay_steps),
        float(plant.y_axis.delay_steps),
    )
    return {
        "method": method,
        "compensation_gain": float(compensation_gain),
        "domain_seed": int(plant_seed),
        "scenario_id": scenario.scenario_id,
        "stress_factor": scenario.factor,
        "stress_level": float(scenario.factor_level),
        "measurement_noise_std_mm": float(
            scenario.measurement_noise_std_mm
        ),
        "extra_delay_steps": int(scenario.extra_delay_steps),
        "mismatch_scale": float(scenario.mismatch_scale),
        "initial_task_score": float(true_metrics[0]["task_score"]),
        "final_task_score": float(final_metric["task_score"]),
        "last_observed_task_score": float(true_metrics[-1]["task_score"]),
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
        "measured_score_auc_normalized": float(
            trapezoid(
                np.asarray(measured_scores)
                / max(float(measured_scores[0]), 1.0e-12),
                dx=1.0,
            )
            / config.iterations
        ),
        "selection_switches": int(
            sum(
                first != second
                for first, second in zip(selections[:-1], selections[1:])
            )
        ),
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
        "median_estimated_lag_steps": median_estimated_lag,
        "median_estimated_axis_lag_steps": json.dumps(median_axis_lag),
        "configured_mean_delay_steps": configured_mean_delay,
        "lag_absolute_error_steps": float(
            np.mean(
                np.abs(
                    np.asarray(median_axis_lag)
                    - np.asarray(configured_axis_delay)
                )
            )
        ),
        "median_peak_correlation": float(np.median(correlation_history)),
        "elapsed_s": float(elapsed),
    }
