"""V13 rolling confidence-adaptive delay compensation experiment."""

import json
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import trapezoid

from .benchmark import BenchmarkSettings
from .delay_compensation_runner import (
    axis_delay_aligned_sensitivity,
    estimate_effective_delay_steps,
)
from .ilc import build_contour_sensitivity, solve_constrained_update
from .metrics import constraint_report, task_errors
from .plant import make_virtual_machine_domain, nominal_config, simulate_machine
from .semantic_task_benchmark import (
    SemanticTaskSpecification,
    _balanced_anchor_weights,
    _dual_anchor_selection,
    _semantic_config,
    _semantic_metrics,
    _trust_limited_candidate,
)
from .trajectory import ReferenceTrajectory


V11_GAIN = 0.25
ROLLING_WINDOW = 2
MIN_ADAPTIVE_GAIN = 0.15
MAX_ADAPTIVE_GAIN = 0.38
MAX_APPLIED_LAG_STEPS = 3.5
ADAPTIVE_METHODS = (
    "dual_anchor_dynamic",
    "fixed_delay_2",
    "delay_aware_dual_anchor",
    "adaptive_rolling_delay",
    "oracle_adaptive_delay",
)


@dataclass(frozen=True)
class BalancedDelayScenario:
    scenario_id: str
    label: str
    mode: str


BALANCED_STATIC = BalancedDelayScenario(
    scenario_id="balanced_static",
    label="Balanced low/high and asymmetric static delays",
    mode="static",
)
BALANCED_SLOW_DRIFT = BalancedDelayScenario(
    scenario_id="balanced_slow_drift",
    label="Balanced rising, falling and asymmetric slow drift",
    mode="slow_drift",
)
BALANCED_SWITCH = BalancedDelayScenario(
    scenario_id="balanced_switch",
    label="Balanced abrupt and asymmetric delay switches",
    mode="switch",
)
BALANCED_DELAY_SCENARIOS = (
    BALANCED_STATIC,
    BALANCED_SLOW_DRIFT,
    BALANCED_SWITCH,
)


def scenario_to_dict(scenario: BalancedDelayScenario) -> Dict[str, object]:
    return {
        "scenario_id": scenario.scenario_id,
        "label": scenario.label,
        "mode": scenario.mode,
    }


def balanced_axis_delay_schedule(
    schedule_slot: int,
    scenario: BalancedDelayScenario,
    trials: int,
) -> np.ndarray:
    """Return a fixed five-trial total axis-delay schedule for one slot."""

    if schedule_slot not in range(4):
        raise ValueError("balanced schedule slot must be in [0, 3]")
    if trials != 5:
        raise ValueError("V13 balanced schedules require exactly five trials")
    if scenario.mode == "static":
        pairs = ((2, 2), (2, 10), (10, 2), (10, 10))
        return np.tile(np.asarray(pairs[schedule_slot], dtype=int), (trials, 1))
    if scenario.mode == "slow_drift":
        schedules = (
            ((2, 2), (4, 4), (6, 6), (8, 8), (10, 10)),
            ((10, 10), (8, 8), (6, 6), (4, 4), (2, 2)),
            ((2, 10), (4, 8), (6, 6), (8, 4), (10, 2)),
            ((4, 8), (5, 7), (6, 6), (7, 5), (8, 4)),
        )
        return np.asarray(schedules[schedule_slot], dtype=int)
    if scenario.mode == "switch":
        schedules = (
            ((2, 2), (2, 2), (10, 10), (10, 10), (10, 10)),
            ((10, 10), (10, 10), (2, 2), (2, 2), (2, 2)),
            ((2, 10), (2, 10), (10, 2), (10, 2), (10, 2)),
            ((4, 8), (4, 8), (8, 4), (8, 4), (4, 8)),
        )
        return np.asarray(schedules[schedule_slot], dtype=int)
    raise ValueError("unknown balanced delay mode: " + scenario.mode)


def _plant_with_total_delay(
    domain_seed: int,
    total_axis_delay: Tuple[int, int],
):
    base = make_virtual_machine_domain(domain_seed)
    return replace(
        base,
        x_axis=replace(base.x_axis, delay_steps=int(total_axis_delay[0])),
        y_axis=replace(base.y_axis, delay_steps=int(total_axis_delay[1])),
    )


def _axis_peak_margins(estimate: Dict[str, object]) -> Tuple[float, float]:
    output = []
    for scores in estimate["axis_correlation_scores"]:
        ordered = np.sort(np.asarray(scores, dtype=float))
        output.append(float(ordered[-1] - ordered[-2]))
    return float(output[0]), float(output[1])


def adaptive_applied_lag(
    residual_lag_history: List[Tuple[int, int]],
    axis_peak_correlations: Tuple[float, float],
    axis_peak_margins: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Compute rolling tracked lag, confidence and nonlinear shrinkage."""

    recent = residual_lag_history[-ROLLING_WINDOW:]
    applied: List[float] = []
    confidences: List[float] = []
    gains: List[float] = []
    for axis in range(2):
        current = float(recent[-1][axis])
        if len(recent) == 1:
            tracked = current
        else:
            previous = float(recent[-2][axis])
            trend = float(np.clip(current - previous, -1.0, 1.0))
            tracked = max(0.0, 0.70 * current + 0.30 * previous + 0.50 * trend)
        correlation_confidence = float(
            np.clip((axis_peak_correlations[axis] - 0.98) / 0.02, 0.0, 1.0)
        )
        margin_confidence = float(
            np.clip(axis_peak_margins[axis] / 5.0e-4, 0.0, 1.0)
        )
        confidence = 0.50 * correlation_confidence + 0.50 * margin_confidence
        target_gain = float(
            np.clip(
                MIN_ADAPTIVE_GAIN + 0.025 * tracked,
                MIN_ADAPTIVE_GAIN,
                MAX_ADAPTIVE_GAIN,
            )
        )
        gain = MIN_ADAPTIVE_GAIN + confidence * (
            target_gain - MIN_ADAPTIVE_GAIN
        )
        applied.append(
            float(np.clip(gain * tracked, 0.0, MAX_APPLIED_LAG_STEPS))
        )
        confidences.append(confidence)
        gains.append(gain)
    return (
        (applied[0], applied[1]),
        (confidences[0], confidences[1]),
        (gains[0], gains[1]),
    )


def _oracle_adaptive_lag(
    actual_axis_delay: Tuple[int, int],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    applied = []
    gains = []
    for delay in actual_axis_delay:
        gain = float(
            np.clip(
                MIN_ADAPTIVE_GAIN + 0.025 * float(delay),
                MIN_ADAPTIVE_GAIN,
                MAX_ADAPTIVE_GAIN,
            )
        )
        applied.append(
            float(
                np.clip(
                    gain * float(delay),
                    0.0,
                    MAX_APPLIED_LAG_STEPS,
                )
            )
        )
        gains.append(gain)
    return (applied[0], applied[1]), (gains[0], gains[1])


def run_adaptive_delay_method(
    method: str,
    reference: ReferenceTrajectory,
    basis: np.ndarray,
    specification: SemanticTaskSpecification,
    plant_seed: int,
    schedule_slot: int,
    settings: BenchmarkSettings,
    scenario: BalancedDelayScenario,
) -> Dict[str, object]:
    """Run one V13 comparator on a balanced total-delay schedule."""

    if method not in ADAPTIVE_METHODS:
        raise ValueError("unknown V13 method: " + method)
    config = _semantic_config(reference, settings)
    nominal_sensitivity = build_contour_sensitivity(
        reference,
        basis,
        nominal_config(),
    )
    delay_schedule = balanced_axis_delay_schedule(
        schedule_slot,
        scenario,
        config.iterations + 1,
    )
    current_command = reference.position.copy()
    true_metrics: List[Dict[str, object]] = []
    measured_scores: List[float] = []
    selections: List[Tuple[int, ...]] = []
    solver_status: List[bool] = []
    acceptance_history: List[bool] = []
    residual_lag_history: List[Tuple[int, int]] = []
    total_lag_history: List[Tuple[int, int]] = []
    nominal_lag_history: List[Tuple[int, int]] = []
    applied_lag_history: List[Tuple[float, float]] = []
    confidence_history: List[Tuple[float, float]] = []
    gain_history: List[Tuple[float, float]] = []
    lag_errors: List[float] = []
    correlation_history: List[float] = []
    trust_radius = 4.0
    accepted_command = None
    accepted_measured_error = None
    accepted_measured_score = None
    accepted_true_metric = None
    rejected_trials = 0

    start_time = time.perf_counter()
    for trial in range(config.iterations + 1):
        actual_axis_delay = tuple(int(item) for item in delay_schedule[trial])
        plant = _plant_with_total_delay(plant_seed, actual_axis_delay)
        true_feedback = simulate_machine(current_command, reference.dt, plant)
        measured_feedback = true_feedback
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
        lag_errors.extend(
            abs(float(estimate) - float(actual))
            for estimate, actual in zip(residual_lag, actual_axis_delay)
        )
        correlation_history.append(
            float(measured_estimate["peak_correlation"])
        )
        axis_correlations = tuple(
            float(item) for item in measured_estimate["axis_peak_correlations"]
        )
        axis_margins = _axis_peak_margins(measured_estimate)

        if method == "adaptive_rolling_delay":
            applied_lag, confidence, gain = adaptive_applied_lag(
                residual_lag_history,
                axis_correlations,
                axis_margins,
            )
        elif method == "oracle_adaptive_delay":
            applied_lag, gain = _oracle_adaptive_lag(actual_axis_delay)
            confidence = (1.0, 1.0)
        elif method == "delay_aware_dual_anchor":
            applied_lag = tuple(
                V11_GAIN
                * float(
                    np.median(
                        [item[axis] for item in residual_lag_history]
                    )
                )
                for axis in range(2)
            )
            confidence = (float("nan"), float("nan"))
            gain = (V11_GAIN, V11_GAIN)
        elif method == "fixed_delay_2":
            applied_lag = (2.0, 2.0)
            confidence = (float("nan"), float("nan"))
            gain = (1.0, 1.0)
        else:
            applied_lag = (0.0, 0.0)
            confidence = (float("nan"), float("nan"))
            gain = (0.0, 0.0)
        applied_lag_history.append(applied_lag)
        confidence_history.append(confidence)
        gain_history.append(gain)

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
    return {
        "method": method,
        "domain_seed": int(plant_seed),
        "schedule_slot": int(schedule_slot),
        "scenario_id": scenario.scenario_id,
        "scenario_mode": scenario.mode,
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
        "actual_axis_delay_history": json.dumps(delay_schedule.tolist()),
        "raw_estimated_lag_history": json.dumps(residual_lag_history),
        "total_estimated_lag_history": json.dumps(total_lag_history),
        "nominal_estimated_lag_history": json.dumps(nominal_lag_history),
        "applied_lag_history": json.dumps(applied_lag_history),
        "confidence_history": json.dumps(confidence_history),
        "gain_history": json.dumps(gain_history),
        "lag_absolute_error_steps": float(np.median(lag_errors)),
        "median_peak_correlation": float(np.median(correlation_history)),
        "elapsed_s": float(elapsed),
    }
