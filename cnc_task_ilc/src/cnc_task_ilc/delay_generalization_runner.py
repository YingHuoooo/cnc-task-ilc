"""V12 unknown and drifting-delay runner built on the frozen V11 method."""

import json
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import trapezoid

from .basis import apply_axis_coefficients
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


ONLINE_COMPENSATION_GAIN = 0.25
FIXED_DELAY_CANDIDATES = (2, 4, 6)
GENERALIZATION_METHODS = (
    "dual_anchor_dynamic",
    "fixed_delay_2",
    "fixed_delay_4",
    "fixed_delay_6",
    "delay_aware_dual_anchor",
    "oracle_true_delay_0p25",
)


@dataclass(frozen=True)
class DelayGeneralizationScenario:
    scenario_id: str
    label: str
    mode: str
    maximum_extra_delay_steps: int = 8
    drift_span_steps: int = 2


UNKNOWN_STATIC = DelayGeneralizationScenario(
    scenario_id="unknown_static_0_8",
    label="Unknown static axis delay in [0, 8] steps",
    mode="static",
)
UNKNOWN_DRIFT = DelayGeneralizationScenario(
    scenario_id="unknown_drift_pm2",
    label="Unknown axis delay with slow +/-2-step trial drift",
    mode="drift",
)
GENERALIZATION_SCENARIOS = (UNKNOWN_STATIC, UNKNOWN_DRIFT)


def scenario_to_dict(
    scenario: DelayGeneralizationScenario,
) -> Dict[str, object]:
    return {
        "scenario_id": scenario.scenario_id,
        "label": scenario.label,
        "mode": scenario.mode,
        "maximum_extra_delay_steps": scenario.maximum_extra_delay_steps,
        "drift_span_steps": scenario.drift_span_steps,
    }


def make_extra_delay_schedule(
    domain_seed: int,
    scenario: DelayGeneralizationScenario,
    trials: int,
) -> np.ndarray:
    """Create a deterministic paired x/y extra-delay schedule."""

    if trials < 2:
        raise ValueError("delay schedule requires at least two trials")
    salt = 41000 if scenario.mode == "static" else 53000
    random = np.random.RandomState(salt + int(domain_seed))
    if scenario.mode == "static":
        base = random.randint(
            0,
            scenario.maximum_extra_delay_steps + 1,
            size=2,
        )
        return np.tile(base, (trials, 1)).astype(int)
    if scenario.mode != "drift":
        raise ValueError("unknown delay-schedule mode: " + scenario.mode)
    lower = scenario.drift_span_steps
    upper = scenario.maximum_extra_delay_steps - scenario.drift_span_steps + 1
    base = random.randint(lower, upper, size=2)
    direction = random.choice((-1, 1), size=2)
    phase = np.arange(trials)
    triangular = np.minimum(phase, trials - 1 - phase)
    triangular = np.minimum(triangular, scenario.drift_span_steps)
    schedule = base[None, :] + triangular[:, None] * direction[None, :]
    return np.clip(
        schedule,
        0,
        scenario.maximum_extra_delay_steps,
    ).astype(int)


def _plant_with_extra_delay(
    domain_seed: int,
    extra_axis_delay: Tuple[int, int],
):
    base = make_virtual_machine_domain(domain_seed)
    return replace(
        base,
        x_axis=replace(
            base.x_axis,
            delay_steps=base.x_axis.delay_steps + int(extra_axis_delay[0]),
        ),
        y_axis=replace(
            base.y_axis,
            delay_steps=base.y_axis.delay_steps + int(extra_axis_delay[1]),
        ),
    )


def _fixed_lag(method: str) -> Tuple[float, float]:
    if not method.startswith("fixed_delay_"):
        raise ValueError("not a fixed-delay method: " + method)
    lag = int(method.rsplit("_", 1)[1])
    if lag not in FIXED_DELAY_CANDIDATES:
        raise ValueError("undeclared fixed delay: " + str(lag))
    return float(lag), float(lag)


def run_delay_generalization_method(
    method: str,
    reference: ReferenceTrajectory,
    basis: np.ndarray,
    specification: SemanticTaskSpecification,
    plant_seed: int,
    settings: BenchmarkSettings,
    scenario: DelayGeneralizationScenario,
) -> Dict[str, object]:
    """Run a paired V12 method under unknown static or drifting delay."""

    if method not in GENERALIZATION_METHODS:
        raise ValueError("unknown V12 method: " + method)
    config = _semantic_config(reference, settings)
    nominal_sensitivity = build_contour_sensitivity(
        reference,
        basis,
        nominal_config(),
    )
    extra_schedule = make_extra_delay_schedule(
        plant_seed,
        scenario,
        config.iterations + 1,
    )
    base_plant = make_virtual_machine_domain(plant_seed)
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
    actual_delay_history: List[Tuple[int, int]] = []
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
        extra_axis = tuple(int(item) for item in extra_schedule[trial])
        plant = _plant_with_extra_delay(plant_seed, extra_axis)
        actual_axis_delay = (
            int(plant.x_axis.delay_steps),
            int(plant.y_axis.delay_steps),
        )
        actual_delay_history.append(actual_axis_delay)
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

        if method == "delay_aware_dual_anchor":
            applied_lag = tuple(
                ONLINE_COMPENSATION_GAIN
                * float(
                    np.median(
                        [item[axis] for item in residual_lag_history]
                    )
                )
                for axis in range(2)
            )
        elif method == "oracle_true_delay_0p25":
            applied_lag = tuple(
                ONLINE_COMPENSATION_GAIN * float(item)
                for item in actual_axis_delay
            )
        elif method.startswith("fixed_delay_"):
            applied_lag = _fixed_lag(method)
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
    return {
        "method": method,
        "domain_seed": int(plant_seed),
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
        "base_axis_delay_steps": json.dumps(
            [base_plant.x_axis.delay_steps, base_plant.y_axis.delay_steps]
        ),
        "extra_delay_schedule": json.dumps(extra_schedule.tolist()),
        "actual_axis_delay_history": json.dumps(actual_delay_history),
        "raw_estimated_lag_history": json.dumps(residual_lag_history),
        "total_estimated_lag_history": json.dumps(total_lag_history),
        "nominal_estimated_lag_history": json.dumps(nominal_lag_history),
        "applied_lag_history": json.dumps(applied_lag_history),
        "lag_absolute_error_steps": float(np.median(lag_errors)),
        "median_peak_correlation": float(np.median(correlation_history)),
        "elapsed_s": float(elapsed),
    }
