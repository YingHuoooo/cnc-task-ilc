"""Noise-, delay- and mismatch-aware runner for post-V8 robustness tests.

This module intentionally leaves the frozen V8 implementation unchanged.  It
reuses the V8 selection and weighting functions, but separates the unobserved
true plant error from the noisy error available to the learning algorithm.
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.integrate import trapezoid

from .basis import apply_axis_coefficients
from .benchmark import BenchmarkSettings
from .ilc import build_contour_sensitivity, solve_constrained_update
from .metrics import constraint_report, task_errors
from .plant import (
    AxisDynamics,
    VirtualPlantConfig,
    make_virtual_machine_domain,
    nominal_config,
    simulate_machine,
)
from .semantic_task_benchmark import (
    SemanticTaskSpecification,
    _balanced_anchor_weights,
    _dual_anchor_selection,
    _dynamic_selection,
    _semantic_config,
    _semantic_metrics,
    _task_weights,
    _trust_limited_candidate,
)
from .trajectory import ReferenceTrajectory


ROBUSTNESS_METHODS = (
    "full_trajectory",
    "error_peak_dynamic",
    "violation_safe",
    "dual_anchor_dynamic",
)


@dataclass(frozen=True)
class StressScenario:
    scenario_id: str
    label: str
    factor: str
    factor_level: float
    measurement_noise_std_mm: float = 0.0
    extra_delay_steps: int = 0
    mismatch_scale: float = 1.0


STRESS_SCENARIOS = (
    StressScenario("baseline", "Baseline", "baseline", 0.0),
    StressScenario(
        "noise_0p02",
        "Noise 0.02 mm",
        "measurement_noise",
        0.02,
        measurement_noise_std_mm=0.02,
    ),
    StressScenario(
        "noise_0p05",
        "Noise 0.05 mm",
        "measurement_noise",
        0.05,
        measurement_noise_std_mm=0.05,
    ),
    StressScenario(
        "delay_2",
        "Added delay 2 steps",
        "added_delay",
        2.0,
        extra_delay_steps=2,
    ),
    StressScenario(
        "delay_4",
        "Added delay 4 steps",
        "added_delay",
        4.0,
        extra_delay_steps=4,
    ),
    StressScenario(
        "mismatch_1p35",
        "Mismatch 1.35x",
        "dynamic_mismatch",
        1.35,
        mismatch_scale=1.35,
    ),
    StressScenario(
        "mismatch_1p70",
        "Mismatch 1.70x",
        "dynamic_mismatch",
        1.70,
        mismatch_scale=1.70,
    ),
)


def scenario_to_dict(scenario: StressScenario) -> Dict[str, object]:
    return {
        "scenario_id": scenario.scenario_id,
        "label": scenario.label,
        "factor": scenario.factor,
        "factor_level": scenario.factor_level,
        "measurement_noise_std_mm": scenario.measurement_noise_std_mm,
        "extra_delay_steps": scenario.extra_delay_steps,
        "mismatch_scale": scenario.mismatch_scale,
    }


def _stress_axis(
    base: AxisDynamics,
    nominal: AxisDynamics,
    mismatch_scale: float,
    extra_delay_steps: int,
) -> AxisDynamics:
    natural_frequency = nominal.natural_frequency + mismatch_scale * (
        base.natural_frequency - nominal.natural_frequency
    )
    damping_ratio = nominal.damping_ratio + mismatch_scale * (
        base.damping_ratio - nominal.damping_ratio
    )
    return AxisDynamics(
        natural_frequency=float(max(5.0, natural_frequency)),
        damping_ratio=float(max(0.25, damping_ratio)),
        delay_steps=int(base.delay_steps + extra_delay_steps),
        friction=float(base.friction * mismatch_scale),
        velocity_scale=float(base.velocity_scale),
    )


def make_stressed_virtual_machine(
    seed: int,
    scenario: StressScenario,
) -> VirtualPlantConfig:
    """Apply one stress factor while holding the other factors at baseline."""

    base = make_virtual_machine_domain(seed)
    nominal = nominal_config()
    return VirtualPlantConfig(
        x_axis=_stress_axis(
            base.x_axis,
            nominal.x_axis,
            scenario.mismatch_scale,
            scenario.extra_delay_steps,
        ),
        y_axis=_stress_axis(
            base.y_axis,
            nominal.y_axis,
            scenario.mismatch_scale,
            scenario.extra_delay_steps,
        ),
        acceleration_limit=base.acceleration_limit,
        velocity_limit=base.velocity_limit,
        cross_coupling=float(base.cross_coupling * scenario.mismatch_scale),
        repeatable_disturbance=float(
            base.repeatable_disturbance * scenario.mismatch_scale
        ),
    )


def _measurement_noise(
    shape: Tuple[int, int],
    standard_deviation: float,
    noise_seed: int,
    trial: int,
) -> np.ndarray:
    if standard_deviation == 0.0:
        return np.zeros(shape, dtype=float)
    random = np.random.RandomState(noise_seed + 7919 * trial)
    return random.normal(0.0, standard_deviation, size=shape)


def run_robustness_method(
    method: str,
    reference: ReferenceTrajectory,
    basis: np.ndarray,
    specification: SemanticTaskSpecification,
    plant_seed: int,
    settings: BenchmarkSettings,
    scenario: StressScenario,
    noise_seed: int,
) -> Dict[str, object]:
    """Run one frozen method using noisy observations and true-error scoring."""

    if method not in ROBUSTNESS_METHODS:
        raise ValueError("unknown robustness method: " + method)
    safe = method in ("violation_safe", "dual_anchor_dynamic")
    config = _semantic_config(reference, settings)
    sensitivity = build_contour_sensitivity(reference, basis, nominal_config())
    plant = make_stressed_virtual_machine(plant_seed, scenario)
    current_command = reference.position.copy()
    true_metrics: List[Dict[str, object]] = []
    measured_scores: List[float] = []
    selections: List[Tuple[int, ...]] = []
    solver_status: List[bool] = []
    acceptance_history: List[bool] = []
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

        base_command = current_command
        base_error = measured_error
        accepted = True
        if safe:
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
        else:
            accepted_command = current_command.copy()
            accepted_measured_error = measured_error.copy()
            accepted_measured_score = float(measured_metric["task_score"])
            accepted_true_metric = dict(true_metric)
        acceptance_history.append(accepted)

        if trial == config.iterations:
            break

        if method == "full_trajectory":
            selection = tuple(range(len(specification.names)))
            weights = np.ones_like(base_error)
        elif method == "dual_anchor_dynamic":
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
        else:
            selection = _dynamic_selection(
                method,
                base_error,
                specification,
                settings.number_of_windows,
            )
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
    selection_switches = sum(
        first != second for first, second in zip(selections[:-1], selections[1:])
    )
    return {
        "method": method,
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
        "selection_switches": int(selection_switches),
        "selection_history": json.dumps(selections),
        "accepted_history": json.dumps(acceptance_history),
        "rejected_trials": int(rejected_trials),
        "final_trust_radius_mm": float(trust_radius if safe else 0.0),
        "constraint_violation": int(constraints["constraint_violation"]),
        "all_updates_succeeded": int(all(solver_status)),
        "finite_result": int(
            np.all(np.isfinite(task_values))
            and np.isfinite(auc)
            and np.all(np.isfinite(final_command))
        ),
        "elapsed_s": float(elapsed),
    }
