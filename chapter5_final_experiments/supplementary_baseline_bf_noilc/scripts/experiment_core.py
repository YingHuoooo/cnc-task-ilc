"""Paired runner for Proposed and parameter-matched constrained BF-NOILC.

This module lives outside the frozen ``cnc_task_ilc`` package.  It imports the
frozen numerical plant, task, basis, sensitivity, and constrained optimizer,
and adds only the supplementary comparator and detailed effort instrumentation.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.integrate import trapezoid

from cnc_task_ilc.basis import apply_axis_coefficients
from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.delay_compensation_runner import (
    axis_delay_aligned_sensitivity,
    estimate_effective_delay_steps,
)
from cnc_task_ilc.ilc import solve_constrained_update
from cnc_task_ilc.metrics import constraint_report, task_errors
from cnc_task_ilc.plant import nominal_config, simulate_machine
from cnc_task_ilc.robustness_runner import (
    StressScenario,
    _measurement_noise,
    make_stressed_virtual_machine,
)
from cnc_task_ilc.semantic_task_benchmark import (
    SemanticTaskSpecification,
    _balanced_anchor_weights,
    _dual_anchor_selection,
    _semantic_config,
    _semantic_metrics,
    _trust_limited_candidate,
)
from cnc_task_ilc.trajectory import ReferenceTrajectory


PROPOSED = "proposed"
BF_NOILC = "parameter_matched_constrained_bf_noilc"
METHODS = (PROPOSED, BF_NOILC)


def _norms(values: np.ndarray) -> Tuple[float, float]:
    flat = np.asarray(values, dtype=float).ravel()
    return float(np.linalg.norm(flat)), float(np.sqrt(np.mean(flat**2)))


def _normalized_auc(values: Sequence[float], iterations: int) -> float:
    array = np.asarray(values, dtype=float)
    normalized = array / max(float(array[0]), 1.0e-12)
    return float(trapezoid(normalized, dx=1.0) / iterations)


def run_method(
    method: str,
    reference: ReferenceTrajectory,
    basis: np.ndarray,
    specification: SemanticTaskSpecification,
    nominal_sensitivity: np.ndarray,
    plant_seed: int,
    settings: BenchmarkSettings,
    scenario: StressScenario,
    noise_seed: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    """Run one method and return run, trial, and update records."""

    if method not in METHODS:
        raise ValueError("unknown method: " + method)

    learning_rate = 0.65 if method == PROPOSED else 1.0
    config = replace(
        _semantic_config(reference, settings),
        learning_rate=learning_rate,
    )
    plant = make_stressed_virtual_machine(int(plant_seed), scenario)
    current_command = reference.position.copy()

    true_metrics: List[Dict[str, object]] = []
    trial_rows: List[Dict[str, object]] = []
    update_rows: List[Dict[str, object]] = []
    solver_status: List[bool] = []
    update_constraints: List[bool] = []
    acceptance_history: List[bool] = []
    selection_history: List[Tuple[int, ...]] = []
    residual_lag_history: List[Tuple[int, int]] = []
    applied_lag_history: List[Tuple[float, float]] = []

    accepted_command = None
    accepted_measured_error = None
    accepted_measured_score = None
    accepted_true_metric = None
    trust_radius = 4.0
    rejected_trials = 0

    cumulative_theta_l2 = 0.0
    cumulative_learned_u_l2 = 0.0
    cumulative_issued_u_l2 = 0.0
    start_time = time.perf_counter()

    for trial in range(config.iterations + 1):
        executed_command = current_command.copy()
        true_feedback = simulate_machine(current_command, reference.dt, plant)
        measured_feedback = true_feedback + _measurement_noise(
            true_feedback.shape,
            scenario.measurement_noise_std_mm,
            int(noise_seed),
            trial,
        )
        true_error = task_errors(reference, true_feedback)["contour"]
        measured_error = task_errors(reference, measured_feedback)["contour"]
        true_metric = _semantic_metrics(true_error, specification)
        measured_metric = _semantic_metrics(measured_error, specification)
        true_metrics.append(dict(true_metric))

        residual_lag: Tuple[int, int] = (0, 0)
        applied_lag: Tuple[float, float] = (0.0, 0.0)
        if method == PROPOSED:
            measured_estimate = estimate_effective_delay_steps(
                current_command,
                measured_feedback,
                smoothing_window=5,
            )
            nominal_feedback = simulate_machine(
                current_command,
                reference.dt,
                nominal_config(),
            )
            nominal_estimate = estimate_effective_delay_steps(
                current_command,
                nominal_feedback,
                smoothing_window=5,
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
            residual_lag_history.append(residual_lag)
            applied_lag = tuple(
                0.25
                * float(
                    np.median([item[axis] for item in residual_lag_history])
                )
                for axis in range(2)
            )
            applied_lag_history.append(applied_lag)

        base_command = current_command.copy()
        base_error = measured_error.copy()
        accepted = True
        if method == PROPOSED:
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
            # Classical norm-optimal progression: every executed command is the
            # baseline for the next update; there is no proposed-specific
            # score acceptance, trust-radius scaling, or rollback.
            accepted_command = current_command.copy()
            accepted_measured_error = measured_error.copy()
            accepted_measured_score = float(measured_metric["task_score"])
            accepted_true_metric = dict(true_metric)

        acceptance_history.append(accepted)
        command_l2, command_rms = _norms(executed_command - reference.position)
        trial_rows.append(
            {
                "trial": int(trial),
                "task_score": float(true_metric["task_score"]),
                "global_rmse": float(true_metric["global_rmse"]),
                "worst_zone_ratio": float(true_metric["task_worst_zone_ratio"]),
                "violation_rate": float(true_metric["task_violation_rate"]),
                "accepted": int(accepted),
                "command_correction_l2": command_l2,
                "command_correction_rms": command_rms,
                "residual_lag_x": int(residual_lag[0]),
                "residual_lag_y": int(residual_lag[1]),
                "applied_lag_x": float(applied_lag[0]),
                "applied_lag_y": float(applied_lag[1]),
            }
        )

        if trial == config.iterations:
            break

        if method == PROPOSED:
            selection = _dual_anchor_selection(base_error, specification, 2)
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
        else:
            selection = tuple(range(len(specification.names)))
            weights = np.ones_like(base_error)
            sensitivity = nominal_sensitivity

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
        solver_status.append(bool(update["success"]))
        selection_history.append(tuple(int(item) for item in selection))
        scaled_delta = config.learning_rate * np.asarray(update["delta"], dtype=float)
        unconstrained_candidate = apply_axis_coefficients(
            base_command,
            basis,
            scaled_delta,
        )
        if method == PROPOSED:
            candidate, implemented_step_max = _trust_limited_candidate(
                base_command,
                basis,
                scaled_delta,
                trust_radius,
            )
            # Recover the actually implemented coefficient scaling when the
            # trust radius clips the candidate.
            raw_step_max = float(
                np.max(np.abs(unconstrained_candidate - base_command))
            )
            if raw_step_max > trust_radius:
                scaled_delta = scaled_delta * (trust_radius / raw_step_max)
        else:
            candidate = unconstrained_candidate
            implemented_step_max = float(np.max(np.abs(candidate - base_command)))

        learned_delta_u = candidate - base_command
        issued_delta_u = candidate - executed_command
        theta_l2 = float(np.linalg.norm(scaled_delta))
        learned_u_l2, learned_u_rms = _norms(learned_delta_u)
        issued_u_l2, issued_u_rms = _norms(issued_delta_u)
        cumulative_theta_l2 += theta_l2
        cumulative_learned_u_l2 += learned_u_l2
        cumulative_issued_u_l2 += issued_u_l2

        candidate_constraints = constraint_report(
            initial_command=reference.position,
            learned_command=candidate,
            dt=reference.dt,
            max_correction=config.correction_limit,
            velocity_limit=config.velocity_limit,
            acceleration_limit=config.acceleration_limit,
        )
        constraint_ok = not bool(candidate_constraints["constraint_violation"])
        update_constraints.append(constraint_ok)
        update_rows.append(
            {
                "update": int(trial + 1),
                "delta_theta_l2": theta_l2,
                "learned_delta_u_l2": learned_u_l2,
                "learned_delta_u_rms": learned_u_rms,
                "issued_delta_u_l2": issued_u_l2,
                "issued_delta_u_rms": issued_u_rms,
                "cumulative_delta_theta_l2": cumulative_theta_l2,
                "cumulative_learned_delta_u_l2": cumulative_learned_u_l2,
                "cumulative_issued_delta_u_l2": cumulative_issued_u_l2,
                "maximum_implemented_step_mm": implemented_step_max,
                "solver_iterations": int(update["iterations"]),
                "solver_objective": float(update["objective"]),
                "constraint_ok": int(constraint_ok),
                "selection": json.dumps(selection),
            }
        )
        current_command = candidate

    elapsed = time.perf_counter() - start_time
    task_scores = [float(item["task_score"]) for item in true_metrics]
    global_rmse = [float(item["global_rmse"]) for item in true_metrics]
    worst_zone = [float(item["task_worst_zone_ratio"]) for item in true_metrics]
    final_metric = dict(accepted_true_metric)
    final_command = np.asarray(accepted_command)
    final_constraints = constraint_report(
        initial_command=reference.position,
        learned_command=final_command,
        dt=reference.dt,
        max_correction=config.correction_limit,
        velocity_limit=config.velocity_limit,
        acceleration_limit=config.acceleration_limit,
    )
    final_correction_l2, final_correction_rms = _norms(
        final_command - reference.position
    )

    summary: Dict[str, object] = {
        "method": method,
        "learning_rate": float(learning_rate),
        "regularization": float(config.regularization),
        "smoothness": float(config.smoothness),
        "scenario_id": scenario.scenario_id,
        "measurement_noise_std_mm": float(scenario.measurement_noise_std_mm),
        "extra_delay_steps": int(scenario.extra_delay_steps),
        "mismatch_scale": float(scenario.mismatch_scale),
        "initial_task_score": task_scores[0],
        "final_task_score": float(final_metric["task_score"]),
        "task_auc_normalized": _normalized_auc(task_scores, config.iterations),
        "final_task_ratio": float(final_metric["task_score"])
        / max(task_scores[0], 1.0e-12),
        "global_rmse_auc_normalized": _normalized_auc(
            global_rmse, config.iterations
        ),
        "final_global_ratio": float(final_metric["global_rmse"])
        / max(global_rmse[0], 1.0e-12),
        "worst_zone_auc_normalized": _normalized_auc(
            worst_zone, config.iterations
        ),
        "final_worst_zone_ratio_relative": float(
            final_metric["task_worst_zone_ratio"]
        )
        / max(worst_zone[0], 1.0e-12),
        "cumulative_delta_theta_l2": cumulative_theta_l2,
        "cumulative_learned_delta_u_l2": cumulative_learned_u_l2,
        "cumulative_issued_delta_u_l2": cumulative_issued_u_l2,
        "final_command_correction_l2": final_correction_l2,
        "final_command_correction_rms": final_correction_rms,
        "rejected_trials": int(rejected_trials),
        "final_trust_radius_mm": float(trust_radius),
        "all_updates_succeeded": int(all(solver_status)),
        "all_update_constraints_satisfied": int(all(update_constraints)),
        "final_constraint_violation": int(
            final_constraints["constraint_violation"]
        ),
        "finite_result": int(
            np.all(np.isfinite(task_scores))
            and np.all(np.isfinite(global_rmse))
            and np.all(np.isfinite(final_command))
        ),
        "trial_task_scores": json.dumps(task_scores),
        "trial_global_rmse": json.dumps(global_rmse),
        "trial_worst_zone": json.dumps(worst_zone),
        "acceptance_history": json.dumps(acceptance_history),
        "selection_history": json.dumps(selection_history),
        "residual_lag_history": json.dumps(residual_lag_history),
        "applied_lag_history": json.dumps(applied_lag_history),
        "elapsed_s": float(elapsed),
    }
    return summary, trial_rows, update_rows
