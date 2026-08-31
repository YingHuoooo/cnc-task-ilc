"""Task-level ILC sensitivity, constrained quadratic update and trial loop."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import Bounds, LinearConstraint, minimize

from .basis import apply_axis_coefficients
from .metrics import summarize_errors, task_errors
from .plant import VirtualPlantConfig, simulate_machine
from .trajectory import ReferenceTrajectory


@dataclass(frozen=True)
class ILCConfig:
    iterations: int = 8
    correction_limit: float = 4.0
    velocity_limit: float = 80.0
    acceleration_limit: float = 500.0
    regularization: float = 2.0e-3
    smoothness: float = 2.0e-8
    learning_rate: float = 0.70
    global_protection_weight: float = 0.20
    critical_boost: float = 5.0
    solver_max_iterations: int = 300


@dataclass
class ILCResult:
    name: str
    metrics: List[Dict[str, float]]
    commands: List[np.ndarray]
    feedbacks: List[np.ndarray]
    contour_errors: List[np.ndarray]
    solver_status: List[Dict[str, object]]


def build_contour_sensitivity(
    reference: ReferenceTrajectory,
    basis: np.ndarray,
    nominal_model: VirtualPlantConfig,
) -> np.ndarray:
    """Map spline coefficient perturbations to contour-normal displacement."""

    samples, columns = basis.shape
    sensitivity = np.empty((samples, 2 * columns), dtype=float)
    zero_state = np.zeros(2, dtype=float)

    for column in range(columns):
        perturbation = np.zeros((samples, 2), dtype=float)
        perturbation[:, 0] = basis[:, column]
        response = simulate_machine(
            perturbation,
            reference.dt,
            nominal_model,
            initial_position=zero_state,
        )
        sensitivity[:, column] = np.sum(
            response * reference.normal,
            axis=1,
        )

        perturbation.fill(0.0)
        perturbation[:, 1] = basis[:, column]
        response = simulate_machine(
            perturbation,
            reference.dt,
            nominal_model,
            initial_position=zero_state,
        )
        sensitivity[:, columns + column] = np.sum(
            response * reference.normal,
            axis=1,
        )

    return sensitivity


def _axis_block(matrix: np.ndarray) -> np.ndarray:
    return block_diag(matrix, matrix)


def solve_constrained_update(
    contour_error: np.ndarray,
    sensitivity: np.ndarray,
    weights: np.ndarray,
    basis: np.ndarray,
    initial_command: np.ndarray,
    current_command: np.ndarray,
    dt: float,
    config: ILCConfig,
) -> Dict[str, object]:
    """Solve a convex quadratic update with linear motion constraints.

    SciPy SLSQP is used as the available QP-capable constrained optimizer. The
    objective itself is a positive-definite quadratic and every constraint is
    linear.
    """

    contour_error = np.asarray(contour_error, dtype=float)
    weights = np.asarray(weights, dtype=float)
    samples = contour_error.size
    variables = sensitivity.shape[1]
    if sensitivity.shape[0] != samples or weights.shape != (samples,):
        raise ValueError("inconsistent error, sensitivity or weight shapes")

    second_basis = np.diff(basis, n=2, axis=0) / (dt**2)
    smooth_map = _axis_block(second_basis)
    weighted_sensitivity = weights[:, None] * sensitivity
    hessian = (
        sensitivity.T @ weighted_sensitivity / samples
        + config.regularization * np.eye(variables)
        + config.smoothness
        * (smooth_map.T @ smooth_map)
        / max(1, smooth_map.shape[0])
    )
    hessian = 0.5 * (hessian + hessian.T)
    linear = sensitivity.T @ (weights * contour_error) / samples

    def objective(delta: np.ndarray) -> float:
        return float(0.5 * delta @ hessian @ delta + linear @ delta)

    def gradient(delta: np.ndarray) -> np.ndarray:
        return hessian @ delta + linear

    position_map = _axis_block(basis)
    velocity_basis = np.diff(basis, axis=0) / dt
    velocity_map = _axis_block(velocity_basis)
    acceleration_basis = np.diff(basis, n=2, axis=0) / (dt**2)
    acceleration_map = _axis_block(acceleration_basis)

    current_correction = current_command - initial_command
    correction_vector = np.concatenate(
        (current_correction[:, 0], current_correction[:, 1])
    )
    current_velocity = np.diff(current_command, axis=0) / dt
    velocity_vector = np.concatenate(
        (current_velocity[:, 0], current_velocity[:, 1])
    )
    current_acceleration = np.diff(current_command, n=2, axis=0) / (dt**2)
    acceleration_vector = np.concatenate(
        (current_acceleration[:, 0], current_acceleration[:, 1])
    )

    constraint_matrix = np.vstack(
        (position_map, velocity_map, acceleration_map)
    )
    lower = np.concatenate(
        (
            -config.correction_limit - correction_vector,
            -config.velocity_limit - velocity_vector,
            -config.acceleration_limit - acceleration_vector,
        )
    )
    upper = np.concatenate(
        (
            config.correction_limit - correction_vector,
            config.velocity_limit - velocity_vector,
            config.acceleration_limit - acceleration_vector,
        )
    )
    constraint = LinearConstraint(constraint_matrix, lower, upper)
    bounds = Bounds(
        -2.0 * config.correction_limit * np.ones(variables),
        2.0 * config.correction_limit * np.ones(variables),
    )

    result = minimize(
        objective,
        np.zeros(variables, dtype=float),
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=[constraint],
        options={
            "maxiter": config.solver_max_iterations,
            "ftol": 1.0e-9,
            "disp": False,
        },
    )
    if not result.success:
        raise RuntimeError("constrained update failed: " + str(result.message))

    return {
        "delta": np.asarray(result.x, dtype=float),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
    }


def run_ilc(
    name: str,
    reference: ReferenceTrajectory,
    initial_command: np.ndarray,
    basis: np.ndarray,
    evaluation_mask: np.ndarray,
    optimization_mask: np.ndarray,
    nominal_model: VirtualPlantConfig,
    virtual_plant: VirtualPlantConfig,
    config: ILCConfig,
    critical_weighting: bool,
) -> ILCResult:
    """Run repeated trials using measured task error and nominal sensitivity."""

    sensitivity = build_contour_sensitivity(
        reference,
        basis,
        nominal_model,
    )
    current_command = np.asarray(initial_command, dtype=float).copy()

    metrics: List[Dict[str, float]] = []
    commands: List[np.ndarray] = []
    feedbacks: List[np.ndarray] = []
    contour_errors: List[np.ndarray] = []
    solver_status: List[Dict[str, object]] = []

    for trial in range(config.iterations + 1):
        feedback = simulate_machine(
            current_command,
            reference.dt,
            virtual_plant,
        )
        contour_error = task_errors(reference, feedback)["contour"]
        summary = summarize_errors(contour_error, evaluation_mask)
        summary["trial"] = float(trial)

        commands.append(current_command.copy())
        feedbacks.append(feedback.copy())
        contour_errors.append(contour_error.copy())
        metrics.append(summary)

        if trial == config.iterations:
            break

        if critical_weighting:
            weights = np.full(
                contour_error.shape,
                config.global_protection_weight,
                dtype=float,
            )
            weights[optimization_mask] += config.critical_boost
            weights /= np.mean(weights)
        else:
            weights = np.ones_like(contour_error)

        update = solve_constrained_update(
            contour_error=contour_error,
            sensitivity=sensitivity,
            weights=weights,
            basis=basis,
            initial_command=initial_command,
            current_command=current_command,
            dt=reference.dt,
            config=config,
        )
        scaled_delta = config.learning_rate * update["delta"]
        current_command = apply_axis_coefficients(
            current_command,
            basis,
            scaled_delta,
        )
        solver_status.append(
            {
                "trial": trial,
                "success": update["success"],
                "status": update["status"],
                "message": update["message"],
                "iterations": update["iterations"],
                "objective": update["objective"],
            }
        )

    return ILCResult(
        name=name,
        metrics=metrics,
        commands=commands,
        feedbacks=feedbacks,
        contour_errors=contour_errors,
        solver_status=solver_status,
    )
