"""Task-space errors, learning metrics and command constraint checks."""

from typing import Dict

import numpy as np

from .trajectory import ReferenceTrajectory


def task_errors(
    reference: ReferenceTrajectory,
    feedback: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return Cartesian, contour-normal and tangent errors."""

    feedback = np.asarray(feedback, dtype=float)
    if feedback.shape != reference.position.shape:
        raise ValueError("feedback and reference shapes differ")
    cartesian = feedback - reference.position
    contour = np.sum(cartesian * reference.normal, axis=1)
    tangent = np.sum(cartesian * reference.tangent, axis=1)
    return {
        "cartesian": cartesian,
        "contour": contour,
        "tangent": tangent,
    }


def summarize_errors(
    contour_error: np.ndarray,
    critical_mask: np.ndarray,
) -> Dict[str, float]:
    """Summarize global and critical-window contour errors."""

    contour_error = np.asarray(contour_error, dtype=float)
    critical_mask = np.asarray(critical_mask, dtype=bool)
    if contour_error.shape != critical_mask.shape:
        raise ValueError("critical mask and error shapes differ")
    critical = contour_error[critical_mask]
    if critical.size == 0:
        raise ValueError("critical mask contains no samples")
    return {
        "global_rmse": float(np.sqrt(np.mean(contour_error**2))),
        "global_max_abs": float(np.max(np.abs(contour_error))),
        "critical_rmse": float(np.sqrt(np.mean(critical**2))),
        "critical_max_abs": float(np.max(np.abs(critical))),
    }


def command_kinematics(command: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    """Return first and second finite differences of a command."""

    command = np.asarray(command, dtype=float)
    velocity = np.diff(command, axis=0) / dt
    acceleration = np.diff(command, n=2, axis=0) / (dt**2)
    return {
        "velocity": velocity,
        "acceleration": acceleration,
    }


def constraint_report(
    initial_command: np.ndarray,
    learned_command: np.ndarray,
    dt: float,
    max_correction: float,
    velocity_limit: float,
    acceleration_limit: float,
    tolerance: float = 1.0e-5,
) -> Dict[str, float]:
    """Report maxima and count configured constraint violations."""

    correction = learned_command - initial_command
    kinematics = command_kinematics(learned_command, dt)
    maximum_correction = float(np.max(np.abs(correction)))
    maximum_velocity = float(np.max(np.abs(kinematics["velocity"])))
    maximum_acceleration = float(np.max(np.abs(kinematics["acceleration"])))
    violations = int(
        maximum_correction > max_correction + tolerance
        or maximum_velocity > velocity_limit + tolerance
        or maximum_acceleration > acceleration_limit + tolerance
    )
    return {
        "max_abs_correction": maximum_correction,
        "max_abs_velocity": maximum_velocity,
        "max_abs_acceleration": maximum_acceleration,
        "configured_max_correction": float(max_correction),
        "configured_velocity_limit": float(velocity_limit),
        "configured_acceleration_limit": float(acceleration_limit),
        "constraint_violation": violations,
    }

