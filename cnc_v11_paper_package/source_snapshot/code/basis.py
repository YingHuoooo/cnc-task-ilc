"""Low-dimensional B-spline representation of command corrections."""

import numpy as np
from scipy.interpolate import BSpline


def cubic_bspline_basis(
    samples: int,
    control_points: int = 18,
    clamp_correction_ends: bool = True,
) -> np.ndarray:
    """Return a cubic B-spline design matrix sampled on [0, 1]."""

    degree = 3
    if control_points <= degree + 1:
        raise ValueError("control_points must be greater than degree + 1")
    if samples < control_points:
        raise ValueError("samples must be at least control_points")

    internal_count = control_points - degree - 1
    internal = np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
    knots = np.concatenate(
        (
            np.zeros(degree + 1),
            internal,
            np.ones(degree + 1),
        )
    )
    phase = np.linspace(0.0, 1.0, samples)
    basis = np.empty((samples, control_points), dtype=float)
    for column in range(control_points):
        coefficient = np.zeros(control_points, dtype=float)
        coefficient[column] = 1.0
        basis[:, column] = BSpline(
            knots,
            coefficient,
            degree,
            extrapolate=False,
        )(phase)

    if clamp_correction_ends:
        basis = basis[:, 1:-1]
    return basis


def apply_axis_coefficients(
    base_command: np.ndarray,
    basis: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Apply axis-blocked coefficients to a two-axis command."""

    columns = basis.shape[1]
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (2 * columns,):
        raise ValueError("coefficient vector has the wrong size")
    correction = np.column_stack(
        (
            basis @ coefficients[:columns],
            basis @ coefficients[columns:],
        )
    )
    return np.asarray(base_command, dtype=float) + correction

