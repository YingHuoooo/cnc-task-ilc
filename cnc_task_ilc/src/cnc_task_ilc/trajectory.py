"""Reference trajectory generation and differential geometry."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferenceTrajectory:
    """A sampled planar reference path and its geometry."""

    time: np.ndarray
    position: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    curvature: np.ndarray
    dt: float


def make_reference_trajectory(
    samples: int = 401,
    duration: float = 6.0,
) -> ReferenceTrajectory:
    """Generate the default smooth closed contour with nonuniform curvature."""

    return make_trajectory_family(
        "harmonic_loop",
        samples=samples,
        duration=duration,
    )


def _assemble_trajectory(
    time: np.ndarray,
    position: np.ndarray,
    first_derivative: np.ndarray,
    second_derivative: np.ndarray,
) -> ReferenceTrajectory:
    """Assemble unit geometry from derivatives of any monotonic path phase."""

    speed_phase = np.linalg.norm(first_derivative, axis=1)
    if np.any(speed_phase < 1.0e-10):
        raise ValueError("trajectory geometry contains a stationary path point")
    tangent = first_derivative / speed_phase[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    cross = (
        first_derivative[:, 0] * second_derivative[:, 1]
        - first_derivative[:, 1] * second_derivative[:, 0]
    )
    curvature = np.abs(cross) / np.maximum(speed_phase**3, 1.0e-12)
    return ReferenceTrajectory(
        time=time,
        position=position,
        tangent=tangent,
        normal=normal,
        curvature=curvature,
        dt=float(time[1] - time[0]),
    )


def make_trajectory_family(
    family: str,
    samples: int = 401,
    duration: float = 6.0,
) -> ReferenceTrajectory:
    """Generate one of five deterministic planar benchmark trajectories.

    A smoothstep phase law makes command velocity zero at the beginning and
    end. Geometry derivatives are evaluated analytically with respect to path
    phase, so tangent and normal remain well-defined at those endpoints.
    """

    if samples < 50:
        raise ValueError("samples must be at least 50")
    if duration <= 0:
        raise ValueError("duration must be positive")

    time = np.linspace(0.0, duration, samples)
    tau = time / duration
    phase = 3.0 * tau**2 - 2.0 * tau**3
    theta = 2.0 * np.pi * phase

    if family == "harmonic_loop":
        x = 28.0 * np.cos(theta) + 3.0 * np.cos(2.0 * theta)
        y = 18.0 * np.sin(theta) + 4.0 * np.sin(3.0 * theta)
        dx = -28.0 * np.sin(theta) - 6.0 * np.sin(2.0 * theta)
        dy = 18.0 * np.cos(theta) + 12.0 * np.cos(3.0 * theta)
        ddx = -28.0 * np.cos(theta) - 12.0 * np.cos(2.0 * theta)
        ddy = -18.0 * np.sin(theta) - 36.0 * np.sin(3.0 * theta)
    elif family == "ellipse":
        x = 30.0 * np.cos(theta)
        y = 18.0 * np.sin(theta)
        dx = -30.0 * np.sin(theta)
        dy = 18.0 * np.cos(theta)
        ddx = -30.0 * np.cos(theta)
        ddy = -18.0 * np.sin(theta)
    elif family == "rounded_square":
        x = 27.0 * np.cos(theta) + 4.5 * np.cos(3.0 * theta)
        y = 27.0 * np.sin(theta) - 4.5 * np.sin(3.0 * theta)
        dx = -27.0 * np.sin(theta) - 13.5 * np.sin(3.0 * theta)
        dy = 27.0 * np.cos(theta) - 13.5 * np.cos(3.0 * theta)
        ddx = -27.0 * np.cos(theta) - 40.5 * np.cos(3.0 * theta)
        ddy = -27.0 * np.sin(theta) + 40.5 * np.sin(3.0 * theta)
    elif family == "figure_eight":
        x = 30.0 * np.sin(theta)
        y = 14.0 * np.sin(2.0 * theta)
        dx = 30.0 * np.cos(theta)
        dy = 28.0 * np.cos(2.0 * theta)
        ddx = -30.0 * np.sin(theta)
        ddy = -56.0 * np.sin(2.0 * theta)
    elif family == "s_curve":
        q = phase
        x = -30.0 + 60.0 * q
        y = 10.0 * np.sin(2.0 * np.pi * q) + 3.0 * np.sin(
            6.0 * np.pi * q
        )
        dx = np.full_like(q, 60.0)
        dy = (
            20.0 * np.pi * np.cos(2.0 * np.pi * q)
            + 18.0 * np.pi * np.cos(6.0 * np.pi * q)
        )
        ddx = np.zeros_like(q)
        ddy = (
            -40.0 * np.pi**2 * np.sin(2.0 * np.pi * q)
            - 108.0 * np.pi**2 * np.sin(6.0 * np.pi * q)
        )
    else:
        raise ValueError("unknown trajectory family: " + family)

    return _assemble_trajectory(
        time=time,
        position=np.column_stack((x, y)),
        first_derivative=np.column_stack((dx, dy)),
        second_derivative=np.column_stack((ddx, ddy)),
    )
