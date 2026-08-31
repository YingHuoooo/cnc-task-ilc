"""Nominal learner model and structurally mismatched virtual machine."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AxisDynamics:
    natural_frequency: float
    damping_ratio: float
    delay_steps: int = 0
    friction: float = 0.0
    velocity_scale: float = 1.0


@dataclass(frozen=True)
class VirtualPlantConfig:
    x_axis: AxisDynamics
    y_axis: AxisDynamics
    acceleration_limit: float = 900.0
    velocity_limit: float = 120.0
    cross_coupling: float = 0.0
    repeatable_disturbance: float = 0.0


def nominal_config() -> VirtualPlantConfig:
    """Low-order model visible to the learning algorithm."""

    return VirtualPlantConfig(
        x_axis=AxisDynamics(16.0, 0.82),
        y_axis=AxisDynamics(16.0, 0.82),
        acceleration_limit=1.0e9,
        velocity_limit=1.0e9,
        cross_coupling=0.0,
        repeatable_disturbance=0.0,
    )


def mismatched_virtual_machine() -> VirtualPlantConfig:
    """Nonlinear evaluator hidden from the learning algorithm."""

    return VirtualPlantConfig(
        x_axis=AxisDynamics(
            natural_frequency=18.0,
            damping_ratio=0.70,
            delay_steps=3,
            friction=2.2,
            velocity_scale=3.0,
        ),
        y_axis=AxisDynamics(
            natural_frequency=12.5,
            damping_ratio=0.64,
            delay_steps=5,
            friction=3.0,
            velocity_scale=2.5,
        ),
        acceleration_limit=720.0,
        velocity_limit=90.0,
        cross_coupling=0.035,
        repeatable_disturbance=0.055,
    )


def make_virtual_machine_domain(seed: int) -> VirtualPlantConfig:
    """Generate a stable, deterministic test domain from a numeric seed."""

    random = np.random.RandomState(seed)
    return VirtualPlantConfig(
        x_axis=AxisDynamics(
            natural_frequency=float(random.uniform(15.0, 21.0)),
            damping_ratio=float(random.uniform(0.60, 0.84)),
            delay_steps=int(random.randint(1, 5)),
            friction=float(random.uniform(1.0, 3.6)),
            velocity_scale=float(random.uniform(2.0, 4.0)),
        ),
        y_axis=AxisDynamics(
            natural_frequency=float(random.uniform(10.5, 17.0)),
            damping_ratio=float(random.uniform(0.58, 0.82)),
            delay_steps=int(random.randint(2, 7)),
            friction=float(random.uniform(1.5, 4.2)),
            velocity_scale=float(random.uniform(1.8, 3.8)),
        ),
        acceleration_limit=float(random.uniform(650.0, 900.0)),
        velocity_limit=float(random.uniform(80.0, 115.0)),
        cross_coupling=float(random.uniform(0.01, 0.06)),
        repeatable_disturbance=float(random.uniform(0.02, 0.075)),
    )


def simulate_machine(
    command: np.ndarray,
    dt: float,
    config: VirtualPlantConfig,
    initial_position: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Simulate two closed-loop axes with optional nonlinear mismatch."""

    command = np.asarray(command, dtype=float)
    if command.ndim != 2 or command.shape[1] != 2:
        raise ValueError("command must have shape (samples, 2)")
    if dt <= 0:
        raise ValueError("dt must be positive")

    samples = command.shape[0]
    position = np.zeros((samples, 2), dtype=float)
    velocity = np.zeros((samples, 2), dtype=float)
    if initial_position is None:
        position[0] = command[0]
    else:
        initial_position = np.asarray(initial_position, dtype=float)
        if initial_position.shape != (2,):
            raise ValueError("initial_position must have shape (2,)")
        position[0] = initial_position

    axes = (config.x_axis, config.y_axis)
    for index in range(1, samples):
        delayed = np.empty(2, dtype=float)
        for axis_index, axis in enumerate(axes):
            source_index = max(0, index - axis.delay_steps)
            delayed[axis_index] = command[source_index, axis_index]

        error = delayed - position[index - 1]
        base_acceleration = np.empty(2, dtype=float)
        for axis_index, axis in enumerate(axes):
            friction = axis.friction * np.tanh(
                velocity[index - 1, axis_index]
                / max(axis.velocity_scale, 1.0e-9)
            )
            base_acceleration[axis_index] = (
                axis.natural_frequency**2 * error[axis_index]
                - 2.0
                * axis.damping_ratio
                * axis.natural_frequency
                * velocity[index - 1, axis_index]
                - friction
            )

        acceleration = base_acceleration.copy()
        acceleration[0] += config.cross_coupling * base_acceleration[1]
        acceleration[1] += config.cross_coupling * base_acceleration[0]
        acceleration = np.clip(
            acceleration,
            -config.acceleration_limit,
            config.acceleration_limit,
        )

        velocity[index] = np.clip(
            velocity[index - 1] + acceleration * dt,
            -config.velocity_limit,
            config.velocity_limit,
        )
        position[index] = position[index - 1] + velocity[index] * dt

    if config.repeatable_disturbance:
        phase = np.linspace(0.0, 1.0, samples)
        disturbance = config.repeatable_disturbance * np.column_stack(
            (
                np.sin(8.0 * np.pi * phase + 0.2),
                np.sin(6.0 * np.pi * phase - 0.4),
            )
        )
        position = position + disturbance

    return position
