"""Automatic scoring and sparse selection of critical trajectory windows."""

from dataclasses import dataclass
from typing import List

import numpy as np

from .trajectory import ReferenceTrajectory


@dataclass(frozen=True)
class CriticalWindowSelection:
    score: np.ndarray
    mask: np.ndarray
    centers: List[int]
    windows: List[List[int]]


def select_windows_from_score(
    score: np.ndarray,
    number_of_windows: int = 3,
    half_width: int = 14,
    margin: int = 0,
) -> CriticalWindowSelection:
    """Select non-overlapping windows from any externally defined score."""

    score = np.asarray(score, dtype=float)
    if score.ndim != 1:
        raise ValueError("score must be one-dimensional")
    if not np.all(np.isfinite(score)):
        raise ValueError("score must be finite")
    if number_of_windows < 1:
        raise ValueError("number_of_windows must be positive")
    if half_width < 1:
        raise ValueError("half_width must be positive")

    effective_margin = max(half_width + 1, margin)
    order = np.argsort(score)[::-1]
    centers: List[int] = []
    for candidate in order:
        if (
            candidate < effective_margin
            or candidate >= score.size - effective_margin
        ):
            continue
        if all(abs(candidate - center) > 2 * half_width for center in centers):
            centers.append(int(candidate))
        if len(centers) == number_of_windows:
            break

    centers.sort()
    mask = np.zeros(score.size, dtype=bool)
    windows: List[List[int]] = []
    for center in centers:
        start = max(0, center - half_width)
        stop = min(score.size, center + half_width + 1)
        mask[start:stop] = True
        windows.append([int(start), int(stop - 1)])

    if len(centers) != number_of_windows or not np.any(mask):
        raise RuntimeError("unable to select the requested critical windows")
    return CriticalWindowSelection(
        score=score,
        mask=mask,
        centers=centers,
        windows=windows,
    )


def select_random_windows(
    samples: int,
    seed: int,
    number_of_windows: int = 3,
    half_width: int = 14,
) -> CriticalWindowSelection:
    """Select a reproducible random-window control with the same budget."""

    random = np.random.RandomState(seed)
    score = random.uniform(0.0, 1.0, size=samples)
    return select_windows_from_score(
        score,
        number_of_windows=number_of_windows,
        half_width=half_width,
    )


def _robust_nonnegative_scale(values: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(values, dtype=float))
    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    scale = float(q75 - q25)
    if scale < 1.0e-12:
        scale = float(np.std(values))
    if scale < 1.0e-12:
        return np.zeros_like(values)
    normalized = (values - median) / scale
    return np.clip(normalized, 0.0, None)


def select_critical_windows(
    reference: ReferenceTrajectory,
    contour_error: np.ndarray,
    number_of_windows: int = 3,
    half_width: int = 14,
) -> CriticalWindowSelection:
    """Select sparse windows from error, error-rate and geometry features."""

    contour_error = np.asarray(contour_error, dtype=float)
    if contour_error.shape != reference.time.shape:
        raise ValueError("contour_error has the wrong shape")
    if number_of_windows < 1:
        raise ValueError("number_of_windows must be positive")
    if half_width < 2:
        raise ValueError("half_width must be at least 2")

    error_rate = np.gradient(contour_error, reference.dt)
    score = (
        0.50 * _robust_nonnegative_scale(contour_error)
        + 0.25 * _robust_nonnegative_scale(error_rate)
        + 0.25 * _robust_nonnegative_scale(reference.curvature)
    )

    smoothing_width = max(3, half_width // 2)
    kernel = np.ones(smoothing_width, dtype=float) / smoothing_width
    score = np.convolve(score, kernel, mode="same")

    selection = select_windows_from_score(
        score=score,
        number_of_windows=number_of_windows,
        half_width=half_width,
        margin=half_width + 2,
    )
    return selection
