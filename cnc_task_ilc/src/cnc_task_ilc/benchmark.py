"""Small multi-trajectory, multi-domain effectiveness gate."""

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(_PROJECT_ROOT / ".matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from scipy.special import expit

from .basis import apply_axis_coefficients, cubic_bspline_basis
from .critical_windows import (
    CriticalWindowSelection,
    select_critical_windows,
    select_random_windows,
    select_windows_from_score,
)
from .ilc import (
    ILCConfig,
    ILCResult,
    build_contour_sensitivity,
    run_ilc,
    solve_constrained_update,
)
from .metrics import (
    command_kinematics,
    constraint_report,
    summarize_errors,
    task_errors,
)
from .plant import (
    make_virtual_machine_domain,
    nominal_config,
    simulate_machine,
)
from .trajectory import ReferenceTrajectory, make_trajectory_family


TRAJECTORY_FAMILIES = (
    "harmonic_loop",
    "ellipse",
    "rounded_square",
    "figure_eight",
    "s_curve",
)

SELECTOR_FEATURES = (
    "absolute_error",
    "error_rate",
    "curvature",
    "jerk",
    "nominal_control_sensitivity",
)

REWARD_RANK_FEATURES = tuple(
    statistic + "_" + feature
    for feature in SELECTOR_FEATURES
    for statistic in ("mean", "max")
) + (
    "mean_error_sensitivity",
    "max_error_sensitivity",
    "nominal_gradient_norm",
)

SAFE_COMBO_FEATURES = REWARD_RANK_FEATURES + (
    "error_peak_score",
    "distance_to_nearest_anchor",
)


@dataclass(frozen=True)
class BenchmarkSettings:
    samples: int = 161
    duration: float = 6.0
    control_points: int = 12
    iterations: int = 4
    domain_seeds: Tuple[int, ...] = (11, 23, 37, 51, 73, 97)
    calibration_seeds: Tuple[int, ...] = (1001, 1003, 1007)
    number_of_windows: int = 3
    half_width: int = 5


def _robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(values, dtype=float))
    upper = float(np.percentile(values, 95.0))
    if upper < 1.0e-12:
        return np.zeros_like(values)
    return np.clip(values / upper, 0.0, 1.5)


def _smooth(values: np.ndarray, width: int = 5) -> np.ndarray:
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(values, kernel, mode="same")


def _reference_jerk(reference: ReferenceTrajectory) -> np.ndarray:
    velocity = np.gradient(reference.position, reference.dt, axis=0)
    acceleration = np.gradient(velocity, reference.dt, axis=0)
    jerk = np.gradient(acceleration, reference.dt, axis=0)
    return np.linalg.norm(jerk, axis=1)


def _window_feature_matrix(
    reference: ReferenceTrajectory,
    contour_error: np.ndarray,
    nominal_control_sensitivity: np.ndarray,
) -> np.ndarray:
    """Build per-sample features without using test-domain labels."""

    error_rate = np.gradient(contour_error, reference.dt)
    return np.column_stack(
        (
            _robust_scale(contour_error),
            _robust_scale(error_rate),
            _robust_scale(reference.curvature),
            _robust_scale(_reference_jerk(reference)),
            _robust_scale(nominal_control_sensitivity),
        )
    )


def calibrate_selector_leave_one_trajectory_out(
    held_out_family: str,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    """Fit nonnegative logistic feature weights on other trajectories only."""

    feature_blocks = []
    label_blocks = []
    for family in TRAJECTORY_FAMILIES:
        if family == held_out_family:
            continue
        reference = make_trajectory_family(
            family,
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(
            samples=settings.samples,
            control_points=settings.control_points,
        )
        sensitivity = build_contour_sensitivity(
            reference,
            basis,
            nominal_config(),
        )
        sensitivity_score = np.linalg.norm(sensitivity, axis=1)
        target = build_independent_evaluation_windows(
            reference,
            settings,
        ).mask.astype(float)
        for seed in settings.calibration_seeds:
            plant = make_virtual_machine_domain(seed)
            feedback = simulate_machine(
                reference.position,
                reference.dt,
                plant,
            )
            contour_error = task_errors(
                reference,
                feedback,
            )["contour"]
            feature_blocks.append(
                _window_feature_matrix(
                    reference,
                    contour_error,
                    sensitivity_score,
                )
            )
            label_blocks.append(target)

    features = np.vstack(feature_blocks)
    labels = np.concatenate(label_blocks)
    positive_fraction = float(np.mean(labels))
    sample_weights = np.where(
        labels > 0.5,
        0.5 / max(positive_fraction, 1.0e-6),
        0.5 / max(1.0 - positive_fraction, 1.0e-6),
    )
    regularization = 2.0e-2

    def loss(parameters: np.ndarray) -> float:
        intercept = parameters[0]
        weights = parameters[1:]
        logits = intercept + features @ weights
        logistic = np.logaddexp(0.0, logits) - labels * logits
        return float(
            np.mean(sample_weights * logistic)
            + 0.5 * regularization * np.sum(weights**2)
        )

    def gradient(parameters: np.ndarray) -> np.ndarray:
        intercept = parameters[0]
        weights = parameters[1:]
        logits = intercept + features @ weights
        residual = sample_weights * (expit(logits) - labels)
        gradient_intercept = float(np.mean(residual))
        gradient_weights = (
            features.T @ residual / features.shape[0]
            + regularization * weights
        )
        return np.concatenate(
            ([gradient_intercept], gradient_weights)
        )

    initial = np.concatenate(
        (
            [np.log(positive_fraction / (1.0 - positive_fraction))],
            0.2 * np.ones(len(SELECTOR_FEATURES), dtype=float),
        )
    )
    result = minimize(
        loss,
        initial,
        method="L-BFGS-B",
        jac=gradient,
        bounds=[(None, None)]
        + [(0.0, 8.0)] * len(SELECTOR_FEATURES),
        options={"maxiter": 300, "ftol": 1.0e-12},
    )
    if not result.success:
        raise RuntimeError(
            "selector calibration failed: " + str(result.message)
        )
    return {
        "held_out_family": held_out_family,
        "feature_names": list(SELECTOR_FEATURES),
        "intercept": float(result.x[0]),
        "weights": [float(value) for value in result.x[1:]],
        "training_samples": int(features.shape[0]),
        "positive_fraction": positive_fraction,
        "objective": float(result.fun),
        "success": bool(result.success),
    }


def apply_calibrated_selector(
    reference: ReferenceTrajectory,
    contour_error: np.ndarray,
    nominal_control_sensitivity: np.ndarray,
    selector: Dict[str, object],
    settings: BenchmarkSettings,
) -> CriticalWindowSelection:
    """Apply frozen calibration weights to one unseen test domain."""

    features = _window_feature_matrix(
        reference,
        contour_error,
        nominal_control_sensitivity,
    )
    weights = np.asarray(selector["weights"], dtype=float)
    score = expit(float(selector["intercept"]) + features @ weights)
    return _score_windows(score, settings)


def _benchmark_ilc_config(
    reference: ReferenceTrajectory,
    settings: BenchmarkSettings,
    iterations: int = None,
    global_protection_weight: float = 1.0,
) -> ILCConfig:
    initial_kinematics = command_kinematics(
        reference.position,
        reference.dt,
    )
    return ILCConfig(
        iterations=(
            settings.iterations if iterations is None else iterations
        ),
        correction_limit=4.0,
        velocity_limit=1.42
        * float(np.max(np.abs(initial_kinematics["velocity"]))),
        acceleration_limit=1.80
        * float(np.max(np.abs(initial_kinematics["acceleration"]))),
        regularization=3.0e-3,
        smoothness=2.0e-8,
        learning_rate=0.65,
        global_protection_weight=global_protection_weight,
        critical_boost=5.0,
        solver_max_iterations=240,
    )


def _reward_candidate_features(
    reference: ReferenceTrajectory,
    contour_error: np.ndarray,
    sensitivity: np.ndarray,
    settings: BenchmarkSettings,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize local features for candidate window centers."""

    sensitivity_score = np.linalg.norm(sensitivity, axis=1)
    sample_features = _window_feature_matrix(
        reference,
        contour_error,
        sensitivity_score,
    )
    interaction = sample_features[:, 0] * sample_features[:, 4]
    margin = settings.half_width + 2
    centers = np.arange(
        margin,
        reference.time.size - margin,
        stride,
        dtype=int,
    )
    rows = []
    raw_gradient_norms = []
    for center in centers:
        start = center - settings.half_width
        stop = center + settings.half_width + 1
        local = sample_features[start:stop]
        row = []
        for feature_index in range(local.shape[1]):
            row.extend(
                (
                    float(np.mean(local[:, feature_index])),
                    float(np.max(local[:, feature_index])),
                )
            )
        local_interaction = interaction[start:stop]
        row.extend(
            (
                float(np.mean(local_interaction)),
                float(np.max(local_interaction)),
            )
        )
        gradient = (
            sensitivity[start:stop].T
            @ contour_error[start:stop]
            / max(1, stop - start)
        )
        raw_gradient_norms.append(float(np.linalg.norm(gradient)))
        row.append(0.0)
        rows.append(row)
    features = np.asarray(rows, dtype=float)
    features[:, -1] = _robust_scale(
        np.asarray(raw_gradient_norms, dtype=float)
    )
    base_score = _smooth(_robust_scale(np.abs(contour_error)))
    return features, centers, base_score[centers]


def _single_update_reward(
    reference: ReferenceTrajectory,
    basis: np.ndarray,
    sensitivity: np.ndarray,
    contour_error: np.ndarray,
    evaluation_mask: np.ndarray,
    candidate_center: int,
    plant_seed: int,
    settings: BenchmarkSettings,
) -> float:
    """Measure real one-update utility on a calibration virtual machine."""

    candidate_mask = np.zeros(reference.time.size, dtype=bool)
    start = candidate_center - settings.half_width
    stop = candidate_center + settings.half_width + 1
    candidate_mask[start:stop] = True
    config = _benchmark_ilc_config(reference, settings, iterations=1)
    weights = np.full(
        contour_error.shape,
        config.global_protection_weight,
        dtype=float,
    )
    weights[candidate_mask] += config.critical_boost
    weights /= np.mean(weights)
    update = solve_constrained_update(
        contour_error=contour_error,
        sensitivity=sensitivity,
        weights=weights,
        basis=basis,
        initial_command=reference.position,
        current_command=reference.position,
        dt=reference.dt,
        config=config,
    )
    learned_command = apply_axis_coefficients(
        reference.position,
        basis,
        config.learning_rate * update["delta"],
    )
    final_feedback = simulate_machine(
        learned_command,
        reference.dt,
        make_virtual_machine_domain(plant_seed),
    )
    final_error = task_errors(reference, final_feedback)["contour"]
    initial_summary = summarize_errors(contour_error, evaluation_mask)
    final_summary = summarize_errors(final_error, evaluation_mask)
    critical_reward = 1.0 - (
        final_summary["critical_rmse"]
        / max(initial_summary["critical_rmse"], 1.0e-12)
    )
    global_ratio = (
        final_summary["global_rmse"]
        / max(initial_summary["global_rmse"], 1.0e-12)
    )
    return float(
        critical_reward - 0.10 * max(0.0, global_ratio - 1.0)
    )


def build_reward_calibration_dataset(
    settings: BenchmarkSettings,
) -> Dict[str, List[Dict[str, object]]]:
    """Generate candidate-window rewards only on calibration domains."""

    dataset: Dict[str, List[Dict[str, object]]] = {}
    stride = settings.half_width + 1
    for family in TRAJECTORY_FAMILIES:
        reference = make_trajectory_family(
            family,
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(
            samples=settings.samples,
            control_points=settings.control_points,
        )
        sensitivity = build_contour_sensitivity(
            reference,
            basis,
            nominal_config(),
        )
        evaluation = build_independent_evaluation_windows(
            reference,
            settings,
        )
        groups = []
        for seed in settings.calibration_seeds:
            plant = make_virtual_machine_domain(seed)
            feedback = simulate_machine(
                reference.position,
                reference.dt,
                plant,
            )
            contour_error = task_errors(
                reference,
                feedback,
            )["contour"]
            features, centers, base_scores = _reward_candidate_features(
                reference,
                contour_error,
                sensitivity,
                settings,
                stride=stride,
            )
            rewards = np.asarray(
                [
                    _single_update_reward(
                        reference,
                        basis,
                        sensitivity,
                        contour_error,
                        evaluation.mask,
                        int(center),
                        int(seed),
                        settings,
                    )
                    for center in centers
                ],
                dtype=float,
            )
            groups.append(
                {
                    "family": family,
                    "domain_seed": int(seed),
                    "features": features,
                    "centers": centers,
                    "base_scores": base_scores,
                    "rewards": rewards,
                }
            )
        dataset[family] = groups
    return dataset


def _expanded_reward_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    return np.column_stack((features, features**2))


def _fit_reward_ridge(
    groups: Sequence[Dict[str, object]],
    alpha: float,
) -> Dict[str, np.ndarray]:
    features = np.vstack(
        [np.asarray(group["features"], dtype=float) for group in groups]
    )
    targets = []
    for group in groups:
        rewards = np.asarray(group["rewards"], dtype=float)
        scale = max(float(np.std(rewards)), 1.0e-8)
        targets.append((rewards - float(np.mean(rewards))) / scale)
    target = np.concatenate(targets)
    expanded = _expanded_reward_features(features)
    mean = np.mean(expanded, axis=0)
    scale = np.std(expanded, axis=0)
    scale[scale < 1.0e-8] = 1.0
    standardized = (expanded - mean) / scale
    coefficients = np.linalg.solve(
        standardized.T @ standardized
        + alpha * np.eye(standardized.shape[1]),
        standardized.T @ target,
    )
    return {
        "mean": mean,
        "scale": scale,
        "coefficients": coefficients,
    }


def _predict_reward(
    features: np.ndarray,
    model: Dict[str, object],
) -> np.ndarray:
    expanded = _expanded_reward_features(features)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    return ((expanded - mean) / scale) @ coefficients


def _rank_unit_interval(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks / max(1, values.size - 1)


def _top_candidate_indices(
    centers: np.ndarray,
    scores: np.ndarray,
    settings: BenchmarkSettings,
) -> List[int]:
    selected: List[int] = []
    for index in np.argsort(scores)[::-1]:
        center = int(centers[index])
        if all(
            abs(center - int(centers[other]))
            > 2 * settings.half_width
            for other in selected
        ):
            selected.append(int(index))
        if len(selected) == settings.number_of_windows:
            break
    if len(selected) != settings.number_of_windows:
        raise RuntimeError("unable to rank enough reward candidates")
    return selected


def calibrate_reward_ranker_leave_one_trajectory_out(
    held_out_family: str,
    settings: BenchmarkSettings,
    dataset: Dict[str, List[Dict[str, object]]],
) -> Dict[str, object]:
    """Fit a reward ranker with nested trajectory-level calibration."""

    training_families = [
        family
        for family in TRAJECTORY_FAMILIES
        if family != held_out_family
    ]
    candidates = []
    for alpha in (0.1, 1.0, 10.0):
        for blend in (0.0, 0.25, 0.50, 0.75, 1.0):
            utilities = []
            for validation_family in training_families:
                fit_groups = [
                    group
                    for family in training_families
                    if family != validation_family
                    for group in dataset[family]
                ]
                validation_groups = dataset[validation_family]
                fold_model = _fit_reward_ridge(fit_groups, alpha)
                for group in validation_groups:
                    prediction = _predict_reward(
                        np.asarray(group["features"], dtype=float),
                        fold_model,
                    )
                    combined = (
                        (1.0 - blend)
                        * _rank_unit_interval(group["base_scores"])
                        + blend * _rank_unit_interval(prediction)
                    )
                    selected = _top_candidate_indices(
                        np.asarray(group["centers"], dtype=int),
                        combined,
                        settings,
                    )
                    rewards = np.asarray(group["rewards"], dtype=float)
                    oracle = _top_candidate_indices(
                        np.asarray(group["centers"], dtype=int),
                        rewards,
                        settings,
                    )
                    selected_reward = float(np.mean(rewards[selected]))
                    oracle_reward = float(np.mean(rewards[oracle]))
                    average_reward = float(np.mean(rewards))
                    utilities.append(
                        (selected_reward - average_reward)
                        / max(oracle_reward - average_reward, 1.0e-8)
                    )
            candidates.append(
                {
                    "alpha": alpha,
                    "blend": blend,
                    "cross_validated_utility": float(
                        np.mean(utilities)
                    ),
                }
            )
    chosen = max(
        candidates,
        key=lambda item: (
            item["cross_validated_utility"],
            -item["blend"],
            -item["alpha"],
        ),
    )
    training_groups = [
        group
        for family in training_families
        for group in dataset[family]
    ]
    fitted = _fit_reward_ridge(training_groups, float(chosen["alpha"]))
    return {
        "held_out_family": held_out_family,
        "feature_names": list(REWARD_RANK_FEATURES),
        "feature_expansion": "linear_and_squared",
        "alpha": float(chosen["alpha"]),
        "reward_blend": float(chosen["blend"]),
        "cross_validated_utility": float(
            chosen["cross_validated_utility"]
        ),
        "training_groups": len(training_groups),
        "training_candidates": int(
            sum(len(group["rewards"]) for group in training_groups)
        ),
        "mean": [float(value) for value in fitted["mean"]],
        "scale": [float(value) for value in fitted["scale"]],
        "coefficients": [
            float(value) for value in fitted["coefficients"]
        ],
    }


def apply_reward_ranker(
    reference: ReferenceTrajectory,
    contour_error: np.ndarray,
    sensitivity: np.ndarray,
    ranker: Dict[str, object],
    settings: BenchmarkSettings,
) -> CriticalWindowSelection:
    """Rank all unseen-test candidate windows by predicted ILC reward."""

    features, centers, base_scores = _reward_candidate_features(
        reference,
        contour_error,
        sensitivity,
        settings,
        stride=1,
    )
    prediction = _predict_reward(features, ranker)
    blend = float(ranker["reward_blend"])
    combined = (
        (1.0 - blend) * _rank_unit_interval(base_scores)
        + blend * _rank_unit_interval(prediction)
    )
    score = np.full(
        reference.time.size,
        float(np.min(combined) - 1.0),
        dtype=float,
    )
    score[centers] = combined
    return select_windows_from_score(
        score,
        number_of_windows=settings.number_of_windows,
        half_width=settings.half_width,
        margin=settings.half_width + 2,
    )


def _selection_from_centers(
    score: np.ndarray,
    centers: Sequence[int],
    settings: BenchmarkSettings,
) -> CriticalWindowSelection:
    centers = sorted(int(center) for center in centers)
    if len(centers) != settings.number_of_windows:
        raise ValueError("the center count must match the window budget")
    if any(
        second - first <= 2 * settings.half_width
        for first, second in zip(centers[:-1], centers[1:])
    ):
        raise ValueError("safe-combination windows overlap")
    mask = np.zeros(len(score), dtype=bool)
    windows = []
    for center in centers:
        start = center - settings.half_width
        stop = center + settings.half_width + 1
        if start < 0 or stop > len(score):
            raise ValueError("safe-combination window exceeds trajectory")
        mask[start:stop] = True
        windows.append([int(start), int(stop - 1)])
    return CriticalWindowSelection(
        score=np.asarray(score, dtype=float),
        mask=mask,
        centers=centers,
        windows=windows,
    )


def _safe_combo_candidates(
    reference: ReferenceTrajectory,
    contour_error: np.ndarray,
    sensitivity: np.ndarray,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    """Create two error-peak anchors and feasible third-window choices."""

    features, all_centers, base_scores = _reward_candidate_features(
        reference,
        contour_error,
        sensitivity,
        settings,
        stride=1,
    )
    full_base_score = _smooth(_robust_scale(np.abs(contour_error)))
    safe_selection = select_windows_from_score(
        full_base_score,
        number_of_windows=settings.number_of_windows,
        half_width=settings.half_width,
        margin=settings.half_width + 2,
    )
    ranked_safe_centers = sorted(
        safe_selection.centers,
        key=lambda center: float(full_base_score[center]),
        reverse=True,
    )
    anchors = ranked_safe_centers[:2]
    safe_third = int(ranked_safe_centers[2])
    center_to_index = {
        int(center): index for index, center in enumerate(all_centers)
    }
    stride = settings.half_width + 1
    candidates = [
        int(center)
        for center in all_centers[::stride]
        if all(
            abs(int(center) - anchor) > 2 * settings.half_width
            for anchor in anchors
        )
    ]
    if safe_third not in candidates:
        candidates.append(safe_third)
    candidates = sorted(set(candidates))
    selected_indices = [center_to_index[center] for center in candidates]
    selected_features = features[selected_indices]
    selected_base_scores = base_scores[selected_indices]
    distances = np.asarray(
        [
            min(abs(center - anchor) for anchor in anchors)
            / max(1, reference.time.size - 1)
            for center in candidates
        ],
        dtype=float,
    )
    extended_features = np.column_stack(
        (selected_features, selected_base_scores, distances)
    )
    return {
        "features": extended_features,
        "centers": np.asarray(candidates, dtype=int),
        "anchors": [int(center) for center in anchors],
        "safe_third": safe_third,
        "safe_index": int(candidates.index(safe_third)),
        "base_score": full_base_score,
    }


def _multi_update_combo_reward(
    reference: ReferenceTrajectory,
    basis: np.ndarray,
    sensitivity: np.ndarray,
    evaluation_mask: np.ndarray,
    optimization_mask: np.ndarray,
    plant_seed: int,
    settings: BenchmarkSettings,
) -> float:
    """Return the real four-update AUC reward of a three-window set."""

    config = _benchmark_ilc_config(
        reference,
        settings,
        global_protection_weight=0.30,
    )
    current_command = reference.position.copy()
    critical_rmse = []
    global_rmse = []
    for trial in range(config.iterations + 1):
        feedback = simulate_machine(
            current_command,
            reference.dt,
            make_virtual_machine_domain(plant_seed),
        )
        contour_error = task_errors(reference, feedback)["contour"]
        summary = summarize_errors(contour_error, evaluation_mask)
        critical_rmse.append(float(summary["critical_rmse"]))
        global_rmse.append(float(summary["global_rmse"]))
        if trial == config.iterations:
            break
        weights = np.full(
            contour_error.shape,
            config.global_protection_weight,
            dtype=float,
        )
        weights[optimization_mask] += config.critical_boost
        weights /= np.mean(weights)
        update = solve_constrained_update(
            contour_error=contour_error,
            sensitivity=sensitivity,
            weights=weights,
            basis=basis,
            initial_command=reference.position,
            current_command=current_command,
            dt=reference.dt,
            config=config,
        )
        current_command = apply_axis_coefficients(
            current_command,
            basis,
            config.learning_rate * update["delta"],
        )
    normalized = np.asarray(critical_rmse) / max(
        critical_rmse[0],
        1.0e-12,
    )
    auc = float(
        trapezoid(normalized, dx=1.0) / config.iterations
    )
    global_ratio = global_rmse[-1] / max(global_rmse[0], 1.0e-12)
    return float(1.0 - auc - 0.10 * max(0.0, global_ratio - 1.0))


def build_safe_combo_calibration_dataset(
    settings: BenchmarkSettings,
) -> Dict[str, List[Dict[str, object]]]:
    """Generate three-window, full-horizon rewards on calibration domains."""

    dataset: Dict[str, List[Dict[str, object]]] = {}
    for family in TRAJECTORY_FAMILIES:
        reference = make_trajectory_family(
            family,
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(
            samples=settings.samples,
            control_points=settings.control_points,
        )
        sensitivity = build_contour_sensitivity(
            reference,
            basis,
            nominal_config(),
        )
        evaluation = build_independent_evaluation_windows(
            reference,
            settings,
        )
        groups = []
        for seed in settings.calibration_seeds:
            feedback = simulate_machine(
                reference.position,
                reference.dt,
                make_virtual_machine_domain(seed),
            )
            contour_error = task_errors(reference, feedback)["contour"]
            specification = _safe_combo_candidates(
                reference,
                contour_error,
                sensitivity,
                settings,
            )
            rewards = []
            for candidate in specification["centers"]:
                selected_centers = list(specification["anchors"]) + [
                    int(candidate)
                ]
                selection = _selection_from_centers(
                    specification["base_score"],
                    selected_centers,
                    settings,
                )
                rewards.append(
                    _multi_update_combo_reward(
                        reference,
                        basis,
                        sensitivity,
                        evaluation.mask,
                        selection.mask,
                        int(seed),
                        settings,
                    )
                )
            groups.append(
                {
                    "family": family,
                    "domain_seed": int(seed),
                    "features": specification["features"],
                    "centers": specification["centers"],
                    "safe_index": specification["safe_index"],
                    "rewards": np.asarray(rewards, dtype=float),
                }
            )
        dataset[family] = groups
    return dataset


def _fit_safe_combo_ensemble(
    families: Sequence[str],
    dataset: Dict[str, List[Dict[str, object]]],
    alpha: float,
) -> List[Dict[str, np.ndarray]]:
    models = []
    for omitted_family in families:
        groups = [
            group
            for family in families
            if family != omitted_family
            for group in dataset[family]
        ]
        models.append(_fit_reward_ridge(groups, alpha))
    return models


def _safe_combo_predictions(
    features: np.ndarray,
    models: Sequence[Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray]:
    predictions = np.stack(
        [_predict_reward(features, model) for model in models],
        axis=0,
    )
    return np.mean(predictions, axis=0), np.std(predictions, axis=0)


def calibrate_safe_combo_leave_one_trajectory_out(
    held_out_family: str,
    settings: BenchmarkSettings,
    dataset: Dict[str, List[Dict[str, object]]],
) -> Dict[str, object]:
    """Calibrate a risk-aware ensemble without the held-out trajectory."""

    training_families = [
        family
        for family in TRAJECTORY_FAMILIES
        if family != held_out_family
    ]
    candidates = []
    for alpha in (1.0, 10.0, 100.0):
        for risk_beta in (0.0, 0.5, 1.0):
            for minimum_advantage in (
                0.0,
                0.10,
                0.25,
                0.50,
                1.00,
                1.0e6,
            ):
                utilities = []
                fallback_count = 0
                degradation_count = 0
                decisions = 0
                for validation_family in training_families:
                    fit_families = [
                        family
                        for family in training_families
                        if family != validation_family
                    ]
                    ensemble = _fit_safe_combo_ensemble(
                        fit_families,
                        dataset,
                        alpha,
                    )
                    for group in dataset[validation_family]:
                        rewards = np.asarray(
                            group["rewards"],
                            dtype=float,
                        )
                        mean, uncertainty = _safe_combo_predictions(
                            np.asarray(group["features"], dtype=float),
                            ensemble,
                        )
                        lower_confidence = mean - risk_beta * uncertainty
                        safe_index = int(group["safe_index"])
                        best_index = int(np.argmax(lower_confidence))
                        if (
                            lower_confidence[best_index]
                            < lower_confidence[safe_index]
                            + minimum_advantage
                        ):
                            best_index = safe_index
                            fallback_count += 1
                        decisions += 1
                        reward_scale = max(float(np.std(rewards)), 1.0e-8)
                        normalized_gain = (
                            rewards[best_index] - rewards[safe_index]
                        ) / reward_scale
                        failure_penalty = 0.50 * float(
                            rewards[best_index] < rewards[safe_index]
                        )
                        degradation_count += int(
                            rewards[best_index]
                            < rewards[safe_index] - 1.0e-12
                        )
                        utilities.append(
                            float(normalized_gain - failure_penalty)
                        )
                candidates.append(
                    {
                        "alpha": alpha,
                        "risk_beta": risk_beta,
                        "minimum_advantage": minimum_advantage,
                        "cross_validated_utility": float(
                            np.mean(utilities)
                        ),
                        "fallback_rate": float(
                            fallback_count / max(decisions, 1)
                        ),
                        "degradation_rate": float(
                            degradation_count / max(decisions, 1)
                        ),
                    }
                )
    risk_feasible = [
        item for item in candidates if item["degradation_rate"] <= 0.10
    ]
    chosen = max(
        risk_feasible if risk_feasible else candidates,
        key=lambda item: (
            item["cross_validated_utility"],
            item["fallback_rate"],
            -item["risk_beta"],
            -item["alpha"],
        ),
    )
    ensemble = _fit_safe_combo_ensemble(
        training_families,
        dataset,
        float(chosen["alpha"]),
    )
    serializable_models = []
    for model in ensemble:
        serializable_models.append(
            {
                "mean": [float(value) for value in model["mean"]],
                "scale": [float(value) for value in model["scale"]],
                "coefficients": [
                    float(value) for value in model["coefficients"]
                ],
            }
        )
    return {
        "held_out_family": held_out_family,
        "feature_names": list(SAFE_COMBO_FEATURES),
        "reward": "four-update normalized critical-error AUC",
        "anchor_windows": 2,
        "exploration_windows": 1,
        "alpha": float(chosen["alpha"]),
        "risk_beta": float(chosen["risk_beta"]),
        "minimum_advantage": float(chosen["minimum_advantage"]),
        "cross_validated_utility": float(
            chosen["cross_validated_utility"]
        ),
        "cross_validated_fallback_rate": float(
            chosen["fallback_rate"]
        ),
        "cross_validated_degradation_rate": float(
            chosen["degradation_rate"]
        ),
        "training_groups": int(
            len(training_families) * len(settings.calibration_seeds)
        ),
        "ensemble": serializable_models,
    }


def apply_safe_combo_selector(
    reference: ReferenceTrajectory,
    contour_error: np.ndarray,
    sensitivity: np.ndarray,
    model: Dict[str, object],
    settings: BenchmarkSettings,
) -> CriticalWindowSelection:
    """Choose one risk-controlled exploration window beside two anchors."""

    specification = _safe_combo_candidates(
        reference,
        contour_error,
        sensitivity,
        settings,
    )
    mean, uncertainty = _safe_combo_predictions(
        np.asarray(specification["features"], dtype=float),
        model["ensemble"],
    )
    lower_confidence = mean - float(model["risk_beta"]) * uncertainty
    safe_index = int(specification["safe_index"])
    best_index = int(np.argmax(lower_confidence))
    if (
        lower_confidence[best_index]
        < lower_confidence[safe_index]
        + float(model["minimum_advantage"])
    ):
        best_index = safe_index
    selected_centers = list(specification["anchors"]) + [
        int(specification["centers"][best_index])
    ]
    score = np.asarray(specification["base_score"], dtype=float).copy()
    return _selection_from_centers(score, selected_centers, settings)


def _score_windows(
    score: np.ndarray,
    settings: BenchmarkSettings,
) -> CriticalWindowSelection:
    return select_windows_from_score(
        _smooth(_robust_scale(score)),
        number_of_windows=settings.number_of_windows,
        half_width=settings.half_width,
    )


def build_independent_evaluation_windows(
    reference: ReferenceTrajectory,
    settings: BenchmarkSettings,
) -> CriticalWindowSelection:
    """Build fixed evaluation windows from held-out calibration domains."""

    calibration_errors = []
    for seed in settings.calibration_seeds:
        plant = make_virtual_machine_domain(seed)
        feedback = simulate_machine(
            reference.position,
            reference.dt,
            plant,
        )
        calibration_errors.append(
            np.abs(task_errors(reference, feedback)["contour"])
        )
    consensus_error = np.median(
        np.stack(calibration_errors, axis=0),
        axis=0,
    )
    evaluation_score = (
        0.85 * _robust_scale(consensus_error)
        + 0.15 * _robust_scale(reference.curvature)
    )
    return _score_windows(evaluation_score, settings)


def build_method_windows(
    reference: ReferenceTrajectory,
    initial_error: np.ndarray,
    settings: BenchmarkSettings,
    random_seed: int,
    nominal_control_sensitivity: np.ndarray = None,
    calibrated_selector: Dict[str, object] = None,
    nominal_sensitivity: np.ndarray = None,
    reward_ranker: Dict[str, object] = None,
    safe_combo_model: Dict[str, object] = None,
) -> Dict[str, CriticalWindowSelection]:
    """Create equal-budget window baselines without using evaluation windows."""

    windows = {
        "curvature": _score_windows(reference.curvature, settings),
        "jerk": _score_windows(_reference_jerk(reference), settings),
        "error_peak": _score_windows(np.abs(initial_error), settings),
        "random": select_random_windows(
            reference.time.size,
            seed=random_seed,
            number_of_windows=settings.number_of_windows,
            half_width=settings.half_width,
        ),
        "automatic": select_critical_windows(
            reference,
            initial_error,
            number_of_windows=settings.number_of_windows,
            half_width=settings.half_width,
        ),
    }
    if calibrated_selector is not None:
        if nominal_control_sensitivity is None:
            raise ValueError(
                "nominal_control_sensitivity is required for learned windows"
            )
        windows["learned_automatic"] = apply_calibrated_selector(
            reference,
            initial_error,
            nominal_control_sensitivity,
            calibrated_selector,
            settings,
        )
    if reward_ranker is not None:
        if nominal_sensitivity is None:
            raise ValueError(
                "nominal_sensitivity is required for reward ranking"
            )
        windows["reward_ranked"] = apply_reward_ranker(
            reference,
            initial_error,
            nominal_sensitivity,
            reward_ranker,
            settings,
        )
    if safe_combo_model is not None:
        if nominal_sensitivity is None:
            raise ValueError(
                "nominal_sensitivity is required for safe combinations"
            )
        windows["safe_combo"] = apply_safe_combo_selector(
            reference,
            initial_error,
            nominal_sensitivity,
            safe_combo_model,
            settings,
        )
    return windows


def _method_definitions(
    include_learned: bool = True,
    include_reward_ranker: bool = False,
    include_safe_combo: bool = False,
) -> List[Tuple[str, str, float, bool]]:
    """Return method name, mask key, global weight and weighting flag."""

    methods = [
        ("full_trajectory", "automatic", 1.0, False),
        ("curvature_window", "curvature", 0.30, True),
        ("jerk_window", "jerk", 0.30, True),
        ("error_peak_window", "error_peak", 0.30, True),
        ("random_window", "random", 0.30, True),
        ("automatic_eta_0.10", "automatic", 0.10, True),
        ("automatic_eta_0.30", "automatic", 0.30, True),
        ("automatic_eta_1.00", "automatic", 1.00, True),
    ]
    if include_learned:
        methods.append(
            (
                "learned_auto_eta_1.00",
                "learned_automatic",
                1.00,
                True,
            )
        )
    if include_reward_ranker:
        methods.append(
            (
                "reward_rank_eta_1.00",
                "reward_ranked",
                1.00,
                True,
            )
        )
    if include_safe_combo:
        methods.append(
            (
                "safe_combo_eta_0.30",
                "safe_combo",
                0.30,
                True,
            )
        )
    return methods


def _normalized_auc(result: ILCResult, metric_name: str) -> float:
    values = np.asarray(
        [metric[metric_name] for metric in result.metrics],
        dtype=float,
    )
    normalized = values / max(values[0], 1.0e-12)
    return float(trapezoid(normalized, dx=1.0) / (values.size - 1))


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second)
    if not np.any(union):
        return 0.0
    return float(np.sum(np.logical_and(first, second)) / np.sum(union))


def _run_one_method(
    method_name: str,
    mask_key: str,
    global_weight: float,
    critical_weighting: bool,
    reference: ReferenceTrajectory,
    basis: np.ndarray,
    evaluation: CriticalWindowSelection,
    method_windows: Dict[str, CriticalWindowSelection],
    plant_seed: int,
    settings: BenchmarkSettings,
) -> Dict[str, object]:
    plant = make_virtual_machine_domain(plant_seed)
    initial_kinematics = command_kinematics(
        reference.position,
        reference.dt,
    )
    config = ILCConfig(
        iterations=settings.iterations,
        correction_limit=4.0,
        velocity_limit=1.42
        * float(np.max(np.abs(initial_kinematics["velocity"]))),
        acceleration_limit=1.80
        * float(np.max(np.abs(initial_kinematics["acceleration"]))),
        regularization=3.0e-3,
        smoothness=2.0e-8,
        learning_rate=0.65,
        global_protection_weight=global_weight,
        critical_boost=5.0,
        solver_max_iterations=240,
    )
    start = time.perf_counter()
    result = run_ilc(
        name=method_name,
        reference=reference,
        initial_command=reference.position,
        basis=basis,
        evaluation_mask=evaluation.mask,
        optimization_mask=method_windows[mask_key].mask,
        nominal_model=nominal_config(),
        virtual_plant=plant,
        config=config,
        critical_weighting=critical_weighting,
    )
    elapsed = time.perf_counter() - start
    constraints = constraint_report(
        initial_command=reference.position,
        learned_command=result.commands[-1],
        dt=reference.dt,
        max_correction=config.correction_limit,
        velocity_limit=config.velocity_limit,
        acceleration_limit=config.acceleration_limit,
    )
    initial = result.metrics[0]
    final = result.metrics[-1]
    return {
        "method": method_name,
        "domain_seed": plant_seed,
        "initial_critical_rmse": initial["critical_rmse"],
        "final_critical_rmse": final["critical_rmse"],
        "initial_global_rmse": initial["global_rmse"],
        "final_global_rmse": final["global_rmse"],
        "critical_auc_normalized": _normalized_auc(
            result,
            "critical_rmse",
        ),
        "global_auc_normalized": _normalized_auc(
            result,
            "global_rmse",
        ),
        "final_critical_ratio": float(
            final["critical_rmse"]
            / max(initial["critical_rmse"], 1.0e-12)
        ),
        "final_global_ratio": float(
            final["global_rmse"]
            / max(initial["global_rmse"], 1.0e-12)
        ),
        "window_iou_with_evaluation": (
            1.0
            if not critical_weighting
            else _mask_iou(
                method_windows[mask_key].mask,
                evaluation.mask,
            )
        ),
        "constraint_violation": constraints["constraint_violation"],
        "all_updates_succeeded": int(
            all(status["success"] for status in result.solver_status)
        ),
        "elapsed_s": float(elapsed),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty benchmark table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    methods = sorted({str(row["method"]) for row in rows})
    summary: List[Dict[str, object]] = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        auc = np.asarray(
            [float(row["critical_auc_normalized"]) for row in selected]
        )
        critical_ratio = np.asarray(
            [float(row["final_critical_ratio"]) for row in selected]
        )
        global_ratio = np.asarray(
            [float(row["final_global_ratio"]) for row in selected]
        )
        overlap = np.asarray(
            [float(row["window_iou_with_evaluation"]) for row in selected]
        )
        summary.append(
            {
                "method": method,
                "runs": len(selected),
                "median_critical_auc": float(np.median(auc)),
                "q25_critical_auc": float(np.percentile(auc, 25.0)),
                "q75_critical_auc": float(np.percentile(auc, 75.0)),
                "median_final_critical_ratio": float(
                    np.median(critical_ratio)
                ),
                "median_final_global_ratio": float(np.median(global_ratio)),
                "median_window_iou": float(np.median(overlap)),
                "success_rate": float(
                    np.mean(
                        [
                            int(row["all_updates_succeeded"])
                            and not int(row["constraint_violation"])
                            for row in selected
                        ]
                    )
                ),
                "median_elapsed_s": float(
                    np.median(
                        [float(row["elapsed_s"]) for row in selected]
                    )
                ),
            }
        )
    return summary


def _evaluate_gate(
    rows: Sequence[Dict[str, object]],
    summary: Sequence[Dict[str, object]],
    primary: str,
) -> Dict[str, object]:
    baseline_candidates = {
        "full_trajectory",
        "curvature_window",
        "jerk_window",
        "error_peak_window",
        "random_window",
    }
    summary_by_method = {
        str(row["method"]): row for row in summary
    }
    strongest_baseline = min(
        baseline_candidates,
        key=lambda method: float(
            summary_by_method[method]["median_critical_auc"]
        ),
    )
    primary_rows = {
        (str(row["trajectory"]), int(row["domain_seed"])): row
        for row in rows
        if row["method"] == primary
    }

    pairwise_vs_baselines: Dict[str, Dict[str, float]] = {}
    for baseline_method in sorted(baseline_candidates):
        baseline_rows = {
            (str(row["trajectory"]), int(row["domain_seed"])): row
            for row in rows
            if row["method"] == baseline_method
        }
        pair_keys = sorted(set(primary_rows).intersection(baseline_rows))
        pair_improvements = []
        pair_global_multipliers = []
        pair_wins = []
        for key in pair_keys:
            proposed_auc = float(
                primary_rows[key]["critical_auc_normalized"]
            )
            baseline_auc = float(
                baseline_rows[key]["critical_auc_normalized"]
            )
            pair_improvements.append(
                100.0 * (baseline_auc - proposed_auc)
                / max(baseline_auc, 1.0e-12)
            )
            pair_wins.append(proposed_auc < baseline_auc)
            proposed_global = float(
                primary_rows[key]["final_global_ratio"]
            )
            baseline_global = float(
                baseline_rows[key]["final_global_ratio"]
            )
            pair_global_multipliers.append(
                proposed_global / max(baseline_global, 1.0e-12)
            )
        pairwise_vs_baselines[baseline_method] = {
            "paired_cases": len(pair_keys),
            "median_auc_improvement_percent": float(
                np.median(pair_improvements)
            ),
            "win_rate": float(np.mean(pair_wins)),
            "median_global_ratio_multiplier": float(
                np.median(pair_global_multipliers)
            ),
        }

    primary_summary = summary_by_method[primary]
    strongest_comparison = pairwise_vs_baselines[strongest_baseline]
    median_improvement = float(
        strongest_comparison["median_auc_improvement_percent"]
    )
    win_rate = float(strongest_comparison["win_rate"])
    median_global_multiplier = float(
        strongest_comparison["median_global_ratio_multiplier"]
    )
    success_rate = float(primary_summary["success_rate"])
    criteria = {
        "median_auc_improvement_at_least_10_percent": (
            median_improvement >= 10.0
        ),
        "paired_win_rate_at_least_60_percent": win_rate >= 0.60,
        "success_rate_at_least_95_percent": success_rate >= 0.95,
        "median_global_tradeoff_no_more_than_1_5x": (
            median_global_multiplier <= 1.50
        ),
    }
    passed = bool(all(criteria.values()))
    return {
        "primary_method": primary,
        "strongest_baseline": strongest_baseline,
        "paired_cases": int(strongest_comparison["paired_cases"]),
        "median_critical_auc_improvement_percent": median_improvement,
        "paired_win_rate": win_rate,
        "primary_success_rate": success_rate,
        "median_global_ratio_multiplier_vs_baseline": (
            median_global_multiplier
        ),
        "criteria": criteria,
        "pairwise_vs_baselines": pairwise_vs_baselines,
        "passed": passed,
        "decision": "GO" if passed else "REVISE",
    }


def _plot_benchmark(
    output_path: Path,
    rows: Sequence[Dict[str, object]],
    summary: Sequence[Dict[str, object]],
    gate: Dict[str, object],
) -> None:
    candidate_order = [
        "full_trajectory",
        "curvature_window",
        "jerk_window",
        "error_peak_window",
        "random_window",
        "automatic_eta_0.10",
        "automatic_eta_0.30",
        "automatic_eta_1.00",
        "learned_auto_eta_1.00",
        "reward_rank_eta_1.00",
        "safe_combo_eta_0.30",
    ]
    candidate_labels = [
        "Full",
        "Curvature",
        "Jerk",
        "Error peak",
        "Random",
        "Auto η=.10",
        "Auto η=.30",
        "Auto η=1.0",
        "Learned η=1.0",
        "Reward rank η=1.0",
        "Safe combo η=.30",
    ]
    available_methods = {
        str(row["method"]) for row in summary
    }
    method_order = [
        method
        for method in candidate_order
        if method in available_methods
    ]
    short_labels = [
        label
        for method, label in zip(candidate_order, candidate_labels)
        if method in available_methods
    ]
    grouped = [
        [
            float(row["critical_auc_normalized"])
            for row in rows
            if row["method"] == method
        ]
        for method in method_order
    ]
    summary_by_method = {
        str(row["method"]): row for row in summary
    }

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))
    axis = axes[0, 0]
    box = axis.boxplot(grouped, patch_artist=True, showfliers=False)
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(
            "tab:orange"
            if (
                method_order[index].startswith("automatic")
                or method_order[index].startswith("learned_auto")
                or method_order[index].startswith("reward_rank")
                or method_order[index].startswith("safe_combo")
            )
            else "tab:blue"
        )
        patch.set_alpha(0.45)
    axis.set_xticklabels(short_labels, rotation=28, ha="right")
    axis.set_ylabel("Normalized critical-error AUC ↓")
    axis.set_title("Paired effectiveness across trajectory–machine cases")
    axis.grid(axis="y", alpha=0.25)

    axis = axes[0, 1]
    for method, label in zip(method_order, short_labels):
        item = summary_by_method[method]
        marker = (
            "s"
            if (
                method.startswith("automatic")
                or method.startswith("learned_auto")
                or method.startswith("reward_rank")
                or method.startswith("safe_combo")
            )
            else "o"
        )
        axis.scatter(
            item["median_final_global_ratio"],
            item["median_final_critical_ratio"],
            s=70,
            marker=marker,
            label=label,
        )
        axis.annotate(
            label,
            (
                item["median_final_global_ratio"],
                item["median_final_critical_ratio"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Median final global-error ratio ↓")
    axis.set_ylabel("Median final critical-error ratio ↓")
    axis.set_title("Critical/global Pareto trade-off")
    axis.grid(alpha=0.25)

    primary = str(gate["primary_method"])
    baseline = str(gate["strongest_baseline"])
    case_keys = sorted(
        {
            (str(row["trajectory"]), int(row["domain_seed"]))
            for row in rows
        }
    )
    advantage_by_case = []
    for trajectory, domain_seed in case_keys:
        values = {
            str(row["method"]): float(
                row["critical_auc_normalized"]
            )
            for row in rows
            if row["trajectory"] == trajectory
            and int(row["domain_seed"]) == domain_seed
            and row["method"] in (primary, baseline)
        }
        advantage_by_case.append(
            100.0
            * (values[baseline] - values[primary])
            / max(values[baseline], 1.0e-12)
        )
    axis = axes[1, 0]
    axis.axhline(0.0, color="black", linewidth=0.8)
    colors = [
        "tab:green" if value > 0.0 else "tab:red"
        for value in advantage_by_case
    ]
    axis.bar(np.arange(len(advantage_by_case)), advantage_by_case, color=colors)
    axis.set_xlabel("Paired trajectory–machine case")
    axis.set_ylabel("AUC improvement over strongest baseline [%]")
    axis.set_title(primary + " paired improvements")
    axis.grid(axis="y", alpha=0.25)

    axis = axes[1, 1]
    criterion_labels = [
        "Median AUC\nimprovement",
        "Paired\nwin rate",
        "Run\nsuccess",
        "Global\ntrade-off",
    ]
    criterion_values = [
        float(gate["median_critical_auc_improvement_percent"]) / 10.0,
        float(gate["paired_win_rate"]) / 0.60,
        float(gate["primary_success_rate"]) / 0.95,
        1.50
        / max(
            float(gate["median_global_ratio_multiplier_vs_baseline"]),
            1.0e-12,
        ),
    ]
    criterion_colors = [
        "tab:green" if value >= 1.0 else "tab:red"
        for value in criterion_values
    ]
    axis.bar(
        np.arange(4),
        criterion_values,
        color=criterion_colors,
        alpha=0.75,
    )
    axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_xticks(np.arange(4))
    axis.set_xticklabels(criterion_labels)
    axis.set_ylabel("Criterion ratio; pass ≥ 1")
    axis.set_title(
        "Effectiveness gate: " + str(gate["decision"])
    )
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    output_path: Path,
    settings: BenchmarkSettings,
    gate: Dict[str, object],
    summary: Sequence[Dict[str, object]],
) -> None:
    summary_by_method = {
        str(row["method"]): row for row in summary
    }
    primary = summary_by_method[str(gate["primary_method"])]
    versus_full = gate["pairwise_vs_baselines"]["full_trajectory"]
    versus_error = gate["pairwise_vs_baselines"]["error_peak_window"]
    safe_combo_protocol = (
        "- Safe-combination protocol: two error-peak anchors plus one "
        "risk-controlled exploration window; labels are three-window, "
        "full-horizon ILC AUC rewards; ensemble risk and conservative "
        "fallback are calibrated; confirmation uses previously unseen "
        "machine seeds\n"
        if str(gate["primary_method"]).startswith("safe_combo")
        else ""
    )
    reward_protocol = (
        "- Reward-ranker protocol: candidate-window one-step ILC rewards "
        "are generated only on calibration domains; nested "
        "trajectory-family validation selects ridge regularization and "
        "error-peak blending; the held-out trajectory family and test "
        "domains remain frozen\n"
        if str(gate["primary_method"]).startswith("reward_rank")
        else ""
    )
    learned_protocol = (
        "- Learned selector protocol: leave-one-trajectory-family-out "
        "calibration; test-domain labels are never used for fitting\n"
        if (
            str(gate["primary_method"]).startswith("learned_auto")
            or str(gate["primary_method"]).startswith("reward_rank")
            or str(gate["primary_method"]).startswith("safe_combo")
        )
        else ""
    )
    method_scope = (
        "full trajectory, curvature, jerk, error peak, random, fixed-score "
        "automatic windows, a calibrated point classifier, and a "
        "risk-controlled three-window combination selector"
        if str(gate["primary_method"]).startswith("safe_combo")
        else
        "full trajectory, curvature, jerk, error peak, random, fixed-score "
        "automatic windows, a calibrated point classifier, and a "
        "simulation-reward window ranker"
        if str(gate["primary_method"]).startswith("reward_rank")
        else
        "full trajectory, curvature, jerk, error peak, random, fixed-score "
        "automatic windows, and a calibrated automatic selector"
        if str(gate["primary_method"]).startswith("learned_auto")
        else "full trajectory, curvature, jerk, error peak, random, and "
        "fixed-score automatic windows"
    )
    window_name = (
        "safe-combination-window"
        if str(gate["primary_method"]).startswith("safe_combo")
        else "reward-ranked-window"
        if str(gate["primary_method"]).startswith("reward_rank")
        else "learned-window"
        if str(gate["primary_method"]).startswith("learned_auto")
        else "automatic-window"
    )
    report = """# Small Effectiveness Gate

## Scope

- Trajectory families: {trajectories}
- Virtual machine domains: {domains}
- Paired trajectory–machine cases: {cases}
- ILC updates per run: {iterations}
- Methods: {method_scope}
- Evaluation windows: fixed per trajectory from separate calibration domains
{learned_protocol}{reward_protocol}{safe_combo_protocol}- Primary method: {primary_method}

## Predefined gate

- median critical-error AUC improvement over the strongest aggregate baseline ≥ 10%
- paired win rate ≥ 60%
- successful constrained runs ≥ 95%
- median global-error ratio multiplier versus the strongest baseline ≤ 1.5

## Result

- Decision: {decision}
- Strongest baseline: {baseline}
- Median critical AUC improvement: {improvement:.2f}%
- Paired win rate: {win_rate:.2%}
- Primary-method success rate: {success_rate:.2%}
- Median global trade-off multiplier: {global_multiplier:.3f}×
- Primary median normalized critical AUC: {primary_auc:.4f}
- Primary median final critical-error ratio: {primary_critical:.4f}
- Primary median final global-error ratio: {primary_global:.4f}
- Versus full-trajectory ILC: {full_improvement:.2f}% median AUC improvement; {full_win:.2%} paired win rate
- Versus error-peak windows: {error_improvement:.2f}% median AUC improvement; {error_win:.2%} paired win rate
- Median {window_name}/evaluation-window IoU: {window_iou:.3f}

## Interpretation

The gate evaluates whether the revised automatic critical-window contribution is
already strong enough to justify adding LinuxCNC and more advanced learning
modules. A REVISE decision does not invalidate ILC feasibility; it means the
automatic-window method has not yet passed the predefined comparative standard.
""".format(
        trajectories=len(TRAJECTORY_FAMILIES),
        domains=len(settings.domain_seeds),
        cases=len(TRAJECTORY_FAMILIES) * len(settings.domain_seeds),
        iterations=settings.iterations,
        learned_protocol=learned_protocol,
        reward_protocol=reward_protocol,
        safe_combo_protocol=safe_combo_protocol,
        primary_method=gate["primary_method"],
        method_scope=method_scope,
        window_name=window_name,
        decision=gate["decision"],
        baseline=gate["strongest_baseline"],
        improvement=gate["median_critical_auc_improvement_percent"],
        win_rate=gate["paired_win_rate"],
        success_rate=gate["primary_success_rate"],
        global_multiplier=gate[
            "median_global_ratio_multiplier_vs_baseline"
        ],
        primary_auc=primary["median_critical_auc"],
        primary_critical=primary["median_final_critical_ratio"],
        primary_global=primary["median_final_global_ratio"],
        full_improvement=versus_full[
            "median_auc_improvement_percent"
        ],
        full_win=versus_full["win_rate"],
        error_improvement=versus_error[
            "median_auc_improvement_percent"
        ],
        error_win=versus_error["win_rate"],
        window_iou=primary["median_window_iou"],
    )
    output_path.write_text(report, encoding="utf-8")


def run_effectiveness_gate(
    output_directory: Path,
    settings: BenchmarkSettings = BenchmarkSettings(),
    primary_method: str = "learned_auto_eta_1.00",
    include_learned: bool = True,
    include_reward_ranker: bool = False,
    include_safe_combo: bool = False,
) -> Dict[str, object]:
    """Run the small gate and create raw, aggregate and visual outputs."""

    output_directory.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    selectors_by_family = (
        {
            family: calibrate_selector_leave_one_trajectory_out(
                family,
                settings,
            )
            for family in TRAJECTORY_FAMILIES
        }
        if include_learned
        else {}
    )
    reward_dataset = (
        build_reward_calibration_dataset(settings)
        if include_reward_ranker
        else {}
    )
    reward_rankers_by_family = (
        {
            family: calibrate_reward_ranker_leave_one_trajectory_out(
                family,
                settings,
                reward_dataset,
            )
            for family in TRAJECTORY_FAMILIES
        }
        if include_reward_ranker
        else {}
    )
    safe_combo_dataset = (
        build_safe_combo_calibration_dataset(settings)
        if include_safe_combo
        else {}
    )
    safe_combo_models_by_family = (
        {
            family: calibrate_safe_combo_leave_one_trajectory_out(
                family,
                settings,
                safe_combo_dataset,
            )
            for family in TRAJECTORY_FAMILIES
        }
        if include_safe_combo
        else {}
    )
    for trajectory_index, family in enumerate(TRAJECTORY_FAMILIES):
        reference = make_trajectory_family(
            family,
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(
            samples=settings.samples,
            control_points=settings.control_points,
        )
        nominal_control_sensitivity = None
        nominal_sensitivity = None
        if include_learned or include_reward_ranker or include_safe_combo:
            nominal_sensitivity = build_contour_sensitivity(
                reference,
                basis,
                nominal_config(),
            )
            nominal_control_sensitivity = np.linalg.norm(
                nominal_sensitivity,
                axis=1,
            )
        evaluation = build_independent_evaluation_windows(
            reference,
            settings,
        )

        for domain_seed in settings.domain_seeds:
            plant = make_virtual_machine_domain(domain_seed)
            initial_feedback = simulate_machine(
                reference.position,
                reference.dt,
                plant,
            )
            initial_error = task_errors(
                reference,
                initial_feedback,
            )["contour"]
            method_windows = build_method_windows(
                reference,
                initial_error,
                settings,
                random_seed=(
                    10000
                    + 100 * trajectory_index
                    + int(domain_seed)
                ),
                nominal_control_sensitivity=(
                    nominal_control_sensitivity
                ),
                calibrated_selector=(
                    selectors_by_family[family]
                    if include_learned
                    else None
                ),
                nominal_sensitivity=nominal_sensitivity,
                reward_ranker=(
                    reward_rankers_by_family[family]
                    if include_reward_ranker
                    else None
                ),
                safe_combo_model=(
                    safe_combo_models_by_family[family]
                    if include_safe_combo
                    else None
                ),
            )
            for (
                method_name,
                mask_key,
                global_weight,
                critical_weighting,
            ) in _method_definitions(
                include_learned=include_learned,
                include_reward_ranker=include_reward_ranker,
                include_safe_combo=include_safe_combo,
            ):
                row = _run_one_method(
                    method_name=method_name,
                    mask_key=mask_key,
                    global_weight=global_weight,
                    critical_weighting=critical_weighting,
                    reference=reference,
                    basis=basis,
                    evaluation=evaluation,
                    method_windows=method_windows,
                    plant_seed=domain_seed,
                    settings=settings,
                )
                row["trajectory"] = family
                row["evaluation_window_fraction"] = float(
                    np.mean(evaluation.mask)
                )
                rows.append(row)

    column_order = [
        "trajectory",
        "domain_seed",
        "method",
        "initial_critical_rmse",
        "final_critical_rmse",
        "initial_global_rmse",
        "final_global_rmse",
        "critical_auc_normalized",
        "global_auc_normalized",
        "final_critical_ratio",
        "final_global_ratio",
        "window_iou_with_evaluation",
        "evaluation_window_fraction",
        "constraint_violation",
        "all_updates_succeeded",
        "elapsed_s",
    ]
    ordered_rows = [
        {column: row[column] for column in column_order}
        for row in rows
    ]
    summary = _aggregate_rows(ordered_rows)
    gate = _evaluate_gate(
        ordered_rows,
        summary,
        primary=primary_method,
    )
    payload = {
        "settings": {
            "trajectories": list(TRAJECTORY_FAMILIES),
            "domain_seeds": list(settings.domain_seeds),
            "calibration_seeds": list(settings.calibration_seeds),
            "samples": settings.samples,
            "iterations": settings.iterations,
            "control_points": settings.control_points,
            "number_of_windows": settings.number_of_windows,
            "half_width": settings.half_width,
            "paired_cases": len(TRAJECTORY_FAMILIES)
            * len(settings.domain_seeds),
            "total_method_runs": len(ordered_rows),
        },
        "gate": gate,
        "summary": summary,
        "selector_calibration": selectors_by_family,
        "reward_ranker_calibration": reward_rankers_by_family,
        "safe_combo_calibration": safe_combo_models_by_family,
    }

    _write_csv(output_directory / "effectiveness_raw.csv", ordered_rows)
    _write_csv(output_directory / "effectiveness_summary.csv", summary)
    (output_directory / "effectiveness_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _plot_benchmark(
        output_directory / "effectiveness_summary.png",
        ordered_rows,
        summary,
        gate,
    )
    _write_report(
        output_directory / "effectiveness_report.md",
        settings,
        gate,
        summary,
    )
    return payload
