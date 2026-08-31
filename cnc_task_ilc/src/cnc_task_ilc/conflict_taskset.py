"""Feedback-independent tolerance-conflict task manifests for V7 experiments."""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from .basis import cubic_bspline_basis
from .benchmark import TRAJECTORY_FAMILIES
from .ilc import build_contour_sensitivity
from .metrics import task_errors
from .plant import make_virtual_machine_domain, nominal_config, simulate_machine
from .semantic_task_benchmark import (
    SemanticTaskSpecification,
    _reference_jerk,
    _robust_unit_scale,
    _zone_statistic,
    make_semantic_task_specification,
)
from .trajectory import ReferenceTrajectory, make_trajectory_family


TASKSET_ID = "tolerance-conflict-v1"
SCHEMA_VERSION = 1
TASK_REGIMES = ("neutral", "demand_aligned", "demand_conflict")
NEUTRAL_TOLERANCE_MM = 0.24
RANKED_TOLERANCES_MM = (0.14, 0.18, 0.22, 0.28, 0.36, 0.46)


def _program_demand_scores(
    reference: ReferenceTrajectory,
    specification: SemanticTaskSpecification,
    control_points: int,
) -> np.ndarray:
    """Score programmed difficulty without observing a virtual machine."""

    basis = cubic_bspline_basis(
        samples=reference.time.size,
        control_points=control_points,
    )
    sensitivity = build_contour_sensitivity(
        reference,
        basis,
        nominal_config(),
    )
    per_sample = (
        0.40 * _robust_unit_scale(reference.curvature)
        + 0.35 * _robust_unit_scale(_reference_jerk(reference))
        + 0.25
        * _robust_unit_scale(np.linalg.norm(sensitivity, axis=1))
    )
    return np.asarray(
        [
            float(np.mean(per_sample[start : stop + 1]))
            for start, stop in specification.windows
        ],
        dtype=float,
    )


def _regime_tolerances(
    demand_scores: np.ndarray,
    regime: str,
) -> Tuple[float, ...]:
    if regime == "neutral":
        return tuple(NEUTRAL_TOLERANCE_MM for _ in demand_scores)
    if regime not in ("demand_aligned", "demand_conflict"):
        raise ValueError("unknown task regime: " + regime)

    demand_order = np.argsort(-np.asarray(demand_scores), kind="mergesort")
    tolerances = np.empty(len(demand_scores), dtype=float)
    ranked = np.asarray(RANKED_TOLERANCES_MM, dtype=float)
    if regime == "demand_aligned":
        # High programmed demand receives the tightest functional tolerance.
        tolerances[demand_order] = ranked
    else:
        # Functional priority deliberately conflicts with programmed demand.
        tolerances[demand_order] = ranked[::-1]
    return tuple(float(value) for value in tolerances)


def build_task_manifest(
    family: str,
    regime: str,
    samples: int = 161,
    duration: float = 6.0,
    control_points: int = 12,
    half_width: int = 5,
) -> Dict[str, object]:
    """Build one resolution-independent manifest without plant feedback."""

    reference = make_trajectory_family(
        family,
        samples=samples,
        duration=duration,
    )
    base = make_semantic_task_specification(
        reference,
        family,
        half_width,
    )
    demand = _program_demand_scores(reference, base, control_points)
    tolerances = _regime_tolerances(demand, regime)
    demand_ranks = np.empty(demand.size, dtype=int)
    demand_ranks[np.argsort(-demand, kind="mergesort")] = np.arange(1, demand.size + 1)

    zones = []
    for index, (
        name,
        role,
        center,
        rule,
        score,
        rank,
        tolerance,
    ) in enumerate(
        zip(
            base.names,
            base.roles,
            base.centers,
            base.generation_rules,
            demand,
            demand_ranks,
            tolerances,
        )
    ):
        zones.append(
            {
                "zone_index": int(index),
                "name": str(name),
                "role": str(role),
                "center_phase": float(center / (samples - 1)),
                "half_width_phase": float(half_width / (samples - 1)),
                "generation_rule": str(rule),
                "program_demand_score": float(score),
                "program_demand_rank": int(rank),
                "tolerance_mm": float(tolerance),
            }
        )

    return {
        "manifest_id": family + "--" + regime,
        "schema_version": SCHEMA_VERSION,
        "taskset_id": TASKSET_ID,
        "trajectory_family": family,
        "regime": regime,
        "reference_sampling": {
            "samples": int(samples),
            "duration_s": float(duration),
            "control_points": int(control_points),
        },
        "construction": {
            "uses_machine_feedback": False,
            "uses_measured_tracking_error": False,
            "demand_features": [
                "programmed_curvature",
                "programmed_jerk",
                "nominal_model_control_sensitivity",
            ],
            "tolerance_assignment": regime,
        },
        "zones": zones,
    }


def build_conflict_taskset(
    samples: int = 161,
    duration: float = 6.0,
    control_points: int = 12,
    half_width: int = 5,
) -> Dict[str, object]:
    manifests = [
        build_task_manifest(
            family,
            regime,
            samples=samples,
            duration=duration,
            control_points=control_points,
            half_width=half_width,
        )
        for family in TRAJECTORY_FAMILIES
        for regime in TASK_REGIMES
    ]
    return {
        "taskset_id": TASKSET_ID,
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Feedback-independent CNC task manifests with neutral, "
            "program-demand-aligned and program-demand-conflict tolerances."
        ),
        "families": list(TRAJECTORY_FAMILIES),
        "regimes": list(TASK_REGIMES),
        "manifest_count": len(manifests),
        "zones_per_manifest": 6,
        "construction_guarantees": {
            "machine_feedback_used": False,
            "measured_tracking_error_used": False,
            "formal_confirmation_seeds_included": False,
        },
        "manifests": manifests,
    }


def specification_from_manifest(
    reference: ReferenceTrajectory,
    manifest: Dict[str, object],
) -> SemanticTaskSpecification:
    """Resolve a manifest at any compatible trajectory sampling resolution."""

    samples = reference.time.size
    zones = list(manifest["zones"])
    names = []
    roles = []
    centers = []
    windows = []
    tolerances = []
    rules = []
    mask = np.zeros(samples, dtype=bool)
    for zone in zones:
        center = int(round(float(zone["center_phase"]) * (samples - 1)))
        half_width = max(
            1,
            int(round(float(zone["half_width_phase"]) * (samples - 1))),
        )
        start = center - half_width
        stop = center + half_width
        if start < 0 or stop >= samples:
            raise ValueError("manifest zone exceeds trajectory boundary")
        names.append(str(zone["name"]))
        roles.append(str(zone["role"]))
        centers.append(center)
        windows.append((start, stop))
        tolerances.append(float(zone["tolerance_mm"]))
        rules.append(str(zone["generation_rule"]))
        mask[start : stop + 1] = True
    if any(
        second - first <= first_width + second_width
        for (first, second), first_width, second_width in zip(
            zip(centers[:-1], centers[1:]),
            [int((stop - start) / 2) for start, stop in windows[:-1]],
            [int((stop - start) / 2) for start, stop in windows[1:]],
        )
    ):
        raise ValueError("resolved manifest zones overlap")
    return SemanticTaskSpecification(
        names=tuple(names),
        roles=tuple(roles),
        centers=tuple(centers),
        windows=tuple(windows),
        tolerances=tuple(tolerances),
        evaluation_mask=mask,
        generation_rules=tuple(rules),
    )


def save_conflict_taskset(path: Path, taskset: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(taskset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_conflict_taskset(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("taskset_id") != TASKSET_ID:
        raise ValueError("unexpected taskset id")
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported taskset schema")
    return payload


def _top_k(values: np.ndarray, k: int) -> Tuple[int, ...]:
    return tuple(sorted(int(index) for index in np.argsort(values)[-k:]))


def audit_conflict_taskset(
    taskset: Dict[str, object],
    audit_seeds: Sequence[int],
    samples: int = 161,
    duration: float = 6.0,
    selection_budget: int = 2,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Audit tolerance-induced rank conflicts without changing manifests."""

    rows = []
    for manifest in taskset["manifests"]:
        family = str(manifest["trajectory_family"])
        regime = str(manifest["regime"])
        reference = make_trajectory_family(
            family,
            samples=samples,
            duration=duration,
        )
        specification = specification_from_manifest(reference, manifest)
        for seed in audit_seeds:
            plant = make_virtual_machine_domain(int(seed))
            feedback = simulate_machine(reference.position, reference.dt, plant)
            contour_error = task_errors(reference, feedback)["contour"]
            absolute_peak = _zone_statistic(
                contour_error,
                specification,
                "max",
            )
            normalized_peak = absolute_peak / np.asarray(
                specification.tolerances,
                dtype=float,
            )
            error_selection = _top_k(absolute_peak, selection_budget)
            tolerance_selection = _top_k(normalized_peak, selection_budget)
            intersection = len(set(error_selection) & set(tolerance_selection))
            union = len(set(error_selection) | set(tolerance_selection))
            correlation = spearmanr(absolute_peak, normalized_peak).statistic
            rows.append(
                {
                    "manifest_id": str(manifest["manifest_id"]),
                    "trajectory": family,
                    "regime": regime,
                    "audit_seed": int(seed),
                    "absolute_error_top2": json.dumps(error_selection),
                    "tolerance_priority_top2": json.dumps(tolerance_selection),
                    "identical_top2": int(error_selection == tolerance_selection),
                    "top2_jaccard": float(intersection / union),
                    "rank_correlation": float(correlation),
                    "initial_zone_violation_rate": float(
                        np.mean(normalized_peak > 1.0)
                    ),
                }
            )

    summary = []
    for regime in TASK_REGIMES:
        selected = [row for row in rows if row["regime"] == regime]
        summary.append(
            {
                "regime": regime,
                "audit_cases": len(selected),
                "identical_top2_rate": float(
                    np.mean([row["identical_top2"] for row in selected])
                ),
                "selection_disagreement_rate": float(
                    1.0 - np.mean([row["identical_top2"] for row in selected])
                ),
                "mean_top2_jaccard": float(
                    np.mean([row["top2_jaccard"] for row in selected])
                ),
                "median_rank_correlation": float(
                    np.median([row["rank_correlation"] for row in selected])
                ),
                "median_initial_zone_violation_rate": float(
                    np.median(
                        [row["initial_zone_violation_rate"] for row in selected]
                    )
                ),
            }
        )
    return rows, summary


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_audit(path: Path, summary: Sequence[Dict[str, object]]) -> None:
    labels = [
        "Neutral",
        "Demand aligned",
        "Demand conflict",
    ]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    axes[0].bar(
        x,
        [item["selection_disagreement_rate"] for item in summary],
        color=["#a9b6bd", "#6d8f5f", "#b46b5a"],
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Selection disagreement rate")
    axes[0].set_title("Raw error vs tolerance priority")

    axes[1].bar(
        x,
        [item["mean_top2_jaccard"] for item in summary],
        color=["#a9b6bd", "#6d8f5f", "#b46b5a"],
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Mean top-2 Jaccard")
    axes[1].set_title("Priority-set overlap")

    axes[2].bar(
        x,
        [item["median_rank_correlation"] for item in summary],
        color=["#a9b6bd", "#6d8f5f", "#b46b5a"],
    )
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].axhline(0.0, color="#666666", linewidth=1.0)
    axes[2].set_ylabel("Median Spearman correlation")
    axes[2].set_title("Full ranking agreement")

    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_taskset_audit(
    output_directory: Path,
    taskset: Dict[str, object],
    audit_seeds: Sequence[int],
) -> Dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "task_manifests.json"
    save_conflict_taskset(manifest_path, taskset)
    rows, summary = audit_conflict_taskset(taskset, audit_seeds)
    _write_csv(output_directory / "audit_raw.csv", rows)
    payload = {
        "taskset_id": TASKSET_ID,
        "audit_only": True,
        "audit_seeds": [int(seed) for seed in audit_seeds],
        "manifests_were_modified_after_audit": False,
        "summary": summary,
        "readiness_criteria": {
            "neutral_identical_top2_rate_at_least_95_percent": (
                summary[0]["identical_top2_rate"] >= 0.95
            ),
            "conflict_selection_disagreement_at_least_70_percent": (
                summary[2]["selection_disagreement_rate"] >= 0.70
            ),
            "conflict_mean_top2_jaccard_no_more_than_50_percent": (
                summary[2]["mean_top2_jaccard"] <= 0.50
            ),
        },
    }
    payload["ready_for_formal_experiment"] = bool(
        all(payload["readiness_criteria"].values())
    )
    (output_directory / "audit_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _plot_audit(output_directory / "audit_summary.png", summary)

    report = f"""# Tolerance-conflict taskset audit

## Construction

- Taskset: `{TASKSET_ID}`
- Manifests: {taskset['manifest_count']} ({len(TRAJECTORY_FAMILIES)} trajectories x {len(TASK_REGIMES)} regimes)
- Zones per manifest: 6
- Machine feedback used to construct manifests: no
- Measured tracking error used to assign tolerances: no
- Audit-only machine seeds: {', '.join(str(seed) for seed in audit_seeds)}

## Regimes

- `neutral`: all zones use {NEUTRAL_TOLERANCE_MM:.2f} mm.
- `demand_aligned`: high programmed demand receives tighter tolerances.
- `demand_conflict`: high programmed demand receives looser tolerances and low-demand zones receive tighter tolerances.

## Audit result

| Regime | Selection disagreement | Mean top-2 Jaccard | Median rank correlation |
|---|---:|---:|---:|
| Neutral | {100.0 * summary[0]['selection_disagreement_rate']:.2f}% | {summary[0]['mean_top2_jaccard']:.3f} | {summary[0]['median_rank_correlation']:.3f} |
| Demand aligned | {100.0 * summary[1]['selection_disagreement_rate']:.2f}% | {summary[1]['mean_top2_jaccard']:.3f} | {summary[1]['median_rank_correlation']:.3f} |
| Demand conflict | {100.0 * summary[2]['selection_disagreement_rate']:.2f}% | {summary[2]['mean_top2_jaccard']:.3f} | {summary[2]['median_rank_correlation']:.3f} |

Ready for a new formal experiment: **{payload['ready_for_formal_experiment']}**

The audit measures whether raw absolute-error priority and tolerance-normalized
priority differ. Audit feedback never changes a manifest. These seeds must not
be reused for the formal algorithm comparison.
"""
    (output_directory / "README.md").write_text(report, encoding="utf-8")
    return payload
