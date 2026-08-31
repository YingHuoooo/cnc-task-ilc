"""End-to-end feasibility experiment and artifact generation."""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(_PROJECT_ROOT / ".matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .basis import cubic_bspline_basis
from .critical_windows import select_critical_windows
from .ilc import ILCConfig, ILCResult, run_ilc
from .metrics import (
    command_kinematics,
    constraint_report,
    task_errors,
)
from .plant import (
    mismatched_virtual_machine,
    nominal_config,
    simulate_machine,
)
from .trajectory import ReferenceTrajectory, make_reference_trajectory


def _percentage_reduction(initial: float, final: float) -> float:
    return float(100.0 * (initial - final) / max(abs(initial), 1.0e-12))


def _method_summary(result: ILCResult) -> Dict[str, object]:
    initial = result.metrics[0]
    final = result.metrics[-1]
    return {
        "initial": initial,
        "final": final,
        "critical_max_reduction_percent": _percentage_reduction(
            initial["critical_max_abs"],
            final["critical_max_abs"],
        ),
        "critical_rmse_reduction_percent": _percentage_reduction(
            initial["critical_rmse"],
            final["critical_rmse"],
        ),
        "global_rmse_reduction_percent": _percentage_reduction(
            initial["global_rmse"],
            final["global_rmse"],
        ),
        "all_qp_updates_succeeded": all(
            status["success"] for status in result.solver_status
        ),
    }


def _save_learning_curves(
    output_path: Path,
    results: Iterable[ILCResult],
) -> None:
    fieldnames = [
        "method",
        "trial",
        "global_rmse",
        "global_max_abs",
        "critical_rmse",
        "critical_max_abs",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for metric in result.metrics:
                writer.writerow(
                    {
                        "method": result.name,
                        "trial": int(metric["trial"]),
                        "global_rmse": metric["global_rmse"],
                        "global_max_abs": metric["global_max_abs"],
                        "critical_rmse": metric["critical_rmse"],
                        "critical_max_abs": metric["critical_max_abs"],
                    }
                )


def _shade_windows(
    axis: plt.Axes,
    reference: ReferenceTrajectory,
    windows: Iterable[Iterable[int]],
) -> None:
    first = True
    for start, stop in windows:
        axis.axvspan(
            reference.time[int(start)],
            reference.time[int(stop)],
            color="tab:orange",
            alpha=0.15,
            label="auto critical windows" if first else None,
        )
        first = False


def _plot_results(
    output_path: Path,
    reference: ReferenceTrajectory,
    windows: Iterable[Iterable[int]],
    full_result: ILCResult,
    critical_result: ILCResult,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))

    axis = axes[0, 0]
    axis.plot(
        reference.position[:, 0],
        reference.position[:, 1],
        color="black",
        linewidth=2.0,
        label="reference",
    )
    axis.plot(
        critical_result.feedbacks[0][:, 0],
        critical_result.feedbacks[0][:, 1],
        color="tab:red",
        alpha=0.75,
        label="trial 0",
    )
    axis.plot(
        critical_result.feedbacks[-1][:, 0],
        critical_result.feedbacks[-1][:, 1],
        color="tab:blue",
        alpha=0.90,
        label="critical-window final",
    )
    axis.set_title("Task-space contour")
    axis.set_xlabel("X position [mm]")
    axis.set_ylabel("Y position [mm]")
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = axes[0, 1]
    axis.plot(
        reference.time,
        critical_result.contour_errors[0],
        color="tab:red",
        label="trial 0",
    )
    axis.plot(
        reference.time,
        critical_result.contour_errors[-1],
        color="tab:blue",
        label="critical-window final",
    )
    _shade_windows(axis, reference, windows)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Contour error and selected windows")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Normal contour error [mm]")
    axis.grid(alpha=0.25)
    axis.legend()

    trials = np.arange(len(full_result.metrics))
    axis = axes[1, 0]
    axis.plot(
        trials,
        [item["critical_max_abs"] for item in full_result.metrics],
        marker="o",
        label="full-trajectory ILC",
    )
    axis.plot(
        trials,
        [item["critical_max_abs"] for item in critical_result.metrics],
        marker="s",
        label="critical-window ILC",
    )
    axis.set_title("Critical-window peak error")
    axis.set_xlabel("Trial")
    axis.set_ylabel("Maximum absolute error [mm]")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = axes[1, 1]
    axis.plot(
        trials,
        [item["global_rmse"] for item in full_result.metrics],
        marker="o",
        label="full-trajectory ILC",
    )
    axis.plot(
        trials,
        [item["global_rmse"] for item in critical_result.metrics],
        marker="s",
        label="critical-window ILC",
    )
    axis.set_title("Global contour RMSE")
    axis.set_xlabel("Trial")
    axis.set_ylabel("RMSE [mm]")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.suptitle(
        "Task-Level ILC feasibility under learner-plant model mismatch",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    output_path: Path,
    metrics: Dict[str, object],
) -> None:
    critical = metrics["methods"]["critical_window_ilc"]
    full = metrics["methods"]["full_trajectory_ilc"]
    comparison = metrics["comparison"]
    report = """# Feasibility Demo Report

## Scope

This is an offline two-axis simulation of the core Task-Level ILC route. The
learner uses a low-order, delay-free and decoupled model. The evaluator contains
axis mismatch, delay, friction, saturation, cross-axis coupling and a
repeatable disturbance.

## Result

- Initial critical-window maximum error: {initial_critical:.6f} mm
- Final critical-window maximum error: {final_critical:.6f} mm
- Critical-window peak reduction: {critical_reduction:.2f}%
- Full-trajectory ILC peak reduction on the same windows: {full_reduction:.2f}%
- Final critical RMSE advantage over full-trajectory ILC: {rmse_advantage:.2f}%
- Final global RMSE for critical-window ILC: {final_global:.6f} mm
- All constrained updates solved: {qp_success}
- Configured constraints violated: {constraint_violation}
- Feasibility checks passed: {feasibility_passed}

## Interpretation

The result demonstrates that measured task error from a structurally mismatched
virtual machine can be mapped through an approximate nominal-model sensitivity
and a constrained low-dimensional update to improve the next repeated trial.
This is evidence for algorithmic feasibility, not evidence of physical machine
or cutting-process performance.
""".format(
        initial_critical=critical["initial"]["critical_max_abs"],
        final_critical=critical["final"]["critical_max_abs"],
        critical_reduction=critical["critical_max_reduction_percent"],
        full_reduction=full["critical_max_reduction_percent"],
        rmse_advantage=comparison[
            "critical_final_rmse_advantage_percent_vs_full"
        ],
        final_global=critical["final"]["global_rmse"],
        qp_success=critical["all_qp_updates_succeeded"],
        constraint_violation=metrics["critical_window_constraints"][
            "constraint_violation"
        ],
        feasibility_passed=metrics["feasibility_passed"],
    )
    output_path.write_text(report, encoding="utf-8")


def run_demo(output_directory: Path) -> Dict[str, object]:
    """Run the comparison and write reproducible result artifacts."""

    output_directory.mkdir(parents=True, exist_ok=True)
    reference = make_reference_trajectory(samples=321, duration=6.0)
    initial_command = reference.position.copy()
    basis = cubic_bspline_basis(
        samples=reference.time.size,
        control_points=18,
    )
    nominal_model = nominal_config()
    virtual_plant = mismatched_virtual_machine()

    initial_feedback = simulate_machine(
        initial_command,
        reference.dt,
        virtual_plant,
    )
    initial_error = task_errors(reference, initial_feedback)["contour"]
    selection = select_critical_windows(
        reference,
        initial_error,
        number_of_windows=3,
        half_width=13,
    )

    initial_kinematics = command_kinematics(initial_command, reference.dt)
    velocity_limit = float(
        1.30 * np.max(np.abs(initial_kinematics["velocity"]))
    )
    acceleration_limit = float(
        1.55 * np.max(np.abs(initial_kinematics["acceleration"]))
    )
    config = ILCConfig(
        iterations=8,
        correction_limit=4.0,
        velocity_limit=velocity_limit,
        acceleration_limit=acceleration_limit,
        regularization=2.0e-3,
        smoothness=2.0e-8,
        learning_rate=0.68,
        global_protection_weight=0.18,
        critical_boost=5.0,
    )

    full_result = run_ilc(
        name="full_trajectory_ilc",
        reference=reference,
        initial_command=initial_command,
        basis=basis,
        evaluation_mask=selection.mask,
        optimization_mask=selection.mask,
        nominal_model=nominal_model,
        virtual_plant=virtual_plant,
        config=config,
        critical_weighting=False,
    )
    critical_result = run_ilc(
        name="critical_window_ilc",
        reference=reference,
        initial_command=initial_command,
        basis=basis,
        evaluation_mask=selection.mask,
        optimization_mask=selection.mask,
        nominal_model=nominal_model,
        virtual_plant=virtual_plant,
        config=config,
        critical_weighting=True,
    )

    constraint_metrics = constraint_report(
        initial_command=initial_command,
        learned_command=critical_result.commands[-1],
        dt=reference.dt,
        max_correction=config.correction_limit,
        velocity_limit=config.velocity_limit,
        acceleration_limit=config.acceleration_limit,
    )
    full_summary = _method_summary(full_result)
    critical_summary = _method_summary(critical_result)
    comparison = {
        "critical_final_rmse_advantage_percent_vs_full": (
            _percentage_reduction(
                full_summary["final"]["critical_rmse"],
                critical_summary["final"]["critical_rmse"],
            )
        ),
        "critical_final_peak_advantage_percent_vs_full": (
            _percentage_reduction(
                full_summary["final"]["critical_max_abs"],
                critical_summary["final"]["critical_max_abs"],
            )
        ),
        "critical_global_rmse_tradeoff_percent_vs_full": float(
            100.0
            * (
                critical_summary["final"]["global_rmse"]
                - full_summary["final"]["global_rmse"]
            )
            / max(full_summary["final"]["global_rmse"], 1.0e-12)
        ),
    }
    feasibility_passed = bool(
        critical_summary["critical_max_reduction_percent"] > 0.0
        and critical_summary["critical_rmse_reduction_percent"] > 0.0
        and critical_summary["all_qp_updates_succeeded"]
        and constraint_metrics["constraint_violation"] == 0
    )
    metrics: Dict[str, object] = {
        "demo": {
            "samples": int(reference.time.size),
            "duration_s": float(reference.time[-1]),
            "iterations": int(config.iterations),
            "bspline_variables": int(2 * basis.shape[1]),
            "critical_windows": selection.windows,
            "critical_window_fraction": float(np.mean(selection.mask)),
            "learner_model": "linear, decoupled, no delay, no friction",
            "virtual_plant": (
                "axis mismatch, delay, friction, saturation, coupling, "
                "repeatable disturbance"
            ),
        },
        "methods": {
            full_result.name: full_summary,
            critical_result.name: critical_summary,
        },
        "comparison": comparison,
        "critical_window_constraints": constraint_metrics,
        "feasibility_passed": feasibility_passed,
    }
    metrics_path = output_directory / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _save_learning_curves(
        output_directory / "learning_curves.csv",
        (full_result, critical_result),
    )
    _plot_results(
        output_directory / "demo_summary.png",
        reference,
        selection.windows,
        full_result,
        critical_result,
    )
    np.savez_compressed(
        output_directory / "trial_data.npz",
        time=reference.time,
        reference=reference.position,
        critical_score=selection.score,
        critical_mask=selection.mask,
        full_commands=np.stack(full_result.commands),
        full_feedbacks=np.stack(full_result.feedbacks),
        full_errors=np.stack(full_result.contour_errors),
        critical_commands=np.stack(critical_result.commands),
        critical_feedbacks=np.stack(critical_result.feedbacks),
        critical_errors=np.stack(critical_result.contour_errors),
    )
    _write_report(output_directory / "demo_report.md", metrics)
    return metrics
