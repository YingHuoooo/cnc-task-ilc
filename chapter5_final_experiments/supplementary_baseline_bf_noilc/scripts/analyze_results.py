"""Analyze the paired Proposed versus constrained BF-NOILC experiment."""

from __future__ import annotations

import csv
import json
import math
import os
import zlib
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bf-noilc-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 20260821
PROPOSED = "proposed"
BF_NOILC = "parameter_matched_constrained_bf_noilc"
SCENARIOS = (
    ("n0_d0_m0", "Baseline"),
    ("delay_2", "Added delay +2"),
    ("n0_d1_m0", "Added delay +4"),
    ("n1_d1_m1", "Triple stress"),
)
SCOPES = (
    ("demand_conflict", "Demand-conflict"),
    ("all_tasks", "All 15 tasks"),
)
PERFORMANCE_METRICS = (
    ("task_auc_normalized", "Task AUC"),
    ("global_rmse_auc_normalized", "Global RMSE AUC"),
    ("worst_zone_auc_normalized", "Worst-zone AUC"),
    ("final_task_ratio", "Final task ratio"),
    ("final_global_ratio", "Final global RMSE ratio"),
    ("final_worst_zone_ratio_relative", "Final worst-zone ratio"),
)
EFFORT_METRICS = (
    ("cumulative_delta_theta_l2", "Cumulative $||\\Delta\\theta||_2$"),
    ("cumulative_learned_delta_u_l2", "Cumulative learned $||\\Delta u||_2$"),
    ("cumulative_issued_delta_u_l2", "Cumulative issued $||\\Delta u||_2$"),
    ("final_command_correction_l2", "Final command-correction $L_2$"),
)
COLORS = {"proposed": "#155A9C", "bf": "#B06C35", "ink": "#242424"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(label: str) -> int:
    return BOOTSTRAP_SEED + int(zlib.crc32(label.encode("utf-8")) % 100000)


def scoped(rows: Sequence[Mapping[str, str]], scenario: str, scope: str):
    return [
        row
        for row in rows
        if row["scenario_id"] == scenario
        and (scope == "all_tasks" or row["regime"] == scope)
    ]


def paired_matrix(
    rows: Sequence[Mapping[str, str]],
    scenario: str,
    scope: str,
    metric: str,
    effort: bool = False,
) -> Tuple[np.ndarray, List[str], List[str], List[Dict[str, object]]]:
    selected = scoped(rows, scenario, scope)
    indexed = {
        (row["plant_id"], row["manifest_id"], row["method"]): row
        for row in selected
    }
    plants = sorted({row["plant_id"] for row in selected})
    tasks = sorted({row["manifest_id"] for row in selected})
    matrix = np.empty((len(plants), len(tasks)), dtype=float)
    records: List[Dict[str, object]] = []
    for i, plant in enumerate(plants):
        for j, task in enumerate(tasks):
            proposed = float(indexed[(plant, task, PROPOSED)][metric])
            baseline = float(indexed[(plant, task, BF_NOILC)][metric])
            if effort:
                value = 100.0 * (proposed - baseline) / max(abs(baseline), 1.0e-12)
            else:
                value = 100.0 * (baseline - proposed) / max(abs(baseline), 1.0e-12)
            matrix[i, j] = value
            records.append(
                {
                    "plant_id": plant,
                    "manifest_id": task,
                    "value": value,
                    "proposed": proposed,
                    "baseline": baseline,
                }
            )
    return matrix, plants, tasks, records


def interval(values: np.ndarray) -> Tuple[float, float]:
    low, high = np.percentile(values, (2.5, 97.5))
    return float(low), float(high)


def bootstrap(matrix: np.ndarray, label: str) -> Dict[str, float]:
    plants, tasks = matrix.shape
    flat = matrix.ravel()
    rng = np.random.default_rng(stable_seed(label))
    paired_draws = np.empty(BOOTSTRAP_REPLICATES)
    plant_draws = np.empty(BOOTSTRAP_REPLICATES)
    hierarchical_draws = np.empty(BOOTSTRAP_REPLICATES)
    batch_size = 250
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        stop = min(BOOTSTRAP_REPLICATES, start + batch_size)
        count = stop - start
        paired_index = rng.integers(0, flat.size, size=(count, flat.size))
        paired_draws[start:stop] = np.median(flat[paired_index], axis=1)
        plant_index = rng.integers(0, plants, size=(count, plants))
        plant_values = matrix[plant_index]
        plant_draws[start:stop] = np.median(
            plant_values.reshape(count, -1), axis=1
        )
        task_index = rng.integers(
            0, tasks, size=(count, plants, tasks)
        )
        sampled = np.take_along_axis(plant_values, task_index, axis=2)
        hierarchical_draws[start:stop] = np.median(
            sampled.reshape(count, -1), axis=1
        )
    paired_ci = interval(paired_draws)
    plant_ci = interval(plant_draws)
    hierarchy_ci = interval(hierarchical_draws)
    leave_one = [
        float(np.median(np.delete(matrix, index, axis=0)))
        for index in range(plants)
    ]
    return {
        "paired_n": int(flat.size),
        "plant_n": int(plants),
        "median": float(np.median(flat)),
        "mean": float(np.mean(flat)),
        "paired_ci_low": paired_ci[0],
        "paired_ci_high": paired_ci[1],
        "plant_ci_low": plant_ci[0],
        "plant_ci_high": plant_ci[1],
        "hierarchical_ci_low": hierarchy_ci[0],
        "hierarchical_ci_high": hierarchy_ci[1],
        "win_rate": float(np.mean(flat > 0.0)),
        "tie_rate": float(np.mean(np.isclose(flat, 0.0, atol=1.0e-12, rtol=0.0))),
        "leave_one_plant_min": float(min(leave_one)),
        "leave_one_plant_max": float(max(leave_one)),
    }


def analyze_runs(rows: Sequence[Mapping[str, str]]):
    performance = []
    effort = []
    plant_rows = []
    pair_rows = []
    for scenario, scenario_label in SCENARIOS:
        for scope, scope_label in SCOPES:
            for metric, metric_label in PERFORMANCE_METRICS:
                matrix, plants, tasks, records = paired_matrix(
                    rows, scenario, scope, metric, effort=False
                )
                stats = bootstrap(matrix, f"performance-{scenario}-{scope}-{metric}")
                stats.update(
                    {
                        "scenario_id": scenario,
                        "scenario_label": scenario_label,
                        "scope": scope,
                        "scope_label": scope_label,
                        "metric": metric,
                        "metric_label": metric_label,
                        "effect_definition": "100*(BF_NOILC-Proposed)/BF_NOILC; positive favors Proposed",
                    }
                )
                performance.append(stats)
                if metric == "task_auc_normalized":
                    for plant_index, plant in enumerate(plants):
                        plant_rows.append(
                            {
                                "scenario_id": scenario,
                                "scope": scope,
                                "plant_id": plant,
                                "plant_median_task_auc_effect_percent": float(
                                    np.median(matrix[plant_index])
                                ),
                            }
                        )
                    for item in records:
                        pair_rows.append(
                            {
                                "scenario_id": scenario,
                                "scope": scope,
                                **item,
                            }
                        )
            for metric, metric_label in EFFORT_METRICS:
                matrix, _, _, _ = paired_matrix(
                    rows, scenario, scope, metric, effort=True
                )
                stats = bootstrap(matrix, f"effort-{scenario}-{scope}-{metric}")
                stats.update(
                    {
                        "scenario_id": scenario,
                        "scenario_label": scenario_label,
                        "scope": scope,
                        "scope_label": scope_label,
                        "metric": metric,
                        "metric_label": metric_label,
                        "effect_definition": "100*(Proposed-BF_NOILC)/BF_NOILC; positive means greater Proposed effort",
                    }
                )
                effort.append(stats)
    return performance, effort, plant_rows, pair_rows


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            FIGURES / f"{name}.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def forest_plot(performance: Sequence[Mapping[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2), sharex=True)
    for ax, (scope, scope_label) in zip(axes, SCOPES):
        rows = [
            row
            for row in performance
            if row["scope"] == scope and row["metric"] == "task_auc_normalized"
        ]
        y = np.arange(len(rows))[::-1]
        values = np.asarray([float(row["median"]) for row in rows])
        low = np.asarray([float(row["hierarchical_ci_low"]) for row in rows])
        high = np.asarray([float(row["hierarchical_ci_high"]) for row in rows])
        ax.errorbar(
            values,
            y,
            xerr=np.vstack((values - low, high - values)),
            fmt="o",
            color=COLORS["proposed"],
            capsize=2.5,
            lw=1.2,
        )
        ax.axvline(0.0, color="#777777", lw=0.9, ls="--")
        ax.set_yticks(y, [row["scenario_label"] for row in rows])
        ax.set_title(scope_label, loc="left", fontweight="bold")
        ax.set_xlabel("Proposed task-AUC improvement (%)")
        ax.grid(axis="x", color="#E6E6E6", lw=0.7)
    fig.suptitle(
        "Proposed vs parameter-matched constrained BF-NOILC\n"
        "median paired effect with hierarchical-bootstrap 95% CI",
        x=0.08,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "paired_task_auc_effects")


def effort_plot(effort: Sequence[Mapping[str, object]]) -> None:
    chosen = [
        row
        for row in effort
        if row["scope"] == "demand_conflict"
        and row["metric"]
        in {"cumulative_delta_theta_l2", "cumulative_learned_delta_u_l2"}
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(SCENARIOS))
    width = 0.34
    for offset, metric in zip(
        (-width / 2, width / 2),
        ("cumulative_delta_theta_l2", "cumulative_learned_delta_u_l2"),
    ):
        rows = [row for row in chosen if row["metric"] == metric]
        values = [float(row["median"]) for row in rows]
        label = next(name for key, name in EFFORT_METRICS if key == metric)
        ax.bar(x + offset, values, width=width, label=label)
    ax.axhline(0.0, color="#777777", lw=0.9)
    ax.set_xticks(x, [label for _, label in SCENARIOS])
    ax.set_ylabel("Proposed effort difference vs BF-NOILC (%)")
    ax.set_title("Demand-conflict update/control effort", loc="left", fontweight="bold")
    ax.legend(ncols=2, fontsize=7)
    ax.grid(axis="y", color="#E6E6E6", lw=0.7)
    fig.tight_layout()
    save_figure(fig, "control_effort_comparison")


def learning_curves(trials: Sequence[Mapping[str, str]]) -> None:
    selected = [row for row in trials if row["regime"] == "demand_conflict"]
    initial = {
        (row["job_id"]): float(row["task_score"])
        for row in selected
        if int(row["trial"]) == 0
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.0), sharex=True, sharey=True)
    for ax, (scenario, label) in zip(axes.ravel(), SCENARIOS):
        for method, color, method_label in (
            (PROPOSED, COLORS["proposed"], "Proposed"),
            (BF_NOILC, COLORS["bf"], "Parameter-matched constrained BF-NOILC"),
        ):
            medians = []
            q25 = []
            q75 = []
            for trial in range(5):
                values = [
                    float(row["task_score"]) / max(initial[row["job_id"]], 1.0e-12)
                    for row in selected
                    if row["scenario_id"] == scenario
                    and row["method"] == method
                    and int(row["trial"]) == trial
                ]
                medians.append(float(np.median(values)))
                q25.append(float(np.percentile(values, 25)))
                q75.append(float(np.percentile(values, 75)))
            x = np.arange(5)
            ax.plot(x, medians, marker="o", color=color, label=method_label)
            ax.fill_between(x, q25, q75, color=color, alpha=0.16)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.grid(color="#E6E6E6", lw=0.7)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Trial")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized task score")
    fig.suptitle("Demand-conflict finite-trial learning curves", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "demand_conflict_learning_curves")


def interpretation(row: Mapping[str, object]) -> str:
    median = float(row["median"])
    low = float(row["hierarchical_ci_low"])
    high = float(row["hierarchical_ci_high"])
    if median > 0.0 and low > 0.0:
        return "supports Proposed superiority"
    if median < 0.0 and high < 0.0:
        return "supports BF-NOILC superiority"
    if median > 0.0:
        return "direction favors Proposed; interval crosses zero"
    if median < 0.0:
        return "direction favors BF-NOILC; interval crosses zero"
    return "no median difference; not an equivalence test"


def write_report(
    performance: Sequence[Mapping[str, object]],
    effort: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, str]],
) -> None:
    primary = [
        row
        for row in performance
        if row["scope"] == "demand_conflict"
        and row["metric"] == "task_auc_normalized"
    ]
    global_rows = [
        row
        for row in performance
        if row["scope"] == "demand_conflict"
        and row["metric"] == "global_rmse_auc_normalized"
    ]
    final_task_rows = [
        row
        for row in performance
        if row["scope"] == "demand_conflict"
        and row["metric"] == "final_task_ratio"
    ]
    final_worst_rows = [
        row
        for row in performance
        if row["scope"] == "demand_conflict"
        and row["metric"] == "final_worst_zone_ratio_relative"
    ]
    learned_effort = [
        row
        for row in effort
        if row["scope"] == "demand_conflict"
        and row["metric"] == "cumulative_learned_delta_u_l2"
    ]
    lookup_global = {row["scenario_id"]: row for row in global_rows}
    lookup_effort = {row["scenario_id"]: row for row in learned_effort}
    lookup_final = {row["scenario_id"]: row for row in final_task_rows}
    lookup_final_worst = {row["scenario_id"]: row for row in final_worst_rows}
    lines = [
        "# Supplementary constrained BF-NOILC comparison",
        "",
        "This is a prospectively specified supplementary comparison on eight new plants. It is separate from the original V11 formal protocol and Table 3.",
        "",
        "## Primary demand-conflict result",
        "",
        "| Condition | Task-AUC effect | Hierarchical 95% CI | Win rate | Interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    for row in primary:
        lines.append(
            "| {scenario_label} | {median:.3f}% | [{hierarchical_ci_low:.3f}%, {hierarchical_ci_high:.3f}%] | {win:.1f}% | {text} |".format(
                **row,
                win=100.0 * float(row["win_rate"]),
                text=interpretation(row),
            )
        )
    lines.extend(
        [
            "",
            "## Finite-budget AUC versus final-trial performance",
            "",
            "| Condition | Task-AUC effect | Final-task-ratio effect | Final worst-zone effect |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in primary:
        final_row = lookup_final[row["scenario_id"]]
        worst_row = lookup_final_worst[row["scenario_id"]]
        lines.append(
            f"| {row['scenario_label']} | {float(row['median']):.3f}% "
            f"[{float(row['hierarchical_ci_low']):.3f}%, {float(row['hierarchical_ci_high']):.3f}%] | "
            f"{float(final_row['median']):.3f}% "
            f"[{float(final_row['hierarchical_ci_low']):.3f}%, {float(final_row['hierarchical_ci_high']):.3f}%] | "
            f"{float(worst_row['median']):.3f}% "
            f"[{float(worst_row['hierarchical_ci_low']):.3f}%, {float(worst_row['hierarchical_ci_high']):.3f}%] |"
        )
    lines.extend(
        [
            "",
            "Positive task-AUC and global-RMSE effects favor Proposed. Positive effort differences mean Proposed used more update effort than BF-NOILC.",
            "",
            "## Objective and effort trade-off",
            "",
            "| Condition | Task-AUC median effect | Global-RMSE-AUC median effect | Learned command-update effort difference |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in primary:
        global_row = lookup_global[row["scenario_id"]]
        effort_row = lookup_effort[row["scenario_id"]]
        lines.append(
            f"| {row['scenario_label']} | {float(row['median']):.3f}% | "
            f"{float(global_row['median']):.3f}% | {float(effort_row['median']):.3f}% |"
        )
    valid = all(int(row["finite_result"]) == 1 for row in rows)
    solver = all(int(row["all_updates_succeeded"]) == 1 for row in rows)
    constraints = all(
        int(row["all_update_constraints_satisfied"]) == 1
        and int(row["final_constraint_violation"]) == 0
        for row in rows
    )
    lines.extend(
        [
            "",
            "## Numerical validation",
            "",
            f"- Finite results: {'PASS' if valid else 'FAIL'}",
            f"- Solver success for every update: {'PASS' if solver else 'FAIL'}",
            f"- Implemented motion constraints: {'PASS' if constraints else 'FAIL'}",
            "",
            "The comparison is configuration-level. It does not attribute any observed difference solely to selection, temporal alignment, the relaxation factor, or rollback.",
            "",
            "## Evidence interpretation",
            "",
            "- Proposed superiority on the primary finite-budget task AUC is not established in any of the four conditions.",
            "- BF-NOILC superiority on task AUC is supported at baseline. Under added delay, the BF-NOILC direction remains favorable for AUC, but the demand-conflict hierarchical intervals cross zero.",
            "- Proposed has a later-trial advantage under temporal mismatch: the final task ratio is significantly better at added delay +4, and the final worst-zone ratio is significantly better at added delay +4 and triple stress.",
            "- BF-NOILC is significantly better on global RMSE AUC in all four demand-conflict conditions, consistent with its full-trajectory quadratic objective.",
            "- Proposed uses significantly less coefficient and learned command-update effort in all four conditions. The result is therefore a speed/aggressiveness versus delayed-condition endpoint trade-off, not across-the-board superiority of either configuration.",
            "",
        ]
    )
    (ROOT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = read_csv(RESULTS / "raw_runs.csv")
    trials = read_csv(RESULTS / "trial_history.csv")
    performance, effort, plant_rows, pair_rows = analyze_runs(rows)
    write_csv(RESULTS / "comparison_summary.csv", performance)
    write_csv(RESULTS / "control_effort_summary.csv", effort)
    write_csv(RESULTS / "plant_level_effects.csv", plant_rows)
    write_csv(RESULTS / "paired_task_auc_effects.csv", pair_rows)
    (RESULTS / "comparison_summary.json").write_text(
        json.dumps(
            {
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "performance": performance,
                "control_effort": effort,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    forest_plot(performance)
    effort_plot(effort)
    learning_curves(trials)
    write_report(performance, effort, rows)
    print("analysis complete")


if __name__ == "__main__":
    main()
