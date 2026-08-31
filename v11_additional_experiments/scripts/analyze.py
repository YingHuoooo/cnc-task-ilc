"""Analyze all five priorities, create the replay and publication figures."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/v11-additional-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parent
PROJECT = WORKSPACE / "cnc_task_ilc"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(HERE))

from cnc_task_ilc.basis import cubic_bspline_basis
from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.conflict_taskset import specification_from_manifest
from cnc_task_ilc.ilc import build_contour_sensitivity
from cnc_task_ilc.metrics import task_errors
from cnc_task_ilc.plant import make_virtual_machine_domain, nominal_config
from cnc_task_ilc.trajectory import make_trajectory_family

from experiment_core import run_matched_method, scenario_from_dict
from run_experiments import build_ablation_jobs, load_manifests


BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 20260820
COLORS = {
    "v11": "#155A9C",
    "no": "#7B86B6",
    "task": "#3F9D86",
    "raw": "#C58A2B",
    "uniform": "#7A7A7A",
    "negative": "#BB4B48",
    "light": "#D7DCE5",
    "ink": "#252525",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "svg.hashsalt": "v11-additional-experiments",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def seed_for(label: str) -> int:
    return BOOTSTRAP_SEED + int(zlib.crc32(label.encode("utf-8")) % 100000)


def percentile_interval(values: np.ndarray) -> Tuple[float, float]:
    low, high = np.percentile(values, (2.5, 97.5))
    return float(low), float(high)


def paired_effects(
    rows: Sequence[Mapping[str, str]],
    proposed: str,
    comparator: str,
    scenario_id: str,
    regime: Optional[str] = None,
    plant_group: Optional[str] = None,
) -> List[Dict[str, object]]:
    scoped = [
        row
        for row in rows
        if row["scenario_id"] == scenario_id
        and (regime is None or row["regime"] == regime)
        and (plant_group is None or row["plant_group"] == plant_group)
    ]
    indexed = {
        (row["manifest_id"], row["plant_id"], row["method"]): row for row in scoped
    }
    base_keys = sorted({(key[0], key[1]) for key in indexed})
    records = []
    for manifest_id, plant_id in base_keys:
        pkey = (manifest_id, plant_id, proposed)
        ckey = (manifest_id, plant_id, comparator)
        if pkey not in indexed or ckey not in indexed:
            continue
        proposed_auc = float(indexed[pkey]["task_auc_normalized"])
        comparator_auc = float(indexed[ckey]["task_auc_normalized"])
        records.append(
            {
                "manifest_id": manifest_id,
                "plant_id": plant_id,
                "effect_percent": 100.0 * (comparator_auc - proposed_auc) / comparator_auc,
                "proposed_auc": proposed_auc,
                "comparator_auc": comparator_auc,
            }
        )
    return records


def bootstrap_statistics(records: Sequence[Mapping[str, object]], label: str) -> Dict[str, object]:
    effects = np.asarray([float(row["effect_percent"]) for row in records])
    domains = np.asarray([str(row["plant_id"]) for row in records])
    unique_domains = sorted(set(domains))
    by_domain = {domain: effects[domains == domain] for domain in unique_domains}
    rng = np.random.RandomState(seed_for(label))

    paired_draws = np.empty(BOOTSTRAP_REPLICATES)
    domain_draws = np.empty(BOOTSTRAP_REPLICATES)
    hierarchy_draws = np.empty(BOOTSTRAP_REPLICATES)
    for start in range(0, BOOTSTRAP_REPLICATES, 1000):
        stop = min(BOOTSTRAP_REPLICATES, start + 1000)
        indices = rng.randint(0, len(effects), size=(stop - start, len(effects)))
        paired_draws[start:stop] = np.median(effects[indices], axis=1)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled_domains = rng.choice(unique_domains, size=len(unique_domains), replace=True)
        domain_values = np.concatenate([by_domain[item] for item in sampled_domains])
        domain_draws[index] = np.median(domain_values)
        hierarchical_values = np.concatenate(
            [
                rng.choice(by_domain[item], size=len(by_domain[item]), replace=True)
                for item in sampled_domains
            ]
        )
        hierarchy_draws[index] = np.median(hierarchical_values)

    leave_one = []
    for domain in unique_domains:
        retained = effects[domains != domain]
        if retained.size:
            leave_one.append(float(np.median(retained)))
    paired_ci = percentile_interval(paired_draws)
    domain_ci = percentile_interval(domain_draws)
    hierarchical_ci = percentile_interval(hierarchy_draws)
    return {
        "paired_n": int(len(effects)),
        "domain_n": int(len(unique_domains)),
        "median_effect_percent": float(np.median(effects)),
        "mean_effect_percent": float(np.mean(effects)),
        "paired_ci_low": paired_ci[0],
        "paired_ci_high": paired_ci[1],
        "domain_ci_low": domain_ci[0],
        "domain_ci_high": domain_ci[1],
        "hierarchical_ci_low": hierarchical_ci[0],
        "hierarchical_ci_high": hierarchical_ci[1],
        "pair_win_rate": float(np.mean(effects > 0.0)),
        "leave_one_domain_min": float(min(leave_one)) if leave_one else math.nan,
        "leave_one_domain_max": float(max(leave_one)) if leave_one else math.nan,
    }


def analyze_ablation(rows: Sequence[Mapping[str, str]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    comparators = (
        "no_residual_alignment",
        "task_top2",
        "raw_top2",
        "uniform_full_trajectory",
    )
    scenarios = ("n0_d0_m0", "n0_d1_m0", "n1_d1_m1")
    summary = []
    stats = []
    for scenario in scenarios:
        for comparator in comparators:
            for regime in ("demand_conflict", None):
                records = paired_effects(rows, "v11_full", comparator, scenario, regime)
                effects = np.asarray([float(item["effect_percent"]) for item in records])
                rng = np.random.RandomState(seed_for(f"ablation-simple-{scenario}-{comparator}-{regime}"))
                draws = np.median(
                    effects[rng.randint(0, len(effects), size=(BOOTSTRAP_REPLICATES, len(effects)))],
                    axis=1,
                )
                ci = percentile_interval(draws)
                summary.append(
                    {
                        "scenario_id": scenario,
                        "scope": regime or "all_tasks",
                        "proposed": "v11_full",
                        "comparator": comparator,
                        "paired_n": len(effects),
                        "median_effect_percent": float(np.median(effects)),
                        "ci95_low": ci[0],
                        "ci95_high": ci[1],
                        "pair_win_rate": float(np.mean(effects > 0.0)),
                    }
                )
                if regime == "demand_conflict":
                    stat = bootstrap_statistics(
                        records, f"ablation-{scenario}-{comparator}"
                    )
                    stat.update(
                        {
                            "dataset": "matched_ablation",
                            "scenario_id": scenario,
                            "scope": regime,
                            "proposed": "v11_full",
                            "comparator": comparator,
                        }
                    )
                    stats.append(stat)
    return summary, stats


def analyze_sensitivity(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    scenarios = ("n0_d1_m0", "n1_d1_m1")
    reference = {
        "residual_delay_shrinkage": "0.0",
        "smoothing_window": "5",
        "nominal_control_points": "12",
        "learning_rate": "0.65",
    }
    display_values = {
        "residual_delay_shrinkage": ["0.0", "0.1", "0.25", "0.4", "0.6", "1.0"],
        "smoothing_window": ["3", "5", "7", "9"],
        "nominal_control_points": ["8", "12", "16"],
        "learning_rate": ["0.5", "0.65", "0.8"],
    }
    standard_rows = [
        row
        for row in rows
        if row["config_group"] == "residual_delay_shrinkage"
        and math.isclose(float(row["config_value"]), 0.25)
    ]
    output = []
    for scenario in scenarios:
        for group, values in display_values.items():
            group_rows = [row for row in rows if row["config_group"] == group]
            if group != "residual_delay_shrinkage":
                default_value = {
                    "smoothing_window": "5",
                    "nominal_control_points": "12",
                    "learning_rate": "0.65",
                }[group]
                group_rows = group_rows + [
                    dict(row, config_group=group, config_value=default_value)
                    for row in standard_rows
                ]
            scoped = [row for row in group_rows if row["scenario_id"] == scenario]
            indexed = {
                (row["manifest_id"], row["plant_id"], row["config_value"]): row
                for row in scoped
            }
            base_keys = sorted({(key[0], key[1]) for key in indexed})
            ref_value = reference[group]
            for value in values:
                effects = []
                aucs = []
                successes = []
                for manifest_id, plant_id in base_keys:
                    key = (manifest_id, plant_id, value)
                    ref_key = (manifest_id, plant_id, ref_value)
                    if key not in indexed or ref_key not in indexed:
                        continue
                    auc = float(indexed[key]["task_auc_normalized"])
                    ref_auc = float(indexed[ref_key]["task_auc_normalized"])
                    effects.append(100.0 * (ref_auc - auc) / ref_auc)
                    aucs.append(auc)
                    successes.append(
                        int(indexed[key]["finite_result"]) == 1
                        and int(indexed[key]["all_updates_succeeded"]) == 1
                        and int(indexed[key]["constraint_violation"]) == 0
                    )
                arr = np.asarray(effects)
                if not len(arr):
                    raise RuntimeError(f"missing sensitivity value {group}={value}")
                if value == ref_value:
                    ci = (0.0, 0.0)
                else:
                    rng = np.random.RandomState(seed_for(f"sensitivity-{scenario}-{group}-{value}"))
                    draws = np.median(
                        arr[rng.randint(0, len(arr), size=(BOOTSTRAP_REPLICATES, len(arr)))],
                        axis=1,
                    )
                    ci = percentile_interval(draws)
                output.append(
                    {
                        "scenario_id": scenario,
                        "parameter": group,
                        "value": value,
                        "reference_value": ref_value,
                        "paired_n": len(arr),
                        "median_effect_vs_reference_percent": float(np.median(arr)),
                        "ci95_low": ci[0],
                        "ci95_high": ci[1],
                        "median_task_auc": float(np.median(aucs)),
                        "success_rate": float(np.mean(successes)),
                    }
                )
    return output


def analyze_plants(rows: Sequence[Mapping[str, str]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    comparators = ("no_residual_alignment", "raw_top2", "uniform_full_trajectory")
    scenarios = ("n0_d0_m0", "n0_d1_m0", "n1_d1_m1")
    plant_effect_rows = []
    summary = []
    stats = []
    for group in ("held_out_lhs", "edge_challenge"):
        for scenario in scenarios:
            for comparator in comparators:
                records = paired_effects(
                    rows,
                    "v11_full",
                    comparator,
                    scenario,
                    regime="demand_conflict",
                    plant_group=group,
                )
                by_plant: Dict[str, List[float]] = {}
                for record in records:
                    by_plant.setdefault(str(record["plant_id"]), []).append(
                        float(record["effect_percent"])
                    )
                plant_values = []
                for plant_id, values in sorted(by_plant.items()):
                    value = float(np.median(values))
                    plant_values.append(value)
                    plant_effect_rows.append(
                        {
                            "plant_group": group,
                            "scenario_id": scenario,
                            "proposed": "v11_full",
                            "comparator": comparator,
                            "plant_id": plant_id,
                            "tasks_within_plant": len(values),
                            "plant_median_effect_percent": value,
                        }
                    )
                rng = np.random.RandomState(seed_for(f"plant-{group}-{scenario}-{comparator}"))
                array = np.asarray(plant_values)
                draws = np.median(
                    array[rng.randint(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))],
                    axis=1,
                )
                ci = percentile_interval(draws)
                summary.append(
                    {
                        "plant_group": group,
                        "scenario_id": scenario,
                        "proposed": "v11_full",
                        "comparator": comparator,
                        "plant_n": len(array),
                        "tasks_per_plant": 5,
                        "median_plant_effect_percent": float(np.median(array)),
                        "plant_ci95_low": ci[0],
                        "plant_ci95_high": ci[1],
                        "plant_win_rate": float(np.mean(array > 0.0)),
                        "worst_plant_effect_percent": float(np.min(array)),
                        "best_plant_effect_percent": float(np.max(array)),
                    }
                )
                if group == "held_out_lhs":
                    stat = bootstrap_statistics(
                        records, f"heldout-{scenario}-{comparator}"
                    )
                    stat.update(
                        {
                            "dataset": "held_out_lhs_plant_family",
                            "scenario_id": scenario,
                            "scope": "demand_conflict",
                            "proposed": "v11_full",
                            "comparator": comparator,
                        }
                    )
                    stats.append(stat)
    return plant_effect_rows, summary, stats


def replay_case(ablation_rows: Sequence[Mapping[str, str]]) -> Dict[str, object]:
    records = paired_effects(
        ablation_rows,
        "v11_full",
        "no_residual_alignment",
        "n0_d1_m0",
        regime="demand_conflict",
    )
    median = float(np.median([float(item["effect_percent"]) for item in records]))
    selected = min(records, key=lambda item: abs(float(item["effect_percent"]) - median))
    manifests = load_manifests()
    manifest = next(item for item in manifests if item["manifest_id"] == selected["manifest_id"])
    plant_seed = int(str(selected["plant_id"])[1:])
    jobs = build_ablation_jobs(manifests)
    job = next(
        item
        for item in jobs
        if item["method"] == "v11_full"
        and item["manifest"]["manifest_id"] == selected["manifest_id"]
        and item["scenario"]["scenario_id"] == "n0_d1_m0"
        and int(item["plant_seed"]) == plant_seed
    )
    reference = make_trajectory_family(str(manifest["trajectory_family"]), 161, 6.0)
    basis = cubic_bspline_basis(161, 12)
    specification = specification_from_manifest(reference, manifest)
    sensitivity = build_contour_sensitivity(reference, basis, nominal_config())
    settings = BenchmarkSettings(number_of_windows=2, half_width=5)
    traces = {}
    summaries = {}
    for method in ("v11_full", "no_residual_alignment"):
        summary, trace = run_matched_method(
            method,
            reference,
            basis,
            specification,
            make_virtual_machine_domain(plant_seed),
            settings,
            scenario_from_dict(job["scenario"]),
            int(job["noise_seed"]),
            compensation_gain=0.25,
            nominal_sensitivity=sensitivity,
            return_trace=True,
        )
        summaries[method] = summary
        traces[method] = trace

    output_dir = ROOT / "results" / "04_representative_replay"
    output_dir.mkdir(parents=True, exist_ok=True)
    full = traces["v11_full"]
    no = traces["no_residual_alignment"]
    zone_index = np.full(reference.time.size, -1, dtype=int)
    zone_tolerance = np.full(reference.time.size, np.nan)
    for index, ((start, stop), tolerance) in enumerate(
        zip(specification.windows, specification.tolerances)
    ):
        zone_index[start : stop + 1] = index
        zone_tolerance[start : stop + 1] = tolerance
    initial_feedback = np.asarray(full["feedbacks"])[0]
    pointwise = []
    for index in range(reference.time.size):
        pointwise.append(
            {
                "sample": index,
                "time_s": float(reference.time[index]),
                "zone_index": int(zone_index[index]),
                "zone_tolerance_mm": float(zone_tolerance[index]),
                "reference_x_mm": float(reference.position[index, 0]),
                "reference_y_mm": float(reference.position[index, 1]),
                "initial_x_mm": float(initial_feedback[index, 0]),
                "initial_y_mm": float(initial_feedback[index, 1]),
                "no_alignment_x_mm": float(no["accepted_feedback"][index, 0]),
                "no_alignment_y_mm": float(no["accepted_feedback"][index, 1]),
                "v11_x_mm": float(full["accepted_feedback"][index, 0]),
                "v11_y_mm": float(full["accepted_feedback"][index, 1]),
                "initial_contour_error_mm": float(full["contour_errors"][0, index]),
                "no_alignment_contour_error_mm": float(no["accepted_contour_error"][index]),
                "v11_contour_error_mm": float(full["accepted_contour_error"][index]),
            }
        )
    write_csv(output_dir / "pointwise_trace.csv", pointwise)

    lag_rows = []
    for method, trace in traces.items():
        for trial, (total, nominal_lag, residual, applied, accepted) in enumerate(
            zip(
                trace["total_lag_history"],
                trace["nominal_lag_history"],
                trace["residual_lag_history"],
                trace["applied_lag_history"],
                trace["acceptance_history"],
            )
        ):
            lag_rows.append(
                {
                    "method": method,
                    "trial": trial,
                    "total_x": total[0], "total_y": total[1],
                    "nominal_x": nominal_lag[0], "nominal_y": nominal_lag[1],
                    "residual_x": residual[0], "residual_y": residual[1],
                    "applied_x": applied[0], "applied_y": applied[1],
                    "accepted": int(accepted),
                }
            )
    write_csv(output_dir / "lag_evolution.csv", lag_rows)

    anchor_rows = []
    for method, trace in traces.items():
        for update, selection in enumerate(trace["selection_history"]):
            for zone in selection:
                anchor_rows.append(
                    {
                        "method": method,
                        "update": update,
                        "zone_index": zone,
                        "zone_name": specification.names[int(zone)],
                        "zone_role": specification.roles[int(zone)],
                    }
                )
    write_csv(output_dir / "anchor_history.csv", anchor_rows)
    np.savez_compressed(
        output_dir / "full_trace_arrays.npz",
        time=reference.time,
        reference=reference.position,
        initial_feedback=initial_feedback,
        v11_commands=full["commands"],
        v11_feedbacks=full["feedbacks"],
        v11_contour_errors=full["contour_errors"],
        no_alignment_commands=no["commands"],
        no_alignment_feedbacks=no["feedbacks"],
        no_alignment_contour_errors=no["contour_errors"],
    )
    payload = {
        "selection_rule": "closest V11-vs-no-alignment effect to the median among delay+4 demand-conflict pairs",
        "median_effect_percent": median,
        "selected_pair_effect_percent": float(selected["effect_percent"]),
        "manifest_id": selected["manifest_id"],
        "plant_id": selected["plant_id"],
        "plant_seed": plant_seed,
        "scenario_id": "n0_d1_m0",
        "noise_seed": int(job["noise_seed"]),
        "v11_summary": summaries["v11_full"],
        "no_alignment_summary": summaries["no_residual_alignment"],
        "independent_versioned_replay": True,
    }
    (output_dir / "replay_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def save_figure(fig, stem: str) -> None:
    directory = ROOT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(
        directory / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(directory / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(stats: Sequence[Mapping[str, object]]) -> None:
    labels = {
        "no_residual_alignment": "No residual alignment",
        "task_top2": "Task-top2 only",
        "raw_top2": "Raw-top2 only",
        "uniform_full_trajectory": "Uniform full trajectory",
    }
    scenarios = [("n0_d0_m0", "Baseline"), ("n0_d1_m0", "Added delay +4"), ("n1_d1_m1", "Triple stress")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharex=True, sharey=True)
    order = list(labels)
    for panel, (ax, (scenario, title)) in enumerate(zip(axes, scenarios)):
        scoped = [row for row in stats if row["scenario_id"] == scenario]
        indexed = {row["comparator"]: row for row in scoped}
        y = np.arange(len(order))[::-1]
        for yi, comparator in zip(y, order):
            row = indexed[comparator]
            estimate = float(row["median_effect_percent"])
            low = float(row["hierarchical_ci_low"])
            high = float(row["hierarchical_ci_high"])
            color = COLORS["v11"] if comparator == "no_residual_alignment" else COLORS["task"] if comparator == "task_top2" else COLORS["raw"] if comparator == "raw_top2" else COLORS["uniform"]
            ax.plot([low, high], [yi, yi], color=color, lw=1.8)
            ax.plot(estimate, yi, "o", color=color, ms=4.5)
        ax.axvline(0, color="#777777", ls="--", lw=0.8)
        ax.set_title(title, fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[item] for item in order])
        ax.text(-0.12, 1.04, chr(ord("a") + panel), transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.set_xlabel("V11 Full improvement (%)")
    fig.suptitle("Matched component ablation · demand-conflict tasks · hierarchical 95% CI", fontsize=8.5, y=1.01)
    fig.subplots_adjust(left=0.24, right=0.99, bottom=0.18, top=0.84, wspace=0.18)
    save_figure(fig, "fig1_matched_ablation")


def plot_statistics(stats: Sequence[Mapping[str, object]]) -> None:
    scoped = [
        row for row in stats
        if row["dataset"] == "matched_ablation" and row["comparator"] == "no_residual_alignment"
    ]
    scenarios = [("n0_d0_m0", "Baseline"), ("n0_d1_m0", "Delay +4"), ("n1_d1_m1", "Triple")]
    methods = [
        ("paired", "paired_ci_low", "paired_ci_high", COLORS["light"]),
        ("domain", "domain_ci_low", "domain_ci_high", COLORS["task"]),
        ("hierarchical", "hierarchical_ci_low", "hierarchical_ci_high", COLORS["v11"]),
    ]
    fig, ax = plt.subplots(figsize=(5.7, 3.2))
    offsets = (-0.18, 0.0, 0.18)
    for method_index, ((method, low_key, high_key, color), offset) in enumerate(zip(methods, offsets)):
        for scenario_index, (scenario, _) in enumerate(scenarios):
            row = next(item for item in scoped if item["scenario_id"] == scenario)
            estimate = float(row["median_effect_percent"])
            low, high = float(row[low_key]), float(row[high_key])
            y = scenario_index + offset
            ax.plot([low, high], [y, y], color=color, lw=2.0)
            ax.plot(estimate, y, "o", color=color, ms=4.5, label=method if scenario_index == 0 else None)
    ax.axvline(0, color="#777777", ls="--", lw=0.8)
    ax.set_yticks(range(3))
    ax.set_yticklabels([item[1] for item in scenarios])
    ax.invert_yaxis()
    ax.set_xlabel("V11 Full vs no alignment · median AUC improvement (%)")
    ax.set_title("Paired and virtual-plant-aware bootstrap intervals", fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=6.5)
    ax.text(-0.08, 1.03, "a", transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.28, top=0.86)
    save_figure(fig, "fig2_domain_aware_statistics")


def plot_sensitivity(rows: Sequence[Mapping[str, object]]) -> None:
    groups = [
        ("residual_delay_shrinkage", "Residual-delay shrinkage γ"),
        ("smoothing_window", "Velocity smoothing window"),
        ("nominal_control_points", "Nominal B-spline points / axis"),
        ("learning_rate", "Learning rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for panel, (ax, (group, title)) in enumerate(zip(axes.flat, groups)):
        scoped = [row for row in rows if row["parameter"] == group]
        values = sorted({float(row["value"]) for row in scoped})
        for scenario, label, color in (
            ("n0_d1_m0", "Delay +4", COLORS["v11"]),
            ("n1_d1_m1", "Triple", COLORS["raw"]),
        ):
            selected = {float(row["value"]): row for row in scoped if row["scenario_id"] == scenario}
            y = np.asarray([float(selected[value]["median_effect_vs_reference_percent"]) for value in values])
            low = np.asarray([float(selected[value]["ci95_low"]) for value in values])
            high = np.asarray([float(selected[value]["ci95_high"]) for value in values])
            ax.plot(values, y, "o-", color=color, lw=1.2, ms=3.8, label=label)
            ax.fill_between(values, low, high, color=color, alpha=0.13, linewidth=0)
        ax.axhline(0, color="#777777", ls="--", lw=0.8)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Improvement vs group reference (%)")
        ax.text(-0.12, 1.04, chr(ord("a") + panel), transform=ax.transAxes, fontweight="bold", fontsize=9)
        if panel == 0:
            ax.legend(fontsize=6.5)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.95, wspace=0.30, hspace=0.38)
    save_figure(fig, "fig3_parameter_sensitivity")


def plot_replay() -> None:
    directory = ROOT / "results" / "04_representative_replay"
    rows = read_csv(directory / "pointwise_trace.csv")
    lag = read_csv(directory / "lag_evolution.csv")
    anchors = read_csv(directory / "anchor_history.csv")
    replay_payload = json.loads((directory / "replay_summary.json").read_text(encoding="utf-8"))
    x = np.asarray([float(row["reference_x_mm"]) for row in rows])
    y = np.asarray([float(row["reference_y_mm"]) for row in rows])
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], hspace=0.38, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, y, color=COLORS["ink"], lw=1.5, label="Reference")
    ax.plot([float(row["initial_x_mm"]) for row in rows], [float(row["initial_y_mm"]) for row in rows], color="#BBBBBB", lw=1.0, label="Initial")
    ax.plot([float(row["no_alignment_x_mm"]) for row in rows], [float(row["no_alignment_y_mm"]) for row in rows], color=COLORS["raw"], lw=1.2, label="No alignment")
    ax.plot([float(row["v11_x_mm"]) for row in rows], [float(row["v11_y_mm"]) for row in rows], color=COLORS["v11"], lw=1.3, label="V11 Full")
    zone_ids = np.asarray([int(row["zone_index"]) for row in rows])
    zone_colors = ["#5DA5DA", "#F5A45D", "#60BD68", "#B276B2", "#DECF3F", "#F17CB0"]
    for zone in range(6):
        mask = zone_ids == zone
        ax.scatter(x[mask], y[mask], s=7, alpha=0.45, color=zone_colors[zone])
        indices = np.where(mask)[0]
        if indices.size:
            center = indices[len(indices) // 2]
            ax.text(x[center], y[center] + 1.2, f"Z{zone + 1}", color=zone_colors[zone], fontsize=5.5, ha="center")
    all_x = np.concatenate(
        [
            x,
            np.asarray([float(row["initial_x_mm"]) for row in rows]),
            np.asarray([float(row["no_alignment_x_mm"]) for row in rows]),
            np.asarray([float(row["v11_x_mm"]) for row in rows]),
        ]
    )
    all_y = np.concatenate(
        [
            y,
            np.asarray([float(row["initial_y_mm"]) for row in rows]),
            np.asarray([float(row["no_alignment_y_mm"]) for row in rows]),
            np.asarray([float(row["v11_y_mm"]) for row in rows]),
        ]
    )
    ax.set_xlim(float(np.min(all_x) - 2.0), float(np.max(all_x) + 2.0))
    ax.set_ylim(float(np.min(all_y) - 2.0), float(np.max(all_y) + 2.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Automatically selected median-effect replay", fontsize=8)
    ax.legend(fontsize=6.2, loc="best")
    ax.text(-0.12, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=9)

    ax_meta = fig.add_subplot(gs[1, 0])
    ax_meta.set_axis_off()
    ax_meta.set_xlim(0, 1)
    ax_meta.set_ylim(0, 1)
    card = patches.FancyBboxPatch(
        (0.03, 0.12),
        0.92,
        0.74,
        boxstyle="round,pad=0.02",
        facecolor="#F4F6F9",
        edgecolor="#D2D7E0",
        linewidth=0.8,
    )
    ax_meta.add_patch(card)
    ax_meta.text(0.08, 0.76, "Replay selection audit", fontsize=7.5, fontweight="bold")
    ax_meta.text(
        0.08,
        0.62,
        f"Case: {replay_payload['manifest_id']} · {replay_payload['plant_id']}",
        fontsize=6.3,
    )
    ax_meta.text(
        0.08,
        0.49,
        f"Pair effect: {float(replay_payload['selected_pair_effect_percent']):.3f}%",
        fontsize=6.3,
        color=COLORS["v11"],
    )
    ax_meta.text(
        0.08,
        0.36,
        f"Candidate median: {float(replay_payload['median_effect_percent']):.3f}%",
        fontsize=6.3,
    )
    ax_meta.text(
        0.08,
        0.21,
        "Rule fixed before trace inspection: closest pair to median effect",
        fontsize=5.8,
        color="#666666",
    )

    ax_err = fig.add_subplot(gs[0, 1])
    sample = np.arange(len(rows))
    for zone in range(6):
        indices = np.where(zone_ids == zone)[0]
        if indices.size:
            ax_err.axvspan(indices.min(), indices.max(), color="#DDE4F1", alpha=0.35)
    ax_err.plot(sample, np.abs([float(row["initial_contour_error_mm"]) for row in rows]), color="#BBBBBB", lw=0.9, label="Initial")
    ax_err.plot(sample, np.abs([float(row["no_alignment_contour_error_mm"]) for row in rows]), color=COLORS["raw"], lw=1.0, label="No alignment")
    ax_err.plot(sample, np.abs([float(row["v11_contour_error_mm"]) for row in rows]), color=COLORS["v11"], lw=1.1, label="V11 Full")
    ax_err.set_ylabel("|Contour error| (mm)")
    ax_err.set_xlabel("Sample index")
    ax_err.set_title("Pointwise contour error", fontsize=8)
    ax_err.legend(fontsize=6.2, ncol=3)
    ax_err.text(-0.10, 1.03, "b", transform=ax_err.transAxes, fontweight="bold", fontsize=9)

    sub = gs[1, 1].subgridspec(1, 2, wspace=0.62)
    ax_lag = fig.add_subplot(sub[0, 0])
    full_lag = [row for row in lag if row["method"] == "v11_full"]
    trials = [int(row["trial"]) for row in full_lag]
    ax_lag.plot(trials, [float(row["applied_x"]) for row in full_lag], "o-", color=COLORS["v11"], label="x")
    ax_lag.plot(trials, [float(row["applied_y"]) for row in full_lag], "s--", color=COLORS["task"], label="y")
    ax_lag.set_xlabel("Trial")
    ax_lag.set_ylabel("Applied shift (samples)")
    ax_lag.set_title("Residual-lag application", fontsize=8)
    ax_lag.legend(fontsize=6.2, loc="lower right")
    ax_lag.text(-0.28, 1.12, "c", transform=ax_lag.transAxes, fontweight="bold", fontsize=9)

    ax_anchor = fig.add_subplot(sub[0, 1])
    methods = ["v11_full", "no_residual_alignment"]
    matrix = np.zeros((8, 6))
    labels = []
    for method_index, method in enumerate(methods):
        for update in range(4):
            labels.append(("V11" if method_index == 0 else "No align") + f" U{update + 1}")
            for row in anchors:
                if row["method"] == method and int(row["update"]) == update:
                    matrix[method_index * 4 + update, int(row["zone_index"])] = 1
    ax_anchor.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax_anchor.set_xticks(range(6))
    ax_anchor.set_xticklabels([f"Z{i+1}" for i in range(6)])
    ax_anchor.set_yticks(range(8))
    ax_anchor.set_yticklabels(labels, fontsize=5.0)
    ax_anchor.set_title("Active-zone history", fontsize=8)
    ax_anchor.text(-0.34, 1.12, "d", transform=ax_anchor.transAxes, fontweight="bold", fontsize=9)
    save_figure(fig, "fig4_representative_replay")


def plot_plants(rows: Sequence[Mapping[str, object]]) -> None:
    scenarios = [("n0_d0_m0", "Baseline"), ("n0_d1_m0", "Delay +4"), ("n1_d1_m1", "Triple stress")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 6.2), sharex=True, sharey=True)
    for panel, (ax, (scenario, title)) in enumerate(zip(axes, scenarios)):
        scoped = [row for row in rows if row["scenario_id"] == scenario and row["comparator"] == "no_residual_alignment"]
        held = sorted([row for row in scoped if row["plant_group"] == "held_out_lhs"], key=lambda row: row["plant_id"])
        challenge = sorted([row for row in scoped if row["plant_group"] == "edge_challenge"], key=lambda row: row["plant_id"])
        combined = held + challenge
        y = np.arange(len(combined))[::-1]
        for yi, row in zip(y, combined):
            challenge_flag = row["plant_group"] == "edge_challenge"
            color = COLORS["negative"] if float(row["plant_median_effect_percent"]) < 0 else COLORS["raw"] if challenge_flag else COLORS["v11"]
            marker = "^" if challenge_flag else "o"
            ax.plot(float(row["plant_median_effect_percent"]), yi, marker=marker, color=color, ms=3.8)
        ax.axvline(0, color="#777777", ls="--", lw=0.8)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Plant median improvement (%)")
        ax.set_yticks(y)
        ax.set_yticklabels([str(row["plant_id"]) for row in combined], fontsize=5.6)
        ax.text(-0.12, 1.02, chr(ord("a") + panel), transform=ax.transAxes, fontweight="bold", fontsize=9)
    legend = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["v11"], markeredgecolor=COLORS["v11"], label="Held-out LHS"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor=COLORS["raw"], markeredgecolor=COLORS["raw"], label="Challenge"),
    ]
    if any(float(row["plant_median_effect_percent"]) < 0 for row in rows if row["comparator"] == "no_residual_alignment"):
        legend.append(
            plt.Line2D([0], [0], marker="^", color="none", markerfacecolor=COLORS["negative"], markeredgecolor=COLORS["negative"], label="Negative effect")
        )
    fig.legend(handles=legend, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(legend))
    fig.suptitle("V11 Full vs no residual alignment across independently sampled virtual plants", fontsize=8.5, y=0.99)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.10, top=0.88, wspace=0.18)
    save_figure(fig, "fig5_virtual_plant_family")


def build_report(
    ablation_summary: Sequence[Mapping[str, object]],
    statistics: Sequence[Mapping[str, object]],
    sensitivity: Sequence[Mapping[str, object]],
    replay: Mapping[str, object],
    plant_summary: Sequence[Mapping[str, object]],
) -> None:
    def stat(dataset: str, scenario: str, comparator: str):
        return next(row for row in statistics if row["dataset"] == dataset and row["scenario_id"] == scenario and row["comparator"] == comparator)

    ab_delay = stat("matched_ablation", "n0_d1_m0", "no_residual_alignment")
    ab_task = stat("matched_ablation", "n0_d1_m0", "task_top2")
    ab_raw = stat("matched_ablation", "n0_d1_m0", "raw_top2")
    gamma_rows = [row for row in sensitivity if row["parameter"] == "residual_delay_shrinkage" and row["scenario_id"] == "n0_d1_m0"]
    gamma_supported = [
        str(row["value"])
        for row in gamma_rows
        if float(row["ci95_low"]) > 0.0
    ]
    gamma_one = next(row for row in gamma_rows if math.isclose(float(row["value"]), 1.0))
    plant_rows = [row for row in plant_summary if row["plant_group"] == "held_out_lhs" and row["comparator"] == "no_residual_alignment"]
    challenge_rows = [row for row in plant_summary if row["plant_group"] == "edge_challenge" and row["comparator"] == "no_residual_alignment"]

    lines = [
        "# 五项 V11 新增实验结果与分析",
        "",
        "## 结论摘要",
        "",
        f"1. **匹配消融**：在 +4 时延、demand-conflict 下，V11 Full 相对无残余时延对齐的中位任务 AUC 改善为 **{float(ab_delay['median_effect_percent']):.3f}%**，hierarchical 95% CI 为 **[{float(ab_delay['hierarchical_ci_low']):.3f}%, {float(ab_delay['hierarchical_ci_high']):.3f}%]**；相对 Task-top2 和 Raw-top2 分别为 **{float(ab_task['median_effect_percent']):.3f}%** 与 **{float(ab_raw['median_effect_percent']):.3f}%**。这组 matched 结果将残余时延对齐识别为主要独立增益来源；若锚点比较区间跨零，则不能声称双锚点组合在匹配框架下已有独立协同证据。",
        f"2. **分层统计**：同一 +4 时延主比较的逐对、按域和分层区间分别为 [{float(ab_delay['paired_ci_low']):.3f}, {float(ab_delay['paired_ci_high']):.3f}]%、[{float(ab_delay['domain_ci_low']):.3f}, {float(ab_delay['domain_ci_high']):.3f}]% 和 [{float(ab_delay['hierarchical_ci_low']):.3f}, {float(ab_delay['hierarchical_ci_high']):.3f}]%。这直接展示了把 plant 当作独立层级后不确定性如何变化。",
        f"3. **参数敏感性**：γ 扫描覆盖 {', '.join(str(row['value']) for row in gamma_rows)}；在 +4 时延开发集上相对 γ=0 区间下界为正的 γ 为 **{', '.join(gamma_supported) if gamma_supported else '无'}**。γ=1 的中位效应为 **{float(gamma_one['median_effect_vs_reference_percent']):.3f}%**（95% CI [{float(gamma_one['ci95_low']):.3f}%, {float(gamma_one['ci95_high']):.3f}%]），用于判断完全相信估计是否出现过补偿。该结果只说明稳定区间/边界，不用于重新选择正式参数。",
        f"4. **代表性回放**：自动选择 `{replay['manifest_id']}`、`{replay['plant_id']}`；其效应为 {float(replay['selected_pair_effect_percent']):.3f}%，而候选配对中位数为 {float(replay['median_effect_percent']):.3f}%。保存的是独立重放的完整点迹，不是由 summary CSV 拼接。",
        "5. **虚拟对象泛化**：24 个 held-out LHS 对象的 plant-level 结果如下表；6 个 challenge 对象仅用于失效边界，不参与参数选择。",
        "",
        "| 场景 | Plant-level median | 95% CI | Plant win rate | Worst plant |",
        "|---|---:|---:|---:|---:|",
    ]
    names = {"n0_d0_m0": "基线", "n0_d1_m0": "+4 时延", "n1_d1_m1": "三重压力"}
    for row in plant_rows:
        lines.append(
            f"| {names[str(row['scenario_id'])]} | {float(row['median_plant_effect_percent']):.3f}% | [{float(row['plant_ci95_low']):.3f}, {float(row['plant_ci95_high']):.3f}]% | {100*float(row['plant_win_rate']):.1f}% | {float(row['worst_plant_effect_percent']):.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Matched ablation 详细结果（demand-conflict）",
            "",
            "正值表示 V11 Full 的任务 AUC 更低。区间为 hierarchical-bootstrap 95% CI。",
            "",
            "| 场景 | 消融比较 | 中位效应 | Hierarchical 95% CI | Pair win rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    comparator_names = {
        "no_residual_alignment": "无残余时延对齐",
        "task_top2": "Task-top2",
        "raw_top2": "Raw-top2",
        "uniform_full_trajectory": "Uniform full trajectory",
    }
    for scenario in ("n0_d0_m0", "n0_d1_m0", "n1_d1_m1"):
        for comparator in comparator_names:
            row = stat("matched_ablation", scenario, comparator)
            lines.append(
                f"| {names[scenario]} | {comparator_names[comparator]} | {float(row['median_effect_percent']):.3f}% | [{float(row['hierarchical_ci_low']):.3f}, {float(row['hierarchical_ci_high']):.3f}]% | {100*float(row['pair_win_rate']):.1f}% |"
            )
    lines.extend(
        [
            "",
            "## 1. Matched ablation 解释",
            "",
            "Task-top2 与 Raw-top2 均保持两个活动区、balanced weights、同一 QP、同一信赖域和回滚，因此本实验比早期 V8 配置比较更接近组件级因果消融。无对齐比较在三个场景的 hierarchical CI 下界均高于零，支持残余时延对齐是稳定的独立增益来源。相反，V11 Full 相对 Task-top2 的中位效应略为负，区间触及或跨越零；相对 Raw-top2 的中位效应为零。因而，当前 matched 实验**没有确认双锚点组合本身优于单一信息源的 top-2 排序**。这不会否定 V11 整体有效，但论文应把双锚点写成可解释的配置设计，而不是已有独立协同因果证明。该结论仍只覆盖当前语义区构造、四次更新预算和数值对象族，不能外推为任意调度器定理。",
            "",
            "## 2. Domain-aware statistics 解释",
            "",
            "逐对 bootstrap 把 task-domain pair 当作交换单位；domain bootstrap 保留每个对象内全部任务；hierarchical bootstrap 先抽对象、再抽对象内任务。主文应优先呈现后两种区间，并把逐对结果作为与旧结果兼容的补充。Leave-one-domain-out 范围用于检查是否存在单域主导。",
            "",
            "## 3. 参数敏感性解释",
            "",
            "该实验是 development-only 的 one-factor-at-a-time 扫描。γ=0 是严格的无对齐对照；γ=1 表示完全采用中位残余估计。平滑窗口、控制点和学习率均围绕冻结默认值做有限扰动。任何最优点都不能反向替换 V11 的正式 γ=0.25。",
            "",
            "## 4. 轨迹回放解释",
            "",
            "回放保存二维参考、初始反馈、两种最终已接受反馈、每次试次的命令/反馈/轮廓误差、残余时延和活动区历史。案例选择规则在读取点迹前固定为最接近中位效应，因此避免人为挑选最漂亮案例。",
            "",
            "## 5. Plant family 解释",
            "",
            "LHS 在 14 维参数空间中覆盖轴动力学、时延、摩擦、饱和、耦合和重复扰动。Plant 是主统计单位。Challenge 对象用于识别最大时延、低带宽、高不对称、强耦合、强摩擦和紧饱和附近的负效应，不应与 held-out 主 CI 混合。",
            "",
            "## 证据边界",
            "",
            "- 本轮没有 LinuxCNC、G 代码控制器链、真实伺服或切削实验。",
            "- LHS 范围是数值假设空间，不是由真实机床群体标定的概率分布。",
            "- bootstrap 区间描述当前对象/任务样本的不确定性，不构成物理安全或全局最优证明。",
            "- 旧 V8–V13 冻结结果未修改；本轮结果应作为独立新增证据层报告。",
            "",
            "## Challenge 摘要",
            "",
        ]
    )
    for row in challenge_rows:
        lines.append(
            f"- {names[str(row['scenario_id'])]}：6 个 challenge 对象中 {100*float(row['plant_win_rate']):.1f}% 的 plant-level 中位效应为正，范围 [{float(row['worst_plant_effect_percent']):.3f}%, {float(row['best_plant_effect_percent']):.3f}%]。"
        )
    (ROOT / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ablation_rows = read_csv(ROOT / "results" / "01_matched_ablation" / "raw_results.csv")
    sensitivity_rows = read_csv(ROOT / "results" / "03_parameter_sensitivity" / "raw_results.csv")
    plant_rows = read_csv(ROOT / "results" / "05_virtual_plant_family" / "raw_results.csv")

    ablation_summary, ablation_stats = analyze_ablation(ablation_rows)
    write_csv(ROOT / "results" / "01_matched_ablation" / "ablation_summary.csv", ablation_summary)
    sensitivity_summary = analyze_sensitivity(sensitivity_rows)
    write_csv(ROOT / "results" / "03_parameter_sensitivity" / "sensitivity_summary.csv", sensitivity_summary)
    plant_effects, plant_summary, plant_stats = analyze_plants(plant_rows)
    write_csv(ROOT / "results" / "05_virtual_plant_family" / "plant_level_effects.csv", plant_effects)
    write_csv(ROOT / "results" / "05_virtual_plant_family" / "plant_family_summary.csv", plant_summary)
    all_stats = ablation_stats + plant_stats
    write_csv(ROOT / "results" / "02_hierarchical_statistics" / "bootstrap_comparison.csv", all_stats)
    replay = replay_case(ablation_rows)

    plot_ablation(ablation_stats)
    plot_statistics(all_stats)
    plot_sensitivity(sensitivity_summary)
    plot_replay()
    plot_plants(plant_effects)
    build_report(ablation_summary, all_stats, sensitivity_summary, replay, plant_summary)
    print(json.dumps({"status": "complete", "figures": 5, "bootstrap_replicates": BOOTSTRAP_REPLICATES}, indent=2))


if __name__ == "__main__":
    main()
