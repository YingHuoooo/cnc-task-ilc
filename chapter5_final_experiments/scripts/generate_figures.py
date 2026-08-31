"""Generate the publication figures for Chapter 5 from frozen source data."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "chapter5-final-mpl"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parent
PAPER = WORKSPACE / "cnc_v11_paper_package"
ADDITIONAL = WORKSPACE / "v11_additional_experiments"
FIGURES = ROOT / "figures"

COLORS = {
    "proposed": "#1F5A94",
    "no_alignment": "#6F78A8",
    "task": "#2E8B78",
    "raw": "#C1862E",
    "uniform": "#777777",
    "fixed": "#7B5BA6",
    "delay": "#B74E4A",
    "light_blue": "#B9CEE2",
    "light_gray": "#D9D9D9",
    "ink": "#242424",
    "grid": "#E6E6E6",
}

SCENARIOS = [
    ("n0_d0_m0", "Baseline"),
    ("delay_2", "Added delay +2"),
    ("n0_d1_m0", "Added delay +4"),
    ("n1_d1_m1", "Triple stress"),
]

THREE_SCENARIOS = [
    ("n0_d0_m0", "Baseline"),
    ("n0_d1_m0", "Added delay +4"),
    ("n1_d1_m1", "Triple stress"),
]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "svg.hashsalt": "chapter5-final-experiments",
        "pdf.fonttype": 42,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
    }
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def f(row: Mapping[str, object], key: str) -> float:
    return float(row[key])


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", fontsize=12, va="bottom")


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def errorbar_h(ax: plt.Axes, estimate: float, low: float, high: float, y: float, color: str, marker: str = "o") -> None:
    ax.plot([low, high], [y, y], color=color, lw=1.8, solid_capstyle="round")
    ax.plot(estimate, y, marker=marker, color=color, ms=4.5, mec="white", mew=0.4)


def conflict_full_trajectory_comparisons() -> Dict[str, Dict[str, object]]:
    """Derive conflict-only full-trajectory comparisons from frozen V11 runs."""

    rows = read_csv(PAPER / "source_snapshot" / "results" / "v11_raw.csv")
    output: Dict[str, Dict[str, object]] = {}
    for scenario, _ in SCENARIOS:
        scoped = [
            row
            for row in rows
            if row["scenario_id"] == scenario and row["regime"] == "demand_conflict"
        ]
        indexed = {
            (row["manifest_id"], row["domain_seed"], row["method"]): row
            for row in scoped
        }
        keys = sorted({(key[0], key[1]) for key in indexed})
        proposed = np.asarray(
            [
                float(indexed[key + ("delay_aware_dual_anchor",)]["task_auc_normalized"])
                for key in keys
            ]
        )
        comparator = np.asarray(
            [
                float(indexed[key + ("full_trajectory",)]["task_auc_normalized"])
                for key in keys
            ]
        )
        effects = 100.0 * (comparator - proposed) / comparator
        label = f"chapter5-conflict-full-{scenario}"
        seed = 20260811 + int(zlib.crc32(label.encode("utf-8")) % 100000)
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, effects.size, size=(20000, effects.size))
        draws = np.median(effects[indices], axis=1)
        low, high = np.percentile(draws, (2.5, 97.5))
        output[scenario] = {
            "paired_n": int(effects.size),
            "median_improvement_percent": float(np.median(effects)),
            "ci95_low_percent": float(low),
            "ci95_high_percent": float(high),
            "win_rate": float(np.mean(effects > 0.0)),
            "bootstrap_seed": seed,
        }
    return output


def figure_reference_strategies() -> None:
    rows = read_csv(PAPER / "source_data" / "fig4_v11_delay_compensation.csv")
    conflict_full = conflict_full_trajectory_comparisons()
    conflict_comparators = [
        ("conflict_full_trajectory", "Uniform full trajectory", COLORS["uniform"]),
        ("V11 vs error-peak", "Raw-error peak", COLORS["raw"]),
        ("V11 vs fixed +4", "Fixed sensitivity shift", COLORS["fixed"]),
    ]
    indexed = {(r["scope"], r["comparison"]): r for r in rows}
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.50)
    ax = fig.add_subplot(gs[0, 0])
    y_base = np.arange(len(SCENARIOS))[::-1]
    offsets = (0.22, 0.0, -0.22)
    for offset, (comparison, display_label, color) in zip(offsets, conflict_comparators):
        for (scenario, _), y in zip(SCENARIOS, y_base):
            row = (
                conflict_full[scenario]
                if comparison == "conflict_full_trajectory"
                else indexed[(scenario, comparison)]
            )
            errorbar_h(
                ax,
                f(row, "median_improvement_percent"),
                f(row, "ci95_low_percent"),
                f(row, "ci95_high_percent"),
                y + offset,
                color,
            )
        ax.plot([], [], "o-", color=color, lw=1.8, ms=4.5, label=display_label)
    ax.axvline(0, color="#777777", lw=0.8, ls="--")
    ax.set_yticks(y_base)
    ax.set_yticklabels([name for _, name in SCENARIOS])
    ax.set_xlim(-4.5, 26.5)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    ax.legend(loc="lower right", ncol=1)
    ax.set_title("Common demand-conflict subset (n=20 per comparison)", loc="left", fontweight="bold")
    panel_label(ax, "a", x=-0.10, y=1.02)

    ax = fig.add_subplot(gs[1, 0])
    for (scenario, _), y in zip(SCENARIOS, y_base):
        row = indexed[(scenario, "V11 vs full-trajectory (all tasks)")]
        errorbar_h(
            ax,
            f(row, "median_improvement_percent"),
            f(row, "ci95_low_percent"),
            f(row, "ci95_high_percent"),
            y,
            COLORS["uniform"],
        )
    ax.axvline(0, color="#777777", lw=0.8, ls="--")
    ax.set_yticks(y_base)
    ax.set_yticklabels([name for _, name in SCENARIOS])
    ax.set_xlabel("Median normalized task-AUC improvement (%)")
    ax.set_xlim(-4.5, 26.5)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    ax.set_title("Broader scope: Uniform full trajectory on all 15 tasks (n=60)", loc="left", fontweight="bold")
    panel_label(ax, "b", x=-0.10, y=1.02)

    fig.suptitle(
        "Configuration-level comparison with reference learning strategies",
        x=0.15,
        y=0.995,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.11, top=0.92)
    save_figure(fig, "fig2_reference_strategy_comparison")


def figure_matched_ablation() -> None:
    rows = read_csv(ADDITIONAL / "results" / "02_hierarchical_statistics" / "bootstrap_comparison.csv")
    rows = [r for r in rows if r["dataset"] == "matched_ablation"]
    labels = {
        "no_residual_alignment": "No residual alignment",
        "task_top2": "Task-top2",
        "raw_top2": "Raw-top2",
        "uniform_full_trajectory": "Uniform full trajectory",
    }
    colors = {
        "no_residual_alignment": COLORS["proposed"],
        "task_top2": COLORS["task"],
        "raw_top2": COLORS["raw"],
        "uniform_full_trajectory": COLORS["uniform"],
    }
    order = list(labels)
    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.48, wspace=0.20)
    for panel, (scenario, title) in enumerate(THREE_SCENARIOS):
        ax = fig.add_subplot(gs[0, panel])
        scoped = {r["comparator"]: r for r in rows if r["scenario_id"] == scenario}
        y = np.arange(len(order))[::-1]
        for yi, comparator in zip(y, order):
            row = scoped[comparator]
            errorbar_h(
                ax,
                f(row, "median_effect_percent"),
                f(row, "hierarchical_ci_low"),
                f(row, "hierarchical_ci_high"),
                yi,
                colors[comparator],
            )
        ax.axvline(0, color="#777777", ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Proposed vs comparator (%)")
        ax.set_xlim(-4.2, 10.5)
        ax.set_yticks(y)
        if panel == 0:
            ax.set_yticklabels([labels[item] for item in order])
        else:
            ax.set_yticklabels([])
        ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
        panel_label(ax, chr(ord("a") + panel), x=-0.16 if panel == 0 else -0.10)

    ax = fig.add_subplot(gs[1, :])
    no_align = [r for r in rows if r["comparator"] == "no_residual_alignment"]
    by_scenario = {r["scenario_id"]: r for r in no_align}
    bootstrap_types = [
        ("Paired", "paired_ci_low", "paired_ci_high", COLORS["light_blue"]),
        ("Plant", "domain_ci_low", "domain_ci_high", COLORS["task"]),
        ("Hierarchical", "hierarchical_ci_low", "hierarchical_ci_high", COLORS["proposed"]),
    ]
    offsets = (0.20, 0.0, -0.20)
    y_base = np.arange(len(THREE_SCENARIOS))[::-1]
    for offset, (label, low_key, high_key, color) in zip(offsets, bootstrap_types):
        for (scenario, _), y in zip(THREE_SCENARIOS, y_base):
            row = by_scenario[scenario]
            errorbar_h(ax, f(row, "median_effect_percent"), f(row, low_key), f(row, high_key), y + offset, color)
        ax.plot([], [], "o-", color=color, lw=1.8, ms=4.5, label=label)
    ax.axvline(0, color="#777777", ls="--", lw=0.8)
    ax.set_yticks(y_base)
    ax.set_yticklabels([name for _, name in THREE_SCENARIOS])
    ax.set_xlabel("Proposed vs No residual alignment: median task-AUC improvement (%)")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    ax.legend(loc="upper right", ncol=3)
    ax.set_title("Dependence-aware inference for the isolated residual-alignment effect", loc="left")
    panel_label(ax, "d", x=-0.10)
    fig.suptitle("Matched component ablation", fontsize=11.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.10, top=0.91)
    save_figure(fig, "fig3_matched_ablation_and_inference")


def figure_temporal_diagnosis() -> None:
    factorial_rows = read_csv(PAPER / "source_data" / "fig3_v10_factorial.csv")
    main = [r for r in factorial_rows if r["scope"] == "factorial effect" and r["comparison"] in {"noise", "delay", "mismatch"}]
    label_map = {"noise": "Noise", "delay": "Added delay", "mismatch": "Dynamic mismatch"}
    color_map = {"noise": COLORS["light_blue"], "delay": COLORS["delay"], "mismatch": COLORS["task"]}
    v9 = read_json(PAPER / "source_snapshot" / "results" / "v9_comparison.json")
    v10 = read_json(PAPER / "source_snapshot" / "results" / "v10_comparison.json")
    degradation = {
        r["scenario_id"]: float(r["median_auc_degradation_percent"])
        for r in v9["degradation_from_baseline"]
        if r["method"] == "dual_anchor_dynamic" and r["scenario_id"] in {"delay_2", "delay_4"}
    }
    triple = v10["extreme_degradation"]["dual_anchor_dynamic"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    main = sorted(main, key=lambda r: ["noise", "delay", "mismatch"].index(r["comparison"]))
    y = np.arange(len(main))[::-1]
    for yi, row in zip(y, main):
        errorbar_h(
            ax,
            f(row, "median_improvement_percent"),
            f(row, "ci95_low_percent"),
            f(row, "ci95_high_percent"),
            yi,
            color_map[row["comparison"]],
        )
    ax.axvline(0, color="#777777", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([label_map[r["comparison"]] for r in main])
    ax.set_xlabel("Median effect on task AUC (%)")
    ax.set_title("Factorial main effects", loc="left")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    panel_label(ax, "a")

    ax = axes[1]
    names = ["Added delay\n+2", "Added delay\n+4", "Triple stress"]
    values = [degradation["delay_2"], degradation["delay_4"], float(triple["median_extreme_auc_degradation_percent"])]
    bars = ax.bar(np.arange(3), values, color=[COLORS["light_blue"], COLORS["delay"], COLORS["fixed"]], width=0.62)
    low, high = triple["bootstrap_extreme_degradation_95ci_percent"]
    ax.errorbar(2, values[2], yerr=[[values[2] - float(low)], [float(high) - values[2]]], fmt="none", ecolor=COLORS["ink"], lw=1.0, capsize=2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names)
    ax.set_ylabel("AUC degradation from baseline (%)")
    ax.set_ylim(0, 84)
    ax.set_title("Absolute finite-trial degradation", loc="left")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
    panel_label(ax, "b")
    fig.suptitle("Temporal mismatch was the dominant tested degradation source", fontsize=11.5, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.22, top=0.83, wspace=0.42)
    save_figure(fig, "fig4_temporal_mismatch_diagnosis")


def figure_plant_generalization() -> None:
    effects = read_csv(ADDITIONAL / "results" / "05_virtual_plant_family" / "plant_level_effects.csv")
    summary = read_csv(ADDITIONAL / "results" / "05_virtual_plant_family" / "plant_family_summary.csv")
    effects = [r for r in effects if r["comparator"] == "no_residual_alignment"]
    summary = [r for r in summary if r["comparator"] == "no_residual_alignment"]
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1.4, 1.25], height_ratios=[1.0, 0.85], hspace=0.50, wspace=0.58)
    ax = fig.add_subplot(gs[:, :2])
    markers = ["o", "s", "^"]
    colors = [COLORS["light_blue"], COLORS["proposed"], COLORS["fixed"]]
    x = np.arange(1, 25)
    for (scenario, label), marker, color in zip(THREE_SCENARIOS, markers, colors):
        scoped = sorted(
            [r for r in effects if r["plant_group"] == "held_out_lhs" and r["scenario_id"] == scenario],
            key=lambda r: r["plant_id"],
        )
        y = [f(r, "plant_median_effect_percent") for r in scoped]
        ax.scatter(x, y, marker=marker, s=18, color=color, label=label, zorder=3)
    ax.axhline(0, color="#777777", ls="--", lw=0.8)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.set_xticklabels(["P01", "P04", "P08", "P12", "P16", "P20", "P24"])
    ax.set_xlabel("Held-out LHS virtual plant")
    ax.set_ylabel("Proposed vs No residual alignment\nplant-level median improvement (%)")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
    ax.legend(loc="upper left", ncol=3)
    ax.set_title("All 24 held-out plants retained a positive median effect", loc="left")
    panel_label(ax, "a", x=-0.10)

    ax = fig.add_subplot(gs[0, 2])
    y_base = np.arange(3)[::-1]
    for yi, ((scenario, _), color) in enumerate(zip(THREE_SCENARIOS, colors)):
        row = next(r for r in summary if r["plant_group"] == "held_out_lhs" and r["scenario_id"] == scenario)
        y = y_base[yi]
        errorbar_h(ax, f(row, "median_plant_effect_percent"), f(row, "plant_ci95_low"), f(row, "plant_ci95_high"), y, color)
    ax.axvline(0, color="#777777", ls="--", lw=0.8)
    ax.set_yticks(y_base)
    ax.set_yticklabels([name for _, name in THREE_SCENARIOS])
    ax.set_xlabel("Median effect (%)")
    ax.set_title("LHS summary", loc="left")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    panel_label(ax, "b", x=-0.28)

    ax = fig.add_subplot(gs[1, 2])
    y_base = np.arange(3)[::-1]
    for yi, ((scenario, _), color) in enumerate(zip(THREE_SCENARIOS, colors)):
        row = next(r for r in summary if r["plant_group"] == "edge_challenge" and r["scenario_id"] == scenario)
        y = y_base[yi]
        ax.plot([f(row, "worst_plant_effect_percent"), f(row, "best_plant_effect_percent")], [y, y], color=color, lw=2.0)
        ax.plot(f(row, "median_plant_effect_percent"), y, "^", color=color, ms=4.5)
    ax.axvline(0, color="#777777", ls="--", lw=0.8)
    ax.set_yticks(y_base)
    ax.set_yticklabels([name for _, name in THREE_SCENARIOS])
    ax.set_xlabel("Range across six plants (%)")
    ax.set_title("Challenge-plant boundary", loc="left")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    panel_label(ax, "c", x=-0.28)
    fig.suptitle("Residual-alignment benefit across previously unseen numerical plants", fontsize=11.5, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.13, top=0.87)
    save_figure(fig, "fig5_held_out_plant_generalization")


def figure_gamma_sensitivity() -> None:
    rows = read_csv(ADDITIONAL / "results" / "03_parameter_sensitivity" / "sensitivity_summary.csv")
    rows = [r for r in rows if r["parameter"] == "residual_delay_shrinkage"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharex=True, sharey=True)
    for panel, (ax, (scenario, title)) in enumerate(zip(axes, THREE_SCENARIOS[1:])):
        scoped = sorted([r for r in rows if r["scenario_id"] == scenario], key=lambda r: f(r, "value"))
        x = np.asarray([f(r, "value") for r in scoped])
        y = np.asarray([f(r, "median_effect_vs_reference_percent") for r in scoped])
        low = np.asarray([f(r, "ci95_low") for r in scoped])
        high = np.asarray([f(r, "ci95_high") for r in scoped])
        ax.fill_between(x, low, high, color=COLORS["light_blue"], alpha=0.45, linewidth=0)
        ax.plot(x, y, "o-", color=COLORS["proposed"], lw=1.5, ms=4.0)
        ax.axhline(0, color="#777777", ls="--", lw=0.8)
        ax.axvline(0.25, color=COLORS["delay"], ls=":", lw=1.2)
        selected = np.where(np.isclose(x, 0.25))[0][0]
        ax.plot(x[selected], y[selected], "o", color=COLORS["delay"], ms=5.0, label="Prespecified γ=0.25")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Residual-alignment gain γ")
        ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
        panel_label(ax, chr(ord("a") + panel))
    axes[0].set_ylabel("Improvement relative to γ=0 (%)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Conservative fractional alignment remained stable over a moderate gain range", fontsize=11.5, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.82, wspace=0.24)
    save_figure(fig, "fig6_gamma_sensitivity")


def figure_representative_replay() -> None:
    directory = ADDITIONAL / "results" / "04_representative_replay"
    trace = read_csv(directory / "pointwise_trace.csv")
    lag = read_csv(directory / "lag_evolution.csv")
    anchors = read_csv(directory / "anchor_history.csv")
    replay = read_json(directory / "replay_summary.json")
    effect = float(replay["selected_pair_effect_percent"])
    arrays = np.load(directory / "full_trace_arrays.npz")
    time = arrays["time"]
    reference = arrays["reference"]
    proposed_feedback = arrays["v11_feedbacks"][-1]
    no_feedback = arrays["no_alignment_feedbacks"][-1]
    proposed_error = np.abs(arrays["v11_contour_errors"][-1])
    no_error = np.abs(arrays["no_alignment_contour_errors"][-1])
    proposed_scores = np.asarray(json.loads(replay["v11_summary"]["trial_task_scores"]))
    no_scores = np.asarray(json.loads(replay["no_alignment_summary"]["trial_task_scores"]))
    zone_ids = np.asarray([int(r["zone_index"]) for r in trace])
    zone_colors = ["#6BAED6", "#F2A65A", "#74B77B", "#A98AC3", "#D3BE45", "#E58AB2"]
    sample_step = float(np.median(np.diff(time)))
    zone_spans: List[Tuple[int, float, float, float]] = []
    for zone in range(6):
        idx = np.where(zone_ids == zone)[0]
        if not idx.size:
            continue
        start = max(float(time[0]), float(time[idx[0]]) - 0.5 * sample_step)
        end = min(float(time[-1]), float(time[idx[-1]]) + 0.5 * sample_step)
        tolerance = float(trace[int(idx[0])]["zone_tolerance_mm"])
        zone_spans.append((zone, start, end, tolerance))

    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.10, 1.35, 1.25], height_ratios=[1.0, 1.05], hspace=0.48, wspace=0.50)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(reference[:, 0], reference[:, 1], color=COLORS["ink"], lw=1.5, label="Reference")
    ax.plot(no_feedback[:, 0], no_feedback[:, 1], color=COLORS["raw"], lw=1.1, label="No alignment")
    ax.plot(proposed_feedback[:, 0], proposed_feedback[:, 1], color=COLORS["proposed"], lw=1.3, label="Proposed")
    for zone in range(6):
        idx = np.where(zone_ids == zone)[0]
        if idx.size:
            mid = idx[len(idx) // 2]
            ax.text(reference[mid, 0], reference[mid, 1] + 1.0, f"Z{zone + 1}", color=zone_colors[zone], fontsize=9.5, ha="center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Final measured contour", loc="left")
    contour_handles, contour_labels = ax.get_legend_handles_labels()
    panel_label(ax, "a", x=-0.16)

    ax = fig.add_subplot(gs[0, 1])
    trials = np.arange(1, 6)
    ax.plot(trials, no_scores, "o-", color=COLORS["raw"], lw=1.2, ms=3.8, label="No alignment")
    ax.plot(trials, proposed_scores, "o-", color=COLORS["proposed"], lw=1.4, ms=4.0, label="Proposed")
    ax.set_xticks(trials)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Tolerance-normalized task score")
    ax.set_title("Finite-trial task-quality evolution", loc="left")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
    ax.legend()
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0:2])
    for zone, start, end, tolerance in zone_spans:
        ax.axvspan(start, end, color=zone_colors[zone], alpha=0.12, linewidth=0, zorder=0)
        ax.axvline(start, color=zone_colors[zone], ls=":", lw=0.6, alpha=0.85, zorder=1)
        ax.axvline(end, color=zone_colors[zone], ls=":", lw=0.6, alpha=0.85, zorder=1)
        label_y = 0.975 if zone % 2 == 0 else 0.875
        ax.text(
            0.5 * (start + end),
            label_y,
            f"Z{zone + 1}\n$\\tau_z$={tolerance:.2f}",
            transform=ax.get_xaxis_transform(),
            color=zone_colors[zone],
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="top",
            linespacing=0.9,
        )
    ax.plot(time, no_error, color=COLORS["raw"], lw=0.9, label="No alignment", zorder=2)
    ax.plot(time, proposed_error, color=COLORS["proposed"], lw=1.0, label="Proposed", zorder=2)
    error_max = float(max(np.max(no_error), np.max(proposed_error)))
    ax.set_ylim(0, 1.18 * error_max)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Contour-normal error| (mm)")
    ax.set_title("Final pointwise contour error by semantic zone", loc="left")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[0, 2])
    full_lag = [r for r in lag if r["method"] == "v11_full"]
    trial = [int(r["trial"]) + 1 for r in full_lag]
    ax.plot(trial, [f(r, "applied_x") for r in full_lag], "o-", color=COLORS["proposed"], label="x axis")
    ax.plot(trial, [f(r, "applied_y") for r in full_lag], "s--", color=COLORS["task"], label="y axis")
    ax.set_xticks(trial)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Applied fractional shift (samples)")
    ax.set_title("Nominal-sensitivity alignment", loc="left")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
    ax.legend()
    panel_label(ax, "d")

    ax = fig.add_subplot(gs[1, 2])
    matrix = np.zeros((8, 6))
    row_labels: List[str] = []
    for method_index, method in enumerate(["v11_full", "no_residual_alignment"]):
        for update in range(4):
            row_labels.append(("Prop." if method_index == 0 else "No align.") + f" U{update + 1}")
            for row in anchors:
                if row["method"] == method and int(row["update"]) == update:
                    matrix[method_index * 4 + update, int(row["zone_index"])] = 1
    ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"Z{i + 1}" for i in range(6)])
    for tick, color in zip(ax.get_xticklabels(), zone_colors):
        tick.set_color(color)
        tick.set_fontweight("bold")
    ax.set_yticks(range(8))
    ax.set_yticklabels(row_labels, fontsize=9.5)
    ax.set_title("Active-zone history", loc="left")
    panel_label(ax, "e")
    fig.suptitle(
        f"Representative replay: S-curve, added delay +4 (AUC improvement {effect:.3f}%)",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(
        contour_handles,
        contour_labels,
        loc="upper left",
        bbox_to_anchor=(0.11, 0.955),
        ncol=3,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.10, top=0.86)
    save_figure(fig, "fig7_representative_trial_replay")


def main() -> None:
    required = [
        PAPER / "source_data" / "fig4_v11_delay_compensation.csv",
        PAPER / "source_snapshot" / "results" / "v11_raw.csv",
        PAPER / "source_data" / "fig3_v10_factorial.csv",
        ADDITIONAL / "results" / "02_hierarchical_statistics" / "bootstrap_comparison.csv",
        ADDITIONAL / "results" / "03_parameter_sensitivity" / "sensitivity_summary.csv",
        ADDITIONAL / "results" / "05_virtual_plant_family" / "plant_family_summary.csv",
        ADDITIONAL / "results" / "04_representative_replay" / "full_trace_arrays.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen source data: " + ", ".join(missing))
    figure_reference_strategies()
    figure_matched_ablation()
    figure_temporal_diagnosis()
    figure_plant_generalization()
    figure_gamma_sensitivity()
    figure_representative_replay()
    print(f"Generated 6 figures in {FIGURES}")


if __name__ == "__main__":
    main()
