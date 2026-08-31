"""Create paired statistics for the external-task V5 confirmation."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


PRIMARY = "nominal_lookahead_dynamic"
COMPARATORS = (
    "violation_dynamic",
    "error_peak_dynamic",
    "full_trajectory",
    "static_tolerance",
    "curvature_zones",
    "jerk_zones",
    "random_zones",
)


def _compare(rows, keys, comparator, rng, primary=PRIMARY):
    indexed = {
        (
            str(row["trajectory"]),
            int(row["domain_seed"]),
            str(row["method"]),
        ): row
        for row in rows
    }
    proposed = np.asarray(
        [float(indexed[key + (primary,)]["task_auc_normalized"]) for key in keys]
    )
    baseline = np.asarray(
        [
            float(indexed[key + (comparator,)]["task_auc_normalized"])
            for key in keys
        ]
    )
    improvement = 100.0 * (baseline - proposed) / baseline
    samples = rng.integers(
        0,
        improvement.size,
        size=(20000, improvement.size),
    )
    bootstrap = np.median(improvement[samples], axis=1)
    proposed_violation = np.asarray(
        [float(indexed[key + (primary,)]["final_violation_rate"]) for key in keys]
    )
    baseline_violation = np.asarray(
        [
            float(indexed[key + (comparator,)]["final_violation_rate"])
            for key in keys
        ]
    )
    test = wilcoxon(proposed, baseline, alternative="less")
    return {
        "comparator": comparator,
        "paired_cases": int(improvement.size),
        "median_task_auc_improvement_percent": float(np.median(improvement)),
        "mean_task_auc_improvement_percent": float(np.mean(improvement)),
        "paired_win_rate": float(np.mean(proposed < baseline)),
        "bootstrap_median_improvement_95ci_percent": [
            float(value) for value in np.percentile(bootstrap, (2.5, 97.5))
        ],
        "one_sided_wilcoxon_p_proposed_lower": float(test.pvalue),
        "median_final_violation_rate_reduction": float(
            np.median(baseline_violation - proposed_violation)
        ),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output_directory = project_root / "results" / "effectiveness_v5_task"
    with (output_directory / "task_effectiveness_raw.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    keys = sorted(
        {
            (str(row["trajectory"]), int(row["domain_seed"]))
            for row in rows
        }
    )
    rng = np.random.default_rng(20260717)
    comparisons = {
        comparator: _compare(rows, keys, comparator, rng)
        for comparator in COMPARATORS
    }
    violation_scheduler_comparisons = {
        comparator: _compare(
            rows,
            keys,
            comparator,
            rng,
            primary="violation_dynamic",
        )
        for comparator in (
            "error_peak_dynamic",
            "full_trajectory",
            "jerk_zones",
        )
    }
    indexed = {
        (
            str(row["trajectory"]),
            int(row["domain_seed"]),
            str(row["method"]),
        ): row
        for row in rows
    }
    by_trajectory = {}
    strongest = min(
        COMPARATORS,
        key=lambda method: np.median(
            [
                float(indexed[key + (method,)]["task_auc_normalized"])
                for key in keys
            ]
        ),
    )
    for family in sorted({key[0] for key in keys}):
        family_keys = [key for key in keys if key[0] == family]
        proposed = np.asarray(
            [
                float(indexed[key + (PRIMARY,)]["task_auc_normalized"])
                for key in family_keys
            ]
        )
        baseline = np.asarray(
            [
                float(indexed[key + (strongest,)]["task_auc_normalized"])
                for key in family_keys
            ]
        )
        improvement = 100.0 * (baseline - proposed) / baseline
        by_trajectory[family] = {
            "median_auc_improvement_vs_strongest_percent": float(
                np.median(improvement)
            ),
            "win_rate_vs_strongest": float(np.mean(proposed < baseline)),
        }
    payload = {
        "primary_method": PRIMARY,
        "confirmation_domain_seeds": [431, 449, 461, 479, 491, 509],
        "zone_budget": 2,
        "task_zones": 6,
        "strongest_baseline": strongest,
        "bootstrap_seed": 20260717,
        "comparisons": comparisons,
        "violation_scheduler_comparisons": (
            violation_scheduler_comparisons
        ),
        "by_trajectory_vs_strongest": by_trajectory,
    }
    destination = output_directory / "comparison_v5.json"
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
