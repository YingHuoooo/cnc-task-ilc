"""Create paired statistics for the V4 confirmation experiment."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


PRIMARY = "safe_combo_eta_0.30"
COMPARATORS = (
    "error_peak_window",
    "learned_auto_eta_1.00",
    "automatic_eta_0.30",
    "automatic_eta_1.00",
    "full_trajectory",
    "curvature_window",
    "jerk_window",
    "random_window",
)


def _paired_comparison(rows, keys, comparator, rng):
    indexed = {
        (
            str(row["trajectory"]),
            int(row["domain_seed"]),
            str(row["method"]),
        ): row
        for row in rows
    }
    proposed = np.asarray(
        [
            float(indexed[key + (PRIMARY,)]["critical_auc_normalized"])
            for key in keys
        ]
    )
    baseline = np.asarray(
        [
            float(
                indexed[key + (comparator,)]["critical_auc_normalized"]
            )
            for key in keys
        ]
    )
    improvement = 100.0 * (baseline - proposed) / baseline
    bootstrap_indices = rng.integers(
        0,
        improvement.size,
        size=(20000, improvement.size),
    )
    bootstrap_medians = np.median(
        improvement[bootstrap_indices],
        axis=1,
    )
    proposed_global = np.asarray(
        [
            float(indexed[key + (PRIMARY,)]["final_global_ratio"])
            for key in keys
        ]
    )
    baseline_global = np.asarray(
        [
            float(indexed[key + (comparator,)]["final_global_ratio"])
            for key in keys
        ]
    )
    difference = proposed - baseline
    nonzero = int(np.count_nonzero(np.abs(difference) > 1.0e-14))
    p_value = (
        float(wilcoxon(proposed, baseline, alternative="less").pvalue)
        if nonzero
        else 1.0
    )
    return {
        "comparator": comparator,
        "paired_cases": int(improvement.size),
        "median_auc_improvement_percent": float(
            np.median(improvement)
        ),
        "mean_auc_improvement_percent": float(np.mean(improvement)),
        "paired_win_rate": float(np.mean(proposed < baseline)),
        "paired_tie_rate": float(
            np.mean(np.abs(difference) <= 1.0e-14)
        ),
        "bootstrap_median_improvement_95ci_percent": [
            float(value)
            for value in np.percentile(
                bootstrap_medians,
                (2.5, 97.5),
            )
        ],
        "one_sided_wilcoxon_p_proposed_lower": p_value,
        "median_global_ratio_multiplier": float(
            np.median(proposed_global / baseline_global)
        ),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output_directory = project_root / "results" / "effectiveness_v4"
    with (output_directory / "effectiveness_raw.csv").open(
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
        comparator: _paired_comparison(
            rows,
            keys,
            comparator,
            rng,
        )
        for comparator in COMPARATORS
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
    for family in sorted({key[0] for key in keys}):
        family_keys = [key for key in keys if key[0] == family]
        proposed = np.asarray(
            [
                float(
                    indexed[key + (PRIMARY,)][
                        "critical_auc_normalized"
                    ]
                )
                for key in family_keys
            ]
        )
        baseline = np.asarray(
            [
                float(
                    indexed[key + ("error_peak_window",)][
                        "critical_auc_normalized"
                    ]
                )
                for key in family_keys
            ]
        )
        improvement = 100.0 * (baseline - proposed) / baseline
        by_trajectory[family] = {
            "median_auc_improvement_vs_error_peak_percent": float(
                np.median(improvement)
            ),
            "win_rate_vs_error_peak": float(
                np.mean(proposed < baseline)
            ),
            "tie_rate_vs_error_peak": float(
                np.mean(np.abs(proposed - baseline) <= 1.0e-14)
            ),
        }
    payload = {
        "primary_method": PRIMARY,
        "confirmation_domain_seeds": [109, 127, 149, 173, 197, 223],
        "comparison_design": (
            "paired on previously unseen virtual-machine domains"
        ),
        "bootstrap_seed": 20260717,
        "comparisons": comparisons,
        "by_trajectory_vs_error_peak": by_trajectory,
    }
    destination = output_directory / "comparison_v4.json"
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
