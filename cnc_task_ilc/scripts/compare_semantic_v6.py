"""Create paired statistics for the frozen V6 semantic-task experiment."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


PRIMARY = "violation_safe"
COMPARATORS = (
    "violation_dynamic",
    "error_peak_dynamic",
    "full_trajectory",
    "static_tolerance",
    "curvature_zones",
    "jerk_zones",
    "random_zones",
)


def _compare(rows, keys, comparator, rng):
    indexed = {
        (
            str(row["trajectory"]),
            int(row["domain_seed"]),
            str(row["method"]),
        ): row
        for row in rows
    }
    proposed = np.asarray(
        [float(indexed[key + (PRIMARY,)]["task_auc_normalized"]) for key in keys]
    )
    baseline = np.asarray(
        [float(indexed[key + (comparator,)]["task_auc_normalized"]) for key in keys]
    )
    proposed_global = np.asarray(
        [float(indexed[key + (PRIMARY,)]["final_global_ratio"]) for key in keys]
    )
    baseline_global = np.asarray(
        [float(indexed[key + (comparator,)]["final_global_ratio"]) for key in keys]
    )
    proposed_violation = np.asarray(
        [float(indexed[key + (PRIMARY,)]["final_violation_rate"]) for key in keys]
    )
    baseline_violation = np.asarray(
        [float(indexed[key + (comparator,)]["final_violation_rate"]) for key in keys]
    )
    improvement = 100.0 * (baseline - proposed) / baseline
    samples = rng.integers(0, improvement.size, size=(20000, improvement.size))
    bootstrap = np.median(improvement[samples], axis=1)
    if np.allclose(proposed, baseline):
        p_value = 1.0
    else:
        p_value = float(
            wilcoxon(proposed, baseline, alternative="less").pvalue
        )
    return {
        "comparator": comparator,
        "paired_cases": int(improvement.size),
        "median_task_auc_improvement_percent": float(np.median(improvement)),
        "mean_task_auc_improvement_percent": float(np.mean(improvement)),
        "paired_win_rate": float(np.mean(proposed < baseline)),
        "paired_tie_rate": float(np.mean(np.isclose(proposed, baseline))),
        "bootstrap_median_improvement_95ci_percent": [
            float(value) for value in np.percentile(bootstrap, (2.5, 97.5))
        ],
        "one_sided_wilcoxon_p_proposed_lower": p_value,
        "median_global_ratio_multiplier": float(
            np.median(proposed_global / np.maximum(baseline_global, 1.0e-12))
        ),
        "median_final_violation_rate_reduction": float(
            np.median(baseline_violation - proposed_violation)
        ),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output_directory = project_root / "results" / "effectiveness_v6_semantic"
    with (output_directory / "semantic_effectiveness_raw.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    keys = sorted(
        {
            (str(row["trajectory"]), int(row["domain_seed"]))
            for row in rows
        }
    )
    rng = np.random.default_rng(20260718)
    comparisons = {
        comparator: _compare(rows, keys, comparator, rng)
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
                float(indexed[key + (PRIMARY,)]["task_auc_normalized"])
                for key in family_keys
            ]
        )
        baseline = np.asarray(
            [
                float(
                    indexed[key + ("full_trajectory",)]["task_auc_normalized"]
                )
                for key in family_keys
            ]
        )
        improvement = 100.0 * (baseline - proposed) / baseline
        by_trajectory[family] = {
            "median_auc_improvement_vs_full_percent": float(
                np.median(improvement)
            ),
            "win_rate_vs_full": float(np.mean(proposed < baseline)),
        }

    different_selection_keys = [
        key
        for key in keys
        if json.loads(indexed[key + (PRIMARY,)]["selection_history"])
        != json.loads(
            indexed[key + ("error_peak_dynamic",)]["selection_history"]
        )
    ]
    proposed_different = np.asarray(
        [
            float(indexed[key + (PRIMARY,)]["task_auc_normalized"])
            for key in different_selection_keys
        ]
    )
    error_peak_different = np.asarray(
        [
            float(
                indexed[key + ("error_peak_dynamic",)]["task_auc_normalized"]
            )
            for key in different_selection_keys
        ]
    )
    different_improvement = 100.0 * (
        error_peak_different - proposed_different
    ) / error_peak_different

    safe_final = np.asarray(
        [float(indexed[key + (PRIMARY,)]["final_task_ratio"]) for key in keys]
    )
    plain_final = np.asarray(
        [
            float(indexed[key + ("violation_dynamic",)]["final_task_ratio"])
            for key in keys
        ]
    )
    rejected_trials = np.asarray(
        [int(indexed[key + (PRIMARY,)]["rejected_trials"]) for key in keys]
    )
    payload = {
        "primary_method": PRIMARY,
        "confirmation_domain_seeds": [631, 647, 659, 677, 691, 709],
        "zone_budget": 2,
        "task_zones": 6,
        "bootstrap_seed": 20260718,
        "comparisons": comparisons,
        "by_trajectory_vs_full": by_trajectory,
        "posthoc_selection_difference_vs_error_peak": {
            "different_selection_cases": len(different_selection_keys),
            "identical_selection_cases": len(keys) - len(different_selection_keys),
            "median_auc_improvement_when_different_percent": float(
                np.median(different_improvement)
            ),
            "win_rate_when_different": float(
                np.mean(proposed_different < error_peak_different)
            ),
        },
        "rollback_diagnostic": {
            "cases_with_rejection": int(np.sum(rejected_trials > 0)),
            "total_rejected_trials": int(np.sum(rejected_trials)),
            "cases_with_better_final_output_than_plain": int(
                np.sum(safe_final < plain_final)
            ),
            "cases_with_worse_final_output_than_plain": int(
                np.sum(safe_final > plain_final)
            ),
        },
    }
    destination = output_directory / "comparison_v6.json"
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
