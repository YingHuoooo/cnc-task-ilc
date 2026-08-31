"""Analyze and plot the frozen V9 robustness confirmation."""

import json
from pathlib import Path

from cnc_task_ilc.robustness_benchmark import analyze_robustness_confirmation


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = analyze_robustness_confirmation(
        root / "results" / "effectiveness_v9_robustness"
    )
    print(json.dumps({
        "classification": payload["classification"],
        "robust_effect_flags": payload["robust_effect_flags"],
        "factor_boundaries": payload["factor_boundaries"],
        "primary_success_rate_by_scenario": payload[
            "primary_success_rate_by_scenario"
        ],
    }, indent=2, ensure_ascii=False))
