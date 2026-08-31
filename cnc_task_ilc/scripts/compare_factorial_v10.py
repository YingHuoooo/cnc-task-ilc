"""Analyze the frozen V10 combined-stress factorial experiment."""

import json
from pathlib import Path

from cnc_task_ilc.factorial_benchmark import analyze_factorial_confirmation


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = analyze_factorial_confirmation(
        root / "results" / "effectiveness_v10_factorial"
    )
    print(json.dumps({
        "classification": payload["classification"],
        "criteria": payload["criteria"],
        "extreme_primary_vs_error_peak": payload[
            "extreme_primary_vs_error_peak"
        ],
        "extreme_primary_absolute_summary": payload[
            "extreme_primary_absolute_summary"
        ],
    }, indent=2, ensure_ascii=False))
