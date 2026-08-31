"""Analyze the frozen V11 online delay-compensation experiment."""

import json
from pathlib import Path

from cnc_task_ilc.delay_compensation_benchmark import analyze_delay_confirmation


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = analyze_delay_confirmation(
        root / "results" / "effectiveness_v11_delay_compensation",
        root
        / "results"
        / "development_v11_delay_compensation"
        / "delay_compensation_development_summary.json",
    )
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "criteria": payload["criteria"],
                "primary_success_rate_by_scenario": payload[
                    "primary_success_rate_by_scenario"
                ],
                "median_axis_lag_absolute_error_steps": payload[
                    "median_axis_lag_absolute_error_steps"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
