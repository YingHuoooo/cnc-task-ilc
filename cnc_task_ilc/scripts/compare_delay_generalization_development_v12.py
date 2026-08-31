"""Analyze the stopped V12 development experiment without formal seeds."""

import json
from pathlib import Path

from cnc_task_ilc.delay_generalization_benchmark import (
    analyze_delay_generalization_development,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = analyze_delay_generalization_development(
        root / "results" / "development_v12_delay_generalization"
    )
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "formal_domain_seeds_used": payload["formal_domain_seeds_used"],
                "selected_best_fixed_method": payload[
                    "selected_best_fixed_method"
                ],
                "development_criteria": payload["development_criteria"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
