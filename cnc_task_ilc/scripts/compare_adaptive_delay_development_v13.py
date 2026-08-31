"""Analyze and archive the stopped V13 development experiment."""

import json
from pathlib import Path

from cnc_task_ilc.adaptive_delay_benchmark import (
    analyze_adaptive_delay_development,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = analyze_adaptive_delay_development(
        root / "results" / "development_v13_adaptive_delay"
    )
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "final_selected_method": payload["final_selected_method"],
                "formal_domain_seeds_used": payload[
                    "formal_domain_seeds_used"
                ],
                "development_criteria": payload["development_criteria"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
