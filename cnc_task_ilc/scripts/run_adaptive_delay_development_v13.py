"""Run the one-shot V13 adaptive-delay development screen."""

import json
from pathlib import Path

from cnc_task_ilc.adaptive_delay_benchmark import (
    DEVELOPMENT_DOMAIN_SEEDS,
    run_adaptive_delay_development,
)
from cnc_task_ilc.benchmark import BenchmarkSettings


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = run_adaptive_delay_development(
        root / "data" / "tolerance_conflict_v1" / "task_manifests.json",
        root / "results" / "development_v13_adaptive_delay",
        BenchmarkSettings(
            domain_seeds=DEVELOPMENT_DOMAIN_SEEDS,
            number_of_windows=2,
            half_width=5,
        ),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
