"""Run the isolated V12 unknown-delay development experiment."""

import json
from pathlib import Path

from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.delay_generalization_benchmark import (
    DEVELOPMENT_DOMAIN_SEEDS,
    run_delay_generalization_development,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = run_delay_generalization_development(
        root / "data" / "tolerance_conflict_v1" / "task_manifests.json",
        root / "results" / "development_v12_delay_generalization",
        BenchmarkSettings(
            domain_seeds=DEVELOPMENT_DOMAIN_SEEDS,
            number_of_windows=2,
            half_width=5,
        ),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
