"""Run the frozen V11 online delay-compensation confirmation."""

import json
from pathlib import Path

from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.delay_compensation_benchmark import (
    FORMAL_DOMAIN_SEEDS,
    run_delay_confirmation,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = run_delay_confirmation(
        root / "data" / "tolerance_conflict_v1" / "task_manifests.json",
        root
        / "results"
        / "development_v11_delay_compensation"
        / "delay_compensation_development_summary.json",
        root / "data" / "tolerance_conflict_v1" / "v10_factorial_protocol.json",
        root / "data" / "tolerance_conflict_v1" / "v11_delay_protocol.json",
        root / "results" / "effectiveness_v11_delay_compensation",
        BenchmarkSettings(
            domain_seeds=FORMAL_DOMAIN_SEEDS,
            number_of_windows=2,
            half_width=5,
        ),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
