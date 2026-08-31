"""Run the frozen V7 tolerance-conflict confirmation."""

import json
from pathlib import Path

from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.conflict_benchmark import (
    FORMAL_DOMAIN_SEEDS,
    run_conflict_confirmation,
)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    payload = run_conflict_confirmation(
        project_root / "data" / "tolerance_conflict_v1" / "task_manifests.json",
        project_root / "results" / "effectiveness_v7_conflict",
        BenchmarkSettings(
            domain_seeds=FORMAL_DOMAIN_SEEDS,
            number_of_windows=2,
            half_width=5,
        ),
    )
    print(json.dumps(payload["settings"], indent=2, ensure_ascii=False))
