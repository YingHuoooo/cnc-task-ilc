"""Run the frozen V9 robustness confirmation."""

import json
from pathlib import Path

from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.robustness_benchmark import (
    FORMAL_DOMAIN_SEEDS,
    run_robustness_confirmation,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = run_robustness_confirmation(
        root / "data" / "tolerance_conflict_v1" / "task_manifests.json",
        root / "results" / "development_v9_robustness" / "robustness_audit_summary.json",
        root / "data" / "tolerance_conflict_v1" / "v8_preregistered_protocol.json",
        root / "data" / "tolerance_conflict_v1" / "v9_robustness_protocol.json",
        root / "results" / "effectiveness_v9_robustness",
        BenchmarkSettings(
            domain_seeds=FORMAL_DOMAIN_SEEDS,
            number_of_windows=2,
            half_width=5,
        ),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
