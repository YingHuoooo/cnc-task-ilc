"""Run the frozen V10 2^3 factorial confirmation."""

import json
from pathlib import Path

from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.factorial_benchmark import (
    FORMAL_DOMAIN_SEEDS,
    run_factorial_confirmation,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = run_factorial_confirmation(
        root / "data" / "tolerance_conflict_v1" / "task_manifests.json",
        root / "results" / "development_v10_factorial" / "factorial_audit_summary.json",
        root / "data" / "tolerance_conflict_v1" / "v9_robustness_protocol.json",
        root / "data" / "tolerance_conflict_v1" / "v10_factorial_protocol.json",
        root / "results" / "effectiveness_v10_factorial",
        BenchmarkSettings(
            domain_seeds=FORMAL_DOMAIN_SEEDS,
            number_of_windows=2,
            half_width=5,
        ),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
