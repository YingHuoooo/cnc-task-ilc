"""Run the V10 baseline/extreme numerical audit."""

import json
from pathlib import Path

from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.factorial_benchmark import AUDIT_DOMAIN_SEEDS, run_factorial_audit


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = run_factorial_audit(
        root / "data" / "tolerance_conflict_v1" / "task_manifests.json",
        root / "results" / "development_v10_factorial",
        BenchmarkSettings(
            domain_seeds=AUDIT_DOMAIN_SEEDS,
            number_of_windows=2,
            half_width=5,
        ),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
