"""Run the frozen V8 dual-anchor confirmation."""

import json
from pathlib import Path

from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.dual_anchor_confirmation import (
    FORMAL_DOMAIN_SEEDS,
    run_dual_anchor_confirmation,
)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    payload = run_dual_anchor_confirmation(
        project_root / "data" / "tolerance_conflict_v1" / "task_manifests.json",
        project_root
        / "results"
        / "development_v8_dual_anchor"
        / "dual_anchor_development_summary.json",
        project_root
        / "data"
        / "tolerance_conflict_v1"
        / "v8_preregistered_protocol.json",
        project_root / "results" / "effectiveness_v8_dual_anchor",
        BenchmarkSettings(
            domain_seeds=FORMAL_DOMAIN_SEEDS,
            number_of_windows=2,
            half_width=5,
        ),
    )
    print(json.dumps({
        "total_paired_cases": payload["total_paired_cases"],
        "total_method_runs": payload["total_method_runs"],
        "protocol_sha256": payload["protocol_sha256"],
    }, indent=2, ensure_ascii=False))
