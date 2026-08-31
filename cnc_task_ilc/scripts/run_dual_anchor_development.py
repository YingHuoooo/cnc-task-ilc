"""Run the isolated V8 dual-anchor development screen."""

import json
from pathlib import Path

from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.dual_anchor_development import (
    DEVELOPMENT_DOMAIN_SEEDS,
    run_dual_anchor_development,
)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    result = run_dual_anchor_development(
        project_root / "data" / "tolerance_conflict_v1" / "task_manifests.json",
        project_root / "results" / "development_v8_dual_anchor",
        BenchmarkSettings(
            domain_seeds=DEVELOPMENT_DOMAIN_SEEDS,
            number_of_windows=2,
            half_width=5,
        ),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
