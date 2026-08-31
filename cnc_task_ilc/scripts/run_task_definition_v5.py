"""Run the externally specified machining-task V5 gate."""

import json
from pathlib import Path

from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.task_benchmark import run_task_definition_gate


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    settings = BenchmarkSettings(
        domain_seeds=(431, 449, 461, 479, 491, 509),
        number_of_windows=2,
        half_width=5,
    )
    payload = run_task_definition_gate(
        project_root / "results" / "effectiveness_v5_task",
        settings,
    )
    print(json.dumps(payload["gate"], indent=2, ensure_ascii=False))
