"""Run the frozen V6 semantic machining-task confirmation."""

import json
from pathlib import Path

from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.semantic_task_benchmark import run_semantic_task_gate


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    settings = BenchmarkSettings(
        domain_seeds=(631, 647, 659, 677, 691, 709),
        number_of_windows=2,
        half_width=5,
    )
    payload = run_semantic_task_gate(
        project_root / "results" / "effectiveness_v6_semantic",
        settings,
    )
    print(json.dumps(payload["gate"], indent=2, ensure_ascii=False))
