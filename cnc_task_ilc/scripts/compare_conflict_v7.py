"""Analyze the frozen V7 tolerance-conflict confirmation."""

import json
from pathlib import Path

from cnc_task_ilc.conflict_benchmark import analyze_conflict_confirmation


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    payload = analyze_conflict_confirmation(
        project_root / "results" / "effectiveness_v7_conflict"
    )
    print(json.dumps(payload["gate"], indent=2, ensure_ascii=False))
