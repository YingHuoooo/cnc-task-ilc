"""Run the revised learned-window effectiveness gate."""

import json
from pathlib import Path

from cnc_task_ilc.benchmark import run_effectiveness_gate


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    payload = run_effectiveness_gate(
        project_root / "results" / "effectiveness_v2"
    )
    print(json.dumps(payload["gate"], indent=2, ensure_ascii=False))
