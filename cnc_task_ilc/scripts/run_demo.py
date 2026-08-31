"""Run the local feasibility demo from the source checkout."""

import json
from pathlib import Path

from cnc_task_ilc.demo import run_demo


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    result = run_demo(project_root / "results")
    print(json.dumps(result, indent=2, ensure_ascii=False))

