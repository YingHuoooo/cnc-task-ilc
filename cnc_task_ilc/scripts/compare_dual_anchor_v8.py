"""Analyze the frozen V8 dual-anchor confirmation."""

import json
from pathlib import Path

from cnc_task_ilc.dual_anchor_confirmation import (
    analyze_dual_anchor_confirmation,
)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    payload = analyze_dual_anchor_confirmation(
        project_root / "results" / "effectiveness_v8_dual_anchor"
    )
    print(json.dumps(payload["gate"], indent=2, ensure_ascii=False))
