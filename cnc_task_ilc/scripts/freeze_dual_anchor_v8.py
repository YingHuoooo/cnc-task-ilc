"""Freeze and verify the V8 protocol before formal execution."""

import json
from pathlib import Path

from cnc_task_ilc.dual_anchor_confirmation import freeze_formal_protocol


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    protocol = freeze_formal_protocol(
        project_root / "data" / "tolerance_conflict_v1" / "task_manifests.json",
        project_root
        / "results"
        / "development_v8_dual_anchor"
        / "dual_anchor_development_summary.json",
        project_root
        / "data"
        / "tolerance_conflict_v1"
        / "v8_preregistered_protocol.json",
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))
