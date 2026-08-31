"""Freeze the V9 robustness protocol after its numerical audit."""

import json
from pathlib import Path

from cnc_task_ilc.robustness_benchmark import freeze_robustness_protocol


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = freeze_robustness_protocol(
        root / "data" / "tolerance_conflict_v1" / "task_manifests.json",
        root / "results" / "development_v9_robustness" / "robustness_audit_summary.json",
        root / "data" / "tolerance_conflict_v1" / "v8_preregistered_protocol.json",
        root / "data" / "tolerance_conflict_v1" / "v9_robustness_protocol.json",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
