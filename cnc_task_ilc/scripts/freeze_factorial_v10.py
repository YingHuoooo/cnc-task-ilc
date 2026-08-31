"""Freeze the V10 combined-stress factorial protocol."""

import json
from pathlib import Path

from cnc_task_ilc.factorial_benchmark import freeze_factorial_protocol


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = freeze_factorial_protocol(
        root / "data" / "tolerance_conflict_v1" / "task_manifests.json",
        root / "results" / "development_v10_factorial" / "factorial_audit_summary.json",
        root / "data" / "tolerance_conflict_v1" / "v9_robustness_protocol.json",
        root / "data" / "tolerance_conflict_v1" / "v10_factorial_protocol.json",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
