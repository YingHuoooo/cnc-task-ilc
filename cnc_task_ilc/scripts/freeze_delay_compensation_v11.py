"""Freeze the V11 fractional online delay-compensation protocol."""

import json
from pathlib import Path

from cnc_task_ilc.delay_compensation_benchmark import freeze_delay_protocol


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    payload = freeze_delay_protocol(
        root / "data" / "tolerance_conflict_v1" / "task_manifests.json",
        root
        / "results"
        / "development_v11_delay_compensation"
        / "delay_compensation_development_summary.json",
        root / "data" / "tolerance_conflict_v1" / "v10_factorial_protocol.json",
        root / "data" / "tolerance_conflict_v1" / "v11_delay_protocol.json",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
