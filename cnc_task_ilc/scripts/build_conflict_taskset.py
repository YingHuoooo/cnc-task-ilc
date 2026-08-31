"""Build and audit the feedback-independent tolerance-conflict taskset."""

import json
from pathlib import Path

from cnc_task_ilc.conflict_taskset import (
    build_conflict_taskset,
    write_taskset_audit,
)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output_directory = project_root / "data" / "tolerance_conflict_v1"
    taskset = build_conflict_taskset()
    payload = write_taskset_audit(
        output_directory,
        taskset,
        audit_seeds=(733, 751, 769),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
