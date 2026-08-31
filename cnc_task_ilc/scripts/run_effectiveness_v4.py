"""Run the risk-controlled safe-combination confirmation gate."""

import json
from pathlib import Path

from cnc_task_ilc.benchmark import (
    BenchmarkSettings,
    run_effectiveness_gate,
)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    confirmation_settings = BenchmarkSettings(
        domain_seeds=(109, 127, 149, 173, 197, 223),
    )
    payload = run_effectiveness_gate(
        project_root / "results" / "effectiveness_v4",
        settings=confirmation_settings,
        primary_method="safe_combo_eta_0.30",
        include_learned=True,
        include_reward_ranker=False,
        include_safe_combo=True,
    )
    print(json.dumps(payload["gate"], indent=2, ensure_ascii=False))
