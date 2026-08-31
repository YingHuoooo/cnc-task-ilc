"""Validate the completed five-priority experiment package and build a manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPORT = ROOT / "qa" / "validation_report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class Checks:
    def __init__(self) -> None:
        self.items: List[Dict[str, object]] = []

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.items.append(
            {"name": name, "status": "pass" if condition else "fail", "detail": detail}
        )


def build_manifest() -> Dict[str, object]:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.name == "MANIFEST.json"
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "package": "v11_additional_experiments",
        "linuxcnc_executed": False,
        "physical_machine_executed": False,
        "file_count_excluding_manifest": len(files),
        "files": files,
    }
    (ROOT / "MANIFEST.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    checks = Checks()
    required = [
        "README.md",
        "experiment_design.md",
        "figure_contracts.md",
        "analysis_report.md",
        "protocol_pre_execution.json",
        "scripts/experiment_core.py",
        "scripts/run_experiments.py",
        "scripts/analyze.py",
        "results/01_matched_ablation/raw_results.csv",
        "results/01_matched_ablation/ablation_summary.csv",
        "results/02_hierarchical_statistics/bootstrap_comparison.csv",
        "results/03_parameter_sensitivity/raw_results.csv",
        "results/03_parameter_sensitivity/sensitivity_summary.csv",
        "results/04_representative_replay/pointwise_trace.csv",
        "results/04_representative_replay/full_trace_arrays.npz",
        "results/05_virtual_plant_family/raw_results.csv",
        "results/05_virtual_plant_family/plant_parameters.csv",
        "results/05_virtual_plant_family/plant_level_effects.csv",
    ]
    missing = [item for item in required if not (ROOT / item).is_file()]
    checks.check("complete_artifact_set", not missing, f"missing={missing}")

    protocol = json.loads((ROOT / "protocol_pre_execution.json").read_text(encoding="utf-8"))
    checks.check(
        "scope_excludes_linuxcnc_and_physical_machine",
        protocol["linuxcnc_executed"] is False
        and protocol["physical_machine_executed"] is False
        and protocol["original_v8_v13_results_modified"] is False,
        "new package must remain numerical-only and preserve old evidence",
    )
    code_hashes = protocol["code_sha256"]
    checks.check(
        "pre_execution_code_hashes",
        code_hashes["experiment_core.py"] == sha256(ROOT / "scripts" / "experiment_core.py")
        and code_hashes["run_experiments.py"] == sha256(ROOT / "scripts" / "run_experiments.py"),
        "execution code must match the pre-execution protocol",
    )

    raw_specs = [
        ("ablation", ROOT / "results/01_matched_ablation/raw_results.csv", 1800, 5),
        ("sensitivity", ROOT / "results/03_parameter_sensitivity/raw_results.csv", 780, 1),
        ("plant_family", ROOT / "results/05_virtual_plant_family/raw_results.csv", 1800, 4),
    ]
    for label, path, expected, method_count in raw_specs:
        data = rows(path)
        checks.check(
            f"{label}_row_count",
            len(data) == expected,
            f"expected={expected}, observed={len(data)}",
        )
        ids = [row["job_id"] for row in data]
        checks.check(
            f"{label}_job_ids_unique",
            len(ids) == len(set(ids)),
            f"duplicates={len(ids)-len(set(ids))}",
        )
        finite = all(
            int(row["finite_result"]) == 1
            and math.isfinite(float(row["task_auc_normalized"]))
            for row in data
        )
        checks.check(f"{label}_finite", finite, "all results must be finite")
        success_rate = sum(
            int(row["all_updates_succeeded"]) == 1
            and int(row["constraint_violation"]) == 0
            for row in data
        ) / len(data)
        checks.check(
            f"{label}_solver_constraint_success_at_least_95_percent",
            success_rate >= 0.95,
            f"observed={success_rate:.6f}",
        )

    ablation = rows(ROOT / "results/01_matched_ablation/raw_results.csv")
    grouped: Dict[tuple, set] = {}
    for row in ablation:
        key = (row["manifest_id"], row["plant_id"], row["scenario_id"])
        grouped.setdefault(key, set()).add(row["method"])
    expected_methods = {
        "v11_full", "no_residual_alignment", "task_top2", "raw_top2", "uniform_full_trajectory"
    }
    checks.check(
        "ablation_matched_method_completeness",
        len(grouped) == 360 and all(value == expected_methods for value in grouped.values()),
        f"paired_keys={len(grouped)}",
    )

    plant_parameters = rows(ROOT / "results/05_virtual_plant_family/plant_parameters.csv")
    held = [row for row in plant_parameters if row["plant_group"] == "held_out_lhs"]
    challenge = [row for row in plant_parameters if row["plant_group"] == "edge_challenge"]
    checks.check(
        "plant_family_counts",
        len(held) == 24 and len(challenge) == 6,
        f"held_out={len(held)}, challenge={len(challenge)}",
    )

    stats = rows(ROOT / "results/02_hierarchical_statistics/bootstrap_comparison.csv")
    checks.check(
        "bootstrap_statistics_complete",
        len(stats) == 21
        and all(int(row["domain_n"]) in (8, 24) for row in stats)
        and all(float(row["paired_ci_low"]) <= float(row["paired_ci_high"]) for row in stats)
        and all(float(row["domain_ci_low"]) <= float(row["domain_ci_high"]) for row in stats)
        and all(float(row["hierarchical_ci_low"]) <= float(row["hierarchical_ci_high"]) for row in stats),
        f"rows={len(stats)}",
    )
    sensitivity = rows(ROOT / "results/03_parameter_sensitivity/sensitivity_summary.csv")
    checks.check(
        "sensitivity_summary_complete",
        len(sensitivity) == 32
        and {row["parameter"] for row in sensitivity}
        == {"residual_delay_shrinkage", "smoothing_window", "nominal_control_points", "learning_rate"},
        f"rows={len(sensitivity)}, parameters={sorted(set(row['parameter'] for row in sensitivity))}",
    )
    replay = rows(ROOT / "results/04_representative_replay/pointwise_trace.csv")
    checks.check(
        "replay_pointwise_complete",
        len(replay) == 161 and all(key in replay[0] for key in ("reference_x_mm", "v11_x_mm", "no_alignment_x_mm")),
        f"rows={len(replay)}",
    )

    figure_stems = [
        "fig1_matched_ablation",
        "fig2_domain_aware_statistics",
        "fig3_parameter_sensitivity",
        "fig4_representative_replay",
        "fig5_virtual_plant_family",
    ]
    for stem in figure_stems:
        paths = [ROOT / "figures" / f"{stem}.{suffix}" for suffix in ("svg", "pdf", "tiff", "png")]
        checks.check(
            f"{stem}_four_formats",
            all(path.is_file() and path.stat().st_size > 1000 for path in paths),
            "SVG/PDF/TIFF/PNG must exist",
        )
        svg = paths[0].read_text(encoding="utf-8") if paths[0].is_file() else ""
        checks.check(
            f"{stem}_editable_svg_text",
            "<text" in svg,
            "SVG must preserve editable text",
        )
        if paths[2].is_file() and paths[3].is_file():
            with Image.open(paths[2]) as image:
                tiff_dpi = image.info.get("dpi", (0, 0))
            with Image.open(paths[3]) as image:
                png_dpi = image.info.get("dpi", (0, 0))
            checks.check(
                f"{stem}_raster_resolution",
                min(tiff_dpi) >= 599 and min(png_dpi) >= 299,
                f"tiff_dpi={tiff_dpi}, png_dpi={png_dpi}",
            )

    failed = [item for item in checks.items if item["status"] == "fail"]
    report = {
        "status": "pass" if not failed else "fail",
        "linuxcnc_executed": False,
        "physical_machine_executed": False,
        "new_numerical_method_runs": 4380,
        "checks_passed": len(checks.items) - len(failed),
        "checks_failed": len(failed),
        "checks": checks.items,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = build_manifest()
    report["manifest_files"] = manifest["file_count_excluding_manifest"]
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    build_manifest()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

