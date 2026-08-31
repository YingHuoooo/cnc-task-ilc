"""Run the prospectively specified supplementary BF-NOILC comparison.

The runner is resumable.  Each completed paired method case is appended to a
JSONL checkpoint before the final CSV files are assembled.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE / "cnc_task_ilc"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(HERE))

from cnc_task_ilc.basis import cubic_bspline_basis
from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.conflict_taskset import specification_from_manifest
from cnc_task_ilc.factorial_benchmark import FACTORIAL_SCENARIOS
from cnc_task_ilc.ilc import build_contour_sensitivity
from cnc_task_ilc.plant import nominal_config
from cnc_task_ilc.robustness_runner import StressScenario
from cnc_task_ilc.trajectory import make_trajectory_family

from experiment_core import BF_NOILC, METHODS, PROPOSED, run_method


FORMAL_PLANT_SEEDS = tuple(range(26001, 26009))
SMOKE_PLANT_SEED = 26999
METHOD_LABELS = {
    PROPOSED: "Proposed",
    BF_NOILC: "Parameter-matched constrained BF-NOILC",
}
CACHE: Dict[Tuple[str, int], Tuple[object, np.ndarray, object, np.ndarray]] = {}


def selected_scenarios() -> Tuple[StressScenario, ...]:
    delay_2 = StressScenario(
        scenario_id="delay_2",
        label="Added delay +2 steps",
        factor="added_delay",
        factor_level=2.0,
        measurement_noise_std_mm=0.0,
        extra_delay_steps=2,
        mismatch_scale=1.0,
    )
    ids = ("n0_d0_m0", "n0_d1_m0", "n1_d1_m1")
    selected = {
        item.scenario_id: item
        for item in FACTORIAL_SCENARIOS
        if item.scenario_id in ids
    }
    return (
        selected["n0_d0_m0"],
        delay_2,
        selected["n0_d1_m0"],
        selected["n1_d1_m1"],
    )


def scenario_dict(scenario: StressScenario) -> Dict[str, object]:
    return {
        "scenario_id": scenario.scenario_id,
        "label": scenario.label,
        "factor": scenario.factor,
        "factor_level": scenario.factor_level,
        "measurement_noise_std_mm": scenario.measurement_noise_std_mm,
        "extra_delay_steps": scenario.extra_delay_steps,
        "mismatch_scale": scenario.mismatch_scale,
    }


def scenario_from_dict(payload: Mapping[str, object]) -> StressScenario:
    return StressScenario(
        scenario_id=str(payload["scenario_id"]),
        label=str(payload["label"]),
        factor=str(payload["factor"]),
        factor_level=float(payload["factor_level"]),
        measurement_noise_std_mm=float(payload["measurement_noise_std_mm"]),
        extra_delay_steps=int(payload["extra_delay_steps"]),
        mismatch_scale=float(payload["mismatch_scale"]),
    )


def load_manifests() -> List[Dict[str, object]]:
    path = PROJECT / "data" / "tolerance_conflict_v1" / "task_manifests.json"
    return list(json.loads(path.read_text(encoding="utf-8"))["manifests"])


def cached_problem(job: Mapping[str, object]):
    manifest = job["manifest"]
    key = (str(manifest["manifest_id"]), 12)
    if key not in CACHE:
        reference = make_trajectory_family(
            str(manifest["trajectory_family"]),
            samples=161,
            duration=6.0,
        )
        basis = cubic_bspline_basis(samples=161, control_points=12)
        specification = specification_from_manifest(reference, manifest)
        sensitivity = build_contour_sensitivity(
            reference,
            basis,
            nominal_config(),
        )
        CACHE[key] = (reference, basis, specification, sensitivity)
    return CACHE[key]


def execute_job(job: Mapping[str, object]) -> Dict[str, object]:
    reference, basis, specification, sensitivity = cached_problem(job)
    settings = BenchmarkSettings(
        samples=161,
        duration=6.0,
        control_points=12,
        iterations=4,
        domain_seeds=(),
        number_of_windows=2,
        half_width=5,
    )
    summary, trials, updates = run_method(
        method=str(job["method"]),
        reference=reference,
        basis=basis,
        specification=specification,
        nominal_sensitivity=sensitivity,
        plant_seed=int(job["plant_seed"]),
        settings=settings,
        scenario=scenario_from_dict(job["scenario"]),
        noise_seed=int(job["noise_seed"]),
    )
    manifest = job["manifest"]
    common = {
        "job_id": str(job["job_id"]),
        "plant_id": f"S{int(job['plant_seed'])}",
        "plant_seed": int(job["plant_seed"]),
        "manifest_id": str(manifest["manifest_id"]),
        "trajectory": str(manifest["trajectory_family"]),
        "regime": str(manifest["regime"]),
        "scenario_id": str(job["scenario"]["scenario_id"]),
        "scenario_label": str(job["scenario"]["label"]),
        "noise_seed": int(job["noise_seed"]),
        "method": str(job["method"]),
        "method_label": METHOD_LABELS[str(job["method"])],
    }
    run_row = dict(common)
    run_row.update(summary)
    trial_rows = []
    for row in trials:
        item = dict(common)
        item.update(row)
        trial_rows.append(item)
    update_rows = []
    for row in updates:
        item = dict(common)
        item.update(row)
        update_rows.append(item)
    return {"run": run_row, "trials": trial_rows, "updates": update_rows}


def build_jobs(smoke: bool) -> List[Dict[str, object]]:
    manifests = load_manifests()
    scenarios = selected_scenarios()
    if smoke:
        manifests = [
            next(item for item in manifests if item["manifest_id"] == "s_curve--demand_conflict")
        ]
        scenarios = (scenarios[0],)
        seeds = (SMOKE_PLANT_SEED,)
    else:
        seeds = FORMAL_PLANT_SEEDS
    jobs = []
    for scenario_index, scenario in enumerate(scenarios):
        for manifest_index, manifest in enumerate(manifests):
            for plant_seed in seeds:
                noise_seed = (
                    310000
                    + 10000 * scenario_index
                    + 100 * manifest_index
                    + (int(plant_seed) % 100)
                )
                for method in METHODS:
                    jobs.append(
                        {
                            "job_id": "|".join(
                                (
                                    str(manifest["manifest_id"]),
                                    scenario.scenario_id,
                                    str(plant_seed),
                                    method,
                                )
                            ),
                            "manifest": dict(manifest),
                            "scenario": scenario_dict(scenario),
                            "plant_seed": int(plant_seed),
                            "noise_seed": int(noise_seed),
                            "method": method,
                        }
                    )
    return jobs


def completed_ids(path: Path) -> set:
    if not path.is_file():
        return set()
    values = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.add(str(json.loads(line)["run"]["job_id"]))
    return values


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError("cannot write an empty CSV: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def assemble(checkpoint: Path, output: Path) -> None:
    payloads = []
    with checkpoint.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payloads.append(json.loads(line))
    payloads.sort(key=lambda item: item["run"]["job_id"])
    runs = [item["run"] for item in payloads]
    trials = [row for item in payloads for row in item["trials"]]
    updates = [row for item in payloads for row in item["updates"]]
    write_csv(output / "raw_runs.csv", runs)
    write_csv(output / "trial_history.csv", trials)
    write_csv(output / "update_history.csv", updates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args.smoke)
    output = ROOT / ("smoke" if args.smoke else "results")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "raw_checkpoint.jsonl"
    done = completed_ids(checkpoint)
    pending = [job for job in jobs if job["job_id"] not in done]
    print(
        f"mode={'smoke' if args.smoke else 'formal'} total={len(jobs)} "
        f"completed={len(done)} pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    start = time.perf_counter()
    if pending:
        context = mp.get_context("fork")
        with checkpoint.open("a", encoding="utf-8") as handle:
            with context.Pool(
                processes=max(1, int(args.workers)),
                maxtasksperchild=80,
            ) as pool:
                for index, payload in enumerate(
                    pool.imap_unordered(execute_job, pending, chunksize=1),
                    1,
                ):
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
                    if index % 25 == 0 or index == len(pending):
                        print(f"completed {index}/{len(pending)} pending jobs", flush=True)
    assemble(checkpoint, output)
    elapsed = time.perf_counter() - start
    metadata = {
        "mode": "smoke" if args.smoke else "formal",
        "expected_method_runs": len(jobs),
        "expected_trial_rows": len(jobs) * 5,
        "expected_update_rows": len(jobs) * 4,
        "plant_seeds": [SMOKE_PLANT_SEED] if args.smoke else list(FORMAL_PLANT_SEEDS),
        "methods": list(METHODS),
        "scenario_ids": [item.scenario_id for item in (selected_scenarios() if not args.smoke else selected_scenarios()[:1])],
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "elapsed_s_this_invocation": elapsed,
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"assembled outputs in {output}", flush=True)


if __name__ == "__main__":
    main()
