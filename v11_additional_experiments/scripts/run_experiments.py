"""Execute priorities 1, 3 and 5 with resumable multiprocessing.

Priorities 2 and 4 are derived after these numerical runs by ``analyze.py``.
The script writes one JSON object per completed job before producing CSV files,
so an interrupted run can resume without repeating finished simulations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/v11-additional-mpl")

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parent
PROJECT = WORKSPACE / "cnc_task_ilc"
sys.path.insert(0, str(PROJECT / "src"))

from cnc_task_ilc.basis import cubic_bspline_basis
from cnc_task_ilc.benchmark import BenchmarkSettings
from cnc_task_ilc.conflict_taskset import specification_from_manifest
from cnc_task_ilc.factorial_benchmark import FACTORIAL_SCENARIOS
from cnc_task_ilc.ilc import build_contour_sensitivity
from cnc_task_ilc.plant import AxisDynamics, VirtualPlantConfig, nominal_config
from cnc_task_ilc.robustness_runner import StressScenario
from cnc_task_ilc.trajectory import make_trajectory_family

from experiment_core import (
    MATCHED_METHODS,
    base_plant_from_job,
    plant_to_dict,
    run_matched_method,
    scenario_dict,
    scenario_from_dict,
)


CACHE: Dict[Tuple[str, int], Tuple[object, np.ndarray, object, np.ndarray]] = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_scenarios() -> Tuple[StressScenario, ...]:
    ids = ("n0_d0_m0", "n0_d1_m0", "n1_d1_m1")
    return tuple(next(item for item in FACTORIAL_SCENARIOS if item.scenario_id == key) for key in ids)


def load_manifests() -> List[Dict[str, object]]:
    path = PROJECT / "data" / "tolerance_conflict_v1" / "task_manifests.json"
    return list(json.loads(path.read_text(encoding="utf-8"))["manifests"])


def cached_problem(job: Mapping[str, object]):
    manifest = job["manifest"]
    control_points = int(job["control_points"])
    key = (str(manifest["manifest_id"]), control_points)
    if key not in CACHE:
        reference = make_trajectory_family(
            str(manifest["trajectory_family"]),
            samples=161,
            duration=6.0,
        )
        basis = cubic_bspline_basis(samples=161, control_points=control_points)
        specification = specification_from_manifest(reference, manifest)
        sensitivity = build_contour_sensitivity(reference, basis, nominal_config())
        CACHE[key] = (reference, basis, specification, sensitivity)
    return CACHE[key]


def execute_job(job: Mapping[str, object]) -> Dict[str, object]:
    reference, basis, specification, sensitivity = cached_problem(job)
    settings = BenchmarkSettings(
        samples=161,
        duration=6.0,
        control_points=int(job["control_points"]),
        iterations=4,
        domain_seeds=(),
        number_of_windows=2,
        half_width=5,
    )
    base_plant = base_plant_from_job(job)
    summary, _ = run_matched_method(
        method=str(job["method"]),
        reference=reference,
        basis=basis,
        specification=specification,
        base_plant=base_plant,
        settings=settings,
        scenario=scenario_from_dict(job["scenario"]),
        noise_seed=int(job["noise_seed"]),
        compensation_gain=float(job["compensation_gain"]),
        smoothing_window=int(job["smoothing_window"]),
        learning_rate=float(job["learning_rate"]),
        nominal_sensitivity=sensitivity,
        return_trace=False,
    )
    manifest = job["manifest"]
    row = {
        "job_id": str(job["job_id"]),
        "experiment": str(job["experiment"]),
        "plant_group": str(job["plant_group"]),
        "plant_id": str(job["plant_id"]),
        "plant_seed": int(job.get("plant_seed", -1)),
        "manifest_id": str(manifest["manifest_id"]),
        "trajectory": str(manifest["trajectory_family"]),
        "regime": str(manifest["regime"]),
        "config_group": str(job.get("config_group", "method")),
        "config_value": str(job.get("config_value", job["method"])),
        "control_points": int(job["control_points"]),
    }
    row.update(summary)
    return row


def existing_job_ids(path: Path) -> set:
    if not path.is_file():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(str(json.loads(line)["job_id"]))
    return completed


def jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> List[Dict[str, object]]:
    rows = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: str(row["job_id"]))
    if not rows:
        raise RuntimeError("no rows were generated")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_jobs(jobs: Sequence[Dict[str, object]], directory: Path, workers: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "raw_results.jsonl"
    completed = existing_job_ids(checkpoint)
    pending = [job for job in jobs if str(job["job_id"]) not in completed]
    print(
        f"[{directory.name}] total={len(jobs)} completed={len(completed)} pending={len(pending)}",
        flush=True,
    )
    if pending:
        context = mp.get_context("fork")
        with checkpoint.open("a", encoding="utf-8") as handle:
            with context.Pool(processes=workers, maxtasksperchild=80) as pool:
                for index, row in enumerate(pool.imap_unordered(execute_job, pending, chunksize=1), 1):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    if index % 50 == 0 or index == len(pending):
                        print(
                            f"[{directory.name}] {index}/{len(pending)} new jobs complete",
                            flush=True,
                        )
    rows = jsonl_to_csv(checkpoint, directory / "raw_results.csv")
    if len(rows) != len(jobs):
        raise RuntimeError(f"row count mismatch: expected {len(jobs)}, got {len(rows)}")


def seeded_job(
    experiment: str,
    method: str,
    manifest: Mapping[str, object],
    scenario: StressScenario,
    plant_seed: int,
    job_suffix: str,
    **overrides,
) -> Dict[str, object]:
    scenario_index = {"n0_d0_m0": 0, "n0_d1_m0": 1, "n1_d1_m1": 2}[scenario.scenario_id]
    manifest_index = int(str(manifest["manifest_id"]).encode("utf-8").hex()[-4:], 16)
    base = {
        "job_id": f"{experiment}|{job_suffix}|{manifest['manifest_id']}|{scenario.scenario_id}|{plant_seed}|{method}",
        "experiment": experiment,
        "plant_kind": "seed",
        "plant_group": "independent_seeded",
        "plant_id": f"S{plant_seed}",
        "plant_seed": int(plant_seed),
        "manifest": dict(manifest),
        "scenario": scenario_dict(scenario),
        "method": method,
        "noise_seed": int(900000 + 10000 * scenario_index + manifest_index + plant_seed),
        "compensation_gain": 0.25,
        "smoothing_window": 5,
        "learning_rate": 0.65,
        "control_points": 12,
    }
    base.update(overrides)
    return base


def build_ablation_jobs(manifests: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    jobs = []
    seeds = tuple(range(24001, 24009))
    for scenario in selected_scenarios():
        for manifest in manifests:
            for seed in seeds:
                for method in MATCHED_METHODS:
                    jobs.append(
                        seeded_job(
                            "01_matched_ablation",
                            method,
                            manifest,
                            scenario,
                            seed,
                            job_suffix="matched",
                        )
                    )
    return jobs


def sensitivity_configurations() -> List[Dict[str, object]]:
    configurations = []
    for gain in (0.0, 0.10, 0.25, 0.40, 0.60, 1.00):
        configurations.append(
            {
                "config_id": "gamma_" + str(gain).replace(".", "p"),
                "config_group": "residual_delay_shrinkage",
                "config_value": gain,
                "compensation_gain": gain,
            }
        )
    for window in (3, 7, 9):
        configurations.append(
            {
                "config_id": f"smoothing_{window}",
                "config_group": "smoothing_window",
                "config_value": window,
                "smoothing_window": window,
            }
        )
    for points in (8, 16):
        configurations.append(
            {
                "config_id": f"control_points_{points}",
                "config_group": "nominal_control_points",
                "config_value": points,
                "control_points": points,
            }
        )
    for rate in (0.50, 0.80):
        configurations.append(
            {
                "config_id": "learning_rate_" + str(rate).replace(".", "p"),
                "config_group": "learning_rate",
                "config_value": rate,
                "learning_rate": rate,
            }
        )
    return configurations


def build_sensitivity_jobs(manifests: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    conflict = [item for item in manifests if item["regime"] == "demand_conflict"]
    scenarios = [item for item in selected_scenarios() if item.scenario_id != "n0_d0_m0"]
    jobs = []
    for scenario in scenarios:
        for manifest in conflict:
            for seed in range(25001, 25007):
                for config in sensitivity_configurations():
                    overrides = dict(config)
                    config_id = str(overrides.pop("config_id"))
                    jobs.append(
                        seeded_job(
                            "03_parameter_sensitivity",
                            "v11_full",
                            manifest,
                            scenario,
                            seed,
                            job_suffix=config_id,
                            **overrides,
                        )
                    )
    return jobs


def lhs_plants(count: int, seed: int, prefix: str) -> List[Dict[str, object]]:
    sampler = qmc.LatinHypercube(d=14, seed=seed, scramble=True)
    unit = sampler.random(n=count)
    ranges = np.asarray(
        [
            [15.0, 21.0], [0.60, 0.84], [1.0, 5.0], [1.0, 3.6], [2.0, 4.0],
            [10.5, 17.0], [0.58, 0.82], [2.0, 7.0], [1.5, 4.2], [1.8, 3.8],
            [650.0, 900.0], [80.0, 115.0], [0.01, 0.06], [0.02, 0.075],
        ],
        dtype=float,
    )
    values = qmc.scale(unit, ranges[:, 0], ranges[:, 1])
    plants = []
    for index, row in enumerate(values, 1):
        plant = VirtualPlantConfig(
            x_axis=AxisDynamics(row[0], row[1], int(np.floor(row[2])), row[3], row[4]),
            y_axis=AxisDynamics(row[5], row[6], int(np.floor(row[7])), row[8], row[9]),
            acceleration_limit=row[10],
            velocity_limit=row[11],
            cross_coupling=row[12],
            repeatable_disturbance=row[13],
        )
        plants.append(
            {
                "plant_id": f"{prefix}{index:02d}",
                "plant_group": "held_out_lhs",
                "plant_parameters": plant_to_dict(plant),
            }
        )
    return plants


def challenge_plants() -> List[Dict[str, object]]:
    base = {
        "x_natural_frequency": 18.0,
        "x_damping_ratio": 0.70,
        "x_delay_steps": 3,
        "x_friction": 2.3,
        "x_velocity_scale": 3.0,
        "y_natural_frequency": 13.5,
        "y_damping_ratio": 0.68,
        "y_delay_steps": 4,
        "y_friction": 2.8,
        "y_velocity_scale": 2.8,
        "acceleration_limit": 760.0,
        "velocity_limit": 95.0,
        "cross_coupling": 0.035,
        "repeatable_disturbance": 0.050,
    }
    variants = {
        "C01_max_delay": {"x_delay_steps": 5, "y_delay_steps": 7},
        "C02_low_bandwidth": {"x_natural_frequency": 14.0, "y_natural_frequency": 9.5},
        "C03_high_asymmetry": {"x_natural_frequency": 22.0, "y_natural_frequency": 9.0, "x_delay_steps": 1, "y_delay_steps": 7},
        "C04_strong_coupling": {"cross_coupling": 0.085},
        "C05_strong_friction": {"x_friction": 4.5, "y_friction": 5.2},
        "C06_tight_saturation": {"acceleration_limit": 560.0, "velocity_limit": 70.0},
    }
    output = []
    for plant_id, changes in variants.items():
        parameters = dict(base)
        parameters.update(changes)
        output.append(
            {
                "plant_id": plant_id,
                "plant_group": "edge_challenge",
                "plant_parameters": parameters,
            }
        )
    return output


def write_plant_table(plants: Sequence[Mapping[str, object]], path: Path) -> None:
    rows = []
    for item in plants:
        row = {"plant_id": item["plant_id"], "plant_group": item["plant_group"]}
        row.update(item["plant_parameters"])
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_plant_jobs(
    manifests: Sequence[Mapping[str, object]],
    plants: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    conflict = [item for item in manifests if item["regime"] == "demand_conflict"]
    methods = (
        "v11_full",
        "no_residual_alignment",
        "raw_top2",
        "uniform_full_trajectory",
    )
    jobs = []
    for scenario_index, scenario in enumerate(selected_scenarios()):
        for manifest_index, manifest in enumerate(conflict):
            for plant_index, plant in enumerate(plants):
                noise_seed = 980000 + 10000 * scenario_index + 100 * manifest_index + plant_index
                for method in methods:
                    jobs.append(
                        {
                            "job_id": f"05_virtual_plant_family|{plant['plant_id']}|{manifest['manifest_id']}|{scenario.scenario_id}|{method}",
                            "experiment": "05_virtual_plant_family",
                            "plant_kind": "explicit",
                            "plant_group": str(plant["plant_group"]),
                            "plant_id": str(plant["plant_id"]),
                            "plant_seed": -1,
                            "plant_parameters": dict(plant["plant_parameters"]),
                            "manifest": dict(manifest),
                            "scenario": scenario_dict(scenario),
                            "method": method,
                            "noise_seed": noise_seed,
                            "compensation_gain": 0.25,
                            "smoothing_window": 5,
                            "learning_rate": 0.65,
                            "control_points": 12,
                            "config_group": "method",
                            "config_value": method,
                        }
                    )
    return jobs


def write_protocol(
    ablation_jobs: Sequence[Mapping[str, object]],
    sensitivity_jobs: Sequence[Mapping[str, object]],
    plant_jobs: Sequence[Mapping[str, object]],
    plants: Sequence[Mapping[str, object]],
) -> None:
    protocol = {
        "protocol_id": "v11-additional-five-priority-numerical-experiments",
        "frozen_before_execution_on": "2026-08-20",
        "scope": "numerical virtual CNC machine-tool models only",
        "linuxcnc_executed": False,
        "physical_machine_executed": False,
        "original_v8_v13_results_modified": False,
        "settings": {
            "samples": 161,
            "duration_s": 6.0,
            "updates": 4,
            "nominal_control_points_per_axis": 12,
            "active_coefficients_per_axis": 10,
            "matched_active_zone_budget": 2,
            "default_residual_delay_gain": 0.25,
            "scenarios": [scenario_dict(item) for item in selected_scenarios()],
        },
        "priority_1": {
            "purpose": "matched module ablation",
            "methods": list(MATCHED_METHODS),
            "new_domain_seeds": list(range(24001, 24009)),
            "tasks": 15,
            "job_count": len(ablation_jobs),
        },
        "priority_2": {
            "purpose": "paired, domain and hierarchical bootstrap",
            "bootstrap_replicates": 20000,
            "bootstrap_seed": 20260820,
        },
        "priority_3": {
            "purpose": "one-factor-at-a-time parameter sensitivity",
            "development_seeds": list(range(25001, 25007)),
            "configurations": sensitivity_configurations(),
            "job_count": len(sensitivity_jobs),
        },
        "priority_4": {
            "purpose": "representative full-trace replay",
            "selection_rule": "delay+4 demand-conflict pair whose V11-vs-no-alignment improvement is closest to the median",
        },
        "priority_5": {
            "purpose": "held-out LHS plant family and edge challenges",
            "lhs_seed": 20260820,
            "held_out_lhs_plants": 24,
            "edge_challenge_plants": 6,
            "methods": ["v11_full", "no_residual_alignment", "raw_top2", "uniform_full_trajectory"],
            "job_count": len(plant_jobs),
        },
        "code_sha256": {
            "experiment_core.py": sha256(HERE / "experiment_core.py"),
            "run_experiments.py": sha256(HERE / "run_experiments.py"),
            "source_delay_runner.py": sha256(PROJECT / "src" / "cnc_task_ilc" / "delay_compensation_runner.py"),
            "source_semantic_runner.py": sha256(PROJECT / "src" / "cnc_task_ilc" / "semantic_task_benchmark.py"),
            "task_manifests.json": sha256(PROJECT / "data" / "tolerance_conflict_v1" / "task_manifests.json"),
        },
    }
    (ROOT / "protocol_pre_execution.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    workers = max(1, min(int(os.environ.get("V11_WORKERS", "4")), os.cpu_count() or 1))
    manifests = load_manifests()
    ablation_jobs = build_ablation_jobs(manifests)
    sensitivity_jobs = build_sensitivity_jobs(manifests)
    lhs = lhs_plants(24, 20260820, "P")
    challenge = challenge_plants()
    all_plants = lhs + challenge
    plant_jobs = build_plant_jobs(manifests, all_plants)
    write_protocol(ablation_jobs, sensitivity_jobs, plant_jobs, all_plants)
    plant_dir = ROOT / "results" / "05_virtual_plant_family"
    write_plant_table(all_plants, plant_dir / "plant_parameters.csv")

    run_jobs(ablation_jobs, ROOT / "results" / "01_matched_ablation", workers)
    run_jobs(sensitivity_jobs, ROOT / "results" / "03_parameter_sensitivity", workers)
    run_jobs(plant_jobs, plant_dir, workers)
    print("All numerical grids completed.", flush=True)


if __name__ == "__main__":
    main()

