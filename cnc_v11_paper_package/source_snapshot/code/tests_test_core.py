"""Core numerical tests for the feasibility demo."""

import json
import unittest
from pathlib import Path

import numpy as np

from cnc_task_ilc.basis import cubic_bspline_basis
from cnc_task_ilc.benchmark import (
    BenchmarkSettings,
    apply_reward_ranker,
    apply_safe_combo_selector,
    build_independent_evaluation_windows,
    build_method_windows,
    build_reward_calibration_dataset,
    build_safe_combo_calibration_dataset,
    calibrate_reward_ranker_leave_one_trajectory_out,
    calibrate_safe_combo_leave_one_trajectory_out,
    calibrate_selector_leave_one_trajectory_out,
)
from cnc_task_ilc.critical_windows import select_critical_windows
from cnc_task_ilc.ilc import ILCConfig, build_contour_sensitivity, run_ilc
from cnc_task_ilc.metrics import command_kinematics, task_errors
from cnc_task_ilc.plant import (
    mismatched_virtual_machine,
    nominal_config,
    simulate_machine,
)
from cnc_task_ilc.trajectory import (
    make_reference_trajectory,
    make_trajectory_family,
)
from cnc_task_ilc.task_benchmark import (
    make_task_specification,
    run_task_method,
)
from cnc_task_ilc.semantic_task_benchmark import (
    _dual_anchor_selection,
    _trust_limited_candidate,
    make_semantic_task_specification,
    run_semantic_task_method,
    zone_quality_ratios,
)
from cnc_task_ilc.conflict_taskset import (
    RANKED_TOLERANCES_MM,
    TASK_REGIMES,
    audit_conflict_taskset,
    build_conflict_taskset,
    specification_from_manifest,
)
from cnc_task_ilc.conflict_benchmark import (
    AUDIT_DOMAIN_SEEDS,
    EXPECTED_TASKSET_SHA256,
    FORMAL_DOMAIN_SEEDS,
    preregistered_protocol,
    taskset_sha256,
    validate_frozen_taskset,
)
from cnc_task_ilc.dual_anchor_confirmation import (
    FORMAL_DOMAIN_SEEDS as V8_FORMAL_DOMAIN_SEEDS,
    FORMAL_METHODS as V8_FORMAL_METHODS,
    validate_frozen_protocol as validate_v8_frozen_protocol,
)
from cnc_task_ilc.dual_anchor_development import DEVELOPMENT_DOMAIN_SEEDS
from cnc_task_ilc.robustness_runner import (
    STRESS_SCENARIOS,
    make_stressed_virtual_machine,
    run_robustness_method,
)
from cnc_task_ilc.robustness_benchmark import (
    AUDIT_DOMAIN_SEEDS as V9_AUDIT_DOMAIN_SEEDS,
    FORMAL_DOMAIN_SEEDS as V9_FORMAL_DOMAIN_SEEDS,
    validate_robustness_protocol,
)
from cnc_task_ilc.factorial_benchmark import (
    AUDIT_DOMAIN_SEEDS as V10_AUDIT_DOMAIN_SEEDS,
    FACTORIAL_SCENARIOS,
    FORMAL_DOMAIN_SEEDS as V10_FORMAL_DOMAIN_SEEDS,
    factor_codes,
    validate_factorial_protocol,
)
from cnc_task_ilc.delay_compensation_runner import (
    axis_delay_aligned_sensitivity,
    estimate_effective_delay_steps,
    run_delay_compensated_method,
)
from cnc_task_ilc.delay_compensation_benchmark import (
    DELAY_2_SCENARIO,
    DEVELOPMENT_DOMAIN_SEEDS as V11_DEVELOPMENT_DOMAIN_SEEDS,
    FORMAL_DOMAIN_SEEDS as V11_FORMAL_DOMAIN_SEEDS,
    SELECTED_COMPENSATION_GAIN,
    validate_delay_protocol,
)
from cnc_task_ilc.delay_generalization_runner import (
    UNKNOWN_DRIFT,
    UNKNOWN_STATIC,
    make_extra_delay_schedule,
    run_delay_generalization_method,
)
from cnc_task_ilc.delay_generalization_benchmark import (
    DEVELOPMENT_DOMAIN_SEEDS as V12_DEVELOPMENT_DOMAIN_SEEDS,
    FORMAL_DOMAIN_SEEDS as V12_FORMAL_DOMAIN_SEEDS,
)
from cnc_task_ilc.adaptive_delay_runner import (
    BALANCED_DELAY_SCENARIOS,
    MAX_ADAPTIVE_GAIN,
    MAX_APPLIED_LAG_STEPS,
    MIN_ADAPTIVE_GAIN,
    adaptive_applied_lag,
    balanced_axis_delay_schedule,
    run_adaptive_delay_method,
)
from cnc_task_ilc.adaptive_delay_benchmark import (
    DEVELOPMENT_DOMAIN_SEEDS as V13_DEVELOPMENT_DOMAIN_SEEDS,
    FORMAL_DOMAIN_SEEDS as V13_FORMAL_DOMAIN_SEEDS,
)


class CoreDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = make_reference_trajectory(samples=161, duration=4.0)

    def test_reference_geometry_is_finite_and_unit_length(self) -> None:
        self.assertTrue(np.all(np.isfinite(self.reference.position)))
        tangent_norm = np.linalg.norm(self.reference.tangent, axis=1)
        normal_norm = np.linalg.norm(self.reference.normal, axis=1)
        self.assertTrue(np.allclose(tangent_norm, 1.0, atol=1.0e-10))
        self.assertTrue(np.allclose(normal_norm, 1.0, atol=1.0e-10))
        orthogonality = np.sum(
            self.reference.tangent * self.reference.normal,
            axis=1,
        )
        self.assertTrue(np.allclose(orthogonality, 0.0, atol=1.0e-10))

    def test_virtual_machine_creates_stable_nonzero_error(self) -> None:
        feedback = simulate_machine(
            self.reference.position,
            self.reference.dt,
            mismatched_virtual_machine(),
        )
        error = feedback - self.reference.position
        self.assertTrue(np.all(np.isfinite(feedback)))
        self.assertGreater(float(np.sqrt(np.mean(error**2))), 1.0e-3)
        self.assertLess(float(np.max(np.abs(feedback))), 100.0)

    def test_clamped_spline_correction_is_zero_at_ends(self) -> None:
        basis = cubic_bspline_basis(
            samples=self.reference.time.size,
            control_points=12,
        )
        self.assertTrue(np.allclose(basis[0], 0.0))
        self.assertTrue(np.allclose(basis[-1], 0.0))
        self.assertGreater(basis.shape[1], 4)

    def test_critical_window_selection_is_sparse(self) -> None:
        feedback = simulate_machine(
            self.reference.position,
            self.reference.dt,
            mismatched_virtual_machine(),
        )
        contour = task_errors(self.reference, feedback)["contour"]
        selection = select_critical_windows(
            self.reference,
            contour,
            number_of_windows=2,
            half_width=8,
        )
        self.assertEqual(len(selection.centers), 2)
        self.assertGreater(int(np.sum(selection.mask)), 0)
        self.assertLess(float(np.mean(selection.mask)), 0.5)

    def test_nominal_sensitivity_has_expected_shape_and_rank(self) -> None:
        basis = cubic_bspline_basis(
            samples=self.reference.time.size,
            control_points=12,
        )
        sensitivity = build_contour_sensitivity(
            self.reference,
            basis,
            nominal_config(),
        )
        self.assertEqual(
            sensitivity.shape,
            (self.reference.time.size, 2 * basis.shape[1]),
        )
        self.assertTrue(np.all(np.isfinite(sensitivity)))
        self.assertGreater(np.linalg.matrix_rank(sensitivity), 4)

    def test_default_configuration_is_valid(self) -> None:
        config = ILCConfig()
        self.assertGreater(config.iterations, 0)
        self.assertGreater(config.correction_limit, 0.0)
        self.assertGreater(config.learning_rate, 0.0)
        self.assertLessEqual(config.learning_rate, 1.0)

    def test_all_benchmark_trajectory_families_are_valid(self) -> None:
        for family in (
            "harmonic_loop",
            "ellipse",
            "rounded_square",
            "figure_eight",
            "s_curve",
        ):
            reference = make_trajectory_family(
                family,
                samples=101,
                duration=4.0,
            )
            self.assertEqual(reference.position.shape, (101, 2))
            self.assertTrue(np.all(np.isfinite(reference.curvature)))

    def test_evaluation_and_method_windows_have_equal_budgets(self) -> None:
        settings = BenchmarkSettings(
            samples=self.reference.time.size,
            duration=4.0,
            control_points=10,
            iterations=1,
            domain_seeds=(11,),
            calibration_seeds=(1001, 1003),
            number_of_windows=2,
            half_width=6,
        )
        evaluation = build_independent_evaluation_windows(
            self.reference,
            settings,
        )
        feedback = simulate_machine(
            self.reference.position,
            self.reference.dt,
            mismatched_virtual_machine(),
        )
        contour = task_errors(self.reference, feedback)["contour"]
        windows = build_method_windows(
            self.reference,
            contour,
            settings,
            random_seed=123,
        )
        budget = int(np.sum(evaluation.mask))
        self.assertGreater(budget, 0)
        for selection in windows.values():
            self.assertEqual(int(np.sum(selection.mask)), budget)

    def test_learned_selector_is_frozen_and_preserves_window_budget(self) -> None:
        settings = BenchmarkSettings(
            samples=81,
            duration=4.0,
            control_points=8,
            iterations=1,
            domain_seeds=(11,),
            calibration_seeds=(1001,),
            number_of_windows=2,
            half_width=4,
        )
        held_out_family = "ellipse"
        selector = calibrate_selector_leave_one_trajectory_out(
            held_out_family,
            settings,
        )
        self.assertEqual(selector["held_out_family"], held_out_family)
        weights = np.asarray(selector["weights"], dtype=float)
        self.assertEqual(weights.size, 5)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights >= 0.0))

        reference = make_trajectory_family(
            held_out_family,
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(
            samples=settings.samples,
            control_points=settings.control_points,
        )
        sensitivity = build_contour_sensitivity(
            reference,
            basis,
            nominal_config(),
        )
        feedback = simulate_machine(
            reference.position,
            reference.dt,
            mismatched_virtual_machine(),
        )
        contour = task_errors(reference, feedback)["contour"]
        windows = build_method_windows(
            reference,
            contour,
            settings,
            random_seed=123,
            nominal_control_sensitivity=np.linalg.norm(
                sensitivity,
                axis=1,
            ),
            calibrated_selector=selector,
        )
        self.assertIn("learned_automatic", windows)
        evaluation = build_independent_evaluation_windows(
            reference,
            settings,
        )
        self.assertEqual(
            int(np.sum(windows["learned_automatic"].mask)),
            int(np.sum(evaluation.mask)),
        )

    def test_reward_ranker_uses_only_other_trajectory_families(self) -> None:
        settings = BenchmarkSettings(
            samples=51,
            duration=3.0,
            control_points=7,
            iterations=1,
            domain_seeds=(11,),
            calibration_seeds=(1001,),
            number_of_windows=2,
            half_width=2,
        )
        dataset = build_reward_calibration_dataset(settings)
        held_out_family = "ellipse"
        ranker = calibrate_reward_ranker_leave_one_trajectory_out(
            held_out_family,
            settings,
            dataset,
        )
        self.assertEqual(ranker["held_out_family"], held_out_family)
        self.assertEqual(ranker["training_groups"], 4)
        self.assertGreater(ranker["training_candidates"], 0)
        self.assertIn(ranker["reward_blend"], (0.0, 0.25, 0.5, 0.75, 1.0))

        reference = make_trajectory_family(
            held_out_family,
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(
            samples=settings.samples,
            control_points=settings.control_points,
        )
        sensitivity = build_contour_sensitivity(
            reference,
            basis,
            nominal_config(),
        )
        feedback = simulate_machine(
            reference.position,
            reference.dt,
            mismatched_virtual_machine(),
        )
        contour = task_errors(reference, feedback)["contour"]
        selection = apply_reward_ranker(
            reference,
            contour,
            sensitivity,
            ranker,
            settings,
        )
        self.assertEqual(len(selection.centers), settings.number_of_windows)
        self.assertEqual(
            int(np.sum(selection.mask)),
            settings.number_of_windows * (2 * settings.half_width + 1),
        )

    def test_safe_combo_keeps_two_anchors_and_uses_an_ensemble(self) -> None:
        settings = BenchmarkSettings(
            samples=51,
            duration=3.0,
            control_points=7,
            iterations=1,
            domain_seeds=(11,),
            calibration_seeds=(1001,),
            number_of_windows=3,
            half_width=2,
        )
        dataset = build_safe_combo_calibration_dataset(settings)
        held_out_family = "ellipse"
        model = calibrate_safe_combo_leave_one_trajectory_out(
            held_out_family,
            settings,
            dataset,
        )
        self.assertEqual(model["held_out_family"], held_out_family)
        self.assertEqual(model["anchor_windows"], 2)
        self.assertEqual(model["exploration_windows"], 1)
        self.assertEqual(model["training_groups"], 4)
        self.assertEqual(len(model["ensemble"]), 4)
        self.assertLessEqual(
            model["cross_validated_degradation_rate"],
            0.10,
        )

        reference = make_trajectory_family(
            held_out_family,
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(
            samples=settings.samples,
            control_points=settings.control_points,
        )
        sensitivity = build_contour_sensitivity(
            reference,
            basis,
            nominal_config(),
        )
        feedback = simulate_machine(
            reference.position,
            reference.dt,
            mismatched_virtual_machine(),
        )
        contour = task_errors(reference, feedback)["contour"]
        selection = apply_safe_combo_selector(
            reference,
            contour,
            sensitivity,
            model,
            settings,
        )
        self.assertEqual(len(selection.centers), 3)
        self.assertEqual(
            int(np.sum(selection.mask)),
            3 * (2 * settings.half_width + 1),
        )

    def test_external_task_zones_do_not_depend_on_machine_error(self) -> None:
        specification = make_task_specification(
            self.reference,
            half_width=5,
        )
        self.assertEqual(len(specification.centers), 6)
        self.assertEqual(len(specification.tolerances), 6)
        self.assertEqual(
            int(np.sum(specification.evaluation_mask)),
            6 * 11,
        )
        self.assertTrue(
            np.all(
                np.isfinite(
                    specification.tolerance_profile[
                        specification.evaluation_mask
                    ]
                )
            )
        )

    def test_nominal_lookahead_reduces_external_task_error(self) -> None:
        settings = BenchmarkSettings(
            samples=81,
            duration=4.0,
            control_points=8,
            iterations=1,
            domain_seeds=(347,),
            calibration_seeds=(1001,),
            number_of_windows=3,
            half_width=3,
        )
        reference = make_trajectory_family(
            "ellipse",
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(
            samples=settings.samples,
            control_points=settings.control_points,
        )
        specification = make_task_specification(
            reference,
            settings.half_width,
        )
        result = run_task_method(
            "nominal_lookahead_dynamic",
            reference,
            basis,
            specification,
            plant_seed=347,
            settings=settings,
            random_seed=123,
        )
        self.assertEqual(result["all_updates_succeeded"], 1)
        self.assertEqual(result["constraint_violation"], 0)
        self.assertLess(
            result["final_task_nrmse"],
            result["initial_task_nrmse"],
        )

    def test_semantic_task_zones_are_program_defined_and_family_specific(self) -> None:
        specifications = {}
        for family in ("ellipse", "rounded_square", "s_curve"):
            reference = make_trajectory_family(
                family,
                samples=161,
                duration=4.0,
            )
            specification = make_semantic_task_specification(
                reference,
                family,
                half_width=5,
            )
            specifications[family] = specification
            self.assertEqual(len(specification.centers), 6)
            self.assertEqual(len(set(specification.roles)), 6)
            self.assertTrue(np.all(np.asarray(specification.tolerances) > 0.0))
            self.assertEqual(int(np.sum(specification.evaluation_mask)), 66)
            for first, second in zip(
                specification.centers[:-1],
                specification.centers[1:],
            ):
                self.assertGreater(second - first, 10)
        self.assertNotEqual(
            specifications["ellipse"].centers,
            specifications["rounded_square"].centers,
        )

    def test_semantic_violation_scheduler_reduces_task_state(self) -> None:
        settings = BenchmarkSettings(
            samples=81,
            duration=4.0,
            control_points=8,
            iterations=1,
            domain_seeds=(607,),
            calibration_seeds=(1001,),
            number_of_windows=2,
            half_width=3,
        )
        reference = make_trajectory_family(
            "rounded_square",
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(
            samples=settings.samples,
            control_points=settings.control_points,
        )
        specification = make_semantic_task_specification(
            reference,
            "rounded_square",
            settings.half_width,
        )
        feedback = simulate_machine(
            reference.position,
            reference.dt,
            mismatched_virtual_machine(),
        )
        contour = task_errors(reference, feedback)["contour"]
        ratios = zone_quality_ratios(contour, specification)
        self.assertEqual(ratios.shape, (6,))
        self.assertTrue(np.all(np.isfinite(ratios)))
        result = run_semantic_task_method(
            "violation_dynamic",
            reference,
            basis,
            specification,
            plant_seed=607,
            settings=settings,
            random_seed=123,
        )
        self.assertEqual(result["all_updates_succeeded"], 1)
        self.assertEqual(result["constraint_violation"], 0)
        self.assertLess(result["final_task_score"], result["initial_task_score"])

    def test_dual_anchor_scheduler_reserves_two_distinct_objectives(self) -> None:
        reference = make_trajectory_family(
            "rounded_square",
            samples=161,
            duration=4.0,
        )
        specification = make_semantic_task_specification(
            reference,
            "rounded_square",
            half_width=5,
        )
        feedback = simulate_machine(
            reference.position,
            reference.dt,
            mismatched_virtual_machine(),
        )
        contour = task_errors(reference, feedback)["contour"]
        selection = _dual_anchor_selection(contour, specification, budget=2)

        ratios = zone_quality_ratios(contour, specification)
        urgency = ratios + 0.50 * np.maximum(ratios - 1.0, 0.0)
        task_anchor = int(np.argmax(urgency))
        raw_peak = np.asarray(
            [
                np.max(np.abs(contour[start : stop + 1]))
                for start, stop in specification.windows
            ]
        )
        expected_error_anchor = next(
            int(index)
            for index in np.argsort(raw_peak)[::-1]
            if int(index) != task_anchor
        )
        self.assertEqual(selection, tuple(sorted((task_anchor, expected_error_anchor))))
        self.assertEqual(len(set(selection)), 2)

    def test_dual_anchor_scheduler_reduces_task_state_safely(self) -> None:
        settings = BenchmarkSettings(
            samples=81,
            duration=4.0,
            control_points=8,
            iterations=1,
            domain_seeds=(607,),
            calibration_seeds=(1001,),
            number_of_windows=2,
            half_width=3,
        )
        reference = make_trajectory_family(
            "rounded_square",
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(
            samples=settings.samples,
            control_points=settings.control_points,
        )
        specification = make_semantic_task_specification(
            reference,
            "rounded_square",
            settings.half_width,
        )
        result = run_semantic_task_method(
            "dual_anchor_dynamic",
            reference,
            basis,
            specification,
            plant_seed=607,
            settings=settings,
            random_seed=123,
        )
        self.assertEqual(result["all_updates_succeeded"], 1)
        self.assertEqual(result["constraint_violation"], 0)
        self.assertLess(result["final_task_score"], result["initial_task_score"])

    def test_output_space_trust_limit_caps_candidate_step(self) -> None:
        basis = cubic_bspline_basis(samples=81, control_points=8)
        command = np.zeros((81, 2), dtype=float)
        delta = np.ones(2 * basis.shape[1], dtype=float)
        candidate, maximum_step = _trust_limited_candidate(
            command,
            basis,
            delta,
            trust_radius=0.10,
        )
        self.assertLessEqual(maximum_step, 0.1000001)
        self.assertLessEqual(float(np.max(np.abs(candidate))), 0.1000001)

    def test_conflict_taskset_is_feedback_independent_and_balanced(self) -> None:
        taskset = build_conflict_taskset()
        self.assertEqual(taskset["manifest_count"], 15)
        self.assertEqual(tuple(taskset["regimes"]), TASK_REGIMES)
        self.assertFalse(
            taskset["construction_guarantees"]["machine_feedback_used"]
        )
        manifests = {
            (item["trajectory_family"], item["regime"]): item
            for item in taskset["manifests"]
        }
        for family in ("ellipse", "rounded_square", "s_curve"):
            neutral = manifests[(family, "neutral")]
            aligned = manifests[(family, "demand_aligned")]
            conflict = manifests[(family, "demand_conflict")]
            for manifest in (neutral, aligned, conflict):
                self.assertFalse(
                    manifest["construction"]["uses_machine_feedback"]
                )
                self.assertFalse(
                    manifest["construction"]["uses_measured_tracking_error"]
                )
                self.assertEqual(len(manifest["zones"]), 6)
            self.assertEqual(
                [zone["center_phase"] for zone in neutral["zones"]],
                [zone["center_phase"] for zone in conflict["zones"]],
            )
            self.assertEqual(
                len({zone["tolerance_mm"] for zone in neutral["zones"]}),
                1,
            )
            aligned_by_rank = sorted(
                aligned["zones"], key=lambda zone: zone["program_demand_rank"]
            )
            conflict_by_rank = sorted(
                conflict["zones"], key=lambda zone: zone["program_demand_rank"]
            )
            self.assertEqual(
                tuple(zone["tolerance_mm"] for zone in aligned_by_rank),
                RANKED_TOLERANCES_MM,
            )
            self.assertEqual(
                tuple(zone["tolerance_mm"] for zone in conflict_by_rank),
                RANKED_TOLERANCES_MM[::-1],
            )

    def test_conflict_manifest_resolves_at_new_sampling_resolution(self) -> None:
        taskset = build_conflict_taskset()
        manifest = next(
            item
            for item in taskset["manifests"]
            if item["manifest_id"] == "figure_eight--demand_conflict"
        )
        reference = make_trajectory_family(
            "figure_eight",
            samples=321,
            duration=6.0,
        )
        specification = specification_from_manifest(reference, manifest)
        self.assertEqual(len(specification.centers), 6)
        self.assertEqual(len(specification.tolerances), 6)
        self.assertTrue(np.all(np.diff(specification.centers) > 20))

    def test_conflict_taskset_audit_separates_neutral_and_conflict(self) -> None:
        taskset = build_conflict_taskset()
        _, summary = audit_conflict_taskset(taskset, audit_seeds=(733,))
        by_regime = {item["regime"]: item for item in summary}
        self.assertEqual(by_regime["neutral"]["identical_top2_rate"], 1.0)
        self.assertGreaterEqual(
            by_regime["demand_conflict"]["selection_disagreement_rate"],
            0.60,
        )

    def test_v7_taskset_hash_and_formal_protocol_are_frozen(self) -> None:
        taskset_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "tolerance_conflict_v1"
            / "task_manifests.json"
        )
        self.assertEqual(taskset_sha256(taskset_path), EXPECTED_TASKSET_SHA256)
        taskset = validate_frozen_taskset(taskset_path)
        self.assertEqual(taskset["manifest_count"], 15)
        self.assertFalse(set(FORMAL_DOMAIN_SEEDS) & set(AUDIT_DOMAIN_SEEDS))
        protocol = preregistered_protocol()
        self.assertEqual(protocol["primary_scope"], "demand_conflict")
        self.assertEqual(protocol["primary_comparator"], "error_peak_dynamic")

    def test_v8_protocol_is_frozen_and_all_seed_groups_are_isolated(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        taskset_path = (
            project_root
            / "data"
            / "tolerance_conflict_v1"
            / "task_manifests.json"
        )
        development_path = (
            project_root
            / "results"
            / "development_v8_dual_anchor"
            / "dual_anchor_development_summary.json"
        )
        protocol_path = (
            project_root
            / "data"
            / "tolerance_conflict_v1"
            / "v8_preregistered_protocol.json"
        )
        protocol = validate_v8_frozen_protocol(
            taskset_path,
            development_path,
            protocol_path,
        )
        seed_groups = (
            set(AUDIT_DOMAIN_SEEDS),
            set(FORMAL_DOMAIN_SEEDS),
            set(DEVELOPMENT_DOMAIN_SEEDS),
            set(V8_FORMAL_DOMAIN_SEEDS),
        )
        for first_index, first in enumerate(seed_groups):
            for second in seed_groups[first_index + 1 :]:
                self.assertFalse(first & second)
        self.assertEqual(protocol["primary_method"], "dual_anchor_dynamic")
        self.assertIn("error_peak_dynamic", V8_FORMAL_METHODS)
        self.assertEqual(protocol["criteria"][
            "conflict_vs_error_peak_strict_win_rate_at_least"
        ], 0.60)

    def test_robustness_scenarios_change_only_declared_stress_factors(self) -> None:
        baseline = make_stressed_virtual_machine(1231, STRESS_SCENARIOS[0])
        noise = make_stressed_virtual_machine(1231, STRESS_SCENARIOS[2])
        delay = make_stressed_virtual_machine(1231, STRESS_SCENARIOS[4])
        mismatch = make_stressed_virtual_machine(1231, STRESS_SCENARIOS[6])
        self.assertEqual(baseline, noise)
        self.assertEqual(delay.x_axis.delay_steps, baseline.x_axis.delay_steps + 4)
        self.assertEqual(delay.y_axis.delay_steps, baseline.y_axis.delay_steps + 4)
        self.assertEqual(delay.cross_coupling, baseline.cross_coupling)
        self.assertGreater(mismatch.cross_coupling, baseline.cross_coupling)
        self.assertEqual(mismatch.x_axis.delay_steps, baseline.x_axis.delay_steps)
        self.assertEqual(mismatch.y_axis.delay_steps, baseline.y_axis.delay_steps)

    def test_noisy_robustness_runner_scores_true_error_and_stays_safe(self) -> None:
        settings = BenchmarkSettings(
            samples=81,
            duration=4.0,
            control_points=8,
            iterations=1,
            domain_seeds=(1231,),
            number_of_windows=2,
            half_width=3,
        )
        reference = make_trajectory_family(
            "rounded_square",
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(
            samples=settings.samples,
            control_points=settings.control_points,
        )
        specification = make_semantic_task_specification(
            reference,
            "rounded_square",
            settings.half_width,
        )
        result = run_robustness_method(
            "dual_anchor_dynamic",
            reference,
            basis,
            specification,
            plant_seed=1231,
            settings=settings,
            scenario=STRESS_SCENARIOS[2],
            noise_seed=44001,
        )
        self.assertEqual(result["measurement_noise_std_mm"], 0.05)
        self.assertEqual(result["finite_result"], 1)
        self.assertEqual(result["all_updates_succeeded"], 1)
        self.assertEqual(result["constraint_violation"], 0)
        self.assertTrue(np.isfinite(result["task_auc_normalized"]))
        self.assertTrue(np.isfinite(result["measured_score_auc_normalized"]))

    def test_v9_robustness_protocol_is_frozen_and_seed_isolated(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        protocol = validate_robustness_protocol(
            project_root
            / "data"
            / "tolerance_conflict_v1"
            / "task_manifests.json",
            project_root
            / "results"
            / "development_v9_robustness"
            / "robustness_audit_summary.json",
            project_root
            / "data"
            / "tolerance_conflict_v1"
            / "v8_preregistered_protocol.json",
            project_root
            / "data"
            / "tolerance_conflict_v1"
            / "v9_robustness_protocol.json",
        )
        prior = (
            set(AUDIT_DOMAIN_SEEDS)
            | set(FORMAL_DOMAIN_SEEDS)
            | set(DEVELOPMENT_DOMAIN_SEEDS)
            | set(V8_FORMAL_DOMAIN_SEEDS)
        )
        self.assertFalse(prior & set(V9_AUDIT_DOMAIN_SEEDS))
        self.assertFalse(prior & set(V9_FORMAL_DOMAIN_SEEDS))
        self.assertFalse(
            set(V9_AUDIT_DOMAIN_SEEDS) & set(V9_FORMAL_DOMAIN_SEEDS)
        )
        self.assertEqual(protocol["formal_method_runs"], 1680)
        self.assertTrue(protocol["true_error_used_for_evaluation"])
        self.assertEqual(len(protocol["scenarios"]), 7)

    def test_v10_factorial_grid_contains_every_binary_combination(self) -> None:
        codes = {factor_codes(item.scenario_id) for item in FACTORIAL_SCENARIOS}
        expected = {
            (noise, delay, mismatch)
            for noise in (0, 1)
            for delay in (0, 1)
            for mismatch in (0, 1)
        }
        self.assertEqual(codes, expected)
        self.assertEqual(len(FACTORIAL_SCENARIOS), 8)
        extreme = next(
            item
            for item in FACTORIAL_SCENARIOS
            if item.scenario_id == "n1_d1_m1"
        )
        self.assertEqual(extreme.measurement_noise_std_mm, 0.05)
        self.assertEqual(extreme.extra_delay_steps, 4)
        self.assertEqual(extreme.mismatch_scale, 1.70)

    def test_v10_factorial_protocol_is_frozen_and_seed_isolated(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        protocol = validate_factorial_protocol(
            project_root
            / "data"
            / "tolerance_conflict_v1"
            / "task_manifests.json",
            project_root
            / "results"
            / "development_v10_factorial"
            / "factorial_audit_summary.json",
            project_root
            / "data"
            / "tolerance_conflict_v1"
            / "v9_robustness_protocol.json",
            project_root
            / "data"
            / "tolerance_conflict_v1"
            / "v10_factorial_protocol.json",
        )
        prior = (
            set(AUDIT_DOMAIN_SEEDS)
            | set(FORMAL_DOMAIN_SEEDS)
            | set(DEVELOPMENT_DOMAIN_SEEDS)
            | set(V8_FORMAL_DOMAIN_SEEDS)
            | set(V9_AUDIT_DOMAIN_SEEDS)
            | set(V9_FORMAL_DOMAIN_SEEDS)
        )
        self.assertFalse(prior & set(V10_AUDIT_DOMAIN_SEEDS))
        self.assertFalse(prior & set(V10_FORMAL_DOMAIN_SEEDS))
        self.assertFalse(
            set(V10_AUDIT_DOMAIN_SEEDS) & set(V10_FORMAL_DOMAIN_SEEDS)
        )
        self.assertEqual(protocol["formal_method_runs"], 1920)
        self.assertEqual(protocol["extreme_scenario_id"], "n1_d1_m1")
        self.assertEqual(len(protocol["scenarios"]), 8)

    def test_delay_estimator_recovers_known_axis_delays(self) -> None:
        time = np.linspace(0.0, 8.0, 801)
        command = np.column_stack(
            (
                np.sin(1.7 * time) + 0.2 * np.sin(4.3 * time),
                np.cos(1.1 * time) + 0.15 * np.sin(3.7 * time),
            )
        )
        feedback = np.empty_like(command)
        feedback[:3, 0] = command[0, 0]
        feedback[3:, 0] = command[:-3, 0]
        feedback[:6, 1] = command[0, 1]
        feedback[6:, 1] = command[:-6, 1]
        estimate = estimate_effective_delay_steps(command, feedback)
        self.assertEqual(estimate["axis_lag_steps"], [3, 6])
        self.assertGreater(estimate["peak_correlation"], 0.98)

    def test_fractional_axis_delay_alignment_interpolates_blocks(self) -> None:
        sensitivity = np.arange(32, dtype=float).reshape(8, 4)
        aligned = axis_delay_aligned_sensitivity(sensitivity, (1.5, 0.5))
        self.assertTrue(np.allclose(aligned[:2, :2], 0.0))
        self.assertTrue(
            np.allclose(aligned[2, :2], 0.5 * sensitivity[0, :2] + 0.5 * sensitivity[1, :2])
        )
        self.assertTrue(
            np.allclose(aligned[1, 2:], 0.5 * sensitivity[0, 2:] + 0.5 * sensitivity[1, 2:])
        )

    def test_short_delay_aware_run_is_finite_and_safe(self) -> None:
        settings = BenchmarkSettings(
            samples=81,
            duration=4.0,
            control_points=8,
            iterations=1,
            domain_seeds=(1601,),
            number_of_windows=2,
            half_width=3,
        )
        reference = make_trajectory_family(
            "rounded_square",
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(settings.samples, settings.control_points)
        specification = make_semantic_task_specification(
            reference,
            "rounded_square",
            settings.half_width,
        )
        result = run_delay_compensated_method(
            "delay_aware_dual_anchor",
            reference,
            basis,
            specification,
            1601,
            settings,
            DELAY_2_SCENARIO,
            88101,
            compensation_gain=SELECTED_COMPENSATION_GAIN,
        )
        self.assertEqual(result["finite_result"], 1)
        self.assertEqual(result["all_updates_succeeded"], 1)
        self.assertEqual(result["constraint_violation"], 0)

    def test_v11_delay_protocol_is_frozen_and_seed_isolated(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        protocol = validate_delay_protocol(
            project_root
            / "data"
            / "tolerance_conflict_v1"
            / "task_manifests.json",
            project_root
            / "results"
            / "development_v11_delay_compensation"
            / "delay_compensation_development_summary.json",
            project_root
            / "data"
            / "tolerance_conflict_v1"
            / "v10_factorial_protocol.json",
            project_root
            / "data"
            / "tolerance_conflict_v1"
            / "v11_delay_protocol.json",
        )
        prior = set(V10_AUDIT_DOMAIN_SEEDS) | set(V10_FORMAL_DOMAIN_SEEDS)
        self.assertFalse(prior & set(V11_DEVELOPMENT_DOMAIN_SEEDS))
        self.assertFalse(prior & set(V11_FORMAL_DOMAIN_SEEDS))
        self.assertFalse(
            set(V11_DEVELOPMENT_DOMAIN_SEEDS) & set(V11_FORMAL_DOMAIN_SEEDS)
        )
        self.assertEqual(protocol["selected_compensation_gain"], 0.25)
        self.assertEqual(protocol["formal_method_runs"], 1200)

    def test_v12_delay_schedules_are_paired_bounded_and_drifting(self) -> None:
        static = make_extra_delay_schedule(1709, UNKNOWN_STATIC, 5)
        drift = make_extra_delay_schedule(1709, UNKNOWN_DRIFT, 5)
        self.assertEqual(static.shape, (5, 2))
        self.assertTrue(np.all(static == static[0]))
        self.assertTrue(np.all((static >= 0) & (static <= 8)))
        self.assertTrue(np.all((drift >= 0) & (drift <= 8)))
        self.assertFalse(np.all(drift == drift[0]))
        self.assertTrue(
            np.array_equal(
                drift,
                make_extra_delay_schedule(1709, UNKNOWN_DRIFT, 5),
            )
        )

    def test_short_v12_unknown_delay_run_is_finite_and_safe(self) -> None:
        settings = BenchmarkSettings(
            samples=81,
            duration=4.0,
            control_points=8,
            iterations=1,
            domain_seeds=(1709,),
            number_of_windows=2,
            half_width=3,
        )
        reference = make_trajectory_family(
            "rounded_square",
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(settings.samples, settings.control_points)
        specification = make_semantic_task_specification(
            reference,
            "rounded_square",
            settings.half_width,
        )
        result = run_delay_generalization_method(
            "delay_aware_dual_anchor",
            reference,
            basis,
            specification,
            1709,
            settings,
            UNKNOWN_STATIC,
        )
        self.assertEqual(result["finite_result"], 1)
        self.assertEqual(result["all_updates_succeeded"], 1)
        self.assertEqual(result["constraint_violation"], 0)

    def test_v12_stopped_before_formal_seed_use(self) -> None:
        prior = set(V11_DEVELOPMENT_DOMAIN_SEEDS) | set(V11_FORMAL_DOMAIN_SEEDS)
        self.assertFalse(prior & set(V12_DEVELOPMENT_DOMAIN_SEEDS))
        self.assertFalse(prior & set(V12_FORMAL_DOMAIN_SEEDS))
        self.assertFalse(
            set(V12_DEVELOPMENT_DOMAIN_SEEDS) & set(V12_FORMAL_DOMAIN_SEEDS)
        )
        summary_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "development_v12_delay_generalization"
            / "comparison_v12_development.json"
        )
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertFalse(payload["formal_domain_seeds_used"])
        self.assertEqual(
            payload["classification"],
            "DEVELOPMENT_GATE_FAILED_NO_FORMAL_RUN",
        )

    def test_v13_balanced_schedules_are_exact_and_bounded(self) -> None:
        for scenario in BALANCED_DELAY_SCENARIOS:
            schedules = [
                balanced_axis_delay_schedule(slot, scenario, 5)
                for slot in range(4)
            ]
            for schedule in schedules:
                self.assertEqual(schedule.shape, (5, 2))
                self.assertTrue(np.all((schedule >= 2) & (schedule <= 10)))
            if scenario.mode == "static":
                self.assertTrue(
                    all(np.all(schedule == schedule[0]) for schedule in schedules)
                )
            else:
                self.assertTrue(
                    all(
                        not np.all(schedule == schedule[0])
                        for schedule in schedules
                    )
                )

    def test_v13_adaptive_lag_is_rolling_confident_and_bounded(self) -> None:
        applied, confidence, gain = adaptive_applied_lag(
            [(4, 8), (6, 6)],
            (0.999, 0.995),
            (0.0010, 0.0007),
        )
        self.assertTrue(all(0.0 <= item <= 1.0 for item in confidence))
        self.assertTrue(
            all(MIN_ADAPTIVE_GAIN <= item <= MAX_ADAPTIVE_GAIN for item in gain)
        )
        self.assertTrue(
            all(0.0 <= item <= MAX_APPLIED_LAG_STEPS for item in applied)
        )
        single_applied, _, _ = adaptive_applied_lag(
            [(6, 6)],
            (0.999, 0.995),
            (0.0010, 0.0007),
        )
        self.assertFalse(np.allclose(applied, single_applied))

    def test_short_v13_adaptive_run_is_finite_and_safe(self) -> None:
        settings = BenchmarkSettings(
            samples=81,
            duration=4.0,
            control_points=8,
            iterations=4,
            domain_seeds=(1901,),
            number_of_windows=2,
            half_width=3,
        )
        reference = make_trajectory_family(
            "rounded_square",
            samples=settings.samples,
            duration=settings.duration,
        )
        basis = cubic_bspline_basis(settings.samples, settings.control_points)
        specification = make_semantic_task_specification(
            reference,
            "rounded_square",
            settings.half_width,
        )
        result = run_adaptive_delay_method(
            "adaptive_rolling_delay",
            reference,
            basis,
            specification,
            1901,
            0,
            settings,
            BALANCED_DELAY_SCENARIOS[1],
        )
        self.assertEqual(result["finite_result"], 1)
        self.assertEqual(result["all_updates_succeeded"], 1)
        self.assertEqual(result["constraint_violation"], 0)

    def test_v13_stopped_and_kept_formal_seeds_unused(self) -> None:
        prior = set(V12_DEVELOPMENT_DOMAIN_SEEDS) | set(
            V12_FORMAL_DOMAIN_SEEDS
        )
        self.assertFalse(prior & set(V13_DEVELOPMENT_DOMAIN_SEEDS))
        self.assertFalse(prior & set(V13_FORMAL_DOMAIN_SEEDS))
        self.assertFalse(
            set(V13_DEVELOPMENT_DOMAIN_SEEDS) & set(V13_FORMAL_DOMAIN_SEEDS)
        )
        summary_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "development_v13_adaptive_delay"
            / "comparison_v13_development.json"
        )
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertFalse(payload["formal_domain_seeds_used"])
        self.assertEqual(
            payload["classification"],
            "V13_DEVELOPMENT_FAILED_FINAL_METHOD_V11",
        )
        self.assertEqual(payload["final_selected_method"], "delay_aware_dual_anchor")

    def test_short_ilc_run_reduces_global_error(self) -> None:
        basis = cubic_bspline_basis(
            samples=self.reference.time.size,
            control_points=10,
        )
        feedback = simulate_machine(
            self.reference.position,
            self.reference.dt,
            mismatched_virtual_machine(),
        )
        contour = task_errors(self.reference, feedback)["contour"]
        selection = select_critical_windows(
            self.reference,
            contour,
            number_of_windows=2,
            half_width=8,
        )
        kinematics = command_kinematics(
            self.reference.position,
            self.reference.dt,
        )
        config = ILCConfig(
            iterations=2,
            correction_limit=4.0,
            velocity_limit=1.4
            * float(np.max(np.abs(kinematics["velocity"]))),
            acceleration_limit=1.7
            * float(np.max(np.abs(kinematics["acceleration"]))),
            learning_rate=0.6,
        )
        result = run_ilc(
            name="integration_test",
            reference=self.reference,
            initial_command=self.reference.position,
            basis=basis,
            evaluation_mask=selection.mask,
            optimization_mask=selection.mask,
            nominal_model=nominal_config(),
            virtual_plant=mismatched_virtual_machine(),
            config=config,
            critical_weighting=False,
        )
        self.assertEqual(len(result.metrics), 3)
        self.assertTrue(
            all(status["success"] for status in result.solver_status)
        )
        self.assertLess(
            result.metrics[-1]["global_rmse"],
            result.metrics[0]["global_rmse"],
        )


if __name__ == "__main__":
    unittest.main()
