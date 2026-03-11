import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from recycling_mrf.config import SimulationConfig
from recycling_mrf.isaac_runner import IsaacConveyorRunner
from recycling_mrf.scenario import build_metadata_frame_times, generate_episode_plan


class ScenarioTests(unittest.TestCase):
    def test_interactive_runner_uses_safe_simulation_app_defaults(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        runner = IsaacConveyorRunner(config, headless=False)
        self.assertEqual(
            runner._simulation_app_config(),
            {
                "headless": False,
                "active_gpu": 0,
                "physics_gpu": 0,
                "multi_gpu": False,
                "renderer": "RaytracedLighting",
                "anti_aliasing": 1,
                "sync_loads": False,
            },
        )

    def test_headless_runner_keeps_explicit_gpu_selection(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        runner = IsaacConveyorRunner(config, headless=True)
        self.assertEqual(
            runner._simulation_app_config(),
            {
                "headless": True,
                "active_gpu": 0,
                "physics_gpu": 0,
            },
        )

    def test_episode_plan_is_deterministic_for_same_seed(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        self.assertEqual(generate_episode_plan(config).to_dict(), generate_episode_plan(config).to_dict())

    def test_spawn_rate_changes_event_count(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        baseline = generate_episode_plan(config)
        faster_config = replace(config, mrf=replace(config.mrf, spawn=replace(config.spawn, rate=config.spawn.rate * 2.0)))
        faster = generate_episode_plan(faster_config)
        self.assertGreater(len(faster.events), len(baseline.events))

    def test_frame_times_cover_episode_endpoints(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        frame_times = build_metadata_frame_times(config)
        self.assertEqual(frame_times[0], 0.0)
        self.assertLessEqual(frame_times[-1], config.episode_duration)

    def test_episode_plan_includes_system_facility_and_summary_metadata(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        plan = generate_episode_plan(config)
        plan_dict = plan.to_dict()
        self.assertIn("system", plan_dict)
        self.assertIn("facility_catalog", plan_dict)
        self.assertIn("summaries", plan_dict)
        self.assertIn("stations", plan.facility)
        self.assertEqual(plan.facility["environment"]["layout_preset"], "full_mrf")
        self.assertAlmostEqual(sum(plan.item_mix.values()), 1.0)

    def test_every_material_unit_has_terminal_disposition(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        plan = generate_episode_plan(config)
        self.assertTrue(plan.material_units)
        self.assertTrue(all(unit.final_status for unit in plan.material_units))

    def test_non_recycling_streams_never_enter_spawn_events(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        plan = generate_episode_plan(config)
        self.assertTrue(plan.events)
        self.assertTrue(all(event.stream == "recycling" for event in plan.events))
        non_recycling = [unit for unit in plan.material_units if unit.stream != "recycling"]
        self.assertTrue(non_recycling)

    def test_recycling_units_include_required_upstream_stages(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        plan = generate_episode_plan(config)
        self.assertTrue(plan.item_lifecycles)
        for lifecycle in plan.item_lifecycles[:5]:
            stages = [stage.stage for stage in lifecycle.lifecycle]
            self.assertEqual(stages[:6], [
                "generated",
                "source_separated",
                "collected",
                "delivered_to_facility",
                "tipped",
                "metered_to_conveyor",
            ])

    def test_ferrous_items_route_to_magnet_then_end_market(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        plan = generate_episode_plan(config)
        steel_routes = [lifecycle for lifecycle in plan.item_lifecycles if lifecycle.item_type == "steel_can"]
        self.assertTrue(steel_routes)
        for lifecycle in steel_routes:
            self.assertIn("ferrous_magnet", [station.station_name for station in lifecycle.route])
            self.assertEqual(lifecycle.final_status, "shipped_to_end_market")

    def test_screenable_fiber_routes_to_screen(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        plan = generate_episode_plan(config)
        fiber_routes = [lifecycle for lifecycle in plan.item_lifecycles if lifecycle.item_type in {"cardboard", "mixed_paper"}]
        self.assertTrue(fiber_routes)
        self.assertTrue(all(lifecycle.route[-1].station_name == "fiber_screen" for lifecycle in fiber_routes))

    def test_residue_is_removed_before_downstream_stations(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        plan = generate_episode_plan(config)
        residue_routes = [lifecycle for lifecycle in plan.item_lifecycles if lifecycle.item_type == "residue"]
        self.assertTrue(residue_routes)
        for lifecycle in residue_routes:
            self.assertEqual(lifecycle.route[-1].station_name, "presort")
            self.assertEqual(lifecycle.final_status, "residual_disposed")

    def test_targeted_plastics_and_nonferrous_are_recovered_downstream(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        plan = generate_episode_plan(config)
        targets = {
            "pet_bottle": "container_optics",
            "hdpe_container": "container_optics",
            "detergent_jug": "container_optics",
            "aluminum_can": "manual_qc",
        }
        for item_type, station_name in targets.items():
            lifecycles = [lifecycle for lifecycle in plan.item_lifecycles if lifecycle.item_type == item_type]
            self.assertTrue(lifecycles)
            self.assertTrue(all(lifecycle.route[-1].station_name == station_name for lifecycle in lifecycles))
            self.assertTrue(all(lifecycle.final_status == "shipped_to_end_market" for lifecycle in lifecycles))

    def test_dry_run_writes_richer_metadata(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            runner_config = replace(config, output_dir=Path(tmpdir) / "dry_run")
            runner = IsaacConveyorRunner(runner_config, headless=True)
            runner.dry_run()
            payload = json.loads((runner_config.output_dir / "episode_plan.json").read_text())
            self.assertIn("material_units", payload)
            self.assertIn("facility_catalog", payload)
            self.assertIn("summaries", payload)

    def test_segment_loop_prefix_and_force_exit_still_work(self) -> None:
        config = SimulationConfig.from_file("configs/edco_conveyor_segment_a.json")
        runner = IsaacConveyorRunner(config, loop=True)
        plan = runner._generate_plan(cycle_index=1)
        self.assertTrue(plan.events)
        self.assertTrue(all(event.item_id.startswith("cycle_0001_") for event in plan.events))
        self.assertTrue(all(lifecycle.final_status == "residual_disposed" for lifecycle in plan.item_lifecycles))


if __name__ == "__main__":
    unittest.main()
