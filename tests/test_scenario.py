import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from recycling_mrf.config import SimulationConfig
from recycling_mrf.isaac_runner import IsaacConveyorRunner
from recycling_mrf.scenario import build_metadata_frame_times, generate_episode_plan, resolve_material_route


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

    def test_dense_topology_scene_exports_surface_path_metadata(self) -> None:
        config = SimulationConfig.from_file("configs/dense_mrf_process_line.json")
        plan = generate_episode_plan(config)
        segments = {segment["id"]: segment for segment in plan.facility["segments"]}
        self.assertIn("s1_infeed_incline", segments)
        infeed = segments["s1_infeed_incline"]
        self.assertIn("surface_path", infeed)
        self.assertEqual(len(infeed["surface_path"]["samples"]), 9)
        self.assertIn("surface_normal", infeed)
        self.assertEqual(tuple(plan.facility["spawn_points"][0]["position"]), tuple(infeed["surface_path"]["start"]))

    def test_dense_topology_spawns_on_conveyor_centerline(self) -> None:
        config = SimulationConfig.from_file("configs/dense_mrf_process_line.json")
        plan = generate_episode_plan(config)
        self.assertTrue(plan.events)
        for event in plan.events[:5]:
            self.assertEqual(event.spawn_segment_id, "s1_infeed_incline")
            self.assertAlmostEqual(event.lane_offset, 0.0)
            self.assertAlmostEqual(event.yaw_deg, 0.0)

    def test_dense_topology_rejects_unanchored_conveyor_endpoints(self) -> None:
        config = SimulationConfig.from_file("configs/dense_mrf_process_line.json")
        broken_nodes = tuple(
            replace(node, upstream_segment_ids=("s10_baler_feed",))
            if node.id == "n11_ferrous_drop"
            else node
            for node in config.routing_nodes
        )
        broken_config = replace(config, mrf=replace(config.mrf, routing_nodes=broken_nodes))
        with self.assertRaisesRegex(ValueError, "meaningful downstream endpoint"):
            broken_config.validate()

    def test_large_v2_config_loads_with_expected_topology_density(self) -> None:
        config = SimulationConfig.from_file("configs/recycling_facility_large_v2.json")
        self.assertEqual(config.environment.layout_preset, "recycling_facility_large_v2")
        self.assertGreaterEqual(len(config.conveyor_segments), 20)
        self.assertGreaterEqual(len(config.machine_zones), 8)
        self.assertGreaterEqual(len(config.platforms), 4)
        self.assertGreaterEqual(len(config.drop_zones), 6)
        split_count = sum(1 for node in config.routing_nodes if node.node_type == "split")
        merge_count = sum(1 for node in config.routing_nodes if node.node_type == "merge")
        self.assertGreaterEqual(split_count, 3)
        self.assertGreaterEqual(merge_count, 2)

    def test_large_v2_dry_run_exports_large_topology_metadata(self) -> None:
        config = SimulationConfig.from_file("configs/recycling_facility_large_v2.json")
        plan = generate_episode_plan(config)
        self.assertEqual(plan.facility["environment"]["layout_preset"], "recycling_facility_large_v2")
        self.assertGreaterEqual(len(plan.facility["segments"]), 20)
        self.assertGreaterEqual(len(plan.facility["machines"]), 8)
        self.assertGreaterEqual(len(plan.facility["platforms"]), 4)
        self.assertGreaterEqual(len(plan.facility["drop_zones"]), 6)
        self.assertIn("E. Bunker, reject, and outbound staging", plan.facility["subareas"])

    def test_large_v2_runner_uses_topology_layout_dispatch(self) -> None:
        config = SimulationConfig.from_file("configs/recycling_facility_large_v2.json")
        runner = IsaacConveyorRunner(config, headless=True)
        self.assertTrue(runner._uses_topology_layout())
        self.assertEqual(runner._topology_layout_preset(), "recycling_facility_large_v2")

    def test_large_v2_runner_uses_internal_camera_poses(self) -> None:
        config = SimulationConfig.from_file("configs/recycling_facility_large_v2.json")
        runner = IsaacConveyorRunner(config, headless=True)
        position, look_at, focal_length = runner._viewport_camera_pose()
        self.assertEqual(position, (-32.0, -20.0, 17.5))
        self.assertEqual(look_at, (34.0, 4.0, 7.4))
        self.assertEqual(focal_length, 18.0)
        self.assertEqual(runner._overview_viewport_camera_path(), "/World/Camera/WideOverviewCamera")

    def test_large_v2_routes_cover_major_commodity_branches(self) -> None:
        config = SimulationConfig.from_file("configs/recycling_facility_large_v2.json")
        pet_route = resolve_material_route(config, config.item_catalog["pet_bottle"])
        hdpe_route = resolve_material_route(config, config.item_catalog["hdpe_jug"])
        ferrous_route = resolve_material_route(config, config.item_catalog["steel_can"])
        nonferrous_route = resolve_material_route(config, config.item_catalog["aluminum_can"])
        glass_route = resolve_material_route(config, config.item_catalog["glass_bottle"])
        residue_route = resolve_material_route(config, config.item_catalog["residue"])
        self.assertIn("s10_pet_branch", pet_route.segment_ids)
        self.assertIn("s11_hdpe_branch", hdpe_route.segment_ids)
        self.assertEqual(ferrous_route.drop_zone_id, "dz_ferrous_bunker")
        self.assertEqual(nonferrous_route.outbound_zone_id, "dz_outbound_staging")
        self.assertEqual(glass_route.drop_zone_id, "dz_glass_bunker")
        self.assertEqual(residue_route.final_status, "residual_disposed")


if __name__ == "__main__":
    unittest.main()
