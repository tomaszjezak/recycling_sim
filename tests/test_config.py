import json
import tempfile
import unittest
from pathlib import Path

from recycling_mrf.config import SimulationConfig


class SimulationConfigTests(unittest.TestCase):
    def test_load_demo_config(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.camera.resolution, (960, 540))
        self.assertEqual(config.main_belt.length, 42.0)
        self.assertEqual(config.environment.mode, "procedural")
        self.assertEqual(config.environment.layout_preset, "edco_conveyor_segment_a")
        self.assertEqual(config.system.sources[0].id, "homes")
        self.assertEqual(config.system.stream_routes["recycling"].initial_facility_id, "edco_mrf")
        self.assertEqual(config.item_catalog["pet_bottle"].stream, "recycling")
        self.assertEqual(config.item_catalog["pet_bottle"].commodity_target, "pet")
        self.assertEqual(
            [station.station_type for station in config.stations[:5]],
            ["feeder", "screen", "magnet", "optical_sorter", "manual_qc"],
        )

    def test_load_segment_anchor_config(self) -> None:
        config = SimulationConfig.from_file("configs/edco_conveyor_segment_a.json")
        self.assertEqual(config.environment.mode, "procedural")
        self.assertEqual(config.environment.layout_preset, "edco_conveyor_segment_a")
        self.assertEqual(config.output_dir.name, "edco_conveyor_segment_a")

    def test_large_v2_robotic_plastic_sort_schema_loads(self) -> None:
        config = SimulationConfig.from_file("configs/recycling_facility_large_v2.json")
        self.assertEqual(config.system.commodity_end_markets["plastic_film"], "plastic_reclaimer")
        self.assertEqual([zone.id for zone in config.perception_zones], ["pz_plastic_sort_line"])
        self.assertEqual(
            {cell.id for cell in config.robot_cells},
            {"rc_pet_arm", "rc_hdpe_arm", "rc_film_arm"},
        )
        self.assertTrue(all(cell.robot_type == "xarm6" for cell in config.robot_cells))
        self.assertEqual(
            {cell.place_segment_id for cell in config.robot_cells},
            {"s10_pet_branch", "s11_hdpe_branch", "s23_film_branch"},
        )
        self.assertTrue(all(cell.controller == "scripted_pick_policy" for cell in config.robot_cells))

    def test_invalid_item_mix_reference_raises(self) -> None:
        invalid = """
        {
          "seed": 1,
          "episode_duration": 2.0,
          "physics_dt": 0.01,
          "render_dt": 0.1,
          "output_dir": "outputs/test",
          "system": {
            "sources": [{"id": "homes", "source_type": "home", "share": 1.0}],
            "facilities": [
              {"id": "edco_mrf", "facility_type": "mrf", "description": ""},
              {"id": "landfill", "facility_type": "disposal", "description": ""}
            ],
            "stream_routes": {
              "recycling": {"initial_facility_id": "edco_mrf", "residual_facility_id": "landfill"},
              "organics": {"initial_facility_id": "landfill"},
              "trash": {"initial_facility_id": "landfill"},
              "construction_demolition": {"initial_facility_id": "landfill"}
            },
            "commodity_end_markets": {},
            "item_mix": {"missing": 1.0},
            "item_catalog": {}
          },
          "mrf": {
            "main_belt": {"length": 2, "width": 1, "height": 0.2, "speed": 0.5},
            "spawn": {"rate": 1.0, "drop_height": 0.2, "lane_jitter": 0.1, "yaw_jitter_deg": 10},
            "camera": {"position": [0, 0, 1], "look_at": [0, 0, 0], "resolution": [64, 64]},
            "environment": {"mode": "warehouse", "environment_id": "simple_warehouse", "layout_preset": "full_mrf", "conveyor_transform": {"position": [0,0,0], "yaw_deg": 0}, "bin_offset": [0,1,0], "station_marker_height": 1.0, "dome_light_intensity": 300.0},
            "stations": [{"name": "presort", "station_type": "presort", "x_range": [-0.8, -0.2], "target_materials": ["residue"], "capture_area": {"offset": [0,0.8,0.2], "size": [0.5,0.5,0.2]}}]
          }
        }
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.json"
            path.write_text(invalid)
            with self.assertRaises(ValueError):
                SimulationConfig.from_file(path)

    def test_invalid_station_type_raises(self) -> None:
        invalid = json.loads(Path("configs/conveyor_demo.json").read_text())
        invalid["mrf"]["stations"][0]["station_type"] = "separator"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid_station.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                SimulationConfig.from_file(path)

    def test_overlapping_station_ranges_raise(self) -> None:
        invalid = json.loads(Path("configs/conveyor_demo.json").read_text())
        invalid["mrf"]["stations"][1]["x_range"] = [-16.0, -14.0]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "overlap.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                SimulationConfig.from_file(path)

    def test_unknown_facility_route_raises(self) -> None:
        invalid = json.loads(Path("configs/conveyor_demo.json").read_text())
        invalid["system"]["stream_routes"]["trash"]["initial_facility_id"] = "missing"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "unknown_route.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                SimulationConfig.from_file(path)

    def test_invalid_facility_type_raises(self) -> None:
        invalid = json.loads(Path("configs/conveyor_demo.json").read_text())
        invalid["system"]["facilities"][0]["facility_type"] = "moon_base"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid_facility.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                SimulationConfig.from_file(path)

    def test_recycling_items_require_commodity_target(self) -> None:
        invalid = json.loads(Path("configs/conveyor_demo.json").read_text())
        del invalid["system"]["item_catalog"]["pet_bottle"]["commodity_target"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid_target.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                SimulationConfig.from_file(path)

    def test_normalized_item_mix_sums_to_one(self) -> None:
        config = SimulationConfig.from_file("configs/conveyor_demo.json")
        self.assertAlmostEqual(sum(config.normalized_item_mix.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
