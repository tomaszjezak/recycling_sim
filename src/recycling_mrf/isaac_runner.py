from __future__ import annotations

import math
import traceback

from recycling_mrf.config import SimulationConfig
from recycling_mrf.scenario import (
    EpisodePlan,
    build_metadata_frame_times,
    estimate_belt_exit_time,
    generate_episode_plan,
)


class IsaacSimUnavailableError(RuntimeError):
    pass


class IsaacConveyorRunner:
    def __init__(
        self,
        config: SimulationConfig,
        headless: bool = False,
        viewport_mode: str = "overview",
        bare_bones: bool = False,
        loop: bool = False,
        loop_cycle_seconds: float = 12.0,
        detections: bool = False,
    ) -> None:
        self.config = config
        self.headless = headless
        self.viewport_mode = viewport_mode
        self.bare_bones = bare_bones
        self.loop = loop
        self.loop_cycle_seconds = loop_cycle_seconds
        self.detections = detections
        self._stage = None

    def run(self) -> EpisodePlan:
        plan = self._generate_plan()
        plan.write_json(self.config.output_dir / "episode_plan.json")
        print(f"IsaacConveyorRunner.run headless={self.headless}", flush=True)

        simulation_app_cls = self._load_simulation_app()
        simulation_app = simulation_app_cls(self._simulation_app_config())
        try:
            modules = self._load_runtime_modules()
            return self._run_with_modules(plan, modules, simulation_app)
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            simulation_app.close()

    def dry_run(self) -> EpisodePlan:
        plan = self._generate_plan()
        plan.write_json(self.config.output_dir / "episode_plan.json")
        return plan

    def _generate_plan(self, cycle_index: int = 0) -> EpisodePlan:
        return generate_episode_plan(
            self.config,
            item_id_prefix=f"cycle_{cycle_index:04d}_" if cycle_index > 0 else "",
            force_mainline_exit=self._force_mainline_exit(),
        )

    def _force_mainline_exit(self) -> bool:
        return self.loop and self.config.environment.layout_preset == "edco_conveyor_segment_a"

    def _use_recirculating_loop(self) -> bool:
        return self.loop and self.config.environment.layout_preset == "edco_conveyor_segment_a"

    def _simulation_app_config(self) -> dict[str, object]:
        if self.headless:
            return {
                "headless": True,
                "active_gpu": 0,
                "physics_gpu": 0,
            }
        return {
            "headless": False,
            "active_gpu": 0,
            "physics_gpu": 0,
            "multi_gpu": False,
            "renderer": "RaytracedLighting",
            "anti_aliasing": 1,
            "sync_loads": False,
        }

    def _load_simulation_app(self):
        try:
            from isaacsim import SimulationApp  # type: ignore
        except ImportError:
            try:
                from omni.isaac.kit import SimulationApp  # type: ignore
            except ImportError as exc:
                raise IsaacSimUnavailableError(
                    "Isaac Sim Python packages are not installed. Run in Isaac Sim's Python environment or use --dry-run."
                ) from exc
        return SimulationApp

    def _load_runtime_modules(self) -> dict[str, object]:
        from pxr import Gf, UsdGeom, UsdLux  # type: ignore
        from omni.isaac.core import World  # type: ignore
        from omni.isaac.core.objects import DynamicCuboid, FixedCuboid  # type: ignore
        from omni.isaac.sensor import Camera  # type: ignore
        from isaacsim.storage.native import get_assets_root_path  # type: ignore
        from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore
        try:
            from omni.kit.viewport.utility import get_active_viewport  # type: ignore
        except ImportError:
            get_active_viewport = None

        try:
            from omni.isaac.core.objects import DynamicCylinder  # type: ignore
        except ImportError:
            DynamicCylinder = None

        return {
            "UsdGeom": UsdGeom,
            "Gf": Gf,
            "UsdLux": UsdLux,
            "World": World,
            "DynamicCuboid": DynamicCuboid,
            "DynamicCylinder": DynamicCylinder,
            "FixedCuboid": FixedCuboid,
            "Camera": Camera,
            "get_assets_root_path": get_assets_root_path,
            "add_reference_to_stage": add_reference_to_stage,
            "get_active_viewport": get_active_viewport,
        }

    def _run_with_modules(self, plan: EpisodePlan, modules: dict[str, object], simulation_app) -> EpisodePlan:
        UsdLux = modules["UsdLux"]
        UsdGeom = modules["UsdGeom"]
        Gf = modules["Gf"]
        World = modules["World"]
        FixedCuboid = modules["FixedCuboid"]
        DynamicCuboid = modules["DynamicCuboid"]
        DynamicCylinder = modules["DynamicCylinder"]
        Camera = modules["Camera"]
        get_assets_root_path = modules["get_assets_root_path"]
        add_reference_to_stage = modules["add_reference_to_stage"]
        get_active_viewport = modules["get_active_viewport"]

        world = World(stage_units_in_meters=1.0, physics_dt=self.config.physics_dt, rendering_dt=self.config.render_dt)
        stage = world.stage
        self._stage = stage

        dome_light = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
        dome_light.CreateIntensityAttr(self.config.environment.dome_light_intensity)
        dome_light.CreateColorAttr((1.0, 1.0, 1.0))
        preset = self.config.environment.layout_preset
        if preset == "edco_conveyor_segment_a":
            dome_light.CreateIntensityAttr(self.config.environment.dome_light_intensity * 0.28)
            self._add_segment_a_lighting(UsdLux, Gf)
        elif preset == "sims_big_sort_video_v2":
            dome_light.CreateIntensityAttr(self.config.environment.dome_light_intensity * 0.2)
            self._add_sims_big_sort_lighting(UsdLux, Gf)
        elif self._uses_topology_layout():
            dome_light.CreateIntensityAttr(self.config.environment.dome_light_intensity * 0.18)
            self._add_dense_mrf_lighting(UsdLux, Gf)

        if self.config.environment.mode == "warehouse":
            self._load_environment(add_reference_to_stage, get_assets_root_path)

        self._define_recycling_root(UsdGeom, Gf)
        self._add_facility_geometry(plan, world, FixedCuboid)
        if not self._uses_topology_layout():
            self._add_station_geometry(plan, world, FixedCuboid)
            self._add_capture_bins(plan, world, FixedCuboid)
        if not self.bare_bones:
            if preset == "edco_conveyor_segment_a":
                self._add_edco_conveyor_segment_visuals(plan, UsdGeom)
            elif preset in ("recycling_facility_large_v2", "recycling_facility_large_v3"):
                self._add_recycling_facility_large_visuals(plan, UsdGeom)
            elif preset == "sims_big_sort_video_v2":
                self._add_sims_big_sort_visuals(plan, UsdGeom)
            elif self._uses_topology_layout():
                self._add_dense_mrf_visuals(plan, UsdGeom)
            else:
                self._add_edco_mrf_visuals(plan, UsdGeom)
            self._add_camera_rig(world, FixedCuboid)
            self._define_viewport_camera(UsdGeom, Gf)
            self._define_overhead_camera(UsdGeom, Gf)
            if self._uses_topology_layout():
                self._define_dense_aux_cameras(UsdGeom, Gf)

        camera = None
        if self.headless:
            camera = Camera(
                prim_path="/World/Camera/RGBCamera",
                position=self.config.camera.position,
                resolution=self.config.camera.resolution,
                frequency=max(1, int(round(1.0 / self.config.render_dt))),
            )
            camera.initialize()
            camera.set_world_pose(
                position=self.config.camera.position,
                orientation=self._camera_orientation(self.config.camera.position, self.config.camera.look_at),
            )

        frame_times = set(build_metadata_frame_times(self.config))
        frames_dir = self.config.output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        world.reset()
        world.play()
        if not self.headless:
            if self.viewport_mode == "camera" and not self.bare_bones:
                self._set_viewport_camera(simulation_app, get_active_viewport, "/World/Camera/ViewportCamera")
            elif self.viewport_mode == "overhead" and not self.bare_bones:
                self._set_viewport_camera(simulation_app, get_active_viewport, "/World/Camera/OverheadCamera")
            else:
                self._frame_viewport_on_conveyor(simulation_app, get_active_viewport)

        if self.loop and not self._use_recirculating_loop():
            self._run_looping_cycles(
                plan,
                DynamicCuboid,
                DynamicCylinder,
                world,
                simulation_app,
                camera,
                frame_times,
                frames_dir,
            )
        elif self.loop and self._use_recirculating_loop():
            self._run_interactive_loop(plan, {}, DynamicCuboid, DynamicCylinder, world, simulation_app)
        elif self.headless:
            self._run_episode_loop(plan, {}, DynamicCuboid, DynamicCylinder, world, camera, frame_times, frames_dir)
        else:
            self._run_interactive_loop(plan, {}, DynamicCuboid, DynamicCylinder, world, simulation_app)

        return plan

    def _add_segment_a_lighting(self, UsdLux, Gf) -> None:
        if self._stage is None:
            return
        from pxr import UsdGeom  # type: ignore

        belt_length = self.config.main_belt.length
        half_length = belt_length / 2
        hall_width = 132.0
        ceiling_z = 110.0

        key = UsdLux.DistantLight.Define(self._stage, "/World/Lights/SegmentAKey")
        key.CreateIntensityAttr(2600.0)
        key.CreateColorAttr((0.98, 0.98, 1.0))
        key_xform = UsdGeom.Xformable(key.GetPrim())
        key_xform.AddRotateXYZOp().Set((54.0, -20.0, -18.0))

        fill = UsdLux.DistantLight.Define(self._stage, "/World/Lights/SegmentAFill")
        fill.CreateIntensityAttr(1500.0)
        fill.CreateColorAttr((0.9, 0.94, 1.0))
        fill_xform = UsdGeom.Xformable(fill.GetPrim())
        fill_xform.AddRotateXYZOp().Set((78.0, 32.0, 95.0))

        x_positions = [
            -half_length + 12.0 + step * ((belt_length - 24.0) / 5.0)
            for step in range(6)
        ]
        y_rows = (-45.0, -15.0, 15.0, 45.0)
        for row_index, y_pos in enumerate(y_rows):
            for col_index, x_pos in enumerate(x_positions):
                light = UsdLux.RectLight.Define(
                    self._stage,
                    f"/World/Lights/SegmentAHighBay_{row_index}_{col_index}",
                )
                light.CreateIntensityAttr(42000.0)
                light.CreateExposureAttr(0.0)
                light.CreateColorAttr((0.98, 0.98, 0.96))
                light.CreateWidthAttr(18.0)
                light.CreateHeightAttr(3.2)
                light.CreateNormalizeAttr(True)
                xform = UsdGeom.Xformable(light.GetPrim())
                xform.AddTranslateOp().Set((x_pos, y_pos, ceiling_z))
                xform.AddRotateXYZOp().Set((90.0, 0.0, 0.0))

        for side, y_pos in (("South", -52.0), ("North", 52.0)):
            strip = UsdLux.RectLight.Define(self._stage, f"/World/Lights/SegmentA{side}Strip")
            strip.CreateIntensityAttr(34000.0)
            strip.CreateColorAttr((0.95, 0.97, 1.0))
            strip.CreateWidthAttr(max(belt_length - 8.0, 20.0))
            strip.CreateHeightAttr(2.6)
            strip.CreateNormalizeAttr(True)
            strip_xform = UsdGeom.Xformable(strip.GetPrim())
            strip_xform.AddTranslateOp().Set((0.0, y_pos, 72.0))
            strip_xform.AddRotateXYZOp().Set((90.0, 0.0, 0.0))

        accent_positions = (
            (-half_length + 42.0, -34.0, 22.0),
            (-half_length + 120.0, 26.0, 30.0),
            (0.0, 4.0, 38.0),
            (half_length - 62.0, -28.0, 24.0),
        )
        for index, position in enumerate(accent_positions):
            accent = UsdLux.SphereLight.Define(self._stage, f"/World/Lights/SegmentAAccent_{index}")
            accent.CreateIntensityAttr(32000.0)
            accent.CreateRadiusAttr(0.45)
            accent.CreateColorAttr((1.0, 0.96, 0.9))
            accent_xform = UsdGeom.Xformable(accent.GetPrim())
            accent_xform.AddTranslateOp().Set(position)

    def _add_dense_mrf_lighting(self, UsdLux, Gf) -> None:
        if self._stage is None:
            return
        from pxr import UsdGeom  # type: ignore

        key = UsdLux.DistantLight.Define(self._stage, "/World/Lights/DenseMRFKey")
        key.CreateIntensityAttr(2400.0)
        key.CreateColorAttr((1.0, 0.98, 0.94))
        key_xform = UsdGeom.Xformable(key.GetPrim())
        key_xform.AddRotateXYZOp().Set((48.0, -14.0, -24.0))

        fill = UsdLux.DistantLight.Define(self._stage, "/World/Lights/DenseMRFFill")
        fill.CreateIntensityAttr(1200.0)
        fill.CreateColorAttr((0.84, 0.9, 1.0))
        fill_xform = UsdGeom.Xformable(fill.GetPrim())
        fill_xform.AddRotateXYZOp().Set((72.0, 28.0, 112.0))

        light_positions = [
            (-14.0, -8.5, 13.6),
            (-2.0, -7.8, 13.6),
            (10.0, -7.6, 13.6),
            (22.0, -6.2, 13.6),
            (34.0, -5.0, 13.6),
            (-12.0, 8.8, 13.2),
            (4.0, 8.6, 13.2),
            (20.0, 8.0, 13.2),
            (36.0, 7.2, 13.2),
        ]
        for idx, position in enumerate(light_positions):
            light = UsdLux.RectLight.Define(self._stage, f"/World/Lights/DenseMRFHighBay_{idx}")
            light.CreateIntensityAttr(36000.0 if idx % 2 == 0 else 30000.0)
            light.CreateColorAttr((0.96, 0.96, 0.92))
            light.CreateWidthAttr(5.0)
            light.CreateHeightAttr(1.1)
            light.CreateNormalizeAttr(True)
            xform = UsdGeom.Xformable(light.GetPrim())
            xform.AddTranslateOp().Set(position)
            xform.AddRotateXYZOp().Set((90.0, 0.0, 0.0))

        for idx, position in enumerate(((-18.0, 5.0, 5.8), (6.0, 0.0, 7.2), (22.0, 5.8, 8.7), (43.0, -1.0, 4.2))):
            accent = UsdLux.SphereLight.Define(self._stage, f"/World/Lights/DenseMRFAccent_{idx}")
            accent.CreateIntensityAttr(1800.0)
            accent.CreateRadiusAttr(0.4)
            accent.CreateColorAttr((1.0, 0.76, 0.48))
            accent_xform = UsdGeom.Xformable(accent.GetPrim())
            accent_xform.AddTranslateOp().Set(position)

    def _add_station_geometry(self, plan: EpisodePlan, world, FixedCuboid) -> None:
        segment_a = self.config.environment.layout_preset == "edco_conveyor_segment_a"
        colors = {
            "feeder": (0.62, 0.62, 0.26),
            "presort": (0.82, 0.58, 0.24),
            "screen": (0.52, 0.72, 0.28),
            "magnet": (0.3, 0.48, 0.82),
            "optical_sorter": (0.42, 0.44, 0.46) if segment_a else (0.5, 0.28, 0.78),
            "manual_qc": (0.44, 0.46, 0.48) if segment_a else (0.82, 0.3, 0.3),
        }
        for station in plan.facility["stations"]:
            if station["capture_center"] is None or station["capture_size"] is None:
                continue
            center_x = (station["x_range"][0] + station["x_range"][1]) / 2
            width = station["x_range"][1] - station["x_range"][0]
            color = colors.get(station["station_type"], (0.5, 0.5, 0.5))
            zone_height = 0.008 if segment_a else 0.02
            zone_z = self.config.environment.station_marker_height if not segment_a else self.config.main_belt.height + 0.02
            zone = world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/RecyclingLine/Stations/{station['name']}/Zone",
                    name=f"{station['name']}_zone",
                    position=self._line_to_world((center_x, 0.0, zone_z)),
                    scale=(width, self.config.main_belt.width * (0.94 if segment_a else 1.0), zone_height),
                    color=self._color_array(color),
                )
            )
            self._disable_collisions(zone)

            capture_color = tuple(min(channel + 0.05, 0.92) for channel in color) if segment_a else tuple(
                min(channel + 0.08, 1.0) for channel in color
            )
            capture = world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/RecyclingLine/Stations/{station['name']}/CaptureArea",
                    name=f"{station['name']}_capture",
                    position=self._line_to_world(tuple(station["capture_center"])),
                    scale=tuple(station["capture_size"]),
                    color=self._color_array(capture_color),
                )
            )
            self._disable_collisions(capture)

    def _add_capture_bins(self, plan: EpisodePlan, world, FixedCuboid) -> None:
        bin_size = (0.9, 0.8, 0.7)
        for station in plan.facility["stations"]:
            if station["capture_center"] is None:
                continue
            center = station["capture_center"]
            bin_position = (
                center[0] + self.config.environment.bin_offset[0],
                center[1] + self.config.environment.bin_offset[1] * self._capture_side_sign(center[1]),
                max(center[2] + self.config.environment.bin_offset[2], bin_size[2] / 2),
            )
            capture_bin = world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/RecyclingLine/Bins/{station['name']}",
                    name=f"{station['name']}_bin",
                    position=self._line_to_world(bin_position),
                    scale=bin_size,
                    color=self._color_array((0.18, 0.28, 0.34)),
                )
            )
            self._disable_collisions(capture_bin)

    def _add_facility_geometry(self, plan: EpisodePlan, world, FixedCuboid) -> None:
        if self._uses_topology_layout():
            self._add_topology_facility_geometry(plan, world, FixedCuboid)
            return
        belt_color = (0.14, 0.14, 0.14)
        world.scene.add(
            FixedCuboid(
                prim_path="/World/RecyclingLine/Facility/MainBelt",
                name="main_belt",
                position=self._line_to_world((0.0, 0.0, self.config.main_belt.height / 2)),
                scale=(self.config.main_belt.length, self.config.main_belt.width, self.config.main_belt.height),
                color=self._color_array(belt_color),
            )
        )
        if self.config.environment.mode == "procedural":
            world.scene.add(
                FixedCuboid(
                    prim_path="/World/RecyclingLine/Facility/Floor",
                    name="floor",
                    position=self._line_to_world((0.0, 0.0, -0.02)),
                    scale=(self.config.main_belt.length + 3.0, self.config.main_belt.width + 4.0, 0.04),
                    color=self._color_array((0.78, 0.8, 0.82)),
                )
            )

    def _add_topology_facility_geometry(self, plan: EpisodePlan, world, FixedCuboid) -> None:
        segment_data = {segment["id"]: segment for segment in plan.facility.get("segments", [])}
        preset = self.config.environment.layout_preset
        if preset in ("recycling_facility_large_v2", "recycling_facility_large_v3"):
            floor_center, floor_size = (32.0, 6.0), (180.0, 48.0)
        elif preset == "sims_big_sort_video_v2":
            floor_center, floor_size = (34.0, 2.0), (196.0, 34.0)
        else:
            floor_center, floor_size = (12.0, 1.0), (72.0, 34.0)
        floor = world.scene.add(
            FixedCuboid(
                prim_path="/World/RecyclingLine/Facility/Floor",
                name="facility_floor",
                position=self._line_to_world((floor_center[0], floor_center[1], -0.04)),
                scale=(floor_size[0], floor_size[1], 0.08),
                color=self._color_array((0.7, 0.72, 0.73)),
            )
        )
        self._disable_collisions(floor)
        for segment in segment_data.values():
            center, length, yaw_deg, incline_deg = self._segment_geometry(segment)
            belt = world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/RecyclingLine/Facility/{self._safe_token(segment['id'])}/Belt",
                    name=f"{segment['id']}_belt",
                    position=self._line_to_world(center),
                    scale=(length, segment["width"], segment["thickness"]),
                    color=self._color_array((0.11, 0.11, 0.12)),
                )
            )
            self._disable_collisions(belt)
            self._rotate_scene_object(
                belt,
                center,
                (0.0, -incline_deg, yaw_deg),
            )

    def _add_recycling_facility_large_visuals(self, plan: EpisodePlan, UsdGeom) -> None:
        if self._stage is None:
            return
        UsdGeom.Xform.Define(self._stage, "/World/RecyclingLine/Visuals")
        machine_green = (0.17, 0.5, 0.32)
        guardrail_yellow = (0.96, 0.82, 0.18)
        walkway_black = (0.16, 0.17, 0.18)
        steel_gray = (0.57, 0.59, 0.61)
        hopper_gray = (0.46, 0.48, 0.5)
        bunker_gray = (0.7, 0.72, 0.74)
        duct_blue = (0.18, 0.36, 0.68)
        accent_orange = (0.95, 0.48, 0.2)
        concrete_gray = (0.54, 0.55, 0.56)
        segment_data = {segment["id"]: segment for segment in plan.facility.get("segments", [])}
        floor_center_x, floor_center_y = 32.0, 6.0
        floor_size_x, floor_size_y = 180.0, 48.0
        wall_height = 16.0
        visual_blocks = plan.facility.get("visual_blocks", [])
        if visual_blocks:
            for block in visual_blocks:
                self._add_static_visual(
                    f"/World/RecyclingLine/Visuals/Blocks/{self._safe_token(block['id'])}",
                    block["shape"],
                    block["pose"]["position"],
                    block["size"],
                    block["color"],
                    rotate=(0.0, 0.0, block["pose"]["yaw_deg"]),
                )
        else:
            self._add_static_visual("/World/RecyclingLine/Visuals/Building/FloorPad", "cube", (floor_center_x, floor_center_y, -0.04), (floor_size_x, floor_size_y, 0.08), (0.68, 0.7, 0.71))
            self._add_static_visual("/World/RecyclingLine/Visuals/Building/NorthWall", "cube", (floor_center_x, floor_center_y + floor_size_y / 2 + 0.2, wall_height / 2), (floor_size_x + 2.0, 0.6, wall_height), (0.8, 0.8, 0.82))
            self._add_static_visual("/World/RecyclingLine/Visuals/Building/SouthWall", "cube", (floor_center_x, floor_center_y - floor_size_y / 2 - 0.2, wall_height / 2), (floor_size_x + 2.0, 0.6, wall_height), (0.78, 0.78, 0.8))
            self._add_static_visual("/World/RecyclingLine/Visuals/Building/WestWall", "cube", (floor_center_x - floor_size_x / 2 - 0.2, floor_center_y, wall_height / 2), (0.6, floor_size_y + 2.0, wall_height), (0.77, 0.77, 0.79))
            self._add_static_visual("/World/RecyclingLine/Visuals/Building/EastWall", "cube", (floor_center_x + floor_size_x / 2 + 0.2, floor_center_y, wall_height / 2), (0.6, floor_size_y + 2.0, wall_height), (0.77, 0.77, 0.79))
            self._add_static_visual("/World/RecyclingLine/Visuals/Building/Roof", "cube", (floor_center_x, floor_center_y, wall_height + 0.12), (floor_size_x, floor_size_y, 0.24), (0.64, 0.66, 0.68))
            self._add_static_visual("/World/RecyclingLine/Visuals/Building/Clerestory", "cube", (floor_center_x, floor_center_y, wall_height - 0.5), (floor_size_x - 8.0, 3.0, 0.2), (0.88, 0.9, 0.92))
        for segment in plan.facility.get("segments", []):
            self._add_conveyor_visual_assembly(segment, machine_green, steel_gray, guardrail_yellow, walkway_black)
        for node in plan.facility.get("nodes", []):
            self._add_routing_node_visual(node, segment_data, machine_green, steel_gray, bunker_gray)
        for platform in plan.facility.get("platforms", []):
            self._add_platform_visual(platform, walkway_black, guardrail_yellow, steel_gray)
        for machine in plan.facility.get("machines", []):
            self._add_machine_visual(machine, machine_green, hopper_gray, bunker_gray, duct_blue, accent_orange)
        for zone in plan.facility.get("drop_zones", []):
            fill = bunker_gray if zone["zone_type"] != "trailer" else (0.62, 0.66, 0.68)
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/DropZones/{self._safe_token(zone['id'])}",
                "cube",
                zone["pose"]["position"],
                zone["size"],
                fill,
            )
        self._add_perception_zone_visuals(plan, (0.2, 0.24, 0.28), accent_orange)
        self._add_robot_cell_visuals(plan, steel_gray, accent_orange, duct_blue)

    def _add_sims_big_sort_lighting(self, UsdLux, Gf) -> None:
        if self._stage is None:
            return
        from pxr import UsdGeom  # type: ignore

        key = UsdLux.DistantLight.Define(self._stage, "/World/Lights/SimsBigSortKey")
        key.CreateIntensityAttr(2100.0)
        key.CreateColorAttr((0.98, 0.99, 1.0))
        key_xform = UsdGeom.Xformable(key.GetPrim())
        key_xform.AddRotateXYZOp().Set((46.0, -10.0, -18.0))

        fill = UsdLux.DistantLight.Define(self._stage, "/World/Lights/SimsBigSortFill")
        fill.CreateIntensityAttr(980.0)
        fill.CreateColorAttr((0.9, 0.96, 1.0))
        fill_xform = UsdGeom.Xformable(fill.GetPrim())
        fill_xform.AddRotateXYZOp().Set((74.0, 18.0, 112.0))

        for index, x_pos in enumerate((-38.0, -8.0, 22.0, 52.0, 82.0, 112.0)):
            light = UsdLux.RectLight.Define(self._stage, f"/World/Lights/SimsBigSortBay_{index}")
            light.CreateIntensityAttr(36000.0)
            light.CreateColorAttr((0.99, 0.99, 0.97))
            light.CreateWidthAttr(18.0)
            light.CreateHeightAttr(1.8)
            light.CreateNormalizeAttr(True)
            xform = UsdGeom.Xformable(light.GetPrim())
            xform.AddTranslateOp().Set((x_pos, 2.0, 14.8))
            xform.AddRotateXYZOp().Set((90.0, 0.0, 0.0))

        for name, position, size in (
            ("DockWash", (-58.0, 7.0, 12.0), (10.0, 4.0)),
            ("NorthSide", (22.0, 11.8, 10.2), (54.0, 2.2)),
            ("ExportHall", (112.0, -8.4, 9.4), (28.0, 2.0)),
        ):
            light = UsdLux.RectLight.Define(self._stage, f"/World/Lights/SimsBigSort{name}")
            light.CreateIntensityAttr(24000.0)
            light.CreateColorAttr((0.96, 0.98, 1.0))
            light.CreateWidthAttr(size[0])
            light.CreateHeightAttr(size[1])
            light.CreateNormalizeAttr(True)
            xform = UsdGeom.Xformable(light.GetPrim())
            xform.AddTranslateOp().Set(position)
            xform.AddRotateXYZOp().Set((90.0, 0.0, 0.0))

    def _add_sims_big_sort_visuals(self, plan: EpisodePlan, UsdGeom) -> None:
        if self._stage is None:
            return
        UsdGeom.Xform.Define(self._stage, "/World/RecyclingLine/Visuals")

        shell_white = (0.94, 0.96, 0.98)
        shell_off_white = (0.9, 0.93, 0.95)
        floor_white = (0.93, 0.93, 0.93)
        belt_black = (0.1, 0.11, 0.12)
        frame_white = (0.97, 0.98, 0.99)
        support_gray = (0.72, 0.75, 0.78)
        catwalk_gray = (0.84, 0.86, 0.88)
        glass_blue = (0.78, 0.88, 0.94)
        panel_teal = (0.7, 0.86, 0.84)
        safety_yellow = (0.96, 0.84, 0.22)
        dark_gray = (0.34, 0.36, 0.38)
        bunker_gray = (0.82, 0.84, 0.86)
        segment_data = {segment["id"]: segment for segment in plan.facility.get("segments", [])}

        for path, shape, position, size, color in (
            ("/World/RecyclingLine/Visuals/Building/Floor", "cube", (34.0, 2.0, -0.04), (196.0, 34.0, 0.08), floor_white),
            ("/World/RecyclingLine/Visuals/Building/NorthWall", "cube", (34.0, 18.25, 8.8), (196.0, 0.55, 17.6), shell_white),
            ("/World/RecyclingLine/Visuals/Building/SouthWall", "cube", (34.0, -14.25, 8.8), (196.0, 0.55, 17.6), shell_white),
            ("/World/RecyclingLine/Visuals/Building/WestWallNorth", "cube", (-64.25, 11.6, 8.8), (0.55, 11.8, 17.6), shell_white),
            ("/World/RecyclingLine/Visuals/Building/WestWallSouth", "cube", (-64.25, -5.4, 8.8), (0.55, 17.6, 17.6), shell_white),
            ("/World/RecyclingLine/Visuals/Building/EastWall", "cube", (132.25, 2.0, 8.8), (0.55, 32.5, 17.6), shell_white),
            ("/World/RecyclingLine/Visuals/Building/Roof", "cube", (34.0, 2.0, 17.45), (196.0, 34.0, 0.22), shell_off_white),
            ("/World/RecyclingLine/Visuals/Building/Clerestory", "cube", (34.0, 2.0, 15.85), (182.0, 2.4, 0.16), glass_blue),
            ("/World/RecyclingLine/Visuals/Building/NorthAnnex", "cube", (18.0, 13.1, 3.6), (68.0, 5.2, 7.2), shell_off_white),
            ("/World/RecyclingLine/Visuals/Building/GlassHall", "cube", (34.0, 13.0, 4.2), (44.0, 5.0, 8.4), shell_white),
            ("/World/RecyclingLine/Visuals/Building/ExportLeanTo", "cube", (114.0, -10.4, 4.4), (40.0, 6.8, 8.8), shell_off_white),
            ("/World/RecyclingLine/Visuals/Building/DockApron", "cube", (-72.0, 8.0, 0.12), (18.0, 18.0, 0.24), (0.72, 0.74, 0.76)),
            ("/World/RecyclingLine/Visuals/Building/DockCanopy", "cube", (-70.0, 8.0, 11.8), (14.0, 18.0, 0.32), shell_off_white),
            ("/World/RecyclingLine/Visuals/Building/TippingPad", "cube", (-44.0, 7.0, 0.14), (28.0, 12.0, 0.18), (0.72, 0.72, 0.72)),
            ("/World/RecyclingLine/Visuals/Building/TippingWall", "cube", (-35.5, 12.8, 2.4), (19.0, 0.45, 4.8), bunker_gray),
            ("/World/RecyclingLine/Visuals/Building/ExportDoorFrame", "cube", (131.8, -6.0, 5.2), (0.36, 9.0, 10.4), dark_gray),
        ):
            self._add_static_visual(path, shape, position, size, color)

        for x_pos in (-44.0, -16.0, 12.0, 40.0, 68.0, 96.0, 124.0):
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Structure/ColumnNorth_{self._safe_token(str(x_pos))}",
                "cube",
                (x_pos, 15.6, 7.3),
                (0.28, 0.28, 14.6),
                support_gray,
            )
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Structure/ColumnSouth_{self._safe_token(str(x_pos))}",
                "cube",
                (x_pos, -11.6, 7.3),
                (0.28, 0.28, 14.6),
                support_gray,
            )
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Structure/Truss_{self._safe_token(str(x_pos))}",
                "cube",
                (x_pos, 2.0, 14.2),
                (0.22, 27.0, 0.22),
                support_gray,
            )

        for block in plan.facility.get("visual_blocks", []):
            block_id = block["id"]
            if not (
                block_id.startswith("white_machine_mass_")
                or block_id.startswith("service_box_")
                or block_id.startswith("bale_stack_")
                or block_id.startswith("highbay_")
            ):
                continue
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Details/{self._safe_token(block_id)}",
                block["shape"],
                block["pose"]["position"],
                block["size"],
                block["color"],
                rotate=(0.0, 0.0, block["pose"]["yaw_deg"]),
            )

        for segment in plan.facility.get("segments", []):
            self._add_sims_big_sort_conveyor(
                segment,
                belt_black,
                frame_white,
                support_gray,
                catwalk_gray,
                panel_teal,
                safety_yellow,
            )

        for node in plan.facility.get("nodes", []):
            self._add_sims_big_sort_node(node, segment_data, frame_white, support_gray, panel_teal, bunker_gray)

        for platform in plan.facility.get("platforms", []):
            self._add_platform_visual(platform, catwalk_gray, safety_yellow, support_gray)

        for machine in plan.facility.get("machines", []):
            self._add_sims_big_sort_machine(
                machine,
                frame_white,
                support_gray,
                dark_gray,
                glass_blue,
                panel_teal,
                safety_yellow,
            )

        for zone in plan.facility.get("drop_zones", []):
            root = f"/World/RecyclingLine/Visuals/DropZones/{self._safe_token(zone['id'])}"
            fill = bunker_gray if zone["zone_type"] in {"bunker", "trailer"} else shell_off_white
            self._add_static_visual(f"{root}/Pad", "cube", zone["pose"]["position"], zone["size"], fill)
            if zone["zone_type"] == "bunker":
                self._add_static_visual(
                    f"{root}/BackWall",
                    "cube",
                    (zone["pose"]["position"][0], zone["pose"]["position"][1] + zone["size"][1] / 2 - 0.12, zone["pose"]["position"][2] + 0.9),
                    (zone["size"][0], 0.24, 1.8),
                    support_gray,
                )
            elif zone["zone_type"] == "trailer":
                self._add_static_visual(
                    f"{root}/TrailerShell",
                    "cube",
                    (zone["pose"]["position"][0], zone["pose"]["position"][1], zone["pose"]["position"][2] + 0.55),
                    (zone["size"][0] * 0.92, zone["size"][1] * 0.86, 1.1),
                    dark_gray,
                )

        self._add_dense_camera_visuals(UsdGeom, safety_yellow, support_gray)

    def _add_dense_mrf_visuals(self, plan: EpisodePlan, UsdGeom) -> None:
        if self._stage is None:
            return
        UsdGeom.Xform.Define(self._stage, "/World/RecyclingLine/Visuals")
        machine_green = (0.17, 0.5, 0.32)
        guardrail_yellow = (0.96, 0.82, 0.18)
        walkway_black = (0.16, 0.17, 0.18)
        steel_gray = (0.57, 0.59, 0.61)
        hopper_gray = (0.46, 0.48, 0.5)
        bunker_gray = (0.7, 0.72, 0.74)
        duct_blue = (0.18, 0.36, 0.68)
        accent_orange = (0.95, 0.48, 0.2)
        concrete_gray = (0.54, 0.55, 0.56)
        rust_brown = (0.56, 0.34, 0.18)
        grime_dark = (0.24, 0.24, 0.22)
        safety_red = (0.7, 0.18, 0.16)
        segment_data = {segment["id"]: segment for segment in plan.facility.get("segments", [])}
        node_data = {node["id"]: node for node in plan.facility.get("nodes", [])}

        self._add_static_visual("/World/RecyclingLine/Visuals/Building/FloorPad", "cube", (12.0, 1.0, -0.04), (72.0, 34.0, 0.08), (0.72, 0.73, 0.74))
        self._add_static_visual("/World/RecyclingLine/Visuals/Building/NorthWall", "cube", (12.0, 18.0, 6.0), (72.0, 0.4, 12.0), (0.82, 0.83, 0.84))
        self._add_static_visual("/World/RecyclingLine/Visuals/Building/SouthWall", "cube", (12.0, -16.0, 6.0), (72.0, 0.4, 12.0), (0.8, 0.8, 0.8))
        self._add_static_visual("/World/RecyclingLine/Visuals/Building/WestWall", "cube", (-24.0, 1.0, 6.0), (0.4, 34.0, 12.0), (0.79, 0.79, 0.79))
        self._add_static_visual("/World/RecyclingLine/Visuals/Building/EastWall", "cube", (48.0, 1.0, 6.0), (0.4, 34.0, 12.0), (0.79, 0.79, 0.79))
        self._add_static_visual("/World/RecyclingLine/Visuals/Building/Roof", "cube", (12.0, 1.0, 13.4), (72.0, 34.0, 0.22), (0.68, 0.69, 0.7))
        self._add_static_visual("/World/RecyclingLine/Visuals/Building/Clerestory", "cube", (12.0, 1.0, 12.2), (70.0, 2.4, 0.16), (0.9, 0.92, 0.94))
        self._add_static_visual("/World/RecyclingLine/Visuals/Building/CableTray", "cube", (12.0, -5.6, 10.8), (62.0, 0.24, 0.18), concrete_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Building/AirMain", "cylinder", (8.0, 10.8, 10.4), (0.24, 0.24, 44.0), duct_blue, rotate=(0.0, 90.0, 0.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Building/AirDropA", "cylinder", (-6.0, 10.8, 7.2), (0.12, 0.12, 6.0), duct_blue)
        self._add_static_visual("/World/RecyclingLine/Visuals/Building/AirDropB", "cylinder", (22.0, 10.8, 7.8), (0.12, 0.12, 5.0), duct_blue)

        self._add_static_visual("/World/RecyclingLine/Visuals/Infeed/TippingFloor", "cube", (-22.0, 9.0, 0.05), (10.0, 8.0, 0.1), (0.62, 0.62, 0.62))
        self._add_static_visual("/World/RecyclingLine/Visuals/Infeed/Hopper", "cube", (-20.5, 6.4, 1.4), (5.0, 4.2, 2.8), hopper_gray, rotate=(0.0, 0.0, -10.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Infeed/LoaderPile", "cube", (-23.2, 8.4, 0.45), (2.8, 2.0, 0.8), (0.78, 0.56, 0.32), rotate=(0.0, 0.0, 12.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Infeed/TipWall", "cube", (-18.8, 9.7, 1.2), (7.4, 0.4, 2.4), concrete_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Infeed/HopperSkirt", "cube", (-17.3, 4.2, 1.1), (2.8, 0.4, 1.8), machine_green)

        for segment in plan.facility.get("segments", []):
            self._add_conveyor_visual_assembly(segment, machine_green, steel_gray, guardrail_yellow, walkway_black)
        for node in plan.facility.get("nodes", []):
            self._add_routing_node_visual(node, segment_data, machine_green, steel_gray, bunker_gray)

        for platform in plan.facility.get("platforms", []):
            self._add_platform_visual(platform, walkway_black, guardrail_yellow, steel_gray)

        for machine in plan.facility.get("machines", []):
            self._add_machine_visual(machine, machine_green, hopper_gray, bunker_gray, duct_blue, accent_orange)

        for zone in plan.facility.get("drop_zones", []):
            fill = bunker_gray if zone["zone_type"] != "trailer" else (0.62, 0.66, 0.68)
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/DropZones/{self._safe_token(zone['id'])}",
                "cube",
                zone["pose"]["position"],
                zone["size"],
                fill,
            )

        self._add_manual_sort_bays(segment_data, guardrail_yellow, bunker_gray, machine_green)
        self._add_dense_belt_clutter(segment_data, grime_dark, rust_brown)
        self._add_floor_clutter(concrete_gray, bunker_gray, rust_brown, grime_dark, safety_red)
        self._add_dense_camera_visuals(UsdGeom, guardrail_yellow, steel_gray)
        self._add_perception_zone_visuals(plan, steel_gray, accent_orange)
        self._add_robot_cell_visuals(plan, steel_gray, accent_orange, duct_blue)

        for x_pos in (-10.0, 2.0, 16.0, 30.0):
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Structure/{self._safe_token(f'Column_{x_pos}')}",
                "cube",
                (x_pos, 15.6, 6.0),
                (0.34, 0.34, 12.0),
                steel_gray,
            )
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Structure/{self._safe_token(f'ColumnSouth_{x_pos}')}",
                "cube",
                (x_pos, -13.6, 6.0),
                (0.34, 0.34, 12.0),
                steel_gray,
            )
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Structure/{self._safe_token(f'Truss_{x_pos}')}",
                "cube",
                (x_pos, 1.0, 11.2),
                (0.26, 28.0, 0.26),
                steel_gray,
            )
        for y_pos in (-10.0, 0.0, 10.0):
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Structure/{self._safe_token(f'LongitudinalTruss_{y_pos}')}",
                "cube",
                (12.0, y_pos, 11.6),
                (68.0, 0.22, 0.22),
                steel_gray,
            )

        for segment_id in ("s2_presort_elevated", "s6_optical_line"):
            segment = segment_data.get(segment_id)
            if segment is None:
                continue
            start = segment["start_pose"]["position"]
            end = segment["end_pose"]["position"]
            mid = (
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2 + (2.0 if segment_id == "s6_optical_line" else -2.0),
                (start[2] + end[2]) / 2 + 1.8,
            )
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Ducts/{self._safe_token(segment_id)}",
                "cylinder",
                mid,
                (0.28, 0.28, max(4.0, self._segment_length(segment) * 0.9)),
                duct_blue,
                rotate=(0.0, 90.0, 0.0),
            )

        for worker_idx, position in enumerate(((-6.0, 1.2, 5.2), (-2.8, -0.1, 5.2), (21.5, 8.0, 6.7))):
            self._add_worker(
                f"/World/RecyclingLine/Visuals/Workers/Dense_{worker_idx}",
                position,
                accent_orange,
                (0.82, 0.96, 0.2),
            )

    def _add_edco_mrf_visuals(self, plan: EpisodePlan, UsdGeom) -> None:
        if self._stage is None:
            return
        UsdGeom.Xform.Define(self._stage, "/World/RecyclingLine/Visuals")

        machine_green = (0.15, 0.52, 0.33)
        guardrail_yellow = (0.94, 0.82, 0.18)
        duct_blue = (0.16, 0.36, 0.72)
        sort_bay_gray = (0.7, 0.72, 0.74)
        steel_gray = (0.56, 0.58, 0.6)
        walkway_black = (0.18, 0.18, 0.2)
        hopper_gray = (0.46, 0.49, 0.5)
        bunker_gray = (0.66, 0.68, 0.7)
        safety_orange = (0.96, 0.44, 0.18)
        vest_lime = (0.82, 0.96, 0.2)
        bale_brown = (0.66, 0.53, 0.34)
        plastic_blue = (0.44, 0.72, 0.9)

        self._add_static_visual("/World/RecyclingLine/Visuals/FloorZone/TippingPad", "cube", (-5.35, 1.55, 0.02), (3.1, 2.8, 0.04), (0.72, 0.72, 0.72))
        self._add_static_visual("/World/RecyclingLine/Visuals/FloorZone/InfeedPit", "cube", (-4.15, 0.9, 0.4), (1.15, 1.0, 0.8), hopper_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/FloorZone/InfeedHopper", "cube", (-3.55, 0.5, 0.95), (1.5, 1.25, 0.72), hopper_gray, rotate=(0.0, 0.0, -18.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/FloorZone/InfeedConveyor", "cube", (-2.65, 0.25, 1.28), (2.6, 0.55, 0.24), machine_green, rotate=(0.0, 0.0, 22.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/FloorZone/MeteringDrum", "cylinder", (-1.6, 0.08, 1.02), (0.3, 0.3, 1.15), steel_gray, rotate=(90.0, 0.0, 0.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/System/MainLineHousing", "cube", (0.7, 0.0, 0.42), (9.6, 1.85, 0.24), machine_green)

        for index, offset in enumerate(((-5.9, 1.8, 0.16), (-5.35, 1.35, 0.14), (-4.8, 1.9, 0.12), (-5.0, 0.95, 0.16))):
            pile_color = (bale_brown, plastic_blue, (0.82, 0.82, 0.76), (0.36, 0.3, 0.26))[index]
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/FloorZone/{self._safe_token(f'TippingPile_{index}')}",
                "cube",
                offset,
                (0.65, 0.48, 0.22),
                pile_color,
                rotate=(0.0, 0.0, 18.0 - index * 10.0),
            )

        self._add_static_visual("/World/RecyclingLine/Visuals/Catwalk/Deck", "cube", (0.3, -1.65, 1.15), (8.6, 0.9, 0.12), walkway_black)
        self._add_static_visual("/World/RecyclingLine/Visuals/Catwalk/Kickplate", "cube", (0.3, -1.22, 1.05), (8.6, 0.08, 0.18), guardrail_yellow)
        for offset_x in (-4.0, -2.4, -0.8, 0.8, 2.4, 4.0):
            for rail_y in (-2.06, -1.24):
                self._add_static_visual(f"/World/RecyclingLine/Visuals/Catwalk/{self._safe_token(f'Post_{offset_x}_{rail_y}')}", "cube", (offset_x, rail_y, 1.5), (0.08, 0.08, 0.9), guardrail_yellow)
        for rail_y in (-2.06, -1.24):
            for rail_z, suffix in ((1.86, "Top"), (1.52, "Mid")):
                self._add_static_visual(f"/World/RecyclingLine/Visuals/Catwalk/{self._safe_token(f'{suffix}Rail_{rail_y}')}", "cube", (0.3, rail_y, rail_z), (8.6, 0.05, 0.06), guardrail_yellow)

        for station in plan.facility["stations"]:
            if station["capture_center"] is None:
                continue
            center_x = (station["x_range"][0] + station["x_range"][1]) / 2
            sign = self._capture_side_sign(station["capture_center"][1])
            bay_y = 1.55 * sign
            station_token = self._safe_token(station["name"])
            self._add_static_visual(f"/World/RecyclingLine/Visuals/SortBays/{station_token}", "cube", (center_x, bay_y, 0.58), (1.55, 1.0, 0.68), sort_bay_gray)
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Equipment/{station_token}_Frame", "cube", (center_x, 0.0, 1.55), (1.25, 1.9, 1.1), machine_green)
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Equipment/{station_token}_Chute", "cube", (center_x, 0.78 * sign, 1.08), (0.92, 0.52, 0.28), machine_green, rotate=(0.0, -24.0 * sign, 0.0))

            if station["station_type"] == "presort":
                self._add_static_visual(f"/World/RecyclingLine/Visuals/Equipment/{station_token}_Cabin", "cube", (center_x - 0.05, -1.62, 1.68), (1.15, 0.7, 0.6), bunker_gray)
                self._add_worker(f"/World/RecyclingLine/Visuals/Workers/{station_token}_A", (center_x - 0.3, -1.45, 1.25), safety_orange, vest_lime)
                self._add_worker(f"/World/RecyclingLine/Visuals/Workers/{station_token}_B", (center_x + 0.15, -1.78, 1.25), safety_orange, vest_lime)
            elif station["station_type"] == "screen":
                self._add_static_visual(f"/World/RecyclingLine/Visuals/Equipment/{station_token}_ScreenDeck", "cube", (center_x, 0.0, 1.62), (1.65, 1.25, 0.22), machine_green, rotate=(0.0, 0.0, -16.0))
                self._add_static_visual(f"/World/RecyclingLine/Visuals/Equipment/{station_token}_FiberChute", "cube", (center_x - 0.15, -1.1, 1.05), (0.92, 0.42, 0.24), machine_green, rotate=(0.0, 22.0, 0.0))
            elif station["station_type"] == "magnet":
                self._add_static_visual(f"/World/RecyclingLine/Visuals/Equipment/{station_token}_MagnetDrum", "cylinder", (center_x, 0.0, 1.78), (0.24, 0.24, 1.55), duct_blue, rotate=(90.0, 0.0, 0.0))
                self._add_static_visual(f"/World/RecyclingLine/Visuals/Equipment/{station_token}_Drop", "cube", (center_x, 1.1, 1.1), (0.6, 0.36, 0.22), machine_green, rotate=(0.0, -18.0, 0.0))
            elif station["station_type"] == "optical_sorter":
                self._add_static_visual(f"/World/RecyclingLine/Visuals/Equipment/{station_token}_OpticTower", "cube", (center_x - 0.22, 0.0, 1.92), (0.42, 0.52, 0.8), duct_blue)
                self._add_static_visual(f"/World/RecyclingLine/Visuals/Equipment/{station_token}_AirJet", "cube", (center_x + 0.28, 0.0, 1.66), (0.34, 0.58, 0.26), steel_gray)
            elif station["station_type"] == "manual_qc":
                self._add_static_visual(f"/World/RecyclingLine/Visuals/Equipment/{station_token}_OpticalTower", "cube", (center_x - 0.2, 0.0, 1.92), (0.42, 0.52, 0.8), duct_blue)
                self._add_static_visual(f"/World/RecyclingLine/Visuals/Equipment/{station_token}_AirJet", "cube", (center_x + 0.28, 0.0, 1.66), (0.34, 0.58, 0.26), steel_gray)
                self._add_worker(f"/World/RecyclingLine/Visuals/Workers/{station_token}_A", (center_x - 0.25, -1.55, 1.25), safety_orange, vest_lime)
                self._add_worker(f"/World/RecyclingLine/Visuals/Workers/{station_token}_B", (center_x + 0.25, -1.78, 1.25), safety_orange, vest_lime)

        self._add_static_visual("/World/RecyclingLine/Visuals/OverheadConveyor/Infeed", "cube", (-2.9, 1.65, 2.55), (3.0, 0.58, 0.36), machine_green, rotate=(0.0, 0.0, 14.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/OverheadConveyor/CrossFeed", "cube", (2.3, -1.1, 2.95), (3.8, 0.52, 0.32), machine_green, rotate=(0.0, 0.0, -22.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/OverheadConveyor/Riser", "cube", (4.05, -0.25, 2.3), (0.7, 0.7, 2.2), machine_green)
        for x_pos in (-3.4, -1.2, 1.0, 3.2):
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Ducts/{self._safe_token(f'Hood_{x_pos}')}", "cylinder", (x_pos, -0.1, 3.22), (0.32, 0.32, 0.78), duct_blue)
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Ducts/{self._safe_token(f'Drop_{x_pos}')}", "cylinder", (x_pos, -0.1, 2.62), (0.14, 0.14, 1.2), duct_blue)
        self._add_static_visual("/World/RecyclingLine/Visuals/Ducts/MainRun", "cylinder", (0.2, 0.55, 3.7), (0.18, 0.18, 8.4), steel_gray, rotate=(0.0, 90.0, 0.0))
        for x_pos in (-3.9, -1.3, 1.3, 3.9):
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Supports/{self._safe_token(f'Support_{x_pos}')}", "cube", (x_pos, -2.0, 0.6), (0.14, 0.14, 1.2), steel_gray)

        bunker_specs = [
            ("OCC", (2.6, -3.0, 0.7), bale_brown),
            ("MixedPaper", (0.9, -3.0, 0.7), (0.84, 0.82, 0.74)),
            ("Ferrous", (2.7, 2.9, 0.7), steel_gray),
            ("Plastics", (4.35, -2.55, 0.7), plastic_blue),
            ("Residue", (4.5, 2.65, 0.7), (0.38, 0.32, 0.28)),
        ]
        for name, position, fill_color in bunker_specs:
            bunker_token = self._safe_token(name)
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Bunkers/{bunker_token}", "cube", position, (1.25, 1.45, 1.1), bunker_gray)
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Bunkers/{bunker_token}Fill", "cube", (position[0], position[1], position[2] + 0.18), (0.88, 1.02, 0.46), fill_color)

        self._add_static_visual("/World/RecyclingLine/Visuals/Outbound/Baler", "cube", (5.45, -1.7, 1.05), (1.4, 1.0, 1.18), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Outbound/BalerFeed", "cube", (4.7, -1.35, 1.22), (1.25, 0.48, 0.28), machine_green, rotate=(0.0, 0.0, -12.0))
        for index, offset_y in enumerate((-2.55, -2.05, -1.55)):
            self._add_bale_stack(f"/World/RecyclingLine/Visuals/Outbound/Bales_{index}", (6.35 + index * 0.32, offset_y, 0.32), bale_brown if index != 1 else plastic_blue)

        self._add_static_visual("/World/RecyclingLine/Visuals/Residuals/Trailer", "cube", (5.9, 2.35, 0.78), (1.9, 1.0, 0.88), (0.64, 0.68, 0.72))
        self._add_static_visual("/World/RecyclingLine/Visuals/Residuals/Conveyor", "cube", (4.95, 1.8, 1.18), (1.5, 0.38, 0.24), machine_green, rotate=(0.0, 0.0, 18.0))

    def _add_edco_conveyor_segment_visuals(self, plan: EpisodePlan, UsdGeom) -> None:
        if self._stage is None:
            return
        UsdGeom.Xform.Define(self._stage, "/World/RecyclingLine/Visuals")

        machine_green = (0.16, 0.54, 0.34)
        guardrail_yellow = (0.96, 0.84, 0.2)
        walkway_black = (0.16, 0.17, 0.18)
        steel_gray = (0.58, 0.6, 0.62)
        housing_gray = (0.74, 0.76, 0.78)
        sensor_orange = (0.96, 0.44, 0.18)
        duct_silver = (0.72, 0.74, 0.76)
        residue_mix = (0.78, 0.76, 0.72)
        sensor_gray = (0.34, 0.36, 0.38)
        pet_blue = (0.42, 0.72, 0.9)
        metal_silver = (0.78, 0.8, 0.82)
        fiber_brown = (0.74, 0.58, 0.36)
        light_wall = (0.9, 0.91, 0.92)
        ladder_orange = (0.92, 0.58, 0.18)
        coil_red = (0.68, 0.28, 0.22)

        belt_length = self.config.main_belt.length
        half_length = belt_length / 2
        loop_radius = 39.0
        return_lane_y = 40.5
        catwalk_y = -39.0
        floor_length = belt_length + 40.0
        floor_width = 174.0
        stringer_length = belt_length + 1.2
        support_positions = [
            -half_length + 8.0 + step * ((belt_length - 16.0) / 17.0)
            for step in range(18)
        ]
        post_positions = [
            -half_length + 6.0 + step * ((belt_length - 12.0) / 23.0)
            for step in range(24)
        ]
        process_module_positions = [
            -half_length + 26.0 + step * ((belt_length - 52.0) / 5.0)
            for step in range(6)
        ]
        overhead_bridge_positions = [
            -half_length + 45.0 + step * ((belt_length - 90.0) / 3.0)
            for step in range(4)
        ]
        roof_column_positions = [
            -half_length + 10.0 + step * ((belt_length - 20.0) / 9.0)
            for step in range(10)
        ]

        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/Floor", "cube", (1.0, 0.0, -0.02), (floor_length, floor_width, 0.04), (0.72, 0.73, 0.74))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/MainLeftStringer", "cube", (0.0, 1.18, 0.42), (stringer_length, 0.18, 0.16), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/MainRightStringer", "cube", (0.0, -1.18, 0.42), (stringer_length, 0.18, 0.16), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/CenterBed", "cube", (0.0, 0.0, 0.18), (belt_length, 1.7, 0.1), (0.12, 0.12, 0.12))

        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/HeadDrum", "cylinder", (half_length + 0.2, 0.0, 0.56), (0.42, 0.42, 2.2), steel_gray, rotate=(90.0, 0.0, 0.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/TailDrum", "cylinder", (-half_length - 0.2, 0.0, 0.56), (0.42, 0.42, 2.2), steel_gray, rotate=(90.0, 0.0, 0.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/ReturnBelt", "cube", (0.0, return_lane_y, 0.28), (belt_length - 2.0, 1.25, 0.12), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/ReturnLeftRail", "cube", (0.0, return_lane_y + 0.74, 0.36), (belt_length - 2.0, 0.12, 0.12), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/ReturnRightRail", "cube", (0.0, return_lane_y - 0.74, 0.36), (belt_length - 2.0, 0.12, 0.12), machine_green)

        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/HeadLiftTower", "cube", (half_length + 1.8, 17.5, 30.0), (6.3, 45.0, 60.0), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/TailLiftTower", "cube", (-half_length - 1.8, 17.5, 30.0), (6.3, 45.0, 60.0), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/HeadLoopOuter", "cube", (half_length + 13.8, 17.5, 44.0), (26.4, 48.0, 0.84), machine_green, rotate=(0.0, 0.0, 88.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/TailLoopOuter", "cube", (-half_length - 13.8, 17.5, 44.0), (26.4, 48.0, 0.84), machine_green, rotate=(0.0, 0.0, -88.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/HeadLoopTop", "cube", (half_length - 3.0, return_lane_y / 2, 50.0), (33.0, 2.16, 0.84), machine_green, rotate=(0.0, 20.0, 56.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/TailLoopTop", "cube", (-half_length + 3.0, return_lane_y / 2, 50.0), (33.0, 2.16, 0.84), machine_green, rotate=(0.0, -20.0, -56.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/HeadTurnBrace", "cube", (half_length + 18.0, 17.5, 25.0), (1.5, 49.5, 49.0), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/TailTurnBrace", "cube", (-half_length - 18.0, 17.5, 25.0), (1.5, 49.5, 49.0), machine_green)

        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/CatwalkDeck", "cube", (0.0, catwalk_y, 1.28), (belt_length - 4.0, 1.3, 0.1), walkway_black)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/CatwalkKickplate", "cube", (0.0, catwalk_y + 0.62, 1.14), (belt_length - 4.0, 0.08, 0.18), guardrail_yellow)
        for offset_x in post_positions:
            for rail_y in (catwalk_y - 0.56, catwalk_y + 0.56):
                post_name = self._safe_token(f"SegPost_{offset_x}_{rail_y}")
                self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/{post_name}", "cube", (offset_x, rail_y, 1.68), (0.08, 0.08, 1.08), guardrail_yellow)
        for rail_y in (catwalk_y - 0.62, catwalk_y + 0.62):
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/{self._safe_token(f'SegTopRail_{rail_y}')}", "cube", (0.0, rail_y, 2.12), (belt_length - 4.0, 0.05, 0.06), guardrail_yellow)
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/{self._safe_token(f'SegMidRail_{rail_y}')}", "cube", (0.0, rail_y, 1.72), (belt_length - 4.0, 0.05, 0.06), guardrail_yellow)

        for support_x in support_positions:
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/{self._safe_token(f'Leg_{support_x}')}", "cube", (support_x, -0.9, 0.86), (0.18, 0.18, 1.72), steel_gray)
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/{self._safe_token(f'CenterLeg_{support_x}')}", "cube", (support_x, 0.0, 0.86), (0.16, 0.16, 1.72), steel_gray)
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/{self._safe_token(f'ReturnLeg_{support_x}')}", "cube", (support_x, return_lane_y, 0.72), (0.16, 0.16, 1.44), steel_gray)
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/{self._safe_token(f'CatwalkLeg_{support_x}')}", "cube", (support_x, catwalk_y, 0.72), (0.12, 0.12, 1.44), steel_gray)

        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/InfeedHousing", "cube", (-half_length + 54.0, 18.6, 25.0), (42.0, 11.4, 29.0), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/SorterHousing", "cube", (-half_length + 186.0, 17.4, 29.0), (84.0, 12.6, 31.0), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/QCHousing", "cube", (half_length - 174.0, 17.4, 27.5), (72.0, 12.0, 29.0), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/EndReturnHousing", "cube", (half_length - 42.0, 16.8, 30.0), (42.0, 10.8, 30.0), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/TransferBridge", "cube", (half_length - 108.0, 14.4, 41.0), (78.0, 3.0, 1.32), machine_green, rotate=(0.0, 0.0, -10.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/LoopCrossFeed", "cube", (half_length + 9.0, 14.4, 54.0), (31.5, 2.7, 1.2), machine_green, rotate=(0.0, 0.0, -72.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/LoopReturnDrop", "cube", (-half_length - 9.0, 27.6, 54.0), (31.5, 2.7, 1.2), machine_green, rotate=(0.0, 0.0, 72.0))

        for duct_x in (-half_length + 22.0, -half_length + 68.0, -half_length + 114.0, -30.0, 40.0, half_length - 82.0, half_length - 28.0):
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/{self._safe_token(f'DuctDrop_{duct_x}')}", "cylinder", (duct_x, 1.2, 11.0), (0.32, 0.32, 5.6), duct_silver)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/DuctRun", "cylinder", (0.0, 4.2, 14.1), (0.42, 0.42, belt_length + 18.0), duct_silver, rotate=(0.0, 90.0, 0.0))

        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/WestWall", "cube", (-half_length - 66.0, 0.0, 55.0), (1.5, floor_width - 4.0, 110.0), (0.8, 0.8, 0.8))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/EastWall", "cube", (half_length + 66.0, 0.0, 55.0), (1.5, floor_width - 4.0, 110.0), (0.8, 0.8, 0.8))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/NorthWall", "cube", (0.0, return_lane_y + loop_radius + 36.0, 55.0), (floor_length - 2.0, 1.5, 110.0), (0.82, 0.82, 0.82))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/SouthWall", "cube", (0.0, catwalk_y - 36.0, 55.0), (floor_length - 2.0, 1.5, 110.0), (0.78, 0.78, 0.78))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/CeilingPanel", "cube", (0.0, 2.0, 112.5), (floor_length - 3.0, floor_width - 3.0, 0.9), light_wall)
        for roof_x in roof_column_positions:
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/{self._safe_token(f'RoofNorthCol_{roof_x}')}", "cube", (roof_x, return_lane_y + loop_radius + 28.5, 50.0), (1.2, 1.2, 100.0), steel_gray)
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/{self._safe_token(f'RoofSouthCol_{roof_x}')}", "cube", (roof_x, catwalk_y - 27.0, 50.0), (1.2, 1.2, 100.0), steel_gray)
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/{self._safe_token(f'RoofTruss_{roof_x}')}", "cube", (roof_x, 2.0, 101.0), (0.96, floor_width - 12.0, 0.96), steel_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/RoofSpine", "cube", (0.0, 2.0, 106.0), (floor_length - 6.0, 1.26, 1.08), steel_gray)

        photo_anchor_x = -half_length + 168.0
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/MainScreenHousing", "cube", (photo_anchor_x, 1.8, 6.1), (24.0, 4.8, 7.0), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/ScreenSlope", "cube", (photo_anchor_x + 8.0, 0.4, 8.1), (18.0, 2.0, 0.7), machine_green, rotate=(0.0, 24.0, -13.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/TopGallery", "cube", (photo_anchor_x - 6.0, -1.8, 10.2), (16.0, 2.2, 0.28), housing_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/TopGalleryRailA", "cube", (photo_anchor_x - 6.0, -0.8, 11.2), (16.0, 0.12, 0.1), housing_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/TopGalleryRailB", "cube", (photo_anchor_x - 6.0, -2.8, 11.2), (16.0, 0.12, 0.1), housing_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/LowerFrame", "cube", (photo_anchor_x - 2.0, 4.6, 2.2), (30.0, 0.48, 4.2), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/LowerCross", "cube", (photo_anchor_x - 2.0, 4.6, 4.9), (30.0, 0.3, 0.3), machine_green)

        for idx, offset in enumerate((-10.0, -4.0, 2.0, 8.0)):
            x = photo_anchor_x + offset
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/PhotoBay/Support_{idx}", "cube", (x, 4.6, 4.2), (0.34, 0.34, 8.4), machine_green)
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/PhotoBay/DiagonalA_{idx}", "cube", (x + 0.9, 4.6, 3.3), (2.2, 0.14, 0.14), machine_green, rotate=(0.0, 42.0, 0.0))
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/PhotoBay/DiagonalB_{idx}", "cube", (x - 0.9, 4.6, 3.3), (2.2, 0.14, 0.14), machine_green, rotate=(0.0, -42.0, 0.0))

        for idx, ladder_x in enumerate((photo_anchor_x - 13.5, photo_anchor_x + 14.5)):
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/PhotoBay/LadderRailL_{idx}", "cube", (ladder_x - 0.22, -0.3, 6.4), (0.08, 0.08, 12.8), ladder_orange)
            self._add_static_visual(f"/World/RecyclingLine/Visuals/Segment/PhotoBay/LadderRailR_{idx}", "cube", (ladder_x + 0.22, -0.3, 6.4), (0.08, 0.08, 12.8), ladder_orange)
            for rung in range(12):
                self._add_static_visual(
                    f"/World/RecyclingLine/Visuals/Segment/PhotoBay/LadderRung_{idx}_{rung}",
                    "cube",
                    (ladder_x, -0.3, 1.0 + rung * 0.9),
                    (0.48, 0.06, 0.06),
                    ladder_orange,
                )

        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/MainDuctRun", "cylinder", (photo_anchor_x + 10.0, 0.8, 12.8), (0.86, 0.86, 52.0), duct_silver, rotate=(0.0, 90.0, 0.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/DuctElbowA", "cylinder", (photo_anchor_x + 1.0, 0.0, 11.0), (0.8, 0.8, 13.0), duct_silver, rotate=(28.0, 14.0, 36.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/DuctElbowB", "cylinder", (photo_anchor_x - 13.0, -2.2, 10.0), (0.7, 0.7, 11.0), duct_silver, rotate=(18.0, -20.0, -42.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/GreenTube", "cylinder", (photo_anchor_x + 27.0, 5.0, 8.2), (0.98, 0.98, 28.0), machine_green, rotate=(0.0, 90.0, 0.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/TubeSupport", "cube", (photo_anchor_x + 18.0, 5.0, 4.6), (0.42, 0.42, 9.2), ladder_orange)

        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/YellowPlatform", "cube", (photo_anchor_x - 22.0, -3.4, 4.8), (10.0, 2.4, 0.22), guardrail_yellow)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/YellowPlatformRail", "cube", (photo_anchor_x - 22.0, -2.3, 5.8), (10.0, 0.1, 0.1), guardrail_yellow)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/StairRun", "cube", (photo_anchor_x - 27.0, -4.6, 2.8), (7.0, 0.68, 0.24), guardrail_yellow, rotate=(0.0, 0.0, 36.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/StairLanding", "cube", (photo_anchor_x - 28.6, -5.9, 1.25), (2.8, 1.4, 0.16), walkway_black)

        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/ForegroundMachine", "cube", (photo_anchor_x - 4.0, -8.2, 1.9), (18.0, 5.6, 3.2), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/ForegroundBaler", "cube", (photo_anchor_x - 8.0, -8.2, 1.1), (8.0, 2.8, 1.5), machine_green)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PhotoBay/ForegroundRamp", "cube", (photo_anchor_x + 7.5, -9.7, 0.18), (8.0, 1.6, 0.14), (0.52, 0.46, 0.42), rotate=(0.0, 12.0, 0.0))
        for idx, coil_x in enumerate((photo_anchor_x - 10.0, photo_anchor_x - 6.5, photo_anchor_x - 2.5, photo_anchor_x + 3.0, photo_anchor_x + 8.0)):
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/PhotoBay/CopperCoil_{idx}",
                "cylinder",
                (coil_x, -11.4 + (idx % 2) * 0.5, 0.74),
                (0.9, 0.9, 1.24),
                coil_red,
                rotate=(90.0, 0.0, 0.0),
            )

        floor_fill_positions = [
            (-half_length + 34.0, -54.0),
            (-half_length + 84.0, -60.0),
            (-half_length + 132.0, 58.0),
            (-half_length + 188.0, -52.0),
            (-half_length + 236.0, 62.0),
            (-half_length + 286.0, -58.0),
            (-half_length + 334.0, 54.0),
            (half_length - 86.0, -62.0),
            (half_length - 42.0, 60.0),
        ]
        for index, (fill_x, fill_y) in enumerate(floor_fill_positions):
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/FloorFillMachine_{index}",
                "cube",
                (fill_x, fill_y, 6.0),
                (18.0, 7.0, 12.0),
                machine_green,
            )
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/FloorFillDeck_{index}",
                "cube",
                (fill_x, fill_y - 6.2 if fill_y > 0 else fill_y + 6.2, 3.6),
                (12.0, 3.2, 0.28),
                walkway_black,
            )
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/FloorFillBunker_{index}",
                "cube",
                (fill_x + 7.0, fill_y, 1.8),
                (8.0, 4.4, 2.8),
                housing_gray,
            )
        for pallet_index, pallet_x in enumerate(range(-150, 151, 30)):
            side = -72.0 if pallet_index % 2 == 0 else 72.0
            self._add_bale_stack(
                f"/World/RecyclingLine/Visuals/Segment/FloorBales_{pallet_index}",
                (float(pallet_x), side, 0.42),
                coil_red if pallet_index % 3 == 0 else fiber_brown,
            )

        residue_positions = [
            (-half_length + 12.0, 0.18, 0.88),
            (-half_length + 15.0, -0.24, 0.88),
            (-half_length + 18.5, 0.26, 0.88),
            (-half_length + 22.0, -0.12, 0.88),
            (-half_length + 26.0, 0.22, 0.88),
        ]
        for index, position in enumerate(residue_positions):
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/{self._safe_token(f'Residue_{index}')}",
                "cube",
                position,
                (0.72, 0.42, 0.1),
                residue_mix,
                rotate=(0.0, 0.0, -12.0 + index * 5.0),
            )

        for module_index, module_x in enumerate(process_module_positions):
            sign = 1.0 if module_index % 2 == 0 else -1.0
            y_base = 17.5 * sign
            housing_length = 18.0 if module_index % 2 == 0 else 14.0
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/ProcessHall_{module_index}",
                "cube",
                (module_x, y_base, 4.8),
                (housing_length, 4.2, 6.2),
                machine_green,
            )
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/ProcessDeck_{module_index}",
                "cube",
                (module_x, y_base - sign * 4.0, 3.1),
                (housing_length - 3.0, 2.6, 0.18),
                walkway_black,
            )
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/ProcessRail_{module_index}",
                "cube",
                (module_x, y_base - sign * 2.8, 4.0),
                (housing_length - 3.0, 0.08, 0.08),
                guardrail_yellow,
            )
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/ProcessChute_{module_index}",
                "cube",
                (module_x, y_base - sign * 6.8, 2.1),
                (4.8, 1.0, 0.4),
                machine_green,
                rotate=(0.0, 18.0 * sign, -12.0 * sign),
            )
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/ProcessBunker_{module_index}",
                "cube",
                (module_x, y_base - sign * 10.5, 1.2),
                (5.8, 3.6, 1.8),
                housing_gray,
            )

        for bridge_index, bridge_x in enumerate(overhead_bridge_positions):
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/OverheadGallery_{bridge_index}",
                "cube",
                (bridge_x, 2.2, 9.2),
                (9.5, 1.6, 0.26),
                walkway_black,
            )
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/OverheadGalleryRail_{bridge_index}",
                "cube",
                (bridge_x, 3.0, 10.0),
                (9.5, 0.08, 0.08),
                guardrail_yellow,
            )
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/OverheadGallerySupportA_{bridge_index}",
                "cube",
                (bridge_x - 4.2, 2.2, 4.6),
                (0.24, 0.24, 9.2),
                steel_gray,
            )
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/OverheadGallerySupportB_{bridge_index}",
                "cube",
                (bridge_x + 4.2, 2.2, 4.6),
                (0.24, 0.24, 9.2),
                steel_gray,
            )

        sorter_center_x = -half_length + 14.0
        qc_center_x = half_length - 10.0
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/OpticalBridgeBeam", "cube", (sorter_center_x, 0.0, 2.52), (5.8, 0.34, 0.32), sensor_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/OpticalBridgeLeftLeg", "cube", (sorter_center_x, -1.25, 1.56), (0.22, 0.22, 1.92), sensor_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/OpticalBridgeRightLeg", "cube", (sorter_center_x, 1.25, 1.56), (0.22, 0.22, 1.92), sensor_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/OpticalSensorBar", "cube", (sorter_center_x, 0.0, 2.06), (4.1, 0.22, 0.18), sensor_orange)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/OpticalCameraPod", "cube", (sorter_center_x - 0.9, 0.0, 2.34), (0.74, 0.46, 0.34), housing_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/OpticalEmitterPod", "cube", (sorter_center_x + 0.95, 0.0, 2.34), (0.74, 0.46, 0.34), housing_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/AirKnifeManifold", "cube", (sorter_center_x + 0.7, 1.02, 1.34), (3.5, 0.18, 0.16), sensor_gray)
        for nozzle_index in range(8):
            nozzle_x = sorter_center_x - 1.45 + nozzle_index * 0.42
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/AirNozzle_{nozzle_index}",
                "cylinder",
                (nozzle_x, 1.08, 1.18),
                (0.04, 0.04, 0.18),
                duct_silver,
                rotate=(90.0, 0.0, 0.0),
            )
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PETChute", "cube", (sorter_center_x + 1.6, 1.58, 1.18), (2.8, 0.44, 0.26), machine_green, rotate=(0.0, -18.0, 10.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PETBunker", "cube", (sorter_center_x + 2.8, 2.95, 0.8), (2.4, 1.55, 1.32), housing_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/PETBunkerFill", "cube", (sorter_center_x + 2.8, 2.95, 1.08), (1.72, 1.08, 0.52), pet_blue)

        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/QCBridgeBeam", "cube", (qc_center_x, 0.0, 2.38), (5.0, 0.28, 0.28), sensor_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/QCBridgeLeftLeg", "cube", (qc_center_x, -1.18, 1.46), (0.2, 0.2, 1.84), sensor_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/QCBridgeRightLeg", "cube", (qc_center_x, 1.18, 1.46), (0.2, 0.2, 1.84), sensor_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/NonferrousSensorPod", "cube", (qc_center_x - 0.7, 0.0, 2.16), (0.66, 0.42, 0.3), housing_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/QCManifold", "cube", (qc_center_x - 0.25, -1.0, 1.28), (2.8, 0.16, 0.14), sensor_gray)
        for nozzle_index in range(6):
            nozzle_x = qc_center_x - 1.05 + nozzle_index * 0.42
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/QCNozzle_{nozzle_index}",
                "cylinder",
                (nozzle_x, -1.04, 1.12),
                (0.035, 0.035, 0.16),
                duct_silver,
                rotate=(90.0, 0.0, 0.0),
            )
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/CanRejectChute", "cube", (qc_center_x - 0.8, -1.52, 1.08), (2.2, 0.38, 0.24), machine_green, rotate=(0.0, 18.0, -10.0))
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/CanBunker", "cube", (qc_center_x - 2.6, -3.15, 0.78), (2.1, 1.42, 1.26), housing_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/CanBunkerFill", "cube", (qc_center_x - 2.6, -3.15, 1.02), (1.48, 0.96, 0.48), metal_silver)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/FiberBunker", "cube", (sorter_center_x - 3.8, -3.05, 0.78), (2.5, 1.5, 1.24), housing_gray)
        self._add_static_visual("/World/RecyclingLine/Visuals/Segment/FiberBunkerFill", "cube", (sorter_center_x - 3.8, -3.05, 1.02), (1.86, 1.04, 0.48), fiber_brown)

        for station in plan.facility["stations"]:
            if station["capture_center"] is None:
                continue
            center_x = (station["x_range"][0] + station["x_range"][1]) / 2
            station_token = self._safe_token(station["name"])
            side = 2.7 if station["capture_center"][1] >= 0 else -2.7
            self._add_static_visual(
                f"/World/RecyclingLine/Visuals/Segment/{station_token}CaptureFrame",
                "cube",
                (center_x, side, 1.02),
                (1.8, 0.68, 0.86),
                housing_gray,
            )

    def _add_static_visual(self, path: str, shape: str, translate, scale, color, rotate=None) -> None:
        from pxr import UsdGeom  # type: ignore

        if self._stage is None:
            return
        if shape == "cube":
            prim = UsdGeom.Cube.Define(self._stage, path)
        elif shape == "cylinder":
            prim = UsdGeom.Cylinder.Define(self._stage, path)
            prim.CreateAxisAttr("Z")
        elif shape == "sphere":
            prim = UsdGeom.Sphere.Define(self._stage, path)
        else:
            return
        xformable = UsdGeom.Xformable(prim.GetPrim())
        xformable.AddTranslateOp().Set(translate)
        if rotate is not None:
            xformable.AddRotateXYZOp().Set(rotate)
        xformable.AddScaleOp().Set(self._visual_scale_for_shape(shape, scale))
        prim.CreateDisplayColorAttr([color])

    def _add_worker(self, path_root: str, position: tuple[float, float, float], shirt_color, vest_color) -> None:
        self._add_static_visual(f"{path_root}/Legs", "cube", (position[0], position[1], position[2] - 0.22), (0.16, 0.12, 0.34), (0.12, 0.12, 0.16))
        self._add_static_visual(f"{path_root}/Torso", "cube", position, (0.28, 0.18, 0.34), shirt_color)
        self._add_static_visual(f"{path_root}/Vest", "cube", (position[0], position[1] + 0.01, position[2] + 0.02), (0.3, 0.06, 0.26), vest_color)
        self._add_static_visual(f"{path_root}/Head", "sphere", (position[0], position[1], position[2] + 0.31), (0.09, 0.09, 0.09), (0.84, 0.72, 0.62))
        self._add_static_visual(f"{path_root}/Helmet", "sphere", (position[0], position[1], position[2] + 0.38), (0.1, 0.1, 0.06), (0.96, 0.9, 0.16))

    def _add_bale_stack(self, path_root: str, position: tuple[float, float, float], color) -> None:
        self._add_static_visual(f"{path_root}/Lower", "cube", position, (0.46, 0.7, 0.36), color)
        self._add_static_visual(f"{path_root}/Upper", "cube", (position[0], position[1], position[2] + 0.38), (0.42, 0.66, 0.32), color)

    def _add_perception_zone_visuals(self, plan: EpisodePlan, camera_color, beam_color) -> None:
        segment_data = {segment["id"]: segment for segment in plan.facility.get("segments", [])}
        for zone in plan.facility.get("perception_zones", []):
            segment = segment_data.get(zone["segment_id"])
            if segment is None:
                continue
            root = f"/World/RecyclingLine/Visuals/Perception/{self._safe_token(zone['id'])}"
            zone_mid = (zone["distance_range"][0] + zone["distance_range"][1]) / 2
            focus = self._point_on_segment_distance(segment, zone_mid, 0.0)
            self._add_static_visual(f"{root}/Pole", "cube", (zone["position"][0], zone["position"][1], zone["position"][2] / 2), (0.12, 0.12, zone["position"][2]), camera_color)
            self._add_static_visual(f"{root}/Head", "cube", zone["position"], (0.46, 0.28, 0.22), camera_color, rotate=(0.0, 0.0, 0.0))
            self._add_static_visual(
                f"{root}/Guide",
                "cube",
                ((zone["position"][0] + focus[0]) / 2, (zone["position"][1] + focus[1]) / 2, (zone["position"][2] + focus[2]) / 2),
                (max(abs(zone["position"][0] - focus[0]), 0.4), 0.04, max(abs(zone["position"][2] - focus[2]), 0.4)),
                beam_color,
                rotate=(0.0, 0.0, 0.0),
            )

    def _add_robot_cell_visuals(self, plan: EpisodePlan, robot_gray, arm_orange, gripper_blue) -> None:
        body_color = self._lighten_color(robot_gray, 0.2)
        joint_color = self._darken_color(robot_gray, 0.04)
        wrist_color = self._lighten_color(arm_orange, 0.08)
        for robot in plan.facility.get("robot_cells", []):
            root = f"/World/RecyclingLine/Visuals/RobotCells/{self._safe_token(robot['id'])}"
            base = tuple(robot["base_pose"]["position"])
            shoulder, elbow, wrist, tool = self._robot_arm_points(base, robot["base_pose"]["yaw_deg"], None)
            self._add_static_visual(
                f"{root}/Pedestal",
                "cube",
                (base[0], base[1], base[2] / 2),
                (0.58, 0.58, max(base[2], 0.88)),
                joint_color,
                rotate=(0.0, 0.0, 0.0),
            )
            self._add_static_visual(
                f"{root}/BaseHousing",
                "cube",
                (base[0], base[1], base[2] + 0.24),
                (0.76, 0.58, 0.46),
                body_color,
                rotate=(0.0, 0.0, robot["base_pose"]["yaw_deg"]),
            )
            self._add_static_visual(f"{root}/ShoulderJoint", "sphere", shoulder, (0.18, 0.18, 0.18), joint_color, rotate=(0.0, 0.0, 0.0))
            self._add_static_visual(f"{root}/ElbowJoint", "sphere", elbow, (0.15, 0.15, 0.15), joint_color, rotate=(0.0, 0.0, 0.0))
            self._add_static_visual(f"{root}/WristJoint", "sphere", wrist, (0.12, 0.12, 0.12), joint_color, rotate=(0.0, 0.0, 0.0))
            upper_mid, upper_scale, upper_rotate = self._arm_segment_transform(shoulder, elbow, 0.16)
            fore_mid, fore_scale, fore_rotate = self._arm_segment_transform(elbow, wrist, 0.14)
            wrist_mid, wrist_scale, wrist_rotate = self._arm_segment_transform(wrist, tool, 0.09)
            self._add_static_visual(f"{root}/UpperArm", "cube", upper_mid, upper_scale, body_color, rotate=upper_rotate)
            self._add_static_visual(f"{root}/Forearm", "cube", fore_mid, fore_scale, body_color, rotate=fore_rotate)
            self._add_static_visual(f"{root}/WristLink", "cube", wrist_mid, wrist_scale, wrist_color, rotate=wrist_rotate)
            self._add_static_visual(f"{root}/ToolHead", "sphere", tool, (0.1, 0.1, 0.1), gripper_blue, rotate=(0.0, 0.0, 0.0))

    def _update_robot_cell_animation(self, plan: EpisodePlan, current_time: float) -> None:
        if self._stage is None:
            return
        for robot in plan.facility.get("robot_cells", []):
            root = f"/World/RecyclingLine/Visuals/RobotCells/{self._safe_token(robot['id'])}"
            base = tuple(robot["base_pose"]["position"])
            active_pick = next(
                (
                    event
                    for event in plan.robot_pick_events
                    if event.robot_cell_id == robot["id"] and event.pick_start_time <= current_time <= event.place_time
                ),
                None,
            )
            gripper_target = None
            if active_pick is not None:
                lifecycle = next(
                    (value for value in plan.item_lifecycles if value.item_id == active_pick.item_id),
                    None,
                )
                if lifecycle is not None:
                    pick_surface = self._point_on_segment_distance(
                        next(segment for segment in plan.facility.get("segments", []) if segment["id"] == active_pick.source_segment_id),
                        active_pick.pick_distance,
                        float(lifecycle.spawn.lane_offset),
                    )
                    pick_position = (
                        pick_surface[0],
                        pick_surface[1],
                        pick_surface[2] + lifecycle.spawn.size[2] / 2 + 0.008,
                    )
                    gripper_target = self._robot_gripper_position(
                        robot,
                        self._robot_place_target_position(plan, active_pick, lifecycle.spawn.size[2]),
                        pick_position,
                        active_pick.pick_start_time,
                        active_pick.place_time,
                        current_time,
                    )
            shoulder, elbow, wrist, tool = self._robot_arm_points(base, robot["base_pose"]["yaw_deg"], gripper_target)
            upper_mid, upper_scale, upper_rotate = self._arm_segment_transform(shoulder, elbow, 0.16)
            fore_mid, fore_scale, fore_rotate = self._arm_segment_transform(elbow, wrist, 0.14)
            wrist_mid, wrist_scale, wrist_rotate = self._arm_segment_transform(wrist, tool, 0.09)
            self._set_static_visual_transform(f"{root}/ShoulderJoint", "sphere", shoulder, (0.18, 0.18, 0.18), (0.0, 0.0, 0.0))
            self._set_static_visual_transform(f"{root}/ElbowJoint", "sphere", elbow, (0.15, 0.15, 0.15), (0.0, 0.0, 0.0))
            self._set_static_visual_transform(f"{root}/WristJoint", "sphere", wrist, (0.12, 0.12, 0.12), (0.0, 0.0, 0.0))
            self._set_static_visual_transform(f"{root}/UpperArm", "cube", upper_mid, upper_scale, upper_rotate)
            self._set_static_visual_transform(f"{root}/Forearm", "cube", fore_mid, fore_scale, fore_rotate)
            self._set_static_visual_transform(f"{root}/WristLink", "cube", wrist_mid, wrist_scale, wrist_rotate)
            self._set_static_visual_transform(f"{root}/ToolHead", "sphere", tool, (0.1, 0.1, 0.1), (0.0, 0.0, 0.0))

    def _robot_arm_points(
        self,
        base: tuple[float, float, float],
        yaw_deg: float,
        gripper_target: tuple[float, float, float] | None,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        yaw_rad = math.radians(yaw_deg)
        shoulder = (base[0], base[1], base[2] + 0.96)
        home_gripper = (
            base[0] + math.cos(yaw_rad) * 1.28,
            base[1] + math.sin(yaw_rad) * 1.28,
            base[2] + 1.46,
        )
        tool = gripper_target or home_gripper
        wrist = (
            tool[0] - math.cos(yaw_rad) * 0.26,
            tool[1] - math.sin(yaw_rad) * 0.26,
            max(base[2] + 1.18, tool[2] + 0.1),
        )
        elbow = (
            (shoulder[0] + wrist[0]) / 2,
            (shoulder[1] + wrist[1]) / 2,
            max(shoulder[2], wrist[2]) + 0.42,
        )
        return shoulder, elbow, wrist, tool

    def _arm_segment_transform(self, start, end, thickness: float) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        mid = (
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
            (start[2] + end[2]) / 2,
        )
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        length = max(math.sqrt(dx * dx + dy * dy + dz * dz), thickness)
        yaw_deg = math.degrees(math.atan2(dy, dx))
        horizontal = math.sqrt(dx * dx + dy * dy)
        incline_deg = math.degrees(math.atan2(dz, max(horizontal, 1e-6)))
        return mid, (length, thickness, thickness), (0.0, -incline_deg, yaw_deg)

    def _set_static_visual_transform(self, path: str, shape: str, translate, scale, rotate) -> None:
        if self._stage is None:
            return
        from pxr import UsdGeom  # type: ignore

        prim = self._stage.GetPrimAtPath(path)
        if not prim:
            return
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        if len(ops) >= 3:
            ops[0].Set(translate)
            ops[1].Set(rotate)
            ops[2].Set(self._visual_scale_for_shape(shape, scale))

    def _uses_topology_layout(self) -> bool:
        return bool(self.config.conveyor_segments)

    def _topology_layout_preset(self) -> str:
        return self.config.environment.layout_preset

    def _viewport_camera_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        if self._uses_topology_layout():
            focal_length = 20.0 if self._topology_layout_preset() == "sims_big_sort_video_v2" else 18.0
            return self.config.camera.position, self.config.camera.look_at, focal_length
        return self.config.camera.position, self.config.camera.look_at, 24.0

    def _overview_viewport_camera_path(self) -> str:
        if self._uses_topology_layout():
            return "/World/Camera/WideOverviewCamera"
        return "/World/Camera/ViewportCamera"

    def _step_topology_scene(self, plan: EpisodePlan, spawned, current_time: float) -> None:
        for lifecycle in plan.item_lifecycles:
            state = spawned.get(lifecycle.item_id)
            if state is None:
                continue
            item = state["object"]
            local_position, yaw_deg, done = self._path_pose_for_lifecycle(plan, lifecycle, current_time)
            item.set_linear_velocity((0.0, 0.0, 0.0))
            item.set_world_pose(
                position=self._line_to_world(local_position),
                orientation=self._compose_yaw_orientation(math.radians(yaw_deg)),
            )
            state["captured"] = done
            if self.detections:
                self._update_detection_visual(plan, lifecycle, state, current_time)
        self._update_robot_cell_animation(plan, current_time)

    def _path_pose_for_lifecycle(self, plan: EpisodePlan, lifecycle, current_time: float) -> tuple[tuple[float, float, float], float, bool]:
        segment_map = {segment["id"]: segment for segment in plan.facility.get("segments", [])}
        robot_pick_event = lifecycle.robot_pick_event
        if robot_pick_event is not None:
            if current_time >= robot_pick_event.place_time:
                if robot_pick_event.place_segment_id is not None:
                    return self._post_pick_path_pose(plan, lifecycle, current_time)
                return self._drop_zone_pose(plan, robot_pick_event.place_drop_zone_id, lifecycle.spawn.size[2])
            if current_time >= robot_pick_event.pick_start_time:
                return self._robot_transfer_pose(plan, lifecycle, current_time)

        elapsed = max(current_time - lifecycle.spawn.spawn_time, 0.0)
        lane_offset = float(lifecycle.spawn.lane_offset)
        item_height = lifecycle.spawn.size[2]
        for segment_id in lifecycle.path_segments:
            segment = segment_map[segment_id]
            length = self._segment_length(segment)
            travel_time = length / max(segment["belt_speed"], 1e-6)
            if elapsed <= travel_time:
                t = max(0.0, min(1.0, elapsed / max(travel_time, 1e-6)))
                belt_surface = self._point_on_segment(segment, t, lane_offset)
                position = (belt_surface[0], belt_surface[1], belt_surface[2] + item_height / 2 + 0.008)
                yaw_deg = self._segment_yaw(segment)
                return position, yaw_deg, False
            elapsed -= travel_time

        drop_zone_id = lifecycle.outbound_zone_id or lifecycle.drop_zone_id
        if drop_zone_id is not None:
            return self._drop_zone_pose(plan, drop_zone_id, item_height)

        last_segment = segment_map[lifecycle.path_segments[-1]]
        end = last_segment["end_pose"]["position"]
        belt_surface_z = end[2] + last_segment.get("thickness", 0.2) / 2
        return (end[0], end[1], belt_surface_z + item_height / 2 + 0.008), self._segment_yaw(last_segment), True

    def _post_pick_path_pose(self, plan: EpisodePlan, lifecycle, current_time: float) -> tuple[tuple[float, float, float], float, bool]:
        robot_pick_event = lifecycle.robot_pick_event
        if robot_pick_event is None or robot_pick_event.place_segment_id is None:
            return self._drop_zone_pose(plan, lifecycle.drop_zone_id or "", lifecycle.spawn.size[2])

        segment_map = {segment["id"]: segment for segment in plan.facility.get("segments", [])}
        if robot_pick_event.place_segment_id not in segment_map:
            return self._drop_zone_pose(plan, robot_pick_event.place_drop_zone_id, lifecycle.spawn.size[2])
        try:
            start_index = lifecycle.path_segments.index(robot_pick_event.place_segment_id)
        except ValueError:
            return self._drop_zone_pose(plan, robot_pick_event.place_drop_zone_id, lifecycle.spawn.size[2])

        elapsed = max(current_time - robot_pick_event.place_time, 0.0)
        item_height = lifecycle.spawn.size[2]
        place_segment = segment_map[robot_pick_event.place_segment_id]
        first_remaining = max(place_segment.get("length", self._segment_length(place_segment)) - robot_pick_event.place_distance, 0.0)
        first_time = first_remaining / max(place_segment["belt_speed"], 1e-6)
        if elapsed <= first_time:
            distance = robot_pick_event.place_distance + elapsed * place_segment["belt_speed"]
            belt_surface = self._point_on_segment_distance(place_segment, distance, 0.0)
            position = (belt_surface[0], belt_surface[1], belt_surface[2] + item_height / 2 + 0.008)
            return position, self._segment_yaw(place_segment), False
        elapsed -= first_time

        for segment_id in lifecycle.path_segments[start_index + 1:]:
            segment = segment_map[segment_id]
            length = self._segment_length(segment)
            travel_time = length / max(segment["belt_speed"], 1e-6)
            if elapsed <= travel_time:
                t = max(0.0, min(1.0, elapsed / max(travel_time, 1e-6)))
                belt_surface = self._point_on_segment(segment, t, 0.0)
                position = (belt_surface[0], belt_surface[1], belt_surface[2] + item_height / 2 + 0.008)
                return position, self._segment_yaw(segment), False
            elapsed -= travel_time

        drop_zone_id = lifecycle.outbound_zone_id or lifecycle.drop_zone_id or robot_pick_event.place_drop_zone_id
        if drop_zone_id is not None:
            return self._drop_zone_pose(plan, drop_zone_id, item_height)

        last_segment = segment_map[lifecycle.path_segments[-1]]
        end = last_segment["end_pose"]["position"]
        belt_surface_z = end[2] + last_segment.get("thickness", 0.2) / 2
        return (end[0], end[1], belt_surface_z + item_height / 2 + 0.008), self._segment_yaw(last_segment), True

    def _drop_zone_pose(self, plan: EpisodePlan, drop_zone_id: str, item_height: float) -> tuple[tuple[float, float, float], float, bool]:
        for zone in plan.facility.get("drop_zones", []):
            if zone["id"] == drop_zone_id:
                pos = zone["pose"]["position"]
                return (pos[0], pos[1], pos[2] + item_height / 2), zone["pose"]["yaw_deg"], True
        return (0.0, 0.0, item_height / 2), 0.0, True

    def _robot_transfer_pose(self, plan: EpisodePlan, lifecycle, current_time: float) -> tuple[tuple[float, float, float], float, bool]:
        robot_pick_event = lifecycle.robot_pick_event
        if robot_pick_event is None:
            return self._drop_zone_pose(plan, lifecycle.drop_zone_id or "", lifecycle.spawn.size[2])

        robot = next(
            (cell for cell in plan.facility.get("robot_cells", []) if cell["id"] == robot_pick_event.robot_cell_id),
            None,
        )
        segment = next(
            (value for value in plan.facility.get("segments", []) if value["id"] == robot_pick_event.source_segment_id),
            None,
        )
        if robot is None or segment is None:
            return self._drop_zone_pose(plan, robot_pick_event.place_drop_zone_id, lifecycle.spawn.size[2])

        pick_surface = self._point_on_segment_distance(
            segment,
            robot_pick_event.pick_distance,
            float(lifecycle.spawn.lane_offset),
        )
        pick_position = (
            pick_surface[0],
            pick_surface[1],
            pick_surface[2] + lifecycle.spawn.size[2] / 2 + 0.008,
        )
        gripper_position = self._robot_gripper_position(
            robot,
            self._robot_place_target_position(plan, robot_pick_event, lifecycle.spawn.size[2]),
            pick_position,
            robot_pick_event.pick_start_time,
            robot_pick_event.place_time,
            current_time,
        )
        return gripper_position, self._segment_yaw(segment), False

    def _robot_place_target_position(self, plan: EpisodePlan, robot_pick_event, item_height: float) -> tuple[float, float, float]:
        if robot_pick_event.place_segment_id is not None:
            segment = next(
                (value for value in plan.facility.get("segments", []) if value["id"] == robot_pick_event.place_segment_id),
                None,
            )
            if segment is not None:
                surface = self._point_on_segment_distance(segment, robot_pick_event.place_distance, 0.0)
                return (surface[0], surface[1], surface[2] + item_height / 2 + 0.008)
        zone = next(
            (value for value in plan.facility.get("drop_zones", []) if value["id"] == robot_pick_event.place_drop_zone_id),
            None,
        )
        if zone is not None:
            position = zone["pose"]["position"]
            return (position[0], position[1], position[2] + item_height / 2)
        return (0.0, 0.0, item_height / 2)

    def _point_on_segment_distance(self, segment: dict, distance: float, lane_offset: float) -> tuple[float, float, float]:
        length = max(self._segment_length(segment), 1e-6)
        return self._point_on_segment(segment, max(0.0, min(1.0, distance / length)), lane_offset)

    def _robot_gripper_position(
        self,
        robot: dict,
        drop_position: tuple[float, float, float],
        pick_position: tuple[float, float, float],
        pick_start_time: float,
        place_time: float,
        current_time: float,
    ) -> tuple[float, float, float]:
        base = tuple(robot["base_pose"]["position"])
        yaw_rad = math.radians(robot["base_pose"]["yaw_deg"])
        home = (
            base[0] + math.cos(yaw_rad) * 1.1,
            base[1] + math.sin(yaw_rad) * 1.1,
            base[2] + 1.5,
        )
        hover_pick = (pick_position[0], pick_position[1], pick_position[2] + 0.55)
        carry = (
            (pick_position[0] + drop_position[0]) / 2,
            (pick_position[1] + drop_position[1]) / 2,
            max(base[2] + 1.9, pick_position[2] + 0.9, drop_position[2] + 1.2),
        )
        hover_place = (drop_position[0], drop_position[1], drop_position[2] + 0.75)
        place = (drop_position[0], drop_position[1], drop_position[2] + 0.25)
        total = max(place_time - pick_start_time, 1e-6)
        alpha = max(0.0, min(1.0, (current_time - pick_start_time) / total))
        waypoints = (
            (0.0, home),
            (0.22, hover_pick),
            (0.38, pick_position),
            (0.68, carry),
            (0.88, hover_place),
            (1.0, place),
        )
        for index in range(len(waypoints) - 1):
            start_alpha, start_point = waypoints[index]
            end_alpha, end_point = waypoints[index + 1]
            if alpha <= end_alpha:
                local = 0.0 if end_alpha == start_alpha else (alpha - start_alpha) / (end_alpha - start_alpha)
                return tuple(
                    start_point[axis] + (end_point[axis] - start_point[axis]) * local
                    for axis in range(3)
                )
        return place

    def _point_on_segment(self, segment: dict, t: float, lane_offset: float) -> tuple[float, float, float]:
        start = segment["start_pose"]["position"]
        end = segment["end_pose"]["position"]
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        z = start[2] + (end[2] - start[2]) * t + segment["thickness"] / 2
        yaw_rad = math.radians(self._segment_yaw(segment))
        x += -math.sin(yaw_rad) * lane_offset
        y += math.cos(yaw_rad) * lane_offset
        return (x, y, z)

    def _segment_length(self, segment: dict) -> float:
        start = segment["start_pose"]["position"]
        end = segment["end_pose"]["position"]
        return math.sqrt(
            (end[0] - start[0]) ** 2 +
            (end[1] - start[1]) ** 2 +
            (end[2] - start[2]) ** 2
        )

    def _segment_yaw(self, segment: dict) -> float:
        start = segment["start_pose"]["position"]
        end = segment["end_pose"]["position"]
        return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))

    def _segment_geometry(self, segment: dict) -> tuple[tuple[float, float, float], float, float, float]:
        start = segment["start_pose"]["position"]
        end = segment["end_pose"]["position"]
        center = (
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
            (start[2] + end[2]) / 2,
        )
        yaw_deg = self._segment_yaw(segment)
        horizontal = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        incline_deg = math.degrees(math.atan2(end[2] - start[2], max(horizontal, 1e-6)))
        return center, self._segment_length(segment), yaw_deg, incline_deg

    def _rotate_scene_object(self, obj, position: tuple[float, float, float], rotate_deg: tuple[float, float, float]) -> None:
        rx, ry, rz = [math.radians(value) for value in rotate_deg]
        cx, sx = math.cos(rx / 2), math.sin(rx / 2)
        cy, sy = math.cos(ry / 2), math.sin(ry / 2)
        cz, sz = math.cos(rz / 2), math.sin(rz / 2)
        quat = (
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        )
        try:
            obj.set_world_pose(position=self._line_to_world(position), orientation=quat)
        except Exception:
            pass

    def _add_conveyor_visual_assembly(self, segment: dict, machine_green, steel_gray, guardrail_yellow, walkway_black) -> None:
        root = f"/World/RecyclingLine/Visuals/Conveyors/{self._safe_token(segment['id'])}"
        center, length, yaw_deg, incline_deg = self._segment_geometry(segment)
        self._add_static_visual(f"{root}/Bed", "cube", center, (length, segment["width"], segment["thickness"]), (0.12, 0.12, 0.12), rotate=(0.0, -incline_deg, yaw_deg))
        self._add_static_visual(f"{root}/Frame", "cube", (center[0], center[1], center[2] - segment["thickness"] * 0.8), (length, segment["width"] + 0.18, segment["thickness"] * 0.45), machine_green, rotate=(0.0, -incline_deg, yaw_deg))
        for suffix, point in (("Head", segment["end_pose"]["position"]), ("Tail", segment["start_pose"]["position"])):
            self._add_static_visual(f"{root}/{suffix}Roller", "cylinder", point, (0.16, 0.16, segment["width"] + 0.08), steel_gray, rotate=(90.0, 0.0, yaw_deg))
        support_count = max(2, int(length / max(segment["support_spacing"], 0.5)) + 1)
        for idx in range(support_count):
            t = idx / max(support_count - 1, 1)
            point = self._point_on_segment(segment, t, 0.0)
            support_root = f"{root}/Supports/Support_{idx:02d}"
            self._add_static_visual(f"{support_root}/LegA", "cube", (point[0], point[1] - segment["width"] * 0.36, point[2] / 2), (0.14, 0.14, max(point[2], 0.6)), steel_gray)
            self._add_static_visual(f"{support_root}/LegB", "cube", (point[0], point[1] + segment["width"] * 0.36, point[2] / 2), (0.14, 0.14, max(point[2], 0.6)), steel_gray)
            self._add_static_visual(f"{support_root}/Brace", "cube", (point[0], point[1], point[2] * 0.55), (0.18, segment["width"] * 0.9, 0.08), steel_gray)
        if segment["sidewalls"]:
            for side in (-1.0, 1.0):
                wall_center = self._point_on_segment(segment, 0.5, side * (segment["width"] / 2 + 0.08))
                self._add_static_visual(f"{root}/Wall_{'L' if side < 0 else 'R'}", "cube", wall_center, (length, 0.08, 0.24), machine_green, rotate=(0.0, -incline_deg, yaw_deg))
        if segment["has_catwalk"]:
            side = -1.0 if segment["access_side"] == "left" else 1.0
            walk_center = self._point_on_segment(segment, 0.5, side * (segment["width"] / 2 + 0.7))
            self._add_static_visual(f"{root}/CatwalkDeck", "cube", walk_center, (length, 0.8, 0.08), walkway_black, rotate=(0.0, -incline_deg, yaw_deg))
            rail_center = self._point_on_segment(segment, 0.5, side * (segment["width"] / 2 + 1.08))
            self._add_static_visual(f"{root}/CatwalkRail", "cube", (rail_center[0], rail_center[1], rail_center[2] + 0.55), (length, 0.06, 0.06), guardrail_yellow, rotate=(0.0, -incline_deg, yaw_deg))

    def _add_platform_visual(self, platform: dict, walkway_black, guardrail_yellow, steel_gray) -> None:
        root = f"/World/RecyclingLine/Visuals/Platforms/{self._safe_token(platform['id'])}"
        self._add_static_visual(f"{root}/Deck", "cube", platform["pose"]["position"], platform["size"], walkway_black, rotate=(0.0, 0.0, platform["pose"]["yaw_deg"]))
        if platform["guard_rails"]:
            self._add_static_visual(f"{root}/RailNorth", "cube", (platform["pose"]["position"][0], platform["pose"]["position"][1] + platform["size"][1] / 2, platform["pose"]["position"][2] + 0.55), (platform["size"][0], 0.06, 0.06), guardrail_yellow, rotate=(0.0, 0.0, platform["pose"]["yaw_deg"]))
            self._add_static_visual(f"{root}/RailSouth", "cube", (platform["pose"]["position"][0], platform["pose"]["position"][1] - platform["size"][1] / 2, platform["pose"]["position"][2] + 0.55), (platform["size"][0], 0.06, 0.06), guardrail_yellow, rotate=(0.0, 0.0, platform["pose"]["yaw_deg"]))
        for side in (-1.0, 1.0):
            self._add_static_visual(f"{root}/Leg_{'L' if side < 0 else 'R'}", "cube", (platform["pose"]["position"][0] + side * platform["size"][0] * 0.3, platform["pose"]["position"][1], platform["pose"]["position"][2] / 2), (0.16, 0.16, max(platform["pose"]["position"][2], 0.8)), steel_gray)
        if platform["stairs"]:
            self._add_static_visual(f"{root}/Stair", "cube", (platform["pose"]["position"][0] - platform["size"][0] / 2 - 1.8, platform["pose"]["position"][1] - platform["size"][1] / 2 - 0.9, platform["pose"]["position"][2] / 2), (4.2, 0.72, 0.18), guardrail_yellow, rotate=(0.0, 0.0, 32.0))
        if platform["ladders"]:
            ladder_x = platform["pose"]["position"][0] + platform["size"][0] / 2 + 0.8
            self._add_static_visual(f"{root}/LadderL", "cube", (ladder_x - 0.18, platform["pose"]["position"][1], platform["pose"]["position"][2] / 2), (0.08, 0.08, platform["pose"]["position"][2]), guardrail_yellow)
            self._add_static_visual(f"{root}/LadderR", "cube", (ladder_x + 0.18, platform["pose"]["position"][1], platform["pose"]["position"][2] / 2), (0.08, 0.08, platform["pose"]["position"][2]), guardrail_yellow)

    def _add_routing_node_visual(self, node: dict, segment_data: dict[str, dict], machine_green, steel_gray, bunker_gray) -> None:
        root = f"/World/RecyclingLine/Visuals/Nodes/{self._safe_token(node['id'])}"
        position = tuple(node["pose"]["position"])
        node_type = node["node_type"]
        color = {
            "handoff": steel_gray,
            "split": machine_green,
            "merge": (0.82, 0.54, 0.18),
            "drop": bunker_gray,
        }.get(node_type, steel_gray)
        scale = {
            "handoff": (0.42, 0.42, 0.18),
            "split": (0.72, 0.72, 0.2),
            "merge": (0.66, 0.66, 0.2),
            "drop": (0.82, 0.82, 0.24),
        }.get(node_type, (0.4, 0.4, 0.18))
        self._add_static_visual(f"{root}/Body", "cube", position, scale, color, rotate=(0.0, 0.0, node["pose"]["yaw_deg"]))
        if node_type == "drop":
            self._add_static_visual(
                f"{root}/Chute",
                "cube",
                (position[0], position[1], max(position[2] - 0.65, 0.3)),
                (0.34, 0.34, 1.1),
                bunker_gray,
            )
        for segment_id in node.get("upstream_segment_ids", []) + node.get("downstream_segment_ids", []):
            segment = segment_data.get(segment_id)
            if segment is None:
                continue
            point = segment["end_pose"]["position"] if segment_id in node.get("upstream_segment_ids", []) else segment["start_pose"]["position"]
            mid = (
                (point[0] + position[0]) / 2,
                (point[1] + position[1]) / 2,
                (point[2] + position[2]) / 2,
            )
            dx = position[0] - point[0]
            dy = position[1] - point[1]
            dz = position[2] - point[2]
            length = max(math.sqrt(dx * dx + dy * dy + dz * dz), 0.1)
            yaw_deg = math.degrees(math.atan2(dy, dx))
            horizontal = math.sqrt(dx * dx + dy * dy)
            incline_deg = math.degrees(math.atan2(dz, max(horizontal, 1e-6)))
            self._add_static_visual(
                f"{root}/Link_{self._safe_token(segment_id)}",
                "cube",
                mid,
                (length, 0.08, 0.08),
                steel_gray,
                rotate=(0.0, -incline_deg, yaw_deg),
            )

    def _add_machine_visual(self, machine: dict, machine_green, hopper_gray, bunker_gray, duct_blue, accent_orange) -> None:
        root = f"/World/RecyclingLine/Visuals/Machines/{self._safe_token(machine['id'])}"
        color = hopper_gray if machine["machine_type"] == "hopper" else machine_green
        self._add_static_visual(f"{root}/Body", "cube", machine["pose"]["position"], machine["size"], color, rotate=(0.0, 0.0, machine["pose"]["yaw_deg"]))
        if machine["machine_type"] in {"screen", "trommel", "disc_screen"}:
            self._add_static_visual(f"{root}/Deck", "cube", (machine["pose"]["position"][0] + 1.8, machine["pose"]["position"][1], machine["pose"]["position"][2] + 1.2), (machine["size"][0] * 0.9, machine["size"][1] * 0.65, 0.26), machine_green, rotate=(0.0, -14.0, machine["pose"]["yaw_deg"]))
        elif machine["machine_type"] == "optical_sorter":
            self._add_static_visual(f"{root}/SensorBridge", "cube", (machine["pose"]["position"][0], machine["pose"]["position"][1], machine["pose"]["position"][2] + machine["size"][2] / 2 + 0.7), (machine["size"][0] * 0.8, 0.22, 0.18), accent_orange)
            self._add_static_visual(f"{root}/Manifold", "cube", (machine["pose"]["position"][0] + 0.8, machine["pose"]["position"][1] + 1.0, machine["pose"]["position"][2]), (machine["size"][0] * 0.55, 0.18, 0.14), duct_blue)
        elif machine["machine_type"] == "magnet":
            self._add_static_visual(f"{root}/Drum", "cylinder", (machine["pose"]["position"][0], machine["pose"]["position"][1], machine["pose"]["position"][2] + 0.9), (0.24, 0.24, machine["size"][1] * 0.9), duct_blue, rotate=(90.0, 0.0, 0.0))
        elif machine["machine_type"] == "eddy_current":
            self._add_static_visual(f"{root}/Rotor", "cylinder", (machine["pose"]["position"][0] + 0.6, machine["pose"]["position"][1], machine["pose"]["position"][2] + 0.3), (0.28, 0.28, machine["size"][1] * 0.75), accent_orange, rotate=(90.0, 0.0, 0.0))
        elif machine["machine_type"] == "baler":
            self._add_static_visual(f"{root}/Ram", "cube", (machine["pose"]["position"][0] + 1.3, machine["pose"]["position"][1], machine["pose"]["position"][2]), (machine["size"][0] * 0.32, machine["size"][1] * 0.65, machine["size"][2] * 0.35), bunker_gray)

    def _add_sims_big_sort_conveyor(self, segment: dict, belt_black, frame_white, support_gray, catwalk_gray, panel_teal, safety_yellow) -> None:
        root = f"/World/RecyclingLine/Visuals/SimsConveyors/{self._safe_token(segment['id'])}"
        center, length, yaw_deg, incline_deg = self._segment_geometry(segment)
        role = segment.get("role", "")
        enclosed_roles = {
            "tipping_meter",
            "infeed_lift",
            "liberator_spine",
            "glass_screen_feed",
            "container_spine",
            "optical_feed",
            "metals_tail",
            "export_baler",
        }

        self._add_static_visual(
            f"{root}/Belt",
            "cube",
            center,
            (length, segment["width"] * 0.96, max(segment["thickness"] * 0.68, 0.12)),
            belt_black,
            rotate=(0.0, -incline_deg, yaw_deg),
        )
        self._add_static_visual(
            f"{root}/Underframe",
            "cube",
            (center[0], center[1], center[2] - segment["thickness"] * 0.55),
            (length, segment["width"] + 0.14, max(segment["thickness"] * 0.2, 0.08)),
            support_gray,
            rotate=(0.0, -incline_deg, yaw_deg),
        )
        for side in (-1.0, 1.0):
            stringer = self._point_on_segment(segment, 0.5, side * (segment["width"] / 2 + 0.06))
            self._add_static_visual(
                f"{root}/Stringer_{'L' if side < 0 else 'R'}",
                "cube",
                stringer,
                (length, 0.06, 0.18),
                frame_white,
                rotate=(0.0, -incline_deg, yaw_deg),
            )

        if role in enclosed_roles:
            hood_height = 0.58 if length < 16.0 else 0.72
            self._add_static_visual(
                f"{root}/Hood",
                "cube",
                (center[0], center[1], center[2] + hood_height * 0.35),
                (length * 0.94, segment["width"] + 0.44, hood_height),
                frame_white,
                rotate=(0.0, -incline_deg, yaw_deg),
            )
            service_panel = self._point_on_segment(segment, 0.52, segment["width"] / 2 + 0.18)
            self._add_static_visual(
                f"{root}/ServicePanel",
                "cube",
                (service_panel[0], service_panel[1], service_panel[2] + hood_height * 0.08),
                (max(length * 0.22, 1.2), 0.05, hood_height * 0.6),
                panel_teal,
                rotate=(0.0, -incline_deg, yaw_deg),
            )
        elif segment["sidewalls"]:
            for side in (-1.0, 1.0):
                rail = self._point_on_segment(segment, 0.5, side * (segment["width"] / 2 + 0.04))
                self._add_static_visual(
                    f"{root}/Rail_{'L' if side < 0 else 'R'}",
                    "cube",
                    rail,
                    (length, 0.05, 0.2),
                    frame_white,
                    rotate=(0.0, -incline_deg, yaw_deg),
                )

        for suffix, point in (("Head", segment["end_pose"]["position"]), ("Tail", segment["start_pose"]["position"])):
            self._add_static_visual(
                f"{root}/{suffix}Drum",
                "cylinder",
                point,
                (0.14, 0.14, segment["width"] + 0.06),
                support_gray,
                rotate=(90.0, 0.0, yaw_deg),
            )

        support_count = max(2, int(length / max(segment["support_spacing"], 0.5)) + 1)
        for idx in range(support_count):
            t = idx / max(support_count - 1, 1)
            point = self._point_on_segment(segment, t, 0.0)
            support_root = f"{root}/Supports/Support_{idx:02d}"
            height = max(point[2] - 0.02, 0.6)
            self._add_static_visual(
                f"{support_root}/LegL",
                "cube",
                (point[0], point[1] - segment["width"] * 0.3, height / 2),
                (0.09, 0.09, height),
                support_gray,
            )
            self._add_static_visual(
                f"{support_root}/LegR",
                "cube",
                (point[0], point[1] + segment["width"] * 0.3, height / 2),
                (0.09, 0.09, height),
                support_gray,
            )
            self._add_static_visual(
                f"{support_root}/Cap",
                "cube",
                (point[0], point[1], max(point[2] - 0.18, 0.2)),
                (0.16, segment["width"] * 0.8, 0.06),
                support_gray,
            )

        if segment["has_catwalk"]:
            side = -1.0 if segment["access_side"] == "left" else 1.0
            deck = self._point_on_segment(segment, 0.5, side * (segment["width"] / 2 + 0.76))
            rail = self._point_on_segment(segment, 0.5, side * (segment["width"] / 2 + 1.08))
            self._add_static_visual(
                f"{root}/CatwalkDeck",
                "cube",
                deck,
                (length, 0.78, 0.08),
                catwalk_gray,
                rotate=(0.0, -incline_deg, yaw_deg),
            )
            self._add_static_visual(
                f"{root}/CatwalkRail",
                "cube",
                (rail[0], rail[1], rail[2] + 0.55),
                (length, 0.05, 0.06),
                safety_yellow,
                rotate=(0.0, -incline_deg, yaw_deg),
            )

    def _add_sims_big_sort_node(self, node: dict, segment_data: dict[str, dict], frame_white, support_gray, panel_teal, bunker_gray) -> None:
        root = f"/World/RecyclingLine/Visuals/SimsNodes/{self._safe_token(node['id'])}"
        position = tuple(node["pose"]["position"])
        node_type = node["node_type"]
        color = {
            "handoff": support_gray,
            "split": panel_teal,
            "merge": frame_white,
            "drop": bunker_gray,
        }.get(node_type, support_gray)
        scale = {
            "handoff": (0.32, 0.32, 0.14),
            "split": (0.46, 0.46, 0.16),
            "merge": (0.46, 0.46, 0.16),
            "drop": (0.62, 0.62, 0.2),
        }.get(node_type, (0.32, 0.32, 0.14))
        self._add_static_visual(f"{root}/Body", "cube", position, scale, color, rotate=(0.0, 0.0, node["pose"]["yaw_deg"]))
        if node_type == "drop":
            self._add_static_visual(
                f"{root}/Chute",
                "cube",
                (position[0], position[1], max(position[2] - 0.56, 0.24)),
                (0.24, 0.24, 0.96),
                support_gray,
            )
        for segment_id in node.get("upstream_segment_ids", []) + node.get("downstream_segment_ids", []):
            segment = segment_data.get(segment_id)
            if segment is None:
                continue
            point = segment["end_pose"]["position"] if segment_id in node.get("upstream_segment_ids", []) else segment["start_pose"]["position"]
            mid = (
                (point[0] + position[0]) / 2,
                (point[1] + position[1]) / 2,
                (point[2] + position[2]) / 2,
            )
            dx = position[0] - point[0]
            dy = position[1] - point[1]
            dz = position[2] - point[2]
            length = max(math.sqrt(dx * dx + dy * dy + dz * dz), 0.1)
            yaw_deg = math.degrees(math.atan2(dy, dx))
            horizontal = math.sqrt(dx * dx + dy * dy)
            incline_deg = math.degrees(math.atan2(dz, max(horizontal, 1e-6)))
            self._add_static_visual(
                f"{root}/Link_{self._safe_token(segment_id)}",
                "cube",
                mid,
                (length, 0.05, 0.05),
                support_gray,
                rotate=(0.0, -incline_deg, yaw_deg),
            )

    def _add_sims_big_sort_machine(self, machine: dict, frame_white, support_gray, dark_gray, glass_blue, panel_teal, safety_yellow) -> None:
        root = f"/World/RecyclingLine/Visuals/SimsMachines/{self._safe_token(machine['id'])}"
        position = machine["pose"]["position"]
        size = machine["size"]
        yaw_deg = machine["pose"]["yaw_deg"]
        machine_type = machine["machine_type"]
        body_color = dark_gray if machine_type == "hopper" else frame_white

        self._add_static_visual(f"{root}/Body", "cube", position, size, body_color, rotate=(0.0, 0.0, yaw_deg))
        self._add_static_visual(
            f"{root}/Base",
            "cube",
            (position[0], position[1], max(position[2] - size[2] * 0.34, 0.2)),
            (size[0] * 0.96, size[1] * 0.92, max(size[2] * 0.18, 0.18)),
            support_gray,
            rotate=(0.0, 0.0, yaw_deg),
        )

        if machine_type in {"screen", "disc_screen"}:
            self._add_static_visual(
                f"{root}/AccessHood",
                "cube",
                (position[0] + size[0] * 0.1, position[1], position[2] + size[2] * 0.32),
                (size[0] * 0.78, size[1] * 0.72, size[2] * 0.3),
                frame_white,
                rotate=(0.0, -10.0, yaw_deg),
            )
            self._add_static_visual(
                f"{root}/Panel",
                "cube",
                (position[0], position[1] + size[1] * 0.46, position[2]),
                (size[0] * 0.36, 0.05, size[2] * 0.5),
                panel_teal,
                rotate=(0.0, 0.0, yaw_deg),
            )
        elif machine_type == "trommel":
            self._add_static_visual(
                f"{root}/Drum",
                "cylinder",
                (position[0], position[1], position[2] + 0.1),
                (size[1] * 0.2, size[1] * 0.2, size[0] * 0.78),
                support_gray,
                rotate=(0.0, 90.0, yaw_deg),
            )
        elif machine_type == "optical_sorter":
            self._add_static_visual(
                f"{root}/SensorWindow",
                "cube",
                (position[0], position[1] + size[1] * 0.34, position[2] + size[2] * 0.18),
                (size[0] * 0.52, 0.04, size[2] * 0.28),
                glass_blue,
                rotate=(0.0, 0.0, yaw_deg),
            )
            self._add_static_visual(
                f"{root}/Manifold",
                "cube",
                (position[0] + size[0] * 0.08, position[1] - size[1] * 0.36, position[2]),
                (size[0] * 0.44, 0.14, 0.12),
                safety_yellow,
                rotate=(0.0, 0.0, yaw_deg),
            )
        elif machine_type == "magnet":
            self._add_static_visual(
                f"{root}/Drum",
                "cylinder",
                (position[0], position[1], position[2] + size[2] * 0.18),
                (0.24, 0.24, size[1] * 0.72),
                support_gray,
                rotate=(90.0, 0.0, yaw_deg),
            )
        elif machine_type == "eddy_current":
            self._add_static_visual(
                f"{root}/Rotor",
                "cylinder",
                (position[0] + size[0] * 0.16, position[1], position[2] + 0.18),
                (0.24, 0.24, size[1] * 0.64),
                safety_yellow,
                rotate=(90.0, 0.0, yaw_deg),
            )
        elif machine_type == "baler":
            self._add_static_visual(
                f"{root}/Ram",
                "cube",
                (position[0] + size[0] * 0.16, position[1], position[2] - size[2] * 0.08),
                (size[0] * 0.36, size[1] * 0.58, size[2] * 0.26),
                dark_gray,
                rotate=(0.0, 0.0, yaw_deg),
            )

    def _add_dense_camera_visuals(self, UsdGeom, guardrail_yellow, steel_gray) -> None:
        if self._stage is None:
            return
        mounts = [
            ("OverviewWide", self.config.camera.position),
            ("Overhead", self._overhead_camera_pose()[0]),
        ]
        for name, position in mounts:
            pole_height = max(position[2] - 0.4, 1.0)
            root = f"/World/RecyclingLine/Visuals/Cameras/{name}"
            self._add_static_visual(f"{root}/Pole", "cube", (position[0], position[1], pole_height / 2), (0.12, 0.12, pole_height), steel_gray)
            self._add_static_visual(f"{root}/Head", "cube", position, (0.44, 0.22, 0.18), guardrail_yellow)

    def _define_dense_aux_cameras(self, UsdGeom, Gf) -> None:
        if self._stage is None:
            return
        if self.config.environment.layout_preset == "sims_big_sort_video_v2":
            specs = (
                ("/World/Camera/WideOverviewCamera", self.config.camera.position, self.config.camera.look_at, 20.0),
                ("/World/Camera/InfeedHeroCamera", (-54.0, 12.0, 8.5), (-38.0, 6.5, 4.8), 26.0),
                ("/World/Camera/SortLineHeroCamera", (18.0, -7.0, 11.0), (34.0, 1.5, 8.4), 24.0),
            )
        else:
            specs = (
                ("/World/Camera/WideOverviewCamera", self.config.camera.position, self.config.camera.look_at, 18.0),
                ("/World/Camera/InfeedHeroCamera", (-40.0, 14.0, 8.0), (-32.0, 8.0, 5.5), 28.0),
                ("/World/Camera/SortLineHeroCamera", (8.0, -8.0, 10.0), (18.0, 2.0, 8.0), 24.0),
            )
        for path, position, look_at, focal in specs:
            camera = UsdGeom.Camera.Define(self._stage, path)
            xformable = UsdGeom.Xformable(camera.GetPrim())
            xformable.AddTranslateOp().Set(position)
            orientation = self._camera_orientation(position, look_at)
            quaternion = Gf.Quatf(
                float(orientation[0]),
                Gf.Vec3f(float(orientation[1]), float(orientation[2]), float(orientation[3])),
            )
            xformable.AddOrientOp().Set(quaternion)
            camera.CreateFocalLengthAttr(focal)

    def _load_environment(self, add_reference_to_stage, get_assets_root_path) -> None:
        assets_root_path = get_assets_root_path()
        if not assets_root_path:
            raise IsaacSimUnavailableError("Could not resolve Isaac Sim assets root for warehouse environment.")
        if self.config.environment.environment_id == "simple_warehouse":
            usd_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        else:
            raise IsaacSimUnavailableError(f"Unsupported environment id: {self.config.environment.environment_id}")
        add_reference_to_stage(usd_path=usd_path, prim_path="/World/Environment")

    def _define_recycling_root(self, UsdGeom, Gf) -> None:
        if self._stage is None:
            return
        root = UsdGeom.Xform.Define(self._stage, "/World/RecyclingLine")
        xformable = UsdGeom.Xformable(root.GetPrim())
        xformable.AddTranslateOp().Set(self.config.environment.conveyor_transform.position)
        yaw_rad = math.radians(self.config.environment.conveyor_transform.yaw_deg)
        quat = Gf.Quatf(float(math.cos(yaw_rad / 2.0)), Gf.Vec3f(0.0, 0.0, float(math.sin(yaw_rad / 2.0))))
        xformable.AddOrientOp().Set(quat)

    def _add_camera_rig(self, world, FixedCuboid) -> None:
        pole_height = max(self.config.camera.position[2] * 0.95, 0.5)
        pole = world.scene.add(
            FixedCuboid(
                prim_path="/World/Camera/Rig/Pole",
                name="camera_rig_pole",
                position=(self.config.camera.position[0], self.config.camera.position[1], pole_height / 2),
                scale=(0.08, 0.08, pole_height),
                color=self._color_array((0.24, 0.24, 0.26)),
            )
        )
        self._disable_collisions(pole)
        head = world.scene.add(
            FixedCuboid(
                prim_path="/World/Camera/Rig/Head",
                name="camera_rig_head",
                position=(
                    self.config.camera.position[0],
                    self.config.camera.position[1],
                    self.config.camera.position[2] - 0.12,
                ),
                scale=(0.24, 0.18, 0.14),
                color=self._color_array((0.16, 0.16, 0.18)),
            )
        )
        self._disable_collisions(head)

    def _define_viewport_camera(self, UsdGeom, Gf) -> None:
        if self._stage is None:
            return
        camera = UsdGeom.Camera.Define(self._stage, "/World/Camera/ViewportCamera")
        xformable = UsdGeom.Xformable(camera.GetPrim())
        xformable.AddTranslateOp().Set(self.config.camera.position)
        orientation = self._camera_orientation(self.config.camera.position, self.config.camera.look_at)
        quaternion = Gf.Quatf(float(orientation[0]), Gf.Vec3f(float(orientation[1]), float(orientation[2]), float(orientation[3])))
        xformable.AddOrientOp().Set(quaternion)
        camera.CreateFocalLengthAttr(24.0)

    def _define_overhead_camera(self, UsdGeom, Gf) -> None:
        if self._stage is None:
            return
        position, look_at = self._overhead_camera_pose()
        camera = UsdGeom.Camera.Define(self._stage, "/World/Camera/OverheadCamera")
        xformable = UsdGeom.Xformable(camera.GetPrim())
        xformable.AddTranslateOp().Set(position)
        orientation = self._camera_orientation(position, look_at)
        quaternion = Gf.Quatf(
            float(orientation[0]),
            Gf.Vec3f(float(orientation[1]), float(orientation[2]), float(orientation[3])),
        )
        xformable.AddOrientOp().Set(quaternion)
        camera.CreateFocalLengthAttr(18.0)

    def _run_episode_loop(
        self,
        plan: EpisodePlan,
        spawned,
        DynamicCuboid,
        DynamicCylinder,
        world,
        camera,
        frame_times,
        frames_dir,
    ) -> None:
        current_time = 0.0
        while current_time <= self.config.episode_duration + self.config.physics_dt:
            self._step_scene(plan, spawned, DynamicCuboid, DynamicCylinder, world, current_time)
            rounded_time = round(current_time, 4)
            if rounded_time in frame_times:
                frame = camera.get_rgba()
                stem = frames_dir / f"frame_{rounded_time:07.4f}"
                self._write_frame(frame, stem)
            current_time += self.config.physics_dt

    def _run_looping_cycles(
        self,
        base_plan: EpisodePlan,
        DynamicCuboid,
        DynamicCylinder,
        world,
        simulation_app,
        camera,
        frame_times,
        frames_dir,
    ) -> None:
        cycle_index = 0
        while self._loop_is_running(simulation_app):
            plan = base_plan if cycle_index == 0 else self._generate_plan(cycle_index)
            print(f"Loop cycle {cycle_index + 1}", flush=True)
            spawned: dict[str, dict[str, object]] = {}
            self._run_single_cycle(
                plan,
                cycle_index,
                spawned,
                DynamicCuboid,
                DynamicCylinder,
                world,
                simulation_app,
                camera,
                frame_times,
                frames_dir,
            )
            self._cleanup_cycle_items(world, spawned)
            if self._loop_is_running(simulation_app):
                self._reset_world_for_next_cycle(world)
            cycle_index += 1

    def _run_single_cycle(
        self,
        plan: EpisodePlan,
        cycle_index: int,
        spawned,
        DynamicCuboid,
        DynamicCylinder,
        world,
        simulation_app,
        camera,
        frame_times,
        frames_dir,
    ) -> None:
        current_time = 0.0
        while current_time <= self.loop_cycle_seconds and self._loop_is_running(simulation_app):
            self._step_scene(plan, spawned, DynamicCuboid, DynamicCylinder, world, current_time)
            if self.headless and camera is not None:
                rounded_time = round(current_time, 4)
                if rounded_time in frame_times:
                    stem = frames_dir / f"cycle_{cycle_index:04d}_{rounded_time:07.4f}"
                    self._write_frame(camera.get_rgba(), stem)
            current_time += self.config.physics_dt

    def _run_interactive_loop(
        self,
        plan: EpisodePlan,
        spawned,
        DynamicCuboid,
        DynamicCylinder,
        world,
        simulation_app,
    ) -> None:
        current_time = 0.0
        while simulation_app.is_running() and not simulation_app.is_exiting():
            self._step_scene(plan, spawned, DynamicCuboid, DynamicCylinder, world, current_time)
            current_time += self.config.physics_dt

    def _loop_is_running(self, simulation_app) -> bool:
        if simulation_app is None:
            return True
        return simulation_app.is_running() and not simulation_app.is_exiting()

    def _cleanup_cycle_items(self, world, spawned) -> None:
        if self._stage is None:
            spawned.clear()
            return
        try:
            world.stop()
        except Exception:
            pass
        for item_id in list(spawned.keys()):
            path = f"/World/RecyclingLine/Items/{item_id}"
            try:
                world.scene.remove_object(item_id)
            except Exception:
                pass
            if self._stage.GetPrimAtPath(path):
                self._stage.RemovePrim(path)
        spawned.clear()

    def _reset_world_for_next_cycle(self, world) -> None:
        world.reset()
        world.play()

    def _step_scene(self, plan: EpisodePlan, spawned, DynamicCuboid, DynamicCylinder, world, current_time: float) -> None:
        rounded_time = round(current_time, 4)

        for lifecycle in plan.item_lifecycles:
            event = lifecycle.spawn
            if event.item_id not in spawned and rounded_time >= event.spawn_time:
                dynamic_item = world.scene.add(self._build_dynamic_item(event, DynamicCuboid, DynamicCylinder))
                if not self.bare_bones:
                    self._decorate_item_visual(event)
                if self.detections:
                    self._decorate_detection_visual(lifecycle)
                loop_birth_time = event.spawn_time
                spawned[event.item_id] = {
                    "object": dynamic_item,
                    "captured": False,
                    "loop_birth_time": loop_birth_time,
                    "spawn_event": event,
                    "lifecycle": lifecycle,
                }
                angle = math.radians(event.yaw_deg)
                spawned[event.item_id]["object"].set_world_pose(
                    position=self._line_to_world(self._spawn_position(event)),
                    orientation=self._compose_yaw_orientation(angle),
                )

        if self._uses_topology_layout():
            self._step_topology_scene(plan, spawned, rounded_time)
            world.step(render=True)
            return

        for lifecycle in plan.item_lifecycles:
            state = spawned.get(lifecycle.item_id)
            if state is None:
                continue
            item = state["object"]
            if state["captured"]:
                item.set_linear_velocity((0.0, 0.0, 0.0))
                continue

            capture_event = next((station for station in lifecycle.route if station.decision == "captured"), None)
            if capture_event is not None and rounded_time >= capture_event.enter_time:
                item.set_linear_velocity((0.0, 0.0, 0.0))
                item.set_world_pose(position=self._line_to_world(self._capture_pose(plan, capture_event.station_name, lifecycle.spawn.position)))
                state["captured"] = True
                continue

            if self._use_recirculating_loop():
                exit_time = estimate_belt_exit_time(self.config, lifecycle.spawn)
                travel_duration = max(exit_time - lifecycle.spawn.spawn_time, self.config.physics_dt)
                if rounded_time >= float(state["loop_birth_time"]) + travel_duration:
                    state["loop_birth_time"] = rounded_time
                    item.set_world_pose(
                        position=self._line_to_world(self._spawn_position(lifecycle.spawn)),
                        orientation=self._compose_yaw_orientation(math.radians(lifecycle.spawn.yaw_deg)),
                    )
                    item.set_linear_velocity((0.0, 0.0, 0.0))
                item.set_linear_velocity(self._line_velocity_to_world((self.config.main_belt.speed, 0.0, 0.0)))
            elif rounded_time <= estimate_belt_exit_time(self.config, lifecycle.spawn):
                item.set_linear_velocity(self._line_velocity_to_world((self.config.main_belt.speed, 0.0, 0.0)))
            else:
                item.set_linear_velocity((0.0, 0.0, 0.0))

            if self.detections:
                self._update_detection_visual(plan, lifecycle, state, rounded_time)

        world.step(render=True)

    def _capture_pose(self, plan: EpisodePlan, station_name: str, original_position: tuple[float, float, float]) -> tuple[float, float, float]:
        for station in plan.facility["stations"]:
            if station["name"] == station_name:
                center = station["capture_center"]
                if center is None:
                    return original_position
                return (center[0], center[1], center[2] + original_position[2] / 3)
        return original_position

    def _spawn_position(self, event) -> tuple[float, float, float]:
        if not self._use_recirculating_loop():
            return event.position
        return (
            event.position[0],
            event.position[1],
            self.config.main_belt.height / 2 + event.size[2] / 2 + 0.01,
        )

    def _disable_collisions(self, obj) -> None:
        try:
            obj.set_collision_enabled(False)
        except AttributeError:
            pass

    def _build_dynamic_item(self, event, DynamicCuboid, DynamicCylinder):
        prim_path = f"/World/RecyclingLine/Items/{event.item_id}"
        common = {
            "prim_path": prim_path,
            "name": event.item_id,
            "position": self._line_to_world(event.position),
            "color": self._color_array(event.color),
            "mass": event.mass,
        }
        if event.shape == "cylinder" and DynamicCylinder is not None:
            return DynamicCylinder(
                radius=max(event.size[0], event.size[1]) / 2,
                height=event.size[2],
                **common,
            )
        return DynamicCuboid(scale=event.size, **common)

    def _decorate_item_visual(self, event) -> None:
        if self._stage is None:
            return

        root_path = f"/World/RecyclingLine/Items/{event.item_id}"
        size = event.size
        profile = event.visual_profile
        if profile == "auto":
            return
        if profile == "bottle":
            radius = max(size[0], size[1]) * 0.22
            self._add_visual_part(root_path, "Shoulder", "cone", (0.0, 0.0, size[2] * 0.2), (radius * 0.95, radius * 0.95, size[2] * 0.18), event.color)
            self._add_visual_part(root_path, "Neck", "cylinder", (0.0, 0.0, size[2] * 0.38), (radius * 0.42, radius * 0.42, size[2] * 0.16), self._lighten_color(event.color, 0.06))
            self._add_visual_part(root_path, "Cap", "cylinder", (0.0, 0.0, size[2] * 0.54), (radius * 0.46, radius * 0.46, size[2] * 0.04), (0.92, 0.92, 0.94))
            return
        if profile == "jug":
            radius = max(size[0], size[1]) * 0.24
            self._add_visual_part(root_path, "Shoulder", "cone", (0.0, 0.0, size[2] * 0.18), (radius, radius, size[2] * 0.18), event.color)
            self._add_visual_part(root_path, "Neck", "cylinder", (0.0, 0.0, size[2] * 0.38), (radius * 0.4, radius * 0.4, size[2] * 0.14), self._lighten_color(event.color, 0.05))
            self._add_visual_part(root_path, "Handle", "cube", (size[0] * 0.42, 0.0, size[2] * 0.08), (size[0] * 0.12, size[1] * 0.12, size[2] * 0.22), (0.94, 0.94, 0.96))
            return
        if profile == "can":
            radius = max(size[0], size[1]) * 0.48
            lip_color = self._lighten_color(event.color, 0.12)
            self._add_visual_part(root_path, "TopLip", "cylinder", (0.0, 0.0, size[2] * 0.48), (radius, radius, size[2] * 0.03), lip_color)
            self._add_visual_part(root_path, "BottomLip", "cylinder", (0.0, 0.0, -size[2] * 0.48), (radius, radius, size[2] * 0.03), lip_color)
            self._add_visual_part(root_path, "LabelBand", "cylinder", (0.0, 0.0, 0.0), (radius * 0.98, radius * 0.98, size[2] * 0.22), self._darken_color(event.color, 0.08))
            return
        if profile == "carton":
            flap_color = self._lighten_color(event.color, 0.08)
            self._add_visual_part(root_path, "LeftFlap", "cube", (-size[0] * 0.16, 0.0, size[2] * 0.52), (size[0] * 0.18, size[1] * 0.48, size[2] * 0.08), flap_color, rotate=(0.0, 28.0, 0.0))
            self._add_visual_part(root_path, "RightFlap", "cube", (size[0] * 0.16, 0.0, size[2] * 0.52), (size[0] * 0.18, size[1] * 0.48, size[2] * 0.08), flap_color, rotate=(0.0, -28.0, 0.0))
            self._add_visual_part(root_path, "Spout", "cylinder", (size[0] * 0.1, 0.0, size[2] * 0.7), (size[0] * 0.08, size[0] * 0.08, size[2] * 0.05), (0.9, 0.22, 0.18))
            return
        if profile == "paper_stack":
            sheet_color = self._lighten_color(event.color, 0.04)
            self._add_visual_part(root_path, "SheetTop", "cube", (0.02, -0.01, size[2] * 0.56), (size[0] * 0.48, size[1] * 0.46, size[2] * 0.08), sheet_color, rotate=(0.0, 0.0, 6.0))
            self._add_visual_part(root_path, "SheetMid", "cube", (-0.01, 0.02, size[2] * 0.5), (size[0] * 0.48, size[1] * 0.46, size[2] * 0.08), sheet_color, rotate=(0.0, 0.0, -4.0))
            return
        if profile == "bag":
            bag_color = self._lighten_color(event.color, 0.03)
            self._add_visual_part(root_path, "FoldA", "cube", (-size[0] * 0.18, 0.0, size[2] * 0.24), (size[0] * 0.2, size[1] * 0.32, size[2] * 0.16), bag_color, rotate=(0.0, 0.0, 18.0))
            self._add_visual_part(root_path, "FoldB", "cube", (size[0] * 0.2, 0.0, -size[2] * 0.08), (size[0] * 0.18, size[1] * 0.28, size[2] * 0.14), bag_color, rotate=(0.0, 0.0, -15.0))
            return
        if profile == "chunk":
            chunk_color = self._lighten_color(event.color, 0.05)
            self._add_visual_part(root_path, "PieceA", "sphere", (-size[0] * 0.18, 0.0, size[2] * 0.14), (size[0] * 0.18, size[0] * 0.18, size[0] * 0.18), chunk_color)
            self._add_visual_part(root_path, "PieceB", "sphere", (size[0] * 0.2, size[1] * 0.06, -size[2] * 0.08), (size[0] * 0.16, size[0] * 0.16, size[0] * 0.16), chunk_color)

    def _decorate_detection_visual(self, lifecycle) -> None:
        event = lifecycle.spawn
        if self._stage is None or not lifecycle.perception_events:
            return
        from pxr import UsdGeom  # type: ignore

        root_path = f"/World/RecyclingLine/Items/{event.item_id}/Detection"
        UsdGeom.Xform.Define(self._stage, root_path)
        commodity = lifecycle.perception_events[-1].predicted_commodity
        frame_color = {
            "pet": (0.2, 0.76, 0.98),
            "hdpe": (0.92, 0.92, 0.96),
            "plastic_film": (0.96, 0.74, 0.32),
        }.get(commodity, (0.94, 0.78, 0.22))
        badge_color = self._darken_color(frame_color, 0.12)
        sx, sy, sz = event.size
        margin = 0.05
        box_x = sx + margin
        box_y = sy + margin
        box_z = sz + margin
        z_mid = 0.0
        z_top = box_z / 2
        z_bottom = -box_z / 2
        x_left = -box_x / 2
        x_right = box_x / 2
        y_front = box_y / 2
        y_back = -box_y / 2
        rail = 0.012

        for name, translate, scale in (
            ("TopFront", (0.0, y_front, z_top), (box_x, rail, rail)),
            ("TopBack", (0.0, y_back, z_top), (box_x, rail, rail)),
            ("TopLeft", (x_left, 0.0, z_top), (rail, box_y, rail)),
            ("TopRight", (x_right, 0.0, z_top), (rail, box_y, rail)),
            ("BottomFront", (0.0, y_front, z_bottom), (box_x, rail, rail)),
            ("BottomBack", (0.0, y_back, z_bottom), (box_x, rail, rail)),
            ("BottomLeft", (x_left, 0.0, z_bottom), (rail, box_y, rail)),
            ("BottomRight", (x_right, 0.0, z_bottom), (rail, box_y, rail)),
            ("PostFL", (x_left, y_front, z_mid), (rail, rail, box_z)),
            ("PostFR", (x_right, y_front, z_mid), (rail, rail, box_z)),
            ("PostBL", (x_left, y_back, z_mid), (rail, rail, box_z)),
            ("PostBR", (x_right, y_back, z_mid), (rail, rail, box_z)),
        ):
            self._add_visual_part(root_path, name, "cube", translate, scale, frame_color)
        self._add_visual_part(root_path, "BadgePlate", "cube", (0.0, 0.0, z_top + 0.08), (box_x * 0.65, box_y * 0.28, 0.03), badge_color)
        self._add_visual_part(root_path, "Beacon", "sphere", (0.0, 0.0, z_top + 0.13), (0.035, 0.035, 0.035), frame_color)
        self._set_detection_visibility(event.item_id, visible=False)

    def _add_visual_part(self, root_path: str, name: str, shape: str, translate, scale, color, rotate=None) -> None:
        from pxr import UsdGeom  # type: ignore

        if self._stage is None:
            return
        path = f"{root_path}/Visuals/{name}"
        if shape == "cube":
            prim = UsdGeom.Cube.Define(self._stage, path)
        elif shape == "cylinder":
            prim = UsdGeom.Cylinder.Define(self._stage, path)
            prim.CreateAxisAttr("Z")
        elif shape == "cone":
            prim = UsdGeom.Cone.Define(self._stage, path)
            prim.CreateAxisAttr("Z")
        elif shape == "sphere":
            prim = UsdGeom.Sphere.Define(self._stage, path)
        else:
            return
        xformable = UsdGeom.Xformable(prim.GetPrim())
        xformable.AddTranslateOp().Set(translate)
        if rotate is not None:
            xformable.AddRotateXYZOp().Set(rotate)
        xformable.AddScaleOp().Set(self._visual_scale_for_shape(shape, scale))
        prim.CreateDisplayColorAttr([color])

    def _visual_scale_for_shape(self, shape: str, scale) -> tuple[float, float, float]:
        if shape == "cube":
            return (scale[0] / 2, scale[1] / 2, scale[2] / 2)
        if shape in {"cylinder", "cone"}:
            return (scale[0], scale[1], scale[2] / 2)
        if shape == "sphere":
            return scale
        return scale

    def _update_detection_visual(self, plan: EpisodePlan, lifecycle, state, current_time: float) -> None:
        event = state.get("spawn_event")
        if event is None or not lifecycle.perception_events:
            return
        start_time = lifecycle.perception_events[0].timestamp
        end_time = (
            lifecycle.robot_pick_event.pick_start_time
            if lifecycle.robot_pick_event is not None
            else lifecycle.perception_events[-1].timestamp + 0.35
        )
        self._set_detection_visibility(event.item_id, start_time <= current_time <= end_time)

    def _current_item_local_x(self, event, state, current_time: float) -> float | None:
        if state.get("captured"):
            return None
        if self._use_recirculating_loop():
            loop_birth_time = float(state.get("loop_birth_time", event.spawn_time))
            return event.position[0] + self.config.main_belt.speed * max(current_time - loop_birth_time, 0.0)
        return event.position[0] + self.config.main_belt.speed * max(current_time - event.spawn_time, 0.0)

    def _set_detection_visibility(self, item_id: str, visible: bool) -> None:
        if self._stage is None:
            return
        from pxr import UsdGeom  # type: ignore

        prim = self._stage.GetPrimAtPath(f"/World/RecyclingLine/Items/{item_id}/Detection")
        if not prim:
            return
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _camera_orientation(
        self,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> tuple[float, float, float, float]:
        forward = (
            target[0] - position[0],
            target[1] - position[1],
            target[2] - position[2],
        )
        norm = math.sqrt(sum(component * component for component in forward))
        if norm == 0:
            return (1.0, 0.0, 0.0, 0.0)
        forward = tuple(component / norm for component in forward)
        yaw = math.atan2(forward[1], forward[0])
        pitch = math.atan2(-forward[2], math.sqrt(forward[0] ** 2 + forward[1] ** 2))
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        return (
            cy * cp,
            -sy * sp,
            sy * cp,
            cy * sp,
        )

    def _overhead_camera_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        belt_length = self.config.main_belt.length
        belt_width = self.config.main_belt.width
        if self._uses_topology_layout():
            if self.config.environment.layout_preset == "sims_big_sort_video_v2":
                return ((30.0, -20.0, 18.0), (34.0, 2.0, 6.8))
            return ((10.0, -18.0, 16.0), (12.0, 2.5, 5.2))
        if self.config.environment.layout_preset == "edco_conveyor_segment_a":
            position = (
                0.0,
                belt_width * 0.2,
                max(12.0, belt_length * 0.22),
            )
            look_at = (
                belt_length * 0.08,
                0.0,
                self.config.main_belt.height,
            )
            return position, look_at
        position = (
            0.0,
            0.0,
            max(7.0, belt_length * 0.35),
        )
        look_at = (0.0, 0.0, self.config.main_belt.height)
        return position, look_at

    def _frame_belt(self, get_active_viewport) -> bool:
        if get_active_viewport is None:
            return False
        import omni.kit.commands  # type: ignore

        viewport = get_active_viewport()
        if viewport is None:
            return False
        if not viewport.camera_path:
            viewport.camera_path = "/OmniverseKit_Persp"
        omni.kit.commands.execute(
            "FramePrimsCommand",
            prim_to_move=viewport.camera_path,
            prims_to_frame=["/World/RecyclingLine/Facility/MainBelt"] if not self._uses_topology_layout() else ["/World/RecyclingLine"],
            aspect_ratio=self.config.camera.resolution[0] / self.config.camera.resolution[1],
            zoom=0.25,
        )
        return True

    def _frame_viewport_on_conveyor(self, simulation_app, get_active_viewport) -> None:
        for _ in range(120):
            simulation_app.update()
            if self._frame_belt(get_active_viewport):
                return

    def _set_viewport_camera(self, simulation_app, get_active_viewport, camera_path: str) -> None:
        if get_active_viewport is None:
            return
        for _ in range(120):
            simulation_app.update()
            viewport = get_active_viewport()
            if viewport is None:
                continue
            viewport.camera_path = camera_path
            return

    def _color_array(self, color):
        import numpy as np

        return np.array(color)

    def _safe_token(self, value: str) -> str:
        token = value.replace("-", "N").replace(".", "_")
        return "".join(character if character.isalnum() or character == "_" else "_" for character in token)

    def _write_frame(self, frame, path) -> None:
        import numpy as np
        from PIL import Image

        np.save(path.with_suffix(".npy"), frame)
        image = np.asarray(frame)
        if image.size == 0:
            return
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        Image.fromarray(image).save(path.with_suffix(".png"))

    def _lighten_color(self, color, amount: float) -> tuple[float, float, float]:
        return tuple(min(channel + amount, 1.0) for channel in color)

    def _darken_color(self, color, amount: float) -> tuple[float, float, float]:
        return tuple(max(channel - amount, 0.0) for channel in color)

    def _line_to_world(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        yaw = math.radians(self.config.environment.conveyor_transform.yaw_deg)
        x = point[0] * math.cos(yaw) - point[1] * math.sin(yaw)
        y = point[0] * math.sin(yaw) + point[1] * math.cos(yaw)
        root = self.config.environment.conveyor_transform.position
        return (root[0] + x, root[1] + y, root[2] + point[2])

    def _line_velocity_to_world(self, velocity: tuple[float, float, float]) -> tuple[float, float, float]:
        yaw = math.radians(self.config.environment.conveyor_transform.yaw_deg)
        x = velocity[0] * math.cos(yaw) - velocity[1] * math.sin(yaw)
        y = velocity[0] * math.sin(yaw) + velocity[1] * math.cos(yaw)
        return (x, y, velocity[2])

    def _compose_yaw_orientation(self, local_yaw_rad: float) -> tuple[float, float, float, float]:
        total_yaw = math.radians(self.config.environment.conveyor_transform.yaw_deg) + local_yaw_rad
        return (math.cos(total_yaw / 2), 0.0, 0.0, math.sin(total_yaw / 2))

    def _capture_side_sign(self, y_value: float) -> float:
        return 1.0 if y_value >= 0 else -1.0
