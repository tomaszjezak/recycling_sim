from __future__ import annotations

from dataclasses import dataclass

from recycling_mrf.config import SimulationConfig
from recycling_mrf.geometry import (
    segment_axes,
    segment_center,
    segment_debug_samples,
    segment_surface_point,
)


@dataclass(frozen=True)
class BeltBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_top: float


@dataclass(frozen=True)
class SpawnZone:
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class StationZone:
    name: str
    station_type: str
    x_range: tuple[float, float]
    target_materials: tuple[str, ...]
    target_commodities: tuple[str, ...]
    capture_center: tuple[float, float, float] | None
    capture_size: tuple[float, float, float] | None


@dataclass(frozen=True)
class CameraPose:
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    resolution: tuple[int, int]


@dataclass(frozen=True)
class EnvironmentLayout:
    mode: str
    environment_id: str
    layout_preset: str
    conveyor_root: tuple[float, float, float]
    conveyor_yaw_deg: float
    bin_offset: tuple[float, float, float]
    station_marker_height: float


@dataclass(frozen=True)
class SceneDefinition:
    belt_bounds: BeltBounds
    spawn_zone: SpawnZone
    stations: tuple[StationZone, ...]
    camera: CameraPose
    environment: EnvironmentLayout
    segments: tuple[dict, ...]
    nodes: tuple[dict, ...]
    machines: tuple[dict, ...]
    platforms: tuple[dict, ...]
    spawn_points: tuple[dict, ...]
    drop_zones: tuple[dict, ...]
    subareas: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "belt_bounds": {
                "x_min": self.belt_bounds.x_min,
                "x_max": self.belt_bounds.x_max,
                "y_min": self.belt_bounds.y_min,
                "y_max": self.belt_bounds.y_max,
                "z_top": self.belt_bounds.z_top,
            },
            "spawn_zone": {
                "center": self.spawn_zone.center,
                "size": self.spawn_zone.size,
            },
            "stations": [
                {
                    "name": station.name,
                    "station_type": station.station_type,
                    "x_range": station.x_range,
                    "target_materials": station.target_materials,
                    "target_commodities": station.target_commodities,
                    "capture_center": station.capture_center,
                    "capture_size": station.capture_size,
                }
                for station in self.stations
            ],
            "camera": {
                "position": self.camera.position,
                "look_at": self.camera.look_at,
                "resolution": self.camera.resolution,
            },
            "environment": {
                "mode": self.environment.mode,
                "environment_id": self.environment.environment_id,
                "layout_preset": self.environment.layout_preset,
                "conveyor_root": self.environment.conveyor_root,
                "conveyor_yaw_deg": self.environment.conveyor_yaw_deg,
                "bin_offset": self.environment.bin_offset,
                "station_marker_height": self.environment.station_marker_height,
            },
            "segments": list(self.segments),
            "nodes": list(self.nodes),
            "machines": list(self.machines),
            "platforms": list(self.platforms),
            "spawn_points": list(self.spawn_points),
            "drop_zones": list(self.drop_zones),
            "subareas": list(self.subareas),
        }


def build_scene_definition(config: SimulationConfig) -> SceneDefinition:
    belt_bounds = BeltBounds(
        x_min=-config.main_belt.length / 2,
        x_max=config.main_belt.length / 2,
        y_min=-config.main_belt.width / 2,
        y_max=config.main_belt.width / 2,
        z_top=config.main_belt.height,
    )
    stations = tuple(
        StationZone(
            name=station.name,
            station_type=station.station_type,
            x_range=station.x_range,
            target_materials=station.target_materials,
            target_commodities=station.target_commodities,
            capture_center=(
                (
                    (station.x_range[0] + station.x_range[1]) / 2 + station.capture_area.offset[0],
                    station.capture_area.offset[1],
                    config.main_belt.height + station.capture_area.offset[2],
                )
                if station.capture_area is not None
                else None
            ),
            capture_size=station.capture_area.size if station.capture_area is not None else None,
        )
        for station in config.stations
    )
    camera = CameraPose(
        position=config.camera.position,
        look_at=config.camera.look_at,
        resolution=config.camera.resolution,
    )
    environment = EnvironmentLayout(
        mode=config.environment.mode,
        environment_id=config.environment.environment_id,
        layout_preset=config.environment.layout_preset,
        conveyor_root=config.environment.conveyor_transform.position,
        conveyor_yaw_deg=config.environment.conveyor_transform.yaw_deg,
        bin_offset=config.environment.bin_offset,
        station_marker_height=config.environment.station_marker_height,
    )
    if config.conveyor_segments and config.spawn_segment_id is not None:
        spawn_segment = next(segment for segment in config.conveyor_segments if segment.id == config.spawn_segment_id)
        tangent, lateral, normal = segment_axes(spawn_segment)
        spawn_surface = segment_surface_point(spawn_segment, 0.0, 0.0)
        spawn_zone = SpawnZone(
            center=tuple(
                spawn_surface[idx] + normal[idx] * (config.spawn.drop_height / 2)
                for idx in range(3)
            ),
            size=(
                0.4,
                min(spawn_segment.width, max(config.spawn.lane_jitter * 2 + 0.2, 0.4)),
                config.spawn.drop_height,
            ),
        )
    else:
        tangent = (1.0, 0.0, 0.0)
        lateral = (0.0, 1.0, 0.0)
        normal = (0.0, 0.0, 1.0)
        spawn_surface = (belt_bounds.x_min + 0.45, 0.0, belt_bounds.z_top)
        spawn_zone = SpawnZone(
            center=(
                belt_bounds.x_min + 0.45,
                0.0,
                belt_bounds.z_top + config.spawn.drop_height / 2,
            ),
            size=(
                0.25,
                min(config.main_belt.width, config.spawn.lane_jitter * 2 + 0.2),
                config.spawn.drop_height,
            ),
        )
    segments = tuple(
        {
            "id": segment.id,
            "start_pose": {
                "position": segment.start_pose.position,
                "yaw_deg": segment.start_pose.yaw_deg,
            },
            "end_pose": {
                "position": segment.end_pose.position,
                "yaw_deg": segment.end_pose.yaw_deg,
            },
            "width": segment.width,
            "belt_height": segment.belt_height,
            "thickness": segment.thickness,
            "belt_speed": segment.belt_speed,
            "support_spacing": segment.support_spacing,
            "sidewalls": segment.sidewalls,
            "skirting": segment.skirting,
            "access_side": segment.access_side,
            "has_catwalk": segment.has_catwalk,
            "role": segment.role,
            "length": segment.length,
            "incline_angle_deg": segment.incline_angle_deg,
            "center_pose": {
                "position": segment_center(segment),
                "yaw_deg": segment.start_pose.yaw_deg,
            },
            "surface_path": {
                "start": segment_surface_point(segment, 0.0, 0.0),
                "end": segment_surface_point(segment, segment.length, 0.0),
                "samples": segment_debug_samples(segment),
            },
            "tangent": segment_axes(segment)[0],
            "lateral_axis": segment_axes(segment)[1],
            "surface_normal": segment_axes(segment)[2],
        }
        for segment in config.conveyor_segments
    ) or (
        {
            "id": "main_belt",
            "start_pose": {"position": (-config.main_belt.length / 2, 0.0, config.main_belt.height / 2), "yaw_deg": 0.0},
            "end_pose": {"position": (config.main_belt.length / 2, 0.0, config.main_belt.height / 2), "yaw_deg": 0.0},
            "width": config.main_belt.width,
            "belt_height": config.main_belt.height,
            "thickness": config.main_belt.height,
            "belt_speed": config.main_belt.speed,
            "support_spacing": max(config.main_belt.length / 4, 1.0),
            "sidewalls": False,
            "skirting": False,
            "access_side": None,
            "has_catwalk": False,
            "role": "legacy_mainline",
            "length": config.main_belt.length,
            "incline_angle_deg": 0.0,
            "center_pose": {"position": (0.0, 0.0, config.main_belt.height / 2), "yaw_deg": 0.0},
            "surface_path": {
                "start": (-config.main_belt.length / 2, 0.0, config.main_belt.height),
                "end": (config.main_belt.length / 2, 0.0, config.main_belt.height),
                "samples": tuple(
                    (
                        -config.main_belt.length / 2 + idx * (config.main_belt.length / 8),
                        0.0,
                        config.main_belt.height,
                    )
                    for idx in range(9)
                ),
            },
            "tangent": (1.0, 0.0, 0.0),
            "lateral_axis": (0.0, 1.0, 0.0),
            "surface_normal": (0.0, 0.0, 1.0),
        },
    )
    nodes = tuple(
        {
            "id": node.id,
            "node_type": node.node_type,
            "pose": {"position": node.pose.position, "yaw_deg": node.pose.yaw_deg},
            "upstream_segment_ids": list(node.upstream_segment_ids),
            "downstream_segment_ids": list(node.downstream_segment_ids),
            "drop_zone_id": node.drop_zone_id,
        }
        for node in config.routing_nodes
    )
    machines = tuple(
        {
            "id": machine.id,
            "machine_type": machine.machine_type,
            "pose": {"position": machine.pose.position, "yaw_deg": machine.pose.yaw_deg},
            "size": machine.size,
            "input_segment_ids": list(machine.input_segment_ids),
            "output_segment_ids": list(machine.output_segment_ids),
            "role": machine.role,
        }
        for machine in config.machine_zones
    )
    platforms = tuple(
        {
            "id": platform.id,
            "pose": {"position": platform.pose.position, "yaw_deg": platform.pose.yaw_deg},
            "size": platform.size,
            "elevation": platform.elevation,
            "guard_rails": platform.guard_rails,
            "stairs": platform.stairs,
            "ladders": platform.ladders,
            "adjacent_segment_ids": list(platform.adjacent_segment_ids),
        }
        for platform in config.platforms
    )
    spawn_points = (
        {
            "id": "primary_spawn",
            "segment_id": config.spawn_segment_id or "main_belt",
            "position": spawn_surface,
            "tangent": tangent,
            "lateral_axis": lateral,
            "surface_normal": normal,
            "drop_height": config.spawn.drop_height,
        },
    )
    drop_zones = tuple(
        {
            "id": zone.id,
            "zone_type": zone.zone_type,
            "pose": {"position": zone.pose.position, "yaw_deg": zone.pose.yaw_deg},
            "size": zone.size,
        }
        for zone in config.drop_zones
    )
    return SceneDefinition(
        belt_bounds=belt_bounds,
        spawn_zone=spawn_zone,
        stations=stations,
        camera=camera,
        environment=environment,
        segments=segments,
        nodes=nodes,
        machines=machines,
        platforms=platforms,
        spawn_points=spawn_points,
        drop_zones=drop_zones,
        subareas=config.subareas,
    )
