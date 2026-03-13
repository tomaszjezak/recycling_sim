from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BeltConfig:
    length: float
    width: float
    height: float
    speed: float


@dataclass(frozen=True)
class CaptureAreaConfig:
    offset: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class StationConfig:
    name: str
    station_type: str
    x_range: tuple[float, float]
    target_materials: tuple[str, ...]
    target_commodities: tuple[str, ...]
    capture_area: CaptureAreaConfig | None = None


@dataclass(frozen=True)
class SpawnConfig:
    rate: float
    drop_height: float
    lane_jitter: float
    yaw_jitter_deg: float


@dataclass(frozen=True)
class CameraConfig:
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    resolution: tuple[int, int]


@dataclass(frozen=True)
class PerceptionZoneConfig:
    id: str
    sensor_type: str
    segment_id: str
    distance_range: tuple[float, float]
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    resolution: tuple[int, int]
    target_materials: tuple[str, ...]
    target_commodities: tuple[str, ...]
    model_name: str = "scripted_classifier"


@dataclass(frozen=True)
class TransformConfig:
    position: tuple[float, float, float]
    yaw_deg: float


@dataclass(frozen=True)
class PoseConfig:
    position: tuple[float, float, float]
    yaw_deg: float


@dataclass(frozen=True)
class AccumulationZoneConfig:
    enabled: bool
    length: float


@dataclass(frozen=True)
class ConveyorSegmentConfig:
    id: str
    start_pose: PoseConfig
    end_pose: PoseConfig
    width: float
    belt_height: float
    thickness: float
    belt_speed: float
    support_spacing: float
    sidewalls: bool
    skirting: bool
    access_side: str | None
    has_catwalk: bool
    accumulation_zone: AccumulationZoneConfig | None
    role: str

    @property
    def length(self) -> float:
        dx = self.end_pose.position[0] - self.start_pose.position[0]
        dy = self.end_pose.position[1] - self.start_pose.position[1]
        dz = self.end_pose.position[2] - self.start_pose.position[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @property
    def incline_angle_deg(self) -> float:
        dx = self.end_pose.position[0] - self.start_pose.position[0]
        dy = self.end_pose.position[1] - self.start_pose.position[1]
        horizontal = math.sqrt(dx * dx + dy * dy)
        dz = self.end_pose.position[2] - self.start_pose.position[2]
        return math.degrees(math.atan2(dz, max(horizontal, 1e-6)))


@dataclass(frozen=True)
class RoutingNodeConfig:
    id: str
    node_type: str
    pose: PoseConfig
    upstream_segment_ids: tuple[str, ...]
    downstream_segment_ids: tuple[str, ...]
    drop_zone_id: str | None = None


@dataclass(frozen=True)
class MachineZoneConfig:
    id: str
    machine_type: str
    pose: PoseConfig
    size: tuple[float, float, float]
    input_segment_ids: tuple[str, ...]
    output_segment_ids: tuple[str, ...]
    role: str


@dataclass(frozen=True)
class RobotCellConfig:
    id: str
    robot_type: str
    base_pose: PoseConfig
    source_segment_id: str
    pick_distance: float
    place_drop_zone_id: str
    target_commodities: tuple[str, ...]
    place_segment_id: str | None = None
    place_distance: float = 0.0
    perception_zone_id: str | None = None
    controller: str = "scripted"
    pick_duration: float = 0.6
    place_duration: float = 0.5
    cooldown: float = 0.25


@dataclass(frozen=True)
class PlatformConfig:
    id: str
    pose: PoseConfig
    size: tuple[float, float, float]
    elevation: float
    guard_rails: bool
    stairs: bool
    ladders: bool
    adjacent_segment_ids: tuple[str, ...]


@dataclass(frozen=True)
class DropZoneConfig:
    id: str
    zone_type: str
    pose: PoseConfig
    size: tuple[float, float, float]


@dataclass(frozen=True)
class VisualBlockConfig:
    id: str
    shape: str
    pose: PoseConfig
    size: tuple[float, float, float]
    color: tuple[float, float, float]


@dataclass(frozen=True)
class MaterialRouteConfig:
    id: str
    material_classes: tuple[str, ...]
    commodity_targets: tuple[str, ...]
    segment_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    machine_ids: tuple[str, ...]
    drop_zone_id: str | None
    outbound_zone_id: str | None
    final_status: str
    station_name: str | None = None


@dataclass(frozen=True)
class EnvironmentConfig:
    mode: str
    environment_id: str
    layout_preset: str
    conveyor_transform: TransformConfig
    bin_offset: tuple[float, float, float]
    station_marker_height: float
    dome_light_intensity: float


@dataclass(frozen=True)
class ItemSpec:
    stream: str
    material_class: str
    shape: str
    size: tuple[float, float, float]
    mass: float
    color: tuple[float, float, float]
    screenable_2d: bool
    magnetic: bool
    container_like: bool
    visual_profile: str = "auto"
    commodity_target: str | None = None


@dataclass(frozen=True)
class SourceConfig:
    id: str
    source_type: str
    share: float


@dataclass(frozen=True)
class FacilityConfig:
    id: str
    facility_type: str
    description: str


@dataclass(frozen=True)
class StreamRouteConfig:
    initial_facility_id: str
    residual_facility_id: str | None = None
    recovered_facility_id: str | None = None
    end_market_facility_id: str | None = None
    recover_material_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SystemConfig:
    sources: tuple[SourceConfig, ...]
    facilities: tuple[FacilityConfig, ...]
    stream_routes: dict[str, StreamRouteConfig]
    item_mix: dict[str, float]
    item_catalog: dict[str, ItemSpec]
    commodity_end_markets: dict[str, str]

    @property
    def normalized_item_mix(self) -> dict[str, float]:
        total_weight = sum(self.item_mix.values())
        return {name: weight / total_weight for name, weight in self.item_mix.items()}

    @property
    def normalized_source_mix(self) -> dict[str, float]:
        total_share = sum(source.share for source in self.sources)
        return {source.id: source.share / total_share for source in self.sources}

    @property
    def facilities_by_id(self) -> dict[str, FacilityConfig]:
        return {facility.id: facility for facility in self.facilities}


@dataclass(frozen=True)
class MRFConfig:
    main_belt: BeltConfig
    spawn: SpawnConfig
    camera: CameraConfig
    environment: EnvironmentConfig
    stations: tuple[StationConfig, ...]
    perception_zones: tuple[PerceptionZoneConfig, ...] = ()
    conveyor_segments: tuple[ConveyorSegmentConfig, ...] = ()
    routing_nodes: tuple[RoutingNodeConfig, ...] = ()
    machine_zones: tuple[MachineZoneConfig, ...] = ()
    robot_cells: tuple[RobotCellConfig, ...] = ()
    platforms: tuple[PlatformConfig, ...] = ()
    material_routes: tuple[MaterialRouteConfig, ...] = ()
    drop_zones: tuple[DropZoneConfig, ...] = ()
    visual_blocks: tuple[VisualBlockConfig, ...] = ()
    spawn_segment_id: str | None = None
    subareas: tuple[str, ...] = ()


@dataclass(frozen=True)
class SimulationConfig:
    seed: int
    episode_duration: float
    physics_dt: float
    render_dt: float
    output_dir: Path
    system: SystemConfig
    mrf: MRFConfig

    @classmethod
    def from_file(cls, path: str | Path) -> "SimulationConfig":
        data = json.loads(Path(path).read_text())
        system = data["system"]
        mrf = data["mrf"]
        config = cls(
            seed=int(data["seed"]),
            episode_duration=float(data["episode_duration"]),
            physics_dt=float(data["physics_dt"]),
            render_dt=float(data["render_dt"]),
            output_dir=Path(data["output_dir"]),
            system=SystemConfig(
                sources=tuple(
                    SourceConfig(
                        id=str(source["id"]),
                        source_type=str(source["source_type"]),
                        share=float(source["share"]),
                    )
                    for source in system["sources"]
                ),
                facilities=tuple(
                    FacilityConfig(
                        id=str(facility["id"]),
                        facility_type=str(facility["facility_type"]),
                        description=str(facility.get("description", "")),
                    )
                    for facility in system["facilities"]
                ),
                stream_routes={
                    str(stream): StreamRouteConfig(
                        initial_facility_id=str(route["initial_facility_id"]),
                        residual_facility_id=(
                            str(route["residual_facility_id"])
                            if route.get("residual_facility_id") is not None
                            else None
                        ),
                        recovered_facility_id=(
                            str(route["recovered_facility_id"])
                            if route.get("recovered_facility_id") is not None
                            else None
                        ),
                        end_market_facility_id=(
                            str(route["end_market_facility_id"])
                            if route.get("end_market_facility_id") is not None
                            else None
                        ),
                        recover_material_classes=tuple(
                            str(value) for value in route.get("recover_material_classes", [])
                        ),
                    )
                    for stream, route in system["stream_routes"].items()
                },
                item_mix={str(k): float(v) for k, v in system["item_mix"].items()},
                item_catalog={
                    str(name): ItemSpec(
                        stream=str(spec["stream"]),
                        material_class=str(spec["material_class"]),
                        shape=str(spec["shape"]),
                        size=tuple(float(v) for v in spec["size"]),
                        mass=float(spec["mass"]),
                        color=tuple(float(v) for v in spec["color"]),
                        screenable_2d=bool(spec.get("screenable_2d", False)),
                        magnetic=bool(spec.get("magnetic", False)),
                        container_like=bool(spec.get("container_like", False)),
                        visual_profile=str(spec.get("visual_profile", "auto")),
                        commodity_target=(
                            str(spec["commodity_target"])
                            if spec.get("commodity_target") is not None
                            else None
                        ),
                    )
                    for name, spec in system["item_catalog"].items()
                },
                commodity_end_markets={
                    str(k): str(v) for k, v in system.get("commodity_end_markets", {}).items()
                },
            ),
            mrf=MRFConfig(
                main_belt=BeltConfig(**mrf["main_belt"]),
                spawn=SpawnConfig(**mrf["spawn"]),
                camera=CameraConfig(
                    position=tuple(float(v) for v in mrf["camera"]["position"]),
                    look_at=tuple(float(v) for v in mrf["camera"]["look_at"]),
                    resolution=tuple(int(v) for v in mrf["camera"]["resolution"]),
                ),
                environment=EnvironmentConfig(
                    mode=str(mrf.get("environment", {}).get("mode", "warehouse")),
                    environment_id=str(mrf.get("environment", {}).get("environment_id", "simple_warehouse")),
                    layout_preset=str(mrf.get("environment", {}).get("layout_preset", "full_mrf")),
                    conveyor_transform=TransformConfig(
                        position=tuple(
                            float(v)
                            for v in mrf.get("environment", {})
                            .get("conveyor_transform", {})
                            .get("position", [10.0, 2.0, -1.06])
                        ),
                        yaw_deg=float(
                            mrf.get("environment", {})
                            .get("conveyor_transform", {})
                            .get("yaw_deg", 90.0)
                        ),
                    ),
                    bin_offset=tuple(
                        float(v) for v in mrf.get("environment", {}).get("bin_offset", [0.0, 1.9, 0.0])
                    ),
                    station_marker_height=float(
                        mrf.get("environment", {}).get("station_marker_height", 1.05)
                    ),
                    dome_light_intensity=float(
                        mrf.get("environment", {}).get("dome_light_intensity", 350.0)
                    ),
                ),
                stations=tuple(
                    StationConfig(
                        name=str(station["name"]),
                        station_type=str(station["station_type"]),
                        x_range=tuple(float(v) for v in station["x_range"]),
                        target_materials=tuple(str(v) for v in station.get("target_materials", [])),
                        target_commodities=tuple(str(v) for v in station.get("target_commodities", [])),
                        capture_area=(
                            CaptureAreaConfig(
                                offset=tuple(float(v) for v in station["capture_area"]["offset"]),
                                size=tuple(float(v) for v in station["capture_area"]["size"]),
                            )
                            if station.get("capture_area") is not None
                            else None
                        ),
                    )
                    for station in mrf.get("stations", [])
                ),
                perception_zones=tuple(
                    PerceptionZoneConfig(
                        id=str(zone["id"]),
                        sensor_type=str(zone.get("sensor_type", "rgb_camera")),
                        segment_id=str(zone["segment_id"]),
                        distance_range=tuple(float(v) for v in zone["distance_range"]),
                        position=tuple(float(v) for v in zone["position"]),
                        look_at=tuple(float(v) for v in zone["look_at"]),
                        resolution=tuple(int(v) for v in zone.get("resolution", [640, 360])),
                        target_materials=tuple(str(v) for v in zone.get("target_materials", [])),
                        target_commodities=tuple(str(v) for v in zone.get("target_commodities", [])),
                        model_name=str(zone.get("model_name", "scripted_classifier")),
                    )
                    for zone in mrf.get("perception_zones", [])
                ),
                conveyor_segments=tuple(
                    ConveyorSegmentConfig(
                        id=str(segment["id"]),
                        start_pose=_parse_pose(segment["start_pose"]),
                        end_pose=_parse_pose(segment["end_pose"]),
                        width=float(segment["width"]),
                        belt_height=float(segment.get("belt_height", segment.get("thickness", 0.18))),
                        thickness=float(segment.get("thickness", 0.18)),
                        belt_speed=float(segment["belt_speed"]),
                        support_spacing=float(segment["support_spacing"]),
                        sidewalls=bool(segment.get("sidewalls", False)),
                        skirting=bool(segment.get("skirting", False)),
                        access_side=(
                            str(segment["access_side"])
                            if segment.get("access_side") is not None
                            else None
                        ),
                        has_catwalk=bool(segment.get("has_catwalk", False)),
                        accumulation_zone=(
                            AccumulationZoneConfig(
                                enabled=bool(segment.get("accumulation_zone", {}).get("enabled", True)),
                                length=float(segment.get("accumulation_zone", {}).get("length", 0.0)),
                            )
                            if segment.get("accumulation_zone") is not None
                            else None
                        ),
                        role=str(segment.get("role", "transfer")),
                    )
                    for segment in mrf.get("conveyor_segments", [])
                ),
                routing_nodes=tuple(
                    RoutingNodeConfig(
                        id=str(node["id"]),
                        node_type=str(node["node_type"]),
                        pose=_parse_pose(node["pose"]),
                        upstream_segment_ids=tuple(str(v) for v in node.get("upstream_segment_ids", [])),
                        downstream_segment_ids=tuple(str(v) for v in node.get("downstream_segment_ids", [])),
                        drop_zone_id=(
                            str(node["drop_zone_id"])
                            if node.get("drop_zone_id") is not None
                            else None
                        ),
                    )
                    for node in mrf.get("routing_nodes", [])
                ),
                machine_zones=tuple(
                    MachineZoneConfig(
                        id=str(zone["id"]),
                        machine_type=str(zone["machine_type"]),
                        pose=_parse_pose(zone["pose"]),
                        size=tuple(float(v) for v in zone["size"]),
                        input_segment_ids=tuple(str(v) for v in zone.get("input_segment_ids", [])),
                        output_segment_ids=tuple(str(v) for v in zone.get("output_segment_ids", [])),
                        role=str(zone.get("role", zone["machine_type"])),
                    )
                    for zone in mrf.get("machine_zones", [])
                ),
                robot_cells=tuple(
                    RobotCellConfig(
                        id=str(cell["id"]),
                        robot_type=str(cell.get("robot_type", "xarm6")),
                        base_pose=_parse_pose(cell["base_pose"]),
                        source_segment_id=str(cell["source_segment_id"]),
                        pick_distance=float(cell["pick_distance"]),
                        place_drop_zone_id=str(cell["place_drop_zone_id"]),
                        place_segment_id=(
                            str(cell["place_segment_id"])
                            if cell.get("place_segment_id") is not None
                            else None
                        ),
                        place_distance=float(cell.get("place_distance", 0.0)),
                        target_commodities=tuple(str(v) for v in cell.get("target_commodities", [])),
                        perception_zone_id=(
                            str(cell["perception_zone_id"])
                            if cell.get("perception_zone_id") is not None
                            else None
                        ),
                        controller=str(cell.get("controller", "scripted")),
                        pick_duration=float(cell.get("pick_duration", 0.6)),
                        place_duration=float(cell.get("place_duration", 0.5)),
                        cooldown=float(cell.get("cooldown", 0.25)),
                    )
                    for cell in mrf.get("robot_cells", [])
                ),
                platforms=tuple(
                    PlatformConfig(
                        id=str(platform["id"]),
                        pose=_parse_pose(platform["pose"]),
                        size=tuple(float(v) for v in platform["size"]),
                        elevation=float(platform.get("elevation", platform["pose"]["position"][2])),
                        guard_rails=bool(platform.get("guard_rails", True)),
                        stairs=bool(platform.get("stairs", False)),
                        ladders=bool(platform.get("ladders", False)),
                        adjacent_segment_ids=tuple(str(v) for v in platform.get("adjacent_segment_ids", [])),
                    )
                    for platform in mrf.get("platforms", [])
                ),
                material_routes=tuple(
                    MaterialRouteConfig(
                        id=str(route["id"]),
                        material_classes=tuple(str(v) for v in route.get("material_classes", [])),
                        commodity_targets=tuple(str(v) for v in route.get("commodity_targets", [])),
                        segment_ids=tuple(str(v) for v in route.get("segment_ids", [])),
                        node_ids=tuple(str(v) for v in route.get("node_ids", [])),
                        machine_ids=tuple(str(v) for v in route.get("machine_ids", [])),
                        drop_zone_id=(
                            str(route["drop_zone_id"])
                            if route.get("drop_zone_id") is not None
                            else None
                        ),
                        outbound_zone_id=(
                            str(route["outbound_zone_id"])
                            if route.get("outbound_zone_id") is not None
                            else None
                        ),
                        final_status=str(route["final_status"]),
                        station_name=(
                            str(route["station_name"])
                            if route.get("station_name") is not None
                            else None
                        ),
                    )
                    for route in mrf.get("material_routes", [])
                ),
                drop_zones=tuple(
                    DropZoneConfig(
                        id=str(zone["id"]),
                        zone_type=str(zone["zone_type"]),
                        pose=_parse_pose(zone["pose"]),
                        size=tuple(float(v) for v in zone["size"]),
                    )
                    for zone in mrf.get("drop_zones", [])
                ),
                visual_blocks=tuple(
                    VisualBlockConfig(
                        id=str(block["id"]),
                        shape=str(block.get("shape", "cube")),
                        pose=_parse_pose(block["pose"]),
                        size=tuple(float(v) for v in block["size"]),
                        color=tuple(float(v) for v in block.get("color", [0.7, 0.7, 0.7])),
                    )
                    for block in mrf.get("visual_blocks", [])
                ),
                spawn_segment_id=(
                    str(mrf["spawn_segment_id"])
                    if mrf.get("spawn_segment_id") is not None
                    else None
                ),
                subareas=tuple(str(value) for value in mrf.get("subareas", [])),
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.episode_duration <= 0:
            raise ValueError("episode_duration must be positive")
        if self.physics_dt <= 0 or self.render_dt <= 0:
            raise ValueError("physics_dt and render_dt must be positive")
        if self.spawn.rate <= 0:
            raise ValueError("mrf.spawn.rate must be positive")
        if self.main_belt.length <= 0 or self.main_belt.width <= 0 or self.main_belt.height <= 0:
            raise ValueError("mrf.main_belt dimensions must be positive")
        if self.main_belt.speed <= 0:
            raise ValueError("mrf.main_belt.speed must be positive")
        if self.environment.mode not in {"warehouse", "procedural"}:
            raise ValueError(f"unsupported mrf.environment.mode: {self.environment.mode}")
        if self.environment.environment_id not in self.supported_environment_ids:
            raise ValueError(f"unsupported mrf.environment.environment_id: {self.environment.environment_id}")
        if self.environment.layout_preset not in self.supported_layout_presets:
            raise ValueError(f"unsupported mrf.environment.layout_preset: {self.environment.layout_preset}")
        if len(self.environment.conveyor_transform.position) != 3:
            raise ValueError("mrf.environment.conveyor_transform.position must have 3 values")
        if len(self.environment.bin_offset) != 3:
            raise ValueError("mrf.environment.bin_offset must have 3 values")
        if self.environment.station_marker_height <= 0:
            raise ValueError("mrf.environment.station_marker_height must be positive")
        if self.environment.dome_light_intensity <= 0:
            raise ValueError("mrf.environment.dome_light_intensity must be positive")

        if not self.system.sources:
            raise ValueError("system.sources must not be empty")
        if sum(source.share for source in self.system.sources) <= 0:
            raise ValueError("system.sources shares must sum to a positive value")
        for source in self.system.sources:
            if source.source_type not in self.supported_source_types:
                raise ValueError(f"unsupported source_type for {source.id}: {source.source_type}")
            if source.share <= 0:
                raise ValueError(f"source share must be positive for {source.id}")

        if not self.system.item_mix:
            raise ValueError("system.item_mix must not be empty")
        if not self.system.facilities:
            raise ValueError("system.facilities must not be empty")

        total_weight = sum(self.system.item_mix.values())
        if total_weight <= 0:
            raise ValueError("system.item_mix weights must sum to a positive value")

        missing = [name for name in self.system.item_mix if name not in self.system.item_catalog]
        if missing:
            raise ValueError(f"system.item_mix references unknown items: {missing}")

        facilities_by_id = self.system.facilities_by_id
        for facility in self.system.facilities:
            if facility.facility_type not in self.supported_facility_types:
                raise ValueError(f"unsupported facility_type for {facility.id}: {facility.facility_type}")

        for stream in self.supported_streams:
            if stream not in self.system.stream_routes:
                raise ValueError(f"missing system.stream_routes entry for {stream}")

        for stream, route in self.system.stream_routes.items():
            if stream not in self.supported_streams:
                raise ValueError(f"unsupported stream route: {stream}")
            self._require_facility(route.initial_facility_id, facilities_by_id, f"stream route {stream}")
            if route.residual_facility_id is not None:
                self._require_facility(route.residual_facility_id, facilities_by_id, f"stream route {stream}")
            if route.recovered_facility_id is not None:
                self._require_facility(route.recovered_facility_id, facilities_by_id, f"stream route {stream}")
            if route.end_market_facility_id is not None:
                self._require_facility(route.end_market_facility_id, facilities_by_id, f"stream route {stream}")
            unknown_materials = [
                material for material in route.recover_material_classes if material not in self.known_material_classes
            ]
            if unknown_materials:
                raise ValueError(
                    f"stream route {stream} references unknown recover_material_classes: {unknown_materials}"
                )

        for commodity, facility_id in self.system.commodity_end_markets.items():
            self._require_facility(facility_id, facilities_by_id, f"commodity end market {commodity}")
            if facilities_by_id[facility_id].facility_type != "end_market":
                raise ValueError(f"commodity end market for {commodity} must reference an end_market facility")

        has_recycling_items = False
        for name, item in self.system.item_catalog.items():
            if item.shape not in {"box", "cylinder"}:
                raise ValueError(f"unsupported shape for {name}: {item.shape}")
            if item.visual_profile not in self.supported_visual_profiles:
                raise ValueError(f"unsupported visual_profile for {name}: {item.visual_profile}")
            if item.stream not in self.supported_streams:
                raise ValueError(f"unsupported stream for {name}: {item.stream}")
            if any(v <= 0 for v in item.size):
                raise ValueError(f"item size must be positive for {name}")
            if item.mass <= 0:
                raise ValueError(f"item mass must be positive for {name}")
            if any(v < 0 or v > 1 for v in item.color):
                raise ValueError(f"item color channels must be between 0 and 1 for {name}")
            if item.material_class not in self.known_material_classes:
                raise ValueError(f"unsupported material_class for {name}: {item.material_class}")
            if item.stream == "recycling":
                has_recycling_items = True
                if item.commodity_target is None and item.material_class not in {"residue", "glass"}:
                    raise ValueError(f"recycling item {name} must declare commodity_target unless it is residue/glass")
                if item.commodity_target is not None and item.commodity_target not in self.system.commodity_end_markets:
                    raise ValueError(f"recycling item {name} references unknown commodity_target: {item.commodity_target}")
        if not has_recycling_items:
            raise ValueError("system.item_catalog must include at least one recycling item for the MRF branch")

        self._validate_stations()
        self._validate_topology()

    def _validate_stations(self) -> None:
        if not self.stations:
            if not self.conveyor_segments:
                raise ValueError("mrf.stations or mrf.conveyor_segments must not be empty")
            return

        allowed_station_types = {
            "tip_floor",
            "feeder",
            "presort",
            "screen",
            "magnet",
            "optical_sorter",
            "manual_qc",
            "commodity_bunker",
            "baler",
            "residual_exit",
        }
        previous_end = None
        for station in self.stations:
            if station.station_type not in allowed_station_types:
                raise ValueError(f"unsupported station_type for {station.name}: {station.station_type}")
            if len(station.x_range) != 2 or station.x_range[0] >= station.x_range[1]:
                raise ValueError(f"station {station.name} must define an increasing x_range")
            if station.x_range[0] < -self.main_belt.length / 2 or station.x_range[1] > self.main_belt.length / 2:
                raise ValueError(f"station {station.name} x_range must stay within the main belt bounds")
            if previous_end is not None and station.x_range[0] < previous_end:
                raise ValueError(f"station {station.name} overlaps or is out of order")
            previous_end = station.x_range[1]
            if station.capture_area is not None and any(v <= 0 for v in station.capture_area.size):
                raise ValueError(f"capture_area size must be positive for {station.name}")
            if station.capture_area is None and station.station_type in self.capture_station_types:
                raise ValueError(f"station {station.name} requires capture_area for station_type {station.station_type}")
            unknown_targets = [
                material for material in station.target_materials if material not in self.known_material_classes
            ]
            if unknown_targets:
                raise ValueError(f"station {station.name} references unknown target materials: {unknown_targets}")

    def _validate_topology(self) -> None:
        if not self.conveyor_segments:
            return

        segment_ids = {segment.id for segment in self.conveyor_segments}
        if len(segment_ids) != len(self.conveyor_segments):
            raise ValueError("mrf.conveyor_segments ids must be unique")
        for segment in self.conveyor_segments:
            if segment.width <= 0 or segment.belt_height <= 0 or segment.thickness <= 0:
                raise ValueError(f"segment {segment.id} dimensions must be positive")
            if segment.belt_speed <= 0:
                raise ValueError(f"segment {segment.id} belt_speed must be positive")
            if segment.support_spacing <= 0:
                raise ValueError(f"segment {segment.id} support_spacing must be positive")
            if segment.length <= 0:
                raise ValueError(f"segment {segment.id} must have distinct start/end poses")
            if segment.access_side not in {None, "left", "right"}:
                raise ValueError(f"segment {segment.id} access_side must be left/right/null")

        if self.spawn_segment_id is None:
            raise ValueError("mrf.spawn_segment_id is required when conveyor_segments are defined")
        if self.spawn_segment_id not in segment_ids:
            raise ValueError("mrf.spawn_segment_id must reference a conveyor segment")

        drop_zone_ids = {zone.id for zone in self.drop_zones}
        if len(drop_zone_ids) != len(self.drop_zones):
            raise ValueError("mrf.drop_zones ids must be unique")
        visual_block_ids = {block.id for block in self.mrf.visual_blocks}
        if len(visual_block_ids) != len(self.mrf.visual_blocks):
            raise ValueError("mrf.visual_blocks ids must be unique")
        for block in self.mrf.visual_blocks:
            if block.shape not in {"cube", "cylinder", "sphere"}:
                raise ValueError(f"unsupported visual block shape: {block.shape}")
            if len(block.size) != 3 or any(value <= 0 for value in block.size):
                raise ValueError(f"visual block {block.id} must have positive 3D size")
            if len(block.color) != 3 or any(value < 0 or value > 1 for value in block.color):
                raise ValueError(f"visual block {block.id} color values must be between 0 and 1")

        node_ids = {node.id for node in self.routing_nodes}
        if len(node_ids) != len(self.routing_nodes):
            raise ValueError("mrf.routing_nodes ids must be unique")
        adjacency: dict[str, set[str]] = {segment.id: set() for segment in self.conveyor_segments}
        referenced_upstreams: set[str] = set()
        for node in self.routing_nodes:
            if node.node_type not in {"handoff", "split", "merge", "drop"}:
                raise ValueError(f"unsupported node_type for {node.id}: {node.node_type}")
            if not node.upstream_segment_ids:
                raise ValueError(f"node {node.id} must have at least one upstream segment")
            for segment_id in node.upstream_segment_ids + node.downstream_segment_ids:
                if segment_id not in segment_ids:
                    raise ValueError(f"node {node.id} references unknown segment: {segment_id}")
            if node.node_type == "split" and len(node.downstream_segment_ids) < 2:
                raise ValueError(f"split node {node.id} must have at least two downstream segments")
            if node.node_type == "merge" and len(node.upstream_segment_ids) < 2:
                raise ValueError(f"merge node {node.id} must have at least two upstream segments")
            if node.node_type == "drop" and node.drop_zone_id is None:
                raise ValueError(f"drop node {node.id} must reference a drop_zone_id")
            if node.drop_zone_id is not None and node.drop_zone_id not in drop_zone_ids:
                raise ValueError(f"node {node.id} references unknown drop zone: {node.drop_zone_id}")
            for upstream in node.upstream_segment_ids:
                referenced_upstreams.add(upstream)
                for downstream in node.downstream_segment_ids:
                    adjacency[upstream].add(downstream)

        perception_zone_ids = {zone.id for zone in self.perception_zones}
        if len(perception_zone_ids) != len(self.perception_zones):
            raise ValueError("mrf.perception_zones ids must be unique")
        for zone in self.perception_zones:
            if zone.sensor_type not in self.supported_perception_sensor_types:
                raise ValueError(f"unsupported perception sensor_type for {zone.id}: {zone.sensor_type}")
            if zone.segment_id not in segment_ids:
                raise ValueError(f"perception zone {zone.id} references unknown segment: {zone.segment_id}")
            if len(zone.distance_range) != 2 or zone.distance_range[0] >= zone.distance_range[1]:
                raise ValueError(f"perception zone {zone.id} must define an increasing distance_range")
            segment = next(segment for segment in self.conveyor_segments if segment.id == zone.segment_id)
            if zone.distance_range[0] < 0 or zone.distance_range[1] > segment.length:
                raise ValueError(f"perception zone {zone.id} distance_range must stay within source segment length")
            if len(zone.position) != 3 or len(zone.look_at) != 3:
                raise ValueError(f"perception zone {zone.id} must define 3D position and look_at")
            if len(zone.resolution) != 2 or any(value <= 0 for value in zone.resolution):
                raise ValueError(f"perception zone {zone.id} resolution must be positive")
            unknown_materials = [
                material for material in zone.target_materials if material not in self.known_material_classes
            ]
            if unknown_materials:
                raise ValueError(f"perception zone {zone.id} references unknown target materials: {unknown_materials}")
            unknown_commodities = [
                commodity for commodity in zone.target_commodities if commodity not in self.system.commodity_end_markets
            ]
            if unknown_commodities:
                raise ValueError(f"perception zone {zone.id} references unknown target commodities: {unknown_commodities}")

        machine_ids = {machine.id for machine in self.machine_zones}
        if len(machine_ids) != len(self.machine_zones):
            raise ValueError("mrf.machine_zones ids must be unique")
        start_anchors: dict[str, set[str]] = {segment.id: set() for segment in self.conveyor_segments}
        end_anchors: dict[str, set[str]] = {segment.id: set() for segment in self.conveyor_segments}
        start_anchors[self.spawn_segment_id].add("spawn")
        for machine in self.machine_zones:
            if any(v <= 0 for v in machine.size):
                raise ValueError(f"machine zone {machine.id} size must be positive")
            for segment_id in machine.input_segment_ids + machine.output_segment_ids:
                if segment_id not in segment_ids:
                    raise ValueError(f"machine zone {machine.id} references unknown segment: {segment_id}")
            for segment_id in machine.input_segment_ids:
                end_anchors[segment_id].add(f"machine:{machine.id}:input")
            for segment_id in machine.output_segment_ids:
                start_anchors[segment_id].add(f"machine:{machine.id}:output")

        robot_cell_ids = {cell.id for cell in self.robot_cells}
        if len(robot_cell_ids) != len(self.robot_cells):
            raise ValueError("mrf.robot_cells ids must be unique")
        for cell in self.robot_cells:
            if cell.robot_type not in self.supported_robot_types:
                raise ValueError(f"unsupported robot_type for {cell.id}: {cell.robot_type}")
            if cell.source_segment_id not in segment_ids:
                raise ValueError(f"robot cell {cell.id} references unknown source segment: {cell.source_segment_id}")
            source_segment = next(segment for segment in self.conveyor_segments if segment.id == cell.source_segment_id)
            if cell.pick_distance < 0 or cell.pick_distance > source_segment.length:
                raise ValueError(f"robot cell {cell.id} pick_distance must stay within source segment length")
            if cell.place_drop_zone_id not in drop_zone_ids:
                raise ValueError(f"robot cell {cell.id} references unknown drop zone: {cell.place_drop_zone_id}")
            if cell.place_segment_id is not None:
                if cell.place_segment_id not in segment_ids:
                    raise ValueError(f"robot cell {cell.id} references unknown placement segment: {cell.place_segment_id}")
                place_segment = next(segment for segment in self.conveyor_segments if segment.id == cell.place_segment_id)
                if cell.place_distance < 0 or cell.place_distance > place_segment.length:
                    raise ValueError(
                        f"robot cell {cell.id} place_distance must stay within placement segment length"
                    )
            if not cell.target_commodities:
                raise ValueError(f"robot cell {cell.id} must target at least one commodity")
            unknown_commodities = [
                commodity for commodity in cell.target_commodities if commodity not in self.system.commodity_end_markets
            ]
            if unknown_commodities:
                raise ValueError(f"robot cell {cell.id} references unknown target commodities: {unknown_commodities}")
            if cell.perception_zone_id is not None:
                if cell.perception_zone_id not in perception_zone_ids:
                    raise ValueError(f"robot cell {cell.id} references unknown perception zone: {cell.perception_zone_id}")
                zone = next(zone for zone in self.perception_zones if zone.id == cell.perception_zone_id)
                if zone.segment_id == cell.source_segment_id and cell.pick_distance <= zone.distance_range[1]:
                    raise ValueError(
                        f"robot cell {cell.id} pick_distance must be downstream of linked perception zone {zone.id}"
                    )
            if not cell.controller:
                raise ValueError(f"robot cell {cell.id} controller must not be empty")
            if cell.pick_duration <= 0 or cell.place_duration <= 0 or cell.cooldown < 0:
                raise ValueError(f"robot cell {cell.id} timing values must be positive")

        platform_ids = {platform.id for platform in self.platforms}
        if len(platform_ids) != len(self.platforms):
            raise ValueError("mrf.platforms ids must be unique")
        for platform in self.platforms:
            if any(v <= 0 for v in platform.size):
                raise ValueError(f"platform {platform.id} size must be positive")
            for segment_id in platform.adjacent_segment_ids:
                if segment_id not in segment_ids:
                    raise ValueError(f"platform {platform.id} references unknown adjacent segment: {segment_id}")

        if not self.material_routes:
            raise ValueError("mrf.material_routes must not be empty when conveyor_segments are defined")
        for route in self.material_routes:
            if not route.segment_ids:
                raise ValueError(f"material route {route.id} must include segment_ids")
            if route.segment_ids[0] != self.spawn_segment_id:
                raise ValueError(f"material route {route.id} must start on spawn_segment_id {self.spawn_segment_id}")
            for segment_id in route.segment_ids:
                if segment_id not in segment_ids:
                    raise ValueError(f"material route {route.id} references unknown segment: {segment_id}")
            for node_id in route.node_ids:
                if node_id not in node_ids:
                    raise ValueError(f"material route {route.id} references unknown node: {node_id}")
            for machine_id in route.machine_ids:
                if machine_id not in machine_ids:
                    raise ValueError(f"material route {route.id} references unknown machine: {machine_id}")
            if route.drop_zone_id is not None and route.drop_zone_id not in drop_zone_ids:
                raise ValueError(f"material route {route.id} references unknown drop zone: {route.drop_zone_id}")
            if route.outbound_zone_id is not None and route.outbound_zone_id not in drop_zone_ids:
                raise ValueError(f"material route {route.id} references unknown outbound zone: {route.outbound_zone_id}")
            if not route.material_classes and not route.commodity_targets:
                raise ValueError(f"material route {route.id} must match at least one material or commodity")

        reachable = set()
        frontier = [self.spawn_segment_id]
        while frontier:
            segment_id = frontier.pop()
            if segment_id in reachable:
                continue
            reachable.add(segment_id)
            frontier.extend(sorted(adjacency[segment_id] - reachable))
        if reachable != segment_ids:
            missing = sorted(segment_ids - reachable)
            raise ValueError(f"disconnected conveyor segments: {missing}")

        for node in self.routing_nodes:
            for segment_id in node.upstream_segment_ids:
                end_anchors[segment_id].add(f"node:{node.id}:upstream")
            for segment_id in node.downstream_segment_ids:
                start_anchors[segment_id].add(f"node:{node.id}:downstream")

        for segment in self.conveyor_segments:
            if not start_anchors[segment.id]:
                raise ValueError(
                    f"segment {segment.id} has no meaningful upstream start anchor; connect it to spawn, a node, or a machine output"
                )
            if not end_anchors[segment.id]:
                raise ValueError(
                    f"segment {segment.id} has no meaningful downstream endpoint; terminate it into a node, drop, or machine inlet"
                )

    def _require_facility(
        self,
        facility_id: str,
        facilities_by_id: dict[str, FacilityConfig],
        context: str,
    ) -> None:
        if facility_id not in facilities_by_id:
            raise ValueError(f"{context} references unknown facility: {facility_id}")

    @property
    def main_belt(self) -> BeltConfig:
        return self.mrf.main_belt

    @property
    def spawn(self) -> SpawnConfig:
        return self.mrf.spawn

    @property
    def camera(self) -> CameraConfig:
        return self.mrf.camera

    @property
    def environment(self) -> EnvironmentConfig:
        return self.mrf.environment

    @property
    def stations(self) -> tuple[StationConfig, ...]:
        return self.mrf.stations

    @property
    def conveyor_segments(self) -> tuple[ConveyorSegmentConfig, ...]:
        return self.mrf.conveyor_segments

    @property
    def routing_nodes(self) -> tuple[RoutingNodeConfig, ...]:
        return self.mrf.routing_nodes

    @property
    def machine_zones(self) -> tuple[MachineZoneConfig, ...]:
        return self.mrf.machine_zones

    @property
    def robot_cells(self) -> tuple[RobotCellConfig, ...]:
        return self.mrf.robot_cells

    @property
    def platforms(self) -> tuple[PlatformConfig, ...]:
        return self.mrf.platforms

    @property
    def material_routes(self) -> tuple[MaterialRouteConfig, ...]:
        return self.mrf.material_routes

    @property
    def drop_zones(self) -> tuple[DropZoneConfig, ...]:
        return self.mrf.drop_zones

    @property
    def perception_zones(self) -> tuple[PerceptionZoneConfig, ...]:
        return self.mrf.perception_zones

    @property
    def spawn_segment_id(self) -> str | None:
        return self.mrf.spawn_segment_id

    @property
    def subareas(self) -> tuple[str, ...]:
        return self.mrf.subareas

    @property
    def item_catalog(self) -> dict[str, ItemSpec]:
        return self.system.item_catalog

    @property
    def item_mix(self) -> dict[str, float]:
        return self.system.item_mix

    @property
    def normalized_item_mix(self) -> dict[str, float]:
        return self.system.normalized_item_mix

    @property
    def supported_streams(self) -> set[str]:
        return {"recycling", "organics", "trash", "construction_demolition"}

    @property
    def supported_source_types(self) -> set[str]:
        return {"home", "business"}

    @property
    def known_material_classes(self) -> set[str]:
        return {
            "paper",
            "plastic",
            "ferrous_metal",
            "nonferrous_metal",
            "glass",
            "organic",
            "mixed_c_and_d",
            "residue",
        }

    @property
    def supported_visual_profiles(self) -> set[str]:
        return {
            "auto",
            "bag",
            "bottle",
            "can",
            "carton",
            "chunk",
            "jug",
            "paper_stack",
        }

    @property
    def supported_environment_ids(self) -> set[str]:
        return {"simple_warehouse"}

    @property
    def supported_layout_presets(self) -> set[str]:
        return {
            "full_mrf",
            "edco_conveyor_segment_a",
            "dense_mrf_process_line",
            "recycling_facility_large_v2",
            "recycling_facility_large_v3",
            "sims_big_sort_video_v2",
        }

    @property
    def supported_facility_types(self) -> set[str]:
        return {
            "mrf",
            "compost",
            "anaerobic_digestion",
            "transfer_recovery",
            "disposal",
            "cdi_processing",
            "end_market",
        }

    @property
    def supported_perception_sensor_types(self) -> set[str]:
        return {"rgb_camera"}

    @property
    def supported_robot_types(self) -> set[str]:
        return {"pick_arm", "xarm6"}

    @property
    def capture_station_types(self) -> set[str]:
        return {"presort", "screen", "magnet", "optical_sorter", "manual_qc"}


def _parse_pose(data: dict) -> PoseConfig:
    return PoseConfig(
        position=tuple(float(v) for v in data["position"]),
        yaw_deg=float(data.get("yaw_deg", 0.0)),
    )
