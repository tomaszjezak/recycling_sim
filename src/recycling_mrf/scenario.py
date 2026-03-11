from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from recycling_mrf.config import (
    ConveyorSegmentConfig,
    ItemSpec,
    MaterialRouteConfig,
    SimulationConfig,
    StationConfig,
)
from recycling_mrf.scene import build_scene_definition


@dataclass(frozen=True)
class SpawnEvent:
    item_id: str
    item_type: str
    stream: str
    spawn_time: float
    position: tuple[float, float, float]
    yaw_deg: float
    size: tuple[float, float, float]
    mass: float
    color: tuple[float, float, float]
    shape: str
    material_class: str
    visual_profile: str
    commodity_target: str | None
    spawn_segment_id: str | None = None


@dataclass(frozen=True)
class StationEvent:
    station_name: str
    station_type: str
    enter_time: float
    decision: str


@dataclass(frozen=True)
class RoutingNodeEvent:
    node_id: str
    node_type: str
    timestamp: float
    decision: str


@dataclass(frozen=True)
class MachineEvent:
    machine_id: str
    machine_type: str
    enter_time: float
    exit_time: float


@dataclass(frozen=True)
class LifecycleStageEvent:
    stage: str
    timestamp: float
    facility_id: str
    decision: str


@dataclass(frozen=True)
class MaterialUnit:
    unit_id: str
    item_type: str
    stream: str
    source_id: str
    material_class: str
    commodity_target: str | None
    lifecycle: list[LifecycleStageEvent]
    final_status: str
    path_segments: list[str]
    node_events: list[RoutingNodeEvent]
    machine_events: list[MachineEvent]
    drop_zone_id: str | None
    outbound_zone_id: str | None


@dataclass(frozen=True)
class ItemLifecycle:
    item_id: str
    item_type: str
    stream: str
    material_class: str
    commodity_target: str | None
    source_id: str
    spawn: SpawnEvent
    route: list[StationEvent]
    lifecycle: list[LifecycleStageEvent]
    final_status: str
    path_segments: list[str]
    node_events: list[RoutingNodeEvent]
    machine_events: list[MachineEvent]
    drop_zone_id: str | None
    outbound_zone_id: str | None


@dataclass(frozen=True)
class EpisodePlan:
    seed: int
    duration: float
    belt_speed: float
    render_dt: float
    item_mix: dict[str, float]
    system: dict
    facility_catalog: list[dict]
    facility: dict
    camera_pose: dict[str, tuple[float, float, float] | tuple[int, int]]
    summaries: dict[str, dict]
    events: list[SpawnEvent]
    material_units: list[MaterialUnit]
    item_lifecycles: list[ItemLifecycle]

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "duration": self.duration,
            "belt_speed": self.belt_speed,
            "render_dt": self.render_dt,
            "item_mix": self.item_mix,
            "system": self.system,
            "facility_catalog": self.facility_catalog,
            "facility": self.facility,
            "camera_pose": self.camera_pose,
            "summaries": self.summaries,
            "events": [asdict(event) for event in self.events],
            "material_units": [asdict(unit) for unit in self.material_units],
            "item_lifecycles": [asdict(lifecycle) for lifecycle in self.item_lifecycles],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def generate_episode_plan(
    config: SimulationConfig,
    item_id_prefix: str = "",
    force_mainline_exit: bool = False,
) -> EpisodePlan:
    rng = random.Random(config.seed)
    weights = list(config.item_mix.values())
    item_names = list(config.item_mix.keys())
    source_weights = [source.share for source in config.system.sources]
    source_ids = [source.id for source in config.system.sources]
    recycling_events: list[SpawnEvent] = []
    recycling_lifecycles: list[ItemLifecycle] = []
    material_units: list[MaterialUnit] = []
    elapsed = 0.0
    item_index = 0
    scene = build_scene_definition(config)
    topology = _build_topology_maps(config)
    spawn_segment = topology["segments"].get(config.spawn_segment_id or "main_belt")
    x_start = -config.main_belt.length / 2 + 0.45

    while elapsed < config.episode_duration:
        interval = rng.expovariate(config.spawn.rate)
        elapsed += interval
        if elapsed >= config.episode_duration:
            break

        item_type = rng.choices(item_names, weights=weights, k=1)[0]
        source_id = rng.choices(source_ids, weights=source_weights, k=1)[0]
        spec = config.item_catalog[item_type]
        quantized_time = round(round(elapsed / config.physics_dt) * config.physics_dt, 4)
        if quantized_time >= config.episode_duration:
            break

        unit_id = f"{item_id_prefix}unit_{item_index:04d}"
        if spec.stream == "recycling":
            lane_offset = rng.uniform(-config.spawn.lane_jitter, config.spawn.lane_jitter)
            yaw_deg = rng.uniform(-config.spawn.yaw_jitter_deg, config.spawn.yaw_jitter_deg)
            if spawn_segment is not None:
                spawn_position = _spawn_position_on_segment(config, spawn_segment, lane_offset, spec.size[2])
                spawn_segment_id = spawn_segment.id
            else:
                z = config.main_belt.height / 2 + config.spawn.drop_height + spec.size[2] / 2
                spawn_position = (round(x_start, 4), round(lane_offset, 4), round(z, 4))
                spawn_segment_id = None
            spawn_event = SpawnEvent(
                item_id=f"{item_id_prefix}item_{item_index:04d}",
                item_type=item_type,
                stream=spec.stream,
                spawn_time=quantized_time,
                position=spawn_position,
                yaw_deg=round(yaw_deg, 3),
                size=spec.size,
                mass=spec.mass,
                color=spec.color,
                shape=spec.shape,
                material_class=spec.material_class,
                visual_profile=spec.visual_profile,
                commodity_target=spec.commodity_target,
                spawn_segment_id=spawn_segment_id,
            )
            lifecycle, material_unit = _build_recycling_lifecycle(
                config=config,
                topology=topology,
                unit_id=unit_id,
                source_id=source_id,
                spawn_event=spawn_event,
                spec=spec,
                force_mainline_exit=force_mainline_exit,
            )
            recycling_events.append(spawn_event)
            recycling_lifecycles.append(lifecycle)
            material_units.append(material_unit)
        else:
            material_units.append(
                _build_non_recycling_unit(
                    config=config,
                    unit_id=unit_id,
                    source_id=source_id,
                    item_type=item_type,
                    spec=spec,
                    start_time=quantized_time,
                )
            )
        item_index += 1

    return EpisodePlan(
        seed=config.seed,
        duration=config.episode_duration,
        belt_speed=config.main_belt.speed,
        render_dt=config.render_dt,
        item_mix=config.normalized_item_mix,
        system=_system_snapshot(config),
        facility_catalog=[asdict(facility) for facility in config.system.facilities],
        facility=scene.to_dict(),
        camera_pose={
            "position": config.camera.position,
            "look_at": config.camera.look_at,
            "resolution": config.camera.resolution,
        },
        summaries=_build_stream_summaries(material_units),
        events=recycling_events,
        material_units=material_units,
        item_lifecycles=recycling_lifecycles,
    )


def _build_recycling_lifecycle(
    config: SimulationConfig,
    topology: dict[str, dict],
    unit_id: str,
    source_id: str,
    spawn_event: SpawnEvent,
    spec: ItemSpec,
    force_mainline_exit: bool,
) -> tuple[ItemLifecycle, MaterialUnit]:
    route: list[StationEvent] = []
    lifecycle: list[LifecycleStageEvent] = []
    node_events: list[RoutingNodeEvent] = []
    machine_events: list[MachineEvent] = []
    recycling_route = config.system.stream_routes["recycling"]

    generated_time = max(spawn_event.spawn_time - 0.9, 0.0)
    source_time = round(generated_time + 0.1, 4)
    collected_time = round(max(spawn_event.spawn_time - 0.45, source_time + 0.05), 4)
    delivered_time = round(max(spawn_event.spawn_time - 0.2, collected_time + 0.05), 4)
    tipped_time = round(max(spawn_event.spawn_time - 0.1, delivered_time + 0.03), 4)
    metered_time = spawn_event.spawn_time

    lifecycle.extend(
        [
            LifecycleStageEvent("generated", generated_time, source_id, "created_at_source"),
            LifecycleStageEvent("source_separated", source_time, source_id, f"assigned_to_{spec.stream}_stream"),
            LifecycleStageEvent("collected", collected_time, source_id, f"collected_{spec.stream}_route"),
            LifecycleStageEvent(
                "delivered_to_facility",
                delivered_time,
                recycling_route.initial_facility_id,
                "delivered_to_mrf",
            ),
            LifecycleStageEvent("tipped", tipped_time, recycling_route.initial_facility_id, "unloaded_on_tipping_floor"),
            LifecycleStageEvent(
                "metered_to_conveyor",
                metered_time,
                recycling_route.initial_facility_id,
                "fed_to_sort_line",
            ),
        ]
    )

    if topology["segments"] and not force_mainline_exit:
        material_route = resolve_material_route(config, spec)
        path_segments = list(material_route.segment_ids)
        station_name = material_route.station_name
        if station_name is not None:
            station = next((value for value in config.stations if value.name == station_name), None)
            if station is not None:
                enter_time = estimate_station_entry_time(config, spawn_event, station)
                route.append(
                    StationEvent(
                        station_name=station.name,
                        station_type=station.station_type,
                        enter_time=enter_time,
                        decision="captured" if material_route.drop_zone_id is not None else "continued_on_mainline",
                    )
                )
        timing = _build_path_timing(config, material_route, spawn_event.spawn_time)
        for node_id, timestamp in timing["node_times"].items():
            node = topology["nodes"][node_id]
            node_events.append(
                RoutingNodeEvent(
                    node_id=node_id,
                    node_type=node["node_type"],
                    timestamp=timestamp,
                    decision=_node_decision(material_route, node_id),
                )
            )
        for machine_id, window in timing["machine_windows"].items():
            machine = topology["machines"][machine_id]
            machine_events.append(
                MachineEvent(
                    machine_id=machine_id,
                    machine_type=machine["machine_type"],
                    enter_time=window[0],
                    exit_time=window[1],
                )
            )
        lifecycle.extend(_route_lifecycle_events(config, material_route, machine_events, node_events, spec))
        final_status = material_route.final_status
        item_lifecycle = ItemLifecycle(
            item_id=spawn_event.item_id,
            item_type=spawn_event.item_type,
            stream=spawn_event.stream,
            material_class=spec.material_class,
            commodity_target=spec.commodity_target,
            source_id=source_id,
            spawn=spawn_event,
            route=route,
            lifecycle=lifecycle,
            final_status=final_status,
            path_segments=path_segments,
            node_events=node_events,
            machine_events=machine_events,
            drop_zone_id=material_route.drop_zone_id,
            outbound_zone_id=material_route.outbound_zone_id,
        )
        material_unit = MaterialUnit(
            unit_id=unit_id,
            item_type=spawn_event.item_type,
            stream=spawn_event.stream,
            source_id=source_id,
            material_class=spec.material_class,
            commodity_target=spec.commodity_target,
            lifecycle=lifecycle,
            final_status=final_status,
            path_segments=path_segments,
            node_events=node_events,
            machine_events=machine_events,
            drop_zone_id=material_route.drop_zone_id,
            outbound_zone_id=material_route.outbound_zone_id,
        )
        return item_lifecycle, material_unit

    final_status = "mainline_exit"
    captured_station_name = None
    capture_time = None
    path_segments = [spawn_event.spawn_segment_id or "main_belt"]

    for station in config.stations:
        enter_time = estimate_station_entry_time(config, spawn_event, station)
        decision = "continued_on_mainline"
        if force_mainline_exit:
            decision = "forced_mainline_exit"
        elif should_capture_item(station, spec):
            decision = "captured"
            captured_station_name = station.name
            capture_time = enter_time
        route.append(
            StationEvent(
                station_name=station.name,
                station_type=station.station_type,
                enter_time=enter_time,
                decision=decision,
            )
        )
        if decision == "captured":
            break

    if captured_station_name is not None and capture_time is not None:
        final_status = _recycling_capture_outcome(
            config=config,
            lifecycle=lifecycle,
            route=route,
            captured_station_name=captured_station_name,
            capture_time=capture_time,
            spec=spec,
        )
    elif force_mainline_exit:
        exit_time = round(estimate_belt_exit_time(config, spawn_event), 4)
        lifecycle.append(
            LifecycleStageEvent(
                "manual_qc_checked",
                exit_time,
                recycling_route.initial_facility_id,
                "passed_all_capture_stations",
            )
        )
        lifecycle.append(
            LifecycleStageEvent(
                "mainline_exit",
                round(exit_time + 0.05, 4),
                recycling_route.initial_facility_id,
                "continued_past_segment_demo",
            )
        )
        final_status = "mainline_exit"
    else:
        residual_facility = recycling_route.residual_facility_id or recycling_route.initial_facility_id
        exit_time = round(estimate_belt_exit_time(config, spawn_event), 4)
        lifecycle.append(
            LifecycleStageEvent("manual_qc_checked", exit_time, recycling_route.initial_facility_id, "passed_all_capture_stations")
        )
        lifecycle.append(
            LifecycleStageEvent("residual_disposed", round(exit_time + 0.15, 4), residual_facility, "exited_as_residual")
        )
        final_status = "residual_disposed"

    item_lifecycle = ItemLifecycle(
        item_id=spawn_event.item_id,
        item_type=spawn_event.item_type,
        stream=spawn_event.stream,
        material_class=spec.material_class,
        commodity_target=spec.commodity_target,
        source_id=source_id,
        spawn=spawn_event,
        route=route,
        lifecycle=lifecycle,
        final_status=final_status,
        path_segments=path_segments,
        node_events=node_events,
        machine_events=machine_events,
        drop_zone_id=None,
        outbound_zone_id=None,
    )
    material_unit = MaterialUnit(
        unit_id=unit_id,
        item_type=spawn_event.item_type,
        stream=spawn_event.stream,
        source_id=source_id,
        material_class=spec.material_class,
        commodity_target=spec.commodity_target,
        lifecycle=lifecycle,
        final_status=final_status,
        path_segments=path_segments,
        node_events=node_events,
        machine_events=machine_events,
        drop_zone_id=None,
        outbound_zone_id=None,
    )
    return item_lifecycle, material_unit


def _route_lifecycle_events(
    config: SimulationConfig,
    material_route: MaterialRouteConfig,
    machine_events: list[MachineEvent],
    node_events: list[RoutingNodeEvent],
    spec: ItemSpec,
) -> list[LifecycleStageEvent]:
    recycling_route = config.system.stream_routes["recycling"]
    residual_facility = recycling_route.residual_facility_id or recycling_route.initial_facility_id
    events: list[LifecycleStageEvent] = []

    for machine_event in machine_events:
        stage_name = {
            "hopper": "metered_to_conveyor",
            "screen": "screen_separated",
            "trommel": "screen_separated",
            "disc_screen": "screen_separated",
            "optical_sorter": "optical_sorted",
            "magnet": "magnet_separated",
            "eddy_current": "manual_qc_checked",
            "baler": "baled_prepared",
        }.get(machine_event.machine_type, "entered_machine_zone")
        events.append(
            LifecycleStageEvent(
                stage_name,
                machine_event.exit_time,
                recycling_route.initial_facility_id,
                f"passed_{machine_event.machine_type}",
            )
        )

    if material_route.final_status == "residual_disposed":
        terminal_time = node_events[-1].timestamp + 0.1 if node_events else 0.1
        events.append(
            LifecycleStageEvent(
                "residual_disposed",
                round(terminal_time, 4),
                residual_facility,
                "disposed_after_process_route",
            )
        )
        return events

    commodity = spec.commodity_target or material_route.id
    bunker_time = node_events[-1].timestamp if node_events else (machine_events[-1].exit_time if machine_events else 0.1)
    events.extend(
        [
            LifecycleStageEvent(
                "commodity_accumulated",
                round(bunker_time, 4),
                recycling_route.initial_facility_id,
                f"accumulated_as_{commodity}",
            ),
            LifecycleStageEvent(
                "shipped_to_end_market",
                round(bunker_time + 0.18, 4),
                config.system.commodity_end_markets.get(
                    spec.commodity_target or "",
                    recycling_route.end_market_facility_id or recycling_route.initial_facility_id,
                ),
                f"shipped_{commodity}",
            ),
        ]
    )
    return events


def _recycling_capture_outcome(
    config: SimulationConfig,
    lifecycle: list[LifecycleStageEvent],
    route: list[StationEvent],
    captured_station_name: str,
    capture_time: float,
    spec: ItemSpec,
) -> str:
    recycling_route = config.system.stream_routes["recycling"]
    residual_facility = recycling_route.residual_facility_id or recycling_route.initial_facility_id
    captured_station = next(station for station in config.stations if station.name == captured_station_name)
    stage_name = {
        "presort": "presort_removed",
        "screen": "screen_separated",
        "magnet": "magnet_separated",
        "optical_sorter": "optical_sorted",
        "manual_qc": "manual_qc_checked",
    }.get(captured_station.station_type, "manual_qc_checked")
    decision = "removed_as_residual"
    if spec.commodity_target is not None and captured_station.station_type != "presort":
        decision = f"captured_for_{spec.commodity_target}"
    lifecycle.append(
        LifecycleStageEvent(stage_name, capture_time, recycling_route.initial_facility_id, decision)
    )

    if spec.commodity_target is None or spec.material_class == "residue":
        lifecycle.append(
            LifecycleStageEvent(
                "residual_disposed",
                round(capture_time + 0.15, 4),
                residual_facility,
                "disposed_after_sorting",
            )
        )
        return "residual_disposed"

    bunker_time = round(capture_time + 0.15, 4)
    baler_time = round(capture_time + 0.3, 4)
    ship_time = round(capture_time + 0.45, 4)
    end_market = config.system.commodity_end_markets.get(
        spec.commodity_target,
        recycling_route.end_market_facility_id or recycling_route.initial_facility_id,
    )
    lifecycle.extend(
        [
            LifecycleStageEvent(
                "commodity_accumulated",
                bunker_time,
                recycling_route.initial_facility_id,
                f"accumulated_as_{spec.commodity_target}",
            ),
            LifecycleStageEvent(
                "baled_prepared",
                baler_time,
                recycling_route.initial_facility_id,
                f"prepared_{spec.commodity_target}_shipment",
            ),
            LifecycleStageEvent(
                "shipped_to_end_market",
                ship_time,
                end_market,
                f"shipped_{spec.commodity_target}",
            ),
        ]
    )
    return "shipped_to_end_market"


def _build_non_recycling_unit(
    config: SimulationConfig,
    unit_id: str,
    source_id: str,
    item_type: str,
    spec: ItemSpec,
    start_time: float,
) -> MaterialUnit:
    route = config.system.stream_routes[spec.stream]
    lifecycle = [
        LifecycleStageEvent("generated", max(start_time - 0.6, 0.0), source_id, "created_at_source"),
        LifecycleStageEvent("source_separated", max(start_time - 0.48, 0.0), source_id, f"assigned_to_{spec.stream}_stream"),
        LifecycleStageEvent("collected", max(start_time - 0.24, 0.0), source_id, f"collected_{spec.stream}_route"),
        LifecycleStageEvent("delivered_to_facility", start_time, route.initial_facility_id, f"delivered_to_{spec.stream}_facility"),
    ]
    final_status = "delivered_to_facility"
    terminal_time = round(start_time + 0.2, 4)

    if spec.stream == "organics":
        facility_type = config.system.facilities_by_id[route.initial_facility_id].facility_type
        stage = "sent_to_anaerobic_digestion" if facility_type == "anaerobic_digestion" else "sent_to_compost"
        lifecycle.append(LifecycleStageEvent(stage, terminal_time, route.initial_facility_id, stage))
        final_status = stage
    elif spec.stream == "trash":
        lifecycle.append(
            LifecycleStageEvent("sent_to_transfer_recovery", terminal_time, route.initial_facility_id, "entered_transfer_recovery")
        )
        recoverable = spec.material_class in route.recover_material_classes
        if recoverable and route.recovered_facility_id is not None:
            lifecycle.append(
                LifecycleStageEvent("recovered", round(terminal_time + 0.15, 4), route.recovered_facility_id, "recovered_from_trash_stream")
            )
            final_status = "recovered"
        else:
            disposal_facility = route.residual_facility_id or route.initial_facility_id
            lifecycle.append(
                LifecycleStageEvent("disposed", round(terminal_time + 0.15, 4), disposal_facility, "disposed_after_transfer")
            )
            final_status = "disposed"
    elif spec.stream == "construction_demolition":
        lifecycle.append(
            LifecycleStageEvent("sent_to_cdi_processing", terminal_time, route.initial_facility_id, "sent_to_cdi_processing")
        )
        final_status = "sent_to_cdi_processing"

    return MaterialUnit(
        unit_id=unit_id,
        item_type=item_type,
        stream=spec.stream,
        source_id=source_id,
        material_class=spec.material_class,
        commodity_target=spec.commodity_target,
        lifecycle=lifecycle,
        final_status=final_status,
        path_segments=[],
        node_events=[],
        machine_events=[],
        drop_zone_id=None,
        outbound_zone_id=None,
    )


def resolve_material_route(config: SimulationConfig, spec: ItemSpec) -> MaterialRouteConfig:
    best_match = None
    best_score = -1
    for route in config.material_routes:
        score = 0
        if route.commodity_targets and spec.commodity_target in route.commodity_targets:
            score += 4
        elif route.commodity_targets:
            continue
        if route.material_classes and spec.material_class in route.material_classes:
            score += 2
        elif route.material_classes:
            continue
        if score > best_score:
            best_match = route
            best_score = score
    if best_match is None:
        raise ValueError(f"no material route matches {spec.material_class}/{spec.commodity_target}")
    return best_match


def should_capture_item(station: StationConfig, spec: ItemSpec) -> bool:
    if station.station_type == "presort":
        return spec.material_class in station.target_materials
    if station.station_type == "screen":
        return spec.screenable_2d and spec.material_class in station.target_materials
    if station.station_type == "magnet":
        return spec.magnetic or spec.material_class in station.target_materials
    if station.station_type == "optical_sorter":
        material_match = not station.target_materials or spec.material_class in station.target_materials
        commodity_match = not station.target_commodities or spec.commodity_target in station.target_commodities
        return material_match and commodity_match and spec.commodity_target is not None
    if station.station_type == "manual_qc":
        return (
            spec.material_class in station.target_materials
            or (
                spec.commodity_target is not None
                and spec.commodity_target in station.target_commodities
            )
        )
    return False


def estimate_station_entry_time(config: SimulationConfig, event: SpawnEvent, station: StationConfig) -> float:
    distance = station.x_range[0] - event.position[0]
    return round(event.spawn_time + max(distance / config.main_belt.speed, 0.0), 4)


def estimate_belt_exit_time(config: SimulationConfig, event: SpawnEvent) -> float:
    if config.conveyor_segments:
        route = resolve_material_route(config, config.item_catalog[event.item_type])
        timing = _build_path_timing(config, route, event.spawn_time)
        return timing["path_end_time"]
    distance = config.main_belt.length - (event.position[0] + config.main_belt.length / 2)
    return round(event.spawn_time + max(distance / config.main_belt.speed, 0.0), 4)


def build_metadata_frame_times(config: SimulationConfig) -> list[float]:
    frame_count = int(math.floor(config.episode_duration / config.render_dt)) + 1
    return [round(index * config.render_dt, 4) for index in range(frame_count)]


def describe_item(spec: ItemSpec) -> str:
    return f"{spec.stream}:{spec.material_class}:{spec.shape}:{spec.size}:{spec.mass}"


def _build_stream_summaries(material_units: list[MaterialUnit]) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for unit in material_units:
        stream_summary = summaries.setdefault(
            unit.stream,
            {
                "generated_count": 0,
                "delivered_count": 0,
                "captured_commodity_count": 0,
                "residual_count": 0,
                "terminal_destination_counts": {},
            },
        )
        stream_summary["generated_count"] += 1
        if any(stage.stage == "delivered_to_facility" for stage in unit.lifecycle):
            stream_summary["delivered_count"] += 1
        if unit.final_status in {"shipped_to_end_market", "recovered", "sent_to_compost", "sent_to_anaerobic_digestion", "sent_to_cdi_processing"}:
            stream_summary["captured_commodity_count"] += 1
        if unit.final_status in {"residual_disposed", "disposed"}:
            stream_summary["residual_count"] += 1
        stream_summary["terminal_destination_counts"][unit.final_status] = (
            stream_summary["terminal_destination_counts"].get(unit.final_status, 0) + 1
        )
    return summaries


def _system_snapshot(config: SimulationConfig) -> dict:
    return {
        "sources": [asdict(source) for source in config.system.sources],
        "stream_routes": {
            stream: asdict(route)
            for stream, route in config.system.stream_routes.items()
        },
        "commodity_end_markets": dict(config.system.commodity_end_markets),
    }


def _build_topology_maps(config: SimulationConfig) -> dict[str, dict]:
    return {
        "segments": {segment.id: segment for segment in config.conveyor_segments},
        "nodes": {
            node.id: {
                "node_type": node.node_type,
                "upstream_segment_ids": node.upstream_segment_ids,
                "downstream_segment_ids": node.downstream_segment_ids,
                "drop_zone_id": node.drop_zone_id,
            }
            for node in config.routing_nodes
        },
        "machines": {
            machine.id: {
                "machine_type": machine.machine_type,
                "input_segment_ids": machine.input_segment_ids,
                "output_segment_ids": machine.output_segment_ids,
            }
            for machine in config.machine_zones
        },
        "drop_zones": {
            zone.id: zone
            for zone in config.drop_zones
        },
    }


def _build_path_timing(config: SimulationConfig, route: MaterialRouteConfig, start_time: float) -> dict[str, object]:
    segment_map = {segment.id: segment for segment in config.conveyor_segments}
    node_map = {node.id: node for node in config.routing_nodes}
    machine_map = {machine.id: machine for machine in config.machine_zones}
    current = start_time
    segment_end_times: dict[str, float] = {}
    for segment_id in route.segment_ids:
        segment = segment_map[segment_id]
        current = round(current + segment.length / segment.belt_speed, 4)
        segment_end_times[segment_id] = current

    node_times: dict[str, float] = {}
    for node_id in route.node_ids:
        node = node_map[node_id]
        upstream_times = [segment_end_times[segment_id] for segment_id in node.upstream_segment_ids if segment_id in segment_end_times]
        node_times[node_id] = round(max(upstream_times) if upstream_times else start_time, 4)

    machine_windows: dict[str, tuple[float, float]] = {}
    for machine_id in route.machine_ids:
        machine = machine_map[machine_id]
        input_times = [
            segment_end_times[segment_id]
            for segment_id in machine.input_segment_ids
            if segment_id in segment_end_times
        ]
        output_times = [
            segment_end_times[segment_id]
            for segment_id in machine.output_segment_ids
            if segment_id in segment_end_times
        ]
        enter_time = round(min(input_times) if input_times else start_time, 4)
        exit_time = round(max(output_times) if output_times else enter_time + 0.05, 4)
        machine_windows[machine_id] = (enter_time, exit_time)

    return {
        "segment_end_times": segment_end_times,
        "node_times": node_times,
        "machine_windows": machine_windows,
        "path_end_time": round(segment_end_times[route.segment_ids[-1]], 4),
    }


def _spawn_position_on_segment(
    config: SimulationConfig,
    segment: ConveyorSegmentConfig,
    lane_offset: float,
    item_height: float,
) -> tuple[float, float, float]:
    start = segment.start_pose.position
    end = segment.end_pose.position
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    planar = math.sqrt(dx * dx + dy * dy)
    if planar <= 1e-6:
        right_x, right_y = 0.0, 1.0
    else:
        right_x = -dy / planar
        right_y = dx / planar
    position = (
        round(start[0] + right_x * lane_offset, 4),
        round(start[1] + right_y * lane_offset, 4),
        round(start[2] + config.spawn.drop_height + item_height / 2, 4),
    )
    return position


def _node_decision(route: MaterialRouteConfig, node_id: str) -> str:
    if route.drop_zone_id is not None and node_id == route.node_ids[-1]:
        return f"dropped_to_{route.drop_zone_id}"
    return f"routed_via_{node_id}"
