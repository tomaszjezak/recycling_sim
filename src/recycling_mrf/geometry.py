from __future__ import annotations

import math
from typing import Any

from recycling_mrf.config import ConveyorSegmentConfig


SegmentLike = ConveyorSegmentConfig | dict[str, Any]


def segment_start_position(segment: SegmentLike) -> tuple[float, float, float]:
    return _tuple3(_segment_pose(segment, "start_pose")["position"])


def segment_end_position(segment: SegmentLike) -> tuple[float, float, float]:
    return _tuple3(_segment_pose(segment, "end_pose")["position"])


def segment_width(segment: SegmentLike) -> float:
    if isinstance(segment, ConveyorSegmentConfig):
        return float(segment.width)
    return float(segment["width"])


def segment_thickness(segment: SegmentLike) -> float:
    if isinstance(segment, ConveyorSegmentConfig):
        return float(segment.thickness)
    return float(segment["thickness"])


def segment_length(segment: SegmentLike) -> float:
    if isinstance(segment, ConveyorSegmentConfig):
        return float(segment.length)
    start = segment_start_position(segment)
    end = segment_end_position(segment)
    return math.sqrt(sum((end[idx] - start[idx]) ** 2 for idx in range(3)))


def segment_center(segment: SegmentLike) -> tuple[float, float, float]:
    start = segment_start_position(segment)
    end = segment_end_position(segment)
    return tuple((start[idx] + end[idx]) / 2 for idx in range(3))


def segment_yaw_deg(segment: SegmentLike) -> float:
    start = segment_start_position(segment)
    end = segment_end_position(segment)
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))


def segment_incline_angle_deg(segment: SegmentLike) -> float:
    start = segment_start_position(segment)
    end = segment_end_position(segment)
    horizontal = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    return math.degrees(math.atan2(end[2] - start[2], max(horizontal, 1e-6)))


def clamp_lane_offset(segment: SegmentLike, lane_offset: float, edge_margin: float = 0.08) -> float:
    half_width = max(segment_width(segment) / 2 - edge_margin, 0.0)
    return max(-half_width, min(half_width, lane_offset))


def segment_axes(segment: SegmentLike) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    start = segment_start_position(segment)
    end = segment_end_position(segment)
    tangent = normalize(tuple(end[idx] - start[idx] for idx in range(3)))
    planar = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    if planar <= 1e-6:
        lateral = (0.0, 1.0, 0.0)
    else:
        lateral = (-(end[1] - start[1]) / planar, (end[0] - start[0]) / planar, 0.0)
    normal = normalize(cross(tangent, lateral))
    if normal[2] < 0:
        normal = scale(normal, -1.0)
        lateral = scale(lateral, -1.0)
    return tangent, lateral, normal


def segment_body_point(segment: SegmentLike, distance: float, lane_offset: float = 0.0) -> tuple[float, float, float]:
    start = segment_start_position(segment)
    tangent, lateral, _ = segment_axes(segment)
    clamped_distance = max(0.0, min(segment_length(segment), distance))
    clamped_lane = clamp_lane_offset(segment, lane_offset)
    return add(add(start, scale(tangent, clamped_distance)), scale(lateral, clamped_lane))


def segment_surface_point(segment: SegmentLike, distance: float, lane_offset: float = 0.0) -> tuple[float, float, float]:
    _, _, normal = segment_axes(segment)
    return add(segment_body_point(segment, distance, lane_offset), scale(normal, segment_thickness(segment) / 2))


def segment_surface_ratio_point(segment: SegmentLike, t: float, lane_offset: float = 0.0) -> tuple[float, float, float]:
    return segment_surface_point(segment, segment_length(segment) * max(0.0, min(1.0, t)), lane_offset)


def segment_item_pose(
    segment: SegmentLike,
    distance: float,
    lane_offset: float,
    item_height: float,
    clearance: float = 0.008,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    tangent, lateral, normal = segment_axes(segment)
    contact_point = segment_surface_point(segment, distance, lane_offset)
    position = add(contact_point, scale(normal, item_height / 2 + clearance))
    return position, tangent, lateral, normal, contact_point


def segment_debug_samples(segment: SegmentLike, count: int = 9) -> tuple[tuple[float, float, float], ...]:
    if count <= 1:
        return (segment_surface_ratio_point(segment, 0.0, 0.0),)
    return tuple(segment_surface_ratio_point(segment, idx / (count - 1), 0.0) for idx in range(count))


def add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(a[idx] + b[idx] for idx in range(3))


def subtract(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(a[idx] - b[idx] for idx in range(3))


def scale(vector: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return tuple(component * factor for component in vector)


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(a[idx] * b[idx] for idx in range(3))


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 1e-6:
        return (0.0, 0.0, 1.0)
    return tuple(component / length for component in vector)


def _segment_pose(segment: SegmentLike, pose_key: str) -> dict[str, Any]:
    if isinstance(segment, ConveyorSegmentConfig):
        pose = segment.start_pose if pose_key == "start_pose" else segment.end_pose
        return {"position": pose.position, "yaw_deg": pose.yaw_deg}
    pose = segment[pose_key]
    if not isinstance(pose, dict):
        raise TypeError(f"segment pose must be a dict, got {type(pose)!r}")
    return pose


def _tuple3(value: Any) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))
