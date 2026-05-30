"""Neutral fetch records from nuScenes (no domain types)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class AnnotationRecord:
    instance_token: str
    annotation_token: str
    category_name: str
    translation_xyz: Tuple[float, float, float]
    centroid_u_norm: float
    centroid_v_norm: float
    bbox_pixels: Optional[Tuple[int, int, int, int]]


@dataclass
class KeyframeRecord:
    sample_token: str
    cam_front_sample_data_token: str
    timestamp_str: str
    ego_xy: Tuple[float, float]
    ego_yaw: float
    image_width: int
    image_height: int
    annotations: List[AnnotationRecord] = field(default_factory=list)


@dataclass
class SceneBundle:
    scene_name: str
    scene_token: str
    sequence_key: str
    keyframes: List[KeyframeRecord] = field(default_factory=list)
