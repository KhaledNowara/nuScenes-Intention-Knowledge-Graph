"""EXP-005 domain model — Trip, Sequence, Scene, Participant, SceneParticipant, Dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Participant:
    """Physical entity across scenes; identity from nuScenes instance_token."""

    participant_id: str  # RDF-safe id, e.g. Participant_<sanitized_instance>
    instance_token: str  # raw nuScenes instance_token
    track_id: Optional[int]  # None for GT nuScenes (EXP-004 used int track_id)
    type_label: str  # nuScenes category name


@dataclass
class Point2D:
    point_id: str
    x: float  # normalized image [0, 1]
    y: float


@dataclass
class SceneParticipant:
    sp_id: str
    scene_id: str
    participant: Participant
    bbox: Tuple[float, float, float, float]  # normalized xyxy placeholder / legacy
    centroid: Tuple[float, float]  # normalized u, v
    confidence: float
    category: str  # ontology bucket
    type_label: str
    position: Optional[Point2D] = None
    motion: Optional[str] = None
    spatial_config: Optional[str] = None
    ttc_risk: Optional[str] = None
    bbox_pixels: Optional[Tuple[int, int, int, int]] = None
    ground_obj_xy: Tuple[float, float] = (0.0, 0.0)  # object center x,y map/global (meters)


@dataclass
class Scene:
    scene_id: str
    index: int
    timestamp: str
    source: str
    participants: List[SceneParticipant] = field(default_factory=list)
    next_scene_id: Optional[str] = None
    previous_scene_id: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    sample_token: str = ""
    cam_front_sample_data_token: str = ""
    ego_xy: Tuple[float, float] = (0.0, 0.0)  # ego translation x,y global (meters)
    ego_yaw: float = 0.0


@dataclass
class Sequence:
    sequence_id: str
    scenes: List[Scene] = field(default_factory=list)


@dataclass
class Trip:
    trip_id: str
    sequence: Sequence
    ego_vehicle_id: str
    sequence_key: str
    participants: Dict[str, Participant] = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    version: str
    trips: List[Trip] = field(default_factory=list)
