"""Ego-relative motion from ground-plane distance deltas between consecutive keyframes."""

from __future__ import annotations

from domain.model import Dataset

from .geometry import planar_distance
from .tracking import iter_consecutive_appearances

DISTANCE_THRESHOLD = 0.5  # meters; |delta distance| below this → Stationary


def compute_motion_states(dataset: Dataset) -> None:
    """Set SceneParticipant.motion from consecutive planar ego–object distances."""
    for _trip, scene_prev, sp_prev, scene_curr, sp_curr in iter_consecutive_appearances(dataset):
        d_prev = planar_distance(sp_prev.ground_obj_xy, scene_prev.ego_xy)
        d_curr = planar_distance(sp_curr.ground_obj_xy, scene_curr.ego_xy)
        delta = d_curr - d_prev
        if delta < -DISTANCE_THRESHOLD:
            sp_curr.motion = "Approaching"
        elif delta > DISTANCE_THRESHOLD:
            sp_curr.motion = "Receding"
        else:
            sp_curr.motion = "Stationary"
