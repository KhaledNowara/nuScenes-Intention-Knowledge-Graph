"""Ego-centric time-to-collision risk label."""

from __future__ import annotations

from domain.model import Dataset

from .geometry import planar_distance
from .tracking import iter_consecutive_appearances

DELTA_T_S = 0.5          # nuScenes keyframes at 2 Hz
MIN_CLOSING_SPEED = 0.1  # m/s; below this → not at risk
# Discrete bins aligned with Manzour et al. (arXiv:2312.06336): High < 4 s;
# Medium ∈ [4, 10) s; Low ≥ 10 s.
TTC_HIGH_UPPER_S = 4.0   # seconds; label High iff TTC < this
TTC_MEDIUM_UPPER_S = 10.0  # seconds; Medium iff TTC < this (and ≥ High bound)


def compute_ttc_risk(dataset: Dataset) -> None:
    """Set SceneParticipant.ttc_risk from one-step closing speed and planar TTC."""
    for _trip, scene_prev, sp_prev, scene_curr, sp_curr in iter_consecutive_appearances(dataset):
        d_prev = planar_distance(sp_prev.ground_obj_xy, scene_prev.ego_xy)
        d_curr = planar_distance(sp_curr.ground_obj_xy, scene_curr.ego_xy)
        closing = (d_prev - d_curr) / DELTA_T_S
        if closing < MIN_CLOSING_SPEED:
            continue
        ttc = d_curr / closing
        if ttc < TTC_HIGH_UPPER_S:
            sp_curr.ttc_risk = "High"
        elif ttc < TTC_MEDIUM_UPPER_S:
            sp_curr.ttc_risk = "Medium"
        else:
            sp_curr.ttc_risk = "Low"
