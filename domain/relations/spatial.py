"""8-zone ego-relative spatial configuration."""

from __future__ import annotations

import math
from typing import Optional

from domain.model import Dataset

from .geometry import body_frame_xy

LONG_BAND_M = 3.0         # |x_body| ≤ 3.0 m → alongside
LANE_HALF_WIDTH_M = 1.75  # |y_body| < 1.75 m → same lane
MAX_RANGE_M = 50.0


def _classify_zone(x_body: float, y_body: float) -> Optional[str]:
    if y_body >= LANE_HALF_WIDTH_M:
        lat = "Left"
    elif y_body <= -LANE_HALF_WIDTH_M:
        lat = "Right"
    else:
        lat = "Same"

    if x_body > LONG_BAND_M:
        lon = "Preceding"
    elif x_body < -LONG_BAND_M:
        lon = "Following"
    else:
        lon = "Alongside"

    if lat == "Same":
        return None if lon == "Alongside" else lon
    return f"{lat}{lon}"


def compute_spatial_configuration(dataset: Dataset) -> None:
    """Set SceneParticipant.spatial_config for each participant within MAX_RANGE_M."""
    for trip in dataset.trips:
        for scene in trip.sequence.scenes:
            for sp in scene.participants:
                x_body, y_body = body_frame_xy(sp.ground_obj_xy, scene.ego_xy, scene.ego_yaw)
                if math.hypot(x_body, y_body) > MAX_RANGE_M:
                    continue
                label = _classify_zone(x_body, y_body)
                if label is not None:
                    sp.spatial_config = label
