"""Pure-math helpers for ego body-frame relations."""

from __future__ import annotations

import math
from typing import Tuple

from pyquaternion import Quaternion


def yaw_from_quaternion(q_wxyz: Tuple[float, float, float, float]) -> float:
    """Yaw angle (radians) from a (w, x, y, z) quaternion via pyquaternion."""
    return float(Quaternion(q_wxyz).yaw_pitch_roll[0])


def planar_distance(obj_xy: Tuple[float, float], ego_xy: Tuple[float, float]) -> float:
    """Euclidean distance in the x-y plane (meters)."""
    return math.hypot(obj_xy[0] - ego_xy[0], obj_xy[1] - ego_xy[1])


def body_frame_xy(
    obj_xy: Tuple[float, float],
    ego_xy: Tuple[float, float],
    ego_yaw: float,
) -> Tuple[float, float]:
    """Object position expressed in the ego body frame.

    Body frame convention: x_body forward (along ego heading), y_body left.
    Translate by -ego_xy, then rotate by -ego_yaw.
    """
    dx = obj_xy[0] - ego_xy[0]
    dy = obj_xy[1] - ego_xy[1]
    cos_y = math.cos(-ego_yaw)
    sin_y = math.sin(-ego_yaw)
    x_body = cos_y * dx - sin_y * dy
    y_body = sin_y * dx + cos_y * dy
    return (x_body, y_body)
