import math

import pytest

from domain.relations.geometry import body_frame_xy, planar_distance, yaw_from_quaternion


def test_planar_distance_is_euclidean():
    assert planar_distance((3.0, 4.0), (0.0, 0.0)) == pytest.approx(5.0)


def test_yaw_from_quaternion_identity_is_zero():
    assert yaw_from_quaternion((1.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0)


def test_yaw_from_quaternion_ninety_degrees_z():
    q = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    assert yaw_from_quaternion(q) == pytest.approx(math.pi / 2.0)


def test_body_frame_xy_zero_yaw_is_translation():
    bx, by = body_frame_xy((10.0, 5.0), (2.0, 3.0), 0.0)
    assert bx == pytest.approx(8.0)
    assert by == pytest.approx(2.0)


def test_body_frame_xy_rotates_by_negative_yaw():
    bx, by = body_frame_xy((0.0, 10.0), (0.0, 0.0), math.pi / 2.0)
    assert bx == pytest.approx(10.0)
    assert by == pytest.approx(0.0)
