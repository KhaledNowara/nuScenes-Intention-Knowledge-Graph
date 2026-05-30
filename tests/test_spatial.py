import math

import pytest

from domain.model import Dataset, Participant, Scene, SceneParticipant, Sequence, Trip
from domain.relations.spatial import (
    LANE_HALF_WIDTH_M,
    LONG_BAND_M,
    MAX_RANGE_M,
    _classify_zone,
    compute_spatial_configuration,
)


def test_classify_preceding_when_ahead_same_lane():
    assert _classify_zone(LONG_BAND_M + 1.0, 0.0) == "Preceding"


def test_classify_following_when_behind_same_lane():
    assert _classify_zone(-(LONG_BAND_M + 1.0), 0.0) == "Following"


def test_classify_left_preceding():
    assert _classify_zone(LONG_BAND_M + 1.0, LANE_HALF_WIDTH_M + 0.5) == "LeftPreceding"


def test_classify_right_following():
    assert _classify_zone(-(LONG_BAND_M + 1.0), -(LANE_HALF_WIDTH_M + 0.5)) == "RightFollowing"


def test_classify_left_alongside():
    assert _classify_zone(0.0, LANE_HALF_WIDTH_M + 0.5) == "LeftAlongside"


def test_classify_same_alongside_is_none():
    assert _classify_zone(0.0, 0.0) is None


def _make_trivial_dataset(x_body, y_body, dist_override=None):
    p = Participant(participant_id="P", instance_token="itok", track_id=None, type_label="car")
    sp = SceneParticipant(
        sp_id="SP",
        scene_id="S1",
        participant=p,
        bbox=(0.0, 0.0, 0.0, 0.0),
        centroid=(0.0, 0.0),
        confidence=1.0,
        category="Vehicle",
        type_label="car",
    )
    # Ego at origin, yaw 0 → body frame == global frame.
    sp.ground_obj_xy = (x_body, y_body)
    scene = Scene(scene_id="S1", index=1, timestamp="0", source="CAM_FRONT", ego_xy=(0.0, 0.0), ego_yaw=0.0)
    scene.participants.append(sp)
    seq = Sequence(sequence_id="Seq", scenes=[scene])
    trip = Trip(trip_id="T", sequence=seq, ego_vehicle_id="Ego", sequence_key="T")
    return Dataset(name="fake", version="v0", trips=[trip]), sp


def test_compute_assigns_label():
    ds, sp = _make_trivial_dataset(20.0, 0.0)
    compute_spatial_configuration(ds)
    assert sp.spatial_config == "Preceding"


def test_compute_skips_beyond_max_range():
    ds, sp = _make_trivial_dataset(MAX_RANGE_M + 5.0, 0.0)
    compute_spatial_configuration(ds)
    assert sp.spatial_config is None
