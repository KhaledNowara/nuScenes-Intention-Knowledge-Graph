from domain.model import Dataset, Participant, Scene, SceneParticipant, Sequence, Trip
from domain.relations.motion import compute_motion_states


def _sp(participant, scene_id, xy):
    sp = SceneParticipant(
        sp_id=f"{scene_id}_{participant.participant_id}",
        scene_id=scene_id,
        participant=participant,
        bbox=(0.0, 0.0, 0.0, 0.0),
        centroid=(0.0, 0.0),
        confidence=1.0,
        category="Vehicle",
        type_label="car",
    )
    sp.ground_obj_xy = xy
    return sp


def _ds(tracks_per_scene):
    """tracks_per_scene: list of dict {itok: (obj_xy, ego_xy)} per scene."""
    participants = {}
    scenes = []
    for i, snapshot in enumerate(tracks_per_scene):
        s = Scene(scene_id=f"S{i + 1}", index=i + 1, timestamp=str(i), source="CAM_FRONT")
        for itok, (obj_xy, ego_xy) in snapshot.items():
            s.ego_xy = ego_xy
            p = participants.setdefault(
                itok,
                Participant(participant_id=f"P_{itok}", instance_token=itok, track_id=None, type_label="car"),
            )
            s.participants.append(_sp(p, s.scene_id, obj_xy))
        scenes.append(s)
    seq = Sequence(sequence_id="Seq", scenes=scenes)
    trip = Trip(trip_id="T", sequence=seq, ego_vehicle_id="Ego", sequence_key="T")
    return Dataset(name="fake", version="v0", trips=[trip])


def test_approaching_when_distance_shrinks():
    ds = _ds([{"a": ((10.0, 0.0), (0.0, 0.0))}, {"a": ((5.0, 0.0), (0.0, 0.0))}])
    compute_motion_states(ds)
    sps = ds.trips[0].sequence.scenes[1].participants
    assert sps[0].motion == "Approaching"


def test_receding_when_distance_grows():
    ds = _ds([{"a": ((5.0, 0.0), (0.0, 0.0))}, {"a": ((10.0, 0.0), (0.0, 0.0))}])
    compute_motion_states(ds)
    assert ds.trips[0].sequence.scenes[1].participants[0].motion == "Receding"


def test_stationary_when_distance_change_within_threshold():
    ds = _ds([{"a": ((5.0, 0.0), (0.0, 0.0))}, {"a": ((5.1, 0.0), (0.0, 0.0))}])
    compute_motion_states(ds)
    assert ds.trips[0].sequence.scenes[1].participants[0].motion == "Stationary"


def test_first_appearance_has_no_motion():
    ds = _ds([{"a": ((5.0, 0.0), (0.0, 0.0))}])
    compute_motion_states(ds)
    assert ds.trips[0].sequence.scenes[0].participants[0].motion is None
