from domain.model import Dataset, Participant, Scene, SceneParticipant, Sequence, Trip
from domain.relations.tracking import iter_consecutive_appearances


def _make_sp(participant, scene_id, xy):
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


def _make_dataset():
    p_a = Participant(participant_id="Pa", instance_token="itok_a", track_id=None, type_label="car")
    p_b = Participant(participant_id="Pb", instance_token="itok_b", track_id=None, type_label="car")
    scenes = []
    for i in range(3):
        s = Scene(
            scene_id=f"S{i + 1}",
            index=i + 1,
            timestamp=str(i),
            source="CAM_FRONT",
            ego_xy=(float(i), 0.0),
            ego_yaw=0.0,
        )
        s.participants.append(_make_sp(p_a, s.scene_id, (float(i) + 5.0, 0.0)))
        if i != 1:
            s.participants.append(_make_sp(p_b, s.scene_id, (float(i) - 2.0, 1.0)))
        scenes.append(s)
    seq = Sequence(sequence_id="Seq", scenes=scenes)
    trip = Trip(trip_id="T", sequence=seq, ego_vehicle_id="Ego", sequence_key="T")
    return Dataset(name="fake", version="v0", trips=[trip])


def test_iter_skips_first_appearance():
    ds = _make_dataset()
    tokens = [sp_curr.participant.instance_token for _, _, _, _, sp_curr in iter_consecutive_appearances(ds)]
    # instance A appears at S1,S2,S3 → yields pairs (S1,S2) and (S2,S3) → 2 for A
    # instance B appears at S1,S3            → yields pair (S1,S3)         → 1 for B
    assert tokens.count("itok_a") == 2
    assert tokens.count("itok_b") == 1


def test_iter_orders_by_scene_index():
    ds = _make_dataset()
    pairs = [
        (scene_prev.index, scene_curr.index)
        for _, scene_prev, _, scene_curr, _ in iter_consecutive_appearances(ds)
    ]
    for prev_idx, curr_idx in pairs:
        assert curr_idx > prev_idx
