from domain.model import Dataset, Participant, Scene, SceneParticipant, Sequence, Trip
from domain.relations.ttc import DELTA_T_S, compute_ttc_risk


def _sp(itok, scene_id, xy):
    p = Participant(participant_id=f"P_{itok}", instance_token=itok, track_id=None, type_label="car")
    sp = SceneParticipant(
        sp_id=f"{scene_id}_{itok}",
        scene_id=scene_id,
        participant=p,
        bbox=(0.0, 0.0, 0.0, 0.0),
        centroid=(0.0, 0.0),
        confidence=1.0,
        category="Vehicle",
        type_label="car",
    )
    sp.ground_obj_xy = xy
    return sp


def _two_frame_dataset(obj_prev, obj_curr, ego_prev=(0.0, 0.0), ego_curr=(0.0, 0.0)):
    s1 = Scene(scene_id="S1", index=1, timestamp="0", source="CAM_FRONT", ego_xy=ego_prev, ego_yaw=0.0)
    s1.participants.append(_sp("a", "S1", obj_prev))
    s2 = Scene(scene_id="S2", index=2, timestamp="1", source="CAM_FRONT", ego_xy=ego_curr, ego_yaw=0.0)
    s2.participants.append(_sp("a", "S2", obj_curr))
    seq = Sequence(sequence_id="Seq", scenes=[s1, s2])
    trip = Trip(trip_id="T", sequence=seq, ego_vehicle_id="Ego", sequence_key="T")
    return Dataset(name="fake", version="v0", trips=[trip])


def test_high_risk_below_four_seconds():
    # TTC < 4 s → High. d_curr = 5 m, Δd = 5 m over 0.5 s → closing 10 m/s → TTC = 0.5 s.
    ds = _two_frame_dataset(obj_prev=(10.0, 0.0), obj_curr=(5.0, 0.0))
    compute_ttc_risk(ds)
    assert ds.trips[0].sequence.scenes[1].participants[0].ttc_risk == "High"


def test_medium_risk_between_four_and_ten_seconds():
    # TTC = 5 s ∈ [4, 10). d_curr = 25 m, Δd = 2.5 m over 0.5 s → closing 5 m/s → TTC = 5 s → Medium.
    ds = _two_frame_dataset(obj_prev=(27.5, 0.0), obj_curr=(25.0, 0.0))
    compute_ttc_risk(ds)
    assert ds.trips[0].sequence.scenes[1].participants[0].ttc_risk == "Medium"


def test_low_risk_at_least_ten_seconds():
    # TTC = 30 s ≥ 10 → Low. d_curr = 30 m, Δd = 0.5 m over 0.5 s → closing 1 m/s → TTC = 30 s.
    ds = _two_frame_dataset(obj_prev=(30.5, 0.0), obj_curr=(30.0, 0.0))
    compute_ttc_risk(ds)
    assert ds.trips[0].sequence.scenes[1].participants[0].ttc_risk == "Low"


def test_no_ttc_when_receding():
    ds = _two_frame_dataset(obj_prev=(5.0, 0.0), obj_curr=(10.0, 0.0))
    compute_ttc_risk(ds)
    assert ds.trips[0].sequence.scenes[1].participants[0].ttc_risk is None


def test_no_ttc_on_first_appearance():
    ds = _two_frame_dataset(obj_prev=(10.0, 0.0), obj_curr=(5.0, 0.0))
    compute_ttc_risk(ds)
    assert ds.trips[0].sequence.scenes[0].participants[0].ttc_risk is None


def test_no_ttc_when_closing_speed_below_threshold():
    # Δd = 0.01 m over 0.5 s → 0.02 m/s, below MIN_CLOSING_SPEED.
    ds = _two_frame_dataset(obj_prev=(10.00, 0.0), obj_curr=(9.99, 0.0))
    compute_ttc_risk(ds)
    assert ds.trips[0].sequence.scenes[1].participants[0].ttc_risk is None
