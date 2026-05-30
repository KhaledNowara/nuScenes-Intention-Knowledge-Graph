from pathlib import Path

from rdflib import Literal
from rdflib.namespace import XSD

from domain.model import Dataset, Participant, Scene, SceneParticipant, Sequence, Trip
from graph_exporter.rdf_exporter import NUS, dataset_to_graph


def _ds_with_labels():
    p = Participant(participant_id="P", instance_token="i", track_id=None, type_label="car")
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
    sp.spatial_config = "LeftPreceding"
    sp.ttc_risk = "High"
    scene = Scene(scene_id="S1", index=1, timestamp="0", source="CAM_FRONT", ego_xy=(0.0, 0.0), ego_yaw=0.0)
    scene.participants.append(sp)
    seq = Sequence(sequence_id="Seq", scenes=[scene])
    trip = Trip(trip_id="T", sequence=seq, ego_vehicle_id="Ego", sequence_key="T")
    trip.participants[p.participant_id] = p
    return Dataset(name="fake", version="v0", trips=[trip])


def test_exporter_emits_spatial_and_ttc_triples():
    g = dataset_to_graph(_ds_with_labels())
    sp_uri = NUS["SP"]
    assert (sp_uri, NUS.spatialConfiguration, NUS["LeftPreceding"]) in g
    assert (sp_uri, NUS.ttcRisk, NUS["High"]) in g


def test_exporter_emits_category_triples():
    g = dataset_to_graph(_ds_with_labels())
    sp_uri = NUS["SP"]
    p_uri = NUS["P"]
    # hasCoarseCategory is asserted on the persistent Participant, not the per-frame SceneParticipant.
    assert (p_uri, NUS.hasCoarseCategory, NUS["Vehicle"]) in g
    # hasNuScenesCategory remains on the SceneParticipant (raw per-observation label).
    assert (sp_uri, NUS.hasNuScenesCategory, Literal("car", datatype=XSD.string)) in g


def test_exporter_merges_ontology_when_path_given(tmp_path):
    ontology_path = Path(__file__).resolve().parent.parent / "ontologies" / "ontology.ttl"
    g = dataset_to_graph(_ds_with_labels(), ontology_path=str(ontology_path))
    # Ontology declares SpatialZone class — presence confirms merge happened.
    triples_about_spatial_zone = list(g.triples((NUS.SpatialZone, None, None)))
    assert len(triples_about_spatial_zone) > 0
