"""Serialize Dataset to Turtle and SPARQL INSERT (aligned with EXP-003/004)."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD

from domain.model import Dataset, Participant, Scene, SceneParticipant, Sequence, Trip

NUS = Namespace("http://www.nuscenes.org/nuScenes/")
DATASET_ROOT_ID = "Dataset_root"


def _bind_prefixes(g: Graph) -> None:
    g.bind("", NUS)
    g.bind("nus", NUS)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)


def _add_trip_to_graph(g: Graph, trip: Trip) -> None:
    trip_uri = NUS[trip.trip_id]
    seq_uri = NUS[trip.sequence.sequence_id]
    ego_uri = NUS[trip.ego_vehicle_id]

    g.add((trip_uri, RDF.type, NUS.Trip))
    g.add((trip_uri, NUS.hasSequence, seq_uri))
    g.add((trip_uri, NUS.hasEgoVehicle, ego_uri))

    g.add((seq_uri, RDF.type, NUS.Sequence))
    g.add(
        (
            seq_uri,
            NUS.hasDescription,
            Literal("nuScenes keyframe sequence (EXP-006)", datatype=XSD.string),
        )
    )

    g.add((ego_uri, RDF.type, NUS.EgoVehicle))

    for participant in trip.participants.values():
        _add_participant(g, participant)

    _add_scenes_and_participants(g, seq_uri, trip.sequence)


def trip_to_graph(trip: Trip) -> Graph:
    g = Graph()
    _bind_prefixes(g)
    _add_trip_to_graph(g, trip)
    return g


def dataset_to_graph(dataset: Dataset, ontology_path: Union[str, Path, None] = None) -> Graph:
    g = Graph()
    _bind_prefixes(g)

    if ontology_path is not None:
        g.parse(str(ontology_path), format="turtle")

    dataset_uri = NUS[DATASET_ROOT_ID]
    g.add((dataset_uri, RDF.type, NUS.Dataset))
    g.add((dataset_uri, NUS.hasName, Literal(dataset.name, datatype=XSD.string)))
    g.add((dataset_uri, NUS.hasVersion, Literal(dataset.version, datatype=XSD.string)))
    g.add((dataset_uri, NUS["sequenceCount"], Literal(len(dataset.trips), datatype=XSD.integer)))

    for trip in dataset.trips:
        trip_uri = NUS[trip.trip_id]
        g.add((dataset_uri, NUS.hasTrip, trip_uri))
        _add_trip_to_graph(g, trip)

    return g


def _add_participant(g: Graph, participant: Participant) -> None:
    p_uri = NUS[participant.participant_id]
    g.add((p_uri, RDF.type, NUS.Participant))
    if participant.track_id is not None:
        g.add((p_uri, NUS.hasTrackId, Literal(participant.track_id, datatype=XSD.integer)))


def _add_scenes_and_participants(g: Graph, seq_uri, sequence: Sequence) -> None:
    for scene in sequence.scenes:
        s_uri = NUS[scene.scene_id]
        g.add((s_uri, RDF.type, NUS.Scene))
        g.add((seq_uri, NUS.hasScene, s_uri))
        g.add((s_uri, NUS.hasFrameIndex, Literal(scene.index, datatype=XSD.integer)))
        g.add((s_uri, NUS.hasTimestamp, Literal(scene.timestamp, datatype=XSD.string)))

        if scene.previous_scene_id is not None:
            g.add((s_uri, NUS.hasPreviousScene, NUS[scene.previous_scene_id]))
        if scene.next_scene_id is not None:
            g.add((s_uri, NUS.hasNextScene, NUS[scene.next_scene_id]))

        for sp in scene.participants:
            sp_uri = NUS[sp.sp_id]
            p_uri = NUS[sp.participant.participant_id]
            g.add((sp_uri, RDF.type, NUS.SceneParticipant))
            g.add((s_uri, NUS.hasSceneParticipant, sp_uri))
            g.add((sp_uri, NUS.isSceneParticipantOf, p_uri))

            if sp.category:
                g.add((p_uri, NUS.hasCoarseCategory, NUS[sp.category]))
            if sp.type_label:
                g.add(
                    (
                        sp_uri,
                        NUS.hasNuScenesCategory,
                        Literal(sp.type_label, datatype=XSD.string),
                    )
                )

            if sp.position is not None:
                pos = sp.position
                pos_uri = NUS[pos.point_id]
                g.add((pos_uri, RDF.type, NUS.Point2D))
                g.add((pos_uri, NUS.hasX, Literal(pos.x, datatype=XSD.float)))
                g.add((pos_uri, NUS.hasY, Literal(pos.y, datatype=XSD.float)))
                g.add((sp_uri, NUS.hasPosition, pos_uri))

            if sp.motion is not None:
                g.add((sp_uri, NUS.motionTowardEgo, NUS[sp.motion]))

            if sp.spatial_config is not None:
                g.add((sp_uri, NUS.spatialConfiguration, NUS[sp.spatial_config]))

            if sp.ttc_risk is not None:
                g.add((sp_uri, NUS.ttcRisk, NUS[sp.ttc_risk]))


def _graph_for_root(root: Union[Trip, Dataset], ontology_path: Union[str, Path, None] = None) -> Graph:
    if isinstance(root, Dataset):
        return dataset_to_graph(root, ontology_path=ontology_path)
    return trip_to_graph(root)


def export_ttl(root: Union[Trip, Dataset], path: str, ontology_path: Union[str, Path, None] = None) -> None:
    g = _graph_for_root(root, ontology_path=ontology_path)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_path), format="turtle")


def export_sparql_insert(root: Union[Trip, Dataset], path: str, ontology_path: Union[str, Path, None] = None) -> None:
    g = _graph_for_root(root, ontology_path=ontology_path)
    ttl_data = g.serialize(format="turtle")
    lines = ["INSERT DATA {"]
    lines.append(ttl_data.decode("utf-8") if isinstance(ttl_data, bytes) else ttl_data)
    lines.append("}")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_dataset(
    dataset: Dataset,
    ttl_path: str,
    sparql_path: str,
    ontology_path: Union[str, Path, None] = None,
) -> None:
    export_ttl(dataset, ttl_path, ontology_path=ontology_path)
    export_sparql_insert(dataset, sparql_path, ontology_path=ontology_path)
