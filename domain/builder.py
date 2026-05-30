"""Build Dataset from neutral SceneBundle records."""

from __future__ import annotations

from data_layer.nuscenes_fetch import sanitize_token
from data_layer.records import SceneBundle

from .category_map import ontology_type_for_category
from .model import Dataset, Participant, Point2D, Scene, SceneParticipant, Sequence, Trip


def build_dataset(
    bundles: list[SceneBundle],
    dataset_name: str = "nuScenes",
    dataset_version: str = "v1.0-mini",
) -> Dataset:
    ds = Dataset(name=dataset_name, version=dataset_version)
    for bundle in bundles:
        trip = _trip_from_bundle(bundle)
        ds.trips.append(trip)
    return ds


def _trip_from_bundle(bundle: SceneBundle) -> Trip:
    sk = bundle.sequence_key
    trip_id = f"Trip_{sk}"
    seq_id = f"Sequence_{sk}"
    ego_id = f"Ego_Vehicle_{sk}"
    sequence = Sequence(sequence_id=seq_id)
    trip = Trip(trip_id=trip_id, sequence=sequence, ego_vehicle_id=ego_id, sequence_key=sk)

    participants_by_instance: dict[str, Participant] = {}

    for frame_idx, kf in enumerate(bundle.keyframes):
        scene_num = frame_idx + 1
        scene_id = f"Scene_{sk}_{scene_num:03d}"
        scene = Scene(
            scene_id=scene_id,
            index=scene_num,
            timestamp=kf.timestamp_str,
            source="CAM_FRONT",
            image_width=kf.image_width,
            image_height=kf.image_height,
            sample_token=kf.sample_token,
            cam_front_sample_data_token=kf.cam_front_sample_data_token,
            ego_xy=kf.ego_xy,
            ego_yaw=kf.ego_yaw,
        )

        w, h = kf.image_width, kf.image_height

        for ann in kf.annotations:
            itok = ann.instance_token
            cat_bucket = ontology_type_for_category(ann.category_name)
            phys_id = f"Participant_{sanitize_token(itok)}"

            if itok not in participants_by_instance:
                participants_by_instance[itok] = Participant(
                    participant_id=phys_id,
                    instance_token=itok,
                    track_id=None,
                    type_label=ann.category_name,
                )
                trip.participants[phys_id] = participants_by_instance[itok]

            participant = participants_by_instance[itok]
            local_id = sanitize_token(itok)[-12:] if len(sanitize_token(itok)) > 12 else sanitize_token(itok)
            sp_id = f"SP_{sk}_{scene_num:03d}_{local_id}"

            u, v = ann.centroid_u_norm, ann.centroid_v_norm
            cx_px, cy_px = u * w, v * h
            bbox_norm = (0.0, 0.0, 0.0, 0.0)
            bbox_px = ann.bbox_pixels
            if bbox_px is not None:
                x1, y1, x2, y2 = bbox_px
                bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)

            point_id = f"Point2D_{sk}_{scene_num:03d}_{local_id}"
            position = Point2D(point_id=point_id, x=u, y=v)

            sp = SceneParticipant(
                sp_id=sp_id,
                scene_id=scene_id,
                participant=participant,
                bbox=bbox_norm,
                centroid=(u, v),
                confidence=1.0,
                category=cat_bucket,
                type_label=ann.category_name,
                position=position,
                bbox_pixels=bbox_px,
                ground_obj_xy=(ann.translation_xyz[0], ann.translation_xyz[1]),
            )
            scene.participants.append(sp)

        sequence.scenes.append(scene)

    for i, scene in enumerate(sequence.scenes):
        if i > 0:
            scene.previous_scene_id = sequence.scenes[i - 1].scene_id
        if i < len(sequence.scenes) - 1:
            scene.next_scene_id = sequence.scenes[i + 1].scene_id

    return trip


def motion_lookup_by_scene_key(dataset: Dataset) -> dict[tuple[str, int], dict[str, str]]:
    """(sequence_key, scene_index) -> instance_token -> motion label."""
    out: dict[tuple[str, int], dict[str, str]] = {}
    for trip in dataset.trips:
        sk = trip.sequence_key
        for scene in trip.sequence.scenes:
            m: dict[str, str] = {}
            for sp in scene.participants:
                if sp.motion:
                    m[sp.participant.instance_token] = sp.motion
            if m:
                out[(sk, scene.index)] = m
    return out
