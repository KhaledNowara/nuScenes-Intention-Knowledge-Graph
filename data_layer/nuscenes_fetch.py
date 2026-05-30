"""Load nuScenes mini (or full) into neutral SceneBundle structures."""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility
from pyquaternion import Quaternion

from .records import AnnotationRecord, KeyframeRecord, SceneBundle


def sanitize_token(raw: str) -> str:
    """RDF-safe token fragment (hex nuScenes tokens stay mostly valid)."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in raw)


def sequence_key_from_scene_name(scene_name: str) -> str:
    return scene_name.replace("-", "_")


def fetch_scene_bundles(
    dataroot: str,
    version: str = "v1.0-mini",
    scene_names: Optional[List[str]] = None,
    verbose: bool = False,
    max_keyframes_per_scene: Optional[int] = None,
) -> tuple[NuScenes, List[SceneBundle]]:
    """Return NuScenes handle and list of SceneBundle (keyframes only per scene).

    If ``max_keyframes_per_scene`` is set, each scene stops after that many keyframes
    (time order along the sample chain).
    """
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
    bundles: List[SceneBundle] = []

    for scene in nusc.scene:
        name = scene["name"]
        if scene_names is not None and name not in scene_names:
            continue

        seq_key = sequence_key_from_scene_name(name)
        bundle = SceneBundle(scene_name=name, scene_token=scene["token"], sequence_key=seq_key)

        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            cam_tok = sample["data"]["CAM_FRONT"]
            sd = nusc.get("sample_data", cam_tok)
            if sd.get("is_key_frame", True):
                kf = _keyframe_from_sample(nusc, sample)
                bundle.keyframes.append(kf)
                if max_keyframes_per_scene is not None and len(bundle.keyframes) >= max_keyframes_per_scene:
                    break
            sample_token = sample["next"]

        bundles.append(bundle)

    return nusc, bundles


def _keyframe_from_sample(nusc: NuScenes, sample: dict) -> KeyframeRecord:
    cam_tok = sample["data"]["CAM_FRONT"]
    sd = nusc.get("sample_data", cam_tok)
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    ego_xy = (float(ego["translation"][0]), float(ego["translation"][1]))
    ego_yaw = float(Quaternion(ego["rotation"]).yaw_pitch_roll[0])

    img_path = os.path.join(nusc.dataroot, sd["filename"])
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read camera image: {img_path}")
    h, w = img.shape[:2]

    _path_sd, boxes, cam_intrinsic = nusc.get_sample_data(
        cam_tok,
        box_vis_level=BoxVisibility.ANY,
    )

    ann_tokens = nusc.field2token("sample_annotation", "sample_token", sample["token"])
    ann_by_token = {t: nusc.get("sample_annotation", t) for t in ann_tokens}

    box_by_ann: dict[str, Box] = {}
    for box in boxes:
        if box.token is not None and box.token in ann_by_token:
            box_by_ann[box.token] = box

    annotations: List[AnnotationRecord] = []
    visual_to_use = cam_intrinsic if cam_intrinsic is not None else np.eye(3)

    for ann_token, ann in ann_by_token.items():
        box = box_by_ann.get(ann_token)
        if "category_name" in ann:
            cat_name = ann["category_name"]
        else:
            cat_rec = nusc.get("category", ann["category_token"])
            cat_name = cat_rec["name"]

        bbox_px: Optional[tuple[int, int, int, int]] = None
        u_norm, v_norm = 0.5, 0.5

        if box is not None:
            corners = view_corners_2d(box, visual_to_use, w, h)
            if corners is not None:
                xs = corners[:, 0]
                ys = corners[:, 1]
                x1, y1 = int(np.clip(np.min(xs), 0, w - 1)), int(np.clip(np.min(ys), 0, h - 1))
                x2, y2 = int(np.clip(np.max(xs), 0, w - 1)), int(np.clip(np.max(ys), 0, h - 1))
                if x2 > x1 and y2 > y1:
                    bbox_px = (x1, y1, x2, y2)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    u_norm = max(0.0, min(1.0, cx / float(w)))
                    v_norm = max(0.0, min(1.0, cy / float(h)))
        else:
            u_norm, v_norm = 0.5, 0.5

        inst = nusc.get("instance", ann["instance_token"])
        annotations.append(
            AnnotationRecord(
                instance_token=inst["token"],
                annotation_token=ann["token"],
                category_name=cat_name,
                translation_xyz=(
                    float(ann["translation"][0]),
                    float(ann["translation"][1]),
                    float(ann["translation"][2]),
                ),
                centroid_u_norm=u_norm,
                centroid_v_norm=v_norm,
                bbox_pixels=bbox_px,
            )
        )

    ts = str(sample.get("timestamp", ""))
    return KeyframeRecord(
        sample_token=sample["token"],
        cam_front_sample_data_token=cam_tok,
        timestamp_str=ts,
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        image_width=w,
        image_height=h,
        annotations=annotations,
    )


def view_corners_2d(box: Box, cam_intrinsic: np.ndarray, im_width: int, im_height: int) -> Optional[np.ndarray]:
    """Project 3D box corners to pixel coordinates; shape (N, 2)."""
    corners_3d = box.corners()  # (3, 8)
    pc = cam_intrinsic @ corners_3d
    depths = pc[2, :]
    valid = depths > 1e-3
    if not np.any(valid):
        return None
    pc = pc[:, valid]
    depths = pc[2, :]
    u = pc[0, :] / depths
    v = pc[1, :] / depths
    return np.column_stack([u, v])
