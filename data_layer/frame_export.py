"""Render CAM_FRONT keyframes with GT boxes and optional motion labels."""

from __future__ import annotations

import os
from typing import Dict, Optional

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility

from .nuscenes_fetch import view_corners_2d


def render_cam_front_keyframe(
    nusc: NuScenes,
    cam_front_sample_data_token: str,
    out_path: str,
    motion_by_instance: Optional[Dict[str, str]] = None,
) -> None:
    """Draw projected 3D boxes and category labels; optional motion text from domain.

    motion_by_instance: instance_token -> Approaching|Receding|Stationary
    """
    data_path, boxes, intrinsic = nusc.get_sample_data(
        cam_front_sample_data_token,
        box_vis_level=BoxVisibility.ANY,
    )
    img_path = os.path.join(nusc.dataroot, data_path)
    img = cv2.imread(img_path)
    if img is None:
        return

    h, w = img.shape[:2]
    color = (0, 255, 0)

    for box in boxes:
        if not isinstance(box, Box):
            continue
        if intrinsic is not None:
            corners = view_corners_2d(box, intrinsic, w, h)
            if corners is not None:
                xs = corners[:, 0]
                ys = corners[:, 1]
                x1 = int(np.clip(np.min(xs), 0, w - 1))
                y1 = int(np.clip(np.min(ys), 0, h - 1))
                x2 = int(np.clip(np.max(xs), 0, w - 1))
                y2 = int(np.clip(np.max(ys), 0, h - 1))
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label_parts = []
        if box.token is not None:
            try:
                ann = nusc.get("sample_annotation", box.token)
                if "category_name" in ann:
                    short = ann["category_name"].split(".")[-1]
                else:
                    cat = nusc.get("category", ann["category_token"])
                    short = cat["name"].split(".")[-1]
                label_parts.append(short)
                inst = ann["instance_token"]
                if motion_by_instance and inst in motion_by_instance:
                    label_parts.append(motion_by_instance[inst])
            except Exception:
                pass

        if label_parts and intrinsic is not None:
            corners = view_corners_2d(box, intrinsic, w, h)
            if corners is not None:
                xs = corners[:, 0]
                ys = corners[:, 1]
                u = float(np.mean(xs))
                v = float(np.min(ys)) - 10
                u_i = int(np.clip(u, 0, w - 1))
                v_i = int(np.clip(v, 0, h - 1))
                text = " | ".join(label_parts)
                cv2.putText(
                    img,
                    text,
                    (u_i, v_i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


def export_bundle_frames(
    nusc: NuScenes,
    bundles,
    motion_per_bundle_scene: Optional[Dict[tuple[str, int], Dict[str, str]]] = None,
    out_root: str = "traces/annotated_frames",
) -> None:
    """For each keyframe in each bundle, write one PNG.

    motion_per_bundle_scene: (sequence_key, scene_index) -> instance_token -> motion
    scene_index is 1-based to match domain Scene.index
    """
    for bundle in bundles:
        for i, kf in enumerate(bundle.keyframes):
            scene_idx = i + 1
            motion: Optional[Dict[str, str]] = None
            if motion_per_bundle_scene:
                motion = motion_per_bundle_scene.get((bundle.sequence_key, scene_idx))
            rel = os.path.join(out_root, bundle.scene_name, f"scene_{scene_idx:03d}.png")
            render_cam_front_keyframe(nusc, kf.cam_front_sample_data_token, rel, motion)
