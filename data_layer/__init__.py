"""nuScenes fetch and optional annotated frame export (cv2 only when exporting frames)."""

from __future__ import annotations

from .nuscenes_fetch import fetch_scene_bundles, sanitize_token, sequence_key_from_scene_name
from .records import AnnotationRecord, KeyframeRecord, SceneBundle

__all__ = [
    "AnnotationRecord",
    "KeyframeRecord",
    "SceneBundle",
    "export_bundle_frames",
    "fetch_scene_bundles",
    "render_cam_front_keyframe",
    "sanitize_token",
    "sequence_key_from_scene_name",
]


def __getattr__(name: str):
    if name == "export_bundle_frames":
        from .frame_export import export_bundle_frames

        return export_bundle_frames
    if name == "render_cam_front_keyframe":
        from .frame_export import render_cam_front_keyframe

        return render_cam_front_keyframe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
