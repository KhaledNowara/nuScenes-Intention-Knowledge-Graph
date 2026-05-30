#!/usr/bin/env python3
"""EXP-006: generate scene study artifacts for a specific nuScenes scene and frame range.

Outputs (all under --out-dir):
    frames/frame_NNN.png      — annotated CAM_FRONT images
    scene_kg.ttl              — full KG for the scene (T-Box + A-Box)
    scene_study_meta.json     — high-level per-frame participant/label counts
    scene_study_details.json  — raw numerical values underlying every label
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

EXP_ROOT = Path(__file__).resolve().parent.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from data_layer import fetch_scene_bundles  # noqa: E402
from domain import (  # noqa: E402
    build_dataset,
    compute_motion_states,
    compute_spatial_configuration,
    compute_ttc_risk,
)
from domain.model import Scene, Trip  # noqa: E402
from domain.relations.geometry import body_frame_xy, planar_distance  # noqa: E402
from graph_exporter import export_ttl  # noqa: E402

# TTC / motion constants (must mirror domain/relations/ttc.py and spatial.py)
_DELTA_T_S           = 0.5
_MIN_CLOSING_SPEED   = 0.1
_TTC_HIGH_UPPER_S    = 4.0
_TTC_MEDIUM_UPPER_S  = 10.0
_MOTION_THRESHOLD_M  = 0.5
_MAX_RANGE_M         = 50.0
_LONG_BAND_M         = 3.0
_LAT_BAND_M          = 1.75

# ── colour palette (BGR order for OpenCV) ───────────────────────────────────
_BOX_COLOUR: Dict[Optional[str], tuple] = {
    "High":   (0,   40, 220),
    "Medium": (0,  140, 255),
    "Low":    (50, 200,  50),
    None:     (180, 180, 180),
}

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_SCALE_CAT  = 0.50
_SCALE_INFO = 0.42
_TEXT_THICK = 1
_TEXT_COLOUR = (255, 255, 255)
_PAD = 3


def _short_cat(type_label: str) -> str:
    return type_label.split(".")[-1] if type_label else "?"


def _draw_annotation(img: np.ndarray, sp) -> None:
    if sp.bbox_pixels is None:
        return
    x1, y1, x2, y2 = sp.bbox_pixels
    colour = _BOX_COLOUR.get(sp.ttc_risk, _BOX_COLOUR[None])
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

    line1 = _short_cat(sp.type_label)
    line2 = f"{sp.motion or chr(8212)} | {sp.spatial_config or chr(8212)} | TTC:{sp.ttc_risk or chr(8212)}"

    (w1, h1), bl1 = cv2.getTextSize(line1, _FONT, _SCALE_CAT,  _TEXT_THICK)
    (w2, h2), bl2 = cv2.getTextSize(line2, _FONT, _SCALE_INFO, _TEXT_THICK)

    label_w = max(w1, w2) + 2 * _PAD
    label_h = h1 + bl1 + h2 + bl2 + 3 * _PAD
    lx, ly = x1, max(y1 - label_h - 2, 0)

    cv2.rectangle(img, (lx, ly), (lx + label_w, ly + label_h), colour, cv2.FILLED)
    cv2.putText(img, line1,
                (lx + _PAD, ly + _PAD + h1),
                _FONT, _SCALE_CAT, _TEXT_COLOUR, _TEXT_THICK, cv2.LINE_AA)
    cv2.putText(img, line2,
                (lx + _PAD, ly + _PAD + h1 + bl1 + _PAD + h2),
                _FONT, _SCALE_INFO, _TEXT_COLOUR, _TEXT_THICK, cv2.LINE_AA)


def _parse_frames(spec: str) -> List[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in spec.split(",")]


def _render_frame(nusc, scene: Scene, out_path: Path) -> dict:
    sd = nusc.get("sample_data", scene.cam_front_sample_data_token)
    img_path = os.path.join(nusc.dataroot, sd["filename"])
    img = cv2.imread(img_path)
    if img is None:
        print(f"  WARNING: cannot read {img_path}", file=sys.stderr)
        return {}
    for sp in scene.participants:
        _draw_annotation(img, sp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return {
        "frame_index":      scene.index,
        "participants":     len(scene.participants),
        "motion_labelled":  sum(1 for sp in scene.participants if sp.motion),
        "spatial_labelled": sum(1 for sp in scene.participants if sp.spatial_config),
        "ttc_labelled":     sum(1 for sp in scene.participants if sp.ttc_risk),
        "ttc_high":         sum(1 for sp in scene.participants if sp.ttc_risk == "High"),
        "ttc_medium":       sum(1 for sp in scene.participants if sp.ttc_risk == "Medium"),
        "ttc_low":          sum(1 for sp in scene.participants if sp.ttc_risk == "Low"),
    }


# ── Raw numerical details ────────────────────────────────────────────────────

def _r(v: float, n: int = 3) -> float:
    """Round to n decimals; return None-safe."""
    return round(v, n) if v is not None else None


def _compute_raw_details(trip: Trip, studied_indices: List[int]) -> dict:
    """Return a dict frame_index -> list-of-participant-detail-dicts.

    For each SceneParticipant in a studied frame, record:
      - the raw distances / delta that produced the motion label
      - the ego-body-frame coordinates and range that produced the spatial label
      - the closing speed and TTC estimate that produced the ttcRisk label
    """
    scenes: Dict[int, Scene] = {s.index: s for s in trip.sequence.scenes}

    # (frame_index, instance_token) → SceneParticipant
    sp_lookup: Dict[tuple, object] = {}
    for s in trip.sequence.scenes:
        for sp in s.participants:
            sp_lookup[(s.index, sp.participant.instance_token)] = sp

    result: Dict[str, list] = {}

    for idx in studied_indices:
        scene = scenes.get(idx)
        if scene is None:
            continue
        prev_scene = scenes.get(idx - 1)

        frame_records = []
        for sp in scene.participants:
            inst = sp.participant.instance_token
            d_curr = planar_distance(sp.ground_obj_xy, scene.ego_xy)

            # ── motion ──────────────────────────────────────────────────────
            if prev_scene and (idx - 1, inst) in sp_lookup:
                sp_prev = sp_lookup[(idx - 1, inst)]
                d_prev  = planar_distance(sp_prev.ground_obj_xy, prev_scene.ego_xy)
                delta_d = d_curr - d_prev
                motion_detail = {
                    "label":         sp.motion,
                    "d_prev_m":      _r(d_prev),
                    "d_curr_m":      _r(d_curr),
                    "delta_d_m":     _r(delta_d),
                    "threshold_m":   _MOTION_THRESHOLD_M,
                    "rule":          (
                        "Approaching if delta_d < -threshold; "
                        "Receding if delta_d > +threshold; else Stationary"
                    ),
                }
            else:
                d_prev = None
                motion_detail = {
                    "label": None,
                    "note":  "first_appearance — no previous keyframe",
                    "d_curr_m": _r(d_curr),
                }

            # ── spatial ──────────────────────────────────────────────────────
            x_body, y_body = body_frame_xy(sp.ground_obj_xy, scene.ego_xy, scene.ego_yaw)
            range_m = math.hypot(x_body, y_body)
            spatial_detail = {
                "label":              sp.spatial_config,
                "x_body_m":          _r(x_body),
                "y_body_m":          _r(y_body),
                "range_m":           _r(range_m),
                "excluded_range":    range_m > _MAX_RANGE_M,
                "thresholds": {
                    "max_range_m":   _MAX_RANGE_M,
                    "long_band_m":   _LONG_BAND_M,
                    "lat_band_m":    _LAT_BAND_M,
                },
                "rule": (
                    "y_body >= +1.75 → Left; y_body <= -1.75 → Right; else Same-lane. "
                    "x_body > +3.0 → Preceding; x_body < -3.0 → Following; else Alongside. "
                    "Same-lane + Alongside → no label."
                ),
            }

            # ── ttc ──────────────────────────────────────────────────────────
            if d_prev is not None:
                closing = (d_prev - d_curr) / _DELTA_T_S
                if closing >= _MIN_CLOSING_SPEED:
                    ttc_s = d_curr / closing
                    ttc_detail = {
                        "label":                  sp.ttc_risk,
                        "d_prev_m":               _r(d_prev),
                        "d_curr_m":               _r(d_curr),
                        "closing_speed_m_per_s":  _r(closing),
                        "ttc_s":                  _r(ttc_s),
                        "delta_t_s":              _DELTA_T_S,
                        "thresholds": {
                            "high_upper_s":   _TTC_HIGH_UPPER_S,
                            "medium_upper_s": _TTC_MEDIUM_UPPER_S,
                        },
                        "rule": "High if ttc < 4 s; Medium if 4 ≤ ttc < 10 s; Low if ttc ≥ 10 s",
                    }
                else:
                    ttc_detail = {
                        "label":                  None,
                        "closing_speed_m_per_s":  _r(closing),
                        "note":                   (
                            f"not emitted — closing speed {_r(closing)} m/s "
                            f"< threshold {_MIN_CLOSING_SPEED} m/s "
                            "(agent receding or nearly stationary)"
                        ),
                    }
            else:
                ttc_detail = {
                    "label": None,
                    "note":  "first_appearance — no previous keyframe",
                }

            frame_records.append({
                "sp_id":          sp.sp_id,
                "instance_token": inst,
                "type_label":     sp.type_label,
                "coarse_category": sp.category,
                "motion":   motion_detail,
                "spatial":  spatial_detail,
                "ttc":      ttc_detail,
            })

        result[str(idx)] = frame_records

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="EXP-006 scene study generator")
    p.add_argument("--dataroot",  type=str, required=True)
    p.add_argument("--version",   type=str, default="v1.0-mini")
    p.add_argument("--scene",     type=str, default="scene-1100",
                   help="nuScenes scene name (default: scene-1100)")
    p.add_argument("--frames",    type=str, default="19-22",
                   help="Frame index range e.g. '19-22' or '19,20,21,22' (1-based)")
    p.add_argument("--out-dir",   type=str, default="evaluation/results/scene_study")
    p.add_argument("--ontology",  type=str, default="ontologies/ontology.ttl")
    args = p.parse_args()

    frame_indices = _parse_frames(args.frames)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.scene} ...")
    nusc, bundles = fetch_scene_bundles(
        args.dataroot, version=args.version,
        scene_names=[args.scene], verbose=False,
    )
    if not bundles:
        print(f"ERROR: scene '{args.scene}' not found.", file=sys.stderr)
        return 1

    dataset = build_dataset(bundles, dataset_name="nuScenes", dataset_version=args.version)
    compute_motion_states(dataset)
    compute_spatial_configuration(dataset)
    compute_ttc_risk(dataset)

    trip = dataset.trips[0]
    total_frames = len(trip.sequence.scenes)
    print(f"Scene has {total_frames} keyframes. Studying frames: {frame_indices}")

    # ── export scene KG ───────────────────────────────────────────────────────
    ontology_path = (
        (EXP_ROOT / args.ontology)
        if not Path(args.ontology).is_absolute()
        else Path(args.ontology)
    )
    kg_path = out_dir / "scene_kg.ttl"
    export_ttl(dataset, str(kg_path), ontology_path=ontology_path)
    print(f"Wrote KG  → {kg_path}")

    # ── render annotated frames ───────────────────────────────────────────────
    scene_by_index: Dict[int, Scene] = {s.index: s for s in trip.sequence.scenes}
    frame_stats: List[dict] = []

    for idx in frame_indices:
        scene = scene_by_index.get(idx)
        if scene is None:
            print(f"  WARNING: frame {idx} out of range (1–{total_frames})", file=sys.stderr)
            continue
        out_img = out_dir / "frames" / f"frame_{idx:03d}.png"
        stats = _render_frame(nusc, scene, out_img)
        if stats:
            frame_stats.append(stats)
            print(
                f"  Frame {idx:3d}: {stats['participants']:2d} agents | "
                f"motion={stats['motion_labelled']} "
                f"spatial={stats['spatial_labelled']} "
                f"ttc={stats['ttc_labelled']} "
                f"(H={stats['ttc_high']} M={stats['ttc_medium']} L={stats['ttc_low']}) "
                f"→ {out_img.name}"
            )

    # ── write high-level metadata ─────────────────────────────────────────────
    meta = {
        "scene": args.scene,
        "frames_studied": frame_indices,
        "total_scene_keyframes": total_frames,
        "per_frame": frame_stats,
    }
    meta_path = out_dir / "scene_study_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote meta    → {meta_path}")

    # ── write raw numerical details ───────────────────────────────────────────
    details = _compute_raw_details(trip, frame_indices)
    details_out = {
        "scene": args.scene,
        "frames_studied": frame_indices,
        "constants": {
            "delta_t_s":           _DELTA_T_S,
            "motion_threshold_m":  _MOTION_THRESHOLD_M,
            "min_closing_speed_m_per_s": _MIN_CLOSING_SPEED,
            "ttc_high_upper_s":    _TTC_HIGH_UPPER_S,
            "ttc_medium_upper_s":  _TTC_MEDIUM_UPPER_S,
            "spatial_max_range_m": _MAX_RANGE_M,
            "spatial_long_band_m": _LONG_BAND_M,
            "spatial_lat_band_m":  _LAT_BAND_M,
        },
        "per_frame": details,
    }
    details_path = out_dir / "scene_study_details.json"
    details_path.write_text(json.dumps(details_out, indent=2))
    print(f"Wrote details → {details_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
