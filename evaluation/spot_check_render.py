#!/usr/bin/env python3
"""EXP-006: render sampled CAM_FRONT keyframes with spatial/ttc overlays for manual review."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import cv2

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


def _annotate(img, sp):
    if sp.bbox_pixels is None:
        return
    x1, y1, x2, y2 = sp.bbox_pixels
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{sp.spatial_config or '-'} | TTC={sp.ttc_risk or '-'}"
    cv2.putText(img, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


def main() -> int:
    p = argparse.ArgumentParser(description="EXP-006 spatial/ttc spot-check renderer")
    p.add_argument("--dataroot", type=str, required=True)
    p.add_argument("--version", type=str, default="v1.0-mini")
    p.add_argument("--scenes", type=str, default=None)
    p.add_argument("--max-keyframes", type=int, default=None)
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="evaluation/results/spot_checks")
    p.add_argument("--out-csv", type=str, default="evaluation/results/spot_check_sheet.csv")
    args = p.parse_args()

    scene_names = [s.strip() for s in args.scenes.split(",")] if args.scenes else None
    nusc, bundles = fetch_scene_bundles(
        args.dataroot,
        version=args.version,
        scene_names=scene_names,
        verbose=False,
        max_keyframes_per_scene=args.max_keyframes,
    )
    dataset = build_dataset(bundles, dataset_name="nuScenes", dataset_version=args.version)
    compute_motion_states(dataset)
    compute_spatial_configuration(dataset)
    compute_ttc_risk(dataset)

    scene_to_cam: dict[str, str] = {}
    for bundle in bundles:
        for idx, kf in enumerate(bundle.keyframes):
            scene_to_cam[f"Scene_{bundle.sequence_key}_{idx + 1:03d}"] = kf.cam_front_sample_data_token

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    scenes_with_labels = [
        scene
        for trip in dataset.trips
        for scene in trip.sequence.scenes
        if any(sp.spatial_config or sp.ttc_risk for sp in scene.participants)
    ]
    sample = rng.sample(scenes_with_labels, min(args.sample, len(scenes_with_labels)))

    rows: list[list[str]] = [["scene_id", "sp_id", "spatial_config", "ttc_risk", "spatial_agree", "ttc_agree"]]

    for scene in sample:
        cam_tok = scene_to_cam.get(scene.scene_id)
        if cam_tok is None:
            continue
        sd = nusc.get("sample_data", cam_tok)
        img_path = os.path.join(nusc.dataroot, sd["filename"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        for sp in scene.participants:
            _annotate(img, sp)
            rows.append([scene.scene_id, sp.sp_id, sp.spatial_config or "", sp.ttc_risk or "", "", ""])
        cv2.imwrite(str(out_dir / f"{scene.scene_id}.png"), img)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"Wrote {len(sample)} frame(s) to {out_dir} and sheet {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
