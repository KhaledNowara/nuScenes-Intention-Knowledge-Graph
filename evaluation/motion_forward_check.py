#!/usr/bin/env python3
"""EXP-006: forward-consistency validation for motionTowardEgo.

Re-runs the pipeline's data loading to obtain (ego_xy, ground_obj_xy) per
(scene, instance), and for each labelled motion at scene t, checks whether the
distance change from t to t+1 is consistent with the label.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from data_layer import fetch_scene_bundles  # noqa: E402
from domain import build_dataset, compute_motion_states  # noqa: E402
from domain.relations.geometry import planar_distance  # noqa: E402
from domain.relations.motion import DISTANCE_THRESHOLD  # noqa: E402


def _expected_forward_sign(label: str, d_curr: float, d_next: float) -> bool:
    delta = d_next - d_curr
    if label == "Approaching":
        return delta < -DISTANCE_THRESHOLD
    if label == "Receding":
        return delta > DISTANCE_THRESHOLD
    if label == "Stationary":
        return abs(delta) <= DISTANCE_THRESHOLD
    return False


def check(dataset) -> dict:
    per_label_total: Counter = Counter()
    per_label_pass: Counter = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)

    for trip in dataset.trips:
        track: dict[str, list[tuple[int, tuple, tuple, str | None]]] = defaultdict(list)
        for scene in trip.sequence.scenes:
            for sp in scene.participants:
                track[sp.participant.instance_token].append(
                    (scene.index, sp.ground_obj_xy, scene.ego_xy, sp.motion)
                )
        for appearances in track.values():
            appearances.sort(key=lambda a: a[0])
            for i in range(len(appearances) - 1):
                _, obj_curr, ego_curr, label = appearances[i]
                _, obj_next, ego_next, _ = appearances[i + 1]
                if label is None:
                    continue
                d_curr = planar_distance(obj_curr, ego_curr)
                d_next = planar_distance(obj_next, ego_next)
                per_label_total[label] += 1
                if _expected_forward_sign(label, d_curr, d_next):
                    per_label_pass[label] += 1

                delta = d_next - d_curr
                if delta < -DISTANCE_THRESHOLD:
                    actual = "Approaching"
                elif delta > DISTANCE_THRESHOLD:
                    actual = "Receding"
                else:
                    actual = "Stationary"
                confusion[label][actual] += 1

    return {
        "per_label_total": dict(per_label_total),
        "per_label_pass": dict(per_label_pass),
        "per_label_accuracy": {
            k: (per_label_pass[k] / per_label_total[k]) if per_label_total[k] else None
            for k in per_label_total
        },
        "overall_accuracy": (
            sum(per_label_pass.values()) / sum(per_label_total.values())
            if sum(per_label_total.values())
            else None
        ),
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def main() -> int:
    p = argparse.ArgumentParser(description="EXP-006 motion forward-consistency check")
    p.add_argument("--dataroot", type=str, required=True)
    p.add_argument("--version", type=str, default="v1.0-mini")
    p.add_argument("--scenes", type=str, default=None)
    p.add_argument("--max-keyframes", type=int, default=None)
    p.add_argument("--out-json", type=str, default="evaluation/results/motion_forward_check.json")
    args = p.parse_args()

    scene_names = [s.strip() for s in args.scenes.split(",")] if args.scenes else None
    _, bundles = fetch_scene_bundles(
        args.dataroot,
        version=args.version,
        scene_names=scene_names,
        verbose=False,
        max_keyframes_per_scene=args.max_keyframes,
    )
    dataset = build_dataset(bundles, dataset_name="nuScenes", dataset_version=args.version)
    compute_motion_states(dataset)
    result = check(dataset)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
