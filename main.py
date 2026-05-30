#!/usr/bin/env python3
"""EXP-006: nuScenes → domain model → motion / spatial / TTC → RDF / optional annotated frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_layer.nuscenes_fetch import fetch_scene_bundles  # noqa: E402
from domain import (  # noqa: E402
    build_dataset,
    compute_motion_states,
    compute_spatial_configuration,
    compute_ttc_risk,
    motion_lookup_by_scene_key,
)
from graph_exporter import export_dataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="EXP-006 intention-relation KG pipeline")
    p.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="nuScenes root (contains samples/, sweeps/, v1.0-mini/ etc.)",
    )
    p.add_argument("--version", type=str, default="v1.0-mini", help="DB version folder name")
    p.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Comma-separated nuScenes scene names (default: all scenes in DB)",
    )
    p.add_argument("--save-artifacts", action="store_true", help="Write traces/annotated_frames PNGs")
    p.add_argument(
        "--max-keyframes",
        type=int,
        default=None,
        metavar="N",
        help="Max CAM_FRONT keyframes per nuScenes scene (default: all keyframes in scene)",
    )
    p.add_argument(
        "--out-ttl",
        type=str,
        default="graphs/multi_scene.ttl",
        help="Output Turtle path (relative to CWD or absolute)",
    )
    p.add_argument(
        "--out-sparql",
        type=str,
        default="graphs/multi_scene.sparql",
        help="Output SPARQL INSERT path",
    )
    p.add_argument(
        "--ontology",
        type=str,
        default="ontologies/ontology.ttl",
        help="Path to the T-Box TTL merged with the output graph",
    )
    p.add_argument(
        "--tbox-out",
        type=str,
        default=None,
        metavar="PATH",
        help="If given, re-serialize the T-Box (ontology only, no instance triples) to this path",
    )
    args = p.parse_args()

    scene_names = None
    if args.scenes:
        scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]

    nusc, bundles = fetch_scene_bundles(
        args.dataroot,
        version=args.version,
        scene_names=scene_names,
        verbose=False,
        max_keyframes_per_scene=args.max_keyframes,
    )
    print(f"Loaded {len(bundles)} scene bundle(s), {sum(len(b.keyframes) for b in bundles)} keyframes")

    dataset = build_dataset(bundles, dataset_name="nuScenes", dataset_version=args.version)
    compute_motion_states(dataset)
    compute_spatial_configuration(dataset)
    compute_ttc_risk(dataset)
    export_dataset(dataset, args.out_ttl, args.out_sparql, ontology_path=args.ontology)
    print(f"Wrote {args.out_ttl} and {args.out_sparql}")

    if args.tbox_out:
        from rdflib import Graph

        tbox_path = Path(args.tbox_out)
        tbox_path.parent.mkdir(parents=True, exist_ok=True)
        g = Graph()
        g.parse(args.ontology, format="turtle")
        g.serialize(destination=str(tbox_path), format="turtle")
        print(f"Wrote T-Box → {args.tbox_out}")

    if args.save_artifacts:
        from data_layer.frame_export import export_bundle_frames  # noqa: E402

        motion_map = motion_lookup_by_scene_key(dataset)
        export_bundle_frames(nusc, bundles, motion_per_bundle_scene=motion_map, out_root="traces/annotated_frames")
        print("Wrote annotated frames under traces/annotated_frames/")


if __name__ == "__main__":
    main()
