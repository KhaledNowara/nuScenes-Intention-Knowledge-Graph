#!/usr/bin/env python3
"""EXP-006: descriptive KG statistics for the paper's Results section."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from rdflib import Graph, Namespace
from rdflib.namespace import RDF

EXP_ROOT = Path(__file__).resolve().parent.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

NUS = Namespace("http://www.nuscenes.org/nuScenes/")

RELATIONS_TO_SUMMARISE = {
    "motionTowardEgo": NUS.motionTowardEgo,
    "spatialConfiguration": NUS.spatialConfiguration,
    "ttcRisk": NUS.ttcRisk,
    "hasCoarseCategory": NUS.hasCoarseCategory,
}


def compute_stats(graph: Graph) -> dict:
    classes = Counter()
    for s, _, o in graph.triples((None, RDF.type, None)):
        classes[str(o)] += 1

    property_counts = Counter()
    for _, p, _ in graph:
        property_counts[str(p)] += 1

    relation_values = {}
    for name, prop in RELATIONS_TO_SUMMARISE.items():
        values = Counter(str(o).rsplit("/", 1)[-1] for _, _, o in graph.triples((None, prop, None)))
        relation_values[name] = dict(values)

    return {
        "total_triples": len(graph),
        "classes": dict(classes),
        "properties": dict(property_counts),
        "relation_values": relation_values,
    }


def format_markdown(stats: dict) -> str:
    lines = [f"# KG Statistics\n\n**Total triples:** {stats['total_triples']}\n"]
    lines.append("## Relation value distribution\n")
    for name, values in stats["relation_values"].items():
        lines.append(f"### {name}\n")
        for value, count in sorted(values.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {value}: {count}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="EXP-006 KG descriptive stats")
    p.add_argument("--ttl", type=str, default="graphs/multi_scene.ttl")
    p.add_argument("--out-json", type=str, default="evaluation/results/kg_stats.json")
    p.add_argument("--out-md", type=str, default="evaluation/results/kg_stats.md")
    args = p.parse_args()

    ttl_path = Path(args.ttl)
    if not ttl_path.exists():
        print(f"ERROR: {ttl_path} does not exist. Run main.py first.", file=sys.stderr)
        return 2

    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    stats = compute_stats(g)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(stats, indent=2))
    Path(args.out_md).write_text(format_markdown(stats))
    print(f"Wrote {args.out_json} and {args.out_md} ({stats['total_triples']} triples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
