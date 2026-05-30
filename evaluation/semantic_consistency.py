#!/usr/bin/env python3
"""EXP-006: semantic consistency checks on the generated KG.

Checks three logical implications derived from the relation definitions:
  1. ttcRisk=High  → motionTowardEgo=Approaching
  2. ttcRisk=High  → spatialConfiguration in forward zones
  3. motionTowardEgo=Stationary → ttcRisk ≠ High

Outputs a JSON summary and a Markdown table for the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rdflib import Graph, Namespace

EXP_ROOT = Path(__file__).resolve().parent.parent
NUS = Namespace("http://www.nuscenes.org/nuScenes/")

FORWARD_ZONES = {
    str(NUS.Preceding),
    str(NUS.LeftPreceding),
    str(NUS.RightPreceding),
}


def run_checks(g: Graph) -> dict:
    results = {}

    # ------------------------------------------------------------------
    # Check 1: ttcRisk=High → motionTowardEgo=Approaching
    # ------------------------------------------------------------------
    high_risk = set(
        s for s, _, o in g.triples((None, NUS.ttcRisk, NUS.High))
    )
    total_high = len(high_risk)
    approaching = set(
        s for s, _, o in g.triples((None, NUS.motionTowardEgo, NUS.Approaching))
    )
    pass_c1 = len(high_risk & approaching)
    results["ttcHigh_implies_approaching"] = {
        "description": "ttcRisk=High → motionTowardEgo=Approaching",
        "total": total_high,
        "pass": pass_c1,
        "pass_rate": round(pass_c1 / total_high, 4) if total_high else None,
    }

    # ------------------------------------------------------------------
    # Check 2: ttcRisk=High → spatialConfiguration in forward zones
    # ------------------------------------------------------------------
    pass_c2 = sum(
        1 for sp in high_risk
        if any(
            str(o) in FORWARD_ZONES
            for _, _, o in g.triples((sp, NUS.spatialConfiguration, None))
        )
    )
    # agents with High TTC that also have a spatialConfiguration label
    high_with_spatial = sum(
        1 for sp in high_risk
        if any(True for _ in g.triples((sp, NUS.spatialConfiguration, None)))
    )
    results["ttcHigh_implies_forward_zone"] = {
        "description": "ttcRisk=High → spatialConfiguration in forward zone",
        "total": high_with_spatial,
        "pass": pass_c2,
        "pass_rate": round(pass_c2 / high_with_spatial, 4) if high_with_spatial else None,
        "note": f"{total_high - high_with_spatial} High-risk agents had no spatialConfiguration label (excluded from denominator)",
    }

    # ------------------------------------------------------------------
    # Check 3: motionTowardEgo=Stationary → ttcRisk ≠ High
    # ------------------------------------------------------------------
    stationary = set(
        s for s, _, o in g.triples((None, NUS.motionTowardEgo, NUS.Stationary))
    )
    total_stationary = len(stationary)
    stationary_and_high = len(stationary & high_risk)
    pass_c3 = total_stationary - stationary_and_high
    results["stationary_implies_not_high"] = {
        "description": "motionTowardEgo=Stationary → ttcRisk ≠ High",
        "total": total_stationary,
        "pass": pass_c3,
        "violations": stationary_and_high,
        "pass_rate": round(pass_c3 / total_stationary, 4) if total_stationary else None,
    }

    return results


def format_markdown(results: dict) -> str:
    lines = [
        "# Semantic Consistency Checks\n",
        "| Constraint | Total | Pass | Pass Rate |",
        "|---|---|---|---|",
    ]
    for check in results.values():
        rate = f"{check['pass_rate'] * 100:.1f}%" if check["pass_rate"] is not None else "N/A"
        lines.append(
            f"| {check['description']} | {check['total']} | {check['pass']} | {rate} |"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="EXP-006 semantic consistency checks")
    p.add_argument("--ttl", type=str, default="graphs/multi_scene.ttl")
    p.add_argument("--out-json", type=str, default="evaluation/results/semantic_consistency.json")
    p.add_argument("--out-md", type=str, default="evaluation/results/semantic_consistency.md")
    args = p.parse_args()

    ttl_path = Path(args.ttl)
    if not ttl_path.exists():
        print(f"ERROR: {ttl_path} does not exist. Run main.py first.", file=sys.stderr)
        return 2

    print(f"Loading {ttl_path} ...")
    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    print(f"Loaded {len(g)} triples.")

    results = run_checks(g)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, indent=2))
    Path(args.out_md).write_text(format_markdown(results))

    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
