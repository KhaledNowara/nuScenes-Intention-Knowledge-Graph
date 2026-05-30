#!/usr/bin/env python3
"""EXP-006: run all validation scripts in sequence.

Short-circuits if the expected output TTL does not exist — this script never
invokes the pipeline. Use main.py to generate the KG first.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description="EXP-006 validation runner")
    p.add_argument("--ttl", type=str, default="graphs/multi_scene.ttl")
    p.add_argument("--dataroot", type=str, default=None, help="Required for motion_forward_check and spot_check_render")
    p.add_argument("--version", type=str, default="v1.0-mini")
    p.add_argument("--scenes", type=str, default=None)
    p.add_argument("--max-keyframes", type=int, default=None)
    p.add_argument("--sample", type=int, default=20)
    args = p.parse_args()

    ttl_path = (EXP_ROOT / args.ttl) if not Path(args.ttl).is_absolute() else Path(args.ttl)
    if not ttl_path.exists():
        print(
            f"ERROR: {ttl_path} does not exist. Run `python main.py ...` first to generate the KG.",
            file=sys.stderr,
        )
        return 2

    print("== kg_stats ==")
    rc = subprocess.call([sys.executable, str(EXP_ROOT / "evaluation" / "kg_stats.py"), "--ttl", str(ttl_path)])
    if rc != 0:
        return rc

    print("== semantic_consistency ==")
    rc = subprocess.call(
        [sys.executable, str(EXP_ROOT / "evaluation" / "semantic_consistency.py"), "--ttl", str(ttl_path)]
    )
    if rc != 0:
        return rc

    if args.dataroot is None:
        print(
            "NOTE: --dataroot not provided; skipping motion_forward_check and spot_check_render.",
            file=sys.stderr,
        )
        return 0

    cmd_common = ["--dataroot", args.dataroot, "--version", args.version]
    if args.scenes:
        cmd_common += ["--scenes", args.scenes]
    if args.max_keyframes is not None:
        cmd_common += ["--max-keyframes", str(args.max_keyframes)]

    print("== motion_forward_check ==")
    rc = subprocess.call(
        [sys.executable, str(EXP_ROOT / "evaluation" / "motion_forward_check.py"), *cmd_common]
    )
    if rc != 0:
        return rc

    print("== spot_check_render ==")
    rc = subprocess.call(
        [sys.executable, str(EXP_ROOT / "evaluation" / "spot_check_render.py"), *cmd_common, "--sample", str(args.sample)]
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
