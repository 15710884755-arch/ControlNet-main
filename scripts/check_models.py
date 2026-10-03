#!/usr/bin/env python
"""Check whether expected runtime model assets are present."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


CORE_WEIGHTS = [
    "control_sd15_canny.pth",
    "control_sd15_depth.pth",
    "control_sd15_hed.pth",
    "control_sd15_mlsd.pth",
    "control_sd15_normal.pth",
    "control_sd15_openpose.pth",
    "control_sd15_scribble.pth",
    "control_sd15_seg.pth",
    "clip-vit-large-patch14/pytorch_model.bin",
]

TRAINING_WEIGHTS = [
    "control_sd15_ini.ckpt",
    "control_sd21_ini.ckpt",
    "v1-5-pruned.ckpt",
    "anything-v3-full.safetensors",
]

DETECTOR_WEIGHTS = [
    "body_pose_model.pth",
    "hand_pose_model.pth",
    "dpt_hybrid-midas-501f0c75.pt",
    "mlsd_large_512_fp32.pth",
]

OPTIONAL_DETECTOR_WEIGHTS = [
    "upernet_global_small.pth",
]


def inspect_file(path):
    if not path.is_file():
        return {"name": str(path), "exists": False, "size_mb": 0.0}
    return {
        "name": str(path),
        "exists": True,
        "size_mb": round(path.stat().st_size / 1024**2, 2),
    }


def inspect_group(label, root, names):
    rows = [inspect_file(root / name) for name in names]
    missing = [row for row in rows if not row["exists"]]
    print(f"\n[{label}] {root}")
    for row in rows:
        state = "OK" if row["exists"] else "MISSING"
        print(f"  {state:7} {row['name']} ({row['size_mb']} MB)")
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    core_missing = inspect_group("core", config.MODEL_DIR, CORE_WEIGHTS)
    training_missing = inspect_group("training", config.MODEL_DIR, TRAINING_WEIGHTS)
    detector_missing = inspect_group("detectors", config.ANNOTATOR_CKPT_DIR, DETECTOR_WEIGHTS)
    optional_missing = inspect_group(
        "optional-detectors",
        config.ANNOTATOR_CKPT_DIR,
        OPTIONAL_DETECTOR_WEIGHTS,
    )

    parts_dir = config.MODEL_DIR / "control_sd15_mlsd.pth.parts"
    if parts_dir.is_dir():
        parts = list(parts_dir.glob("*"))
        print(f"\n[note] {parts_dir} exists with {len(parts)} files; verify before deleting.")

    result = {
        "model_dir": str(config.MODEL_DIR),
        "annotator_ckpt_dir": str(config.ANNOTATOR_CKPT_DIR),
        "core_missing": [row["name"] for row in core_missing],
        "training_missing": [row["name"] for row in training_missing],
        "detector_missing": [row["name"] for row in detector_missing],
        "optional_detector_missing": [row["name"] for row in optional_missing],
        "mlsd_parts_dir": str(parts_dir) if parts_dir.is_dir() else None,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    if core_missing:
        print("\nERROR: core inference weights are missing.", file=sys.stderr)
        return 1
    print("\nCore inference weights are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
