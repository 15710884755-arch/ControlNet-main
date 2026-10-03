#!/usr/bin/env python
"""Verify and optionally remove the split-copy of the M-LSD checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


CHECKPOINT_NAME = "control_sd15_mlsd.pth"
PARTS_NAME = "control_sd15_mlsd.pth.parts"
CHUNK_SIZE = 8 * 1024 * 1024


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_parts(paths):
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete-parts",
        action="store_true",
        help="delete the parts directory after a verified hash match",
    )
    args = parser.parse_args()

    model_dir = config.MODEL_DIR.resolve()
    checkpoint = model_dir / CHECKPOINT_NAME
    parts_dir = model_dir / PARTS_NAME

    if not parts_dir.is_dir():
        print(f"No parts directory found: {parts_dir}")
        return 0
    if not checkpoint.is_file():
        print(f"Checkpoint is missing: {checkpoint}", file=sys.stderr)
        return 1

    parts = sorted(path for path in parts_dir.iterdir() if path.is_file())
    if not parts:
        print(f"Parts directory is empty: {parts_dir}")
        return 1

    checkpoint_size = checkpoint.stat().st_size
    parts_size = sum(path.stat().st_size for path in parts)
    print(f"Checkpoint: {checkpoint_size} bytes")
    print(f"Parts     : {parts_size} bytes in {len(parts)} files")
    if checkpoint_size != parts_size:
        print("Size mismatch; keeping the parts directory.")
        return 1

    print("Hashing checkpoint ...")
    checkpoint_hash = hash_file(checkpoint)
    print("Hashing parts ...")
    parts_hash = hash_parts(parts)
    print(f"Checkpoint SHA-256: {checkpoint_hash}")
    print(f"Parts SHA-256     : {parts_hash}")

    if checkpoint_hash != parts_hash:
        print("Hash mismatch; keeping the parts directory.")
        return 1

    print("Verified duplicate.")
    if not args.delete_parts:
        print("Re-run with --delete-parts to remove the duplicate.")
        return 0

    resolved_parts = parts_dir.resolve()
    if resolved_parts.parent != model_dir or resolved_parts.name != PARTS_NAME:
        raise RuntimeError(f"Refusing to delete unexpected path: {resolved_parts}")
    shutil.rmtree(resolved_parts)
    print(f"Deleted {resolved_parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
