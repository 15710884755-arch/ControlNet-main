#!/usr/bin/env python
"""List or download model assets declared in configs/model_manifest.json."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


MANIFEST = ROOT / "configs" / "model_manifest.json"


def load_manifest():
    with MANIFEST.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def target_path(group, entry):
    if group == "detectors":
        return config.ANNOTATOR_CKPT_DIR / entry["name"]
    return config.MODEL_DIR / entry["name"]


def print_status(group, entries):
    print(f"\n[{group}]")
    for entry in entries:
        path = target_path(group, entry)
        state = "OK" if path.is_file() else "MISSING"
        size_mb = round(path.stat().st_size / 1024**2, 2) if path.is_file() else 0
        url = entry.get("url") or "manual"
        print(f"  {state:7} {path} ({size_mb} MB)")
        print(f"          {url}")
        if entry.get("note"):
            print(f"          note: {entry['note']}")


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0

    headers = {}
    mode = "wb"
    if offset:
        headers["Range"] = f"bytes={offset}-"
        mode = "ab"

    request = urllib.request.Request(url, headers=headers)
    print(f"Downloading {url}")
    print(f"  -> {destination}")
    with urllib.request.urlopen(request, timeout=60) as response:
        if offset and response.status != 206:
            print("  server does not support resume; restarting")
            mode = "wb"
            offset = 0
        total = response.headers.get("Content-Length")
        total_bytes = int(total) + offset if total else None
        with part.open(mode) as handle:
            downloaded = offset
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    percent = downloaded * 100 / total_bytes
                    print(f"\r  {percent:5.1f}% ({downloaded / 1024**3:.2f} GB)", end="")
                else:
                    print(f"\r  {downloaded / 1024**3:.2f} GB", end="")
    print()
    part.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=["inference", "training", "detectors", "all"],
        default="all",
    )
    parser.add_argument("--apply", action="store_true", help="actually download missing files")
    args = parser.parse_args()

    manifest = load_manifest()
    groups = list(manifest) if args.group == "all" else [args.group]

    for group in groups:
        entries = manifest[group]
        print_status(group, entries)

    if not args.apply:
        print("\nDry run only. Re-run with --apply to download entries that have URLs.")
        return 0

    for group in groups:
        for entry in manifest[group]:
            destination = target_path(group, entry)
            if destination.is_file():
                continue
            url = entry.get("url")
            if not url:
                print(f"Skipping manual asset: {destination}")
                continue
            try:
                download(url, destination)
            except urllib.error.URLError as exc:
                print(f"Download failed for {url}: {exc}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
