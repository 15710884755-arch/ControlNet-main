#!/usr/bin/env python
"""Print runtime environment details for debugging ControlNet."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402


MODULES = [
    "torch",
    "torchvision",
    "gradio",
    "transformers",
    "open_clip",
    "omegaconf",
    "safetensors",
    "einops",
    "cv2",
]


def probe_module(name):
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"MISSING ({exc})"
    version = getattr(module, "__version__", "unknown")
    return f"{version} ({getattr(module, '__file__', 'unknown')})"


def main():
    print("=== Python ===")
    print(f"python: {sys.version}")
    print(f"executable: {sys.executable}")
    print(f"platform: {platform.platform()}")
    print(f"repo: {ROOT}")
    print()
    print("=== Config ===")
    print(f"MODEL_DIR: {config.MODEL_DIR}")
    print(f"ANNOTATOR_CKPT_DIR: {config.ANNOTATOR_CKPT_DIR}")
    print(f"TRAINING_DIR: {config.TRAINING_DIR}")
    print(f"save_memory: {config.save_memory}")
    print()
    print("=== Imports ===")
    for name in MODULES:
        print(f"{name}: {probe_module(name)}")
    print()
    print("=== PyTorch Lightning ===")
    try:
        import lightning_compat
        print(f"module: {getattr(lightning_compat.pl, '__file__', 'unknown')}")
        print(f"using shim: {lightning_compat.USING_SHIM}")
        print(f"has Trainer: {hasattr(lightning_compat.pl, 'Trainer')}")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"MISSING ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
