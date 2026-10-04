#!/usr/bin/env python
"""Print runtime environment details for debugging ControlNet."""

from __future__ import annotations

import argparse
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


def declared_versions():
    try:
        from omegaconf import OmegaConf

        environment = OmegaConf.load(ROOT / "environment.yaml")
    except Exception:
        return {}

    versions = {}
    for dependency in environment.get("dependencies", []):
        if isinstance(dependency, str):
            name, separator, version = dependency.partition("=")
            if separator:
                versions[name.strip()] = version.strip()
            continue
        try:
            pip_dependencies = dependency.get("pip", [])
        except Exception:
            continue
        for item in pip_dependencies or []:
            name, separator, version = str(item).partition("==")
            if separator:
                versions[name.strip()] = version.strip()
    return versions


def module_version(name):
    try:
        return getattr(importlib.import_module(name), "__version__", "unknown")
    except Exception:
        return "missing"


def normalize_version(value):
    return str(value).split("+")[0].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero exit code when declared and installed versions differ",
    )
    args = parser.parse_args()

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
        lightning_version = (
            "local shim"
            if lightning_compat.USING_SHIM
            else module_version("pytorch_lightning")
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"MISSING ({exc})")
        lightning_version = "missing"

    print()
    print("=== Declared Environment ===")
    declared = declared_versions()
    checks = [
        ("python", declared.get("python"), platform.python_version()),
        ("torch", declared.get("pytorch"), module_version("torch")),
        ("torchvision", declared.get("torchvision"), module_version("torchvision")),
        ("gradio", declared.get("gradio"), module_version("gradio")),
        ("pytorch-lightning", declared.get("pytorch-lightning"), lightning_version),
    ]
    drift = []
    for name, expected, installed in checks:
        if expected is None:
            continue
        status = "OK" if normalize_version(expected) == normalize_version(installed) else "DRIFT"
        print(f"{status:5} {name}: declared {expected}, installed {installed}")
        if status == "DRIFT":
            drift.append(name)
    if drift and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
