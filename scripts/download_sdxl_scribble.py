"""Download the SDXL base model, VAE and ControlNet weights used by the SDXL scribble UI.

Hugging Face is tried first and ModelScope is the mirror, because in practice one
of the two is unreachable or throttled depending on the network. Transfers go
through ``parallel_download`` (several ranged GETs at once) instead of
huggingface_hub, whose transfer backends stall on high latency links. Files that
are already complete on disk are skipped, so the script can be re-run after an
interrupted download without starting over.

Usage:
    python scripts/download_sdxl_scribble.py
    python scripts/download_sdxl_scribble.py --source modelscope
    python scripts/download_sdxl_scribble.py --groups base scribble
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parallel_download import download, is_complete  # noqa: E402 - helper next to this script


ROOT = SCRIPT_DIR.parent
SDXL_DIR = ROOT / "models" / "sdxl"

HF_URL = "https://huggingface.co/{repo}/resolve/main/{path}"
MODELSCOPE_URL = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"
CONNECTIONS = int(os.environ.get("SDXL_DOWNLOAD_CONNECTIONS", "8"))

BASE_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)

GROUPS = {
    "vae": {
        "dir": SDXL_DIR / "vae-fp16-fix",
        "files": ("config.json", "diffusion_pytorch_model.safetensors"),
        "hf": "madebyollin/sdxl-vae-fp16-fix",
        "modelscope": "AI-ModelScope/sdxl-vae-fp16-fix",
    },
    "base": {
        "dir": SDXL_DIR / "stable-diffusion-xl-base-1.0",
        "files": BASE_FILES,
        "hf": "stabilityai/stable-diffusion-xl-base-1.0",
        "modelscope": "AI-ModelScope/stable-diffusion-xl-base-1.0",
    },
    "scribble": {
        "dir": SDXL_DIR / "controlnet-scribble-sdxl-1.0",
        "files": ("config.json", "diffusion_pytorch_model.safetensors"),
        "hf": "xinsir/controlnet-scribble-sdxl-1.0",
        "modelscope": None,
    },
    "canny": {
        "dir": SDXL_DIR / "controlnet-canny-sdxl-1.0",
        "files": ("config.json", "diffusion_pytorch_model.fp16.safetensors"),
        "hf": None,
        "modelscope": "AI-ModelScope/controlnet-canny-sdxl-1.0",
    },
}

DEFAULT_GROUPS = ("vae", "base", "scribble")


def missing_files(group):
    # A file left behind by an interrupted transfer keeps its journal sidecar, so
    # existence alone is not enough to call a group complete.
    return [name for name in group["files"] if not is_complete(group["dir"] / name)]


def url_for(source, group, filename):
    template = HF_URL if source == "hf" else MODELSCOPE_URL
    return template.format(repo=group[source], path=filename)


def source_order(requested, group):
    order = [requested] if requested != "auto" else ["hf", "modelscope"]
    return [name for name in order if group.get(name)]


def report(message):
    print("    " + message, flush=True)


def resolve(name, requested):
    group = GROUPS[name]
    group["dir"].mkdir(parents=True, exist_ok=True)
    if not missing_files(group):
        print("[" + name + "] already complete -> " + str(group["dir"]))
        return True

    for source in source_order(requested, group):
        for filename in missing_files(group):
            print("[" + name + "] " + source + " <- " + filename)
            try:
                download(
                    url_for(source, group, filename),
                    group["dir"] / filename,
                    connections=CONNECTIONS,
                    log=report,
                )
            except Exception as error:
                print(
                    "[" + name + "] " + source + " failed on " + filename + ": "
                    + type(error).__name__ + ": " + str(error),
                    file=sys.stderr,
                )
                break
        if not missing_files(group):
            print("[" + name + "] done -> " + str(group["dir"]))
            return True

    print("[" + name + "] INCOMPLETE, still missing: " + str(missing_files(group)), file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("auto", "hf", "modelscope"), default="auto")
    parser.add_argument("--groups", nargs="+", choices=tuple(GROUPS), default=list(DEFAULT_GROUPS))
    args = parser.parse_args()

    SDXL_DIR.mkdir(parents=True, exist_ok=True)
    results = {name: resolve(name, args.source) for name in args.groups}

    if results.get("scribble") is False and "canny" not in results:
        print("Scribble ControlNet is unavailable, fetching the Canny fallback instead.")
        results["canny"] = resolve("canny", args.source)

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print("Failed groups: " + str(failed), file=sys.stderr)
        return 1

    controlnet_groups = [name for name in ("scribble", "canny") if name in results]
    if controlnet_groups and not any(results[name] for name in controlnet_groups):
        print("No SDXL ControlNet weights are available.", file=sys.stderr)
        return 1

    print("SDXL weights are ready. Start the UI with: python gradio_scribble2image_sdxl.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
