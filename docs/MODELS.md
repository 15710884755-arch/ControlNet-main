# Model Assets

Model weights are external runtime assets. They are intentionally excluded from
Git and should be stored outside the source tree when possible.

## Default Locations

- ControlNet and CLIP weights: `models/`
- Preprocessor detector weights: `src/annotator/ckpts/`

The repository supports environment overrides:

```powershell
$env:CONTROLNET_MODEL_DIR = 'D:\ControlNet-models'
$env:CONTROLNET_ANNOTATOR_CKPT_DIR = 'D:\ControlNet-annotator-ckpts'
$env:CONTROLNET_TRAINING_DIR = 'D:\ControlNet-training'
```

`config.py` resolves relative override paths against the repository root.

## Required Core Weights

```text
models/
  control_sd15_canny.pth
  control_sd15_depth.pth
  control_sd15_hed.pth
  control_sd15_mlsd.pth
  control_sd15_normal.pth
  control_sd15_openpose.pth
  control_sd15_scribble.pth
  control_sd15_seg.pth
  clip-vit-large-patch14/
```

Training and weight-transfer scripts also expect:

```text
models/control_sd15_ini.ckpt
models/control_sd21_ini.ckpt
models/v1-5-pruned.ckpt
models/v2-1_512-ema-pruned.ckpt
models/anything-v3-full.safetensors
```

The two `control_sd*_ini.ckpt` files are generated assets, not files hosted in
the official ControlNet model repository. Generate them from the corresponding
base checkpoints:

```powershell
python tool_add_control.py models\v1-5-pruned.ckpt models\control_sd15_ini.ckpt
python tool_add_control_sd21.py models\v2-1_512-ema-pruned.ckpt models\control_sd21_ini.ckpt
```

`gradio_seg2image.py` downloads `upernet_global_small.pth` on first use if it is
missing from `src/annotator/ckpts/`.

## SDXL Scribble Mode

The optional SDXL entrypoint uses a separate diffusers pipeline:

```text
models/sdxl/
  stable-diffusion-xl-base-1.0/
  vae-fp16-fix/
  controlnet-scribble-sdxl-1.0/   (preferred)
  controlnet-canny-sdxl-1.0/      (fallback)
```

Prepare the weights with:

```powershell
python scripts\download_sdxl_scribble.py
```

Every weight goes through `scripts/parallel_download.py`, which issues several
ranged GETs at once because `huggingface_hub`'s transfer backends stall on high
latency links while the CDN keeps serving plain range requests. Hugging Face is
tried first and ModelScope is used as the mirror; files that are already complete
are skipped, so re-running the script resumes an interrupted download instead of
starting over. When the Scribble ControlNet cannot be fetched, the script falls
back to the SDXL Canny ControlNet. Set `SDXL_DOWNLOAD_CONNECTIONS` to change the
number of parallel connections (default 8).

Run the SDXL UI with sequential CPU offload on 8 GB GPUs:

```powershell
$env:SDXL_OFFLOAD = 'sequential'
$env:GRADIO_SERVER_PORT = '7867'
python gradio_scribble2image_sdxl.py
```

`SDXL_OFFLOAD=model` is faster but needs more VRAM. SDXL is substantially
slower than the SD 1.5 ControlNet entrypoint on an 8 GB laptop GPU. Measured on
an RTX 5060 Laptop (8 GB) with sequential offload: about 95 s for 1024x1024 at
20 steps, and about 65 s for 512x512 at 12 steps.

The resolution slider defaults to 1024 because SDXL is trained at that size;
at 512 or 768 the model tends to tile the composition into repeated panels.

## Download Manifest

Known download URLs are recorded in `configs/model_manifest.json`. Use the
downloader in dry-run mode first:

```powershell
python scripts\download_models.py
python scripts\download_models.py --group training
python scripts\download_models.py --group training --apply
```

All training entries are intentionally marked `manual` because the base
checkpoint source, access terms, and license depend on the user's chosen
distribution. Only inference and detector assets are downloaded automatically.

## Externalize The Weight Directory

Run a dry run first:

```powershell
.\scripts\externalize_models.ps1
```

After reviewing the output, apply the move and create a junction so the
existing `models/...` paths keep working:

```powershell
.\scripts\externalize_models.ps1 -Destination D:\ControlNet-models -Apply
```

The destination must not already exist. The script moves `models/` to the
destination and creates a directory junction at the original path.

## Duplicate Check

`models/control_sd15_mlsd.pth.parts/` was verified against
`control_sd15_mlsd.pth` by SHA-256 and was an exact duplicate. The parts
directory was removed to reclaim about 5.32 GB.

Use the verification script before removing similar split copies in the
future:

```powershell
python scripts\verify_mlsd_parts.py
python scripts\verify_mlsd_parts.py --delete-parts
```
