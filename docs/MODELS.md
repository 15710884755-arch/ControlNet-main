# Model Assets

Model weights are external runtime assets. They are intentionally excluded from
Git and should be stored outside the source tree when possible.

## Default Locations

- ControlNet and CLIP weights: `models/`
- Preprocessor detector weights: `annotator/ckpts/`

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
models/anything-v3-full.safetensors
```

`gradio_seg2image.py` downloads `upernet_global_small.pth` on first use if it is
missing from `annotator/ckpts/`.

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

`models/control_sd15_mlsd.pth.parts/` contains a second copy-shaped set of
files. Do not delete it until the primary `control_sd15_mlsd.pth` checkpoint has
been loaded successfully and its output has been verified.
