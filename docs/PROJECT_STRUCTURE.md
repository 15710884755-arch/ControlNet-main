# Project Structure

This repository is the ControlNet 1.0 reference implementation plus local
runtime patches. It is a research/demo codebase, not a packaged service.

## Current Layout

```text
ControlNet-main/
|-- configs/                  YAML model configs
|-- src/
|   |-- cldm/                 ControlNet model and LDM integration
|   |-- ldm/                  Latent Diffusion building blocks
|   |-- annotator/            condition preprocessors and detector code
|   `-- pytorch_lightning_shim/ local inference-only compatibility layer
|-- assets/                   README images, test images, fonts, samples
|-- scripts/                  maintenance and diagnostic scripts
|-- tests/                    standard-library test suite
|-- docs/                     engineering and usage docs
|-- config.py                 path and runtime configuration
|-- lightning_compat.py       official Lightning import or local shim fallback
|-- sitecustomize.py          adds src/ to sys.path for direct scripts
|-- gradio_*.py               inference entrypoints
|-- tutorial_*.py             training examples
|-- tool_*.py                 weight injection and transfer tools
|-- models/                   external model assets (gitignored)
|-- environment.yaml
|-- requirements.txt
|-- requirements-dev.txt
|-- pyproject.toml
`-- README.md
```

## Runtime Flow

```text
gradio_*.py
  -> share.py / config.py
  -> cldm.model.create_model()
  -> configs/cldm_v15.yaml
  -> cldm.cldm.ControlLDM
  -> annotator.* produces a hint map
  -> cldm.cldm.ControlNet
  -> ldm UNet / VAE / CLIP
  -> cldm.ddim_hacked.DDIMSampler
  -> Gradio output
```

## Package Migration Status

The base `src/` layout is in place. Package names remain `cldm`, `ldm`, and
`annotator` so existing imports and YAML targets keep working.

The next optional step is a semantic rename:

```text
src/
  controlnet/       renamed cldm core
  diffusion/        renamed ldm building blocks
  preprocessors/    renamed annotator wrappers
  runtime/          model loading and device policy
  apps/             Gradio and CLI entrypoints
```

That rename changes import paths and YAML `target` strings, so it still needs a
separate change set. The current layout is intentionally non-breaking.

## Validation

```powershell
.\scripts\run_tests.ps1
python scripts\check_models.py
python scripts\check_env.py
```
