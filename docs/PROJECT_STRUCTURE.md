# Project Structure

This repository is the ControlNet 1.0 reference implementation plus local
runtime patches. It is a research/demo codebase, not a packaged service.

## Current Layout

```text
ControlNet-main/
|-- configs/                  YAML model configs
|-- cldm/                     ControlNet model and LDM integration
|-- ldm/                      Latent Diffusion building blocks
|-- annotator/                condition preprocessors and detector code
|-- assets/                   README images, test images, fonts, samples
|-- scripts/                  maintenance and diagnostic scripts
|-- tests/                    standard-library test suite
|-- docs/                     engineering and usage docs
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

## Deferred Migration

The long-term target is a `src/` package layout:

```text
src/
  controlnet/       cldm core
  diffusion/        ldm building blocks
  preprocessors/    annotator wrappers
  runtime/          model loading and device policy
  apps/             Gradio and CLI entrypoints
```

Move the packages only after the baseline is green. The package move changes
import paths and YAML `target` strings, so it needs its own change set under the
delivery workflow.

## Validation

```powershell
.\scripts\run_tests.ps1
python scripts\check_models.py
python scripts\check_env.py
```
