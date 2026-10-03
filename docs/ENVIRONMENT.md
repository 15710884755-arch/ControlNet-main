# Environment

## Primary Environment

The authoritative environment is `environment.yaml`:

- Python 3.8.5
- CUDA toolkit 11.3
- PyTorch 1.12.1
- Torchvision 0.13.1
- Gradio 3.16.2
- PyTorch Lightning 1.5.0

Create it with:

```text
conda env create -f environment.yaml
conda activate control
```

## Pip Mirror

`requirements.txt` mirrors the pip-level dependencies for environments that do
not use conda. PyTorch must be installed separately with the correct CUDA
wheel. `requirements-dev.txt` adds test and lint tooling.

## Local Runtime Notes

The repository contains a `src/pytorch_lightning_shim/` package so inference-only
code can run without importing the full Lightning package. `lightning_compat.py`
imports official PyTorch Lightning when it is installed and falls back to the
shim otherwise. The shim does not provide `Trainer`; training scripts need the
real PyTorch Lightning package from `environment.yaml`.

Verify the active interpreter and imports:

```powershell
python scripts\check_env.py
python scripts\check_env.py --strict
```

`--strict` returns a non-zero exit code when the active interpreter differs
from `environment.yaml`. Resolve training and reproducibility work inside the
conda environment created from `environment.yaml`.
