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

The repository contains a `pytorch_lightning/` shim so inference-only code can
run without importing the full Lightning package. The shim does not provide
`Trainer`; training scripts need the real PyTorch Lightning package from
`environment.yaml`.

Verify the active interpreter and imports:

```powershell
python scripts\check_env.py
```
