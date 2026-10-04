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

## Verified Local Environment

`environment.control-gpu.lock.yaml` is an exported snapshot of the local
`control-gpu` environment used to run the current test suite and inference
entrypoints. It is a reproducibility record, not a replacement for
`environment.yaml`.

- Python 3.10.21
- PyTorch 2.7.0+cu128
- Torchvision 0.22.0+cu128
- Gradio 3.16.2
- Transformers 4.19.2

Recreate it with:

```text
conda env create -f environment.control-gpu.lock.yaml
conda activate control-gpu
```

This lock file does not include an official PyTorch Lightning installation with
`Trainer`. It is suitable for inference and the standard-library test suite,
but not for training. Use the `control` environment created from
`environment.yaml` for training and weight-transfer work.

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

The lock file has no machine-specific `prefix:` entry, so it can be committed
and recreated on another checkout.

## Verified Training Environment

`environment.control.lock.yaml` records the verified `control` training
environment. It keeps the versions declared by `environment.yaml`, including
Python 3.8.5, PyTorch 1.12.1, Torchvision 0.13.1, Gradio 3.16.2, and PyTorch
Lightning 1.5.0. The exported environment has an official `Trainer`.

```text
conda env create -f environment.control.lock.yaml
conda activate control
python scripts\check_env.py
```

On this machine, the user-level `.condarc` routed conda through a Tsinghua
mirror that returned HTTP 403, and the user-level pip index lacked the pinned
Gradio release. The verified lock removes the mirror entries and can be used
with an official PyPI index when installing or repairing pip dependencies:

```powershell
$env:PIP_INDEX_URL = 'https://pypi.org/simple'
```
