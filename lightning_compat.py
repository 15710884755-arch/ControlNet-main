"""Use official PyTorch Lightning when available, otherwise use the local shim."""

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.utilities.distributed import rank_zero_only

    USING_SHIM = False
except ImportError:  # pragma: no cover - depends on the local environment
    import pytorch_lightning_shim as pl
    from pytorch_lightning_shim.callbacks import Callback
    from pytorch_lightning_shim.utilities.distributed import rank_zero_only

    USING_SHIM = True


LightningModule = pl.LightningModule


__all__ = [
    "Callback",
    "LightningModule",
    "USING_SHIM",
    "pl",
    "rank_zero_only",
]
