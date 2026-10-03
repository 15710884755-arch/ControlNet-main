"""Shim package to replace pytorch_lightning for inference-only use."""
import torch
import torch.nn as nn
import os
import random
import numpy as np


class LightningModule(nn.Module):
    """Minimal LightningModule shim."""
    
    def __init__(self):
        super().__init__()
        self.trainer = None
    
    def log(self, *args, **kwargs):
        """No-op log."""
        pass
    
    def configure_optimizers(self):
        return None


# re-export seed_everything for compat
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


from pytorch_lightning.utilities.distributed import rank_zero_only  # noqa: E402
from pytorch_lightning.callbacks import Callback  # noqa: E402
