from share import *

import os

import lightning_compat as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


if not hasattr(pl, "Trainer"):
    raise SystemExit(
        "Training requires the official pytorch-lightning package. "
        "Create the conda environment from environment.yaml first."
    )


# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = int(os.environ.get('CONTROLNET_BATCH_SIZE', '1'))
accumulate_grad_batches = int(os.environ.get('CONTROLNET_ACCUMULATE_GRAD_BATCHES', '16'))
effective_batch_size = batch_size * accumulate_grad_batches
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./configs/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(
    gpus=1,
    precision=32,
    callbacks=[logger],
    accumulate_grad_batches=accumulate_grad_batches,
)

print(
    f'Gradient accumulation: batch_size={batch_size}, '
    f'accumulate_grad_batches={accumulate_grad_batches}, '
    f'effective_batch_size={effective_batch_size}'
)

# Train!
trainer.fit(model, dataloader)
