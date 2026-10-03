import os
from pathlib import Path

import torch

import config
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def resolve_ckpt_path(ckpt_path):
    """Resolve a checkpoint path, falling back to CONTROLNET_MODEL_DIR."""
    path = Path(ckpt_path).expanduser()
    if path.exists():
        return path
    candidate = config.MODEL_DIR / path.name
    if candidate.exists():
        return candidate
    return path


def resolve_config_path(config_path):
    """Resolve a YAML config path, falling back to the repository configs dir."""
    path = Path(config_path).expanduser()
    if path.exists():
        return path
    candidate = config.CONFIG_DIR / path.name
    if candidate.exists():
        return candidate
    return path


def load_state_dict(ckpt_path, location='cpu'):
    resolved = resolve_ckpt_path(ckpt_path)
    extension = os.path.splitext(str(resolved))[1]
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(str(resolved), device=location)
    else:
        state_dict = get_state_dict(torch.load(str(resolved), map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    # Strip keys that don't exist in the original ControlNet architecture
    # (newer transformers CLIP checkpoints add position_ids)
    state_dict.pop('cond_stage_model.transformer.text_model.embeddings.position_ids', None)
    print(f'Loaded state_dict from [{resolved}]')
    return state_dict


def create_model(config_path):
    resolved = resolve_config_path(config_path)
    model_config = OmegaConf.load(str(resolved))
    model = instantiate_from_config(model_config.model).cpu()
    print(f'Loaded model config from [{resolved}]')
    return model
