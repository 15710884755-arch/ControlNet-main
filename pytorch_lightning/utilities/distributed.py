"""Shim for pytorch_lightning.utilities.distributed."""
import functools


def rank_zero_only(fn):
    """No-op rank_zero_only decorator."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper
