import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class CoreImportTests(unittest.TestCase):
    def test_core_model_helpers_import(self):
        for name in ["torch", "einops", "omegaconf"]:
            try:
                importlib.import_module(name)
            except Exception as exc:
                self.skipTest(f"{name} is unavailable: {exc}")

        from cldm.model import create_model, load_state_dict, resolve_ckpt_path

        self.assertTrue(callable(create_model))
        self.assertTrue(callable(load_state_dict))
        self.assertTrue(callable(resolve_ckpt_path))
