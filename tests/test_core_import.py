import importlib
import unittest


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

