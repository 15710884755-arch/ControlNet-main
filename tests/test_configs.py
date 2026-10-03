import importlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ModelConfigTests(unittest.TestCase):
    def setUp(self):
        try:
            self.omegaconf = importlib.import_module("omegaconf")
        except Exception as exc:
            self.skipTest(f"omegaconf is unavailable: {exc}")

    def test_v15_core_targets(self):
        cfg = self.omegaconf.OmegaConf.load(str(ROOT / "configs" / "cldm_v15.yaml"))
        self.assertEqual(cfg.model.target, "cldm.cldm.ControlLDM")
        self.assertEqual(
            cfg.model.params.control_stage_config.target,
            "cldm.cldm.ControlNet",
        )
        self.assertEqual(
            cfg.model.params.unet_config.target,
            "cldm.cldm.ControlledUnetModel",
        )

    def test_v21_core_targets(self):
        cfg = self.omegaconf.OmegaConf.load(str(ROOT / "configs" / "cldm_v21.yaml"))
        self.assertEqual(cfg.model.target, "cldm.cldm.ControlLDM")
        self.assertEqual(
            cfg.model.params.cond_stage_config.target,
            "ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder",
        )

