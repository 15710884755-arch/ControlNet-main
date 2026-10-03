import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTests(unittest.TestCase):
    def test_target_directories_exist(self):
        expected = [
            "configs",
            "scripts",
            "tests",
            "docs",
            "assets",
            "assets/samples",
            "src/cldm",
            "src/ldm",
            "src/annotator",
            "src/pytorch_lightning_shim",
            "claude/plain",
            "plan/review",
        ]
        for name in expected:
            self.assertTrue((ROOT / name).is_dir(), f"missing directory: {name}")

    def test_yaml_configs_are_outside_model_weights(self):
        self.assertTrue((ROOT / "configs" / "cldm_v15.yaml").is_file())
        self.assertTrue((ROOT / "configs" / "cldm_v21.yaml").is_file())
        self.assertFalse((ROOT / "models" / "cldm_v15.yaml").exists())
        self.assertFalse((ROOT / "models" / "cldm_v21.yaml").exists())

    def test_large_runtime_assets_are_ignored(self):
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ["models/", "src/annotator/ckpts/", "*.pth", "*.bin"]:
            self.assertIn(pattern, ignored)

    def test_lightning_shim_does_not_shadow_official_package(self):
        self.assertFalse((ROOT / "pytorch_lightning").exists())
        self.assertTrue((ROOT / "lightning_compat.py").is_file())
        self.assertTrue((ROOT / "src" / "pytorch_lightning_shim").is_dir())

    def test_gradio_entrypoints_parse_and_use_configs(self):
        entries = sorted(ROOT.glob("gradio_*.py"))
        self.assertGreaterEqual(len(entries), 8)
        for path in entries:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            if path.name != "gradio_annotator.py":
                self.assertIn("create_model", source, path.name)
            self.assertNotIn("models/cldm_v15.yaml", source, path.name)
