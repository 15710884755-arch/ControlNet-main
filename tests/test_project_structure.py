import ast
import sys
import tempfile
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

    def test_plain_templates_use_chinese_names(self):
        template_dir = ROOT / "claude" / "plain"
        expected = [
            "说明.md",
            "_模板-计划文档.md",
            "_模板-产品需求文档.md",
            "_模板-测试用例.md",
            "_模板-待定事项.md",
        ]
        for name in expected:
            self.assertTrue((template_dir / name).is_file(), f"missing template: {name}")
        legacy = [
            "README.md",
            "_TEMPLATE-plan.md",
            "_TEMPLATE-prd.md",
            "_TEMPLATE-test-cases.md",
            "_TEMPLATE-tbd.md",
        ]
        for name in legacy:
            self.assertFalse((template_dir / name).exists(), f"legacy template remains: {name}")

    def test_large_runtime_assets_are_ignored(self):
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ["models/", "src/annotator/ckpts/", "*.pth", "*.bin"]:
            self.assertIn(pattern, ignored)

    def test_lightning_shim_does_not_shadow_official_package(self):
        self.assertFalse((ROOT / "pytorch_lightning").exists())
        self.assertTrue((ROOT / "lightning_compat.py").is_file())
        self.assertTrue((ROOT / "bootstrap.py").is_file())
        self.assertTrue((ROOT / "src" / "pytorch_lightning_shim").is_dir())

    def test_gradio_entrypoints_parse_and_use_configs(self):
        entries = sorted(ROOT.glob("gradio_*.py"))
        self.assertGreaterEqual(len(entries), 8)
        for path in entries:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            if path.name not in {"gradio_annotator.py", "gradio_scribble2image_sdxl.py"}:
                self.assertIn("create_model", source, path.name)
            self.assertNotIn("models/cldm_v15.yaml", source, path.name)

    def test_canny_web_ui_supports_cancellation(self):
        source = (ROOT / "gradio_canny2image.py").read_text(encoding="utf-8")
        self.assertIn('stop_button = gr.Button(value="停止生成", variant="stop")', source)
        self.assertIn("cancel_event.set()", source)
        self.assertIn("callback=lambda _: raise_if_cancelled()", source)
        self.assertIn("stop_button.click(fn=stop_generation", source)
        self.assertIn("stop_button.click(fn=None", source)
        self.assertIn("cancels=[run_event]", source)

    def test_scribble_web_ui_supports_canvas_and_cancellation(self):
        source = (ROOT / "gradio_scribble2image_interactive.py").read_text(encoding="utf-8")
        self.assertIn("control_sd15_scribble.pth", source)
        self.assertIn("gr.Blocks(title='ControlNet 与用户涂鸦')", source)
        self.assertIn("source='canvas'", source)
        self.assertIn("cancel_event.set()", source)
        self.assertIn("callback=lambda _: raise_if_cancelled()", source)
        self.assertIn("cancels=[run_event]", source)
        self.assertIn("GRADIO_SERVER_PORT", source)
        self.assertIn("MAX_IMAGE_RESOLUTION", source)
        self.assertIn("clamp_image_size", source)
        self.assertIn("samples[i:i + 1]", source)
        self.assertIn("prompt_weight", source)
        self.assertIn("effective_scale", source)
        self.assertIn("translate_prompt_to_english", source)
        self.assertIn("enhance_prompt_for_generation", source)

    def test_prompt_translation_has_chinese_fallback(self):
        from prompt_translation import contains_chinese, translate_prompt_to_english

        self.assertTrue(contains_chinese("笑脸"))
        self.assertFalse(contains_chinese("smiling face"))
        self.assertEqual(translate_prompt_to_english("笑脸"), "smiling face")

    def test_guangzhou_tower_prompt_includes_visual_landmark_details(self):
        from prompt_translation import enhance_prompt_for_generation

        prompt = enhance_prompt_for_generation("广州塔", "Canton Tower.")
        self.assertIn("hyperboloid lattice TV tower", prompt)
        self.assertIn("narrow twisted waist", prompt)
        self.assertIn("Guangzhou city skyline", prompt)

    def test_training_scripts_support_gradient_accumulation(self):
        for name in ("tutorial_train.py", "tutorial_train_sd21.py"):
            source = (ROOT / name).read_text(encoding="utf-8")
            ast.parse(source, filename=name)
            self.assertIn("CONTROLNET_ACCUMULATE_GRAD_BATCHES", source, name)
            self.assertIn("effective_batch_size = batch_size * accumulate_grad_batches", source, name)
            self.assertIn("accumulate_grad_batches=accumulate_grad_batches", source, name)

    def test_sdxl_scribble_web_ui_supports_offload_and_controlnet_fallback(self):
        source = (ROOT / "gradio_scribble2image_sdxl.py").read_text(encoding="utf-8")
        ast.parse(source, filename="gradio_scribble2image_sdxl.py")
        self.assertIn("StableDiffusionXLControlNetPipeline", source)
        self.assertIn("controlnet-scribble-sdxl-1.0", source)
        self.assertIn("controlnet-canny-sdxl-1.0", source)
        self.assertIn("enable_sequential_cpu_offload", source)
        self.assertIn("prompt_weight", source)
        self.assertIn("GRADIO_SERVER_PORT", source)

    def test_sdxl_download_script_covers_both_mirrors(self):
        source = (ROOT / "scripts" / "download_sdxl_scribble.py").read_text(encoding="utf-8")
        ast.parse(source, filename="scripts/download_sdxl_scribble.py")
        self.assertIn("parallel_download", source)
        self.assertIn("huggingface.co", source)
        self.assertIn("modelscope.cn", source)
        self.assertIn("controlnet-scribble-sdxl-1.0", source)
        self.assertIn("diffusion_pytorch_model.fp16.safetensors", source)

    def test_parallel_download_plans_byte_ranges(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import parallel_download
        finally:
            sys.path.pop(0)
        self.assertEqual(parallel_download.plan_chunks(10, 4), [(0, 3), (4, 7), (8, 9)])
        self.assertEqual(parallel_download.plan_chunks(8, 4), [(0, 3), (4, 7)])
        self.assertEqual(parallel_download.plan_chunks(0, 4), [])

    def test_parallel_download_treats_resumed_files_as_incomplete(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import parallel_download
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "weights.safetensors"
            self.assertFalse(parallel_download.is_complete(target))
            target.write_bytes(b"partially written")
            self.assertTrue(parallel_download.is_complete(target))
            parallel_download.journal_path(target).write_text("[]", encoding="utf-8")
            self.assertFalse(parallel_download.is_complete(target))

    def test_local_environment_lock_is_portable(self):
        lock = (ROOT / "environment.control-gpu.lock.yaml").read_text(encoding="utf-8")
        self.assertIn("name: control-gpu", lock)
        self.assertIn("torch==2.7.0+cu128", lock)
        self.assertIn("torchvision==0.22.0+cu128", lock)
        self.assertNotIn("\nprefix:", lock)

    def test_training_environment_lock_is_portable(self):
        lock = (ROOT / "environment.control.lock.yaml").read_text(encoding="utf-8")
        self.assertIn("name: control", lock)
        self.assertIn("python=3.8.5", lock)
        self.assertIn("pytorch=1.12.1", lock)
        self.assertIn("pytorch-lightning==1.5.0", lock)
        self.assertIn("lmdb==1.8.1", lock)
        self.assertNotIn("\nprefix:", lock)
        self.assertNotIn("mirrors.tuna.tsinghua.edu.cn", lock)
