import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SrcLayoutTests(unittest.TestCase):
    def test_sitecustomize_exposes_src_packages(self):
        code = "import bootstrap, cldm, ldm, annotator, lightning_compat; print('ok')"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)
