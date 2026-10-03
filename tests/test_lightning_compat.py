import unittest


class LightningCompatTests(unittest.TestCase):
    def test_lightning_compat_selects_a_module(self):
        import lightning_compat

        self.assertIn(
            lightning_compat.pl.__name__,
            {"pytorch_lightning", "pytorch_lightning_shim"},
        )
        self.assertTrue(hasattr(lightning_compat, "LightningModule"))
        self.assertTrue(hasattr(lightning_compat, "Callback"))
