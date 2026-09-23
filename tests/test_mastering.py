import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import MASTERING_STYLES, Song, mastering  # noqa: E402
from paprika.fx import FxContext, apply_fx  # noqa: E402
from paprika.render import render_song  # noqa: E402

SR = 44100


class TestMastering(unittest.TestCase):
    def test_styles(self):
        self.assertEqual(set(MASTERING_STYLES), {"cinematic", "darkpop", "lofi"})

    def test_fresh_instances(self):
        a, b = mastering("cinematic"), mastering("cinematic")
        self.assertTrue(a)
        for x, y in zip(a, b):
            self.assertIsNot(x, y)

    def test_every_style_processes_stereo(self):
        x = np.random.default_rng(0).normal(0, 0.2, (2 * SR, 2)).astype(np.float32)
        ctx = FxContext(sr=SR)
        for style in MASTERING_STYLES:
            y = x
            for fx in mastering(style):
                y = apply_fx(fx, y, ctx)
            self.assertEqual(y.shape, x.shape, style)
            self.assertTrue(np.all(np.isfinite(y)), style)
            self.assertLess(np.abs(y).max(), 2.0, style)

    def test_cinematic_cuts_rumble(self):
        t = np.arange(2 * SR) / SR
        x = np.stack([np.sin(2 * np.pi * 20 * t)] * 2, 1).astype(np.float32) * 0.3
        y = x
        for fx in mastering("cinematic"):
            y = apply_fx(fx, y, FxContext(sr=SR))
        self.assertLess(np.abs(y[SR:]).max(), 0.3 * 0.6)

    def test_unknown_style(self):
        with self.assertRaises(KeyError) as cm:
            mastering("polka")
        self.assertIn("cinematic", str(cm.exception))

    def test_on_master_bus(self):
        s = Song(bars=2)
        s.track("organ").notes("C4:4 | E4:4")
        s.master.fx(*mastering("cinematic"))
        out = render_song(s, None, report=False)
        self.assertTrue(np.all(np.isfinite(out.audio)))


if __name__ == "__main__":
    unittest.main()
