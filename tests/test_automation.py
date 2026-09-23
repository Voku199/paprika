import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import LowPass, Reverb, Song  # noqa: E402
from paprika.automation import Automation  # noqa: E402
from paprika.render import render_song  # noqa: E402

SR = 44100


class TestCurves(unittest.TestCase):
    def sample(self, pts, curve):
        # 4 s při 120 BPM = 2 takty
        return Automation("vol", pts, curve).sample(4 * SR, SR, 120, 4)

    def test_lin_hits_points(self):
        y = self.sample([(0, 0.0), (1, 1.0)], "lin")
        self.assertAlmostEqual(y[0], 0.0)
        self.assertAlmostEqual(y[SR], 0.5, places=3)
        self.assertAlmostEqual(y[-1], 1.0)

    def test_exp_geometric_midpoint(self):
        y = self.sample([(0, 100.0), (1, 10000.0)], "exp")
        self.assertAlmostEqual(y[SR], 1000.0, delta=5)

    def test_s_curve_and_step(self):
        s = self.sample([(0, 0.0), (1, 1.0)], "s")
        self.assertLess(s[SR // 2], 0.25)
        self.assertAlmostEqual(s[SR], 0.5, places=2)
        st = self.sample([(0, 0.0), (1, 1.0)], "step")
        self.assertEqual(st[SR], 0.0)
        self.assertEqual(st[2 * SR + 5], 1.0)

    def test_single_point_is_constant(self):
        y = self.sample([(3, 0.4)], "lin")
        self.assertTrue(np.allclose(y, 0.4))

    def test_exp_rejects_zero(self):
        with self.assertRaises(ValueError) as cm:
            Automation("x.cutoff", [(0, 0.0), (1, 10.0)], "exp")
        self.assertIn("kladn", str(cm.exception))

    def test_unknown_curve(self):
        with self.assertRaises(ValueError):
            Automation("vol", [(0, 1)], "wobble")

    def test_points_must_not_go_back(self):
        with self.assertRaises(ValueError):
            Automation("vol", [(4, 1), (2, 0)])

    def test_empty_points(self):
        with self.assertRaises(ValueError):
            Automation("vol", [])

    def test_offset_for_bar_slice(self):
        y = Automation("vol", [(0, 0.0), (2, 1.0)]).sample(10, SR, 120, 4, beat_offset=4)
        self.assertAlmostEqual(y[0], 0.5, places=3)


class TestRenderAutomation(unittest.TestCase):
    def song(self):
        s = Song(bpm=120, bars=2)
        t = s.track("organ", name="o")
        t.notes("C4:4 | C4:4")
        return s, t

    def test_vol_fade_in(self):
        s, t = self.song()
        t.automate("vol", [(0, 0.0), (2, 0.8)])
        a = render_song(s, None, report=False).audio
        self.assertLess(np.abs(a[:SR // 4]).max(), np.abs(a[3 * SR:4 * SR]).max() * 0.3)

    def test_pan_sweep(self):
        s, t = self.song()
        t.automate("pan", [(0, -1.0), (2, 1.0)])
        a = render_song(s, None, report=False).audio
        early = slice(SR // 10, SR // 2)
        self.assertGreater(np.abs(a[early, 0]).sum(), np.abs(a[early, 1]).sum() * 5)

    def test_filter_sweep_brightens(self):
        s, t = self.song()
        t.fx(LowPass(200))
        t.automate("lowpass.cutoff", [(0, 200), (2, 10000)], "exp")
        a = render_song(s, None, report=False).audio[:, 0]

        def hi(x):
            # podíl výkonu nad 600 Hz (organ C4 má harmonické do ~1.6 kHz)
            sp = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
            f = np.fft.rfftfreq(x.size, 1 / SR)
            return sp[f > 600].sum() / sp.sum()
        self.assertGreater(hi(a[3 * SR:3 * SR + 8192]), hi(a[SR // 2:SR // 2 + 8192]) * 5)

    def test_named_and_duplicate_fx(self):
        s, t = self.song()
        t.fx(LowPass(5000), LowPass(5000), Reverb(mix=0.2, name="space"))
        t.automate("lowpass2.cutoff", [(0, 300)]).automate("space.mix", [(0, 0.5)])
        render_song(s, None, report=False)

    def test_unknown_target_lists_options(self):
        s, t = self.song()
        t.fx(LowPass(200))
        t.automate("lowpas.cutoff", [(0, 200)])
        with self.assertRaises(KeyError) as cm:
            render_song(s, None, report=False)
        self.assertIn("lowpass.cutoff", str(cm.exception))

    def test_non_automatable_param(self):
        s, t = self.song()
        t.fx(LowPass(200))
        t.automate("lowpass.poles", [(0, 1)])
        with self.assertRaises(KeyError) as cm:
            render_song(s, None, report=False)
        self.assertIn("lowpass.res", str(cm.exception))

    def test_bar_slice_respects_automation(self):
        s, t = self.song()
        t.automate("vol", [(0, 0.0), (1, 0.0), (1.01, 0.8)])
        a = render_song(s, None, report=False, bars=(1, 2)).audio
        self.assertGreater(np.abs(a[SR // 2:SR]).max(), 0.05)

    def test_curves_cleared_after_render(self):
        s, t = self.song()
        lp = LowPass(200)
        t.fx(lp)
        t.automate("lowpass.cutoff", [(0, 5000)])
        render_song(s, None, report=False)
        self.assertEqual(lp.value("cutoff"), 200.0)


if __name__ == "__main__":
    unittest.main()
