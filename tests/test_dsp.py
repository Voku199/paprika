import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import dsp  # noqa: E402

SR = 44100


def tone(f, sec=0.5):
    t = np.arange(int(sec * SR)) / SR
    return np.sin(2 * np.pi * f * t).astype(np.float32)


def gain_db(y, x):
    s = slice(len(x) // 2, None)  # bez přechodového jevu
    return 20 * np.log10(np.sqrt(np.mean(y[s] ** 2)) / np.sqrt(np.mean(x[s] ** 2)))


class TestBiquad(unittest.TestCase):
    def test_lp_minus3db_at_cutoff(self):
        x = tone(1000)
        self.assertAlmostEqual(gain_db(dsp.filt(x, "lp", 1000, SR), x), -3.0, delta=1.0)

    def test_hp_cuts_lows(self):
        x = tone(100)
        self.assertLess(gain_db(dsp.filt(x, "hp", 2000, SR), x), -30)

    def test_resonance_peak_grows(self):
        x = tone(1000)
        flat = gain_db(dsp.filt(x, "lp", 1000, SR, q=dsp.res_to_q(0.0)), x)
        peak = gain_db(dsp.filt(x, "lp", 1000, SR, q=dsp.res_to_q(0.8)), x)
        self.assertGreater(peak, flat + 10)

    def test_res_to_q_range(self):
        self.assertAlmostEqual(dsp.res_to_q(0.0), 0.707, places=3)
        self.assertAlmostEqual(dsp.res_to_q(1.0), 12.0, places=3)

    def test_all_kinds_finite(self):
        x = np.random.default_rng(0).normal(0, 0.3, SR).astype(np.float32)
        for k in dsp.KINDS:
            y = dsp.filt(x, k, 800, SR, q=2.0, gain_db=6.0)
            self.assertTrue(np.all(np.isfinite(y)), k)

    def test_unknown_kind_lists_options(self):
        with self.assertRaises(ValueError) as cm:
            dsp.biquad("lowpas", 100, SR)
        self.assertIn("lp", str(cm.exception))

    def test_stereo_axis0(self):
        x = np.stack([tone(100), tone(5000)], axis=1)
        y = dsp.filt(x, "lp", 500, SR)
        self.assertEqual(y.shape, x.shape)
        self.assertGreater(np.abs(y[:, 0]).max(), np.abs(y[:, 1]).max() * 5)


class TestVarying(unittest.TestCase):
    def test_constant_curve_matches_static(self):
        x = tone(300)
        a = dsp.filt(x, "lp", 1200, SR, q=2.0)
        b = dsp.filt_varying(x, "lp", np.full(x.size, 1200.0), SR, q=2.0)
        np.testing.assert_allclose(a, b, atol=1e-4)

    def test_sweep_has_no_block_clicks(self):
        x = tone(220, 1.0)
        cut = np.geomspace(200, 8000, x.size)
        y = dsp.filt_varying(x, "lp", cut, SR, q=dsp.res_to_q(0.5))
        self.assertLess(np.abs(np.diff(y)).max(), 0.2)

    def test_onepole_varying_and_static(self):
        x = tone(3000)
        s = dsp.onepole(x, 500.0, SR, poles=2)
        v = dsp.onepole(x, np.full(x.size, 500.0), SR, poles=2)
        np.testing.assert_allclose(s, v, atol=1e-4)
        self.assertLess(gain_db(s, x), -12)


if __name__ == "__main__":
    unittest.main()
