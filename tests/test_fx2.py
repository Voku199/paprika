import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika.fx import EQ, Filter, FxContext  # noqa: E402
from paprika.fx_dyn import Compressor, Gate  # noqa: E402

SR = 44100
CTX = FxContext(sr=SR, bpm=120)


def tone(f, sec=1.0, amp=0.9):
    return (amp * np.sin(2 * np.pi * f * np.arange(int(sec * SR)) / SR)).astype(np.float32)


def band_amp(y, f):
    sp = np.abs(np.fft.rfft(y[SR // 2:] * np.hanning(y.size - SR // 2)))
    fr = np.fft.rfftfreq(y.size - SR // 2, 1 / SR)
    return sp[np.argmin(np.abs(fr - f))]


class TestFilterEq(unittest.TestCase):
    def test_filter_modes(self):
        x = tone(100, amp=0.4) + tone(5000, amp=0.4)
        lo = Filter("lp", 500).process(x, CTX)
        hi = Filter("hp", 1000).process(x, CTX)
        bp = Filter("bp", 5000, res=0.5).process(x, CTX)
        self.assertLess(band_amp(lo, 5000), band_amp(x, 5000) * 0.05)
        self.assertLess(band_amp(hi, 100), band_amp(x, 100) * 0.05)
        self.assertLess(band_amp(bp, 100), band_amp(bp, 5000) * 0.05)

    def test_filter_bad_mode(self):
        with self.assertRaises(ValueError) as cm:
            Filter("wobble")
        self.assertIn("notch", str(cm.exception))

    def test_filter_automatable(self):
        self.assertEqual(Filter.AUTOMATABLE, ("cutoff", "res"))

    def test_eq_boosts_low_shelf(self):
        x = tone(100, amp=0.2)
        y = EQ(low=(200, 12.0)).process(x, CTX)
        self.assertGreater(np.abs(y[SR // 2:]).max(), np.abs(x).max() * 2.5)

    def test_eq_peak_cut_and_high_shelf(self):
        x = tone(1000, amp=0.5) + tone(8000, amp=0.3)
        y = EQ(mid=[(1000, -12.0, 2.0)], high=(6000, 6.0)).process(x, CTX)
        self.assertLess(band_amp(y, 1000), band_amp(x, 1000) * 0.35)
        self.assertGreater(band_amp(y, 8000), band_amp(x, 8000) * 1.6)

    def test_eq_stereo(self):
        x = np.stack([tone(100), tone(100)], 1)
        self.assertEqual(EQ(low=(200, 6)).process(x, CTX).shape, x.shape)


class TestDynamics(unittest.TestCase):
    def test_compressor_reduces_loud_keeps_quiet(self):
        c = Compressor(threshold=-20, ratio=8, attack=0.001, release=0.05)
        self.assertLess(np.abs(c.process(tone(200, amp=0.9), CTX)[SR // 2:]).max(), 0.4)
        quiet = c.process(tone(200, amp=0.02), CTX)
        self.assertAlmostEqual(np.abs(quiet[SR // 2:]).max(), 0.02, delta=0.004)

    def test_compressor_makeup(self):
        c = Compressor(threshold=-40, ratio=1.0, makeup=6.0)
        y = c.process(tone(200, amp=0.1), CTX)
        self.assertAlmostEqual(np.abs(y[SR // 2:]).max(), 0.2, delta=0.02)

    def test_compressor_stereo_linked(self):
        x = np.stack([tone(200, amp=0.9), tone(200, amp=0.05)], 1)
        y = Compressor(threshold=-20, ratio=10, attack=0.001).process(x, CTX)
        # levý kanál stáhne i pravý — stereo obraz se nerozjede
        self.assertLess(np.abs(y[SR // 2:, 1]).max(), 0.04)

    def test_gate_follows_pattern(self):
        x = np.ones(SR, np.float32)  # 1 s = 2 beaty = 8 šestnáctin
        y = Gate("x.......", rate=1 / 16).process(x, CTX)
        step = int(0.125 * SR)
        self.assertGreater(y[step // 2], 0.9)
        self.assertLess(y[step + step // 2], 0.05)

    def test_gate_depth(self):
        y = Gate("x.", depth=0.5).process(np.ones(SR, np.float32), CTX)
        step = int(0.125 * SR)
        self.assertAlmostEqual(float(y[step + step // 2]), 0.5, delta=0.05)

    def test_gate_respects_beat_offset(self):
        ctx = FxContext(sr=SR, bpm=120, beat_offset=0.25)  # začátek na 2. šestnáctině
        y = Gate("x.").process(np.ones(SR, np.float32), ctx)
        self.assertLess(y[int(0.03 * SR)], 0.05)

    def test_gate_bad_pattern(self):
        with self.assertRaises(ValueError):
            Gate("x-y")

    def test_gate_stereo(self):
        y = Gate("x.").process(np.ones((SR, 2), np.float32), CTX)
        self.assertEqual(y.shape, (SR, 2))


if __name__ == "__main__":
    unittest.main()
