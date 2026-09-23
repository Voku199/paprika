import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika.oscillators import (FM, Additive, Noise, Pluck, Pulse, Saw,  # noqa: E402
                                 Sine, Square, SuperSaw, Tri)

SR = 44100


def zc_freq(x):
    s = np.signbit(x)
    return (np.count_nonzero(s[1:] != s[:-1]) / 2) / (x.size / SR)


class TestFrequencyCurves(unittest.TestCase):
    def test_scalar_matches_v1_sine(self):
        y = Sine().render(440.0, 1000, SR)
        t = np.arange(1000) / SR
        np.testing.assert_allclose(y, np.sin(2 * np.pi * 440 * t), atol=1e-5)

    def test_sine_follows_curve(self):
        n = SR
        f = np.concatenate([np.full(n // 2, 220.0), np.full(n - n // 2, 440.0)])
        y = Sine().render(f, n, SR)
        self.assertAlmostEqual(zc_freq(y[:SR // 4]), 220, delta=10)
        self.assertAlmostEqual(zc_freq(y[-SR // 4:]), 440, delta=10)

    def test_every_osc_accepts_curve(self):
        f = np.geomspace(110, 440, 8000)
        for osc in (Sine(), Saw(), Square(), Tri(), FM(), SuperSaw(), Pulse(),
                    Pluck(), Additive(), Noise()):
            y = osc.render(f, 8000, SR)
            self.assertEqual(y.shape[0], 8000, type(osc).__name__)
            self.assertTrue(np.all(np.isfinite(y)), type(osc).__name__)

    def test_bandlimit_uses_max_freq(self):
        f = np.geomspace(100, 8000, SR)
        y = Saw().render(f, SR, SR)
        sp = np.abs(np.fft.rfft(y[-4096:] * np.hanning(4096)))
        fr = np.fft.rfftfreq(4096, 1 / SR)
        self.assertLess(sp[fr > 21000].sum() / sp.sum(), 0.01)

    def test_harmonic_sum_matches_sine_series(self):
        from paprika.oscillators import _harmonic_sum
        cyc = 220.0 * np.arange(5000) / SR
        ks = np.arange(1, 40)
        w = 1.0 / ks
        ref = sum(np.sin(2 * np.pi * k * cyc) * wk for k, wk in zip(ks, w))
        np.testing.assert_allclose(_harmonic_sum(cyc, ks, w), ref, atol=1e-9)

    def test_phase_offset_changes_output(self):
        a = Saw().render(220.0, 1000, SR)
        b = Saw().render(220.0, 1000, SR, phase=0.3)
        self.assertGreater(np.abs(a - b).max(), 0.1)


class TestNewOscillators(unittest.TestCase):
    def test_supersaw_stereo_and_wide(self):
        y = SuperSaw(voices=7, spread=1.0).render(220.0, SR // 2, SR)
        self.assertEqual(y.shape, (SR // 2, 2))
        self.assertLess(np.corrcoef(y[:, 0], y[:, 1])[0, 1], 0.95)
        self.assertLessEqual(np.abs(y).max(), 1.5)

    def test_supersaw_mono_when_no_spread(self):
        osc = SuperSaw(spread=0)
        self.assertFalse(osc.stereo)
        self.assertEqual(osc.render(220.0, 1000, SR).ndim, 1)

    def test_pulse_width_changes_timbre(self):
        a = Pulse(width=0.5).render(220.0, 8192, SR)
        b = Pulse(width=0.1).render(220.0, 8192, SR)
        self.assertGreater(np.abs(a - b).mean(), 0.1)
        self.assertLessEqual(np.abs(b).max(), 1.2)

    def test_pluck_pitch_and_key(self):
        y = Pluck(decay=2.0).render(330.0, SR, SR)
        self.assertGreater(np.abs(y[:2000]).max(), np.abs(y[-2000:]).max())
        self.assertNotEqual(Pluck(bright=0.2).key(), Pluck(bright=0.8).key())

    def test_additive_decays_per_partial(self):
        osc = Additive(partials=((1, 1.0, 0.0), (5, 1.0, 0.05)))
        y = osc.render(200.0, SR, SR)
        fr = np.fft.rfftfreq(4096, 1 / SR)
        k = int(np.argmin(abs(fr - 1000)))
        early = np.abs(np.fft.rfft(y[:4096]))[k]
        late = np.abs(np.fft.rfft(y[-4096:]))[k]
        self.assertGreater(early, late * 10)

    def test_additive_inharmonicity_in_key(self):
        self.assertNotEqual(Additive(inharm=0.001).key(), Additive().key())

    def test_noise_color_in_key(self):
        self.assertNotEqual(Noise(color="pink").key(), Noise().key())


if __name__ == "__main__":
    unittest.main()
