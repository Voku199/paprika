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


def zero_cross_freq(x, sr=SR):
    s = np.signbit(x)
    idx = np.nonzero(s[1:] != s[:-1])[0]
    return (len(idx) / 2) / (x.size / sr)


def acf_freq(x, sr=SR, fmin=50.0, fmax=2000.0):
    """Výška tónu z autokorelace s parabolickou interpolací vrcholu."""
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    r = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * x.size)) ** 2)[:x.size]
    lo, hi = int(sr / fmax), int(sr / fmin)
    k = lo + int(np.argmax(r[lo:hi]))
    a, b, c = r[k - 1], r[k], r[k + 1]
    return sr / (k + 0.5 * (a - c) / (a - 2 * b + c))


class TestLfo(unittest.TestCase):
    def test_shapes_range_and_rate(self):
        for sh in dsp.LFO_SHAPES:
            y = dsp.lfo(sh, 4.0, SR, SR, seed=1)
            self.assertLessEqual(np.abs(y).max(), 1.0 + 1e-9, sh)
        self.assertAlmostEqual(zero_cross_freq(dsp.lfo("sine", 4.0, SR, SR)), 4.0,
                               delta=0.5)

    def test_unknown_shape(self):
        with self.assertRaises(ValueError):
            dsp.lfo("wobble", 1, 10, SR)


class TestNoiseAndPluck(unittest.TestCase):
    def test_colors_spectral_tilt(self):
        def hi_share(x):
            sp = np.abs(np.fft.rfft(x)) ** 2
            f = np.fft.rfftfreq(x.size, 1 / SR)
            return sp[f > 4000].sum() / sp.sum()
        w, p, b = (dsp.colored_noise(SR, c, seed=3) for c in ("white", "pink", "brown"))
        self.assertGreater(hi_share(w), hi_share(p))
        self.assertGreater(hi_share(p), hi_share(b))
        self.assertLessEqual(np.abs(b).max(), 1.0)

    def test_unknown_color(self):
        with self.assertRaises(ValueError):
            dsp.colored_noise(10, "blue")

    def test_karplus_pitch(self):
        # autokorelace: nejsilnější harmonická nemusí být základní tón
        y = dsp.karplus(220.0, SR, SR, decay=2.0, bright=0.3)
        self.assertAlmostEqual(acf_freq(y[2000:2000 + 8192]), 220.0, delta=220 * 0.01)
        self.assertAlmostEqual(acf_freq(dsp.karplus(1318.5, SR, SR)[2000:10192]),
                               1318.5, delta=1318.5 * 0.01)

    def test_karplus_decays(self):
        y = dsp.karplus(330.0, 2 * SR, SR, decay=0.5)
        self.assertLess(np.abs(y[-2000:]).max(), 0.01)


class TestIrAndBlep(unittest.TestCase):
    def test_ir_shape_energy_decorrelation(self):
        ir = dsp.synth_ir(1.5, SR, seed=2)
        self.assertEqual(ir.shape[1], 2)
        self.assertAlmostEqual(float((ir[:, 0].astype(np.float64) ** 2).sum()), 1.0,
                               places=2)
        self.assertLess(abs(np.corrcoef(ir[:, 0], ir[:, 1])[0, 1]), 0.3)

    def test_ir_decays_60db(self):
        ir = dsp.synth_ir(1.0, SR, predelay=0.0, early=0.0)
        head = np.abs(ir[:2000]).max()
        tail = np.abs(ir[int(1.0 * SR):]).max()
        self.assertLess(tail, head * 10 ** (-50 / 20))

    def test_blep_saw_bounded(self):
        cyc = 110.0 * np.arange(SR) / SR
        y = dsp.blep_saw(cyc, 110.0 / SR)
        self.assertLessEqual(np.abs(y).max(), 1.1)
        self.assertGreater(np.abs(y).max(), 0.9)


if __name__ == "__main__":
    unittest.main()
