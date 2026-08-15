import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika.patch import (ADSR, FM, LP, Noise, Patch, Saw,  # noqa: E402
                           Sine, Square, Tri, midi_to_freq)
from paprika.presets import preset, preset_names  # noqa: E402

SR = 44100


class TestADSR(unittest.TestCase):
    def test_length(self):
        env = ADSR(0.01, 0.1, 0.5, 0.2).render(SR, SR + SR // 5, SR)
        self.assertEqual(env.size, SR + SR // 5)

    def test_sustain_level(self):
        adsr = ADSR(0.01, 0.1, 0.5, 0.2)
        env = adsr.render(SR, SR + SR // 5, SR)
        # uprostřed držené noty musí sedět sustain
        self.assertAlmostEqual(float(env[SR // 2]), 0.5, places=5)

    def test_starts_at_zero_ends_at_zero(self):
        env = ADSR(0.01, 0.1, 0.5, 0.2).render(SR, SR + SR // 5, SR)
        self.assertAlmostEqual(float(env[0]), 0.0, places=6)
        self.assertAlmostEqual(float(env[-1]), 0.0, places=6)

    def test_attack_peaks_at_one(self):
        env = ADSR(0.05, 0.1, 0.4, 0.2).render(SR, SR + SR // 5, SR)
        self.assertAlmostEqual(float(env.max()), 1.0, places=3)

    def test_short_gate_does_not_overflow(self):
        env = ADSR(0.5, 0.5, 0.5, 0.5).render(100, 300, SR)
        self.assertEqual(env.size, 300)
        self.assertTrue(np.all(env >= 0.0))


class TestOscillators(unittest.TestCase):
    def test_all_oscillators_are_finite_and_bounded(self):
        for osc in (Sine(), Saw(), Square(), Tri(), Noise(), FM()):
            buf = osc.render(440.0, SR // 10, SR)
            self.assertEqual(buf.size, SR // 10, type(osc).__name__)
            self.assertTrue(np.all(np.isfinite(buf)), type(osc).__name__)
            self.assertLessEqual(float(np.abs(buf).max()), 1.25,
                                 type(osc).__name__)

    def test_ratio_shifts_frequency(self):
        low = Sine(1.0).render(220.0, SR, SR)
        high = Sine(2.0).render(220.0, SR, SR)
        # oktáva nahoru = dvakrát tolik průchodů nulou
        crossings = lambda b: int(np.sum(np.diff(np.sign(b)) != 0))  # noqa: E731
        self.assertAlmostEqual(crossings(high) / crossings(low), 2.0, places=1)

    def test_noise_is_deterministic(self):
        a = Noise(seed=5).render(440.0, 1000, SR)
        b = Noise(seed=5).render(440.0, 1000, SR)
        np.testing.assert_array_equal(a, b)


class TestVoice(unittest.TestCase):
    def test_midi_to_freq(self):
        self.assertAlmostEqual(midi_to_freq(69), 440.0)
        self.assertAlmostEqual(midi_to_freq(60), 261.6255653, places=5)

    def test_length_is_duration_plus_release(self):
        p = Patch(osc=[Sine()], adsr=ADSR(0.01, 0.1, 0.6, 0.3))
        voice = p.voice(440.0, 1.0, 100, SR)
        self.assertGreaterEqual(voice.size, int(1.3 * SR))
        self.assertLessEqual(voice.size, int(1.31 * SR) + 2)

    def test_no_nan(self):
        for name in preset_names():
            voice = preset(name).voice(220.0, 0.5, 100, SR)
            self.assertTrue(np.all(np.isfinite(voice)), name)

    def test_fade_on_edges(self):
        p = Patch(osc=[Sine()], adsr=ADSR(0.0, 0.0, 1.0, 0.0))
        voice = p.voice(440.0, 0.5, 127, SR)
        self.assertLess(abs(float(voice[0])), 1e-3)
        self.assertLess(abs(float(voice[-1])), 1e-3)
        self.assertGreater(float(np.abs(voice).max()), 0.1)

    def test_velocity_scales_amplitude(self):
        p = preset("organ")
        quiet = float(np.abs(p.voice(440.0, 0.5, 40, SR)).max())
        loud = float(np.abs(p.voice(440.0, 0.5, 120, SR)).max())
        self.assertLess(quiet, loud)

    def test_filter_removes_highs(self):
        bright = Patch(osc=[Saw()], adsr=ADSR(0.01, 0.1, 1.0, 0.1))
        dark = Patch(osc=[Saw()], adsr=ADSR(0.01, 0.1, 1.0, 0.1), filter=LP(400))
        b = np.abs(np.fft.rfft(bright.voice(220.0, 0.5, 127, SR)))
        d = np.abs(np.fft.rfft(dark.voice(220.0, 0.5, 127, SR)))
        freqs = np.fft.rfftfreq(bright.voice(220.0, 0.5, 127, SR).size, 1 / SR)
        high = freqs > 3000
        self.assertLess(d[high].sum(), b[high].sum() * 0.2)

    def test_presets_stay_below_unity(self):
        for name in preset_names():
            peak = float(np.abs(preset(name).voice(220.0, 1.0, 127, SR)).max())
            self.assertLess(peak, 1.0, f"{name} peak {peak:.3f}")

    def test_preset_instances_are_independent(self):
        a, b = preset("lead"), preset("lead")
        self.assertIsNot(a, b)
        self.assertIsNot(a.osc, b.osc)


if __name__ == "__main__":
    unittest.main()
