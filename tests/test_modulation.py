import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika.patch import (ADSR, BP, LFO, LP, Formant, Patch,  # noqa: E402
                           PitchEnv, Pulse, Saw, Sine, SuperSaw)

SR = 44100
FLAT = ADSR(0.001, 0.01, 1.0, 0.01)


def inst_freq(x):
    """Okamžitá frekvence z rozestupů nulových průchodů."""
    s = np.signbit(x)
    idx = np.nonzero(s[1:] != s[:-1])[0]
    return SR / (2 * np.diff(idx))


class TestLfoObject(unittest.TestCase):
    def test_delay_silences_start(self):
        c = LFO(5, 1.0, delay=0.2).curve(SR, SR)
        self.assertLess(np.abs(c[:int(0.19 * SR)]).max(), 1e-9)
        self.assertGreater(np.abs(c[int(0.5 * SR):]).max(), 0.9)

    def test_sync_uses_bpm(self):
        self.assertAlmostEqual(LFO(sync=1 / 4).hz(120), 2.0)

    def test_pitch_env_curve(self):
        c = PitchEnv(12, 0.1).curve(SR, SR)
        self.assertAlmostEqual(c[0], 12.0)
        self.assertLess(abs(c[int(0.1 * SR)]), 0.2)
        with self.assertRaises(ValueError):
            PitchEnv(shape="wobble")


class TestVoiceModulation(unittest.TestCase):
    def test_vibrato_moves_pitch(self):
        p = Patch(osc=[Sine()], adsr=FLAT, vibrato=LFO(4, 1.0))
        f = inst_freq(p.voice(440.0, 1.0, sr=SR)[1000:-2000])
        self.assertGreater(f.max(), 440 * 2 ** (0.8 / 12))
        self.assertLess(f.min(), 440 * 2 ** (-0.8 / 12))

    def test_pitch_env_starts_high(self):
        p = Patch(osc=[Sine()], adsr=FLAT, pitch_env=PitchEnv(12, 0.2))
        f = inst_freq(p.voice(220.0, 1.0, sr=SR))
        self.assertGreater(f[:5].mean(), 350)
        self.assertAlmostEqual(f[-20:].mean(), 220, delta=5)

    def test_glide_path(self):
        p = Patch(osc=[Sine()], adsr=FLAT)
        y = p.voice(220.0, 1.0, sr=SR, glide_path=((0.0, 220.0), (0.5, 440.0)),
                    glide_time=0.05)
        f = inst_freq(y)
        self.assertAlmostEqual(f[5:30].mean(), 220, delta=5)
        self.assertAlmostEqual(f[-40:-10].mean(), 440, delta=8)

    def test_rr_variants_differ(self):
        p = Patch(osc=[Saw()], rr=3)
        a = p.voice(220.0, 0.2, sr=SR, rr_index=0)
        b = p.voice(220.0, 0.2, sr=SR, rr_index=1)
        self.assertGreater(np.abs(a - b).max(), 1e-3)

    def test_tremolo_modulates_level(self):
        p = Patch(osc=[Sine()], adsr=FLAT, tremolo=LFO(4, 0.8))
        y = np.abs(p.voice(440.0, 1.0, sr=SR))
        blocks = [y[i:i + 2000].max() for i in range(0, 40000, 2000)]
        self.assertLess(min(blocks), max(blocks) * 0.5)

    def test_vel_to_cutoff_brightens(self):
        p = Patch(osc=[Saw()], filter=LP(600, res=0.2), vel_to_cutoff=1.0)

        def hi(y):
            return np.abs(np.fft.rfft(y))[2000:].sum()
        loud = p.voice(110.0, 0.3, 127, SR, tone_vel=127)
        soft = p.voice(110.0, 0.3, 127, SR, tone_vel=30)
        self.assertGreater(hi(loud), hi(soft) * 1.5)

    def test_fast_path_matches_v1_without_filter(self):
        p = Patch(osc=[Sine()], adsr=FLAT)
        y = p.voice(440.0, 0.1, sr=SR)
        t = np.arange(y.size) / SR
        env = FLAT.render(int(round(0.1 * SR)), y.size, SR)
        ref = np.sin(2 * np.pi * 440 * t) * env * (100 / 127) ** 1.6
        np.testing.assert_allclose(y[200:-200], ref[200:-200], atol=1e-4)


class TestPatchFilters(unittest.TestCase):
    def test_resonant_lp_bp_formant_finite(self):
        for flt in (LP(800, res=0.9, env=0.5), BP(1200, res=0.5), Formant("o"),
                    Formant("a", morph=("a", "u", 0.5)), [LP(3000), Formant("e")],
                    LP(900, lfo=LFO(2, 1.0))):
            y = Patch(osc=[Saw()], filter=flt).voice(165.0, 0.6, sr=SR)
            self.assertTrue(np.all(np.isfinite(y)), repr(flt))
            self.assertLess(np.abs(y).max(), 1.5, repr(flt))
            self.assertGreater(np.abs(y).max(), 0.01, repr(flt))

    def test_formant_vowels_differ(self):
        a = Patch(osc=[Saw()], filter=Formant("a")).voice(165.0, 0.4, sr=SR)
        u = Patch(osc=[Saw()], filter=Formant("u")).voice(165.0, 0.4, sr=SR)
        diff = np.abs(np.abs(np.fft.rfft(a)) - np.abs(np.fft.rfft(u))).sum()
        self.assertGreater(diff, 1.0)

    def test_formant_unknown_vowel(self):
        with self.assertRaises(ValueError):
            Formant("y")

    def test_filter_lfo_moves_brightness(self):
        p = Patch(osc=[Saw()], adsr=FLAT, filter=LP(400, lfo=LFO(1, 2.0)))
        y = p.voice(110.0, 1.0, sr=SR)

        def bright(seg):
            sp = np.abs(np.fft.rfft(seg))
            return sp[200:].sum() / sp.sum()
        vals = [bright(y[i:i + 4096]) for i in range(0, 40000, 4096)]
        self.assertGreater(max(vals), min(vals) * 1.5)

    def test_every_field_changes_key(self):
        base = Patch(osc=[Saw()])
        variants = [
            Patch(osc=[Saw()], vibrato=LFO()), Patch(osc=[Saw()], tremolo=LFO()),
            Patch(osc=[Saw()], pitch_env=PitchEnv()),
            Patch(osc=[Saw()], vel_to_cutoff=0.5), Patch(osc=[Saw()], rr=2),
            Patch(osc=[Saw()], drift=5.0),
            Patch(osc=[Saw()], filter=LP(1000, res=0.5)),
            Patch(osc=[Saw()], filter=LP(1000, lfo=LFO())),
            Patch(osc=[Saw()], filter=[LP(1000), BP(500)]),
            Patch(osc=[Pulse(pwm=LFO(0.5, 0.2))]),
        ]
        keys = {base.key()} | {v.key() for v in variants}
        self.assertEqual(len(keys), len(variants) + 1)

    def test_stereo_voice_shape(self):
        y = Patch(osc=[SuperSaw(spread=1)], filter=LP(2000)).voice(220.0, 0.2, sr=SR)
        self.assertEqual(y.ndim, 2)
        self.assertEqual(y.shape[1], 2)

    def test_pulse_pwm_with_lfo(self):
        y = Patch(osc=[Pulse(pwm=LFO(1, 0.3))]).voice(110.0, 0.5, sr=SR)
        self.assertTrue(np.all(np.isfinite(y)))


if __name__ == "__main__":
    unittest.main()
