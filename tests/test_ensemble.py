import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika.patch import ADSR, LFO, LP, Ensemble, Patch, Saw, Sine, SuperSaw  # noqa: E402

SR = 44100
FLAT = ADSR(0.001, 0.01, 1.0, 0.05)


def section(**kw):
    return Patch(osc=[Saw()], adsr=FLAT, ensemble=Ensemble(**kw))


class TestEnsemble(unittest.TestCase):
    def test_output_is_stereo_and_wide(self):
        y = section(voices=4, spread=0.8).voice(220.0, 0.5, sr=SR)
        self.assertEqual(y.shape[1], 2)
        self.assertLess(np.corrcoef(y[:, 0], y[:, 1])[0, 1], 0.9)

    def test_onsets_are_smeared(self):
        p = Patch(osc=[Sine()], adsr=ADSR(0.0005, 0.01, 1.0, 0.05),
                  ensemble=Ensemble(voices=2, spread=1.0, smear=0.02, detune=0.0, seed=3))
        y = p.voice(440.0, 0.3, sr=SR)
        first = [int(np.argmax(np.abs(y[:, c]) > 0.01)) for c in (0, 1)]
        self.assertGreater(abs(first[0] - first[1]), int(0.001 * SR))

    def test_length_includes_smear(self):
        plain = Patch(osc=[Saw()], adsr=FLAT).voice(220.0, 0.5, sr=SR)
        ens = section(smear=0.03).voice(220.0, 0.5, sr=SR)
        self.assertEqual(ens.shape[0], plain.shape[0] + int(0.03 * SR))

    def test_deterministic_per_rr_index(self):
        p = section(voices=3)
        a = p.voice(220.0, 0.3, sr=SR, rr_index=1)
        b = p.voice(220.0, 0.3, sr=SR, rr_index=1)
        c = p.voice(220.0, 0.3, sr=SR, rr_index=2)
        np.testing.assert_array_equal(a, b)
        self.assertGreater(np.abs(a - c).max(), 1e-3)

    def test_key_changes(self):
        base = Patch(osc=[Saw()])
        keys = {base.key(), Patch(osc=[Saw()], ensemble=Ensemble()).key(),
                Patch(osc=[Saw()], ensemble=Ensemble(voices=6)).key(),
                Patch(osc=[Saw()], ensemble=Ensemble(detune=20)).key()}
        self.assertEqual(len(keys), 4)

    def test_pitch_stays_centered_with_vibrato_and_filter(self):
        p = Patch(osc=[Saw()], adsr=FLAT, filter=LP(3000, env=0.3),
                  vibrato=LFO(5, 0.1), ensemble=Ensemble(voices=4, detune=8))
        y = p.voice(220.0, 1.0, sr=SR).mean(axis=1)
        seg = y[SR // 4:SR // 4 + 16384] * np.hanning(16384)
        f0 = np.fft.rfftfreq(16384, 1 / SR)[np.argmax(np.abs(np.fft.rfft(seg)))]
        self.assertAlmostEqual(f0, 220.0, delta=220 * 0.03)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_stereo_oscillator_inside_ensemble(self):
        p = Patch(osc=[SuperSaw(voices=3, spread=0.5)], adsr=FLAT,
                  ensemble=Ensemble(voices=2))
        y = p.voice(220.0, 0.3, sr=SR)
        self.assertEqual(y.shape[1], 2)
        self.assertLess(np.abs(y).max(), 2.0)

    def test_level_comparable_to_single_voice(self):
        single = Patch(osc=[Saw()], adsr=FLAT).voice(220.0, 0.5, sr=SR)
        ens = section(voices=6).voice(220.0, 0.5, sr=SR)
        r1 = np.sqrt(np.mean(single ** 2))
        r2 = np.sqrt(np.mean(ens ** 2))
        self.assertLess(abs(20 * np.log10(r2 / r1)), 6.0)

    def test_bad_voices(self):
        with self.assertRaises(ValueError):
            Ensemble(voices=0)


if __name__ == "__main__":
    unittest.main()
