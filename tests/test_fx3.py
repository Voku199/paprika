import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika.fx import FxContext  # noqa: E402
from paprika.fx_lofi import BitCrush, Tape  # noqa: E402
from paprika.fx_space import Chorus, Hall, Width  # noqa: E402

SR = 44100
CTX = FxContext(sr=SR, bpm=120)


def noise(shape):
    return np.random.default_rng(1).normal(0, 0.2, shape).astype(np.float32)


def corr(y):
    return np.corrcoef(y[:, 0], y[:, 1])[0, 1]


class TestHall(unittest.TestCase):
    def test_tail_bounds_every_preset(self):
        for p in ("room", "plate", "hall", "cathedral"):
            h = Hall(p, mix=1.0)
            x = np.zeros((int((h.tail(CTX) + 1) * SR), 2), np.float32)
            x[0] = 1
            y = h.process(x, CTX)
            loud = np.nonzero(np.abs(y).max(axis=1) > 1e-3)[0]
            self.assertLessEqual(loud[-1] / SR, h.tail(CTX) + 0.1, p)

    def test_cathedral_longer_than_room(self):
        self.assertGreater(Hall("cathedral").tail(CTX), Hall("room").tail(CTX) * 4)

    def test_size_scales_decay(self):
        self.assertGreater(Hall("hall", size=2.0).tail(CTX), Hall("hall").tail(CTX) * 1.8)

    def test_unknown_preset(self):
        with self.assertRaises(KeyError) as cm:
            Hall("stadium")
        self.assertIn("cathedral", str(cm.exception))

    def test_hall_widens_mono_source(self):
        x = noise(SR)
        x = np.stack([x, x], 1)
        self.assertLess(corr(Hall("hall", mix=1.0).process(x, CTX)), 0.5)

    def test_mono_input_stays_mono(self):
        self.assertEqual(Hall("room", mix=0.5).process(noise(SR), CTX).ndim, 1)

    def test_zero_mix_transparent(self):
        x = noise((1000, 2))
        np.testing.assert_array_equal(Hall(mix=0.0).process(x, CTX), x)


class TestChorusWidth(unittest.TestCase):
    def test_chorus_decorrelates(self):
        x = noise(SR)
        x = np.stack([x, x], 1)
        self.assertLess(corr(Chorus(mix=0.5).process(x, CTX)), 0.99)

    def test_chorus_mono_input(self):
        self.assertEqual(Chorus().process(noise(4000), CTX).shape, (4000,))

    def test_width_zero_is_mono(self):
        self.assertGreater(corr(Width(0.0).process(noise((SR, 2)), CTX)), 0.999)

    def test_width_two_is_wider(self):
        x = noise((SR, 2))
        x[:, 1] = 0.7 * x[:, 0] + 0.3 * x[:, 1]
        self.assertLess(corr(Width(2.0).process(x, CTX)), corr(x) - 0.1)

    def test_width_mono_input_unchanged(self):
        x = noise(100)
        np.testing.assert_array_equal(Width(2.0).process(x, CTX), x)


class TestLofi(unittest.TestCase):
    def test_bitcrush_quantizes(self):
        y = BitCrush(bits=3, downsample=1).process(noise(4000), CTX)
        self.assertLessEqual(len(np.unique(np.round(y, 6))), 2 ** 3 + 1)

    def test_bitcrush_downsample_holds(self):
        y = BitCrush(bits=16, downsample=4).process(noise(4000), CTX)
        self.assertEqual(y[0], y[3])

    def test_tape_finite_bounded(self):
        y = Tape(wow=1.0, flutter=1.0, drive=3.0, hiss=0.5).process(noise((SR, 2)), CTX)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertLess(np.abs(y).max(), 1.5)

    def test_tape_wobbles_pitch(self):
        t = np.arange(SR) / SR
        x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        y = Tape(wow=1.0, flutter=0.0, drive=1.0).process(x, CTX)
        self.assertGreater(np.abs(y - x).max(), 0.1)


if __name__ == "__main__":
    unittest.main()
