import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika.fx import (Delay, FxContext, HighPass, LowPass,  # noqa: E402
                        Reverb, Saturate, Sidechain)

SR = 44100


def impulse(n=SR, at=0):
    buf = np.zeros(n, dtype=np.float32)
    buf[at] = 1.0
    return buf


class TestReverb(unittest.TestCase):
    def setUp(self):
        self.ctx = FxContext(sr=SR, bpm=120)

    def test_tail_is_bounded(self):
        rv = Reverb(room=0.8, mix=0.5)
        self.assertGreater(rv.tail(self.ctx), 0.3)
        self.assertLess(rv.tail(self.ctx), 12.1)

    def test_does_not_outlast_its_tail(self):
        rv = Reverb(room=0.7, mix=1.0)
        n = 6 * SR
        out = rv.process(impulse(n), self.ctx)
        loud = np.nonzero(np.abs(out) > 1e-4)[0]
        self.assertLessEqual(float(loud[-1]) / SR, rv.tail(self.ctx) + 0.5)

    def test_bigger_room_rings_longer(self):
        small = Reverb(room=0.2, mix=1.0).process(impulse(4 * SR), self.ctx)
        big = Reverb(room=0.9, mix=1.0).process(impulse(4 * SR), self.ctx)
        tail = slice(2 * SR, None)
        self.assertGreater(float(np.abs(big[tail]).sum()),
                           float(np.abs(small[tail]).sum()))

    def test_zero_mix_is_transparent(self):
        dry = impulse(1000)
        np.testing.assert_array_equal(Reverb(mix=0.0).process(dry, self.ctx), dry)

    def test_finite(self):
        out = Reverb(room=0.95, mix=0.6).process(
            np.random.default_rng(0).normal(0, 0.3, SR).astype(np.float32),
            self.ctx)
        self.assertTrue(np.all(np.isfinite(out)))


class TestDelay(unittest.TestCase):
    def setUp(self):
        self.ctx = FxContext(sr=SR, bpm=120, beats_per_bar=4)

    def test_time_is_tempo_synced(self):
        d = Delay(1 / 8, fb=0.0, mix=1.0)
        # osminka při 120 BPM = 0,25 s
        self.assertAlmostEqual(d._sec(self.ctx), 0.25, places=6)
        out = d.process(impulse(SR), self.ctx)
        peak = int(np.argmax(np.abs(out[100:]))) + 100
        self.assertAlmostEqual(peak / SR, 0.25, places=2)

    def test_feedback_converges_to_zero(self):
        out = Delay(1 / 8, fb=0.5, mix=1.0).process(impulse(4 * SR), self.ctx)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertLess(float(np.abs(out[-SR:]).max()), 1e-3)

    def test_echo_decays_monotonically(self):
        d = Delay(1 / 8, fb=0.5, mix=1.0)
        out = d.process(impulse(3 * SR), self.ctx)
        step = int(0.25 * SR)
        peaks = [float(np.abs(out[i * step:(i + 1) * step]).max())
                 for i in range(1, 5)]
        self.assertTrue(all(a > b for a, b in zip(peaks, peaks[1:])), peaks)


class TestSaturate(unittest.TestCase):
    def test_stays_in_range(self):
        x = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
        out = Saturate(drive=8.0).process(x, FxContext(sr=SR))
        self.assertLessEqual(float(np.abs(out).max()), 1.0 + 1e-6)

    def test_adds_harmonics(self):
        t = np.arange(SR) / SR
        sine = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        out = Saturate(drive=6.0).process(sine, FxContext(sr=SR))
        spec = np.abs(np.fft.rfft(out))
        # třetí harmonická u 660 Hz musí být znatelně nad šumem
        self.assertGreater(spec[660], spec[1000] * 5)


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.ctx = FxContext(sr=SR)
        rng = np.random.default_rng(3)
        self.noise = rng.normal(0, 0.3, SR).astype(np.float32)

    def _band(self, buf, lo, hi):
        spec = np.abs(np.fft.rfft(buf)) ** 2
        freqs = np.fft.rfftfreq(buf.size, 1 / SR)
        return float(spec[(freqs >= lo) & (freqs < hi)].sum())

    def test_lowpass_cuts_highs(self):
        out = LowPass(500).process(self.noise, self.ctx)
        self.assertLess(self._band(out, 5000, 20000),
                        self._band(self.noise, 5000, 20000) * 0.05)

    def test_highpass_cuts_lows(self):
        out = HighPass(2000).process(self.noise, self.ctx)
        self.assertLess(self._band(out, 0, 200),
                        self._band(self.noise, 0, 200) * 0.05)


class TestSidechain(unittest.TestCase):
    def test_pumps_with_source(self):
        sr = SR
        source = np.zeros(sr, dtype=np.float32)
        source[:sr // 10] = 1.0            # "kopák" na začátku
        pad = np.ones(sr, dtype=np.float32) * 0.5
        ctx = FxContext(sr=sr, bpm=120, track_buffers={"drums": source})
        out = Sidechain("drums", amount=0.8).process(pad, ctx)
        self.assertLess(float(np.abs(out[:sr // 20]).mean()),
                        float(np.abs(out[-sr // 5:]).mean()))

    def test_missing_source_raises(self):
        ctx = FxContext(sr=SR, track_buffers={})
        with self.assertRaises(KeyError):
            Sidechain("neexistuje").process(np.ones(100, dtype=np.float32), ctx)


if __name__ == "__main__":
    unittest.main()
