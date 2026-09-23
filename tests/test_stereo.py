import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Delay, HighPass, LowPass, Reverb, Saturate, Song  # noqa: E402
from paprika.fx import Fx, FxContext  # noqa: E402
from paprika.patch import Patch, SuperSaw  # noqa: E402
from paprika.render import render_song  # noqa: E402

SR = 44100


class Halve(Fx):
    def process_mono(self, x, ctx):
        return x * 0.5


def noise(shape):
    return np.random.default_rng(0).normal(0, 0.2, shape).astype(np.float32)


class TestFxBase(unittest.TestCase):
    def test_mono_subclass_on_stereo(self):
        out = Halve().process(np.ones((100, 2), np.float32), FxContext())
        self.assertEqual(out.shape, (100, 2))
        self.assertAlmostEqual(float(out.max()), 0.5)

    def test_mono_subclass_on_mono(self):
        out = Halve().process(np.ones(100, np.float32), FxContext())
        self.assertEqual(out.shape, (100,))

    def test_legacy_process_override_still_works(self):
        class Legacy(Fx):
            def process(self, buf, ctx):
                return buf * 2

        s = Song(bars=1)
        t = s.track("organ")
        t.note("C4", 0, 1)
        t.fx(Legacy())
        out = render_song(s, None, report=False)
        self.assertTrue(np.all(np.isfinite(out.audio)))

    def test_legacy_mono_only_override(self):
        class MonoOnly(Fx):
            def process(self, buf, ctx):
                return buf.mean(axis=1) if buf.ndim == 2 else buf

        s = Song(bars=1)
        t = s.track("organ")
        t.note("C4", 0, 1)
        t.fx(MonoOnly())
        out = render_song(s, None, report=False)
        self.assertEqual(out.tracks["organ"].shape[1], 2)

    def test_existing_fx_accept_both_shapes(self):
        ctx = FxContext(sr=SR)
        for fx in (Reverb(mix=0.5), Delay(1 / 8, 0.3, 0.3), LowPass(1000, res=0.3),
                   HighPass(200), Saturate(3),
                   Delay(1 / 8, 0.4, 0.4, ping_pong=True, tone=2000)):
            m = fx.process(noise(4000), ctx)
            st = fx.process(noise((4000, 2)), ctx)
            self.assertEqual(m.ndim, 1, type(fx).__name__)
            self.assertEqual(st.shape, (4000, 2), type(fx).__name__)

    def test_reverb_is_stereo(self):
        x = np.zeros((SR, 2), np.float32)
        x[0] = 1
        y = Reverb(room=0.8, mix=1.0).process(x, FxContext(sr=SR))
        self.assertGreater(np.abs(y[:, 0] - y[:, 1]).max(), 1e-3)

    def test_ping_pong_alternates(self):
        x = np.zeros((SR, 2), np.float32)
        x[0] = 1
        y = Delay(1 / 8, fb=0.5, mix=1.0, ping_pong=True).process(
            x, FxContext(sr=SR, bpm=120))
        d = int(0.25 * SR)
        self.assertGreater(abs(y[d, 0]), abs(y[d, 1]) * 5)
        self.assertGreater(abs(y[2 * d, 1]), abs(y[2 * d, 0]) * 5)

    def test_delay_tone_darkens_repeats(self):
        x = np.zeros(SR, np.float32)
        x[:200] = noise(200)
        bright = Delay(1 / 8, fb=0.6, mix=1.0).process(x, FxContext(sr=SR))
        dark = Delay(1 / 8, fb=0.6, mix=1.0, tone=1500).process(x, FxContext(sr=SR))
        d = int(0.25 * SR) * 3
        seg = slice(d, d + 400)

        def hf(y):
            return np.abs(np.diff(y[seg])).sum()
        self.assertLess(hf(dark), hf(bright) * 0.7)


class TestStereoRender(unittest.TestCase):
    def test_track_buffers_are_stereo(self):
        s = Song(bars=1)
        t = s.track(Patch(osc=[SuperSaw(spread=1)], name="ss"))
        t.note("C4", 0, 2)
        out = render_song(s, None, report=True, print_report=False)
        self.assertEqual(out.tracks["ss"].shape[1], 2)
        self.assertLess(np.corrcoef(out.audio[:, 0], out.audio[:, 1])[0, 1], 0.99)
        self.assertIn("PAPRIKA", out.report)


if __name__ == "__main__":
    unittest.main()
