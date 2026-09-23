"""Regrese nalezené finálním review enginu v2."""

import os
import sys
import tracemalloc
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Delay, Patch, Song  # noqa: E402
from paprika.fx import Fx, FxContext  # noqa: E402
from paprika.oscillators import Osc  # noqa: E402
from paprika.render import render_song  # noqa: E402

SR = 44100


def peak_freq(seg):
    sp = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    return np.fft.rfftfreq(seg.size, 1 / SR)[np.argmax(sp)]


class TestDownwardGlide(unittest.TestCase):
    """`>` na nižší notu musí skončit na nižší notě (spec: `C2:1 >G1:1`)."""

    def ends_at(self, text, mono):
        s = Song(bpm=60, bars=1)
        t = s.track("organ", name="o", mono=mono, glide=0.25)
        t.notes(text)
        buf = render_song(s, None, report=False).tracks["o"][:, 0]
        return peak_freq(buf[int(3.2 * SR):int(3.9 * SR)])

    def test_poly_slide_down(self):
        self.assertAlmostEqual(self.ends_at("A4:2 >A3:2", mono=False), 220, delta=6)

    def test_mono_slide_down_after_rest(self):
        self.assertAlmostEqual(self.ends_at("A4:1 _:1 >A3:2", mono=True), 220, delta=6)


class TestV1CustomExtensions(unittest.TestCase):
    """Rozšíření napsaná podle v1 dokumentace musí dál fungovat."""

    def test_v1_process_override_gets_mono_buffer(self):
        class Tremolo(Fx):
            def process(self, buf, ctx):
                t = np.arange(buf.shape[0]) / ctx.sr
                return buf * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))

        s = Song(bars=1)
        s.track("organ", name="o").fx(Tremolo()).notes("C4:4")
        out = render_song(s, None, report=False)
        self.assertEqual(out.tracks["o"].shape[1], 2)

    def test_v1_oscillator_signature(self):
        class MyOsc(Osc):
            def render(self, freq, n, sr):
                return np.sin(2 * np.pi * freq * np.arange(n) / sr).astype(np.float32)

        y = Patch(osc=[MyOsc()]).voice(220.0, 0.2, sr=SR)
        self.assertTrue(np.any(y))

    def test_v1_oscillator_in_round_robin_patch(self):
        class MyOsc(Osc):
            def render(self, freq, n, sr):
                return np.ones(n, np.float32)

        y = Patch(osc=[MyOsc()], rr=2).voice(220.0, 0.2, sr=SR, rr_index=1)
        self.assertTrue(np.any(y))

    def test_v1_filter_signature(self):
        class MyFlt:
            def apply(self, x, sr, env=None):
                return x * 0.5

            def key(self):
                return ("MyFlt",)

        y = Patch(filter=MyFlt()).voice(220.0, 0.2, sr=SR)
        self.assertTrue(np.any(y))


class TestMasterRouting(unittest.TestCase):
    def test_to_master_keeps_track_audible(self):
        s = Song(bars=1)
        s.track("organ", name="o").notes("C4:4")
        s.tracks[0].to(s.master)
        self.assertGreater(np.abs(render_song(s, None, report=False).audio).max(), 0.01)

    def test_to_master_by_name(self):
        s = Song(bars=1)
        t = s.track("organ", name="o")
        t.notes("C4:4").to("master")
        self.assertIsNone(t.bus)

    def test_foreign_bus_rejected(self):
        other = Song(bars=1).bus("x")
        s = Song(bars=1)
        with self.assertRaises(ValueError):
            s.track("organ").to(other)


class TestDelayMemory(unittest.TestCase):
    def test_tone_delay_memory_is_linear(self):
        x = np.random.default_rng(0).normal(0, 0.1, (10 * SR, 2)).astype(np.float32)
        ctx = FxContext(sr=SR, bpm=120)
        tracemalloc.start()
        Delay(1 / 8, fb=0.9, mix=0.3, tone=3000, ping_pong=True).process(x, ctx)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # vstup má ~3.5 MB; lineární implementace se vejde do pár kopií
        self.assertLess(peak, 20 * x.nbytes)


if __name__ == "__main__":
    unittest.main()
