import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Song  # noqa: E402
from paprika.render import render_song  # noqa: E402
from paprika.sfx import SFX_NAMES, render_sfx  # noqa: E402

SR = 44100


class TestSfx(unittest.TestCase):
    def test_riser_ends_on_bar(self):
        s = Song(bpm=120, bars=4)
        fx = s.sfx(name="fx")
        fx.riser(to_bar=2, bars=1)
        buf = render_song(s, None, report=False).tracks["fx"]
        edge = 2 * 2 * SR  # takt 2 při 120 BPM = 4 s
        self.assertGreater(np.abs(buf[edge - 2000:edge]).max(), 0.05)
        self.assertLess(np.abs(buf[edge + 300:edge + 3000]).max(), 0.01)
        self.assertLess(np.abs(buf[:edge - 2 * SR - 10]).max(), 1e-6)

    def test_riser_grows(self):
        y = render_sfx("riser", ("noise", 4.0), SR, 120)
        n = y.shape[0]
        self.assertGreater(np.abs(y[-n // 8:]).max(), np.abs(y[:n // 8]).max() * 3)

    def test_impact_peaks_early(self):
        y = render_sfx("impact", (1.0,), SR, 120)
        self.assertLess(np.argmax(np.abs(y).max(axis=1)), int(0.1 * SR))

    def test_reverse_swell_ends_loud_on_bar(self):
        s = Song(bpm=120, bars=4)
        s.sfx(name="fx").reverse_swell(to_bar=2, beats=2)
        buf = render_song(s, None, report=False).tracks["fx"]
        edge = 4 * SR
        self.assertGreater(np.abs(buf[edge - 3000:edge]).max(),
                           np.abs(buf[edge - SR:edge - SR + 3000]).max() * 2)
        self.assertLess(np.abs(buf[edge + 500:edge + 3000]).max(), 0.01)

    def test_whoosh_pans(self):
        y = render_sfx("whoosh", (2.0,), SR, 120)
        n = y.shape[0]
        first, last = y[:n // 4], y[-n // 4:]
        self.assertGreater(np.abs(first[:, 0]).sum(), np.abs(first[:, 1]).sum())
        self.assertGreater(np.abs(last[:, 1]).sum(), np.abs(last[:, 0]).sum())

    def test_all_kinds_finite_stereo_bounded(self):
        s = Song(bpm=100, bars=8)
        fx = s.sfx()
        fx.riser(2).impact(2).braam(3, "D2").subdrop(4).reverse_swell(6).whoosh(6, beats=2)
        self.assertEqual(sorted({h.name for h in fx.hits}), sorted(SFX_NAMES))
        out = render_song(s, None, report=False)
        self.assertTrue(np.all(np.isfinite(out.audio)))
        for name, params in {(h.name, h.params) for h in fx.hits}:
            y = render_sfx(name, params, SR, 100)
            self.assertEqual(y.ndim, 2, name)
            self.assertLess(np.abs(y).max(), 1.0, name)
            self.assertGreater(np.abs(y).max(), 0.1, name)

    def test_riser_kinds(self):
        for kind in ("noise", "tone", "both"):
            self.assertTrue(np.all(np.isfinite(render_sfx("riser", (kind, 2.0), SR, 120))))
        with self.assertRaises(ValueError):
            Song(bars=2).sfx().riser(1, kind="laser")

    def test_braam_pitch_by_name(self):
        s = Song(bars=4)
        fx = s.sfx()
        fx.braam(1, "D2", bars=1)
        self.assertEqual(fx.hits[0].params[0], 38)

    def test_humanize_keeps_timing(self):
        s = Song(bars=4)
        fx = s.sfx()
        fx.riser(2)
        start = fx.hits[0].start_beat
        fx.humanize(timing=0.1, vel=0.2, seed=1)
        self.assertEqual(fx.hits[0].start_beat, start)

    def test_show_symbols(self):
        s = Song(bars=4)
        s.sfx(name="fx").impact(1).riser(3, bars=1)
        roll = s.show(res=4)
        self.assertIn("I", roll)
        self.assertIn("R", roll)


if __name__ == "__main__":
    unittest.main()
