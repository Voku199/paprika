import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Saturate, Sidechain, Song  # noqa: E402
from paprika.render import render_song  # noqa: E402


def base():
    s = Song(bpm=120, bars=1)
    a = s.track("organ", name="a")
    a.notes("C4:4")
    d = s.drums(name="d")
    d.pattern("k...k...k...k...", bars=(0, 1))
    return s, a, d


class TestBus(unittest.TestCase):
    def test_bus_sums_members(self):
        s, a, d = base()
        b = s.bus("grp", vol=1.0)
        a.to(b)
        d.to("grp")
        out = render_song(s, None, report=False)
        self.assertIn("grp", out.buses)
        self.assertTrue(np.any(out.buses["grp"]))

    def test_bus_matches_direct_mix_at_unity(self):
        s, a, d = base()
        direct = render_song(s, None, report=False).audio
        s2, a2, d2 = base()
        b = s2.bus("grp")
        a2.to(b)
        d2.to(b)
        bussed = render_song(s2, None, report=False).audio
        # bus s vol 1 a pan 0 je průhledný (balance, ne další equal-power pan)
        np.testing.assert_allclose(bussed, direct, atol=1e-5)

    def test_bus_vol_zero_silences(self):
        s, a, d = base()
        b = s.bus("grp", vol=0.0)
        a.to(b)
        d.to(b)
        self.assertLess(np.abs(render_song(s, None, report=False).audio).max(), 1e-6)

    def test_bus_automation(self):
        s, a, d = base()
        b = s.bus("grp")
        a.to(b)
        d.to(b)
        b.automate("vol", [(0, 0.0), (0.5, 0.0), (0.51, 1.0)])
        out = render_song(s, None, report=False).audio
        self.assertLess(np.abs(out[:40000]).max(), 1e-6)

    def test_master_fx_applies(self):
        s, a, d = base()
        s.master.fx(Saturate(8.0))
        dry = render_song(base()[0], None, report=False).audio
        wet = render_song(s, None, report=False).audio
        n = min(dry.shape[0], wet.shape[0])
        self.assertGreater(np.abs(wet[:n] - dry[:n]).max(), 1e-3)

    def test_master_vol_automation(self):
        s, a, d = base()
        s.master.automate("vol", [(0, 0.0)])
        self.assertLess(np.abs(render_song(s, None, report=False).audio).max(), 1e-6)

    def test_sidechain_from_bus(self):
        s, a, d = base()
        b = s.bus("dr")
        d.to(b)
        a.fx(Sidechain("dr", amount=0.9))
        out = render_song(s, None, report=False)
        self.assertTrue(np.all(np.isfinite(out.audio)))

    def test_unknown_bus(self):
        s, a, d = base()
        with self.assertRaises(KeyError) as cm:
            a.to("nope")
        self.assertIn("nope", str(cm.exception))

    def test_bus_name_unique_against_tracks(self):
        s, a, d = base()
        self.assertEqual(s.bus("a").name, "a2")

    def test_cycle_is_reported(self):
        s, a, d = base()
        b = s.bus("grp")
        a.to(b)
        a.fx(Sidechain("grp"))
        with self.assertRaises(RuntimeError) as cm:
            render_song(s, None, report=False)
        self.assertIn("cyklick", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
