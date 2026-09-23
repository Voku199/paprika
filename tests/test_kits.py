import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Song  # noqa: E402
from paprika.drums import DRUM_CHARS, parse_pattern, render_hit  # noqa: E402
from paprika.kits import kit_names  # noqa: E402
from paprika.render import render_song  # noqa: E402

SR = 44100


def logspec(x, n=8192):
    x = np.pad(x[:n], (0, max(0, n - x.size)))
    return np.log10(np.abs(np.fft.rfft(x * np.hanning(n))) + 1e-6)


class TestKits(unittest.TestCase):
    def test_kit_list(self):
        self.assertEqual(set(kit_names()),
                         {"classic", "trap", "cinematic", "breakbeat", "industrial", "lofi"})

    def test_new_chars(self):
        for ch, name in (("x", "rim"), ("b", "boom"), ("y", "shaker"), ("p", "perc")):
            self.assertEqual(DRUM_CHARS[ch], name)
        self.assertEqual(parse_pattern("b..x")[1].name, "rim")

    def test_every_kit_every_instrument(self):
        for kit in kit_names():
            for name in set(DRUM_CHARS.values()):
                y = render_hit(name, SR, kit)
                self.assertTrue(np.all(np.isfinite(y)), (kit, name))
                self.assertLess(np.abs(y).max(), 1.0, (kit, name))
                self.assertGreater(np.abs(y).max(), 0.05, (kit, name))

    def test_classic_kick_unchanged(self):
        from paprika import kits
        np.testing.assert_array_equal(render_hit("kick", SR, "classic"),
                                      kits._kick(SR).astype(np.float32))

    def test_kits_sound_different(self):
        for name in ("kick", "snare", "hat"):
            specs = {k: logspec(render_hit(name, SR, k)) for k in kit_names()}
            ks = list(specs)
            for i, a in enumerate(ks):
                for b in ks[i + 1:]:
                    self.assertGreater(np.abs(specs[a] - specs[b]).mean(), 0.15,
                                       (name, a, b))

    def test_unknown_kit(self):
        with self.assertRaises(KeyError) as cm:
            render_hit("kick", SR, "jazz")
        self.assertIn("trap", str(cm.exception))

    def test_song_drums_kit(self):
        s = Song(bars=1)
        d = s.drums(kit="cinematic", name="d")
        d.pattern("b...k...b...k...", bars=(0, 1))
        self.assertEqual(d.kit, "cinematic")
        out = render_song(s, None, report=False)
        self.assertTrue(np.any(out.tracks["d"]))

    def test_song_drums_unknown_kit(self):
        with self.assertRaises(KeyError):
            Song(bars=1).drums(kit="jazz")


class TestRollSwing(unittest.TestCase):
    def test_roll_count_and_ramp(self):
        s = Song(bpm=120, bars=1)
        d = s.drums(kit="trap")
        d.roll("h", bar=0, beat=2, beats=1, rate=1 / 32, vel=(60, 110))
        self.assertEqual(len(d.hits), 8)
        self.assertEqual(d.hits[0].name, "hat")
        self.assertLess(d.hits[0].velocity, d.hits[-1].velocity)
        self.assertAlmostEqual(d.hits[0].start_beat, 2.0)
        self.assertAlmostEqual(d.hits[1].start_beat - d.hits[0].start_beat, 0.125)

    def test_roll_by_full_name(self):
        s = Song(bars=1)
        d = s.drums()
        d.roll("snare", bar=0, beats=0.5, rate=1 / 16)
        self.assertEqual({h.name for h in d.hits}, {"snare"})

    def test_roll_unknown_instrument(self):
        with self.assertRaises(ValueError):
            Song(bars=1).drums().roll("q", bar=0)

    def test_swing_shifts_odd_steps(self):
        s = Song(bars=1)
        d = s.drums()
        d.pattern("hhhh", bars=(0, 1), swing=0.5)
        st = sorted(h.start_beat for h in d.hits)
        self.assertAlmostEqual(st[0], 0.0)
        self.assertAlmostEqual(st[1], 0.25 + 0.125)
        self.assertAlmostEqual(st[2], 0.5)


if __name__ == "__main__":
    unittest.main()
