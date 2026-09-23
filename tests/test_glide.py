import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Patch, Saw, Song, parse  # noqa: E402
from paprika.render import _phrases, render_song  # noqa: E402

SR = 44100


def peak_freq(seg):
    sp = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    return np.fft.rfftfreq(seg.size, 1 / SR)[np.argmax(sp)]


class TestGlideNotation(unittest.TestCase):
    def test_prefix(self):
        n = parse("C2:1 >G1:1")
        self.assertFalse(n[0].glide)
        self.assertTrue(n[1].glide)
        self.assertEqual(n[1].pitch, 31)  # G1 (C4 = 60)

    def test_bar_check_unaffected(self):
        parse("C2:2 >G1:2 | C2:4")

    def test_glide_survives_all_notes(self):
        s = Song(bars=1)
        t = s.track("lead")
        t.notes("C4:2 >G4:2")
        self.assertTrue(t.all_notes()[1].glide)


class TestPhrases(unittest.TestCase):
    def test_mono_merges_legato(self):
        s = Song(bars=1)
        t = s.track("subbass", mono=True, glide=0.25)
        t.notes("C2:1 D2:1 _:1 E2:1")
        ph = _phrases(t)
        self.assertEqual(len(ph), 2)
        self.assertEqual(len(ph[0][3]), 2)
        self.assertEqual(ph[0][4], 0.25)
        self.assertAlmostEqual(ph[0][1], 2.0)

    def test_poly_glide_note(self):
        s = Song(bars=1)
        t = s.track("lead")
        t.notes("C4:2 >G4:2")
        ph = _phrases(t)
        self.assertEqual(ph[1][3][0][1], 60)
        self.assertEqual(ph[1][3][-1][1], 67)

    def test_poly_without_glide_is_one_phrase_per_note(self):
        s = Song(bars=1)
        t = s.track("lead")
        t.notes("(C4 E4 G4):4")
        self.assertEqual(len(_phrases(t)), 3)


class TestGlideRender(unittest.TestCase):
    def test_render_glide_ends_on_target(self):
        s = Song(bpm=60, bars=1)
        t = s.track("organ", mono=True, glide=0.5, name="o")
        t.notes("A3:2 A4:2")
        out = render_song(s, None, report=False).tracks["o"][:, 0]
        self.assertAlmostEqual(peak_freq(out[int(3.2 * SR):int(3.9 * SR)]), 440, delta=6)
        self.assertAlmostEqual(peak_freq(out[int(0.5 * SR):int(1.5 * SR)]), 220, delta=6)

    def test_rr_used_in_render(self):
        s = Song(bpm=120, bars=1)
        t = s.track(Patch(osc=[Saw()], rr=2, name="p"))
        t.notes("C4:1 C4:1 C4:1 C4:1")
        buf = render_song(s, None, report=False).tracks["p"][:, 0]
        a = buf[100:5000]
        b = buf[int(0.5 * SR) + 100:int(0.5 * SR) + 5000]
        self.assertGreater(np.abs(a - b).max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
