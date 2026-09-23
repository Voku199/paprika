import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import LowPass, Patch, Saw, Song  # noqa: E402
from paprika.patch import LFO  # noqa: E402
from paprika.analyze import warnings_for  # noqa: E402
from paprika.render import render_song  # noqa: E402

SILENT = np.zeros((10, 2), np.float32)


def long_song(patch):
    s = Song(bars=20)
    t = s.track(patch)
    t.notes(" | ".join(["C4:4"] * 20))
    return s, t


def texts(warns):
    return [w.text for w in warns]


class TestMonoCompat(unittest.TestCase):
    def test_phase_flip_is_warned(self):
        s = Song(bars=1)
        s.track("organ", name="o").notes("C4:4")
        buf = np.random.default_rng(0).normal(0, .2, 88200).astype(np.float32)
        st = np.stack([buf, -buf], 1)
        w = warnings_for(s, st, {"o": st}, 0.5)
        self.assertTrue(any("mono" in t for t in texts(w)))

    def test_normal_stereo_not_warned(self):
        s = Song(bars=1)
        s.track("organ", name="o").notes("C4:4")
        buf = np.random.default_rng(0).normal(0, .2, 88200).astype(np.float32)
        st = np.stack([buf, buf * 0.8], 1)
        self.assertFalse(any("mono" in t for t in texts(warnings_for(s, st, {"o": st}, 0.5))))


class TestStaticSound(unittest.TestCase):
    def test_static_patch_gets_info(self):
        s, t = long_song(Patch(osc=[Saw()], name="p"))
        w = [x for x in warnings_for(s, SILENT, {}, 0.1) if "statický" in x.text]
        self.assertEqual(len(w), 1)
        self.assertEqual(w[0].level, "INFO")

    def test_humanize_silences(self):
        s, t = long_song(Patch(osc=[Saw()], name="p"))
        t.humanize(seed=1)
        self.assertFalse(any("statický" in x for x in texts(warnings_for(s, SILENT, {}, 0.1))))

    def test_modulated_patch_silences(self):
        s, t = long_song(Patch(osc=[Saw()], vibrato=LFO(), name="p"))
        self.assertFalse(any("statický" in x for x in texts(warnings_for(s, SILENT, {}, 0.1))))

    def test_automation_silences(self):
        s, t = long_song(Patch(osc=[Saw()], name="p"))
        t.fx(LowPass(900)).automate("lowpass.cutoff", [(0, 900), (20, 3000)], "exp")
        self.assertFalse(any("statický" in x for x in texts(warnings_for(s, SILENT, {}, 0.1))))

    def test_short_track_not_reported(self):
        s = Song(bars=8)
        s.track(Patch(osc=[Saw()], name="p")).notes(" | ".join(["C4:4"] * 8))
        self.assertFalse(any("statický" in x for x in texts(warnings_for(s, SILENT, {}, 0.1))))

    def test_capped_output(self):
        s = Song(bars=20)
        for i in range(6):
            s.track(Patch(osc=[Saw()], name=f"p{i}")).notes(" | ".join(["C4:4"] * 20))
        hits = [x for x in warnings_for(s, SILENT, {}, 0.1) if "statický" in x.text]
        self.assertLessEqual(len(hits), 3)


class TestReport(unittest.TestCase):
    def test_report_has_stereo_line(self):
        s = Song(bars=1)
        s.track("organ").notes("C4:4")
        rep = render_song(s, None, print_report=False).report
        self.assertIn("STEREO", rep)
        self.assertIn("korelace", rep)


if __name__ == "__main__":
    unittest.main()
