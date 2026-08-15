import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Song  # noqa: E402
from paprika.analyze import db, report, warnings_for  # noqa: E402
from paprika.render import render_song  # noqa: E402


def texts(warns, level=None):
    return [f"{w.where} {w.text}" for w in warns
            if level is None or w.level == level]


class TestOutOfScale(unittest.TestCase):
    def _warns(self, pitch_name, chromatic=False):
        s = Song(bpm=120, key="C", scale="major", bars=2)
        t = s.track("organ", chromatic=chromatic, name="lead")
        t.notes(f"C4:1 {pitch_name}:1 E4:1 G4:1 | C4:4", bar=0)
        out = render_song(s, None, report=False)
        return warnings_for(s, out.audio, out.tracks, out.raw_peak)

    def test_flags_note_outside_scale(self):
        found = texts(self._warns("D#5"))
        self.assertTrue(any("D#5" in f and "mimo" in f for f in found), found)

    def test_clean_song_has_no_scale_warning(self):
        found = texts(self._warns("D4"))
        self.assertFalse(any("mimo" in f for f in found), found)

    def test_chromatic_track_is_silent_about_it(self):
        found = texts(self._warns("D#5", chromatic=True))
        self.assertFalse(any("mimo" in f for f in found), found)


class TestMixWarnings(unittest.TestCase):
    def test_clipping_is_reported(self):
        s = Song(bpm=120, bars=2)
        for i in range(6):
            t = s.track("organ", vol=1.0, name=f"t{i}")
            t.notes("C4:4 | C4:4", bar=0)
        out = render_song(s, None, report=False)
        warns = warnings_for(s, out.audio, out.tracks, out.raw_peak)
        self.assertGreater(out.raw_peak, 1.0)
        self.assertTrue(any("clipping" in w.text for w in warns))

    def test_quiet_mix_is_reported(self):
        s = Song(bpm=120, bars=2)
        t = s.track("organ", vol=0.001, name="tichy")
        t.notes("C4:4 | C4:4", bar=0)
        out = render_song(s, None, report=False)
        warns = warnings_for(s, out.audio, out.tracks, out.raw_peak)
        self.assertTrue(any("potichu" in w.text for w in warns))


class TestPerformanceWarnings(unittest.TestCase):
    def test_flat_velocity_is_robotic(self):
        s = Song(bpm=120, bars=2)
        t = s.track("organ", name="lead")
        t.notes("C4:1 D4:1 E4:1 F4:1 | G4:1 A4:1 B4:1 C5:1", bar=0, vel=100)
        out = render_song(s, None, report=False)
        warns = warnings_for(s, out.audio, out.tracks, out.raw_peak)
        self.assertTrue(any("roboticky" in w.text for w in warns))

    def test_humanize_silences_the_warning(self):
        s = Song(bpm=120, bars=2)
        t = s.track("organ", name="lead")
        t.notes("C4:1 D4:1 E4:1 F4:1 | G4:1 A4:1 B4:1 C5:1", bar=0, vel=100)
        t.humanize(timing=0.01, vel=0.15, seed=1)
        out = render_song(s, None, report=False)
        warns = warnings_for(s, out.audio, out.tracks, out.raw_peak)
        self.assertFalse(any("roboticky" in w.text for w in warns))

    def test_low_minor_second_is_dissonant(self):
        s = Song(bpm=120, key="C", scale="chromatic", bars=2)
        t = s.track("subbass", chromatic=True, name="bass")
        t.chord([36, 37], bar=0, dur=4)
        t.chord([36, 37], bar=1, dur=4)
        out = render_song(s, None, report=False)
        warns = warnings_for(s, out.audio, out.tracks, out.raw_peak)
        self.assertTrue(any("disonance" in w.text for w in warns))

    def test_gap_in_arrangement_is_info(self):
        s = Song(bpm=120, bars=12)
        t = s.track("organ", name="lead")
        t.notes("C4:4", bar=0)
        t.notes("C4:4", bar=8)
        out = render_song(s, None, report=False)
        warns = warnings_for(s, out.audio, out.tracks, out.raw_peak)
        self.assertTrue(any("ticho" in w.text for w in warns))


class TestReportText(unittest.TestCase):
    def test_report_has_all_sections(self):
        s = Song(bpm=120, bars=2, title="test")
        t = s.track("epiano", name="piano")
        t.notes("C4:2 E4:2 | G4:4", bar=0)
        out = render_song(s, None, report=False)
        text = report(s, out.audio, out.tracks, out.raw_peak)
        for token in ("PAPRIKA", "render", "MIX", "BANDS", "TRACKS", "piano"):
            self.assertIn(token, text)

    def test_clean_song_says_ok(self):
        s = Song(bpm=120, bars=2)
        t = s.track("epiano", name="piano")
        t.notes("C4:2 E4:2 | G4:2 E4:2", bar=0)
        t.humanize(timing=0.01, vel=0.15, seed=2)
        out = render_song(s, None, report=False)
        warns = warnings_for(s, out.audio, out.tracks, out.raw_peak)
        self.assertEqual([w.text for w in warns if w.level == "WARN"], [])

    def test_db_helper(self):
        self.assertAlmostEqual(db(1.0), 0.0)
        self.assertAlmostEqual(db(0.5), -6.0206, places=3)
        self.assertEqual(db(0.0), -99.0)


if __name__ == "__main__":
    unittest.main()
