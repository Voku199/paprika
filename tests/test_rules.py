import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Song  # noqa: E402
from paprika.analyze import check_report, model_warnings, warnings_for  # noqa: E402
from paprika.harmony import HarmonyError  # noqa: E402
from paprika.render import render_song  # noqa: E402
from paprika.rules import theory_warnings  # noqa: E402


def song(key="C", scale="major", bars=4, theory="free"):
    return Song(bpm=120, key=key, scale=scale, bars=bars, theory=theory)


def found(warns, word, level=None):
    return [w for w in warns if word in w.text and (level is None or w.level == level)]


class TestPlanModel(unittest.TestCase):
    def test_bar_offset(self):
        s = song().harmony("C | F", bar=2)
        self.assertEqual([(p.start_beat, p.chord.label) for p in s.plan],
                         [(8.0, "C"), (12.0, "F")])

    def test_new_call_replaces_covered_bars(self):
        s = song().harmony("C | F | G | C").harmony("Am", bar=1)
        self.assertEqual([p.chord.label for p in s.plan], ["C", "Am", "G", "C"])

    def test_long_chord_is_truncated(self):
        s = song().harmony("C:8").harmony("F", bar=1)
        self.assertEqual([(p.start_beat, p.dur_beats, p.chord.label) for p in s.plan],
                         [(0.0, 4.0, "C"), (4.0, 4.0, "F")])

    def test_rest_is_not_in_plan(self):
        s = song().harmony("_ | C")
        self.assertEqual([p.start_beat for p in s.plan], [4.0])

    def test_chord_at(self):
        s = song().harmony("C | F")
        self.assertEqual(s.chord_at(3.99).chord.label, "C")
        self.assertEqual(s.chord_at(4.0).chord.label, "F")
        self.assertIsNone(s.chord_at(8.0))

    def test_roman_uses_song_key(self):
        s = song(key="D", scale="minor").harmony("i | V7")
        self.assertEqual([p.chord.label for p in s.plan], ["Dm", "A7"])

    def test_bad_plan_raises(self):
        with self.assertRaises(HarmonyError):
            song().harmony("C | F:3")

    def test_invalid_theory(self):
        with self.assertRaises(ValueError) as cm:
            Song(theory="loose")
        self.assertIn("strict", str(cm.exception))

    def test_section(self):
        s = song(bars=16).section("B", bars=(8, 16)).section("A", bars=(0, 8))
        self.assertEqual([x.name for x in s.sections], ["A", "B"])
        with self.assertRaises(ValueError):
            s.section("X", bars=(4, 4))


class TestBass(unittest.TestCase):
    def test_wrong_bass(self):
        s = song().harmony("C | F")
        s.track("organ", name="bass").notes("C3:4 | D3:4")
        hits = found(theory_warnings(s), "nepatří do akordu", "WARN")
        self.assertEqual(len(hits), 1)
        self.assertIn("bar 1", hits[0].text)
        self.assertEqual(hits[0].bar, 1)

    def test_inversion_is_fine(self):
        s = song().harmony("C | F")
        s.track("organ", name="bass").notes("C3:4 | A2:4")
        self.assertFalse(found(theory_warnings(s), "nepatří do akordu"))

    def test_pedal_point_is_not_reported(self):
        s = song().harmony("C | G")
        s.track("organ", name="bass").note("C3", at=0, dur=8)
        self.assertFalse(found(theory_warnings(s), "nepatří do akordu"))

    def test_without_plan_nothing(self):
        s = song()
        s.track("organ", name="bass").notes("C3:4 | D3:4")
        self.assertFalse(found(theory_warnings(s), "nepatří do akordu"))

    def test_capped(self):
        s = song(bars=8).harmony("C C C C C C C C")
        s.track("organ", name="bass").notes(" | ".join(["D3:4"] * 8))
        self.assertEqual(len(found(theory_warnings(s), "nepatří do akordu")), 3)


class TestCoverage(unittest.TestCase):
    def _song(self, theory):
        s = song(theory=theory).harmony("C | F")
        s.track("organ", name="lead").notes("E4:4 | D4:4")
        return s

    def test_missing_root_and_third(self):
        hits = found(theory_warnings(self._song("free")), "není slyšet", "INFO")
        self.assertEqual(len(hits), 2)
        self.assertIn("základ C", hits[0].text)
        self.assertIn("tercie A", hits[1].text)

    def test_strict_is_warn(self):
        self.assertTrue(found(theory_warnings(self._song("strict")), "není slyšet", "WARN"))


class TestNonChordTones(unittest.TestCase):
    def _warns(self, melody, theory="free", plan="C"):
        s = song(theory=theory).harmony(plan)
        s.track("organ", name="lead").notes(melody)
        return found(theory_warnings(s), "mimo akord")

    def test_appoggiatura_resolves_by_step(self):
        self.assertFalse(self._warns("D4:1 C4:1 E4:1 G4:1"))

    def test_unresolved_diatonic_is_info(self):
        hits = self._warns("F4:1 C4:1 E4:1 G4:1")
        self.assertEqual([w.level for w in hits], ["INFO"])

    def test_unresolved_chromatic_is_warn(self):
        hits = self._warns("F#4:1 C4:1 E4:1 G4:1")
        self.assertEqual([w.level for w in hits], ["WARN"])
        self.assertIn("mimo stupnici", hits[0].text)

    def test_weak_beat_is_ignored(self):
        self.assertFalse(self._warns("C4:1 F4:1 A4:1 G4:1"))

    def test_suspension_is_fine(self):
        self.assertFalse(self._warns("E4:4 | E4:1 D4:3", plan="C | G"))

    def test_strict_warns_diatonic(self):
        hits = self._warns("F4:1 C4:1 E4:1 G4:1", theory="strict")
        self.assertEqual([w.level for w in hits], ["WARN"])

    def test_humanized_correct_line_is_clean(self):
        s = song(bars=4).harmony("C | F | G | C")
        lead = s.track("organ", name="lead")
        lead.notes("E4:1 D4:1 C4:2 | A4:1 G4:1 F4:2 | D4:1 E4:1 D4:1 B3:1 | C4:4")
        lead.humanize(timing=0.015, seed=1)
        s.track("organ", name="bass").notes("C3:4 | F2:4 | G2:4 | C3:4")
        warns = theory_warnings(s)
        self.assertFalse(found(warns, "mimo akord"), warns)
        self.assertFalse(found(warns, "nepatří do akordu"), warns)


class TestParallels(unittest.TestCase):
    def _song(self, upper, lower, theory="free"):
        s = song(theory=theory)
        s.track("organ", name="upper").notes(upper)
        s.track("organ", name="lower").notes(lower)
        return s

    def test_isolated_fifths_are_info(self):
        s = self._song("G4:1 A4:1 G4:1 G4:1", "C4:1 D4:1 E4:1 E4:1")
        hits = found(theory_warnings(s), "paralelní kvinty")
        self.assertEqual([w.level for w in hits], ["INFO"])
        self.assertIn("G4/C4 → A4/D4", hits[0].text)

    def test_octave_doubling_is_intentional_in_free(self):
        s = self._song("C5:1 D5:1 E5:1 F5:1", "C4:1 D4:1 E4:1 F4:1")
        self.assertFalse(found(theory_warnings(s), "paralelní"))

    def test_strict_reports_doubling(self):
        s = self._song("C5:1 D5:1 E5:1 F5:1", "C4:1 D4:1 E4:1 F4:1", theory="strict")
        hits = found(theory_warnings(s), "paralelní oktávy", "WARN")
        self.assertEqual(len(hits), 1)

    def test_contrary_motion_is_fine(self):
        s = self._song("G4:1 F4:1 E4:1 D4:1", "C4:1 D4:1 C4:1 B3:1")
        self.assertFalse(found(theory_warnings(s), "paralelní"))

    def test_figuration_is_texture_in_free(self):
        s = self._song("C4:0.5 D4:0.5 F4:0.5 A4:0.5 F4:0.5 D4:0.5 C4:1",
                       "C3:0.5 D3:1.5 E3:2")
        self.assertFalse(found(theory_warnings(s), "paralelní"))
        s.theory = "strict"
        self.assertTrue(found(theory_warnings(s), "paralelní"))

    def test_chord_tracks_are_texture_in_free(self):
        s = song()
        s.track("organ", name="pad").notes("(C4 G4):1 (D4 A4):1 (C4 G4):1 (C4 G4):1")
        self.assertFalse(found(theory_warnings(s), "paralelní"))


class TestLeadingTone(unittest.TestCase):
    def _warns(self, melody, theory="free"):
        s = song(theory=theory).harmony("G7 | C")
        s.track("organ", name="lead").notes(melody)
        return found(theory_warnings(s), "citlivý tón")

    def test_resolves_up(self):
        self.assertFalse(self._warns("D5:3 B4:1 | C5:4"))

    def test_not_resolved(self):
        self.assertEqual([w.level for w in self._warns("D5:3 B4:1 | G4:4")], ["INFO"])
        self.assertEqual([w.level for w in self._warns("D5:3 B4:1 | G4:4", "strict")],
                         ["WARN"])


class TestLeaps(unittest.TestCase):
    def _warns(self, melody):
        s = song()
        s.track("organ", name="lead").notes(melody)
        return found(theory_warnings(s), "skok")

    def test_leap_with_recovery(self):
        self.assertFalse(self._warns("C5:1 D5:1 E5:1 C6:1 | B5:1 A5:1 G5:1 F5:1"))

    def test_leap_without_recovery(self):
        hits = self._warns("C5:1 D5:1 E5:1 C6:1 | D6:1 E6:1 D6:1 C6:1")
        self.assertEqual(len(hits), 1)
        self.assertIn("malou sextu", hits[0].text)

    def test_tritone(self):
        self.assertTrue(self._warns("E5:1 D5:1 F5:1 B4:1 | C5:4"))

    def test_bass_lines_are_not_melodies(self):
        self.assertFalse(self._warns("C3:1 D3:1 E3:1 C4:1 | D3:1 E3:1 D3:1 C3:1"))


class TestLowThirds(unittest.TestCase):
    def _warns(self, chord):
        s = song()
        s.track("organ", name="pad").notes(chord)
        return found(theory_warnings(s), "zablácené")

    def test_low_third(self):
        self.assertEqual(len(self._warns("(C2 E2):4")), 1)

    def test_high_enough(self):
        self.assertFalse(self._warns("(C3 E3):4"))

    def test_short_arpeggio_note(self):
        self.assertFalse(self._warns("(C2 E2):0.5 _:3.5"))


class TestStrictOnly(unittest.TestCase):
    def test_spacing(self):
        for theory, expect in (("free", 0), ("strict", 1)):
            s = song(theory=theory)
            s.track("organ", name="choir").notes("(C3 E3 G4 C5):4")
            self.assertEqual(len(found(theory_warnings(s), "dál než oktávu")), expect)

    def test_crossing(self):
        s = song(theory="strict")
        s.track("organ", name="alto").notes("C4:1 D4:1 E4:1 A4:1")
        s.track("organ", name="soprano").notes("G4:1 G4:1 G4:1 G4:1")
        self.assertTrue(found(theory_warnings(s), "křížení"))

    def test_off(self):
        s = song(theory="off").harmony("C | F")
        s.track("organ", name="bass").notes("C3:4 | D3:4")
        s.track("organ", name="pad").notes("(C2 E2):4 | (C2 E2):4")
        self.assertEqual(theory_warnings(s), [])


class TestScaleUsesPlan(unittest.TestCase):
    def test_chord_tone_outside_scale_is_fine(self):
        s = song(key="D", scale="minor").harmony("Dm | A")
        s.track("organ", name="lead").notes("D4:4 | C#4:2 E4:2")
        self.assertFalse(found(model_warnings(s), "mimo D minor"))

    def test_without_plan_still_warns(self):
        s = song(key="D", scale="minor")
        s.track("organ", name="lead").notes("D4:4 | C#4:2 E4:2")
        self.assertTrue(found(model_warnings(s), "mimo D minor"))


class TestCheckReport(unittest.TestCase):
    def _song(self):
        s = song(bars=4).harmony("C | F | G7 | C").section("A", bars=(0, 4))
        s.track("organ", name="bass").notes("C3:4 | D3:4 | G2:4 | C3:4")
        return s

    def test_sections(self):
        text = check_report(self._song(), name="x.py")
        self.assertIn("PAPRIKA CHECK  x.py", text)
        self.assertIn("HARMONIE A (0–3)", text)
        self.assertIn("I | IV | V7 | I", text)
        self.assertIn("AKORDY   I=C  IV=F  V7=G7", text)
        self.assertIn("V7 → I autentická", text)
        self.assertIn("nepatří do akordu", text)

    def test_bars_filter(self):
        text = check_report(self._song(), bars=(2, 4))
        self.assertNotIn("nepatří do akordu", text)

    def test_without_plan_hint(self):
        s = song()
        s.track("organ", name="lead").notes("C4:4")
        self.assertIn("bez plánu", check_report(s))

    def test_render_report_includes_theory(self):
        s = song(bars=2).harmony("C | F")
        s.track("organ", name="bass").notes("C3:4 | D3:4")
        out = render_song(s, None, report=False)
        warns = warnings_for(s, out.audio, out.tracks, out.raw_peak)
        self.assertTrue(found(warns, "nepatří do akordu"))


if __name__ == "__main__":
    unittest.main()
