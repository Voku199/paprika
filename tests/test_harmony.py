import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika.harmony import (HarmonyError, cadence, key_signature,  # noqa: E402
                             parse_chord, parse_harmony, pc_name, roman_of)


def label(tok, key="C", scale="major"):
    return parse_chord(tok, key, scale).label


class TestNamedChords(unittest.TestCase):
    def test_qualities_and_bass(self):
        c = parse_chord("Dm/F")
        self.assertEqual((c.root, c.quality, c.bass), (2, "m", 5))
        self.assertEqual(c.pcs, frozenset({2, 5, 9}))
        self.assertEqual(c.label, "Dm/F")
        self.assertEqual(parse_chord("Bbmaj7").pcs, frozenset({10, 2, 5, 9}))

    def test_slash_bass_outside_chord_is_part_of_sound(self):
        self.assertIn(2, parse_chord("C/D").pcs)

    def test_third(self):
        self.assertEqual(parse_chord("A").third_pc, 1)
        self.assertEqual(parse_chord("Am").third_pc, 0)
        self.assertFalse(parse_chord("Asus4").has_third)
        self.assertFalse(parse_chord("A5").has_third)

    def test_unknown_quality_lists_known(self):
        with self.assertRaises(HarmonyError) as cm:
            parse_chord("Cblah")
        self.assertIn("maj7", str(cm.exception))


class TestRomanParse(unittest.TestCase):
    def test_major_diatonic(self):
        got = [label(t) for t in ("I", "ii", "iii", "IV", "V", "vi", "viio")]
        self.assertEqual(got, ["C", "Dm", "Em", "F", "G", "Am", "Bdim"])

    def test_minor_uses_natural_minor_degrees(self):
        got = [label(t, "D", "minor") for t in ("i", "iio", "III", "iv", "v", "VI", "VII")]
        self.assertEqual(got, ["Dm", "Edim", "F", "Gm", "Am", "Bb", "C"])

    def test_minor_dominant_and_leading_tone(self):
        self.assertEqual(label("V", "D", "minor"), "A")
        self.assertEqual(label("V7", "D", "minor"), "A7")
        self.assertEqual(label("viio7", "D", "minor"), "C#dim7")
        self.assertEqual(label("viiø7", "A", "minor"), "G#m7b5")

    def test_diatonic_sevenths(self):
        got = [label(t) for t in ("I7", "ii7", "IV7", "V7", "viiø7")]
        self.assertEqual(got, ["Cmaj7", "Dm7", "Fmaj7", "G7", "Bm7b5"])
        self.assertEqual(label("bVII7"), "Bb7")          # chromatický základ

    def test_inversions(self):
        self.assertEqual(label("i6", "D", "minor"), "Dm/F")
        self.assertEqual(label("I64"), "C/G")
        self.assertEqual(label("V65", "D", "minor"), "A7/C#")
        self.assertEqual(label("V42"), "G7/F")
        self.assertEqual(label("Iadd6"), "C6")            # přidaná sexta, ne obrat

    def test_applied_chords(self):
        self.assertEqual(label("V/V"), "D")
        self.assertEqual(label("V7/IV"), "C7")
        self.assertEqual(label("V7/iv", "D", "minor"), "D7")
        self.assertEqual(label("viio7/V"), "F#dim7")
        self.assertEqual(label("viio6/ii"), "C#dim/E")

    def test_mode_degrees(self):
        self.assertEqual(label("IV", "D", "dorian"), "G")
        self.assertEqual(label("bVII", "C", "major"), "Bb")

    def test_bad_roman(self):
        for tok in ("X", "Vx7", "V/", "iii13"):
            with self.assertRaises(HarmonyError, msg=tok):
                parse_chord(tok)


class TestRomanOf(unittest.TestCase):
    def test_round_trip_diatonic(self):
        for key, scale, toks in (
                ("C", "major", ["I", "ii", "iii", "IV", "V", "vi", "viio", "V7",
                                "ii7", "Imaj7", "I6", "V65", "IV64"]),
                ("D", "minor", ["i", "iio", "III", "iv", "V", "VI", "VII",
                                "viio7", "V7", "i6"])):
            for tok in toks:
                self.assertEqual(roman_of(parse_chord(tok, key, scale), key, scale),
                                 tok, (key, scale, tok))

    def test_named_chords_get_numerals(self):
        self.assertEqual(roman_of(parse_chord("A"), "D", "minor"), "V")
        self.assertEqual(roman_of(parse_chord("Bb"), "D", "minor"), "VI")
        self.assertEqual(roman_of(parse_chord("Dm/F"), "D", "minor"), "i6")

    def test_applied_and_borrowed(self):
        self.assertEqual(roman_of(parse_chord("D")), "V/V")
        self.assertEqual(roman_of(parse_chord("E"), "D", "minor"), "V/V")
        self.assertEqual(roman_of(parse_chord("C7")), "V7/IV")
        self.assertEqual(roman_of(parse_chord("F#dim7")), "viio7/V")
        self.assertEqual(roman_of(parse_chord("Bb")), "bVII")
        self.assertEqual(roman_of(parse_chord("Ab")), "bVI")
        self.assertEqual(roman_of(parse_chord("Eb"), "D", "minor"), "bII")


class TestParseHarmony(unittest.TestCase):
    def spans(self, text, **kw):
        return [(s.start_beat, s.dur_beats, s.chord.label if s.chord else None)
                for s in parse_harmony(text, **kw)]

    def test_bars_and_durations(self):
        got = self.spans("Dm | Bb | Gm:2 A:2 | Gm A", key="D", scale="minor")
        self.assertEqual(got, [(0, 4, "Dm"), (4, 4, "Bb"), (8, 2, "Gm"),
                               (10, 2, "A"), (12, 2, "Gm"), (14, 2, "A")])

    def test_mixed_explicit_and_shared(self):
        got = self.spans("| C:2 F G |")
        self.assertEqual(got, [(0, 2, "C"), (2, 1, "F"), (3, 1, "G")])

    def test_without_bars_one_chord_per_bar(self):
        got = self.spans("i VI iv V:2 V7:2", key="D", scale="minor", beats_per_bar=3)
        self.assertEqual(got, [(0, 3, "Dm"), (3, 3, "Bb"), (6, 3, "Gm"),
                               (9, 2, "A"), (11, 2, "A7")])

    def test_rest(self):
        got = self.spans("_ | C")
        self.assertEqual(got, [(0, 4, None), (4, 4, "C")])

    def test_mixed_notations(self):
        got = self.spans("i | Bb | V7/iv | Gm", key="D", scale="minor")
        self.assertEqual([g[2] for g in got], ["Dm", "Bb", "D7", "Gm"])

    def test_wrong_bar_length_names_bar(self):
        with self.assertRaises(HarmonyError) as cm:
            parse_harmony("C | F:2 G:1 | C", bar_offset=8)
        self.assertIn("takt 10", str(cm.exception))

    def test_overfull_bar(self):
        with self.assertRaises(HarmonyError):
            parse_harmony("| C:4 F |")

    def test_unknown_chord_names_bar(self):
        with self.assertRaises(HarmonyError) as cm:
            parse_harmony("C | Fblah")
        self.assertIn("takt 2", str(cm.exception))

    def test_empty_bar(self):
        with self.assertRaises(HarmonyError):
            parse_harmony("C | | G")


class TestKeyAndSpelling(unittest.TestCase):
    def test_key_signature(self):
        self.assertEqual(key_signature("C", "major"), (0, 0))
        self.assertEqual(key_signature("F", "major"), (-1, 0))
        self.assertEqual(key_signature("A", "major"), (3, 0))
        self.assertEqual(key_signature("D", "minor"), (-1, 1))
        self.assertEqual(key_signature("Eb", "minor"), (-6, 1))
        self.assertEqual(key_signature("F#", "major"), (6, 0))
        self.assertEqual(key_signature("D", "dorian"), (0, 0))

    def test_pc_name(self):
        self.assertEqual(pc_name(10, "F", "major"), "Bb")
        self.assertEqual(pc_name(10, "E", "major"), "A#")
        self.assertEqual(pc_name(1, "D", "minor"), "C#")     # citlivý tón
        self.assertEqual(pc_name(11, "Gb", "major"), "Cb")
        self.assertEqual(pc_name(2, "Gb", "major"), "D")     # ne Ebb


class TestCadence(unittest.TestCase):
    def test_kinds(self):
        C = parse_chord
        self.assertEqual(cadence(C("G7"), C("C")), "autentická")
        self.assertEqual(cadence(C("A7"), C("Dm"), "D", "minor"), "autentická")
        self.assertEqual(cadence(C("F"), C("C")), "plagální")
        self.assertEqual(cadence(C("Dm"), C("G")), "půlová")
        self.assertEqual(cadence(C("G"), C("Am")), "klamná")
        self.assertIsNone(cadence(C("Dm"), C("F")))


if __name__ == "__main__":
    unittest.main()
