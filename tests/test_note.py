import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika.note import NoteLengthError, NoteParseError, parse  # noqa: E402


class TestSingleNote(unittest.TestCase):
    def test_pitch_and_duration(self):
        n = parse("C#4:1.5")[0]
        self.assertEqual(n.pitch, 61)
        self.assertEqual(n.dur_beats, 1.5)
        self.assertEqual(n.start_beat, 0.0)

    def test_default_duration(self):
        n = parse("C4")[0]
        self.assertEqual(n.dur_beats, 1.0)

    def test_velocity_suffix(self):
        n = parse("C4:1@110")[0]
        self.assertEqual(n.velocity, 110)


class TestSequence(unittest.TestCase):
    def test_starts_accumulate(self):
        notes = parse("E5:1 G5:1 A5:2")
        self.assertEqual([n.start_beat for n in notes], [0.0, 1.0, 2.0])
        self.assertEqual([n.pitch for n in notes], [76, 79, 81])

    def test_rest(self):
        notes = parse("_:2 C4:2")
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].start_beat, 2.0)

    def test_tie_extends_previous(self):
        notes = parse("D5:3 ~:1")
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].dur_beats, 4.0)

    def test_tie_without_note(self):
        with self.assertRaises(NoteParseError):
            parse("~:1")

    def test_chord_in_parens(self):
        notes = parse("(C4 E4 G4):2")
        self.assertEqual([n.pitch for n in notes], [60, 64, 67])
        self.assertEqual({n.start_beat for n in notes}, {0.0})
        self.assertEqual({n.dur_beats for n in notes}, {2.0})

    def test_chord_then_note(self):
        notes = parse("(C4 E4):2 G4:2")
        self.assertEqual(notes[-1].start_beat, 2.0)

    def test_unknown_token(self):
        with self.assertRaises(NoteParseError):
            parse("Q4:1")


class TestBarChecking(unittest.TestCase):
    def test_correct_bars_pass(self):
        notes = parse("E5:1 G5:1 A5:2 | G5:1 E5:1 D5:2 | C5:4")
        self.assertEqual(len(notes), 7)
        self.assertEqual(notes[-1].start_beat, 8.0)

    def test_short_bar_raises(self):
        with self.assertRaises(NoteLengthError):
            parse("C4:1 D4:1 | E4:4")

    def test_long_bar_raises(self):
        with self.assertRaises(NoteLengthError):
            parse("C4:5 | E4:4")

    def test_error_names_the_bar(self):
        with self.assertRaises(NoteLengthError) as cm:
            parse("C4:4 | D4:2 | E4:4")
        self.assertIn("takt 2", str(cm.exception))

    def test_bar_offset_in_message(self):
        with self.assertRaises(NoteLengthError) as cm:
            parse("C4:2 | D4:4", bar_offset=8)
        self.assertIn("takt 9", str(cm.exception))

    def test_no_pipes_no_check(self):
        parse("C4:1 D4:1 E4:1")  # nesmí spadnout

    def test_other_meter(self):
        parse("C4:3 | D4:3", beats_per_bar=3)
        with self.assertRaises(NoteLengthError):
            parse("C4:4 | D4:3", beats_per_bar=3)

    def test_leading_and_trailing_pipe(self):
        notes = parse("|C4:4|")
        self.assertEqual(len(notes), 1)


if __name__ == "__main__":
    unittest.main()
