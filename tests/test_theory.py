import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import theory  # noqa: E402


class TestPitch(unittest.TestCase):
    def test_middle_c(self):
        self.assertEqual(theory.pitch("C4"), 60)

    def test_accidentals(self):
        self.assertEqual(theory.pitch("C#4"), 61)
        self.assertEqual(theory.pitch("Db4"), 61)
        self.assertEqual(theory.pitch("A0"), 21)

    def test_enharmonics_across_octave(self):
        # oktáva patří písmenu, ne výsledné třídě tónu
        self.assertEqual(theory.pitch("B#3"), 60)
        self.assertEqual(theory.pitch("Cb4"), 59)
        self.assertEqual(theory.pitch("E#4"), 65)
        self.assertEqual(theory.pitch("Fb4"), 64)

    def test_roundtrip(self):
        for p in range(21, 108):
            self.assertEqual(theory.pitch(theory.note_name(p)), p)

    def test_bad_name(self):
        with self.assertRaises(theory.TheoryError):
            theory.pitch("H4")
        with self.assertRaises(theory.TheoryError):
            theory.pitch("C")


class TestChords(unittest.TestCase):
    def test_cmaj7(self):
        self.assertEqual(theory.chord("Cmaj7"), [60, 64, 67, 71])

    def test_triads(self):
        self.assertEqual(theory.chord("C"), [60, 64, 67])
        self.assertEqual(theory.chord("Am"), [69, 72, 76])
        self.assertEqual(theory.chord("G7"), [67, 71, 74, 77])

    def test_octave_in_name(self):
        self.assertEqual(theory.chord("F3maj7"), [53, 57, 60, 64])
        self.assertEqual(theory.chord("Fmaj7", octave=3), [53, 57, 60, 64])

    def test_inversions(self):
        self.assertEqual(theory.chord("C", inv=1), [64, 67, 72])
        self.assertEqual(theory.chord("C", inv=2), [67, 72, 76])
        self.assertEqual(theory.invert([60, 64, 67], -1), [55, 60, 64])

    def test_oct_shift(self):
        self.assertEqual(theory.chord("C", oct=-1), [48, 52, 55])

    def test_unknown(self):
        with self.assertRaises(theory.TheoryError):
            theory.chord("Cnope")


class TestScales(unittest.TestCase):
    def test_major(self):
        self.assertEqual(theory.scale("C", "major"),
                         [60, 62, 64, 65, 67, 69, 71, 72])

    def test_minor_pcs(self):
        self.assertEqual(theory.scale_pcs("A", "minor"),
                         [0, 2, 4, 5, 7, 9, 11])

    def test_in_scale(self):
        self.assertTrue(theory.in_scale(62, "C", "major"))
        self.assertFalse(theory.in_scale(61, "C", "major"))

    def test_unknown_scale(self):
        with self.assertRaises(theory.TheoryError):
            theory.scale("C", "hyperlydian")


class TestArp(unittest.TestCase):
    def test_up(self):
        self.assertEqual(theory.arp([60, 64, 67], "up"), [60, 64, 67])

    def test_down(self):
        self.assertEqual(theory.arp([60, 64, 67], "down"), [67, 64, 60])

    def test_updown_no_repeat_at_turn(self):
        self.assertEqual(theory.arp([60, 64, 67], "updown"), [60, 64, 67, 64])

    def test_octaves_and_steps(self):
        seq = theory.arp([60, 64, 67], "up", steps=8, octaves=2)
        self.assertEqual(len(seq), 8)
        self.assertEqual(seq[:6], [60, 64, 67, 72, 76, 79])
        self.assertEqual(seq[6], 60)  # cyklí dokola

    def test_random_is_seeded(self):
        a = theory.arp([60, 64, 67], "random", steps=6, seed=1)
        b = theory.arp([60, 64, 67], "random", steps=6, seed=1)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 6)

    def test_unknown_pattern(self):
        with self.assertRaises(theory.TheoryError):
            theory.arp([60], "zigzag")


if __name__ == "__main__":
    unittest.main()
