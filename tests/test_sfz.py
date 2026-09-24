import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika.sfz import SfzError, load, note_number, parse  # noqa: E402


def norm(p):
    return os.path.normpath(p)


class TestNoteNumber(unittest.TestCase):
    def test_numbers_and_names(self):
        self.assertEqual(note_number("60"), 60)
        self.assertEqual(note_number("c4"), 60)
        self.assertEqual(note_number("C4"), 60)
        self.assertEqual(note_number("f#3"), 54)
        self.assertEqual(note_number("eb2"), 39)
        self.assertEqual(note_number("bb3"), 58)
        self.assertEqual(note_number("b3"), 59)
        self.assertEqual(note_number("a0"), 21)
        self.assertEqual(note_number("c-1"), 0)

    def test_bad(self):
        with self.assertRaises(SfzError):
            note_number("h4")


class TestParse(unittest.TestCase):
    def test_header_inheritance(self):
        text = """
<global> volume=-6 ampeg_release=0.5
<group> lovel=1 hivel=64
<region> sample=a.wav key=60
<region> sample=b.wav key=62 volume=-3
<group> lovel=65
<region> sample=c.wav lokey=c4 hikey=d4 pitch_keycenter=c#4
"""
        a, b, c = parse(text, "/lib")
        self.assertEqual((a.lokey, a.hikey, a.keycenter), (60, 60, 60))
        self.assertEqual(a.volume, -6.0)
        self.assertEqual(a.release, 0.5)
        self.assertEqual((a.lovel, a.hivel), (1, 64))
        self.assertEqual(b.volume, -3.0)
        self.assertEqual((c.lovel, c.hivel), (65, 127))
        self.assertEqual((c.lokey, c.hikey, c.keycenter), (60, 62, 61))
        self.assertEqual(norm(a.sample), norm("/lib/a.wav"))

    def test_global_resets_group(self):
        rs = parse("<group> volume=-10\n<global>\n<region> sample=x.wav", "/l")
        self.assertEqual(rs[0].volume, 0.0)

    def test_comments_define_spaces_default_path(self):
        text = """
// komentář <region> sample=nic.wav
/* blok
<region> sample=taky_ne.wav */
#define $ROOT 57
<control> default_path=../samples/
<region> sample=Solo Violin/Arco Vib/x y.wav pitch_keycenter=$ROOT lokey=55 hikey=$ROOT
"""
        rs = parse(text, "/lib/sfz")
        self.assertEqual(len(rs), 1)
        self.assertEqual(rs[0].keycenter, 57)
        self.assertEqual(rs[0].hikey, 57)
        self.assertEqual(norm(rs[0].sample),
                         norm("/lib/samples/Solo Violin/Arco Vib/x y.wav"))

    def test_backslash_paths_and_options(self):
        r = parse(r"<region> sample=44.1khz16bit\A0v1.wav lokey=21 hikey=22 "
                  r"lovel=1 hivel=26 pitch_keycenter=21 tune=10 amp_veltrack=73 "
                  r"loop_mode=one_shot trigger=release offset=100 seq_length=2 "
                  r"seq_position=2 pan=-50 transpose=-12", "/p")[0]
        self.assertTrue(r.sample.replace("\\", "/").endswith("44.1khz16bit/A0v1.wav"))
        self.assertEqual(r.tune, 10.0)
        self.assertEqual(r.veltrack, 73.0)
        self.assertEqual(r.loop_mode, "one_shot")
        self.assertEqual(r.trigger, "release")
        self.assertEqual((r.offset, r.seq_length, r.seq_position, r.pan, r.transpose),
                         (100, 2, 2, -50.0, -12))

    def test_unknown_opcodes_are_listed(self):
        r = parse("<region> sample=a.wav cutoff=500 fil_type=lpf_2p", "/")[0]
        self.assertEqual(r.ignored, ("cutoff", "fil_type"))

    def test_generators_and_other_headers_skipped(self):
        rs = parse("<region> sample=*sine\n<curve> v000=0\n<region> sample=a.wav", "/")
        self.assertEqual(len(rs), 1)

    def test_bad_value(self):
        with self.assertRaises(SfzError) as cm:
            parse("<region> sample=a.wav lokey=xyz", "/", source="x.sfz")
        self.assertIn("lokey", str(cm.exception))
        self.assertIn("x.sfz", str(cm.exception))

    def test_load_include_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "inc.sfz"), "w") as f:
                f.write("<region> sample=b.wav key=70\n")
            with open(os.path.join(d, "main.sfz"), "w") as f:
                f.write('<region> sample=a.wav key=60\n#include "inc.sfz"\n')
            rs = load(os.path.join(d, "main.sfz"))
            self.assertEqual([r.keycenter for r in rs], [60, 70])
            with self.assertRaises(SfzError):
                load(os.path.join(d, "nope.sfz"))


if __name__ == "__main__":
    unittest.main()
