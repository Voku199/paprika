import io
import os
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Song  # noqa: E402
from paprika.cli import main  # noqa: E402
from paprika.midi import (PPQ, MidiError, _program, export, read,  # noqa: E402
                          to_bytes)


def metas(track, kind):
    return [(t, d[1]) for t, k, d in track.events if k == "meta" and d[0] == kind]


def roundtrip(song):
    data, notes = to_bytes(song)
    return read(data), notes


class TestHeader(unittest.TestCase):
    def test_format_and_tracks(self):
        s = Song(bpm=120, bars=2)
        s.track("organ", name="a").notes("C4:4")
        s.track("organ", name="b").notes("E4:4")
        data, _ = to_bytes(s)
        self.assertEqual(data[:4], b"MThd")
        mid = read(data)
        self.assertEqual((mid.fmt, mid.ppq, len(mid.tracks)), (1, PPQ, 3))
        self.assertEqual([t.name for t in mid.tracks[1:]], ["a", "b"])

    def test_deterministic(self):
        s = Song(bpm=100, bars=1)
        s.track("organ").notes("C4:1 D4:1 E4:2").humanize(seed=4)
        self.assertEqual(to_bytes(s)[0], to_bytes(s)[0])


class TestNotes(unittest.TestCase):
    def test_round_trip(self):
        s = Song(bpm=90, bars=2)
        s.track("organ", name="lead").notes("C4:1 E4:0.5@90 G4:2.5 | (C4 E4):4@70")
        mid, _ = roundtrip(s)
        got = [(n.start, n.dur, n.pitch, n.velocity) for n in mid.tracks[1].notes]
        self.assertEqual(got, [(0, 480, 60, 100), (480, 240, 64, 90),
                               (720, 1200, 67, 100), (1920, 1920, 60, 70),
                               (1920, 1920, 64, 70)])
        self.assertEqual({n.channel for n in mid.tracks[1].notes}, {0})

    def test_same_pitch_overlap_is_shortened(self):
        s = Song(bars=1)
        t = s.track("organ", name="lead")
        t.note("C4", at=0, dur=2).note("C4", at=1, dur=1)
        mid, _ = roundtrip(s)
        got = [(n.start, n.dur) for n in mid.tracks[1].notes]
        self.assertEqual(got, [(0, 480), (480, 480)])

    def test_pedal_cc64(self):
        s = Song(bars=1)
        s.track("organ", name="keys").notes("C4:4").pedal(bars=(0, 1))
        mid, _ = roundtrip(s)
        cc = [(t, d) for t, k, d in mid.tracks[1].events if k == "cc"]
        self.assertEqual(cc, [(30, (0, 64, 127)), (1920, (0, 64, 0))])

    def test_channels_skip_drums(self):
        s = Song(bars=1)
        for i in range(11):
            s.track("organ", name=f"t{i}").notes("C4:4")
        mid, _ = roundtrip(s)
        channels = [t.notes[0].channel for t in mid.tracks[1:]]
        self.assertEqual(channels, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11])

    def test_program_change(self):
        s = Song(bars=1)
        s.track("epiano", name="ep").notes("C4:4")
        mid, _ = roundtrip(s)
        progs = [d for _, k, d in mid.tracks[1].events if k == "program"]
        self.assertEqual(progs, [(0, 4)])

    def test_program_map(self):
        def track(name):
            return SimpleNamespace(patch=SimpleNamespace(name=name))
        self.assertEqual(_program(track("violins")), 48)
        self.assertEqual(_program(track("violins.pizz")), 45)
        self.assertEqual(_program(track("celli.stac")), 48)
        self.assertIsNone(_program(track("muj_patch")))


class TestDrums(unittest.TestCase):
    def test_gm_channel_10(self):
        s = Song(bars=1)
        s.drums().pattern("k...s...")
        mid, _ = roundtrip(s)
        got = [(n.start, n.pitch, n.channel) for n in mid.tracks[1].notes]
        self.assertEqual(got, [(0, 36, 9), (480, 38, 9)])

    def test_sfx_is_skipped_with_note(self):
        s = Song(bars=2)
        s.track("organ", name="lead").notes("C4:4")
        s.sfx().impact(bar=1)
        mid, notes = roundtrip(s)
        self.assertEqual(len(mid.tracks), 2)
        self.assertTrue(any("SFX" in n for n in notes), notes)


class TestConductor(unittest.TestCase):
    def test_tempo_time_and_key(self):
        s = Song(bpm=90, key="D", scale="minor", bars=1)
        mid, _ = roundtrip(s)
        c = mid.tracks[0]
        (_, tempo), = metas(c, 0x51)
        self.assertEqual(int.from_bytes(tempo, "big"), 666667)
        (_, ts), = metas(c, 0x58)
        self.assertEqual(ts[:2], bytes([4, 2]))
        (_, ks), = metas(c, 0x59)
        self.assertEqual(struct.unpack(">bB", ks), (-1, 1))

    def test_odd_meters(self):
        for bpb, expect in ((3, (3, 2)), (3.5, (7, 3))):
            mid, _ = roundtrip(Song(bars=1, beats_per_bar=bpb))
            (_, ts), = metas(mid.tracks[0], 0x58)
            self.assertEqual(tuple(ts[:2]), expect)

    def test_sections_and_chords(self):
        s = Song(key="D", scale="minor", bars=4).harmony("Dm | A | i | V7")
        s.section("A", bars=(0, 2)).section("B", bars=(2, 4))
        mid, _ = roundtrip(s)
        c = mid.tracks[0]
        self.assertEqual([(t, d.decode()) for t, d in metas(c, 0x06)],
                         [(0, "A"), (3840, "B")])
        self.assertEqual([(t, d.decode()) for t, d in metas(c, 0x01)],
                         [(0, "Dm (i)"), (1920, "A (V)"), (3840, "Dm (i)"),
                          (5760, "A7 (V7)")])

    def test_automation_note(self):
        s = Song(bars=2)
        s.track("organ", name="pad").notes("C4:4").automate("vol", [(0, 0), (1, 1)])
        _, notes = roundtrip(s)
        self.assertTrue(any("automatizace" in n and "pad" in n for n in notes), notes)


class TestReader(unittest.TestCase):
    def test_running_status(self):
        body = bytes([0x00, 0x90, 60, 100,     # note on
                      0x60, 64, 100,           # running status: další note on
                      0x60, 60, 0,             # note on vel 0 = note off
                      0x00, 64, 0,
                      0x00, 0xFF, 0x2F, 0x00])
        data = (b"MThd" + struct.pack(">IHHH", 6, 0, 1, 96)
                + b"MTrk" + struct.pack(">I", len(body)) + body)
        mid = read(data)
        got = [(n.start, n.dur, n.pitch) for n in mid.tracks[0].notes]
        self.assertEqual(got, [(0, 192, 60), (96, 96, 64)])

    def test_not_midi(self):
        with self.assertRaises(MidiError):
            read(b"RIFF....WAVE")

    def test_export_file(self):
        s = Song(bars=1)
        s.track("organ", name="lead").notes("C4:4")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "x.mid")
            export(s, path)
            mid = read(path)
        self.assertEqual(mid.tracks[1].notes[0].pitch, 60)


class TestMidiCli(unittest.TestCase):
    def test_default_path_next_to_wav(self):
        src = ("from paprika import Song\n"
               "s = Song(bars=1)\n"
               "s.track('organ', name='lead').notes('C4:4')\n"
               "s.sfx().impact(0)\n")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "songs", "x.py")
            os.makedirs(os.path.dirname(path))
            with open(path, "w", encoding="utf-8") as f:
                f.write(src)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(["midi", path])
            out = os.path.join(d, "output", "x.mid")
            self.assertEqual(code, 0)
            self.assertTrue(os.path.isfile(out))
            self.assertIn("SFX", buf.getvalue())
            self.assertEqual(read(out).tracks[1].notes[0].pitch, 60)


if __name__ == "__main__":
    unittest.main()
