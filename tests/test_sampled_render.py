import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from paprika import Song  # noqa: E402
from paprika import library  # noqa: E402
from paprika.fx import FxContext  # noqa: E402
from paprika.install import write_sfz  # noqa: E402
from paprika.note import Note  # noqa: E402
from paprika.render import (_pedaled, _phrases, _sampled_events,  # noqa: E402
                            _tail_seconds, render_song)
from paprika.sampler import Sampler, clear_cache  # noqa: E402
from paprika.song import PEDAL_LIFT  # noqa: E402
from sample_fixture import make_sfz_library, make_vsco_violins, midi_freq  # noqa: E402


class TestPedal(unittest.TestCase):
    def test_pedal_segments(self):
        t = Song(bpm=120, bars=4).track("epiano")
        self.assertIs(t.pedal(bars=(0, 2), every=1), t)
        self.assertEqual(t.pedals, [(PEDAL_LIFT, 4.0), (4.0 + PEDAL_LIFT, 8.0)])
        with self.assertRaises(ValueError):
            t.pedal(bars=(2, 2))

    def test_pedal_extends_notes(self):
        notes = [Note(60, 0, 1, 100), Note(64, 1, 1, 100),
                 Note(67, 3.5, 0.99, 100), Note(62, 4.0, 0.5, 100)]
        out = _pedaled(notes, [(PEDAL_LIFT, 4.0), (4.0 + PEDAL_LIFT, 8.0)])
        self.assertEqual([n.end_beat for n in out], [4.0, 4.0, 8.0, 8.0])
        self.assertEqual(notes[0].end_beat, 1)          # vstup se nemění

    def test_note_ending_at_pedal_change_is_not_caught(self):
        out = _pedaled([Note(60, 0, 4, 100)], [(PEDAL_LIFT, 4.0), (4.0 + PEDAL_LIFT, 8.0)])
        self.assertEqual(out[0].end_beat, 4.0)

    def test_restrike_cuts(self):
        out = _pedaled([Note(60, 0, 1, 100), Note(60, 2, 1, 100)], [(0.0, 8.0)])
        self.assertEqual([n.end_beat for n in out], [2.0, 8.0])

    def test_pedal_on_synth_track(self):
        t = Song(bpm=120, bars=2).track("epiano")
        t.note("C4", at=0, dur=0.5)
        t.pedal(bars=(0, 1))
        self.assertEqual(_phrases(t)[0][1], 4.0)


class TestSampledTrack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.sfz = make_sfz_library(cls.tmp.name, sr=44100)

    @classmethod
    def tearDownClass(cls):
        clear_cache()
        cls.tmp.cleanup()

    def sampler(self, **kw):
        return Sampler(self.sfz, name="test", **kw)

    def test_render_sampled_track(self):
        s = Song(bpm=120, bars=2)
        t = s.track(self.sampler(legato=True))
        t.notes("C4:1 E4:1 G#4:2", bar=0)
        res = render_song(s, None, report=False)
        buf = res.tracks["test"]
        self.assertGreater(np.abs(buf).max(), 0.05)
        self.assertGreater(np.abs(buf[int(1.5 * 44100):int(1.7 * 44100)]).max(), 0.01)

    def test_legato_events(self):
        t = Song(bpm=120, bars=2).track(self.sampler(legato=True), mono=True)
        t.notes("C4:1 E4:1 G#4:2", bar=0)
        ev = _sampled_events(t)
        self.assertEqual([e[4] for e in ev], [False, True, True])
        self.assertEqual([e[5] for e in ev], [True, True, False])
        self.assertEqual([e[1] for e in ev], [1.0, 1.0, 2.0])

    def test_no_legato_when_poly_or_one_shot(self):
        poly = Song(bars=1).track(self.sampler(legato=True))
        poly.notes("C4:1 E4:1 G#4:2")
        self.assertFalse(any(e[4] or e[5] for e in _sampled_events(poly)))
        shot = Song(bars=1).track(self.sampler(legato=False), mono=True)
        shot.notes("C4:1 E4:1 G#4:2")
        self.assertFalse(any(e[4] or e[5] for e in _sampled_events(shot)))

    def test_glide_mark_is_legato(self):
        t = Song(bars=1).track(self.sampler(legato=True))
        t.notes("C4:2 >E4:2")
        self.assertEqual([e[4] for e in _sampled_events(t)], [False, True])

    def test_tail_uses_sampler(self):
        s = Song(bpm=120, bars=1)
        s.track(self.sampler()).note("C4", 0, 1)
        ctx = FxContext(sr=44100, bpm=120, beats_per_bar=4)
        self.assertGreaterEqual(_tail_seconds(s, ctx), 0.2 + 0.3)

    def test_bars_slice(self):
        s = Song(bpm=120, bars=4)
        t = s.track(self.sampler())
        t.notes("C4:4 | C4:4 | C4:4 | C4:4", bar=0)
        a = render_song(s, None, report=False, bars=(2, 4)).tracks["test"]
        b = render_song(s, None, report=False, bars=(2, 4)).tracks["test"]
        self.assertTrue(np.array_equal(a, b))
        self.assertGreater(np.abs(a[:4410]).max(), 0.05)

    def test_song_48k(self):
        s = Song(bpm=120, bars=1, sample_rate=48000)
        s.track(self.sampler()).note("E4", 0, 2)
        buf = render_song(s, None, report=False).tracks["test"].mean(axis=1)
        size = 1 << 15
        spec = np.abs(np.fft.rfft(buf[:size] * np.hanning(size)))
        f = np.argmax(spec) * 48000 / size
        self.assertAlmostEqual(f / midi_freq(64), 1.0, delta=0.01)


class TestSongTrackByName(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.env = mock.patch.dict(os.environ, {"PAPRIKA_SAMPLES": self.tmp.name})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        clear_cache()
        self.tmp.cleanup()

    def test_missing_library(self):
        with self.assertRaises(library.SamplesMissing):
            Song().track("grand_piano")

    def test_track_by_name(self):
        make_vsco_violins(self.tmp.name)
        inst = library.INSTRUMENTS["violins"]
        for art in inst.arts:
            write_sfz(inst, art, self.tmp.name)
        s = Song(bars=1)
        t = s.track("violins.pizz", fx="default")
        self.assertEqual(t.name, "violins_pizz")
        self.assertIsInstance(t.patch, Sampler)
        self.assertTrue(t.fx_chain)
        self.assertFalse(t.mono)
        t.notes("C4:1 E4:1 C4:2")
        res = render_song(s, None, report=False)
        self.assertGreater(np.abs(res.tracks["violins_pizz"]).max(), 0.01)


if __name__ == "__main__":
    unittest.main()
