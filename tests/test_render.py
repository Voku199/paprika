import os
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Reverb, Song  # noqa: E402
from paprika.render import limiter, render_song, write_wav  # noqa: E402


def demo_song(bars=4, bpm=120):
    s = Song(bpm=bpm, key="C", scale="major", bars=bars)
    lead = s.track("pluck", vol=0.7, name="lead")
    lead.notes(" | ".join(["C4:1 E4:1 G4:1 E4:1"] * bars), bar=0)
    bass = s.track("subbass", vol=0.6, name="bass")
    bass.notes(" | ".join(["C2:4"] * bars), bar=0)
    dr = s.drums(vol=0.6)
    dr.pattern("k...s...k...s...", bars=(0, bars))
    return s


class TestRender(unittest.TestCase):
    def setUp(self):
        self.song = demo_song()
        self.result = render_song(self.song, None, report=False)

    def test_no_nan_or_inf(self):
        self.assertTrue(np.all(np.isfinite(self.result.audio)))

    def test_peak_within_range(self):
        self.assertLessEqual(float(np.abs(self.result.audio).max()), 1.0)

    def test_is_stereo(self):
        self.assertEqual(self.result.audio.ndim, 2)
        self.assertEqual(self.result.audio.shape[1], 2)

    def test_length_matches_bars_and_bpm(self):
        # 4 takty při 120 BPM = 8 s, plus ocásek efektů a release
        self.assertGreaterEqual(self.result.seconds, 8.0)
        self.assertLess(self.result.seconds, 8.0 + 3.0)

    def test_bpm_halves_length(self):
        slow = render_song(demo_song(bpm=60), None, report=False)
        self.assertGreater(slow.seconds, self.result.seconds * 1.7)

    def test_track_buffers_present(self):
        self.assertEqual(set(self.result.tracks), {"lead", "bass", "drums"})
        for buf in self.result.tracks.values():
            self.assertTrue(np.any(buf))

    def test_bars_slice_is_shorter(self):
        part = render_song(demo_song(bars=8), None, report=False, bars=(0, 2))
        self.assertLess(part.seconds, 8.0)

    def test_empty_song_does_not_crash(self):
        out = render_song(Song(bars=2), None, report=False)
        self.assertTrue(np.all(np.isfinite(out.audio)))

    def test_pan_moves_energy(self):
        s = Song(bpm=120, bars=1)
        t = s.track("organ", vol=0.8, pan=-1.0)
        t.note("C4", at=0, dur=1)
        out = render_song(s, None, report=False)
        left = float(np.abs(out.audio[:, 0]).sum())
        right = float(np.abs(out.audio[:, 1]).sum())
        self.assertGreater(left, right * 10)

    def test_reverb_tail_fits_in_buffer(self):
        s = demo_song(bars=2)
        s.tracks[0].fx(Reverb(room=0.9, mix=0.5))
        out = render_song(s, None, report=False)
        self.assertTrue(np.all(np.isfinite(out.audio)))
        self.assertLess(float(np.abs(out.audio[-100:]).max()), 0.02)


class TestLimiter(unittest.TestCase):
    def test_tames_overs(self):
        loud = np.linspace(-3.0, 3.0, 1000).astype(np.float32)
        out = limiter(loud)
        self.assertLess(float(np.abs(out).max()), 1.0)

    def test_leaves_quiet_signal_alone(self):
        quiet = (np.linspace(-0.5, 0.5, 100)).astype(np.float32)
        np.testing.assert_allclose(limiter(quiet), quiet)


class TestWav(unittest.TestCase):
    def test_writes_readable_16bit_stereo(self):
        result = render_song(demo_song(bars=2), None, report=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "out.wav")
            write_wav(path, result.audio, 44100)
            self.assertTrue(os.path.isfile(path))
            with wave.open(path, "rb") as w:
                self.assertEqual(w.getnchannels(), 2)
                self.assertEqual(w.getsampwidth(), 2)
                self.assertEqual(w.getframerate(), 44100)
                self.assertEqual(w.getnframes(), result.audio.shape[0])


if __name__ == "__main__":
    unittest.main()
