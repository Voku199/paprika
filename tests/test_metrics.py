import io
import os
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Song  # noqa: E402
from paprika import metrics  # noqa: E402
from paprika.analyze import warnings_for  # noqa: E402
from paprika.cli import main  # noqa: E402
from paprika.render import render_song, write_wav  # noqa: E402

SR = 44100


def sine(f, sec=2.0, amp=0.5):
    return (amp * np.sin(2 * np.pi * f * np.arange(int(sec * SR)) / SR)).astype(np.float32)


def stereo(x):
    return np.stack([x, x], axis=1)


class TestMetrics(unittest.TestCase):
    def test_bands_sum_and_sub(self):
        shares = metrics.band_shares(stereo(sine(40)), SR)
        self.assertEqual([b[0] for b in metrics.BANDS7], list(shares))
        self.assertAlmostEqual(sum(shares.values()), 100.0, places=3)
        self.assertGreater(shares["sub"], 90)

    def test_mid_tone(self):
        self.assertGreater(metrics.band_shares(stereo(sine(1000)), SR)["mid"], 90)

    def test_width(self):
        rng = np.random.default_rng(0)
        indep = rng.normal(0, 0.2, (SR, 2)).astype(np.float32)
        self.assertTrue(40 < metrics.stereo_width(indep) < 60)
        self.assertAlmostEqual(metrics.stereo_width(stereo(sine(300))), 0.0, places=6)

    def test_correlation_and_mono(self):
        self.assertAlmostEqual(metrics.correlation(stereo(sine(300))), 1.0, places=5)
        self.assertIsNone(metrics.correlation(sine(300)))
        self.assertEqual(metrics.stereo_width(sine(300)), 0.0)

    def test_summary_keys_and_short_input(self):
        s = metrics.summary(stereo(sine(300, sec=0.05)), SR)
        for k in ("bands", "mids", "width", "corr", "centroid", "crest", "lra"):
            self.assertIn(k, s)

    def test_silence_does_not_crash(self):
        s = metrics.summary(np.zeros((SR, 2), np.float32), SR)
        self.assertEqual(s["width"], 0.0)


class TestMixRules(unittest.TestCase):
    def texts(self, song, audio):
        return [w.text for w in warnings_for(song, audio, {}, 0.5)]

    def test_sub_heavy(self):
        s = Song(bars=16)
        self.assertTrue(any("sub" in t for t in self.texts(s, stereo(sine(40)))))
        self.assertFalse(any("sub přetížený" in t for t in self.texts(s, stereo(sine(440)))))

    def test_narrow_only_for_longer_songs(self):
        mono_mix = stereo(sine(440))
        self.assertTrue(any("úzké stereo" in t for t in self.texts(Song(bars=16), mono_mix)))
        self.assertFalse(any("úzké stereo" in t for t in self.texts(Song(bars=4), mono_mix)))

    def test_thin_mids(self):
        s = Song(bars=16)
        bass_only = stereo(sine(80) + sine(9000, amp=0.1))
        self.assertTrue(any("tenké středy" in t for t in self.texts(s, bass_only)))

    def test_report_has_seven_bands(self):
        s = Song(bars=1)
        s.track("organ").notes("C4:4")
        rep = render_song(s, None, print_report=False).report
        for name in ("sub", "lowmid", "himid", "air"):
            self.assertIn(name, rep)


class TestCompareCli(unittest.TestCase):
    def run_cli(self, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(list(args))
        return code, buf.getvalue()

    def test_compare_two_wavs(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.wav"), os.path.join(d, "b.wav")
            write_wav(a, stereo(sine(60)), SR)
            write_wav(b, stereo(sine(1000)), SR)
            code, out = self.run_cli("compare", a, b)
        self.assertEqual(code, 0)
        self.assertIn("a.wav", out)
        self.assertIn("b.wav", out)
        self.assertIn("sub", out)

    def test_compare_mono_wav(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mono.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes((sine(300) * 32767).astype(np.int16).tobytes())
            code, out = self.run_cli("compare", path)
        self.assertEqual(code, 0)
        self.assertIn("mono.wav", out)

    def test_compare_missing_file(self):
        buf = io.StringIO()
        from contextlib import redirect_stderr
        with redirect_stderr(buf):
            code = main(["compare", "nope.wav"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
