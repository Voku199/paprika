import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from paprika.reference import _match_chord, analyze_reference  # noqa: E402
from paprika.theory import pitch  # noqa: E402

SR = 44100


def _freq(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _tone(midis, dur_sec, sr=SR, amp=0.3):
    t = np.arange(int(sr * dur_sec)) / sr
    x = np.zeros_like(t, dtype=np.float32)
    for m in midis:
        x += (amp * np.sin(2 * np.pi * _freq(m) * t)).astype(np.float32)
    return x


def _click_track(bpm: float, dur_sec: float, sr=SR):
    t = np.arange(int(sr * dur_sec)) / sr
    x = np.zeros_like(t, dtype=np.float32)
    period = 60.0 / bpm
    click_times = np.arange(0, dur_sec, period)
    for ct in click_times:
        i = int(ct * sr)
        if i + 200 <= x.size:
            x[i:i + 200] += np.hanning(200)
    return x


class TestTempo(unittest.TestCase):
    def test_detects_120_bpm_click_track(self):
        x = _click_track(120.0, 8.0)
        rep = analyze_reference(x, SR)
        self.assertAlmostEqual(rep.bpm, 120.0, delta=6.0)
        self.assertGreater(rep.bpm_confidence, 0.3)

    def test_detects_90_bpm_click_track(self):
        x = _click_track(90.0, 8.0)
        rep = analyze_reference(x, SR)
        self.assertAlmostEqual(rep.bpm, 90.0, delta=6.0)


class TestKey(unittest.TestCase):
    def test_c_major_triad(self):
        x = _tone([pitch("C4"), pitch("E4"), pitch("G4")], 6.0)
        rep = analyze_reference(x, SR)
        self.assertEqual((rep.key, rep.scale), ("C", "major"))
        self.assertGreater(rep.key_confidence, 0.5)

    def test_a_minor_triad(self):
        x = _tone([pitch("A3"), pitch("C4"), pitch("E4")], 6.0)
        rep = analyze_reference(x, SR)
        self.assertEqual((rep.key, rep.scale), ("A", "minor"))
        self.assertGreater(rep.key_confidence, 0.5)

    def test_g_major_triad(self):
        x = _tone([pitch("G3"), pitch("B3"), pitch("D4")], 6.0)
        rep = analyze_reference(x, SR)
        self.assertEqual((rep.key, rep.scale), ("G", "major"))


class TestChordMatch(unittest.TestCase):
    def test_pure_c_major_chroma(self):
        chroma = np.zeros(12)
        for pc in (0, 4, 7):  # C E G
            chroma[pc] = 1.0
        self.assertEqual(_match_chord(chroma), "C")

    def test_pure_d_minor_chroma(self):
        chroma = np.zeros(12)
        for pc in (2, 5, 9):  # D F A
            chroma[pc] = 1.0
        self.assertEqual(_match_chord(chroma), "Dm")

    def test_silence_is_unknown(self):
        self.assertEqual(_match_chord(np.zeros(12)), "?")


class TestAnalyzeShape(unittest.TestCase):
    def test_bar_lists_align_and_cover_duration(self):
        x = _click_track(100.0, 10.0)
        rep = analyze_reference(x, SR)
        self.assertEqual(len(rep.chords), len(rep.bar_energy_db))
        self.assertGreater(len(rep.chords), 0)
        self.assertAlmostEqual(rep.duration_sec, 10.0, delta=0.1)

    def test_empty_signal_does_not_crash(self):
        rep = analyze_reference(np.zeros(0, dtype=np.float32), SR)
        self.assertEqual(rep.chords, [])


if __name__ == "__main__":
    unittest.main()
