import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Song  # noqa: E402
from paprika.fx import FxContext  # noqa: E402
from paprika.render import render_track  # noqa: E402


class TestBeatToSample(unittest.TestCase):
    def test_bar4_beat2_at_120bpm(self):
        s = Song(bpm=120, bars=8)
        # bar 4 začíná na beatu 16, plus 2 beaty = 18 beatů = 9 s při 120 BPM
        beat = 4 * s.beats_per_bar + 2
        self.assertEqual(s.beat_to_sample(beat), 9 * 44100)

    def test_bpm_change_scales_everything(self):
        fast, slow = Song(bpm=120), Song(bpm=60)
        self.assertEqual(slow.beat_to_sample(4), 2 * fast.beat_to_sample(4))

    def test_duration(self):
        s = Song(bpm=120, bars=8)
        self.assertAlmostEqual(s.duration, 16.0)


class TestNotePlacement(unittest.TestCase):
    def _first_nonzero(self, buf, floor=1e-4):
        nz = np.nonzero(np.abs(buf) > floor)[0]
        return int(nz[0]) if nz.size else -1

    def test_note_starts_at_expected_sample(self):
        s = Song(bpm=120, bars=8)
        t = s.track("organ", vol=1.0)
        t.note("C4", at=4 * s.beats_per_bar + 2, dur=1)
        ctx = FxContext(sr=s.sample_rate, bpm=s.bpm, beats_per_bar=s.beats_per_bar)
        buf = render_track(t, s.sample_rate * 12, ctx)
        expected = 9 * 44100
        start = self._first_nonzero(buf)
        # 2 ms fade-in znamená, že první vzorky jsou skoro nulové
        self.assertLess(abs(start - expected), int(0.004 * 44100))

    def test_notes_string_lands_on_bar(self):
        s = Song(bpm=120, bars=8)
        t = s.track("organ")
        t.notes("C4:4", bar=2)
        self.assertEqual(t.all_notes()[0].start_beat, 8.0)

    def test_arp_step_length(self):
        s = Song(bpm=120, bars=4)
        t = s.track("pluck")
        t.arp("C", bar=0, pattern="up", rate=1 / 8, octaves=1)
        starts = [n.start_beat for n in t.all_notes()]
        self.assertEqual(len(starts), 8)
        self.assertAlmostEqual(starts[1] - starts[0], 0.5)

    def test_drum_pattern_steps(self):
        s = Song(bpm=120, bars=4)
        d = s.drums()
        d.pattern("k...s...k...s...", bars=(0, 2))
        starts = sorted(h.start_beat for h in d.hits)
        self.assertEqual(starts[:4], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(len(starts), 8)  # dva takty po čtyřech úderech

    def test_drum_pattern_respects_rate(self):
        s = Song(bpm=120, bars=4)
        d = s.drums()
        d.pattern("khkh", bars=(0, 1), rate=1 / 4)
        self.assertEqual(sorted(h.start_beat for h in d.hits),
                         [0.0, 1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
