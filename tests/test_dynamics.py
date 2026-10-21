"""Plynulá dynamika samplované stopy: automatizace `dyn` (jako modwheel)."""

import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from paprika import Song  # noqa: E402
from paprika.automation import targets  # noqa: E402
from paprika.render import render_song  # noqa: E402
from paprika.sampler import Sampler, clear_cache  # noqa: E402
from sample_fixture import make_sfz_library, sine_wav  # noqa: E402

SR = 44100


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12)


def tone_share(x, freq, sr=SR):
    """Podíl energie kolem `freq` (±15 Hz) v mono signálu."""
    x = np.asarray(x, np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return spec[(f > freq - 15) & (f < freq + 15)].sum() / (spec.sum() + 1e-12)


def make_two_tone_layers(root):
    """Jedna klávesa, dvě vrstvy s jinou výškou: p = 440 Hz, f = 880 Hz.

    `amp_veltrack=0`, takže se mění jen barva (vrstva), ne hlasitost.
    """
    sine_wav(os.path.join(root, "p.wav"), 440.0, SR, 5.0, 0.3)
    sine_wav(os.path.join(root, "f.wav"), 880.0, SR, 5.0, 0.3)
    path = os.path.join(root, "layers.sfz")
    with open(path, "w", encoding="utf-8") as f:
        f.write("<global> amp_veltrack=0 ampeg_release=0.1\n"
                "<region> sample=p.wav lokey=0 hikey=127 pitch_keycenter=69 lovel=1 hivel=63\n"
                "<region> sample=f.wav lokey=0 hikey=127 pitch_keycenter=69 lovel=64 hivel=127\n")
    return path


class TestDyn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.sfz = make_sfz_library(cls.tmp.name, sr=SR, seconds=5.0)
        cls.layers = make_two_tone_layers(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        clear_cache()
        cls.tmp.cleanup()

    def song(self, sfz, vel, dyn=None, bars=1):
        s = Song(bpm=60, bars=bars)
        t = s.track(Sampler(sfz, name="t", legato=True))
        t.note("A4" if sfz == self.layers else "C4", at=0, dur=4 * bars, vel=vel)
        if dyn is not None:
            t.automate("dyn", dyn, "lin")
        return render_song(s, None, report=False).tracks["t"].mean(axis=1)

    def test_swell_gets_louder_inside_one_note(self):
        x = self.song(self.sfz, 100, [(0, 0.3), (1, 1.0)])
        q = SR                                    # 60 BPM: 1 s na dobu
        self.assertGreater(rms_db(x[3 * q:4 * q]) - rms_db(x[0:q]), 6.0)

    def test_without_dyn_note_is_flat(self):
        x = self.song(self.sfz, 100)
        q = SR
        self.assertLess(abs(rms_db(x[3 * q:4 * q]) - rms_db(x[0:q])), 1.0)

    def test_layers_crossfade_over_time(self):
        x = self.song(self.layers, 100, [(0, 0.3), (1, 1.0)])
        q = SR
        start, end = x[int(0.1 * q):q], x[3 * q:int(3.9 * q)]
        self.assertGreater(tone_share(start, 440), 0.9)       # jen p vrstva
        self.assertGreater(tone_share(end, 880), 0.9)         # jen f vrstva
        # efektivní velocity 64 je v polovině: obě vrstvy zní spolu
        mid = x[int(1.8 * q):int(2.1 * q)]
        self.assertGreater(tone_share(mid, 440), 0.15)
        self.assertGreater(tone_share(mid, 880), 0.15)

    def test_constant_dyn_inside_layer_matches_plain_render(self):
        plain = self.song(self.sfz, 100)
        flat = self.song(self.sfz, 100, [(0, 1.0)])
        self.assertTrue(np.allclose(plain, flat, atol=1e-5))

    def test_dyn_scales_velocity_and_clamps(self):
        loud = self.song(self.sfz, 100, [(0, 5.0)])           # 500 -> 127
        top = self.song(self.sfz, 127)
        self.assertTrue(np.allclose(loud, top, atol=1e-5))

    def test_target_only_on_sampled_tracks(self):
        s = Song(bpm=60, bars=1)
        synth = s.track("epiano").automate("dyn", [(0, 1.0)])
        synth.note("C4", 0, 1)
        with self.assertRaises(KeyError) as cm:
            render_song(s, None, report=False)
        self.assertIn("dyn", str(cm.exception))
        self.assertIn("dyn", targets([], sampled=True))
        self.assertNotIn("dyn", targets([]))


if __name__ == "__main__":
    unittest.main()
