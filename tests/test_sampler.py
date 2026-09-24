import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from paprika.sampler import (LEGATO_XFADE, TAIL_CAP, SampleError,  # noqa: E402
                             Sampler, clear_cache, load_wav)
from sample_fixture import make_sfz_library, midi_freq, sine_wav  # noqa: E402

SR = 44100


def peak_freq(x, sr=SR):
    m = x.mean(axis=1) if x.ndim == 2 else x
    size = 1 << 15
    m = m[:size] if m.size >= size else np.pad(m, (0, size - m.size))
    spec = np.abs(np.fft.rfft(m * np.hanning(size)))
    k = int(np.argmax(spec))
    a, b, c = np.log(spec[k - 1:k + 2] + 1e-12)
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * sr / size


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


class TestSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dir = cls.tmp.name
        cls.sfz = make_sfz_library(cls.dir, sr=48000)
        cls.s = Sampler(cls.sfz, legato=True)

    @classmethod
    def tearDownClass(cls):
        clear_cache()
        cls.tmp.cleanup()

    def test_select_by_key_and_velocity(self):
        r = self.s.select(64, 100)[0]
        self.assertEqual((r.keycenter, r.lovel), (64, 64))
        r = self.s.select(65, 20)[0]
        self.assertEqual((r.keycenter, r.hivel), (64, 63))

    def test_round_robin(self):
        a, b, c = (self.s.select(60, 100, i) for i in range(3))
        self.assertEqual(len(a), 1)
        self.assertNotEqual(a[0].sample, b[0].sample)
        self.assertEqual(a[0].sample, c[0].sample)

    def test_out_of_range_uses_nearest_root(self):
        self.assertEqual(self.s.select(80, 100)[0].keycenter, 68)
        self.assertEqual(self.s.range, (58, 70))

    def test_pitch_shift_and_resample(self):
        # vzorky jsou 48 kHz, render 44.1 kHz; E4 region zahraje F4
        r = self.s.select(65, 100)[0]
        v = self.s.voice(r, 65, 0.8, SR)
        self.assertEqual(v.shape[1], 2)
        self.assertEqual(v.dtype, np.float32)
        self.assertAlmostEqual(peak_freq(v) / midi_freq(65), 1.0, delta=0.005)

    def test_gate_and_release_length(self):
        r = self.s.select(64, 100)[0]
        v = self.s.voice(r, 64, 0.3, SR)
        self.assertEqual(v.shape[0], round(0.3 * SR) + round(0.2 * SR))
        self.assertLess(np.abs(v[-10:]).max(), 1e-3)

    def test_sample_ends_without_loop(self):
        r = self.s.select(64, 100)[0]
        v = self.s.voice(r, 64, 5.0, SR)        # vzorek má jen 1 s
        self.assertAlmostEqual(v.shape[0], SR, delta=1)

    def test_gain_law(self):
        r = self.s.select(64, 100)[0]
        self.assertAlmostEqual(self.s.gain(r, 127), 1.0)
        self.assertAlmostEqual(self.s.gain(r, 64), (64 / 127) ** 2, places=4)
        self.assertAlmostEqual(self.s.gain(r, 500), 1.0)

    def test_legato_in_fades_and_out_is_short(self):
        r = self.s.select(64, 100)[0]
        v = self.s.voice(r, 64, 0.3, SR, legato_in=True, legato_out=True)
        self.assertEqual(v.shape[0], round(0.3 * SR) + round(LEGATO_XFADE * SR))
        head = np.abs(v[: int(0.01 * SR)]).max()
        body = np.abs(v[int(0.15 * SR):int(0.25 * SR)]).max()
        self.assertLess(head, body * 0.5)

    def test_missing_sfz_and_wav(self):
        with self.assertRaises(SampleError) as cm:
            Sampler(os.path.join(self.dir, "nic.sfz"))
        self.assertIn("nic.sfz", str(cm.exception))
        s = Sampler(write(os.path.join(self.dir, "bad.sfz"),
                          "<region> sample=chybi.wav key=60\n"))
        with self.assertRaises(SampleError) as cm:
            s.voice(s.regions[0], 60, 0.5, SR)
        self.assertIn("chybi.wav", str(cm.exception))

    def test_release_regions_ignored(self):
        s = Sampler(write(os.path.join(self.dir, "rel.sfz"),
                          "<region> sample=wav dir/s_60_v1_rr1.wav key=60\n"
                          "<region> sample=wav dir/s_64_v1_rr1.wav key=60 "
                          "trigger=release\n"))
        self.assertEqual(len(s.regions), 1)
        self.assertEqual(len(s.select(60, 100)), 1)

    def test_regions_without_keys_ignored(self):
        # Salamander: pedálový šum spouští CC64, ne klávesa (lokey=-1 hikey=-1)
        s = Sampler(write(os.path.join(self.dir, "cc.sfz"),
                          "<region> sample=wav dir/s_60_v1_rr1.wav key=60\n"
                          "<group> lokey=-1 hikey=-1 on_locc64=126\n"
                          "<region> sample=wav dir/s_64_v1_rr1.wav\n"))
        self.assertEqual(len(s.regions), 1)
        self.assertEqual(s.range, (60, 60))
        self.assertEqual(s.select(10, 100)[0].keycenter, 60)

    def test_layered_regions_play_together(self):
        s = Sampler(write(os.path.join(self.dir, "layer.sfz"),
                          "<region> sample=wav dir/s_60_v1_rr1.wav key=60\n"
                          "<region> sample=wav dir/s_64_v1_rr1.wav key=60\n"))
        self.assertEqual(len(s.select(60, 100)), 2)

    def test_loop_extends_long_note(self):
        sine_wav(os.path.join(self.dir, "loop.wav"), 220.0, SR, 0.5, 0.5)
        s = Sampler(write(os.path.join(self.dir, "loop.sfz"),
                          "<region> sample=loop.wav key=57 loop_mode=loop_continuous "
                          "loop_start=10000 loop_end=20000 ampeg_release=0.1\n"))
        v = s.voice(s.regions[0], 57, 3.0, SR)
        self.assertEqual(v.shape[0], 3 * SR + round(0.1 * SR))
        mid = v[SR:2 * SR, 0]
        self.assertGreater(np.sqrt(np.mean(mid ** 2)), 0.2)
        self.assertLess(np.abs(np.diff(v[:3 * SR, 0])).max(), 0.1)

    def test_one_shot_ignores_gate(self):
        s = Sampler(write(os.path.join(self.dir, "os.sfz"),
                          "<region> sample=wav dir/s_60_v1_rr1.wav key=60 "
                          "loop_mode=one_shot\n"))
        v = s.voice(s.regions[0], 60, 0.05, SR)
        self.assertAlmostEqual(v.shape[0], SR, delta=1)
        self.assertGreater(np.abs(v[SR // 2:SR // 2 + 500]).max(), 0.01)
        self.assertEqual(s.tail, TAIL_CAP)

    def test_undamped_high_notes_ring(self):
        s = Sampler(self.sfz, undamped_from=64)
        r = s.select(64, 100)[0]
        self.assertGreater(s.voice(r, 64, 0.1, SR).shape[0],
                           self.s.voice(r, 64, 0.1, SR).shape[0])

    def test_tail_and_key(self):
        self.assertAlmostEqual(self.s.tail, 0.2)
        self.assertNotEqual(Sampler(self.sfz, gain_db=-3).key(), self.s.key())
        self.assertEqual(Sampler(self.sfz).key(), Sampler(self.sfz).key())

    def test_float_mono_wav(self):
        from scipy.io import wavfile
        p = os.path.join(self.dir, "mono.wav")
        wavfile.write(p, SR, (0.3 * np.sin(np.arange(SR) * 0.05)).astype(np.float32))
        sr, data = load_wav(p)
        self.assertEqual((sr, data.shape[1]), (SR, 1))
        s = Sampler(write(os.path.join(self.dir, "mono.sfz"),
                          "<region> sample=mono.wav key=60\n"))
        v = s.voice(s.regions[0], 60, 0.2, SR)
        self.assertTrue(np.allclose(v[:, 0], v[:, 1]))


if __name__ == "__main__":
    unittest.main()
