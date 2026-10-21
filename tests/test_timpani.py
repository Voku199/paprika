"""Tympány z VSCO 2 CE: kotle bez noty ve jméně, výška ze změřené mapy."""

import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from paprika import Song, install, library  # noqa: E402
from paprika.render import render_song  # noqa: E402
from paprika.sampler import clear_cache  # noqa: E402
from paprika.sfz import load as load_sfz  # noqa: E402


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def wav_bytes(freq, seconds=1.0, sr=44100):
    buf = io.BytesIO()
    t = np.arange(int(sr * seconds)) / sr
    wavfile.write(buf, sr, (8000 * np.sin(2 * np.pi * freq * t)).astype(np.int16))
    return buf.getvalue()


class TestNames(unittest.TestCase):
    def test_named_pitch_without_note(self):
        p = round(install.VSCO_NAMED_PITCH["Timpani3_Hit"])
        self.assertEqual(install.parse_sample_name("Timpani3_Hit_v4_rr2_Sum.wav"), (p, 4, 2))
        r = round(install.VSCO_NAMED_PITCH["Timpani1_Roll"])
        self.assertEqual(install.parse_sample_name("Timpani1_Roll_v5_rr1_Sum.wav"), (r, 5, 1))

    def test_unknown_name_without_note_is_skipped(self):
        self.assertIsNone(install.parse_sample_name("Tumba-HitN_v1.wav"))

    def test_catalog(self):
        inst = library.INSTRUMENTS["timpani"]
        self.assertEqual(inst.group, "percussion")
        self.assertIn("percussion", library.GROUPS)
        self.assertTrue(inst.is_one_shot("hit"))
        self.assertFalse(inst.is_one_shot("roll"))
        self.assertTrue(inst.loops("roll"))
        self.assertFalse(inst.loops("hit"))


class TestTimpaniInstall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = self.tmp.name

    def tearDown(self):
        clear_cache()
        self.tmp.cleanup()

    def install(self):
        hit = {d: wav_bytes(440 * 2 ** ((install.VSCO_NAMED_PITCH[f"Timpani{d}_Hit"] - 69) / 12))
               for d in (1, 3)}
        roll = wav_bytes(100.0, seconds=3.0)
        files = {f"Percussion/Timpani/Timpani{d}_Hit_v{v}_rr1_Sum.wav": hit[d]
                 for d in (1, 3) for v in (1, 3)}
        files.update({f"Percussion/Timpani/Rolls/Timpani{d}_Roll_v3_rr1_Sum.wav": roll
                      for d in (1, 3)})
        tree = [{"type": "blob", "path": p, "size": len(b)} for p, b in files.items()]

        def fake(url, timeout=60.0):
            if "api.github.com" in url:
                return FakeResp(json.dumps({"tree": tree}).encode())
            path = url.split(install.VSCO_SHA + "/", 1)[1].replace("%20", " ")
            return FakeResp(files[path])

        with mock.patch.object(install, "_urlopen", side_effect=fake):
            return install.install("percussion", root=self.root, log=lambda *a: None)

    def test_install_writes_tuned_regions(self):
        self.assertEqual(self.install(), ["timpani"])
        self.assertTrue(library.is_installed("timpani", self.root))
        hit = load_sfz(os.path.join(self.root, "vsco2", "sfz", "timpani.hit.sfz"))
        exact = install.VSCO_NAMED_PITCH["Timpani3_Hit"]
        r = next(x for x in hit if "Timpani3" in x.sample)
        self.assertEqual(r.keycenter, round(exact))
        self.assertAlmostEqual(r.tune, (round(exact) - exact) * 100, delta=1)
        self.assertEqual(r.loop_mode, "one_shot")
        roll = load_sfz(os.path.join(self.root, "vsco2", "sfz", "timpani.roll.sfz"))
        self.assertTrue(all(x.loop_mode == "loop_continuous" for x in roll))

    def test_hit_plays_in_tune(self):
        self.install()
        with mock.patch.dict(os.environ, {"PAPRIKA_SAMPLES": self.root}):
            s = Song(bpm=60, bars=1)
            s.track("timpani", chromatic=True).note("C#3", at=0, dur=1, vel=100)
            buf = render_song(s, None, report=False).tracks["timpani"].mean(axis=1)
        size = 1 << 15
        spec = np.abs(np.fft.rfft(buf[:size] * np.hanning(size)))
        f = np.argmax(spec) * 44100 / size
        self.assertAlmostEqual(f, 138.59, delta=2.0)          # C#3 přesně


if __name__ == "__main__":
    unittest.main()
