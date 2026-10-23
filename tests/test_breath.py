import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from paprika import Sampler, Song, breath, cli, voices  # noqa: E402
from paprika.library import SamplesMissing  # noqa: E402
from paprika.render import render_song  # noqa: E402
from paprika.sampler import clear_cache  # noqa: E402
from paprika.sfx import _CACHE, render_sfx  # noqa: E402
from sample_fixture import (SINGER_BREATH_HZ, fake_vocalset_breaths,  # noqa: E402
                            make_sfz_library)

SR = 44100


def _centroid(y, sr=SR):
    y = np.asarray(y, dtype=np.float64)
    y = y[:, 0] if y.ndim == 2 else y
    p = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(y.size, 1.0 / sr)
    return float((p * f).sum() / p.sum())


class TestFind(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.truths = fake_vocalset_breaths(os.path.join(cls.tmp.name, "vocalset", "FULL"),
                                           singers=("male1",), vowels="ao")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_finds_breaths_before_tones(self):
        path = next(p for p in self.truths if "straight_a" in p)
        sr, x = voices._load_mono(path)
        found, rejects = breath.find_breaths(x, sr)
        self.assertEqual(len(found), 3)
        for b, (t0, t1) in zip(found, self.truths[path]):
            self.assertAlmostEqual(b.t0, t0, delta=0.05)
            self.assertLess(b.t1, t1 + 0.03)
            self.assertGreater(b.t1, t1 - 0.12)       # doznívání konce je pod prahem
            self.assertLess(b.rel_db, -8.0)           # nádech je tišší než zpěv
        self.assertEqual(sum(rejects.values()), 0)

    def test_rejects_clicks_rumble_and_exhale(self):
        path = next(p for p in self.truths if "distract" in p)
        sr, x = voices._load_mono(path)
        found, rejects = breath.find_breaths(x, sr)
        self.assertEqual(len(found), 1)               # jen nádech před 3. tónem
        self.assertAlmostEqual(found[0].t0, self.truths[path][0][0], delta=0.05)
        self.assertEqual(sum(rejects.values()), 2)
        self.assertIn("dunění", rejects)

    def test_check_rejects_tone(self):
        from sample_fixture import sung_tone
        why, _ = breath._check(sung_tone(55, SR, 0.4), SR)
        self.assertEqual(why, "tónový")

    def test_tone_tail_is_not_breath(self):
        """YIN doznívání tónu přehlédne (VocalSet male3): nádech nesmí začít v něm."""
        from sample_fixture import breath_sound, sung_tone
        tail_n = int(0.12 * SR)
        decay = np.exp(-np.arange(tail_n) / SR / 0.04)
        tail = sung_tone(48, SR, 0.5)[int(0.1 * SR):int(0.1 * SR) + tail_n] * decay  # ~60 ms nad nádechem
        b = breath_sound(SR, 0.4, 1300.0, 3)
        b[:tail_n] += tail                               # doznívání přechází do nádechu
        parts = [np.zeros(int(0.3 * SR)), sung_tone(48, SR, 1.0), b,
                 np.zeros(int(0.06 * SR)), sung_tone(55, SR, 1.0), np.zeros(int(0.3 * SR))]
        x = np.concatenate(parts) + 0.001 * np.random.default_rng(1).standard_normal(
            sum(p.size for p in parts))
        tail0 = 1.3
        real = voices.track_pitch

        def deaf(sig, sr, *a, **k):                      # YIN doznívání neslyší
            pitch, rms, hop = real(sig, sr, *a, **k)
            t = np.arange(pitch.size) * hop
            pitch[(t >= tail0 - 0.02) & (t < tail0 + 0.12)] = np.nan
            return pitch, rms, hop

        with mock.patch.object(voices, "track_pitch", deaf):
            found, _ = breath.find_breaths(x, SR)
            with mock.patch.object(breath, "TONAL_FLAT", 0.0):
                naive, _ = breath.find_breaths(x, SR)
        self.assertEqual(len(found), 1)
        self.assertGreater(found[0].t0, tail0 + 0.04)    # až po doznívání
        self.assertLess(naive[0].t0, tail0 + 0.02)       # bez kontroly začne v tónu

    def test_tonal_frames(self):
        from sample_fixture import breath_sound, sung_tone
        fw = int(0.02 * SR)
        tone = sung_tone(48, SR, 0.2)[2000:2000 + fw]
        noise = breath_sound(SR, 0.2, 2000.0, 1)[3000:3000 + fw]
        self.assertEqual(breath._tonal(np.stack([tone, noise]), SR).tolist(), [True, False])

    def test_silence_has_no_breaths(self):
        found, _ = breath.find_breaths(np.zeros(SR), SR)
        self.assertEqual(found, [])


class TestDescribe(unittest.TestCase):
    def test_vector_layout_and_color(self):
        from sample_fixture import breath_sound
        lo = breath.describe(breath_sound(SR, 0.4, 1000.0, 1), SR)
        hi = breath.describe(breath_sound(SR, 0.4, 4000.0, 1), SR)
        k, b = breath.K, breath.BANDS
        self.assertEqual(lo.size, k + k * b)
        self.assertAlmostEqual(float(lo[:k].max()), 0.0, delta=0.5)
        self.assertGreaterEqual(float(lo[:k].min()), breath.FLOOR_DB)
        _, centers = breath._bank(SR, breath._stft_size(SR)[0])
        peak = lambda v: centers[int(np.argmax(v[k:].reshape(k, b).mean(axis=0)))]
        self.assertLess(peak(lo), 1600.0)
        self.assertGreater(peak(hi), 2500.0)

    def test_bank_is_area_normalized(self):
        fb, centers = breath._bank(SR, 1024)
        np.testing.assert_allclose(fb.sum(axis=1), 1.0, rtol=1e-9)
        self.assertTrue(np.all(np.diff(centers) > 0))


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        fake_vocalset_breaths(os.path.join(cls.root, "vocalset", "FULL"),
                              singers=("male1", "female2"))
        # male3 má jen jeden soubor (3 nádechy): vlastní model nedostane
        fake_vocalset_breaths(os.path.join(cls.root, "vocalset", "FULL"),
                              singers=("male3",), vowels="a", distractors=False)
        cls.path = breath.build(cls.root, log=lambda *a: None, reuse=False,
                                singers=["male1", "female2", "male3"])
        cls.env = mock.patch.dict(os.environ, {"PAPRIKA_SAMPLES": cls.root})
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        clear_cache()
        breath._CACHE.clear()
        cls.tmp.cleanup()

    def setUp(self):
        _CACHE.clear()

    def test_models_written(self):
        import json
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(sorted(data["models"]), ["female", "female2", "male", "male1"])
        self.assertEqual(data["stats"]["male1"]["breaths"], 16)    # 5 x 3 + distractor
        self.assertEqual(data["stats"]["male1"]["rejected"], {"krátký": 1, "dunění": 1})
        self.assertEqual(data["models"]["male"]["count"], 19)       # male1 + male3
        self.assertIn("CC BY 4.0", data["source"])

    def test_synthesis_reproduces_corpus(self):
        for name in ("male1", "female2"):
            m = breath.load(name)
            self.assertLess(m["lsd_db"], 2.0, name)
            self.assertLess(m["env_db"], 2.0, name)
            self.assertAlmostEqual(m["synth_centroid"] / m["centroid"], 1.0, delta=0.15)

    def test_singer_color_survives(self):
        rng = np.random.default_rng(0)
        male = np.mean([_centroid(breath.synth(breath.load("male1"), SR // 2, SR, rng))
                        for _ in range(4)])
        female = np.mean([_centroid(breath.synth(breath.load("female2"), SR // 2, SR, rng))
                          for _ in range(4)])
        self.assertLess(male, female - 800.0)
        self.assertLess(SINGER_BREATH_HZ["male1"], SINGER_BREATH_HZ["female2"])

    def test_resolve_names(self):
        self.assertEqual(breath.resolve("soprano"), "female2")
        self.assertEqual(breath.resolve("alto.ooh"), "female7")
        self.assertEqual(breath.resolve("male_choir"), "male")
        self.assertEqual(breath.resolve("female_choir.aah"), "female")
        self.assertEqual(breath.resolve("male3"), "male3")
        with self.assertRaises(KeyError):
            breath.resolve("flute")

    def test_singer_without_model_gets_group(self):
        self.assertEqual(breath.load("male3")["name"], "male")
        self.assertEqual(breath.load("tenor")["name"], "male")       # male9 chybí

    def test_synth_deterministic_and_exact(self):
        m = breath.load("female2")
        a = breath.synth(m, 12345, SR, np.random.default_rng(5))
        b = breath.synth(m, 12345, SR, np.random.default_rng(5))
        c = breath.synth(m, 12345, SR, np.random.default_rng(6))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.size, 12345)
        self.assertAlmostEqual(float(np.sqrt(np.mean(a ** 2))), 1.0, places=6)
        self.assertLess(abs(np.corrcoef(a, c)[0, 1]), 0.2)

    def test_generator_uses_model_and_physics_level(self):
        ref = render_sfx("breath", ("mouth", 0.9, 0.6, "a", "male", "female2", 0), SR, 120)
        catch = render_sfx("breath", ("catch", 0.4, 0.35, "a", "male", "female2", 0), SR, 120)
        slow = render_sfx("breath", ("mouth", 2.4, 1.0, "a", "male", "female2", 0), SR, 120)
        db = lambda y: 20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        self.assertGreater(db(catch), db(ref) + 2.0)
        self.assertLess(db(slow), db(ref) - 3.0)
        self.assertGreater(_centroid(catch), _centroid(slow))
        phys = render_sfx("breath", ("mouth", 0.9, 0.6, "a", "male", None, 0), SR, 120)
        self.assertAlmostEqual(db(ref), db(phys), delta=1.0)   # stejná kalibrace
        self.assertFalse(np.allclose(ref[:1000], phys[:1000]))

    def test_track_with_model(self):
        s = Song(bpm=120, bars=4, sample_rate=SR)
        lead = s.track("cello_solo", name="lead")
        lead.notes("D4:3 _:1 | F4:2 E4:1.5 _:0.5 | D4:4 | _:4", bar=0)
        h = s.human("dech", performer=lead, breath_model="soprano", mic=False)
        self.assertEqual(h.breath_model, "female2")
        h.breaths().breath(to_bar=3, kind="nose")
        models = [x.params[5] for x in h.hits]
        self.assertEqual(models, ["female2", "female2", None])
        buf = render_song(s, None, report=False).tracks["dech"]
        edge = 2 * SR                                    # takt 1 = 2 s
        self.assertGreater(float(np.abs(buf[edge - int(0.3 * SR):edge]).max()), 0.01)

    def test_auto_takes_vocalset_singer(self):
        with tempfile.TemporaryDirectory() as d:
            sfz = make_sfz_library(d)
            s = Song()
            sop = s.track(Sampler(sfz, name="soprano.aah"), name="sop")
            self.assertEqual(s.human(performer=sop).breath_model, "female2")
            self.assertIsNone(s.human(performer=s.track("strings")).breath_model)
            self.assertIsNone(s.human(performer=sop, breath_model=None).breath_model)
            clear_cache()

    def test_cli_stats_and_examples(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, redirect_stdout(out):
            code = cli.main(["samples", "breaths", "--examples", d])
            files = sorted(os.listdir(d))
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("female2", text)
        self.assertIn("dunění", text)
        self.assertIn("male3", text)                     # bez vlastního modelu
        self.assertIn("female2_real_1.wav", files)
        self.assertIn("female2_synth_3.wav", files)


class TestMissing(unittest.TestCase):
    def test_missing_model_raises_install_hint(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {"PAPRIKA_SAMPLES": d}):
            breath._CACHE.clear()
            s = Song()
            with self.assertRaises(SamplesMissing) as cm:
                s.human(breath_model="female2")
            self.assertIn("samples install voices", str(cm.exception))
            self.assertIsNone(s.human().breath_model)    # bez performera: fyzika
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), mock.patch.object(sys, "stderr", err):
                self.assertEqual(cli.main(["samples", "breaths"]), 2)
            self.assertIn("samples install voices", err.getvalue())

    def test_build_without_recordings(self):
        with tempfile.TemporaryDirectory() as d:
            path = breath.build(d, log=lambda *a: None, reuse=False)
            breath._CACHE.clear()
            with mock.patch.dict(os.environ, {"PAPRIKA_SAMPLES": d}):
                self.assertTrue(os.path.isfile(path))
                with self.assertRaises(SamplesMissing):
                    breath.load("soprano")


if __name__ == "__main__":
    unittest.main()
