import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Song, corpus, model_warnings  # noqa: E402
from paprika.analyze import _gestures_only  # noqa: E402
from paprika.cli import main  # noqa: E402
from paprika.library import SamplesMissing  # noqa: E402
from paprika.render import render_song  # noqa: E402
from paprika.sfx import _CACHE, render_sfx  # noqa: E402
from paprika.voices import track_pitch  # noqa: E402

from sample_fixture import breath_sound, sung_tone  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "tools", "human_worker.py")
SR = 44100


def _wav(path, x, sr=SR, meta=None):
    wavfile.write(path, sr, (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))
    if meta is not None:
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f)


def _info(**kw):
    base = {"dur": 1.0, "peak": 0.5, "peak_db": -10.0, "clipped": 0, "voiced": 0.5,
            "pitch": 60.0, "bursts": 4.0}
    base.update(kw)
    return base


def _worker(request_path, out):
    return subprocess.run([sys.executable, WORKER, request_path, "--out", out],
                          capture_output=True, text=True, encoding="utf-8", cwd=ROOT)


class TestSieve(unittest.TestCase):
    def test_analyze_sung_tone(self):
        x = sung_tone(57, seconds=1.0)
        info = corpus.analyze_clip(x, SR)
        self.assertAlmostEqual(info["dur"], 1.0, places=2)
        self.assertGreater(info["voiced"], 0.8)
        self.assertAlmostEqual(info["pitch"], 57.0, delta=0.3)
        self.assertEqual(info["clipped"], 0)

    def test_analyze_noise_has_no_pitch(self):
        info = corpus.analyze_clip(breath_sound(SR, 0.6, 1500.0, 3), SR)
        self.assertLess(info["voiced"], 0.2)
        self.assertIsNone(info["pitch"])

    def test_check_reasons(self):
        self.assertIsNone(corpus.check(_info(), "laugh"))
        self.assertEqual(corpus.check(_info(peak_db=-70.0), "laugh"), "ticho")
        self.assertEqual(corpus.check(_info(clipped=50), "laugh"), "clipping")
        self.assertEqual(corpus.check(_info(dur=9.0), "laugh"), "dlouhý")
        self.assertEqual(corpus.check(_info(dur=0.05), "cough"), "krátký")
        self.assertEqual(corpus.check(_info(voiced=0.9), "cough"), "moc znělý")
        self.assertEqual(corpus.check(_info(voiced=0.1), "hum"), "málo znělý")
        self.assertEqual(corpus.check(_info(bursts=0.5), "laugh"), "rytmus")

    def test_check_classifier_and_transcript(self):
        speech = {"labels": {"Speech": 0.9, "Laughter": 0.3}}
        self.assertEqual(corpus.check(_info(), "laugh", speech), "řeč (klasifikátor)")
        other = {"labels": {"Speech": 0.1, "Music": 0.8, "Laughter": 0.05}}
        self.assertEqual(corpus.check(_info(), "laugh", other), "jiný zvuk (klasifikátor)")
        good = {"labels": {"Speech": 0.4, "Laughter": 0.7}}
        self.assertIsNone(corpus.check(_info(), "laugh", good))
        words = {"transcript": "Well, that was funny."}
        self.assertEqual(corpus.check(_info(), "laugh", words), "řeč (přepis)")
        fillers = {"transcript": "Hahaha, hah. Mm."}
        self.assertIsNone(corpus.check(_info(), "laugh", fillers))

    def test_prepare_trims_and_levels(self):
        tone = sung_tone(55, seconds=0.8, amp=0.05)
        x = np.concatenate([np.zeros(SR), tone, np.zeros(SR // 2)])
        y = corpus.prepare(x, SR)
        self.assertLess(y.size, 0.95 * SR)
        self.assertGreater(y.size, 0.8 * SR)
        mid = y[y.size // 4: 3 * y.size // 4]
        self.assertAlmostEqual(20 * np.log10(np.sqrt(np.mean(mid ** 2))),
                               corpus.TARGET_RMS_DB, delta=1.5)
        self.assertLessEqual(np.abs(y).max(), corpus.PEAK_CEILING + 1e-9)
        self.assertEqual(y[0], 0.0)

    def test_request(self):
        req = corpus.request(["laugh", "sigh"], backend="dia", count=3, seed=10)
        self.assertEqual(len(req["jobs"]), 6)
        self.assertEqual([j["seed"] for j in req["jobs"][:3]], [10, 11, 12])
        self.assertEqual(req["jobs"][0]["prompt"], corpus.PROMPTS["dia"]["laugh"])
        with self.assertRaisesRegex(ValueError, "neznámý backend"):
            corpus.request(["laugh"], backend="xyz")
        with self.assertRaisesRegex(ValueError, "neznámý druh"):
            corpus.request(["burp"])
        with self.assertRaisesRegex(ValueError, "stable_audio"):
            corpus.request(["steps"], backend="dia")


class TestCorpus(unittest.TestCase):
    """Celý protokol: zadání -> worker (backend paprika) -> síto -> korpus."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = os.path.join(cls.tmp.name, "samples")
        cls.incoming = os.path.join(cls.tmp.name, "incoming")
        cls.env = mock.patch.dict(os.environ, {"PAPRIKA_SAMPLES": cls.root})
        cls.env.start()
        req = corpus.request(["laugh", "sigh", "hum", "cloth"], backend="paprika", count=3)
        cls.req_path = os.path.join(cls.tmp.name, "req.json")
        with open(cls.req_path, "w", encoding="utf-8") as f:
            json.dump(req, f)
        cls.run1 = _worker(cls.req_path, cls.incoming)
        cls.log = []
        cls.result = corpus.ingest(cls.incoming, log=cls.log.append)

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        corpus._CACHE.clear()
        corpus._AUDIO.clear()
        cls.tmp.cleanup()

    def setUp(self):
        _CACHE.clear()

    def test_worker_made_clips_with_sidecars(self):
        self.assertEqual(self.run1.returncode, 0, self.run1.stderr)
        wavs = [f for f in os.listdir(self.incoming) if f.endswith(".wav")]
        self.assertEqual(len(wavs), 12)
        with open(os.path.join(self.incoming, wavs[0].replace(".wav", ".json")),
                  encoding="utf-8") as f:
            meta = json.load(f)
        for key in ("kind", "backend", "model", "prompt", "seed", "license", "attribution"):
            self.assertIn(key, meta)

    def test_worker_skips_finished_jobs(self):
        again = _worker(self.req_path, self.incoming)
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("z cache 12", again.stdout)

    def test_worker_unknown_backend(self):
        bad = os.path.join(self.tmp.name, "bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            json.dump({"jobs": [{"kind": "laugh", "backend": "nope", "seed": 0}]}, f)
        res = _worker(bad, os.path.join(self.tmp.name, "bad_out"))
        self.assertEqual(res.returncode, 2)
        self.assertIn("neznámý backend", res.stderr)

    def test_ingest_accepts_and_writes_manifest(self):
        self.assertEqual(len(self.result["accepted"]), 12, self.result["rejected"])
        base = corpus.corpus_root()
        with open(os.path.join(base, corpus.MANIFEST), encoding="utf-8") as f:
            data = json.load(f)
        clip = data["clips"][0]
        self.assertTrue(os.path.isfile(os.path.join(base, *clip["file"].split("/"))))
        self.assertEqual(clip["source"]["backend"], "paprika")
        self.assertIn("license", clip)
        with open(os.path.join(base, corpus.CREDITS), encoding="utf-8") as f:
            self.assertIn("Paprika, vestavěná syntéza", f.read())
        hums = corpus.clips("hum")
        self.assertTrue(all(c["pitch"] is not None for c in hums))
        self.assertTrue(all(c["pitch"] is None for c in corpus.clips("cloth")))

    def test_stored_level(self):
        sr, x = corpus.read_clip(corpus.clips("hum")[0]["id"])
        self.assertEqual(sr, SR)
        rms = 20 * np.log10(np.sqrt(np.mean(x[x.size // 4: 3 * x.size // 4] ** 2)))
        self.assertAlmostEqual(rms, corpus.TARGET_RMS_DB, delta=2.0)

    def test_reimport_is_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            res = corpus.ingest(self.incoming, root=d, log=lambda s: None)
            self.assertEqual(len(res["accepted"]), 12)
            again = corpus.ingest(self.incoming, root=d, log=lambda s: None)
            self.assertEqual(again["accepted"], [])
            self.assertEqual(set(again["rejected"].values()), {"duplicitní"})
            # posunutý a tišší, ale stejný klip je pořád duplicita
            src = sorted(f for f in os.listdir(self.incoming)
                         if f.startswith("hum") and f.endswith(".wav"))[0]
            sr, x = corpus._load_wav(os.path.join(self.incoming, src))
            own = os.path.join(d, "own")
            os.makedirs(own)
            _wav(os.path.join(own, "kopie.wav"), np.concatenate([np.zeros(sr // 3), 0.5 * x]),
                 sr)
            res = corpus.ingest(own, root=d, kind="hum", license="CC0", log=lambda s: None)
            self.assertEqual(res["rejected"], {"kopie.wav": "duplicitní"})

    def test_own_recordings_need_license(self):
        with tempfile.TemporaryDirectory() as d:
            own = os.path.join(d, "own")
            os.makedirs(own)
            noise = breath_sound(SR, 0.4, 2500.0, 7) * 0.3
            _wav(os.path.join(own, "sniff.wav"), noise)
            _wav(os.path.join(own, "ticho.wav"), np.zeros(SR // 2))
            clipped = np.clip(sung_tone(50, seconds=1.0) * 6.0, -1.0, 1.0)
            _wav(os.path.join(own, "nahlas.wav"), clipped)
            _wav(os.path.join(own, "dlouhy.wav"), np.tile(noise, 10))
            res = corpus.ingest(own, root=d, kind="sniff", log=lambda s: None)
            self.assertEqual(set(res["rejected"].values()), {"chybí licence (--license)"})
            res = corpus.ingest(own, root=d, kind="sniff", license="CC0",
                                attribution="vlastní nahrávka", log=lambda s: None)
            self.assertEqual(len(res["accepted"]), 1)
            self.assertEqual(res["rejected"], {"ticho.wav": "ticho", "nahlas.wav": "clipping",
                                               "dlouhy.wav": "dlouhý"})
            entry = corpus.clips("sniff", root=d)[0]
            self.assertEqual(entry["license"], "CC0")
            self.assertEqual(corpus.credit_line(entry), "vlastní nahrávka (CC0)")
            res = corpus.ingest(own, root=d, kind="burp", license="CC0", log=lambda s: None)
            self.assertTrue(all("neznámý druh" in w for w in res["rejected"].values()))

    def test_empty_kind_is_samples_missing(self):
        with self.assertRaises(SamplesMissing) as ctx:
            corpus.clips("cough")
        self.assertIn("samples human", str(ctx.exception))
        with self.assertRaisesRegex(ValueError, "neznámý druh"):
            corpus.clips("burp")

    def _song(self, performer=False):
        s = Song(bpm=90, key="D", scale="minor", bars=4)
        s.harmony("i | VI | iv | V")
        lead = None
        if performer:
            lead = s.track("lead", name="lead").notes("D5:2 _:2 | A4:4 | _:4 | _:4")
        return s, s.human("hlas", performer=lead)

    def test_clip_variants_and_params(self):
        s, h = self._song()
        for bar in range(3):
            h.clip("laugh", bar)
        ids = [hit.params[1] for hit in h.hits]
        self.assertEqual(len(set(ids)), 3)
        h.clip("laugh", 3, variant=1)
        self.assertEqual(h.hits[-1].params[1], corpus.clips("laugh")[1]["id"])
        self.assertEqual(h.hits[0].params[2], None)
        n = len(h.hits)
        h.clip("hum", 2, pitch="chord")
        midi = h.hits[n].params[2]
        self.assertIn(int(midi) % 12, {7, 10, 2})            # Gm v taktu 2
        own = corpus.clips("hum")[h.hits[n].params[4]]["pitch"]
        self.assertLessEqual(abs(midi - own), 7)
        h.clip("hum", 3, pitch="A3")
        self.assertEqual(h.hits[-1].params[2], 57.0)
        with self.assertRaisesRegex(ValueError, "hlasových"):
            h.clip("cloth", 0, pitch="chord")
        with self.assertRaisesRegex(ValueError, "kladné"):
            h.clip("hum", 0, beats=0)

    def test_clip_beats_and_render(self):
        s, h = self._song()
        e = corpus.clips("hum")[0]
        h.clip("hum", 0, beats=1.0, variant=0)
        self.assertAlmostEqual(h.hits[0].dur_beats, 1.0)
        y = render_sfx("clip", h.hits[0].params, SR, 90.0)
        self.assertAlmostEqual(y.shape[0] / SR, 60.0 / 90.0, delta=0.01)
        self.assertEqual(y.shape[1], 2)
        full = render_sfx("clip", ("hum", e["id"], None, None, 0), SR, 90.0)
        self.assertAlmostEqual(full.shape[0] / SR, e["dur"], delta=0.01)
        self.assertLess(np.abs(full).max(), 1.0)
        self.assertGreater(np.sqrt(np.mean(full ** 2)), 0.01)

    def test_clip_pitch_shift(self):
        e = corpus.clips("hum")[0]
        for step in (2.0, -3.0):
            y = render_sfx("clip", ("hum", e["id"], e["pitch"] + step, None, 0), SR, 120.0)
            pitch, _, _ = track_pitch(y[:, 0], SR)
            self.assertAlmostEqual(float(np.nanmedian(pitch)), e["pitch"] + step, delta=0.35)
        # cíl o oktávu níž: posune se jen o půltóny k nejbližší oktávě cíle
        y = render_sfx("clip", ("hum", e["id"], e["pitch"] - 10.0, None, 0), SR, 120.0)
        pitch, _, _ = track_pitch(y[:, 0], SR)
        self.assertAlmostEqual(float(np.nanmedian(pitch)), e["pitch"] + 2.0, delta=0.35)

    def test_song_renders_with_credits(self):
        s, h = self._song()
        h.clip("laugh", 0, beat=1)
        h.clip("sigh", 1, pitch="chord")
        out = render_song(s, None, report=False)
        self.assertGreater(float(np.abs(out.tracks["hlas"]).max()), 0.001)
        infos = [w for w in model_warnings(s) if w.where == "korpus"]
        self.assertEqual(len(infos), 1)
        self.assertIn("Paprika, vestavěná syntéza (bez omezení)", infos[0].text)
        self.assertTrue(_gestures_only(h))     # krátká gesta: maskování je přeskočí
        h.clip("cloth", 2)
        self.assertFalse(_gestures_only(h))

    def test_vocal_clip_over_performer_warns(self):
        s, h = self._song(performer=True)
        h.clip("laugh", 1)                 # přes A4 v taktu 1
        h.clip("cloth", 1)                 # šustění oblečení hraní nevadí
        h.clip("sigh", 2)                  # pauza
        warns = [w for w in model_warnings(s) if "jde přes noty" in w.text]
        self.assertEqual(len(warns), 1)
        self.assertIn("klip laugh", warns[0].text)
        self.assertEqual(warns[0].bar, 1)

    def test_repeated_clip_warns(self):
        s, h = self._song()
        h.clip("laugh", 0, variant=2)
        h.clip("sigh", 1)
        h.clip("laugh", 2, variant=2)
        warns = [w for w in model_warnings(s) if "stejný vzorek" in w.text]
        self.assertEqual(len(warns), 1)
        self.assertIn("doplň", warns[0].text)

    def test_cli(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["samples", "human"]), 0)
        self.assertRegex(out.getvalue(), r"laugh\s+3 klipů")
        path = os.path.join(self.tmp.name, "cli_req.json")
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["samples", "human", "request", "cough", "--count", "2",
                                   "--out", path]), 0)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)["jobs"]), 2)
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(main(["samples", "human", "request", "burp", "--out", path]), 2)
        self.assertIn("neznámý druh", err.getvalue())
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["samples", "human", "import", self.incoming]), 0)
        self.assertIn("odmítnuto 12", out.getvalue())

    def test_cli_missing_corpus_is_clean_error(self):
        song = os.path.join(self.tmp.name, "songs", "kasel.py")
        os.makedirs(os.path.dirname(song), exist_ok=True)
        with open(song, "w", encoding="utf-8") as f:
            f.write("from paprika import Song\ns = Song(bars=1)\n"
                    "s.human('h').clip('cough', 0)\n")
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(main(["check", song]), 2)
        self.assertIn("chyba: korpus lidských zvuků nemá žádný klip druhu 'cough'",
                      err.getvalue())


if __name__ == "__main__":
    unittest.main()
