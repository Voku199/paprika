import dataclasses
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import (Formant, Glottal, Patch, Saw, Song, Tract, dsp,  # noqa: E402
                     model_warnings, preset, presets_by_tag)
from paprika.filters import FORMANTS, VOICE_LENGTH, tube_formants, vowel_areas  # noqa: E402
from paprika.human import pitch_curve  # noqa: E402
from paprika.oscillators import glottal, lf_timing  # noqa: E402
from paprika.sfx import _CACHE, render_sfx  # noqa: E402
from paprika.voices import track_pitch, voice_quality  # noqa: E402

SR = 44100


def _peak_hz(y, lo, hi, sr=SR):
    p = np.abs(np.fft.rfft(np.asarray(y, dtype=np.float64) * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(p[m])])


def _centroid(y, sr=SR):
    y = np.asarray(y, dtype=np.float64)
    p = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(y.size, 1.0 / sr)
    return float((p * f).sum() / p.sum())


def _harmonic_db(y, f0, k, sr=SR):
    p = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(y.size, 1.0 / sr)
    return 10.0 * np.log10(p[np.argmin(abs(f - k * f0))] + 1e-20)


def _median_midi(y, sr=SR):
    p, _, _ = track_pitch(np.asarray(y, dtype=np.float64), sr)
    return float(np.nanmedian(p))


class TestGlottal(unittest.TestCase):
    def test_lf_timing(self):
        tp, te, ta = lf_timing(1.0)
        self.assertAlmostEqual(tp, 0.484, delta=0.005)
        self.assertAlmostEqual(te, 0.650, delta=0.005)
        self.assertAlmostEqual(ta, 0.038, delta=0.002)
        tes = [lf_timing(rd)[1] for rd in (0.3, 1.0, 2.7)]
        self.assertEqual(tes, sorted(tes))

    def test_breathier_has_steeper_tilt(self):
        h12 = []
        for rd in (0.3, 1.0, 2.7):
            y = glottal(120.0, SR, SR, rd=rd, jitter=0, shimmer=0, aspiration=0).astype(float)
            h12.append(_harmonic_db(y, 120.0, 1) - _harmonic_db(y, 120.0, 2))
        self.assertLess(h12[0], 0.0)                 # tlačený: H2 nad H1
        self.assertTrue(0.0 < h12[1] < 8.0)          # modální
        self.assertGreater(h12[2], 10.0)             # dyšný

    def test_level_pitch_determinism(self):
        y = glottal(150.0, SR, SR, aspiration=0).astype(float)
        self.assertAlmostEqual(float(np.sqrt(np.mean(y ** 2))), 1.0, delta=0.2)
        self.assertAlmostEqual(_peak_hz(y, 100, 200), 150.0, delta=2.0)
        np.testing.assert_array_equal(glottal(150.0, 5000, SR, seed=2),
                                      glottal(150.0, 5000, SR, seed=2))
        self.assertFalse(np.allclose(glottal(150.0, 5000, SR, seed=2),
                                     glottal(150.0, 5000, SR, seed=3)))
        curve = np.linspace(100.0, 200.0, SR)
        self.assertTrue(np.all(np.isfinite(glottal(curve, SR, SR, rd=np.linspace(0.5, 2.5, SR)))))

    def test_osc_in_patch_and_key(self):
        y = Patch(osc=[Glottal()], filter=Tract("a")).voice(130.0, 0.3, sr=SR)
        self.assertTrue(np.all(np.isfinite(y)))
        base = Glottal()
        changed = [Glottal(rd=1.5), Glottal(jitter=0.01), Glottal(shimmer=0.05),
                   Glottal(aspiration=0.2), Glottal(leak=0.1), Glottal(doubling=0.3),
                   Glottal(seed=4), Glottal(ratio=2.0)]
        self.assertEqual(len({base.key()} | {g.key() for g in changed}), len(changed) + 1)
        with self.assertRaises(ValueError):
            Glottal(rd=4.0)


class TestVoiceQuality(unittest.TestCase):
    TRACT = [(650, 80), (1080, 90), (2650, 120)]

    def _measure(self, **kw):
        y = glottal(130.0, SR, SR, **kw).astype(float)
        return voice_quality(dsp.cascade(y, self.TRACT, SR), SR, 130.0)

    def test_sterile_source(self):
        q = self._measure(jitter=0, shimmer=0, aspiration=0)
        self.assertLess(q["jitter"], 0.05)
        self.assertLess(q["shimmer"], 0.3)
        self.assertGreater(q["hnr"], 35.0)

    def test_default_is_human(self):
        q = self._measure()
        self.assertTrue(0.2 <= q["jitter"] <= 1.0, q)
        self.assertTrue(1.0 <= q["shimmer"] <= 4.0, q)
        self.assertTrue(15.0 <= q["hnr"] <= 30.0, q)

    def test_more_jitter_measures_more(self):
        lo = self._measure(jitter=0.004, aspiration=0)["jitter"]
        hi = self._measure(jitter=0.016, aspiration=0)["jitter"]
        self.assertGreater(hi, 2.5 * lo)

    def test_aspiration_lowers_hnr(self):
        self.assertLess(self._measure(aspiration=0.3)["hnr"],
                        self._measure(aspiration=0.03)["hnr"] - 6.0)


class TestTract(unittest.TestCase):
    def test_uniform_tube(self):
        f = [fi for fi, _ in tube_formants((3.0,) * 12, 17.5, 3)]
        for got, want in zip(f, (500.0, 1500.0, 2500.0)):
            self.assertAlmostEqual(got / want, 1.0, delta=0.01)
        b = [bi for _, bi in tube_formants((3.0,) * 12, 17.5, 3)]
        self.assertTrue(40.0 < b[0] < 120.0)

    def test_vowels_match_table(self):
        for v, tol in (("a", 0.02), ("e", 0.02), ("i", 0.02), ("o", 0.05), ("u", 0.07)):
            got = [f for f, _ in tube_formants(vowel_areas(v), 17.0, 3)]
            for g, w in zip(got, FORMANTS["tenor"][v][0][:3]):
                self.assertAlmostEqual(g / w, 1.0, delta=tol, msg=v)

    def test_length_scales_formants(self):
        a = Tract("a", voice="tenor").formants()
        b = Tract("a", voice="soprano").formants()
        ratio = VOICE_LENGTH["tenor"] / VOICE_LENGTH["soprano"]
        self.assertAlmostEqual(b[0][0] / a[0][0], ratio, delta=0.01)

    def test_morph_moves_f2(self):
        x = glottal(120.0, 2 * SR, SR).astype(float)
        y = Tract("a", morph=("a", "i", 1.0)).apply(x, SR, freq=120.0)
        early = _peak_hz(y[2000:2000 + 8192], 900, 2400)
        late = _peak_hz(y[-8192:], 900, 2400)
        self.assertLess(early, 1300.0)
        self.assertGreater(late, 1600.0)

    def test_nasal_hum_is_dark(self):
        x = glottal(120.0, SR, SR).astype(float)
        oral = Tract("a").apply(x, SR, freq=120.0)
        nasal = Tract("a", nasal=1.0).apply(x, SR, freq=120.0)
        self.assertLess(_centroid(nasal), _centroid(oral) * 0.7)

    def test_vary_norm_stereo_key(self):
        x = glottal(220.0, 8000, SR).astype(float)
        a = Tract("o", vary=0.05).apply(x, SR, freq=220.0)
        np.testing.assert_array_equal(a, Tract("o", vary=0.05).apply(x, SR, freq=220.0))
        self.assertFalse(np.allclose(a, Tract("o").apply(x, SR, freq=220.0)))
        n = Tract("a", norm=True, gain=1.0).apply(x, SR, freq=220.0)
        self.assertAlmostEqual(float(np.sqrt(np.mean(n.astype(float) ** 2))),
                               float(np.sqrt(np.mean(x ** 2))), delta=0.01)
        st = Tract("e").apply(np.stack([x, x], axis=1), SR, freq=220.0)
        self.assertEqual(st.shape, (8000, 2))
        base = Tract()
        variants = []
        for fld in dataclasses.fields(Tract):
            val = {"vowel": "i", "voice": "bass", "length": 16.0, "areas": (2.0, 3.0, 4.0),
                   "morph": ("a", "u", 0.3), "nasal": 0.5, "vary": 0.02, "gain": 0.7,
                   "norm": True}[fld.name]
            variants.append(dataclasses.replace(base, **{fld.name: val}))
        self.assertEqual(len({base.key()} | {v.key() for v in variants}), len(variants) + 1)
        with self.assertRaises(ValueError):
            Tract("y")


class TestFormantCascade(unittest.TestCase):
    def test_default_unchanged_and_level_even(self):
        x = Saw().render(220.0, SR // 2, SR)
        np.testing.assert_array_equal(Formant("a").apply(x, SR, freq=220.0),
                                      Formant("a", mode="parallel").apply(x, SR, freq=220.0))
        levels = [float(np.sqrt(np.mean(Formant(v, mode="cascade").apply(x, SR, freq=220.0)
                                        .astype(float) ** 2))) for v in "aiu"]
        self.assertLess(max(levels) / min(levels), 1.05)
        self.assertNotEqual(Formant("a").key(), Formant("a", mode="cascade").key())
        self.assertNotEqual(Formant("a").key(), Formant("a", vary=0.03).key())
        with self.assertRaises(ValueError):
            Formant("a", mode="serial")


class TestPitchCurve(unittest.TestCase):
    def test_overshoot_then_settle(self):
        rng = np.random.default_rng(0)
        n = SR
        y = pitch_curve(n, SR, [(0.0, 60.0), (0.3, 60.0), (0.301, 62.0), (1.0, 62.0)], rng,
                        drift=0.0, fine=0.0)
        self.assertGreater(float(y.max()), 62.05)                  # přestřel
        self.assertAlmostEqual(float(y[-SR // 5:].mean()), 62.0, delta=0.02)

    def test_drift_and_determinism(self):
        a = pitch_curve(SR, SR, [(0, 60), (1, 60)], np.random.default_rng(1), drift=10.0, fine=0.0)
        b = pitch_curve(SR, SR, [(0, 60), (1, 60)], np.random.default_rng(1), drift=10.0, fine=0.0)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(0.03 < float(np.std(a)) < 0.2)             # ~10 centů


class TestGestures(unittest.TestCase):
    def setUp(self):
        _CACHE.clear()

    def test_hum_pitch_and_nasal(self):
        y = render_sfx("hum", (48.0, 3.0, "male", 0), SR, 120)[:, 0].astype(float)
        self.assertAlmostEqual(_median_midi(y[SR // 4:-SR // 4]), 48.0, delta=0.3)
        self.assertLess(_centroid(y), 800.0)
        q = voice_quality(y[int(0.4 * SR):int(1.2 * SR)], SR, 130.8)
        self.assertTrue(0.2 <= q["jitter"] <= 1.5, q)

    def test_sigh_falls_and_is_breathy(self):
        y = render_sfx("sigh", (62.0, 5.0, 2.6, "female", 0), SR, 120)[:, 0].astype(float)
        p, _, hop = track_pitch(y, SR, fmin=180.0, fmax=600.0)
        voiced = np.flatnonzero(~np.isnan(p))
        self.assertGreater(voiced.size, 0.5 * p.size)              # znělý, ne šum
        q = voiced.size // 4
        first = float(np.median(p[voiced[:q]]))
        last = float(np.median(p[voiced[-q:]]))
        self.assertLess(last, first - 1.5)
        hum = render_sfx("hum", (62.0, 2.6, "female", 0), SR, 120)[:, 0].astype(float)
        f0 = 440.0 * 2 ** ((62 - 69) / 12)
        self.assertLess(voice_quality(y[int(0.15 * SR):int(0.5 * SR)], SR, f0)["hnr"],
                        voice_quality(hum[int(0.4 * SR):int(0.8 * SR)], SR, f0)["hnr"] - 5.0)

    def test_effort_hard_onset_and_rough(self):
        y = render_sfx("effort", (50.0, 0.6, "a", "male", 0), SR, 120)[:, 0].astype(float)
        env = np.abs(y)
        self.assertLess(int(np.argmax(env)), int(0.06 * SR))      # rázové nasazení
        f0 = 440.0 * 2 ** ((50 - 69) / 12)
        q = voice_quality(y[int(0.03 * SR):int(0.2 * SR)], SR, f0)
        self.assertGreater(q["shimmer"], 5.0)                       # hrubý hlas

    def test_laugh_calls_and_falling_pitch(self):
        y = render_sfx("laugh", (60.0, 5, 4.5, "a", "male", 0), SR, 120)[:, 0].astype(float)
        env = dsp.onepole(np.abs(y), 15.0, SR).astype(float)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(env, distance=int(0.12 * SR), height=0.25 * env.max())
        self.assertEqual(len(peaks), 5)
        self.assertGreater(env[peaks[0]], env[peaks[-1]])          # síla klesá
        p, _, hop = track_pitch(y, SR)
        pitch_at = [float(np.nanmedian(p[max(0, int(k / SR / hop) - 3):int(k / SR / hop) + 3]))
                    for k in peaks[[0, -1]]]
        self.assertLess(pitch_at[1], pitch_at[0])                  # výška klesá


class TestHumanTrackVoice(unittest.TestCase):
    def test_pitch_from_plan(self):
        s = Song(bpm=120, key="D", scale="minor", bars=4)
        s.harmony("i | VI | iv | V")
        male = s.human("m", voice="male", mic=False).hum(0)
        female = s.human("f", voice="female", mic=False).hum(0).laugh(3)
        self.assertEqual(male.hits[0].params[0], 50.0)             # D3 blízko 48
        self.assertEqual(female.hits[0].params[0], 57.0)           # A3 blízko 57
        self.assertEqual(female.hits[1].params[0], 69.0)           # A4 nad mluvenou výškou
        t = s.human("x", mic=False).hum(1, pitch="F3").effort(2, pitch=55)
        self.assertEqual([h.params[0] for h in t.hits], [53.0, 55.0])

    def test_without_plan_uses_tonic(self):
        s = Song(key="G", scale="major")
        self.assertEqual(s.human(mic=False).hum(0).hits[0].params[0] % 12, 7)

    def test_sigh_inhales_first_and_laugh_length(self):
        s = Song(bpm=120, bars=4)
        h = s.human(mic=False).sigh(2).laugh(3, calls=3, rate=3.0)
        names = [x.name for x in h.hits]
        self.assertEqual(names, ["breath", "sigh", "laugh"])
        b = h.hits[0]
        self.assertAlmostEqual(b.start_beat + b.dur_beats, h.hits[1].start_beat)
        self.assertAlmostEqual(h.hits[2].dur_beats, (3 / 3.0 + 0.12) * 2.0)

    def test_renders_with_mic(self):
        s = Song(bpm=100, bars=2, sample_rate=22050)
        s.human(voice="female").hum(0, beats=2).sigh(1).effort(1, beat=3)
        out = s.render(None, report=False)
        self.assertTrue(np.all(np.isfinite(out.audio)))
        self.assertGreater(float(np.abs(out.audio).max()), 0.01)

    def test_gesture_over_notes_warns(self):
        s = Song(bpm=120, bars=4)
        lead = s.track("cello_solo", name="lead")
        lead.notes("D4:4 | F4:4 | _:4 | _:4", bar=0)
        s.human("hlas", performer=lead, mic=False).hum(1).hum(2)
        texts = [w.text for w in model_warnings(s)]
        self.assertEqual(sum("hlasové gesto hum jde přes noty" in t for t in texts), 1)


class TestRulesAndPreset(unittest.TestCase):
    def test_sterile_voice(self):
        s = Song()
        s.track(Patch(osc=[Glottal(jitter=0, shimmer=0)], filter=Tract("a")), name="robot")
        s.track(Patch(osc=[Glottal()], filter=Tract("a")), name="clovek")
        warns = [w for w in model_warnings(s) if "sterilní hlas" in w.text]
        self.assertEqual([w.where for w in warns], ["robot"])

    def test_choir_tract_preset(self):
        self.assertIn("choir_tract", presets_by_tag("choir"))
        y = preset("choir_tract").voice(220.0, 0.6, vel=100, sr=SR)
        self.assertEqual(y.ndim, 2)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertLess(float(np.abs(y).max()), 1.0)


if __name__ == "__main__":
    unittest.main()
