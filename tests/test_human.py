import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import (DynEQ, HumanTrack, Proximity, RoomTone, Song,  # noqa: E402
                     dsp, model_warnings)
from paprika.analyze import warnings_for  # noqa: E402
from paprika.fx import FxContext  # noqa: E402
from paprika.fx_mic import SPEED_OF_SOUND  # noqa: E402
from paprika.human import (CEILING, GENERATORS, HUMAN_NAMES, air_seconds,  # noqa: E402
                           breath_phrases)
from paprika.midi import to_bytes  # noqa: E402
from paprika.render import render_song  # noqa: E402
from paprika.sfx import _CACHE, render_sfx  # noqa: E402

SR = 44100


def _rms_db(y) -> float:
    return 20.0 * np.log10(float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2))) + 1e-12)


def _centroid(y, sr=SR) -> float:
    m = np.asarray(y, dtype=np.float64)
    m = m[:, 0] if m.ndim == 2 else m
    p = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(m.size, 1.0 / sr)
    return float((p * f).sum() / p.sum())


def _gain_db(y, freq, sr=SR) -> float:
    h = np.fft.rfft(np.asarray(y, dtype=np.float64))
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    return 20.0 * np.log10(abs(h[np.argmin(abs(f - freq))]))


BREATH = ("mouth", 0.9, 0.6, "a", "male", 0)      # 0.45 s při 120 BPM


class TestNoise(unittest.TestCase):
    def test_ar1_variance_and_memory(self):
        tau = 0.002
        y = dsp.ar1(200000, tau, SR, seed=4)
        self.assertAlmostEqual(float(np.var(y)), 1.0, delta=0.1)
        lag1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
        self.assertAlmostEqual(lag1, np.exp(-1.0 / (tau * SR)), delta=0.01)

    def test_ar1_deterministic(self):
        np.testing.assert_array_equal(dsp.ar1(500, 5.0, seed=2), dsp.ar1(500, 5.0, seed=2))
        self.assertFalse(np.allclose(dsp.ar1(500, 5.0, seed=2), dsp.ar1(500, 5.0, seed=3)))

    def test_pink_walk_has_long_memory(self):
        w = dsp.pink_walk(400000, SR, tau_min=0.001, octaves=8, seed=1)
        self.assertAlmostEqual(float(np.var(w)), 1.0, delta=0.25)
        lag = int(0.02 * SR)                    # 20 ms: ar1 s tau 1 ms už zapomněl
        short = dsp.ar1(400000, 0.001, SR, seed=1)
        self.assertGreater(np.corrcoef(w[:-lag], w[lag:])[0, 1], 0.15)
        self.assertLess(abs(np.corrcoef(short[:-lag], short[lag:])[0, 1]), 0.05)

    def test_filt_varying_scalar_gain_unchanged(self):
        x = np.random.default_rng(0).standard_normal(4000)
        np.testing.assert_array_equal(
            dsp.filt_varying(x, "peak", 1000.0, SR, q=1.0, gain_db=-6.0),
            dsp.filt(x, "peak", 1000.0, SR, q=1.0, gain_db=-6.0))

    def test_filt_varying_gain_curve(self):
        x = np.random.default_rng(0).standard_normal(SR)
        g = np.where(np.arange(SR) < SR // 2, 0.0, -18.0)
        y = dsp.filt_varying(x, "peak", 3000.0, SR, q=1.0, gain_db=g)
        first = dsp.filt(y[:SR // 2], "bp", 3000.0, SR, q=2.0)
        second = dsp.filt(y[SR // 2 + 2000:], "bp", 3000.0, SR, q=2.0)
        self.assertGreater(_rms_db(first) - _rms_db(second), 8.0)   # měří i boky zvonu


class TestHumanize(unittest.TestCase):
    def _track(self, n=400):
        s = Song(bpm=120, bars=n // 4 + 1)
        t = s.track("epiano")
        t.notes(" | ".join(["C4 D4 E4 F4"] * (n // 4)), bar=0)
        return s, t

    def test_corr_zero_is_legacy(self):
        _, a = self._track(16)
        _, b = self._track(16)
        a.humanize(timing=0.02, vel=0.1, seed=5)
        rng = random.Random(5)                   # původní algoritmus v2
        for clip in b.clips:
            for n in clip.notes:
                n.start_beat = max(0.0, n.start_beat + rng.uniform(-0.02, 0.02))
                n.velocity = int(max(1, min(127, round(
                    n.velocity * (1.0 + rng.uniform(-0.1, 0.1))))))
        self.assertEqual([(n.start_beat, n.velocity) for n in a.all_notes()],
                         [(n.start_beat, n.velocity) for n in b.all_notes()])

    def test_corr_makes_smooth_deviations(self):
        _, t = self._track(400)
        before = [n.start_beat for n in t.all_notes()]
        t.humanize(timing=0.02, vel=0.0, seed=1, corr=0.9)
        dev = np.array([n.start_beat for n in t.all_notes()]) - np.array(before)
        self.assertLessEqual(float(np.abs(dev).max()), 0.02 + 1e-12)
        self.assertGreater(float(np.corrcoef(dev[:-1], dev[1:])[0, 1]), 0.6)
        _, u = self._track(400)
        u.humanize(timing=0.02, vel=0.0, seed=1)
        dev0 = np.array([n.start_beat for n in u.all_notes()]) - np.array(before)
        self.assertLess(abs(float(np.corrcoef(dev0[:-1], dev0[1:])[0, 1])), 0.15)

    def test_corr_deterministic(self):
        _, a = self._track(40)
        _, b = self._track(40)
        a.humanize(seed=3, corr=0.8)
        b.humanize(seed=3, corr=0.8)
        self.assertEqual([(n.start_beat, n.velocity) for n in a.all_notes()],
                         [(n.start_beat, n.velocity) for n in b.all_notes()])

    def test_chord_moves_together(self):
        s = Song(bpm=120, bars=40)
        t = s.track("epiano")
        t.notes(" | ".join(["(C4 E4 G4):4"] * 40), bar=0)
        t.humanize(timing=0.02, vel=0.0, seed=2, corr=0.8)
        starts = {}
        for n in t.all_notes():
            starts.setdefault(round(n.start_beat / 4), []).append(n.start_beat)
        spread = np.mean([max(v) - min(v) for v in starts.values()])
        moves = np.std([np.mean(v) - 4 * k for k, v in starts.items()])
        self.assertLess(spread, moves)

    def test_louder_comes_earlier(self):
        _, t = self._track(400)
        before = {id(n): n.start_beat for c in t.clips for n in c.notes}
        t.humanize(timing=0.02, vel=0.2, seed=7, corr=0.5)
        dt = [n.start_beat - before[id(n)] for c in t.clips for n in c.notes]
        dv = [n.velocity for c in t.clips for n in c.notes]
        self.assertLess(float(np.corrcoef(dt, dv)[0, 1]), -0.1)

    def test_bad_corr(self):
        _, t = self._track(4)
        with self.assertRaises(ValueError):
            t.humanize(corr=1.0)

    def test_drums_and_sfx(self):
        s = Song(bpm=120, bars=4)
        d = s.drums().pattern("k.h.s.h.", bars=(0, 4))
        d.humanize(timing=0.01, seed=1, corr=0.9)
        self.assertTrue(any(h.start_beat % 0.25 > 1e-9 for h in d.hits))
        fx = s.sfx().impact(1).impact(2)
        fx.humanize(seed=1, corr=0.9)
        self.assertEqual([h.start_beat for h in fx.hits], [4.0, 8.0])


class TestProximity(unittest.TestCase):
    def _response(self, fx):
        imp = np.zeros((SR, 2), dtype=np.float32)
        imp[0] = 1.0
        return fx.process(imp, FxContext(sr=SR))[:, 0]

    def test_matches_formula(self):
        for dist in (0.05, 0.3):
            y = self._response(Proximity(dist=dist, max_db=None, hp=None))
            for f in (100.0, 200.0, 500.0):
                w = 2 * np.pi * f
                exact = 20 * np.log10(np.sqrt(1 + (SPEED_OF_SOUND / (2 * w * dist)) ** 2)
                                      / np.sqrt(1 + (SPEED_OF_SOUND / (2 * w)) ** 2))
                self.assertAlmostEqual(_gain_db(y, f), exact, delta=0.1)
            self.assertAlmostEqual(_gain_db(y, 8000.0), 0.0, delta=0.1)

    def test_cap_and_hp(self):
        y = self._response(Proximity(dist=0.03, max_db=12.0, hp=None))
        self.assertLess(_gain_db(y, 20.0), 12.1)
        y = self._response(Proximity(dist=0.03, max_db=12.0, hp=50.0))
        self.assertLess(_gain_db(y, 20.0), _gain_db(y, 200.0) - 6.0)

    def test_omni_has_no_effect(self):
        y = self._response(Proximity(dist=0.02, pattern="omni", hp=None))
        self.assertAlmostEqual(_gain_db(y, 100.0), 0.0, delta=0.01)

    def test_automated_distance(self):
        fx = Proximity(dist=1.0, max_db=None, hp=None)
        n = 2 * SR
        x = np.random.default_rng(1).standard_normal((n, 2)).astype(np.float32)
        fx.set_curve("dist", np.where(np.arange(n) < SR, 1.0, 0.05))
        y = fx.process(x, FxContext(sr=SR))
        fx.clear_curves()
        lo = [dsp.filt(y[a:a + SR // 2, 0], "lp", 150.0, SR) for a in (SR // 4, SR + SR // 4)]
        self.assertGreater(_rms_db(lo[1]) - _rms_db(lo[0]), 8.0)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            Proximity(pattern="shotgun")
        with self.assertRaises(ValueError):
            Proximity(dist=0)


class TestRoomTone(unittest.TestCase):
    def test_silence_stays_silent(self):
        x = np.zeros((SR, 2), dtype=np.float32)
        np.testing.assert_array_equal(RoomTone().process(x, FxContext(sr=SR)), x)

    def test_noise_follows_sound(self):
        x = np.zeros((3 * SR, 2), dtype=np.float32)
        x[SR:SR + SR // 2] = 0.3 * np.random.default_rng(0).standard_normal((SR // 2, 2))
        fx = RoomTone(level=-60.0, attack=0.08, release=0.5)
        tone = fx.process(x, FxContext(sr=SR)) - x
        self.assertAlmostEqual(_rms_db(tone[SR + 5000:SR + 20000]), -60.0, delta=1.0)
        self.assertGreater(_rms_db(tone[SR - 2000:SR]), -75.0)        # náběh před
        self.assertGreater(_rms_db(tone[SR // 2 * 3 + 5000:SR * 2]), -64.0)  # doběh
        self.assertLess(float(np.abs(tone[int(2.3 * SR):]).max()), 1e-6)
        self.assertLess(float(np.abs(tone[:SR - int(0.15 * SR)]).max()), 1e-6)
        self.assertEqual(fx.tail(FxContext()), 0.5)
        self.assertLess(abs(np.corrcoef(tone[SR:SR + 20000, 0], tone[SR:SR + 20000, 1])[0, 1]), 0.1)


class TestDynEQ(unittest.TestCase):
    def _sine(self, amp, f=5000.0, n=SR):
        t = np.arange(n) / SR
        return np.repeat((amp * np.sin(2 * np.pi * f * t))[:, None], 2, axis=1).astype(np.float32)

    def test_tames_loud_band_only(self):
        loud, quiet = self._sine(0.5), self._sine(0.002)
        fx = DynEQ(freq=5000.0, threshold=-30.0, ratio=4.0, range_db=12.0)
        out = fx.process(loud, FxContext(sr=SR))
        self.assertLess(_rms_db(out[SR // 2:]) - _rms_db(loud[SR // 2:]), -6.0)
        np.testing.assert_array_equal(fx.process(quiet, FxContext(sr=SR)), quiet)
        low = self._sine(0.5, f=200.0)
        out = fx.process(low, FxContext(sr=SR))
        self.assertAlmostEqual(_rms_db(out[SR // 2:]), _rms_db(low[SR // 2:]), delta=0.5)

    def test_source_ducks_music(self):
        music = self._sine(0.3, f=3500.0, n=2 * SR)
        key = np.zeros((2 * SR, 2), dtype=np.float32)
        key[SR:] = self._sine(0.5, f=3500.0)
        ctx = FxContext(sr=SR, track_buffers={"dech": key})
        fx = DynEQ(freq=3500.0, threshold=-30.0, source="dech")
        self.assertTrue(fx.needs_tracks)
        out = fx.process(music, ctx)
        self.assertAlmostEqual(_rms_db(out[SR // 4:SR - 2000]), _rms_db(music[SR // 4:SR - 2000]), delta=0.2)
        self.assertLess(_rms_db(out[SR + SR // 4:]) - _rms_db(music[SR + SR // 4:]), -4.0)

    def test_missing_source(self):
        with self.assertRaises(KeyError):
            DynEQ(source="nic").process(self._sine(0.1, n=1000), FxContext(sr=SR, track_buffers={}))


class TestGenerators(unittest.TestCase):
    PARAMS = {
        "breath": BREATH,
        "exhale": ("sigh", 2.0, 0.8, "a", "female", 1),
        "lip_smack": (2,),
        "heartbeat": (70.0, 0),
        "cloth": ("nylon", 1.0, "fidget", 0),
        "step": ("gravel", 1, 3),
    }

    def setUp(self):
        _CACHE.clear()

    def test_all_render_bounded(self):
        self.assertEqual(sorted(GENERATORS), sorted(HUMAN_NAMES))
        for name in HUMAN_NAMES:
            y = render_sfx(name, self.PARAMS[name], SR, 120)
            self.assertEqual(y.ndim, 2)
            self.assertEqual(y.shape[1], 2)
            self.assertTrue(np.all(np.isfinite(y)), name)
            self.assertLessEqual(float(np.abs(y).max()), CEILING + 1e-6, name)
            self.assertGreater(float(np.abs(y).max()), 0.01, name)

    def test_variants_differ_and_repeat(self):
        for name, params in self.PARAMS.items():
            a = render_sfx(name, params, SR, 120)
            _CACHE.clear()
            np.testing.assert_array_equal(a, render_sfx(name, params, SR, 120))
            b = render_sfx(name, params[:-1] + (params[-1] + 1,), SR, 120)
            m = min(len(a), len(b))
            r = abs(np.corrcoef(a[:m, 0], b[:m, 0])[0, 1])
            # tep a kroky po stejné podlaze si jsou podobné i ve skutečnosti
            self.assertLess(r, 0.95 if name in ("heartbeat", "step") else 0.3, name)

    def test_breath_length_follows_beats(self):
        y = render_sfx("breath", BREATH, SR, 120)
        self.assertEqual(y.shape[0], int(round(0.45 * SR)))

    def test_faster_breath_is_louder_and_brighter(self):
        ref = render_sfx("breath", BREATH, SR, 120)
        catch = render_sfx("breath", ("catch", 0.4, 0.35, "a", "male", 0), SR, 120)
        slow = render_sfx("breath", ("mouth", 2.4, 1.0, "a", "male", 0), SR, 120)
        self.assertGreater(_rms_db(catch), _rms_db(ref) + 2.0)
        self.assertLess(_rms_db(slow), _rms_db(ref) - 3.0)
        self.assertGreater(_centroid(catch), _centroid(slow) + 300.0)

    def test_nose_quieter_and_darker(self):
        mouth = render_sfx("breath", ("mouth", 1.2, 0.6, "a", "male", 0), SR, 120)
        nose = render_sfx("breath", ("nose", 1.2, 0.6, "a", "male", 0), SR, 120)
        self.assertLess(_rms_db(nose), _rms_db(mouth))
        self.assertLess(_centroid(nose), _centroid(mouth))

    def test_vowel_shapes_spectrum(self):
        u = render_sfx("breath", ("mouth", 0.9, 0.6, "u", "male", 0), SR, 120)
        i = render_sfx("breath", ("mouth", 0.9, 0.6, "i", "male", 0), SR, 120)
        self.assertGreater(_centroid(i), _centroid(u) + 500.0)

    def test_breath_has_little_sub_and_air(self):
        y = render_sfx("breath", BREATH, SR, 120)[:, 0].astype(np.float64)
        p = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(y.size, 1.0 / SR)
        self.assertLess(p[f < 200].sum() / p.sum(), 0.02)
        self.assertLess(p[f > 10000].sum() / p.sum(), 0.05)
        self.assertGreater(p[(f > 1000) & (f < 6000)].sum() / p.sum(), 0.6)

    def test_heartbeat_lub_dub(self):
        y = render_sfx("heartbeat", (60.0, 0), SR, 120)[:, 0]
        env = dsp.onepole(np.abs(y), 30.0, SR)
        first = int(np.argmax(env[:int(0.2 * SR)]))
        lo = int(0.25 * SR)
        second = lo + int(np.argmax(env[lo:]))
        self.assertAlmostEqual((second - first) / SR, 0.35, delta=0.06)

    def test_step_feet_alternate_sides(self):
        left = render_sfx("step", ("stone", 0, 0), SR, 120)
        right = render_sfx("step", ("stone", 1, 0), SR, 120)
        self.assertGreater(np.abs(left[:, 0]).sum(), np.abs(left[:, 1]).sum())
        self.assertGreater(np.abs(right[:, 1]).sum(), np.abs(right[:, 0]).sum())

    def test_bad_params(self):
        for name, params in (("breath", ("yawn",) + BREATH[1:]),
                             ("breath", BREATH[:3] + ("y",) + BREATH[4:]),
                             ("cloth", ("silk", 1.0, "swing", 0)),
                             ("step", ("ice", 0, 0))):
            with self.assertRaises(ValueError):
                render_sfx(name, params, SR, 120)


class TestHumanTrack(unittest.TestCase):
    def _song(self, mic=False):
        s = Song(bpm=120, bars=8, sample_rate=22050)
        lead = s.track("cello_solo", name="lead")
        lead.notes("D4:3 _:1 | F4:2 E4:1.5 _:0.5 | D4:4 | _:4", bar=0)
        lead.notes("A4:4 | G4:4 | F4:2 E4:2 | D4:4", bar=4)
        h = s.human("dech", performer=lead, mic=mic)
        return s, lead, h

    def test_breath_ends_on_target(self):
        s, _, h = self._song()
        h.breath(to_bar=2, beat=-0.5)
        b = h.hits[0]
        self.assertAlmostEqual(b.start_beat + b.dur_beats, 7.5)
        self.assertAlmostEqual(b.dur_beats, 0.45 * 2.0)       # 0.45 s při 120 BPM

    def test_auto_variants(self):
        _, _, h = self._song()
        h.breath(2).breath(4).breath(6)
        self.assertEqual(len({x.params[-1] for x in h.hits}), 3)
        h.steps((0, 2), every=1)
        steps = [x for x in h.hits if x.name == "step"]
        self.assertEqual([x.params[1] for x in steps[:4]], [0, 1, 0, 1])
        self.assertTrue(all(a.params != b.params for a, b in zip(steps, steps[1:])))

    def test_breaths_fit_rests(self):
        s, lead, h = self._song()
        h.breaths()
        breaths = [x for x in h.hits if x.name == "breath"]
        self.assertEqual(len(breaths), 3)          # před frázemi v taktu 1, 2 a 4; první nemá kdy
        notes = lead.all_notes()
        for b in breaths:
            lo, hi = b.start_beat, b.start_beat + b.dur_beats
            self.assertFalse(any(n.start_beat < hi and n.start_beat + n.dur_beats > lo
                                 for n in notes))
        kinds = [b.params[0] for b in breaths]
        self.assertEqual(kinds[1], "catch")                     # pauza 0.5 beatu = 0.25 s
        self.assertGreater(breaths[2].params[2], breaths[1].params[2])  # delší fráze = hlubší

    def test_breaths_need_performer(self):
        s = Song()
        with self.assertRaises(ValueError):
            s.human().breaths()

    def test_humanize_moves_timing(self):
        _, _, h = self._song()
        h.breath(2).breath(4)
        before = [x.start_beat for x in h.hits]
        h.humanize(seed=1)
        self.assertNotEqual(before, [x.start_beat for x in h.hits])
        self.assertTrue(all(abs(a - b.start_beat) <= 0.012 * 2.0 + 1e-12   # 12 ms při 120 BPM
                            for a, b in zip(before, h.hits)))

    def test_mic_and_performer_by_name(self):
        s = Song()
        s.track("strings", name="soprano_line")
        h = s.human(performer="soprano_line")
        self.assertIsInstance(h, HumanTrack)
        self.assertEqual(h.voice, "female")
        self.assertEqual([type(f) for f in h.fx_chain], [Proximity, RoomTone])
        self.assertEqual(s.human(mic=False).fx_chain, [])

    def test_render_places_breath(self):
        s, _, h = self._song()
        h.breath(to_bar=2)
        out = render_song(s, None, report=False)
        buf = out.tracks["dech"]
        sr = s.sample_rate
        edge = int(4.0 * sr)                                    # takt 2 = 4 s
        self.assertGreater(float(np.abs(buf[edge - int(0.3 * sr):edge]).max()), 0.02)
        self.assertLess(float(np.abs(buf[:edge - int(0.5 * sr)]).max()), 1e-6)

    def test_show_and_midi(self):
        s, _, h = self._song()
        h.breaths().heartbeat((0, 2), every=2)
        roll = s.show(res=16)
        self.assertIn("^", roll)
        self.assertIn("@", roll)
        _, notes = to_bytes(s)
        self.assertTrue(any("lidské zvuky" in n for n in notes))


class TestBreathRules(unittest.TestCase):
    def _texts(self, s):
        return [w.text for w in model_warnings(s)]

    def test_auto_breaths_are_clean(self):
        s, _, h = TestHumanTrack()._song()
        h.breaths()
        bad = [w for w in model_warnings(s) if w.where == "dech"]
        self.assertEqual(bad, [])

    def test_breath_over_notes(self):
        s, _, h = TestHumanTrack()._song()
        h.breath(to_bar=5)
        self.assertTrue(any("přes noty" in t for t in self._texts(s)))

    def test_section_skips_overlap(self):
        s, lead, _ = TestHumanTrack()._song()
        s.human("sbor_dech", performer=lead, section=True, mic=False).breath(to_bar=5)
        self.assertFalse(any("přes noty" in t for t in self._texts(s)))

    def test_long_phrase(self):
        s = Song(bpm=60, bars=8)
        lead = s.track("cello_solo", name="lead")
        lead.notes("D4:4 | F4:4 | E4:4 | D4:4 | A4:4", bar=0)
        s.human(performer=lead)
        self.assertTrue(any("bez pauzy na nádech" in t for t in self._texts(s)))
        self.assertGreater(air_seconds(40), air_seconds(127))

    def test_repeated_variant(self):
        s, _, h = TestHumanTrack()._song()
        h.breath(to_bar=1, variant=2).breath(to_bar=4, variant=2)
        self.assertTrue(any("kulometný" in t for t in self._texts(s)))

    def test_grid_info(self):
        s, _, h = TestHumanTrack()._song()
        for bar in (1, 4, 6, 7):
            h.breath(to_bar=bar)
        self.assertTrue(any("stejný odstup od mřížky" in t for t in self._texts(s)))

    def test_loud_breath_info(self):
        s, _, h = TestHumanTrack()._song()
        h.vol = 4.0
        h.breath(to_bar=4)
        out = render_song(s, None, report=False)
        warns = warnings_for(s, out.audio, out.tracks, out.raw_peak)
        self.assertTrue(any("pod frází" in w.text for w in warns))

    def test_phrases_split_on_rests(self):
        s, lead, _ = TestHumanTrack()._song()
        phrases = breath_phrases(lead.all_notes(), 0.3)
        self.assertEqual([round(p[0], 3) for p in phrases], [0.0, 4.0, 8.0, 16.0])


if __name__ == "__main__":
    unittest.main()
