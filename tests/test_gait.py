import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Song, foley, gait, metrics, model_warnings  # noqa: E402
from paprika.analyze import check_report, step_stats, warnings_for  # noqa: E402
from paprika.render import render_song  # noqa: E402
from paprika.human import CEILING  # noqa: E402
from paprika.sfx import _CACHE, render_sfx  # noqa: E402

SR = 22050
WALK = lambda b: 1.4  # noqa: E731


def _centroid(y) -> float:
    m = np.asarray(y, dtype=np.float64)[:, 0]
    p = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(m.size, 1.0 / SR)
    return float((p * f).sum() / p.sum())


def _steps(**kw):
    s = Song(bpm=120, bars=16, sample_rate=SR)
    h = s.human("telo", mic=False, seed=kw.pop("seed", 3))
    if not {"pace", "speed", "weight", "jitter"} & set(kw):
        kw["pace"] = "walk"
    h.steps(kw.pop("bars", (0, 16)), **kw)
    return s, h, [x for x in h.hits if x.name == "step"]


def _log(steps, i):
    return np.log([x.params[i] for x in steps])


class TestPlan(unittest.TestCase):
    def test_cadence_follows_speed(self):
        self.assertAlmostEqual(gait.cadence(1.4), 1.9, places=6)
        self.assertGreater(gait.cadence(3.5), 2.8)
        plan = gait.plan(0.0, 64.0, 120, WALK, jitter=0.0)
        d = np.diff([p.beat for p in plan[1:]])
        # 1.9 kroku/s při 120 BPM = interval 1.05 beatu (stálý posun nohou zůstane)
        self.assertAlmostEqual(float(np.mean(d)), 2.0 / 1.9, delta=0.01)

    def test_jitter_is_interval_cv(self):
        for jit in (0.01, 0.025, 0.05):
            plan = gait.plan(0.0, 400.0, 120, WALK, jitter=jit, seed=1)
            cv = gait.interval_cv([p.beat for p in plan])
            self.assertAlmostEqual(cv, jit, delta=0.006, msg=jit)

    def test_steps_stay_on_grid(self):
        # 1/f odchylka se měří od mřížky: chodec po 400 beatech neuteče
        plan = gait.plan(0.0, 400.0, 120, WALK, seed=5)
        grid = np.arange(len(plan)) * (2.0 / 1.9)
        err = np.abs(np.array([p.beat for p in plan]) - grid)
        self.assertLess(float(err[1:].max()), gait.MAX_SHIFT * 1.06 + 0.02)

    def test_pink_has_long_memory(self):
        x = gait.pink(4096, np.random.default_rng(0))
        self.assertAlmostEqual(float(x.std()), 1.0, places=6)
        p = np.abs(np.fft.rfft(x)) ** 2
        lo, hi = p[1:64].mean(), p[1024:].mean()
        self.assertGreater(lo, 10.0 * hi)            # víc energie v pomalých změnách

    def test_params_correlated_not_identical(self):
        _, _, st = _steps()
        lv = _log(st, 4)
        lag1 = float(np.corrcoef(lv[:-1], lv[1:])[0, 1])
        self.assertGreater(lag1, 0.25)
        self.assertLess(lag1, 0.9)
        self.assertEqual(len({x.params for x in st}), len(st))

    def test_feet_differ_persistently(self):
        w3, w4 = gait.Walker(3), gait.Walker(4)
        self.assertNotAlmostEqual(w3.factor(0, "level"), w3.factor(1, "level"))
        self.assertAlmostEqual(w3.factor(0, "level") * w3.factor(1, "level"), 1.0)
        # stejný šum, jiný chodec: poměr síly je u každé nohy po celou dobu stejný
        a = gait.plan(0.0, 64.0, 120, WALK, walker=w3, seed=1)
        b = gait.plan(0.0, 64.0, 120, WALK, walker=w4, seed=1)
        for foot in (0, 1):
            ratio = [p.level / q.level for p, q in zip(a, b) if p.foot == foot]
            want = w3.factor(foot, "level") / w4.factor(foot, "level")
            np.testing.assert_allclose(ratio, want, rtol=1e-9)
        # rozdíl nohou mezi chodci je řádově BIAS (ne nula, ne obří)
        diff = [np.log(gait.Walker(k).factor(0, "level") / gait.Walker(k).factor(1, "level"))
                for k in range(200)]
        self.assertAlmostEqual(float(np.std(diff)), gait.BIAS["level"], delta=0.015)
        # dvě volání steps() na jedné stopě: stejný chodec, jiný šum
        s = Song(bpm=120, bars=8, sample_rate=SR)
        h = s.human("telo", mic=False, seed=3)
        h.steps((0, 4), pace="walk").steps((4, 8), pace="walk")
        first = [x for x in h.hits if x.start_beat < 16]
        second = [x for x in h.hits if x.start_beat >= 16]
        self.assertNotEqual(first[0].params[4:], second[0].params[4:])

    def test_deterministic(self):
        a, b = _steps()[2], _steps()[2]
        self.assertEqual([(x.start_beat, x.params) for x in a],
                         [(x.start_beat, x.params) for x in b])
        c = _steps(seed=4)[2]
        self.assertNotEqual([x.params for x in a], [x.params for x in c])

    def test_speed_ramp_densifies(self):
        _, _, st = _steps(bars=(0, 8), speed=[(0, 1.0), (8, 4.0)])
        d = np.diff([x.start_beat for x in st])
        self.assertGreater(float(d[:3].mean()), 1.4 * float(d[-3:].mean()))
        self.assertLess(st[0].params[3], st[-1].params[3])

    def test_every_overrides_cadence(self):
        _, _, st = _steps(bars=(0, 4), every=1, pace="run", jitter=0.0)
        self.assertEqual(len(st), 16)
        self.assertEqual(st[0].params[3], gait.PACES["run"])

    def test_physics(self):
        walk, run = gait.physics(75, 1.4), gait.physics(75, 3.5)
        self.assertFalse(walk.run)
        self.assertTrue(run.run)
        self.assertAlmostEqual(walk.level, 1.0)
        self.assertGreater(run.level, 1.4)
        self.assertLess(run.contact, walk.contact)
        self.assertLess(run.gap, walk.gap)
        heavy = gait.physics(110, 1.4)
        self.assertGreater(heavy.level, walk.level)
        self.assertLess(heavy.pitch, walk.pitch)

    def test_bad_args(self):
        s = Song(bpm=120, bars=4)
        h = s.human("telo")
        for kw in ({"pace": "sprint"}, {"pace": "walk", "speed": 2.0},
                   {"speed": 20.0}, {"speed": "fast"}, {"weight": 5},
                   {"jitter": 0.5}, {"every": 0, "pace": "walk"}):
            with self.assertRaises(ValueError, msg=kw):
                h.steps((0, 2), **kw)


class TestStepAudio(unittest.TestCase):
    def setUp(self):
        _CACHE.clear()

    @staticmethod
    def _one(surface="stone", weight=75.0, speed=1.4, variant=0, foot=0):
        return render_sfx("step", (surface, foot, weight, speed, 1.0, 1.0, 1.0, 1.0,
                                   1.0, "boot", None, variant), SR, 120)

    def test_legacy_step_unchanged(self):
        # pevný krok bez plánu chůze musí zůstat bit-identický se starými songy
        sums = {"stone": 531.3258435211235, "wood": 1251.12365682602,
                "gravel": 833.8897426070496, "dirt": 1536.0396043644114}
        for surface, ref in sums.items():
            y = render_sfx("step", (surface, 1, 3), 44100, 120).astype(np.float64)
            self.assertAlmostEqual(float(np.abs(y).sum()), ref, places=3, msg=surface)
        s = Song(bpm=120, bars=2)
        h = s.human("telo").steps((0, 1))
        self.assertTrue(all(len(x.params) == 3 for x in h.hits))

    def test_bounded_on_all_surfaces(self):
        for surface in foley.SURFACES:
            for weight, speed in ((40, 0.9), (75, 1.4), (120, 3.5), (250, 8.0)):
                y = self._one(surface, weight, speed)
                self.assertTrue(np.all(np.isfinite(y)))
                self.assertLessEqual(float(np.abs(y).max()), CEILING + 1e-6)
                self.assertGreater(float(np.abs(y).max()), 0.05)

    def test_reference_level(self):
        for surface, target in foley.SURFACE_TARGET.items():
            peaks = [float(np.abs(render_sfx("step", (surface, v % 2, 75.0, 1.4, 1.0, 1.0,
                                                      1.0, 1.0, 1.0, "boot", None, v),
                                             44100, 120)).max())
                     for v in range(30)]
            self.assertAlmostEqual(float(np.median(peaks)), target, delta=0.06, msg=surface)

    def test_heavier_and_faster_louder(self):
        def med(**kw):
            return float(np.median([np.abs(self._one(variant=v, **kw)).max()
                                    for v in range(20)]))
        ref = med()
        self.assertGreater(med(weight=110), ref * 1.15)
        self.assertGreater(med(speed=3.2), ref * 1.3)

    def test_run_is_brighter_than_slow_walk(self):
        slow = np.mean([_centroid(self._one(speed=0.9, variant=v)) for v in range(10)])
        run = np.mean([_centroid(self._one(speed=3.5, variant=v)) for v in range(10)])
        self.assertGreater(run, slow * 1.1)

    def test_bad_params(self):
        for params in (("stone", 0, 10.0, 1.4, 1, 1, 1, 1, 1, "boot", None, 0),
                       ("stone", 0, 75.0, 20.0, 1, 1, 1, 1, 1, "boot", None, 0),
                       ("ice", 0, 75.0, 1.4, 1, 1, 1, 1, 1, "boot", None, 0),
                       ("stone", 0, 75.0, 1.4, 1, 1, 1, 1, 1, "clogs", None, 0),
                       ("stone", 0, 75.0, 1.4, 1, 1, 1, 1, 1, "boot", "sword", 0),
                       (("stone", "mud", 1.5), 0, 75.0, 1.4, 1, 1, 1, 1, 1, "boot", None, 0),
                       ("stone", 0, 75.0, 1.4, 1, 1, 1, 1, 1, 0),
                       ("stone", 0, 75.0, 1.4, 0)):
            with self.assertRaises(ValueError, msg=params):
                render_sfx("step", params, SR, 120)


class TestIrregularity(unittest.TestCase):
    def test_matches_jitter(self):
        for jit in (0.01, 0.025, 0.05):
            plan = gait.plan(0.0, 400.0, 120, WALK, jitter=jit, seed=2)
            self.assertAlmostEqual(gait.irregularity([p.beat for p in plan]), jit,
                                   delta=0.006 + 0.3 * jit, msg=jit)

    def test_ignores_ramp_and_stops(self):
        ramp = gait.plan(0.0, 60.0, 120, lambda b: 1.0 + b / 20.0, jitter=0.0,
                         walker=gait.Walker(0))
        self.assertGreater(gait.interval_cv([p.beat for p in ramp]), 0.1)
        self.assertLess(gait.irregularity([p.beat for p in ramp]), 0.02)
        two = list(range(10)) + [30 + i for i in range(10)]
        self.assertEqual(len(gait.walks(two)), 2)
        self.assertAlmostEqual(gait.irregularity(two), 0.0, places=9)

    def test_too_few(self):
        self.assertIsNone(gait.irregularity([0, 1, 2, 3]))


class TestStepMetrics(unittest.TestCase):
    @staticmethod
    def _burst(rng, n=4000):
        t = np.arange(n) / SR
        return (0.5 * np.sin(2 * np.pi * 80 * t) + 0.3 * rng.standard_normal(n)) * np.exp(-t / 0.03)

    def _track(self, bursts, gap=0.5):
        step = int(gap * SR)
        buf = np.zeros(step * (len(bursts) + 1))
        for i, b in enumerate(bursts):
            buf[i * step:i * step + b.size] += b
        return buf, [i * step for i in range(len(bursts))]

    def test_copies_detected(self):
        rng = np.random.default_rng(1)
        one = self._burst(rng)
        buf, on = self._track([one * g for g in (1.0, 0.7, 1.2, 0.9, 1.0, 0.8)])
        st = metrics.step_stats(buf, SR, on)
        self.assertEqual(st["repeats"], 5)
        self.assertGreater(st["sim_max"], 0.99)
        self.assertEqual(st["repeat_of"][1], 0)

    def test_same_thud_different_noise_is_not_copy(self):
        # tlumený sinus je u všech stejný, náhodná část ne -> žádná kopie
        rng = np.random.default_rng(2)
        buf, on = self._track([self._burst(rng) for _ in range(8)])
        st = metrics.step_stats(buf, SR, on)
        self.assertEqual(st["repeats"], 0)
        self.assertLess(st["sim_max"], 0.5)
        self.assertGreater(st["level_sd"], 0.0)

    def test_silence_and_few(self):
        self.assertIsNone(metrics.step_stats(np.zeros(SR), SR, [0, 100, 200, 300]))
        rng = np.random.default_rng(3)
        buf, on = self._track([self._burst(rng), self._burst(rng)])
        self.assertIsNone(metrics.step_stats(buf, SR, on))


class TestReport(unittest.TestCase):
    def test_gait_steps_not_robotic(self):
        s, _, _ = _steps(bars=(0, 4), pace="walk")
        texts = [w.text for w in model_warnings(s)]
        self.assertFalse(any("roboticky" in t for t in texts), texts)

    @staticmethod
    def _texts(warns, name="telo"):
        return [w.text for w in warns if w.where == name]

    def test_legacy_steps_flagged_without_render(self):
        s = Song(bpm=120, bars=8, sample_rate=SR)
        s.human("telo", mic=False).steps((0, 8), every=1)
        texts = self._texts(model_warnings(s))
        self.assertTrue(any("metronom" in t for t in texts), texts)
        self.assertTrue(any("stejný vzorek jako dřívější" in t and "24 z 32" in t
                            for t in texts), texts)
        self.assertIn("KROKY    telo: 32 kroků   nepravidelnost 0.0 %   stejný vzorek 24/32",
                      check_report(s))

    def test_gait_steps_quiet(self):
        s, _, _ = _steps(bars=(0, 8), pace="walk")
        texts = self._texts(model_warnings(s))
        self.assertFalse(any("metronom" in t or "vzorek" in t for t in texts), texts)

    def test_report_line_and_audio_rules(self):
        s = Song(bpm=120, bars=6, sample_rate=SR)
        s.human("stary", mic=False, seed=1).steps((0, 6), every=1)
        s.human("chuze", mic=False, seed=1).steps((0, 6), pace="walk")
        r = render_song(s, report=False, print_report=False)
        st = step_stats(s, r.tracks)
        self.assertEqual(st["stary"]["repeats"], 16)       # 24 kroků, 8 variant
        self.assertEqual(st["chuze"]["repeats"], 0)
        self.assertLess(st["stary"]["level_sd"], 0.5)
        self.assertGreater(st["chuze"]["level_sd"], 0.8)
        warns = warnings_for(s, r.audio, r.tracks, r.raw_peak)
        self.assertTrue(any("stejně silné" in t for t in self._texts(warns, "stary")))
        self.assertFalse(self._texts(warns, "chuze"), self._texts(warns, "chuze"))
        r = render_song(s, print_report=False)
        self.assertIn("KROKY    stary: 24 kroků", r.report)
        self.assertIn("kopie 0/", r.report)

    def test_bars_slice_aligns_onsets(self):
        s = Song(bpm=120, bars=8, sample_rate=SR)
        h = s.human("telo", mic=False, seed=2).steps((0, 8), every=1)
        r = render_song(s, bars=(4, 8), report=False, print_report=False)
        st = step_stats(s, r.tracks, (4, 8))["telo"]
        self.assertEqual(st["n"], 16)
        self.assertEqual(st["repeats"], 8)
        # nástupy sedí: okno začíná těsně před úderem, ne v tichu mezi kroky
        first = [x for x in h.hits if x.start_beat >= 16][0]
        self.assertIs(st["hits"][0], first)
        self.assertGreater(min(st["peak_db"]), -40.0)


if __name__ == "__main__":
    unittest.main()
