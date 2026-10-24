import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Song, foley, gait  # noqa: E402
from paprika.analyze import step_stats  # noqa: E402
from paprika.render import render_song  # noqa: E402

SR = 44100


def _step(surface="stone", speed=1.4, weight=75.0, shoe="boot", gear=None, v=0, foot=0):
    return foley.footstep(SR, surface, foot, weight, speed, 1.0, 1.0, 1.0, 1.0, 1.0,
                          shoe, gear, v)


def _spec(y):
    m = np.asarray(y, dtype=np.float64).mean(axis=1)
    p = np.abs(np.fft.rfft(m)) ** 2
    return p, np.fft.rfftfreq(m.size, 1.0 / SR)


def _centroid(y) -> float:
    p, f = _spec(y)
    return float((p * f).sum() / p.sum())


def _share(y, lo, hi=None) -> float:
    p, f = _spec(y)
    sel = (f >= lo) & ((f < hi) if hi else True)
    return float(p[sel].sum() / p.sum())


def _ring(y, db=-30.0) -> float:
    """Jak dlouho zvuk trvá nad `db` vůči špičce (s)."""
    e = np.convolve(np.abs(y).mean(axis=1), np.ones(220) / 220, "same")
    idx = np.where(e > e.max() * 10 ** (db / 20.0))[0]
    return (idx[-1] - idx[0]) / SR


def _med(fn, surface, k=8, **kw):
    return float(np.median([fn(_step(surface, v=v, **kw)) for v in range(k)]))


class TestPhysics(unittest.TestCase):
    def test_pulse_gain(self):
        self.assertAlmostEqual(float(foley.pulse_gain(0.0, 0.001)), 1.0)
        # na f = 1/(2τ) je limita π/4, žádné dělení nulou
        self.assertAlmostEqual(float(foley.pulse_gain(500.0, 0.001)), np.pi / 4, places=6)
        g = foley.pulse_gain(np.array([100.0, 1000.0, 4000.0]), 0.002)
        self.assertGreater(g[0], g[1])
        self.assertGreater(g[1], g[2])

    def test_soft_contact_is_longer(self):
        hard = foley.contact_ms("boot", foley.SURFACES["stone"], 1.4)
        soft = foley.contact_ms("bare", foley.SURFACES["mud"], 1.4)
        self.assertAlmostEqual(hard, 0.5)
        self.assertGreater(soft, 30 * hard)
        self.assertLess(foley.contact_ms("boot", foley.SURFACES["stone"], 3.5), hard)

    def test_grf_shapes(self):
        t = np.arange(int(0.7 * SR)) / SR
        walk = foley.grf(t, 0.56, False)
        run = foley.grf(t, 0.27, True)
        self.assertAlmostEqual(float(walk.max()), 1.0)
        self.assertEqual(float(walk[t > 0.57].max()), 0.0)
        # chůze: dva vrcholy s údolím, běh: jeden vrchol (+ ráz dopadu)
        mid = walk[(t > 0.25) & (t < 0.31)].min()
        self.assertLess(mid, walk[(t > 0.1) & (t < 0.18)].max())
        self.assertLess(mid, walk[(t > 0.38) & (t < 0.46)].max())
        body = run[t > 0.05]
        peak = int(np.argmax(body))
        self.assertTrue(np.all(np.diff(body[peak:peak + int(0.05 * SR)]) <= 1e-9))


class TestSurfaces(unittest.TestCase):
    def test_bounded_and_deterministic(self):
        for name in foley.SURFACES:
            for shoe in foley.SHOES:
                y = _step(name, shoe=shoe)
                self.assertEqual(y.shape[1], 2)
                self.assertTrue(np.all(np.isfinite(y)), (name, shoe))
                self.assertLessEqual(float(np.abs(y).max()), foley.CEILING + 1e-9)
                self.assertGreater(float(np.abs(y).max()), 0.02, (name, shoe))
            np.testing.assert_array_equal(_step(name, v=3), _step(name, v=3))

    def test_variants_differ(self):
        for name in foley.SURFACES:
            a, b = _step(name, v=1), _step(name, v=2)
            m = min(len(a), len(b))
            self.assertFalse(np.allclose(a[:m], b[:m]), name)

    def test_surfaces_differ_spectrally(self):
        names = list(foley.SURFACES)
        shapes = {}
        for n in names:
            p, f = _spec(_step(n))
            edges = np.geomspace(40, 16000, 17)
            bands = np.array([p[(f >= a) & (f < b)].sum() for a, b in zip(edges, edges[1:])])
            shapes[n] = 10 * np.log10(bands / bands.sum() + 1e-12)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = float(np.sqrt(np.mean((shapes[a] - shapes[b]) ** 2)))
                self.assertGreater(d, 2.0, (a, b))

    def test_character(self):
        cent = {n: _med(_centroid, n) for n in foley.SURFACES}
        # sníh a tráva jsou jasné (úlomky), bláto a hlína tmavé (dupnutí)
        self.assertGreater(cent["snow"], 2000)
        self.assertGreater(cent["grass"], cent["dirt"] * 3)
        self.assertGreater(cent["stone"], cent["dirt"] * 2)
        self.assertLess(cent["mud"], 300)
        self.assertGreater(_med(lambda y: _share(y, 4000), "grass"), 0.1)
        # kov zvoní nejdéle, bláto má po odlepení sání
        ring = {n: _med(_ring, n) for n in foley.SURFACES}
        self.assertEqual(max(ring, key=ring.get), "metal")
        self.assertGreater(ring["mud"], ring["stone"] + 0.08)

    def test_shoes(self):
        bare = _med(lambda y: float(np.abs(y).max()), "wood", shoe="bare")
        heels = _med(lambda y: float(np.abs(y).max()), "wood", shoe="heels")
        self.assertGreater(heels, 3 * bare)
        self.assertGreater(_med(_centroid, "stone", shoe="heels"),
                           _med(_centroid, "stone", shoe="bare") * 1.5)

    def test_gear_is_centered_and_bright(self):
        for g in foley.GEARS:
            with_g, without = _step("dirt", gear=g, v=4), _step("dirt", v=4)
            # stejná noha, výbava navíc: rozdíl je v obou kanálech stejný
            m = min(len(with_g), len(without))
            diff = with_g[:m] - without[:m]
            self.assertGreater(float(np.abs(diff).max()), 0.01, g)
        keys = _med(lambda y: _share(y, 2500), "dirt", gear="keys")
        self.assertGreater(keys, 3 * _med(lambda y: _share(y, 2500), "dirt"))

    def test_heavier_runner_louder(self):
        for name in ("stone", "grass", "snow"):
            walk = _med(lambda y: float(np.abs(y).max()), name)
            self.assertGreater(_med(lambda y: float(np.abs(y).max()), name, speed=3.2),
                               walk * 1.2, name)
            self.assertGreater(_med(lambda y: float(np.abs(y).max()), name, weight=110),
                               walk * 1.1, name)


class TestBlend(unittest.TestCase):
    def test_endpoints_are_pure(self):
        np.testing.assert_array_equal(_step(("stone", "grass", 0.0), v=5), _step("stone", v=5))
        np.testing.assert_array_equal(_step(("stone", "grass", 1.0), v=5), _step("grass", v=5))

    def test_middle_has_both(self):
        mid = _step(("stone", "snow", 0.5), v=5)
        self.assertGreater(_centroid(mid), _centroid(_step("stone", v=5)))
        self.assertLess(_centroid(mid), _centroid(_step("snow", v=5)))

    def test_path(self):
        s = Song(bpm=120, bars=8, sample_rate=SR)
        h = s.human("telo", mic=False)
        h.steps((0, 8), pace="walk", surface=[(0, "stone"), (4, "grass")], blend=2.0)
        surf = [x.params[0] for x in h.hits]
        self.assertEqual(surf[0], "stone")
        self.assertEqual(surf[-1], "grass")
        mixed = [x for x in h.hits if isinstance(x.params[0], tuple)]
        self.assertTrue(mixed)
        self.assertTrue(all(15.0 - 1e-9 <= x.start_beat <= 17.5 for x in mixed))
        ws = [x.params[0][2] for x in mixed]
        self.assertEqual(ws, sorted(ws))


class TestTrack(unittest.TestCase):
    def _song(self, **kw):
        s = Song(bpm=120, bars=4, sample_rate=SR)
        h = s.human("telo", mic=False, seed=2)
        h.steps((0, 4), **kw)
        return s, h

    def test_new_surface_walks(self):
        _, h = self._song(surface="snow", every=1)
        self.assertEqual(len(h.hits), 16)
        self.assertTrue(all(len(x.params) == 12 for x in h.hits))
        self.assertGreater(gait.irregularity([x.start_beat for x in h.hits]), 0.005)

    def test_shoe_and_gear_in_params(self):
        _, h = self._song(pace="walk", shoe="sneaker", gear="keys")
        self.assertTrue(all(x.params[9:11] == ("sneaker", "keys") for x in h.hits))

    def test_cloth_hits_in_swing(self):
        _, h = self._song(pace="brisk", cloth="nylon")
        steps = [x for x in h.hits if x.name == "step"]
        cloth = [x for x in h.hits if x.name == "cloth"]
        self.assertEqual(len(cloth), len(steps))
        for st, nxt, c in zip(steps, steps[1:], cloth):
            self.assertGreater(c.start_beat, st.start_beat)
            self.assertLess(c.start_beat + c.dur_beats, nxt.start_beat + 0.05)
            self.assertEqual(c.params[0], "nylon")
        self.assertEqual(len({c.params for c in cloth}), len(cloth))

    def test_walk_continues_into_run(self):
        s = Song(bpm=120, bars=8, sample_rate=SR)
        h = s.human("telo", mic=False, seed=7)
        h.steps((0, 4), pace="walk").steps((4, 8), pace="run", surface="snow")
        b = [x.start_beat for x in h.hits]
        feet = [x.params[1] for x in h.hits]
        self.assertTrue(all(f != g for f, g in zip(feet, feet[1:])))
        self.assertGreater(min(np.diff(b)), 0.5)          # žádné zakopnutí na taktu 4
        self.assertLess(gait.irregularity(b), 0.05)

    def test_walk_after_pause_starts_fresh(self):
        h = Song(bpm=120, bars=8, sample_rate=SR).human("telo", mic=False, seed=7)
        h.steps((0, 2), pace="walk").steps((6, 8), pace="walk")
        second = [x for x in h.hits if x.start_beat > 20]
        self.assertAlmostEqual(second[0].start_beat, 24.0, delta=0.3)
        self.assertEqual(second[0].params[1], 0)

    def test_bad_args(self):
        h = Song(bpm=120, bars=4).human("telo")
        for kw in ({"surface": "ice"}, {"surface": [(0, "stone"), (2, "lava")]},
                   {"surface": [], "pace": "walk"}, {"shoe": "clogs"},
                   {"gear": "sword"}, {"cloth": "silk"}, {"blend": -1, "pace": "walk"}):
            with self.assertRaises(ValueError, msg=kw):
                h.steps((0, 2), **kw)

    def test_report_no_copies_on_any_surface(self):
        s = Song(bpm=120, bars=4, sample_rate=SR)
        for name in foley.SURFACES:
            s.human(name, mic=False, seed=3).steps((0, 4), pace="walk", surface=name)
        r = render_song(s, report=False, print_report=False)
        for name, st in step_stats(s, r.tracks).items():
            self.assertEqual(st["repeats"], 0, name)
            # kov: rezonance plechu jsou pevné, mění se jen jejich amplitudy
            self.assertLess(st["sim_max"], 0.95 if name == "metal" else 0.8, name)


if __name__ == "__main__":
    unittest.main()
