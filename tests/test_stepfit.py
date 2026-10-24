import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Song, cli, foley, stepfit  # noqa: E402
from paprika.render import render_song, write_wav  # noqa: E402

SR = 44100
# „skutečný“ povrch, který fit nezná: rezonance 310 / 780 / 1150 Hz, dupnutí 95 Hz
TRUTH = foley.Surface(0.9, modes=((310.0, 0.04, 0.6), (780.0, 0.025, 0.5),
                                  (1150.0, 0.018, 0.5)),
                      body=(95.0, 0.03, 0.5), click=(3600.0, 1.0),
                      friction=(3500.0, 1.0, 0.15))


def recording(surf=TRUTH, n=12, seed=0, noise_db=-65.0):
    """Nahrávka chůze: kroky z `surf` s náhodnou hlasitostí a mezerami + šum."""
    rng = np.random.default_rng(seed)
    parts, onsets, pos = [], [], 0
    for k in range(n):
        y, _ = foley._render(SR, surf, k % 2, 75.0, 1.4, rng.uniform(0.7, 1.2), 1, 1, 1, 1,
                             "boot", None, np.random.default_rng(1000 + k))
        gap = np.zeros(int(rng.uniform(0.15, 0.4) * SR))
        parts += [gap, y]
        pos += gap.size
        onsets.append(pos)
        pos += y.size
    x = np.concatenate(parts + [np.zeros(SR // 2)]) * 0.3
    x += 10 ** (noise_db / 20) * np.random.default_rng(seed + 9).standard_normal(x.size)
    return x, onsets


class TestModes(unittest.TestCase):
    def test_matrix_pencil_recovers_damped_sines(self):
        t = np.arange(int(0.4 * SR)) / SR
        x = (0.6 * np.sin(2 * np.pi * 310 * t) * np.exp(-t / 0.04)
             + 0.4 * np.sin(2 * np.pi * 780 * t + 1) * np.exp(-t / 0.025)
             + 0.5 * np.sin(2 * np.pi * 95 * t) * np.exp(-t / 0.03))
        x += 1e-2 * np.random.default_rng(0).standard_normal(x.size)
        found = {round(f): (f, tau) for f, tau, _ in stepfit.modes(x, SR)[:3]}
        for f0, tau0 in ((310, 0.04), (780, 0.025), (95, 0.03)):
            f, tau = min(found.values(), key=lambda m: abs(m[0] - f0))
            self.assertAlmostEqual(f, f0, delta=0.01 * f0)
            self.assertAlmostEqual(tau, tau0, delta=0.1 * tau0)

    def test_silence(self):
        self.assertEqual(stepfit.modes(np.zeros(4000), SR), [])
        self.assertEqual(stepfit.modes(np.ones(10), SR), [])


class TestSegment(unittest.TestCase):
    def test_finds_every_step_once(self):
        x, onsets = recording()
        seg = stepfit.segment(x, SR)
        self.assertEqual(len(seg), len(onsets))      # odval špičky není další krok
        for (a, b), o in zip(seg, onsets):
            self.assertAlmostEqual(a + int(stepfit.PRE * SR), o, delta=int(0.002 * SR))
            self.assertGreater((b - a) / SR, 0.45)    # okno drží i odval špičky

    def test_silence_and_noise(self):
        self.assertEqual(stepfit.segment(np.zeros(SR), SR), [])
        noise = 1e-3 * np.random.default_rng(1).standard_normal(2 * SR)
        self.assertEqual(stepfit.segment(noise, SR), [])

    def test_noise_floor(self):
        x, _ = recording(noise_db=-50.0)
        seg = stepfit.segment(x, SR)
        nz = stepfit.noise_floor(x, SR, seg)
        self.assertIsNotNone(nz)
        # bílý šum −50 dBFS: energie na Σw² ~ výkon v pásmu (rozprostřený do 22 kHz)
        self.assertAlmostEqual(10 * np.log10(nz["b7"].sum()), -50.0, delta=1.5)
        self.assertIsNone(stepfit.noise_floor(x[:SR // 4], SR, [(0, SR // 4)]))


class TestFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x, _ = recording()
        seg = stepfit.segment(x, SR)
        cls.segs = [x[a:b] for a, b in seg]
        cls.noise = stepfit.noise_floor(x, SR, seg)
        cls.rec = stepfit.fit(cls.segs, SR, base="wood", noise=cls.noise)

    def test_recovers_modes_and_body(self):
        s = self.rec["surface"]
        freqs = [m[0] for m in s["modes"]]
        for f0 in (310.0, 780.0, 1150.0):
            self.assertLess(min(abs(f - f0) / f0 for f in freqs), 0.03, (f0, freqs))
        self.assertAlmostEqual(s["body"][0], 95.0, delta=5.0)
        self.assertAlmostEqual(s["body"][1], 0.03, delta=0.005)

    def test_closer_than_base(self):
        err = self.rec["error_db"]
        self.assertLess(err["fit"], 1.5)
        self.assertLess(err["fit"], err["base"] / 3)
        self.assertLess(self.rec["shape_db"]["fit"], 2.0)
        self.assertAlmostEqual(self.rec["ring"]["fit"], self.rec["ring"]["real"], delta=0.05)

    def test_eq_bounded(self):
        eq = self.rec["surface"]["eq"]
        self.assertEqual(len(eq), 24)
        self.assertLessEqual(max(abs(g) for g in eq), stepfit.EQ_MAX)

    def test_too_few_steps(self):
        with self.assertRaises(stepfit.FitError):
            stepfit.fit(self.segs[:3], SR)
        with self.assertRaises(stepfit.FitError):
            stepfit.fit(self.segs, SR, base="lava")

    def test_lines(self):
        text = "\n".join(stepfit.fit_lines("dlazba", self.rec))
        self.assertIn("POVRCH   dlazba", text)
        self.assertIn("chyba pásem", text)
        text = "\n".join(stepfit.compare_lines(self.segs, SR, "stone", noise=self.noise))
        self.assertIn("SROVNÁNÍ", text)


class TestCustomSurface(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"PAPRIKA_SAMPLES": self.tmp.name})
        self.env.start()
        x, _ = recording(n=8)
        self.wav = os.path.join(self.tmp.name, "chuze.wav")
        write_wav(self.wav, np.stack([x, x], axis=1).astype(np.float32), SR)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def _fit(self, name="dlazba"):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["samples", "steps", "fit", self.wav, "--name", name,
                             "--base", "wood", "--license", "CC0"])
        return code, out.getvalue()

    def test_fit_save_and_use(self):
        code, text = self._fit()
        self.assertEqual(code, 0, text)
        self.assertIn('surface="dlazba"', text)
        self.assertIn("dlazba", foley.surface_names())
        surf = foley.lookup("dlazba")
        self.assertEqual(len(surf.eq), 24)
        # zisk kalibrovaný na cíl základu (wood) u referenčního chodce
        peaks = [float(np.abs(foley.footstep(SR, "dlazba", v % 2, 75.0, 1.4, 1, 1, 1, 1, 1,
                                             "boot", None, v)).max()) for v in range(20)]
        self.assertAlmostEqual(float(np.median(peaks)), foley.SURFACE_TARGET["wood"], delta=0.08)
        s = Song(bpm=100, bars=2, sample_rate=SR)
        s.human("telo", mic=False).steps((0, 2), pace="walk",
                                         surface=[(0, "stone"), (1, "dlazba")])
        r = render_song(s, report=False, print_report=False)
        self.assertTrue(np.all(np.isfinite(r.audio)))
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["samples", "steps"]), 0)
            self.assertEqual(cli.main(["samples", "steps", "compare", self.wav,
                                       "--surface", "dlazba"]), 0)
            self.assertEqual(cli.main(["samples", "steps", "cut", self.wav, "--out",
                                       os.path.join(self.tmp.name, "kroky")]), 0)
        text = out.getvalue()
        self.assertIn("licence CC0", text)
        self.assertIn("-> 8 kroků", text)

    def test_names(self):
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), mock.patch.object(sys, "stderr", err):
            self.assertEqual(self._fit("stone")[0], 2)
            self.assertEqual(self._fit("moje dlazba")[0], 2)
        self.assertIn("vestavěný povrch", err.getvalue())

    def test_missing_surface_hint(self):
        with self.assertRaises(foley.MissingSurface) as cm:
            Song(bpm=100, bars=2).human("telo").steps((0, 2), surface="dlazba")
        self.assertIn("samples steps fit", str(cm.exception))


class TestEq(unittest.TestCase):
    def test_flat_eq_is_identity(self):
        y = np.random.default_rng(0).standard_normal(5000)
        np.testing.assert_allclose(foley.apply_eq(y, SR, (0.0,) * 24), y, atol=1e-9)

    def test_eq_changes_band(self):
        y = np.random.default_rng(0).standard_normal(SR)
        eq = [0.0] * 24
        eq[-6:] = [-12.0] * 6
        z = foley.apply_eq(y, SR, eq)
        p, q = np.abs(np.fft.rfft(y)) ** 2, np.abs(np.fft.rfft(z)) ** 2
        f = np.fft.rfftfreq(y.size, 1.0 / SR)
        hi = f > 9000
        self.assertLess(10 * np.log10(q[hi].sum() / p[hi].sum()), -9.0)
        lo = (f > 200) & (f < 1000)
        self.assertAlmostEqual(10 * np.log10(q[lo].sum() / p[lo].sum()), 0.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
