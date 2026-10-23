"""Skutečné vzorky: každý nástroj zní, nekřičí a ladí. Jen s PAPRIKA_SLOW=1."""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import library  # noqa: E402
from paprika.sampler import clear_cache  # noqa: E402

SR = 44100
SLOW = os.environ.get("PAPRIKA_SLOW") == "1"
PITCHED = ("piano", "strings", "winds", "brass")


def f0(x, sr=SR, fmin=40.0, fmax=2000.0):
    """Základní frekvence autokorelací (YIN-lite) — odolné proti oktávám."""
    x = x - x.mean()
    n = x.size
    size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(x, size)
    acf = np.fft.irfft(spec * np.conj(spec), size)[: int(sr / fmin) + 1]
    energy = np.concatenate([[0.0], np.cumsum(x ** 2)])
    taus = np.arange(acf.size)
    d = energy[n] + (energy[n] - energy[taus]) - 2 * acf
    d[0] = 0.0
    cmnd = np.ones_like(d)
    cmnd[1:] = d[1:] * taus[1:] / np.maximum(np.cumsum(d[1:]), 1e-12)
    lo = int(sr / fmax)
    below = np.nonzero(cmnd[lo:] < 0.15)[0]
    t = lo + (int(below[0]) if below.size else int(np.argmin(cmnd[lo:])))
    while t + 1 < cmnd.size and cmnd[t + 1] < cmnd[t]:
        t += 1
    return sr / t


@unittest.skipUnless(SLOW, "PAPRIKA_SLOW=1 zapne test skutečných vzorků")
class TestRealSamples(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        clear_cache()

    def test_every_instrument(self):
        tested = 0
        for inst in library.INSTRUMENTS.values():
            for art in inst.arts:
                name = f"{inst.name}.{art}"
                if not library.is_installed(name):
                    continue            # skupiny jdou instalovat zvlášť
                tested += 1
                with self.subTest(name=name):
                    s = library.instrument(name)
                    lo, hi = s.range
                    pitch = (lo + hi) // 2
                    r = s.select(pitch, 100)[0]
                    v = s.voice(r, pitch, 1.0, SR) * s.gain(r, 100)
                    peak = float(np.abs(v).max())
                    self.assertLess(peak, 1.0, name)
                    self.assertGreater(peak, 0.01, name)
                    if inst.group in PITCHED and not inst.is_one_shot(art):
                        seg = v[int(0.3 * SR):int(0.5 * SR)].mean(axis=1)
                        got = 69 + 12 * math.log2(f0(seg) / 440.0)
                        self.assertLess(abs(got - pitch), 0.6,
                                        f"{name}: {got:.2f} vs {pitch}")
        if not tested:
            self.skipTest("žádné vzorky nejsou nainstalované")


if __name__ == "__main__":
    unittest.main()
