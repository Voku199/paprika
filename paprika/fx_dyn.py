"""Dynamické efekty: Compressor a rytmický Gate.

Obojí počítá křivku zisku vektorově (detektor přes jednopólový filtr,
žádný cyklus po samplech) a násobí jí buffer. Stereo je linkované — oba
kanály dostanou stejný zisk, takže se stereo obraz nerozjede.
"""

from __future__ import annotations

import math

import numpy as np

from . import dsp
from .fx import Fx, FxContext, _col

__all__ = ["Compressor", "Gate"]


class Compressor(Fx):
    """Feed-forward kompresor se soft knee.

    `threshold` v dBFS, `ratio` (4 = nad prahem jen čtvrtina nárůstu),
    `attack`/`release` v sekundách, `makeup` v dB. Na busu bicích drží
    groove pohromadě, na masteru „lepí“ mix.
    """

    STEREO = True

    AUTOMATABLE = ("threshold",)

    def __init__(self, threshold: float = -18.0, ratio: float = 4.0,
                 attack: float = 0.01, release: float = 0.15, makeup: float = 0.0,
                 knee: float = 6.0, name: str | None = None):
        self.threshold = float(threshold)
        self.ratio = max(1.0, float(ratio))
        self.attack = max(1e-4, float(attack))
        self.release = max(1e-3, float(release))
        self.makeup = float(makeup)
        self.knee = max(0.0, float(knee))
        self.name = name

    def gain(self, buf: np.ndarray, sr: int) -> np.ndarray:
        """Křivka zisku (lineární) pro každý sample."""
        det = np.abs(buf).max(axis=1) if buf.ndim == 2 else np.abs(buf)
        env = dsp.onepole(det, 1.0 / (2.0 * math.pi * self.release), sr)
        level = 20.0 * np.log10(np.maximum(env.astype(np.float64), 1e-9))
        over = level - np.asarray(self.value("threshold"), dtype=np.float64)
        k = self.knee
        if k > 0:
            over_k = np.where(over <= -k / 2, 0.0,
                              np.where(over >= k / 2, over, (over + k / 2) ** 2 / (2 * k)))
        else:
            over_k = np.maximum(over, 0.0)
        gain_db = -over_k * (1.0 - 1.0 / self.ratio) + self.makeup
        g = 10.0 ** (gain_db / 20.0)
        return dsp.onepole(g, 1.0 / (2.0 * math.pi * self.attack), sr)

    def process(self, buf, ctx: FxContext):
        buf = np.asarray(buf, dtype=np.float32)
        if buf.shape[0] == 0:
            return buf
        return (buf * _col(self.gain(buf, ctx.sr), buf)).astype(np.float32)


class Gate(Fx):
    """Rytmický gate (trance gate / stutter) synchronní s tempem.

    `pattern`: `x` otevřeno, `.` nebo `_` zavřeno; opakuje se po krocích
    `rate` (zlomek celé noty). `depth` 1 = úplné ticho, 0.5 = na půl.
    Počítá se od začátku skladby, takže sedí i při renderu výřezu.
    """

    STEREO = True

    OPEN = "xX"
    CLOSED = "._"
    AUTOMATABLE = ("depth",)

    def __init__(self, pattern: str = "x.x.xx..", rate: float = 1 / 16,
                 depth: float = 1.0, smooth: float = 0.004, name: str | None = None):
        bad = [c for c in pattern if c not in self.OPEN + self.CLOSED]
        if bad or not pattern:
            raise ValueError(f"Gate pattern {pattern!r}: povolené znaky jsou "
                             f"'x' (otevřeno) a '.' nebo '_' (zavřeno)")
        self.pattern = pattern
        self.rate = float(rate)
        self.depth = float(np.clip(depth, 0.0, 1.0))
        self.smooth = max(1e-4, float(smooth))
        self.name = name

    def process(self, buf, ctx: FxContext):
        buf = np.asarray(buf, dtype=np.float32)
        n = buf.shape[0]
        if n == 0:
            return buf
        step = self.rate * ctx.beats_per_bar
        beats = ctx.beat_offset + np.arange(n, dtype=np.float64) / ctx.sr * (ctx.bpm / 60.0)
        mask = np.array([1.0 if c in self.OPEN else 0.0 for c in self.pattern])
        idx = np.floor(beats / step + 1e-9).astype(np.int64) % len(self.pattern)
        opened = dsp.onepole(mask[idx], 1.0 / (2.0 * math.pi * self.smooth), ctx.sr)
        depth = np.asarray(self.value("depth"), dtype=np.float32)
        gain = 1.0 - depth * (1.0 - opened)
        return (buf * _col(gain, buf)).astype(np.float32)
