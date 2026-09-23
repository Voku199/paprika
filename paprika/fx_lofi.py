"""Lo-fi efekty: BitCrush a Tape.

Dreamy / lo-fi zvuk (SEAPOOL, lofi keys) stojí na nedokonalosti: hrubé
rozlišení, kolísání rychlosti pásky, saturace a šum.
"""

from __future__ import annotations

import numpy as np

from . import dsp
from .fx import Fx, FxContext, _col
from .fx_space import _delayed

__all__ = ["BitCrush", "Tape"]


class BitCrush(Fx):
    """Snížení bitové hloubky a vzorkovací frekvence (sample & hold)."""

    AUTOMATABLE = ("mix",)

    def __init__(self, bits: int = 8, downsample: int = 4, mix: float = 1.0,
                 name: str | None = None):
        self.bits = max(1, int(bits))
        self.downsample = max(1, int(downsample))
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.name = name

    def process(self, buf, ctx: FxContext):
        buf = np.asarray(buf, dtype=np.float32)
        n = buf.shape[0]
        if n == 0:
            return buf
        held = np.repeat(buf[::self.downsample], self.downsample, axis=0)[:n]
        steps = 2.0 ** (self.bits - 1)
        wet = np.round(held * steps) / steps
        m = _col(self.value("mix"), buf)
        return ((1.0 - m) * buf + m * wet).astype(np.float32)


class Tape(Fx):
    """Páska: pomalé kolísání rychlosti (`wow`), rychlé chvění (`flutter`),
    saturace (`drive`) a šum (`hiss`). Oba kanály kolísají spolu — páska je
    jedna."""

    BASE_MS = 5.0

    def __init__(self, wow: float = 0.3, flutter: float = 0.2, drive: float = 1.5,
                 hiss: float = 0.0, seed: int = 0, name: str | None = None):
        self.wow = float(max(0.0, wow))
        self.flutter = float(max(0.0, flutter))
        self.drive = max(1e-3, float(drive))
        self.hiss = float(max(0.0, hiss))
        self.seed = int(seed)
        self.name = name

    def process(self, buf, ctx: FxContext):
        buf = np.asarray(buf, dtype=np.float32)
        n = buf.shape[0]
        if n == 0:
            return buf
        sr = ctx.sr
        ms = (self.BASE_MS
              + self.wow * 3.0 * dsp.lfo("sine", 0.5, n, sr)
              + self.wow * 1.0 * dsp.lfo("random", 0.3, n, sr, seed=self.seed)
              + self.flutter * 0.3 * dsp.lfo("sine", 7.0, n, sr))
        d = ms * sr / 1000.0
        if buf.ndim == 1:
            wet = _delayed(buf.astype(np.float64), d)
        else:
            wet = np.stack([_delayed(buf[:, c].astype(np.float64), d)
                            for c in range(buf.shape[1])], axis=1)
        wet = np.tanh(wet * self.drive) / np.tanh(self.drive)
        if self.hiss > 0:
            noise = dsp.colored_noise(n, "pink", self.seed + 101) * self.hiss * 0.01
            wet = wet + (noise[:, None] if wet.ndim == 2 else noise)
        return wet.astype(np.float32)
