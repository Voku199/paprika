"""Prostorové efekty: konvoluční Hall, Chorus, Width.

`Hall` konvoluuje stopu se syntetickou stereo impulzní odezvou
(`dsp.synth_ir`) přes FFT — zní výrazně přirozeněji než Freeverb a hodí se
na orchestr, sbor a pady. Odezvy se cachují, stejná místnost se počítá jednou.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import dsp
from .fx import Fx, FxContext, _col

__all__ = ["Hall", "HALL_PRESETS", "Chorus", "Width"]

# preset -> (decay s na -60 dB, predelay s, damp 0..1)
HALL_PRESETS = {
    "room": (0.6, 0.01, 0.6),
    "plate": (1.8, 0.0, 0.3),
    "hall": (2.8, 0.03, 0.5),
    "cathedral": (6.0, 0.06, 0.6),
}

_IR_CACHE: dict = {}


class Hall(Fx):
    """Konvoluční reverb. `preset` room / plate / hall / cathedral, `size`
    násobí dobu doznění, jednotlivé parametry jdou přepsat."""

    AUTOMATABLE = ("mix",)

    def __init__(self, preset: str = "hall", size: float | None = None,
                 decay: float | None = None, predelay: float | None = None,
                 mix: float = 0.3, damp: float | None = None, seed: int = 0,
                 name: str | None = None):
        if preset not in HALL_PRESETS:
            raise KeyError(f"neznámý Hall preset {preset!r}; k dispozici: "
                           f"{', '.join(HALL_PRESETS)}")
        p_decay, p_pre, p_damp = HALL_PRESETS[preset]
        self.preset = preset
        self.decay = float(decay if decay is not None else p_decay) * float(size or 1.0)
        self.predelay = float(predelay if predelay is not None else p_pre)
        self.damp = float(damp if damp is not None else p_damp)
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.seed = int(seed)
        self.name = name

    def tail(self, ctx: FxContext) -> float:
        return self.decay * 1.1 + self.predelay

    def _ir(self, sr: int) -> np.ndarray:
        key = (round(self.decay, 4), round(self.predelay, 4), round(self.damp, 4),
               self.seed, sr)
        if key not in _IR_CACHE:
            _IR_CACHE[key] = dsp.synth_ir(self.decay, sr, self.predelay,
                                          self.damp, early=0.3, seed=self.seed)
        return _IR_CACHE[key]

    def process(self, buf, ctx: FxContext):
        buf = np.asarray(buf, dtype=np.float32)
        mix = self.value("mix")
        n = buf.shape[0]
        if (np.isscalar(mix) and mix <= 0.0) or n == 0:
            return buf
        ir = self._ir(ctx.sr)
        m = _col(mix, buf)
        if buf.ndim == 1:
            wet = signal.fftconvolve(buf, ir[:, 0])[:n]
        else:
            wet = np.stack([signal.fftconvolve(buf[:, c], ir[:, c % 2])[:n]
                            for c in range(buf.shape[1])], axis=1)
        return ((1.0 - m) * buf + m * wet).astype(np.float32)


def _delayed(x: np.ndarray, delay_samples: np.ndarray) -> np.ndarray:
    """x čtené s proměnným zpožděním (lineární interpolace)."""
    idx = np.arange(x.shape[0], dtype=np.float64) - delay_samples
    return np.interp(idx, np.arange(x.shape[0], dtype=np.float64), x, left=0.0)


class Chorus(Fx):
    """Chorus: pár kopií se zpožděním, které pomalu kolísá. Levý a pravý
    kanál kolísají v protifázi, takže se zvuk rozšíří do stran."""

    AUTOMATABLE = ("mix",)
    BASE_MS = 15.0

    def __init__(self, rate: float = 0.8, depth_ms: float = 7.0, mix: float = 0.4,
                 voices: int = 2, name: str | None = None):
        self.rate = float(rate)
        self.depth_ms = float(depth_ms)
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.voices = max(1, int(voices))
        self.name = name

    def _wet(self, x: np.ndarray, sr: int, phase: float) -> np.ndarray:
        n = x.shape[0]
        out = np.zeros(n, dtype=np.float64)
        for v in range(self.voices):
            mod = dsp.lfo("sine", self.rate * (1.0 + 0.13 * v), n, sr,
                          phase=phase + v / self.voices)
            d = (self.BASE_MS + self.depth_ms * 0.5 * (1.0 + mod)) * sr / 1000.0
            out += _delayed(x.astype(np.float64), d)
        return out / self.voices

    def process(self, buf, ctx: FxContext):
        buf = np.asarray(buf, dtype=np.float32)
        if buf.shape[0] == 0:
            return buf
        m = _col(self.value("mix"), buf)
        if buf.ndim == 1:
            wet = self._wet(buf, ctx.sr, 0.0)
        else:
            wet = np.stack([self._wet(buf[:, c], ctx.sr, 0.5 * (c % 2))
                            for c in range(buf.shape[1])], axis=1)
        return ((1.0 - m) * buf + m * wet).astype(np.float32)


class Width(Fx):
    """Stereo šířka přes mid/side: 0 = mono, 1 = beze změny, 2 = extra široké.

    Pozor na víc než ~1.6 — v mono se široký signál začne vyrušovat
    (report to hlásí jako fázový rozpad)."""

    AUTOMATABLE = ("amount",)

    def __init__(self, amount: float = 1.3, name: str | None = None):
        self.amount = float(max(0.0, amount))
        self.name = name

    def process(self, buf, ctx: FxContext):
        buf = np.asarray(buf, dtype=np.float32)
        if buf.ndim == 1:
            return buf
        amt = _col(self.value("amount"), buf[:, 0])
        mid = (buf[:, 0] + buf[:, 1]) * 0.5
        side = (buf[:, 0] - buf[:, 1]) * 0.5 * amt
        return np.stack([mid + side, mid - side], axis=1).astype(np.float32)

