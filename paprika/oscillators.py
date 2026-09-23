"""Oscilátory: jeden periodický (nebo šumový) průběh na dané frekvenci.

`ratio` je poměr k základní frekvenci noty, takže `Sine(2.01)` je mírně
rozladěná oktáva. Vše je float32.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["Osc", "Sine", "Saw", "Square", "Tri", "Noise", "FM"]

TWO_PI = 2.0 * math.pi


class Osc:
    """Základ oscilátoru. `ratio` je poměr k základní frekvenci noty."""

    def __init__(self, ratio: float = 1.0, amp: float = 1.0, phase: float = 0.0):
        self.ratio = float(ratio)
        self.amp = float(amp)
        self.phase = float(phase)

    def _phase_array(self, freq: float, n: int, sr: int) -> np.ndarray:
        t = np.arange(n, dtype=np.float64) / sr
        return (freq * self.ratio) * t + self.phase

    def render(self, freq: float, n: int, sr: int) -> np.ndarray:
        raise NotImplementedError

    def key(self) -> tuple:
        return (type(self).__name__, self.ratio, self.amp, self.phase)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}({self.ratio:g}, amp={self.amp:g})"


class Sine(Osc):
    def render(self, freq, n, sr):
        return np.sin(TWO_PI * self._phase_array(freq, n, sr)).astype(np.float32)


class Saw(Osc):
    """Pilový průběh, band-limited přes sčítání harmonických (bez aliasu)."""

    def render(self, freq, n, sr):
        f = freq * self.ratio
        if f <= 0:
            return np.zeros(n, dtype=np.float32)
        nyq = sr * 0.5
        harmonics = max(1, min(64, int(nyq / f)))
        t = np.arange(n, dtype=np.float64) / sr
        out = np.zeros(n, dtype=np.float64)
        for k in range(1, harmonics + 1):
            out -= np.sin(TWO_PI * k * f * t + self.phase) / k
        out *= 2.0 / math.pi
        return out.astype(np.float32)


class Square(Osc):
    """Obdélník, band-limited (jen liché harmonické)."""

    def __init__(self, ratio=1.0, amp=1.0, phase=0.0, width: float = 0.5):
        super().__init__(ratio, amp, phase)
        self.width = float(width)

    def render(self, freq, n, sr):
        f = freq * self.ratio
        if f <= 0:
            return np.zeros(n, dtype=np.float32)
        nyq = sr * 0.5
        harmonics = max(1, min(64, int(nyq / f)))
        t = np.arange(n, dtype=np.float64) / sr
        out = np.zeros(n, dtype=np.float64)
        for k in range(1, harmonics + 1, 2):
            out += np.sin(TWO_PI * k * f * t + self.phase) / k
        out *= 4.0 / math.pi
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def key(self):
        return super().key() + (self.width,)


class Tri(Osc):
    def render(self, freq, n, sr):
        f = freq * self.ratio
        if f <= 0:
            return np.zeros(n, dtype=np.float32)
        nyq = sr * 0.5
        harmonics = max(1, min(32, int(nyq / f)))
        t = np.arange(n, dtype=np.float64) / sr
        out = np.zeros(n, dtype=np.float64)
        sign = 1.0
        for k in range(1, harmonics + 1, 2):
            out += sign * np.sin(TWO_PI * k * f * t + self.phase) / (k * k)
            sign = -sign
        out *= 8.0 / (math.pi ** 2)
        return out.astype(np.float32)


class Noise(Osc):
    """Bílý šum. `seed` drží zvuk deterministický mezi rendery."""

    def __init__(self, ratio=1.0, amp=1.0, seed: int = 12345):
        super().__init__(ratio, amp, 0.0)
        self.seed = int(seed)

    def render(self, freq, n, sr):
        rng = np.random.default_rng(self.seed)
        return rng.uniform(-1.0, 1.0, n).astype(np.float32)

    def key(self):
        return super().key() + (self.seed,)


class FM(Osc):
    """Dvouoperátorové FM: nosič + modulátor s vlastním decay.

    Základ zvuku typu Rhodes (`epiano`) a zvonu (`bell`).
    """

    def __init__(self, ratio=1.0, amp=1.0, mod: float = 2.0, index: float = 3.0,
                 decay: float = 0.6, phase: float = 0.0):
        super().__init__(ratio, amp, phase)
        self.mod = float(mod)
        self.index = float(index)
        self.decay = float(decay)

    def render(self, freq, n, sr):
        t = np.arange(n, dtype=np.float64) / sr
        f = freq * self.ratio
        if self.decay > 0:
            idx_env = np.exp(-t / self.decay)
        else:
            idx_env = np.ones(n)
        modulator = np.sin(TWO_PI * f * self.mod * t)
        out = np.sin(TWO_PI * f * t + self.phase + self.index * idx_env * modulator)
        return out.astype(np.float32)

    def key(self):
        return super().key() + (self.mod, self.index, self.decay)
