"""Syntéza jednoho hlasu: oscilátory, ADSR, filtry, Patch.

Jedna nota = jeden `voice` buffer. Vše je float32 mono.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Osc", "Sine", "Saw", "Square", "Tri", "Noise", "FM",
    "ADSR", "LP", "HP", "Patch", "midi_to_freq",
    "onepole_lp", "onepole_hp",
]

FADE_SEC = 0.002  # povinný fade na krajích hlasu, jinak to lupe
TWO_PI = 2.0 * math.pi


def midi_to_freq(pitch: float) -> float:
    return 440.0 * 2.0 ** ((float(pitch) - 69.0) / 12.0)


# --------------------------------------------------------------------------
# oscilátory
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# obálka
# --------------------------------------------------------------------------

@dataclass
class ADSR:
    """Attack / decay / sustain / release. Časy v sekundách, sustain 0-1."""

    a: float = 0.01
    d: float = 0.1
    s: float = 0.7
    r: float = 0.2

    def render(self, gate_n: int, total_n: int, sr: int) -> np.ndarray:
        env = np.zeros(total_n, dtype=np.float32)
        if total_n <= 0:
            return env
        gate_n = max(1, min(int(gate_n), total_n))
        a_n = max(1, int(self.a * sr))
        d_n = max(1, int(self.d * sr))
        r_n = max(1, int(self.r * sr))

        i = 0
        seg = min(a_n, gate_n)
        env[:seg] = np.linspace(0.0, 1.0, seg, endpoint=False, dtype=np.float32)
        i = seg
        if i < gate_n:
            seg = min(d_n, gate_n - i)
            env[i:i + seg] = np.linspace(1.0, self.s, seg, endpoint=False,
                                         dtype=np.float32)
            i += seg
        if i < gate_n:
            env[i:gate_n] = self.s
            i = gate_n
        level = float(env[gate_n - 1]) if gate_n > 0 else 0.0
        tail = total_n - gate_n
        if tail > 0:
            seg = min(r_n, tail)
            env[gate_n:gate_n + seg] = np.linspace(level, 0.0, seg, dtype=np.float32)
            # za releasem je ticho, což už v poli je
        return env

    def key(self) -> tuple:
        return (self.a, self.d, self.s, self.r)


# --------------------------------------------------------------------------
# filtry
# --------------------------------------------------------------------------

def _coef(cutoff: float, sr: int) -> float:
    cutoff = float(np.clip(cutoff, 10.0, sr * 0.49))
    return float(1.0 - math.exp(-TWO_PI * cutoff / sr))


def _fir_kernel(a: float, max_taps: int = 8192) -> np.ndarray:
    """Impulsní odezva jednopólového LP: h[n] = a*(1-a)^n."""
    a = float(np.clip(a, 1e-5, 0.999999))
    taps = int(min(max_taps, max(4, math.ceil(math.log(1e-4) / math.log(1 - a)))))
    n = np.arange(taps, dtype=np.float64)
    h = a * (1.0 - a) ** n
    return (h / h.sum()).astype(np.float64)


def onepole_lp(x: np.ndarray, cutoff, sr: int, poles: int = 2) -> np.ndarray:
    """Lowpass. Konstantní cutoff jede přes FIR aproximaci (rychlé),
    proměnný cutoff (envelope) přes cyklus po samplech."""
    if x.size == 0:
        return x
    if np.isscalar(cutoff):
        h = _fir_kernel(_coef(float(cutoff), sr))
        y = x.astype(np.float64)
        for _ in range(max(1, poles)):
            y = np.convolve(y, h)[:x.size]
        return y.astype(np.float32)
    a = np.clip(1.0 - np.exp(-TWO_PI * np.clip(cutoff, 10.0, sr * 0.49) / sr),
                1e-5, 0.999999)
    y = x.astype(np.float64)
    for _ in range(max(1, poles)):
        out = np.empty_like(y)
        prev = 0.0
        for i in range(y.size):
            prev += a[i] * (y[i] - prev)
            out[i] = prev
        y = out
    return y.astype(np.float32)


def onepole_hp(x: np.ndarray, cutoff, sr: int, poles: int = 2) -> np.ndarray:
    """Highpass jako signál minus jeho lowpass."""
    return (x - onepole_lp(x, cutoff, sr, poles)).astype(np.float32)


@dataclass
class LP:
    """Lowpass v patchi. `env` posune cutoff nahoru podle obálky."""

    cutoff: float = 3000.0
    env: float = 0.0
    poles: int = 2

    def apply(self, x: np.ndarray, sr: int, env: np.ndarray | None = None):
        if self.env and env is not None:
            top = min(sr * 0.49, self.cutoff * (1.0 + 7.0 * self.env))
            curve = self.cutoff + (top - self.cutoff) * env
            return onepole_lp(x, curve, sr, self.poles)
        return onepole_lp(x, self.cutoff, sr, self.poles)

    def key(self):
        return ("LP", self.cutoff, self.env, self.poles)


@dataclass
class HP:
    """Highpass v patchi."""

    cutoff: float = 200.0
    env: float = 0.0
    poles: int = 2

    def apply(self, x: np.ndarray, sr: int, env: np.ndarray | None = None):
        if self.env and env is not None:
            top = min(sr * 0.49, self.cutoff * (1.0 + 7.0 * self.env))
            curve = self.cutoff + (top - self.cutoff) * env
            return onepole_hp(x, curve, sr, self.poles)
        return onepole_hp(x, self.cutoff, sr, self.poles)

    def key(self):
        return ("HP", self.cutoff, self.env, self.poles)


# --------------------------------------------------------------------------
# patch
# --------------------------------------------------------------------------

@dataclass
class Patch:
    """Nástroj: sada oscilátorů, obálka, volitelný filtr."""

    osc: list = field(default_factory=lambda: [Sine()])
    adsr: ADSR = field(default_factory=ADSR)
    filter: object | None = None
    gain: float = 1.0
    name: str = "patch"

    def key(self) -> tuple:
        return (
            tuple(o.key() for o in self.osc),
            self.adsr.key(),
            self.filter.key() if self.filter is not None else None,
            self.gain,
        )

    def voice(self, freq: float, dur: float, vel: int = 100,
              sr: int = 44100) -> np.ndarray:
        """Jedna nota. `dur` je znějící délka v sekundách; buffer je delší
        o release obálky."""
        gate_n = max(1, int(round(dur * sr)))
        total_n = gate_n + max(1, int(round(self.adsr.r * sr))) + 1
        buf = np.zeros(total_n, dtype=np.float32)
        for o in self.osc:
            buf += o.amp * o.render(freq, total_n, sr)
        env = self.adsr.render(gate_n, total_n, sr)
        buf *= env
        if self.filter is not None:
            buf = self.filter.apply(buf, sr, env)
        buf *= self.gain * ((max(0, min(127, int(vel))) / 127.0) ** 1.6)
        _fade(buf, sr)
        return buf.astype(np.float32)


def _fade(buf: np.ndarray, sr: int, sec: float = FADE_SEC) -> None:
    """Povinný fade-in/out na krajích hlasu. Mění buffer na místě."""
    n = min(int(sec * sr), buf.size // 2)
    if n <= 0:
        return
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    buf[:n] *= ramp
    buf[-n:] *= ramp[::-1]
