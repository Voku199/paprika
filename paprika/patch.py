"""Syntéza jednoho hlasu: obálka a Patch.

Jedna nota = jeden `voice` buffer. Oscilátory žijí v `oscillators.py`,
filtry v `filters.py`; tenhle modul je re-exportuje, takže
`from paprika.patch import Saw, LP` funguje dál.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .filters import HP, LP, onepole_hp, onepole_lp
from .oscillators import FM, Noise, Osc, Saw, Sine, Square, Tri

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
