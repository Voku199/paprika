"""Oscilátory: jeden periodický (nebo šumový) průběh na dané frekvenci.

`ratio` je poměr k základní frekvenci noty, takže `Sine(2.01)` je mírně
rozladěná oktáva. `render(freq, n, sr, phase)` bere frekvenci jako číslo,
nebo jako pole délky `n` (frekvenční křivka v Hz) — na tom stojí vibrato,
glide i pitch envelope. `phase` je posun v periodách (round-robin varianty).

Mono oscilátor vrací `(n,)`, stereo (`SuperSaw` se `spread > 0`) `(n, 2)`.
"""

from __future__ import annotations

import math

import numpy as np

from . import dsp

__all__ = ["Osc", "Sine", "Saw", "Square", "Tri", "Noise", "FM",
           "SuperSaw", "Pulse", "Pluck", "Additive"]

TWO_PI = 2.0 * math.pi
MAX_HARMONICS = 64


class Osc:
    """Základ oscilátoru. `ratio` je poměr k základní frekvenci noty."""

    def __init__(self, ratio: float = 1.0, amp: float = 1.0, phase: float = 0.0):
        self.ratio = float(ratio)
        self.amp = float(amp)
        self.phase = float(phase)

    @property
    def stereo(self) -> bool:
        return False

    def _cycles(self, freq, n: int, sr: int, phase: float = 0.0) -> np.ndarray:
        """Fáze v periodách. Pro křivku je to integrál frekvence."""
        ph = self.phase + float(phase)
        if np.isscalar(freq):
            return float(freq) * self.ratio * np.arange(n, dtype=np.float64) / sr + ph
        f = np.asarray(freq, dtype=np.float64)[:n] * self.ratio
        return np.cumsum(f) / sr - f[0] / sr + ph

    def _fmax(self, freq) -> float:
        return float(np.max(freq)) * self.ratio

    def _harmonics(self, freq, sr: int, cap: int = MAX_HARMONICS) -> int:
        """Kolik harmonických se vejde pod Nyquist u nejvyšší frekvence."""
        return max(1, min(cap, int(sr * 0.5 / self._fmax(freq))))

    def render(self, freq, n: int, sr: int, phase: float = 0.0) -> np.ndarray:
        raise NotImplementedError

    def key(self) -> tuple:
        return (type(self).__name__, self.ratio, self.amp, self.phase)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}({self.ratio:g}, amp={self.amp:g})"


class Sine(Osc):
    def render(self, freq, n, sr, phase=0.0):
        return np.sin(TWO_PI * self._cycles(freq, n, sr, phase)).astype(np.float32)


class Saw(Osc):
    """Pilový průběh, band-limited přes sčítání harmonických (bez aliasu)."""

    def render(self, freq, n, sr, phase=0.0):
        if self._fmax(freq) <= 0:
            return np.zeros(n, dtype=np.float32)
        cyc = TWO_PI * self._cycles(freq, n, sr, phase)
        out = np.zeros(n, dtype=np.float64)
        for k in range(1, self._harmonics(freq, sr) + 1):
            out -= np.sin(k * cyc) / k
        out *= 2.0 / math.pi
        return out.astype(np.float32)


class Square(Osc):
    """Obdélník, band-limited (jen liché harmonické)."""

    def __init__(self, ratio=1.0, amp=1.0, phase=0.0, width: float = 0.5):
        super().__init__(ratio, amp, phase)
        self.width = float(width)

    def render(self, freq, n, sr, phase=0.0):
        if self._fmax(freq) <= 0:
            return np.zeros(n, dtype=np.float32)
        cyc = TWO_PI * self._cycles(freq, n, sr, phase)
        out = np.zeros(n, dtype=np.float64)
        for k in range(1, self._harmonics(freq, sr) + 1, 2):
            out += np.sin(k * cyc) / k
        out *= 4.0 / math.pi
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def key(self):
        return super().key() + (self.width,)


class Tri(Osc):
    def render(self, freq, n, sr, phase=0.0):
        if self._fmax(freq) <= 0:
            return np.zeros(n, dtype=np.float32)
        cyc = TWO_PI * self._cycles(freq, n, sr, phase)
        out = np.zeros(n, dtype=np.float64)
        sign = 1.0
        for k in range(1, self._harmonics(freq, sr, cap=32) + 1, 2):
            out += sign * np.sin(k * cyc) / (k * k)
            sign = -sign
        out *= 8.0 / (math.pi ** 2)
        return out.astype(np.float32)


class Noise(Osc):
    """Šum. `seed` drží zvuk deterministický, `color` white/pink/brown."""

    def __init__(self, ratio=1.0, amp=1.0, seed: int = 12345,
                 color: str = "white"):
        super().__init__(ratio, amp, 0.0)
        self.seed = int(seed)
        if color not in dsp.NOISE_COLORS:
            raise ValueError(f"neznámá barva šumu {color!r}; k dispozici: "
                             f"{', '.join(dsp.NOISE_COLORS)}")
        self.color = color

    def render(self, freq, n, sr, phase=0.0):
        seed = self.seed + int(round(float(phase) * 1000))
        if self.color == "white":
            rng = np.random.default_rng(seed)
            return rng.uniform(-1.0, 1.0, n).astype(np.float32)
        return dsp.colored_noise(n, self.color, seed)

    def key(self):
        return super().key() + (self.seed, self.color)


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

    def render(self, freq, n, sr, phase=0.0):
        t = np.arange(n, dtype=np.float64) / sr
        cyc = self._cycles(freq, n, sr, phase)
        idx_env = np.exp(-t / self.decay) if self.decay > 0 else np.ones(n)
        modulator = np.sin(TWO_PI * self.mod * (cyc - self.phase - float(phase)))
        out = np.sin(TWO_PI * cyc + self.index * idx_env * modulator)
        return out.astype(np.float32)

    def key(self):
        return super().key() + (self.mod, self.index, self.decay)


# --------------------------------------------------------------------------
# v2
# --------------------------------------------------------------------------

def _dt(freq, ratio: float, n: int, sr: int) -> np.ndarray:
    """f / sr pro polyBLEP — skalár rozkopírovaný, nebo křivka."""
    f = np.asarray(freq, dtype=np.float64) * ratio
    return np.broadcast_to(f / sr, (n,)) if f.ndim == 0 else f[:n] / sr


class SuperSaw(Osc):
    """Unison pil: `voices` hlasů rozladěných v rozsahu `detune` půltónů,
    rozprostřených do stran podle `spread` (0 = mono, 1 = plně L/R).

    Každý hlas má náhodnou startovní fázi (podle `seed`), takže to nezní jako
    jedna fázující pila. Pily jsou polyBLEP — levné i při sedmi hlasech.
    """

    def __init__(self, ratio=1.0, amp=1.0, voices: int = 7, detune: float = 0.25,
                 spread: float = 0.8, seed: int = 0):
        super().__init__(ratio, amp, 0.0)
        self.voices = max(1, int(voices))
        self.detune = float(detune)
        self.spread = float(np.clip(spread, 0.0, 1.0))
        self.seed = int(seed)

    @property
    def stereo(self) -> bool:
        return self.spread > 0

    def render(self, freq, n, sr, phase=0.0):
        v = self.voices
        offs = (np.linspace(-self.detune / 2, self.detune / 2, v) if v > 1
                else np.zeros(1))
        pans = np.linspace(-self.spread, self.spread, v) if v > 1 else np.zeros(1)
        starts = np.random.default_rng(self.seed).uniform(0.0, 1.0, v)
        base = self._cycles(freq, n, sr, phase)
        dt0 = _dt(freq, self.ratio, n, sr)
        left = np.zeros(n)
        right = np.zeros(n)
        for off, pan, st in zip(offs, pans, starts):
            r = 2.0 ** (off / 12.0)
            y = dsp.blep_saw(base * r + st, dt0 * r)
            ang = (pan + 1.0) * math.pi / 4.0
            left += y * math.cos(ang)
            right += y * math.sin(ang)
        norm = 1.0 / math.sqrt(v)
        if not self.stereo:
            # všechny hlasy uprostřed: levý kanál je součet × cos(π/4)
            return (left * norm / math.cos(math.pi / 4)).astype(np.float32)
        return (np.stack([left, right], axis=1) * norm).astype(np.float32)

    def key(self):
        return super().key() + (self.voices, self.detune, self.spread, self.seed)


class Pulse(Osc):
    """Obdélník s proměnnou šířkou (rozdíl dvou polyBLEP pil).

    `pwm` je `LFO` — hloubka v jednotkách šířky, typicky 0.1–0.4.
    """

    def __init__(self, ratio=1.0, amp=1.0, width: float = 0.5, pwm=None):
        super().__init__(ratio, amp, 0.0)
        self.width = float(width)
        self.pwm = pwm

    def render(self, freq, n, sr, phase=0.0):
        cyc = self._cycles(freq, n, sr, phase)
        dt = _dt(freq, self.ratio, n, sr)
        w = np.full(n, self.width)
        if self.pwm is not None:
            w = w + self.pwm.curve(n, sr)
        w = np.clip(w, 0.02, 0.98)
        y = dsp.blep_saw(cyc, dt) - dsp.blep_saw(cyc + w, dt)
        wm = float(np.mean(w))
        return (y / (2.0 * max(wm, 1.0 - wm))).astype(np.float32)

    def key(self):
        pwm = self.pwm.key() if self.pwm is not None else None
        return super().key() + (self.width, pwm)


class Pluck(Osc):
    """Karplus-Strong struna: pizzicato, harfa, kytara, koto.

    Ladí se podle frekvence na začátku noty (glide struna neumí).
    """

    def __init__(self, ratio=1.0, amp=1.0, decay: float = 1.0,
                 bright: float = 0.5, seed: int = 0):
        super().__init__(ratio, amp, 0.0)
        self.decay = float(decay)
        self.bright = float(bright)
        self.seed = int(seed)

    def render(self, freq, n, sr, phase=0.0):
        f0 = float(freq if np.isscalar(freq) else np.asarray(freq)[0]) * self.ratio
        seed = self.seed + int(round(float(phase) * 1000))
        return dsp.karplus(f0, n, sr, self.decay, self.bright, seed)

    def key(self):
        return super().key() + (self.decay, self.bright, self.seed)


class Additive(Osc):
    """Součet parciálů `(ratio, amp, decay_s)`; decay 0 = bez útlumu.

    `inharm` natahuje vyšší parciály jako u struny piana:
    f_k = f · r · sqrt(1 + inharm · r²). Normalizováno součtem amplitud.
    """

    def __init__(self, partials=((1.0, 1.0, 0.0),), inharm: float = 0.0,
                 ratio=1.0, amp=1.0):
        super().__init__(ratio, amp, 0.0)
        self.partials = tuple((float(r), float(a), float(d)) for r, a, d in partials)
        self.inharm = float(inharm)

    def render(self, freq, n, sr, phase=0.0):
        cyc = self._cycles(freq, n, sr, phase)
        fmax = self._fmax(freq)
        t = np.arange(n, dtype=np.float64) / sr
        out = np.zeros(n, dtype=np.float64)
        total = 0.0
        for k, (r, a, d) in enumerate(self.partials):
            r_eff = r * math.sqrt(1.0 + self.inharm * r * r)
            if fmax * r_eff >= sr * 0.5:
                continue
            part = np.sin(TWO_PI * (cyc * r_eff + 0.13 * k)) * a
            if d > 0:
                part *= np.exp(-t / d)
            out += part
            total += abs(a)
        if total > 0:
            out /= total
        return out.astype(np.float32)

    def key(self):
        return super().key() + (self.partials, self.inharm)
