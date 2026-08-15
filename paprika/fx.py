"""Efekty na úrovni stopy: Reverb, Delay, LowPass, HighPass, Saturate, Sidechain.

Každý efekt má `process(buf, ctx)` a `tail(ctx)` — kolik sekund navíc si
`render()` musí nechat v bufferu, aby se ocásek vešel.

Zpětné vazby (comb, allpass, delay) jsou počítané po blocích délky zpoždění,
takže i přes rekurzi jde o vektorizované numpy operace, ne cyklus po samplech.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .patch import onepole_hp, onepole_lp

__all__ = ["Fx", "FxContext", "Reverb", "Delay", "LowPass", "HighPass",
           "Saturate", "Sidechain"]


@dataclass
class FxContext:
    """Co efekt potřebuje vědět o okolí."""

    sr: int = 44100
    bpm: float = 120.0
    beats_per_bar: float = 4.0
    track_buffers: dict | None = None

    def beats_to_sec(self, beats: float) -> float:
        return float(beats) * 60.0 / float(self.bpm)


class Fx:
    """Základ efektu."""

    def process(self, buf: np.ndarray, ctx: FxContext) -> np.ndarray:
        raise NotImplementedError

    def tail(self, ctx: FxContext) -> float:
        return 0.0

    @property
    def needs_tracks(self) -> bool:
        return False


def _feedback_delay(x: np.ndarray, d: int, fb: float) -> np.ndarray:
    """y[n] = x[n] + fb * y[n-d], počítané po blocích délky d."""
    if d <= 0:
        return x.copy()
    y = x.astype(np.float64).copy()
    for start in range(d, y.size, d):
        end = min(start + d, y.size)
        y[start:end] += fb * y[start - d:start - d + (end - start)]
    return y


def _allpass(x: np.ndarray, d: int, g: float) -> np.ndarray:
    """y[n] = -x[n] + x[n-d] + g*y[n-d]."""
    if d <= 0:
        return x.copy()
    xd = np.concatenate([np.zeros(d), x.astype(np.float64)])[:x.size]
    y = (-x.astype(np.float64) + xd)
    for start in range(d, y.size, d):
        end = min(start + d, y.size)
        y[start:end] += g * y[start - d:start - d + (end - start)]
    return y


class Reverb(Fx):
    """Schroeder / Freeverb: 8 paralelních comb filtrů, pak 4 allpassy."""

    COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
    ALLPASS = (556, 441, 341, 225)

    def __init__(self, room: float = 0.6, mix: float = 0.3, damp: float = 0.4,
                 width: float = 1.0):
        self.room = float(np.clip(room, 0.0, 1.0))
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.damp = float(np.clip(damp, 0.0, 1.0))
        self.width = float(width)

    def _feedback(self) -> float:
        return 0.70 + 0.28 * self.room

    def tail(self, ctx: FxContext) -> float:
        # doba, za kterou feedback klesne pod -60 dB
        fb = self._feedback()
        loop_sec = max(self.COMBS) / 44100.0
        n_loops = np.log(1e-3) / np.log(max(fb, 1e-6))
        return float(min(12.0, loop_sec * n_loops + 0.1))

    def process(self, buf: np.ndarray, ctx: FxContext) -> np.ndarray:
        if self.mix <= 0.0 or buf.size == 0:
            return buf
        scale = ctx.sr / 44100.0
        fb = self._feedback()
        wet = np.zeros(buf.size, dtype=np.float64)
        for d in self.COMBS:
            wet += _feedback_delay(buf, int(d * scale), fb)
        wet /= len(self.COMBS)
        if self.damp > 0:
            cutoff = 12000.0 * (1.0 - self.damp) + 1200.0 * self.damp
            wet = onepole_lp(wet.astype(np.float32), cutoff, ctx.sr, poles=1)
        for d in self.ALLPASS:
            wet = _allpass(np.asarray(wet, dtype=np.float64), int(d * scale), 0.5)
        wet = np.asarray(wet, dtype=np.float32) * 0.35
        return ((1.0 - self.mix) * buf + self.mix * wet).astype(np.float32)


class Delay(Fx):
    """Tempo-synchronní echo. `time` je zlomek celé noty (1/8 = osminka)."""

    def __init__(self, time: float = 0.25, fb: float = 0.3, mix: float = 0.25):
        self.time = float(time)
        self.fb = float(np.clip(fb, 0.0, 0.95))
        self.mix = float(np.clip(mix, 0.0, 1.0))

    def _sec(self, ctx: FxContext) -> float:
        return ctx.beats_to_sec(self.time * ctx.beats_per_bar)

    def tail(self, ctx: FxContext) -> float:
        if self.fb <= 0:
            return self._sec(ctx)
        n_loops = np.log(1e-3) / np.log(max(self.fb, 1e-6))
        return float(min(15.0, self._sec(ctx) * (n_loops + 1)))

    def process(self, buf: np.ndarray, ctx: FxContext) -> np.ndarray:
        if self.mix <= 0.0 or buf.size == 0:
            return buf
        d = int(round(self._sec(ctx) * ctx.sr))
        if d <= 0:
            return buf
        echo = _feedback_delay(buf, d, self.fb)
        echo = np.concatenate([np.zeros(d, dtype=np.float64), echo])[:buf.size]
        return (buf + self.mix * echo).astype(np.float32)


class LowPass(Fx):
    """Lowpass na celé stopě."""

    def __init__(self, cutoff: float = 2000.0, poles: int = 2):
        self.cutoff = float(cutoff)
        self.poles = int(poles)

    def process(self, buf, ctx):
        return onepole_lp(buf, self.cutoff, ctx.sr, self.poles)


class HighPass(Fx):
    """Highpass na celé stopě — nejlevnější způsob, jak uvolnit basy v mixu."""

    def __init__(self, cutoff: float = 200.0, poles: int = 2):
        self.cutoff = float(cutoff)
        self.poles = int(poles)

    def process(self, buf, ctx):
        return onepole_hp(buf, self.cutoff, ctx.sr, self.poles)


class Saturate(Fx):
    """Měkké zkreslení: tanh normalizovaný tak, aby nezvedal hlasitost."""

    def __init__(self, drive: float = 2.0, mix: float = 1.0):
        self.drive = max(1e-3, float(drive))
        self.mix = float(np.clip(mix, 0.0, 1.0))

    def process(self, buf, ctx):
        wet = np.tanh(buf * self.drive) / np.tanh(self.drive)
        return ((1.0 - self.mix) * buf + self.mix * wet).astype(np.float32)


class Sidechain(Fx):
    """Klasické pumpování: obálka jiné stopy stahuje hlasitost téhle."""

    def __init__(self, source: str, amount: float = 0.6, attack: float = 0.005,
                 release: float = 0.18):
        self.source = str(source)
        self.amount = float(np.clip(amount, 0.0, 1.0))
        self.attack = float(attack)
        self.release = float(release)

    @property
    def needs_tracks(self) -> bool:
        return True

    def process(self, buf, ctx):
        tracks = ctx.track_buffers or {}
        src = tracks.get(self.source)
        if src is None:
            raise KeyError(
                f"Sidechain: stopa {self.source!r} neexistuje nebo ještě není "
                f"vyrenderovaná (k dispozici: {', '.join(sorted(tracks)) or '—'})")
        n = buf.size
        env = np.abs(src[:n].astype(np.float32))
        if env.size < n:
            env = np.concatenate([env, np.zeros(n - env.size, dtype=np.float32)])
        env = onepole_lp(env, 1.0 / max(self.release, 1e-3) / 6.28, ctx.sr, poles=1)
        peak = float(env.max()) if env.size else 0.0
        if peak <= 1e-9:
            return buf
        gain = 1.0 - self.amount * (env / peak)
        atk = onepole_lp(gain.astype(np.float32),
                         1.0 / max(self.attack, 1e-3) / 6.28, ctx.sr, poles=1)
        return (buf * atk).astype(np.float32)
