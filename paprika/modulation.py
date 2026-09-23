"""Modulační zdroje uvnitř patche: LFO a pitch envelope.

Obojí vrací křivku po samplech pro jeden hlas, takže se dá cachovat spolu
s hlasem (`key()`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dsp

__all__ = ["LFO", "PitchEnv"]

PITCH_CURVES = ("exp", "lin")


@dataclass
class LFO:
    """Pomalý oscilátor. Jednotka `depth` záleží na cíli:

    - `vibrato` — půltóny,
    - `tremolo` — podíl hlasitosti 0..1,
    - `lfo` filtru — oktávy cutoffu,
    - `pwm` pulzu — jednotky šířky.

    `sync` (zlomek celé noty, 1/8 = osminka) má přednost před `rate` v Hz.
    `delay` v sekundách odloží nástup (živý hráč začne vibrovat až po chvíli),
    `fade` je délka náběhu hloubky; při `delay > 0` a `fade == 0` je 50 ms.
    """

    rate: float = 5.0
    depth: float = 0.1
    shape: str = "sine"
    sync: float | None = None
    delay: float = 0.0
    fade: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if self.shape not in dsp.LFO_SHAPES:
            raise ValueError(f"neznámý tvar LFO {self.shape!r}; k dispozici: "
                             f"{', '.join(dsp.LFO_SHAPES)}")

    def hz(self, bpm: float = 120.0, beats_per_bar: float = 4.0) -> float:
        if self.sync:
            period_beats = float(self.sync) * beats_per_bar
            return float(bpm) / 60.0 / max(1e-6, period_beats)
        return float(self.rate)

    def curve(self, n: int, sr: int, bpm: float = 120.0) -> np.ndarray:
        raw = dsp.lfo(self.shape, self.hz(bpm), n, sr, seed=self.seed) * self.depth
        fade = self.fade if self.fade > 0 else (0.05 if self.delay > 0 else 0.0)
        if self.delay > 0 or fade > 0:
            t = np.arange(n, dtype=np.float64) / sr
            ramp = np.clip((t - self.delay) / fade, 0.0, 1.0) if fade > 0 \
                else (t >= self.delay).astype(np.float64)
            raw = raw * ramp
        return raw

    def key(self) -> tuple:
        return ("LFO", self.rate, self.depth, self.shape, self.sync, self.delay,
                self.fade, self.seed)


@dataclass
class PitchEnv:
    """Nota začne o `semi` půltónů výš (záporné = níž) a za `time` sekund
    dojede na svou výšku. Kopák, 808, pluck, „braam“."""

    semi: float = 12.0
    time: float = 0.05
    shape: str = "exp"

    def __post_init__(self):
        if self.shape not in PITCH_CURVES:
            raise ValueError(f"neznámý tvar pitch envelope {self.shape!r}; "
                             f"k dispozici: {', '.join(PITCH_CURVES)}")

    def curve(self, n: int, sr: int) -> np.ndarray:
        """Posun v půltónech pro každý sample hlasu."""
        t = np.arange(n, dtype=np.float64) / sr
        tm = max(1e-4, float(self.time))
        if self.shape == "exp":
            return self.semi * np.exp(-5.0 * t / tm)
        return self.semi * np.clip(1.0 - t / tm, 0.0, 1.0)

    def key(self) -> tuple:
        return ("PitchEnv", self.semi, self.time, self.shape)
