"""Filtry uvnitř patche (LP, HP) a jednopólové pomocné funkce.

`onepole_lp/hp` používají i bicí a efekty; počítají se přes `dsp.onepole`
(IIR v scipy), takže ani proměnný cutoff nemá cyklus po samplech.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dsp

__all__ = ["LP", "HP", "onepole_lp", "onepole_hp"]


def onepole_lp(x: np.ndarray, cutoff, sr: int, poles: int = 2) -> np.ndarray:
    """Lowpass s konstantním nebo proměnným (pole) cutoffem."""
    if x.size == 0:
        return x
    return dsp.onepole(x, cutoff, sr, poles=max(1, poles))


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
