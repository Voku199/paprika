"""Filtry uvnitř patche (LP, HP, BP, Formant) a jednopólové pomocné funkce.

`onepole_lp/hp` používají i bicí a efekty; počítají se přes `dsp.onepole`
(IIR v scipy), takže ani proměnný cutoff nemá cyklus po samplech.

Všechny patch filtry mají `apply(x, sr, env=None, scale=1.0, bpm=120.0,
freq=440.0)`: `env` je obálka hlasu (0..1), `scale` násobí cutoff
(velocity → jas), `freq` je výška noty (formanty podle hlasového rozsahu).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import dsp

__all__ = ["LP", "HP", "BP", "Formant", "FORMANTS", "VOWELS", "VOICE_TYPES",
           "onepole_lp", "onepole_hp"]


def onepole_lp(x: np.ndarray, cutoff, sr: int, poles: int = 2) -> np.ndarray:
    """Lowpass s konstantním nebo proměnným (pole) cutoffem."""
    if x.size == 0:
        return x
    return dsp.onepole(x, cutoff, sr, poles=max(1, poles))


def onepole_hp(x: np.ndarray, cutoff, sr: int, poles: int = 2) -> np.ndarray:
    """Highpass jako signál minus jeho lowpass."""
    return (x - onepole_lp(x, cutoff, sr, poles)).astype(np.float32)


def _cutoff_curve(cutoff, env_amt, env, scale, lfo, n, sr, bpm):
    """Cutoff jako skalár (nic se nehýbe) nebo pole po samplech."""
    base = float(cutoff) * float(scale)
    curve = base
    if env_amt and env is not None:
        top = min(sr * 0.49, base * (1.0 + 7.0 * env_amt))
        curve = base + (top - base) * np.asarray(env, dtype=np.float64)[:n]
    if lfo is not None:
        curve = curve * 2.0 ** lfo.curve(n, sr, bpm)
    if np.isscalar(curve):
        return float(min(curve, sr * 0.49))
    return np.clip(curve, 10.0, sr * 0.49)


def _lfo_key(lfo):
    return lfo.key() if lfo is not None else None


@dataclass
class LP:
    """Lowpass v patchi. `env` posune cutoff nahoru podle obálky.

    `res` 0..1 přidá rezonanční špičku (acid, growl), `lfo` hýbe cutoffem
    (hloubka v oktávách). S `res=0` a bez `lfo` je to jednopólový filtr
    z v1 — staré songy zní stejně.
    """

    cutoff: float = 3000.0
    env: float = 0.0
    poles: int = 2
    res: float = 0.0
    lfo: object | None = None

    _kind = "lp"

    def apply(self, x, sr, env=None, scale=1.0, bpm=120.0, freq=440.0):
        n = x.shape[0]
        cut = _cutoff_curve(self.cutoff, self.env, env, scale, self.lfo, n, sr, bpm)
        if self.res <= 0 and self.lfo is None:
            low = onepole_lp(x, cut, sr, self.poles)
            return low if self._kind == "lp" else (x - low).astype(np.float32)
        q = dsp.res_to_q(self.res)
        y = dsp.filt_varying(x, self._kind, cut, sr, q=q,
                             stages=max(1, self.poles // 2))
        # rezonanční špička by jinak s `res` rostla o až +25 dB; mírná
        # kompenzace drží hlasitost patche zhruba stejnou jako bez rezonance
        return (y * (dsp.Q_MIN / q) ** 0.35).astype(np.float32)

    def key(self):
        return (type(self).__name__, self.cutoff, self.env, self.poles, self.res,
                _lfo_key(self.lfo))


@dataclass
class HP(LP):
    """Highpass v patchi."""

    cutoff: float = 200.0

    _kind = "hp"


@dataclass
class BP:
    """Pásmová propust: nechá projít jen okolí `cutoff`. Užší s vyšším `res`."""

    cutoff: float = 1000.0
    res: float = 0.3
    env: float = 0.0
    lfo: object | None = None

    def apply(self, x, sr, env=None, scale=1.0, bpm=120.0, freq=440.0):
        cut = _cutoff_curve(self.cutoff, self.env, env, scale, self.lfo,
                            x.shape[0], sr, bpm)
        return dsp.filt_varying(x, "bp", cut, sr, q=dsp.res_to_q(self.res))

    def key(self):
        return ("BP", self.cutoff, self.res, self.env, _lfo_key(self.lfo))


# --------------------------------------------------------------------------
# formanty
# --------------------------------------------------------------------------

VOWELS = ("a", "e", "i", "o", "u")
VOICE_TYPES = ("auto", "bass", "tenor", "alto", "soprano")

# (frekvence Hz, zisk dB, šířka pásma Hz) pěti formantů — tabulka z Csound
# manuálu („Formant values“).
FORMANTS = {
    "soprano": {
        "a": ((800, 1150, 2900, 3900, 4950), (0, -6, -32, -20, -50), (80, 90, 120, 130, 140)),
        "e": ((350, 2000, 2800, 3600, 4950), (0, -20, -15, -40, -56), (60, 100, 120, 150, 200)),
        "i": ((270, 2140, 2950, 3900, 4950), (0, -12, -26, -26, -44), (60, 90, 100, 120, 120)),
        "o": ((450, 800, 2830, 3800, 4950), (0, -11, -22, -22, -50), (70, 80, 100, 130, 135)),
        "u": ((325, 700, 2700, 3800, 4950), (0, -16, -35, -40, -60), (50, 60, 170, 180, 200)),
    },
    "alto": {
        "a": ((800, 1150, 2800, 3500, 4950), (0, -4, -20, -36, -60), (80, 90, 120, 130, 140)),
        "e": ((400, 1600, 2700, 3300, 4950), (0, -24, -30, -35, -60), (60, 80, 120, 150, 200)),
        "i": ((350, 1700, 2700, 3700, 4950), (0, -20, -30, -36, -60), (50, 100, 120, 150, 200)),
        "o": ((450, 800, 2830, 3500, 4950), (0, -9, -16, -28, -55), (70, 80, 100, 130, 135)),
        "u": ((325, 700, 2530, 3500, 4950), (0, -12, -30, -40, -64), (50, 60, 170, 180, 200)),
    },
    "tenor": {
        "a": ((650, 1080, 2650, 2900, 3250), (0, -6, -7, -8, -22), (80, 90, 120, 130, 140)),
        "e": ((400, 1700, 2600, 3200, 3580), (0, -14, -12, -14, -20), (70, 80, 100, 120, 120)),
        "i": ((290, 1870, 2800, 3250, 3540), (0, -15, -18, -20, -30), (40, 90, 100, 120, 120)),
        "o": ((400, 800, 2600, 2800, 3000), (0, -10, -12, -12, -26), (40, 80, 100, 120, 120)),
        "u": ((350, 600, 2700, 2900, 3300), (0, -20, -17, -14, -26), (40, 60, 100, 120, 120)),
    },
    "bass": {
        "a": ((600, 1040, 2250, 2450, 2750), (0, -7, -9, -9, -20), (60, 70, 110, 120, 130)),
        "e": ((400, 1620, 2400, 2800, 3100), (0, -12, -9, -12, -18), (40, 80, 100, 120, 120)),
        "i": ((250, 1750, 2600, 3050, 3340), (0, -30, -16, -22, -28), (60, 90, 100, 120, 120)),
        "o": ((400, 750, 2400, 2600, 2900), (0, -11, -21, -20, -40), (40, 80, 100, 120, 120)),
        "u": ((350, 600, 2400, 2675, 2950), (0, -20, -32, -28, -36), (40, 80, 100, 120, 120)),
    },
}


def voice_for_pitch(freq: float) -> str:
    """Hlasový rozsah podle výšky noty (MIDI < 48 bas, < 57 tenor, < 65 alt)."""
    midi = 69.0 + 12.0 * math.log2(max(1.0, float(freq)) / 440.0)
    if midi < 48:
        return "bass"
    if midi < 57:
        return "tenor"
    if midi < 65:
        return "alto"
    return "soprano"


@dataclass
class Formant:
    """Samohláska: pět paralelních pásmových filtrů na formantech.

    Pilu nebo pulz udělá „óó“ / „áá“ — sbor, vocal chop, braam. `voice="auto"`
    vybere tabulku podle výšky noty. `morph=("a", "o", 0.3)` přejde z jedné
    samohlásky do druhé za 0.3 s. `gain` dorovná hlasitost po filtraci.
    """

    vowel: str = "a"
    voice: str = "auto"
    morph: tuple | None = None
    gain: float = 3.0

    def __post_init__(self):
        vowels = [self.vowel] + (list(self.morph[:2]) if self.morph else [])
        for v in vowels:
            if v not in VOWELS:
                raise ValueError(f"neznámá samohláska {v!r}; k dispozici: "
                                 f"{', '.join(VOWELS)}")
        if self.voice not in VOICE_TYPES:
            raise ValueError(f"neznámý hlas {self.voice!r}; k dispozici: "
                             f"{', '.join(VOICE_TYPES)}")

    def apply(self, x, sr, env=None, scale=1.0, bpm=120.0, freq=440.0):
        voice = voice_for_pitch(freq) if self.voice == "auto" else self.voice
        table = FORMANTS[voice]
        x = np.asarray(x, dtype=np.float32)
        out = np.zeros(x.shape, dtype=np.float64)
        if self.morph is None:
            for f, db, bw in zip(*table[self.vowel]):
                if f >= sr * 0.45:
                    continue
                out += dsp.filt(x, "bp", f, sr, q=f / bw) * 10.0 ** (db / 20.0)
        else:
            v1, v2, sec = self.morph
            n = x.shape[0]
            u = np.clip(np.arange(n) / sr / max(1e-3, float(sec)), 0.0, 1.0)
            for (f1, db1, bw1), (f2, db2, bw2) in zip(zip(*table[v1]), zip(*table[v2])):
                f = f1 + (f2 - f1) * u
                q = f / (bw1 + (bw2 - bw1) * u)
                amp = 10.0 ** ((db1 + (db2 - db1) * u) / 20.0)
                y = dsp.filt_varying(x, "bp", f, sr, q=q)
                out += y * (amp[:, None] if y.ndim == 2 else amp)
        return (out * self.gain).astype(np.float32)

    def key(self):
        return ("Formant", self.vowel, self.voice, self.morph, self.gain)
