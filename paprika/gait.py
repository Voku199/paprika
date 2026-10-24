"""Chůze a běh jako model: kdy, kterou nohou a jak došlápnout. Bez audia.

Kroky v herním audiu zní strojově ze tří důvodů a každý tu má protilék:

- **Metronom.** Lidský krok není na mřížce. Interval mezi kroky kolísá
  o 2–4 % a kolísání má dlouhou paměť (1/f, Hausdorff et al. 1995: DFA
  α ≈ 0.8–1 u zdravé chůze). `plan` proto posouvá kroky o 1/f šum
  s rozptylem intervalu `jitter` (variační koeficient). Odchylka se měří
  od mřížky, ne od předchozího kroku, takže chodec nikdy neuteče hudbě.
- **Nezávislá náhoda.** Losovat každý krok znovu zní jako „jiný vzorek
  pokaždé“, ne jako jeden člověk. Síla, jas, výška a odstup pata–špička
  jsou proto Ornsteinův–Uhlenbeckův proces přes kroky (AR(1), `CORR`):
  sousední kroky jsou příbuzné, vzdálené nezávislé.
- **Dokonalá symetrie.** Každý má jednu nohu těžší. `Walker` vylosuje ze
  `seed` stálé rozdíly levé a pravé nohy (síla, jas, odstup, časování) —
  stejné po celou skladbu, ať se `steps()` zavolá kolikrát chce.

Rychlost (m/s) a váha (kg) mění fyziku kroku (`physics`): kadenci, dobu
kontaktu, odstup paty a špičky, sílu dopadu a barvu. Nad 2.2 m/s chodec
přejde do běhu — jeden silný dopad místo paty a špičky.

Čísla jsou orientační z biomechaniky chůze (kadence ~1.9 kroku/s při
1.4 m/s, ~3 kroky/s při běhu 3.5 m/s; reakční síla podložky ~1.1–1.3×
tělesné váhy při chůzi, 2–3× při běhu). Audio z nich dělá `human._step`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["PACES", "RUN_SPEED", "REF_KG", "REF_SPEED", "JITTER", "CORR",
           "Physics", "Walker", "StepSpec", "cadence", "physics", "plan",
           "pink", "interval_cv"]

PACES = {"slow": 0.9, "walk": 1.4, "brisk": 1.9, "run": 3.2}   # m/s
RUN_SPEED = 2.2       # m/s: přechod chůze -> běh
REF_KG = 75.0         # referenční chodec: 75 kg, 1.4 m/s
REF_SPEED = PACES["walk"]
MIN_KG, MAX_KG = 20.0, 250.0
MIN_SPEED, MAX_SPEED = 0.3, 8.0
JITTER = 0.025        # variační koeficient intervalu (zdravá chůze 2–4 %)
PINK_BETA = 0.8       # spektrum odchylek ~ 1/f^β
MAX_SHIFT = 0.3       # krok se neposune o víc než 30 % intervalu

CORR = 0.6            # AR(1) korelace parametrů sousedních kroků
# směrodatné odchylky OU procesů (log poměr): síla, odstup, poměr špičky, jas, výška
SPREAD = {"level": 0.12, "gap": 0.15, "ratio": 0.12, "bright": 0.06, "pitch": 0.025}
BRIGHT_FROM_LEVEL = 0.5   # silnější došlap je i jasnější
# stálé rozdíly nohou (log poměr, směrodatná odchylka); časování v podílu intervalu
BIAS = {"level": 0.06, "gap": 0.08, "bright": 0.04, "pitch": 0.015, "time": 0.008}


@dataclass(frozen=True)
class Physics:
    """Fyzika jednoho kroku při dané rychlosti a váze."""
    run: bool
    contact: float     # s: doba kontaktu chodidla s podložkou
    gap: float         # s: pata -> špička (u běhu dopad -> odval)
    ratio: float       # síla druhého úderu vůči prvnímu
    level: float       # amplituda vůči referenčnímu chodci
    pitch: float       # násobek frekvence dupnutí (těžší = hlubší)
    bright: float      # násobek frekvence kliku (rychlejší = jasnější)
    decay: float       # násobek doznění dupnutí (těžší a delší kontakt = delší)


def cadence(speed: float) -> float:
    """Kroky za sekundu při rychlosti `speed` m/s."""
    return 1.2 + 0.5 * float(speed)


def _grf(speed: float) -> float:
    """Vrchol reakční síly podložky v násobcích tělesné váhy."""
    v = float(speed)
    if v >= RUN_SPEED:
        return 2.3 + 0.15 * (v - 3.5)
    return 1.1 + 0.15 * (v - REF_SPEED)


def physics(weight: float = REF_KG, speed: float = REF_SPEED) -> Physics:
    """Rychlost a váha -> doba kontaktu, odstup pata–špička, síla a barva."""
    kg, v = float(weight), float(speed)
    run = v >= RUN_SPEED
    if run:
        contact = max(0.15, 0.34 - 0.05 * (v - RUN_SPEED))
        gap, ratio = 0.08 * contact, 0.35
    else:
        contact = max(0.3, 0.7 - 0.1 * v)
        gap, ratio = 0.12 * contact, 0.6
    force = (kg / REF_KG) * _grf(v) / _grf(REF_SPEED)
    ref_contact = 0.7 - 0.1 * REF_SPEED
    return Physics(
        run=run, contact=contact, gap=gap, ratio=ratio,
        level=force ** 0.6,
        pitch=(REF_KG / kg) ** 0.25,
        bright=float(np.clip(1.0 + 0.12 * (v - REF_SPEED), 0.85, 1.6)),
        decay=(kg / REF_KG) ** 0.3 * (contact / ref_contact) ** 0.3)


def pink(n: int, rng: np.random.Generator, beta: float = PINK_BETA) -> np.ndarray:
    """`n` hodnot šumu se spektrem 1/f^β, nulový průměr, jednotkový rozptyl."""
    if n <= 0:
        return np.zeros(0)
    if n < 8:
        x = rng.standard_normal(n)
    else:
        spec = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n)
        f[0] = f[1]
        x = np.fft.irfft(spec * f ** (-beta / 2.0), n)
    x = x - x.mean()
    sd = float(x.std())
    return x / sd if sd > 0 else x


class Walker:
    """Jeden chodec: stálé rozdíly levé a pravé nohy ze `seed`."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(1_000_003 + int(seed))
        half = {k: 0.5 * float(rng.standard_normal()) * s for k, s in BIAS.items()}
        # noha 0 (levá) dostane +polovinu rozdílu, pravá -polovinu
        self.bias = ({k: v for k, v in half.items()}, {k: -v for k, v in half.items()})

    def factor(self, foot: int, name: str) -> float:
        return math.exp(self.bias[int(foot) % 2][name])

    def shift(self, foot: int) -> float:
        """Stálé zpoždění nohy v podílu intervalu (kladné = později)."""
        return self.bias[int(foot) % 2]["time"]


@dataclass(frozen=True)
class StepSpec:
    """Naplánovaný krok: čas v beatech, noha a parametry pro `human._step`."""
    beat: float
    foot: int
    weight: float
    speed: float
    level: float
    gap: float
    ratio: float
    bright: float
    pitch: float

    def params(self, surface: str, variant: int) -> tuple:
        """Parametry hitu `step` (variant poslední, jako u všech generátorů)."""
        r = lambda x: round(float(x), 4)  # noqa: E731 — stabilní klíč cache
        return (surface, int(self.foot), r(self.weight), r(self.speed), r(self.level),
                r(self.gap), r(self.ratio), r(self.bright), r(self.pitch), int(variant))


def plan(start: float, end: float, bpm: float, speed: Callable[[float], float],
         weight: float = REF_KG, every: float | None = None,
         jitter: float = JITTER, walker: Walker | None = None,
         seed: int = 0) -> list[StepSpec]:
    """Kroky v rozsahu beatů `[start, end)`.

    Interval je `every` beatů, nebo (bez `every`) z kadence při rychlosti
    `speed(beat)` m/s — zrychlení tedy zhustí kroky. Krok se posune od
    mřížky o 1/f šum, jehož interval má variační koeficient `jitter`.
    Parametry kroku jsou korelované přes kroky (OU, `CORR`) a nesou stálé
    rozdíly nohou z `walker`.
    """
    if end <= start:
        return []
    walker = walker or Walker(seed)
    spb = 60.0 / float(bpm)

    def interval(beat: float) -> float:
        if every is not None:
            return float(every)
        return 1.0 / cadence(speed(beat)) / spb

    grid = [float(start)]
    while True:
        nxt = grid[-1] + interval(grid[-1])
        if nxt >= end - 1e-9:
            break
        grid.append(nxt)
    n = len(grid)
    gaps = np.array([interval(b) for b in grid])
    rng = np.random.default_rng(2_000_029 + int(seed))
    x = pink(n, rng)
    if n >= 3 and jitter > 0:
        # odchylka od mřížky e_k; rozptyl intervalu (e_k - e_{k-1}) = jitter
        sd = float(np.std(np.diff(x)))
        e = x * (float(jitter) / sd if sd > 0 else 0.0)
    else:
        e = x * float(jitter)
    e = np.clip(e, -MAX_SHIFT, MAX_SHIFT)

    z = {k: float(rng.standard_normal()) for k in SPREAD}
    innov = math.sqrt(1.0 - CORR * CORR)
    out: list[StepSpec] = []
    for k, beat in enumerate(grid):
        if k:
            z = {name: CORR * v + innov * float(rng.standard_normal())
                 for name, v in z.items()}
        foot = k % 2
        t = beat + (float(e[k]) + walker.shift(foot)) * gaps[k]
        bright_z = (BRIGHT_FROM_LEVEL * z["level"]
                    + math.sqrt(1.0 - BRIGHT_FROM_LEVEL ** 2) * z["bright"])
        out.append(StepSpec(
            beat=max(0.0, t), foot=foot, weight=float(weight),
            speed=float(speed(beat)),
            level=math.exp(SPREAD["level"] * z["level"]) * walker.factor(foot, "level"),
            gap=math.exp(SPREAD["gap"] * z["gap"]) * walker.factor(foot, "gap"),
            ratio=math.exp(SPREAD["ratio"] * z["ratio"]),
            bright=math.exp(SPREAD["bright"] * bright_z) * walker.factor(foot, "bright"),
            pitch=math.exp(SPREAD["pitch"] * z["pitch"]) * walker.factor(foot, "pitch")))
    return out


def interval_cv(beats) -> float:
    """Variační koeficient intervalů mezi kroky (0 = metronom)."""
    d = np.diff(np.sort(np.asarray(beats, dtype=np.float64)))
    if d.size < 2 or float(d.mean()) <= 0:
        return 0.0
    return float(d.std() / d.mean())
