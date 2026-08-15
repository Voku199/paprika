"""Syntetizované bicí a parsing pattern stringů.

Znaky patternu::

    k  kick     s  snare    h  hi-hat   c  clap    t  tom
    o  open hat r  ride     .  pauza    _  pauza

Velké písmeno = akcent (vyšší velocity). Jeden znak = jeden krok; výchozí
krok je šestnáctina, tedy šestnáct znaků = jeden takt ve 4/4.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["DRUM_CHARS", "DrumHit", "parse_pattern", "render_hit", "drum_names"]

DRUM_CHARS = {
    "k": "kick",
    "s": "snare",
    "h": "hat",
    "o": "openhat",
    "c": "clap",
    "t": "tom",
    "r": "ride",
}

REST_CHARS = "._ -"
ACCENT_VELOCITY = 118
NORMAL_VELOCITY = 92

TWO_PI = 2.0 * math.pi


class DrumHit:
    """Jeden úder: jméno nástroje, start v beatech, velocity."""

    __slots__ = ("name", "start_beat", "velocity")

    def __init__(self, name: str, start_beat: float, velocity: int):
        self.name = name
        self.start_beat = float(start_beat)
        self.velocity = int(velocity)

    def __repr__(self) -> str:  # pragma: no cover
        return f"DrumHit({self.name} @{self.start_beat:g} vel={self.velocity})"


def drum_names() -> list[str]:
    return sorted(set(DRUM_CHARS.values()))


def parse_pattern(text: str, step_beats: float = 0.25,
                  start_beat: float = 0.0) -> list[DrumHit]:
    """Pattern string -> seznam úderů. Víc řádků = víc vrstev nad sebou."""
    hits: list[DrumHit] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for i, ch in enumerate(line):
            if ch in REST_CHARS:
                continue
            low = ch.lower()
            if low not in DRUM_CHARS:
                raise ValueError(
                    f"neznámý znak bicích {ch!r}; povolené: "
                    f"{''.join(sorted(DRUM_CHARS))} plus '.' a '_'")
            vel = ACCENT_VELOCITY if ch.isupper() else NORMAL_VELOCITY
            hits.append(DrumHit(DRUM_CHARS[low], start_beat + i * step_beats, vel))
    return hits


def pattern_steps(text: str) -> int:
    """Kolik kroků má nejdelší řádek patternu."""
    return max((len(l.strip()) for l in text.splitlines() if l.strip()), default=0)


# --------------------------------------------------------------------------
# syntéza
# --------------------------------------------------------------------------

def _t(n: int, sr: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / sr


def _noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, n)


def _fade_out(buf: np.ndarray, sr: int, sec: float = 0.002) -> np.ndarray:
    n = min(int(sec * sr), buf.size // 2)
    if n > 0:
        ramp = np.linspace(1.0, 0.0, n)
        buf[-n:] *= ramp
        buf[:n] *= ramp[::-1]
    return buf


def _kick(sr: int) -> np.ndarray:
    n = int(0.42 * sr)
    t = _t(n, sr)
    # pitch envelope 110 Hz -> 45 Hz, to dělá ten "dunk"
    f = 45.0 + 75.0 * np.exp(-t / 0.030)
    phase = TWO_PI * np.cumsum(f) / sr
    body = np.sin(phase) * np.exp(-t / 0.16)
    click = _noise(n, 7) * np.exp(-t / 0.004) * 0.25
    return _fade_out((body + click) * 0.80, sr)


def _snare(sr: int) -> np.ndarray:
    from .patch import onepole_hp, onepole_lp
    n = int(0.30 * sr)
    t = _t(n, sr)
    noise = _noise(n, 11) * np.exp(-t / 0.085)
    noise = onepole_hp(noise.astype(np.float32), 900.0, sr, poles=1)
    noise = onepole_lp(noise, 7500.0, sr, poles=1)
    tone = (np.sin(TWO_PI * 185.0 * t) * 0.6 +
            np.sin(TWO_PI * 330.0 * t) * 0.3) * np.exp(-t / 0.045)
    return _fade_out(noise.astype(np.float64) * 0.9 + tone * 0.5, sr)


def _hat(sr: int, decay: float = 0.045, seed: int = 21) -> np.ndarray:
    from .patch import onepole_hp
    n = int(max(0.05, decay * 6) * sr)
    t = _t(n, sr)
    noise = _noise(n, seed) * np.exp(-t / decay)
    noise = onepole_hp(noise.astype(np.float32), 7000.0, sr, poles=2)
    return _fade_out(noise.astype(np.float64) * 0.55, sr)


def _openhat(sr: int) -> np.ndarray:
    return _hat(sr, decay=0.28, seed=22)


def _clap(sr: int) -> np.ndarray:
    from .patch import onepole_hp, onepole_lp
    n = int(0.35 * sr)
    t = _t(n, sr)
    out = np.zeros(n)
    # tři rychlé odrazy plus ocásek — tak zní tleskání skupiny lidí
    for i, delay in enumerate((0.0, 0.011, 0.021)):
        off = int(delay * sr)
        seg = _noise(n - off, 31 + i) * np.exp(-_t(n - off, sr) / 0.014)
        out[off:] += seg * (1.0 - 0.18 * i)
    out += _noise(n, 41) * np.exp(-t / 0.12) * 0.35
    out = onepole_hp(out.astype(np.float32), 900.0, sr, poles=1)
    out = onepole_lp(out, 6000.0, sr, poles=1)
    return _fade_out(out.astype(np.float64) * 0.6, sr)


def _tom(sr: int) -> np.ndarray:
    n = int(0.38 * sr)
    t = _t(n, sr)
    f = 120.0 + 90.0 * np.exp(-t / 0.05)
    phase = TWO_PI * np.cumsum(f) / sr
    body = np.sin(phase) * np.exp(-t / 0.18)
    noise = _noise(n, 51) * np.exp(-t / 0.02) * 0.15
    return _fade_out((body + noise) * 0.8, sr)


def _ride(sr: int) -> np.ndarray:
    from .patch import onepole_hp
    n = int(0.9 * sr)
    t = _t(n, sr)
    metal = np.zeros(n)
    for k, f in enumerate((2100.0, 3170.0, 4210.0, 5390.0, 6700.0)):
        metal += np.sin(TWO_PI * f * t) * (0.6 ** k)
    metal *= np.exp(-t / 0.35)
    noise = _noise(n, 61) * np.exp(-t / 0.10) * 0.5
    out = onepole_hp((metal * 0.35 + noise).astype(np.float32), 3000.0, sr, poles=1)
    return _fade_out(out.astype(np.float64) * 0.45, sr)


_GENERATORS = {
    "kick": _kick,
    "snare": _snare,
    "hat": lambda sr: _hat(sr),
    "openhat": _openhat,
    "clap": _clap,
    "tom": _tom,
    "ride": _ride,
}

_CACHE: dict[tuple[str, int], np.ndarray] = {}


def render_hit(name: str, sr: int = 44100) -> np.ndarray:
    """Vyrenderuje (a nacachuje) jeden bicí zvuk při velocity 127."""
    key = (name, sr)
    if key not in _CACHE:
        if name not in _GENERATORS:
            raise KeyError(f"neznámý bicí nástroj {name!r}")
        _CACHE[key] = _GENERATORS[name](sr).astype(np.float32)
    return _CACHE[key]
