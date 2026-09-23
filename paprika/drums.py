"""Syntetizované bicí a parsing pattern stringů.

Znaky patternu::

    k  kick     s  snare    h  hi-hat   c  clap    t  tom
    o  open hat r  ride     x  rim      b  boom    y  shaker
    p  perc     .  pauza    _  pauza

Jak který nástroj zní, určuje bicí sada (`kits.py`): stejný pattern zní
v `trap` jako 808 a v `cinematic` jako orchestr s taiko.

Velké písmeno = akcent (vyšší velocity). Jeden znak = jeden krok; výchozí
krok je šestnáctina, tedy šestnáct znaků = jeden takt ve 4/4.
"""

from __future__ import annotations

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
    "x": "rim",
    "b": "boom",
    "y": "shaker",
    "p": "perc",
}

REST_CHARS = "._ -"
ACCENT_VELOCITY = 118
NORMAL_VELOCITY = 92


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
# render
# --------------------------------------------------------------------------

_CACHE: dict[tuple[str, str, int], np.ndarray] = {}


def render_hit(name: str, sr: int = 44100, kit: str = "classic") -> np.ndarray:
    """Vyrenderuje (a nacachuje) jeden bicí zvuk při velocity 127.

    Sada, která nástroj nemá, použije ten z `classic`.
    """
    from .kits import KITS
    if kit not in KITS:
        raise KeyError(f"neznámá bicí sada {kit!r}; k dispozici: "
                       f"{', '.join(KITS)}")
    key = (kit, name, sr)
    if key not in _CACHE:
        gen = KITS[kit].get(name) or KITS["classic"].get(name)
        if gen is None:
            raise KeyError(f"neznámý bicí nástroj {name!r}; k dispozici: "
                           f"{', '.join(drum_names())}")
        _CACHE[key] = np.asarray(gen(sr), dtype=np.float32)
    return _CACHE[key]
