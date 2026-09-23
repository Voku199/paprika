"""Hotové mastering řetězy na `song.master`.

    s.master.fx(*mastering("cinematic"))

Každé volání vrací nové instance efektů. Řetěz je jemný — lepí mix
dohromady, ořízne rumble pod 30 Hz a přidá trochu analogové nedokonalosti
(páska), aby výsledek nezněl „z čipu“. Hlasitost řeší až limiter v renderu.
"""

from __future__ import annotations

from .fx import EQ, HighPass, LowPass
from .fx_dyn import Compressor
from .fx_lofi import BitCrush, Tape

__all__ = ["mastering", "MASTERING_STYLES"]


def _cinematic() -> list:
    return [
        HighPass(30.0),
        EQ(low=(90.0, -2.0), high=(8000.0, 1.5)),
        Compressor(threshold=-14.0, ratio=2.0, attack=0.03, release=0.25),
        Tape(wow=0.05, flutter=0.05, drive=1.2),
    ]


def _darkpop() -> list:
    return [
        HighPass(28.0),
        EQ(low=(70.0, -1.0), mid=[(350.0, -1.5, 1.0)], high=(9000.0, 2.0)),
        Compressor(threshold=-12.0, ratio=3.0, attack=0.02, release=0.18),
        Tape(wow=0.08, flutter=0.1, drive=1.8),
    ]


def _lofi() -> list:
    return [
        HighPass(40.0),
        Tape(wow=0.4, flutter=0.3, drive=1.5, hiss=0.4),
        BitCrush(bits=12, downsample=2, mix=0.5),
        LowPass(9000.0),
        Compressor(threshold=-16.0, ratio=2.5, attack=0.02, release=0.2),
    ]


_STYLES = {"cinematic": _cinematic, "darkpop": _darkpop, "lofi": _lofi}
MASTERING_STYLES = tuple(_STYLES)


def mastering(style: str = "cinematic") -> list:
    """Nový seznam efektů pro master: cinematic, darkpop nebo lofi."""
    if style not in _STYLES:
        raise KeyError(f"neznámý mastering {style!r}; k dispozici: "
                       f"{', '.join(MASTERING_STYLES)}")
    return _STYLES[style]()
