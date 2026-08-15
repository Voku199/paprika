"""Osm hotových nástrojů. Každý je obyčejný `Patch`, takže jde po vytvoření
doladit (`p = preset("lead"); p.filter.cutoff = 900`).
"""

from __future__ import annotations

from .patch import ADSR, FM, LP, Patch, Saw, Sine, Square, Tri

__all__ = ["PRESETS", "preset", "preset_names"]


def _epiano() -> Patch:
    # Rhodes: FM nosič s modulátorem, který rychle odezní, plus lehká oktáva.
    return Patch(
        osc=[FM(1.0, amp=1.0, mod=2.0, index=3.2, decay=0.45),
             Sine(1.0, amp=0.25),
             Sine(2.005, amp=0.10)],
        adsr=ADSR(0.004, 1.6, 0.22, 0.55),
        filter=LP(4200, env=0.25),
        gain=0.62,
        name="epiano",
    )


def _warmpad() -> Patch:
    return Patch(
        osc=[Saw(1.0, amp=0.5), Saw(1.006, amp=0.45), Saw(0.994, amp=0.45),
             Sine(0.5, amp=0.35)],
        adsr=ADSR(0.55, 1.2, 0.75, 1.6),
        filter=LP(1700, env=0.35),
        gain=0.5,
        name="warmpad",
    )


def _subbass() -> Patch:
    return Patch(
        osc=[Sine(1.0, amp=1.0), Sine(2.0, amp=0.18), Tri(1.0, amp=0.12)],
        adsr=ADSR(0.008, 0.25, 0.85, 0.16),
        filter=LP(600, env=0.4),
        gain=0.68,
        name="subbass",
    )


def _pluck() -> Patch:
    return Patch(
        osc=[Saw(1.0, amp=0.8), Square(1.002, amp=0.3), Sine(2.0, amp=0.15)],
        adsr=ADSR(0.003, 0.32, 0.0, 0.22),
        filter=LP(2600, env=0.85),
        gain=0.7,
        name="pluck",
    )


def _lead() -> Patch:
    return Patch(
        osc=[Square(1.0, amp=0.55), Saw(1.004, amp=0.45), Saw(0.996, amp=0.35),
             Sine(2.0, amp=0.12)],
        adsr=ADSR(0.02, 0.25, 0.65, 0.3),
        filter=LP(3200, env=0.45),
        gain=0.6,
        name="lead",
    )


def _strings() -> Patch:
    return Patch(
        osc=[Saw(1.0, amp=0.4), Saw(1.008, amp=0.4), Saw(0.992, amp=0.4),
             Saw(2.003, amp=0.15)],
        adsr=ADSR(0.28, 0.7, 0.8, 1.1),
        filter=LP(2400, env=0.3),
        gain=0.45,
        name="strings",
    )


def _bell() -> Patch:
    return Patch(
        osc=[FM(1.0, amp=1.0, mod=3.5, index=5.0, decay=0.25),
             Sine(2.76, amp=0.22),
             Sine(5.4, amp=0.08)],
        adsr=ADSR(0.002, 2.4, 0.0, 1.2),
        filter=None,
        gain=0.6,
        name="bell",
    )


def _organ() -> Patch:
    return Patch(
        osc=[Sine(1.0, amp=0.9), Sine(2.0, amp=0.5), Sine(3.0, amp=0.3),
             Sine(4.0, amp=0.18), Sine(6.0, amp=0.08)],
        adsr=ADSR(0.012, 0.05, 0.95, 0.08),
        filter=LP(5200),
        gain=0.42,
        name="organ",
    )


PRESETS = {
    "epiano": _epiano,
    "warmpad": _warmpad,
    "subbass": _subbass,
    "pluck": _pluck,
    "lead": _lead,
    "strings": _strings,
    "bell": _bell,
    "organ": _organ,
}


def preset_names() -> list[str]:
    return sorted(PRESETS)


def preset(name: str) -> Patch:
    """Vrátí novou instanci presetu, takže úpravy neovlivní ostatní stopy."""
    key = str(name).lower()
    if key not in PRESETS:
        raise KeyError(f"neznámý preset {name!r}; k dispozici: "
                       f"{', '.join(preset_names())}")
    return PRESETS[key]()
