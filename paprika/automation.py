"""Automatizace: hodnota parametru, která se mění v čase.

Body jsou `(bar, value)` — takt smí být desetinný (`16.5` = půlka taktu).
Před prvním bodem platí jeho hodnota, za posledním taky. Převod na samply
se děje až v renderu (`sample`), takže změna BPM nic nerozbije.

Cíl je `vol`, `pan`, nebo `<efekt>.<parametr>`, kde `<efekt>` je `Fx.label`
(jméno třídy malými písmeny, u duplicit `lowpass2`, nebo `name=` efektu).
Samplované stopy mají navíc `dyn` — plynulou dynamiku (jako modwheel):
násobí velocity not v čase a prolíná velocity vrstvy nahrávky, takže
jeden dlouhý tón umí crescendo. Řeší ji render sampleru, ne mix.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Automation", "CURVES", "DYN", "fx_labels", "targets", "resolve"]

CURVES = ("lin", "exp", "s", "step")
MIX_TARGETS = ("vol", "pan")
DYN = "dyn"


class Automation:
    """Jedna automatizační křivka jednoho cíle."""

    def __init__(self, target: str, points, curve: str = "lin"):
        if curve not in CURVES:
            raise ValueError(f"neznámá křivka {curve!r}; k dispozici: "
                             f"{', '.join(CURVES)}")
        pts = [(float(b), float(v)) for b, v in points]
        if not pts:
            raise ValueError(f"automatizace {target!r} nemá žádný bod; "
                             f"čekám seznam (takt, hodnota)")
        for (a, _), (b, _) in zip(pts, pts[1:]):
            if b < a:
                raise ValueError(f"automatizace {target!r}: takty bodů musí jít "
                                 f"po sobě, {b:g} je před {a:g}")
        if curve == "exp" and any(v <= 0 for _, v in pts):
            raise ValueError(f"automatizace {target!r}: křivka 'exp' potřebuje "
                             f"kladné hodnoty (geometrický průběh); pro nulu "
                             f"použij 'lin' nebo 's'")
        self.target = str(target)
        self.points = pts
        self.curve = curve

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Automation {self.target} {self.curve} {self.points}>"

    def sample(self, n: int, sr: int, bpm: float, beats_per_bar: float,
               beat_offset: float = 0.0) -> np.ndarray:
        """Hodnota pro každý sample bufferu, který začíná na `beat_offset`."""
        t = beat_offset + np.arange(n, dtype=np.float64) / sr * (bpm / 60.0)
        xs = np.array([b * beats_per_bar for b, _ in self.points])
        ys = np.array([v for _, v in self.points])
        if len(xs) == 1:
            return np.full(n, ys[0])
        if self.curve == "lin":
            return np.interp(t, xs, ys)
        if self.curve == "exp":
            return np.exp(np.interp(t, xs, np.log(ys)))
        idx = np.clip(np.searchsorted(xs, t, side="right") - 1, 0, len(xs) - 1)
        if self.curve == "step":
            return ys[idx]
        # smoothstep po segmentech
        seg = np.clip(idx, 0, len(xs) - 2)
        x0, x1 = xs[seg], xs[seg + 1]
        y0, y1 = ys[seg], ys[seg + 1]
        width = np.where(x1 > x0, x1 - x0, 1.0)
        u = np.clip((t - x0) / width, 0.0, 1.0)
        u = u * u * (3.0 - 2.0 * u)
        return y0 + (y1 - y0) * u


def fx_labels(chain) -> dict:
    """Jméno pro automatizaci -> efekt. Duplicity dostanou číslo: lowpass2."""
    out: dict = {}
    for fx in chain:
        base = fx.label
        label = base
        i = 2
        while label in out:
            label = f"{base}{i}"
            i += 1
        out[label] = fx
    return out


def targets(chain, sampled: bool = False) -> list[str]:
    """Všechny cíle, které jdou na stopě s tímhle řetězem automatizovat."""
    out = list(MIX_TARGETS) + ([DYN] if sampled else [])
    for label, fx in fx_labels(chain).items():
        out.extend(f"{label}.{p}" for p in getattr(fx, "AUTOMATABLE", ()))
    return out


def resolve(owner: str, automations, chain, sampled: bool = False):
    """Rozdělí automatizace na (vol, pan, [(fx, param, Automation)]).

    `dyn` na samplované stopě přeskočí (řeší ho render sampleru).
    Neznámý cíl → `KeyError` se seznamem dostupných cílů.
    """
    labels = fx_labels(chain)
    vol = pan = None
    fx_autos = []
    for a in automations:
        if a.target == DYN:
            if sampled:
                continue
            raise KeyError(
                f"{owner!r}: cíl 'dyn' (plynulá dynamika) umí jen samplované "
                f"nástroje (grand_piano, violins, ...); k dispozici: "
                f"{', '.join(targets(chain))}")
        if a.target == "vol":
            vol = a
            continue
        if a.target == "pan":
            pan = a
            continue
        label, _, param = a.target.partition(".")
        fx = labels.get(label)
        if fx is None or param not in getattr(fx, "AUTOMATABLE", ()):
            raise KeyError(
                f"{owner!r}: neznámý cíl automatizace {a.target!r}; "
                f"k dispozici: {', '.join(targets(chain, sampled))}")
        fx_autos.append((fx, param, a))
    return vol, pan, fx_autos
