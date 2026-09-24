"""Instalace vzorků: stažení, rozbalení, vygenerování SFZ.

Jediný modul Paprika, který sahá na síť. Stahování pokračuje: hotové
soubory se správnou velikostí se přeskočí, rozstahované leží jako `.part`.
"""

from __future__ import annotations

import math
import os
import re

import numpy as np

from .library import Instrument
from .sampler import load_wav, to_float
from .sfz import note_number

__all__ = ["InstallError", "parse_sample_name", "plan_regions", "write_sfz",
           "VSCO_SHA", "VSCO_OCTAVE_ZERO", "TARGET_RMS_DB"]

VSCO_SHA = "440300901dfe9275fd84e0b7763af1f8443ae62e"
TARGET_RMS_DB = -18.0     # max RMS nejhlasitější vrstvy prostředního kořene
# VSCO pojmenovává většinu složek s C3 = střední C (výška z jména + 12);
# tyhle používají C4 = 60. Změřeno detekcí výšky všech složek 2026-09-24.
VSCO_OCTAVE_ZERO = ("Keys/Upright Nr1", "Strings/Harp",
                    "Strings/Solo Violin/Arco Vib", "Strings/Solo Violin/Pizz")
LOOP_START, LOOP_END = 0.45, 0.90
_DYN = {"ppp": 1, "pp": 2, "p": 3, "mp": 4, "mf": 5, "f": 6, "ff": 7, "fff": 8,
        "soft": 3, "medium": 5, "loud": 7}
_NOTE = re.compile(r"^[A-G][#b]?-?\d$")


class InstallError(RuntimeError):
    """Instalace selhala — zpráva říká co a jak pokračovat."""


def parse_sample_name(filename: str, octave_shift: int = 0):
    """Jméno VSCO souboru -> `(výška, vrstva, rr)`, bez noty `None`.

    Tokeny oddělené `_`: nota (`A#3`), vrstva (`v2` nebo dynamika `mf`),
    round robin (`rr2`, `RR2` nebo koncové číslo `_1`). Před notou se
    ignoruje všechno (jméno nástroje, tvrdost paliček).
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    pitch, layer, rr = None, 0, 1
    for tok in stem.split("_"):
        low = tok.lower()
        if pitch is None:
            if _NOTE.match(tok):
                pitch = note_number(tok) + octave_shift
            continue
        if re.fullmatch(r"v\d+", low):
            layer = int(low[1:])
        elif re.fullmatch(r"rr\d+", low):
            rr = int(low[2:])
        elif low in _DYN:
            layer = _DYN[low]
        elif tok.isdigit():
            rr = int(tok)
    return None if pitch is None else (pitch, layer, rr)


def plan_regions(entries) -> list[dict]:
    """`[(cesta, výška, vrstva, rr)]` -> regiony (dict opcodů).

    Velocity se dělí rovnoměrně mezi vrstvy; klávesy vrstvy napůl mezi
    sousední kořeny, krajní o 2 půltóny dál.
    """
    layers = sorted({e[2] for e in entries})
    out = []
    for li, layer in enumerate(layers):
        lovel = 1 + (li * 127) // len(layers)
        hivel = ((li + 1) * 127) // len(layers)
        mine = [e for e in entries if e[2] == layer]
        roots = sorted({e[1] for e in mine})
        for k, root in enumerate(roots):
            lokey = roots[0] - 2 if k == 0 else (roots[k - 1] + root) // 2 + 1
            hikey = (roots[-1] + 2 if k == len(roots) - 1
                     else (root + roots[k + 1]) // 2)
            group = sorted((e for e in mine if e[1] == root),
                           key=lambda e: (e[3], e[0]))
            for pos, e in enumerate(group, start=1):
                out.append({"sample": e[0], "lokey": max(0, lokey),
                            "hikey": min(127, hikey), "pitch_keycenter": root,
                            "lovel": lovel, "hivel": hivel,
                            "seq_length": len(group), "seq_position": pos})
    return out


def _loudest_rms_db(regions: list, base: str) -> float:
    """Max RMS (100 ms okna) nejhlasitější vrstvy u prostředního kořene."""
    top = max(r["hivel"] for r in regions)
    layer = [r for r in regions if r["hivel"] == top]
    roots = sorted({r["pitch_keycenter"] for r in layer})
    mid = roots[len(roots) // 2]
    r = next(r for r in layer if r["pitch_keycenter"] == mid)
    sr, data = load_wav(os.path.join(base, *r["sample"].split("/")))
    x = to_float(data[: sr * 4]).mean(axis=1)
    win = max(1, int(0.1 * sr))
    n = (x.size // win) * win
    if n == 0:
        return TARGET_RMS_DB
    rms = float(np.sqrt(np.mean(x[:n].reshape(-1, win) ** 2, axis=1)).max())
    return 20.0 * math.log10(max(rms, 1e-6))


def write_sfz(inst: Instrument, art: str, root: str) -> str:
    """Vygeneruje `<root>/vsco2/sfz/<nástroj>.<art>.sfz` ze stažených WAV."""
    folder = inst.arts[art]
    base = os.path.join(root, "vsco2")
    src_dir = os.path.join(base, *folder.split("/"))
    shift = 0 if folder in VSCO_OCTAVE_ZERO else 12
    entries = []
    names = sorted(os.listdir(src_dir)) if os.path.isdir(src_dir) else []
    for fn in names:
        if not fn.lower().endswith(".wav"):
            continue
        parsed = parse_sample_name(fn, shift)
        if parsed:
            entries.append((f"{folder}/{fn}", *parsed))
    if not entries:
        raise InstallError(f"ve složce {src_dir} nejsou vzorky pro "
                           f"{inst.name}.{art} — spusť instalaci znovu")
    regions = plan_regions(entries)
    for r in regions:
        if inst.loops(art):
            n = load_wav(os.path.join(base, *r["sample"].split("/")))[1].shape[0]
            r["loop_mode"] = "loop_continuous"
            r["loop_start"] = int(n * LOOP_START)
            r["loop_end"] = int(n * LOOP_END)
        elif inst.is_one_shot(art):
            r["loop_mode"] = "one_shot"
    volume = TARGET_RMS_DB - _loudest_rms_db(regions, base)
    veltrack = 100 if len({r["lovel"] for r in regions}) == 1 else 40
    lines = ["// vygenerováno Paprikou z VSCO 2 CE (CC0) — neupravuj, přepíše se",
             "<control> default_path=../",
             f"<global> volume={volume:.2f} amp_veltrack={veltrack} "
             f"ampeg_release={inst.release:g}"]
    for r in regions:
        lines.append("<region> " + " ".join(f"{k}={v}" for k, v in r.items()))
    path = os.path.join(base, "sfz", f"{inst.name}.{art}.sfz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
