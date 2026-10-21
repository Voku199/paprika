"""Hudební teorie: noty, stupnice, akordy, arpeggia.

Modul nezná nic o zvuku ani o renderu. Pracuje jen s MIDI čísly (C4 = 60).
"""

from __future__ import annotations

import random

__all__ = [
    "NOTE_TO_PC",
    "PC_TO_NOTE",
    "SCALES",
    "CHORDS",
    "TheoryError",
    "pitch",
    "note_name",
    "scale",
    "scale_pcs",
    "chord",
    "invert",
    "arp",
    "in_scale",
]


class TheoryError(ValueError):
    """Neznámý akord, stupnice nebo jméno noty."""


NOTE_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4,
    "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11, "Cb": 11, "B#": 0,
}

PC_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "chromatic": list(range(12)),
}

CHORDS = {
    "": [0, 4, 7],
    "maj": [0, 4, 7],
    "M": [0, 4, 7],
    "m": [0, 3, 7],
    "min": [0, 3, 7],
    "-": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "+": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "5": [0, 7],
    "6": [0, 4, 7, 9],
    "m6": [0, 3, 7, 9],
    "min6": [0, 3, 7, 9],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "M7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "min7": [0, 3, 7, 10],
    "mmaj7": [0, 3, 7, 11],
    "dim7": [0, 3, 6, 9],
    "m7b5": [0, 3, 6, 10],
    "7sus4": [0, 5, 7, 10],
    "7b9": [0, 4, 7, 10, 13],
    "7#9": [0, 4, 7, 10, 15],
    "add9": [0, 4, 7, 14],
    "madd9": [0, 3, 7, 14],
    "9": [0, 4, 7, 10, 14],
    "maj9": [0, 4, 7, 11, 14],
    "m9": [0, 3, 7, 10, 14],
    "min9": [0, 3, 7, 10, 14],
    "11": [0, 4, 7, 10, 14, 17],
    "m11": [0, 3, 7, 10, 14, 17],
    "13": [0, 4, 7, 10, 14, 21],
    "maj13": [0, 4, 7, 11, 14, 21],
}


def pitch(name: str) -> int:
    """'C#4' -> 61. Přijímá i rovnou int nebo string s číslem."""
    if isinstance(name, (int, float)):
        return int(name)
    s = name.strip()
    if not s:
        raise TheoryError("prázdné jméno noty")
    if s.lstrip("-").isdigit():
        return int(s)
    i = 1
    if len(s) > 1 and s[1] in "#b":
        i = 2
    head, tail = s[:i], s[i:]
    head = head[0].upper() + head[1:]
    if head not in NOTE_TO_PC:
        raise TheoryError(f"neznámá nota: {name!r}")
    if not tail or not (tail.lstrip("-").isdigit()):
        raise TheoryError(f"chybí oktáva: {name!r}")
    octave = int(tail)
    # oktáva patří písmenu: B#3 je C4 (60), Cb4 je H3 (59) - ne pc modulo 12
    letter = NOTE_TO_PC[head[0]]
    return (octave + 1) * 12 + letter + {"#": 1, "b": -1}.get(head[1:], 0)


def note_name(p: int) -> str:
    """61 -> 'C#4'."""
    p = int(round(p))
    return f"{PC_TO_NOTE[p % 12]}{p // 12 - 1}"


def _root_pc(root: str | int) -> int:
    """Pitch class kořene: 'C', 'Eb', 'C#4' i MIDI číslo."""
    if not isinstance(root, str):
        return int(root) % 12
    s = root.strip()
    i = 2 if len(s) > 1 and s[1] in "#b" else 1
    head = s[0].upper() + s[1:i]
    if head not in NOTE_TO_PC:
        raise TheoryError(f"neznámý kořen: {root!r}")
    return NOTE_TO_PC[head]


def _scale_key(name: str) -> str:
    key = str(name).lower().replace(" ", "_").replace("-", "_")
    if key not in SCALES:
        raise TheoryError(f"neznámá stupnice: {name!r}")
    return key


def scale_pcs(root: str | int, name: str = "major") -> list[int]:
    """Pitch classes stupnice, tedy hodnoty 0-11."""
    key = _scale_key(name)
    root_pc = _root_pc(root)
    return sorted({(root_pc + iv) % 12 for iv in SCALES[key]})


def scale(root: str | int = "C", name: str = "major", octaves: int = 1,
          octave: int = 4) -> list[int]:
    """Stupnice jako MIDI pitche, vzestupně od rootu v dané oktávě."""
    key = _scale_key(name)
    if isinstance(root, str) and not root[-1:].isdigit():
        base = (octave + 1) * 12 + _root_pc(root)
    else:
        base = pitch(root)
    out = []
    for o in range(octaves):
        out.extend(base + 12 * o + iv for iv in SCALES[key])
    out.append(base + 12 * octaves)
    return out


def _split_chord(name: str) -> tuple[str, int | None, str]:
    s = name.strip()
    if not s:
        raise TheoryError("prázdné jméno akordu")
    i = 1
    if len(s) > 1 and s[1] in "#b":
        i = 2
    root = s[0].upper() + s[1:i]
    if root not in NOTE_TO_PC:
        raise TheoryError(f"neznámý základ akordu: {name!r}")
    rest = s[i:]
    # Celý zbytek je známá kvalita ("Cmaj7", "C7", "C5") -> žádná oktáva.
    if rest in CHORDS:
        return root, None, rest
    # Jinak může být první číslice oktáva ("C3maj7", "C3").
    if rest and rest[0].isdigit() and rest[1:] in CHORDS:
        return root, int(rest[0]), rest[1:]
    return root, None, rest


def chord(name: str, octave: int = 4, inv: int = 0, oct: int = 0) -> list[int]:
    """'Cmaj7' -> [60, 64, 67, 71].

    `octave` je oktáva základního tónu, `oct` posune celý akord o oktávy,
    `inv` udělá obrat (nejnižší tón nahoru).
    """
    root, oct_in_name, quality = _split_chord(name)
    if oct_in_name is not None:
        octave = oct_in_name
    if quality not in CHORDS:
        raise TheoryError(f"neznámý akord: {name!r} (kvalita {quality!r})")
    base = (octave + 1 + oct) * 12 + NOTE_TO_PC[root]
    return invert([base + iv for iv in CHORDS[quality]], inv)


def invert(pitches: list[int], n: int = 0) -> list[int]:
    """Obrat akordu. Kladné n zvedá spodní tóny, záporné sklápí horní."""
    out = list(pitches)
    for _ in range(abs(int(n))):
        if n > 0:
            out = out[1:] + [out[0] + 12]
        else:
            out = [out[-1] - 12] + out[:-1]
    return out


def arp(pitches: list[int], pattern: str = "up", steps: int | None = None,
        octaves: int = 1, seed: int | None = None) -> list[int]:
    """Rozloží akord na sekvenci pitchů.

    pattern: up | down | updown | downup | random
    """
    pool = []
    for o in range(max(1, int(octaves))):
        pool.extend(p + 12 * o for p in pitches)
    pool.sort()
    p = pattern.lower()
    if p == "up":
        seq = pool
    elif p == "down":
        seq = pool[::-1]
    elif p == "updown":
        seq = pool + pool[-2:0:-1] if len(pool) > 2 else pool + pool[::-1]
    elif p == "downup":
        rev = pool[::-1]
        seq = rev + rev[-2:0:-1] if len(pool) > 2 else rev + pool
    elif p == "random":
        rng = random.Random(seed)
        n = steps or len(pool)
        return [rng.choice(pool) for _ in range(n)]
    else:
        raise TheoryError(f"neznámý arp pattern: {pattern!r}")
    if steps is None:
        return list(seq)
    return [seq[i % len(seq)] for i in range(steps)]


def in_scale(p: int, root: str | int, name: str) -> bool:
    """Leží pitch ve stupnici?"""
    try:
        pcs = scale_pcs(root, name)
    except TheoryError:
        return True
    return p % 12 in pcs
