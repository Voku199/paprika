"""Harmonický plán: akordy po taktech jako jména (``Dm``, ``Bbmaj7``,
``C/E``) nebo jako římské číslice vůči tónině (``i``, ``VI``, ``V7``,
``V/V``).

Modul pracuje jen s pitch classes (0–11) a MIDI čísly, stejně jako
`theory.py`. Nezná `Song`, audio ani render.

Gramatika plánu kopíruje notaci not::

    Dm | Bb | Gm | Gm:2 A:2      | odděluje takty a hlídá délky
    Dm Bb Gm A                   bez | je každý akord jeden takt
    Gm A                         tokeny bez délky si dělí takt rovným dílem
    _                            bez akordu (jen bicí, pauza)

Jméno akordu je základ + kvalita z `theory.CHORDS`, lomítko určuje bas
(``Dm/F``). Římská číslice: velké písmeno = durový, malé = mollový akord,
``o`` zmenšený, ``ø`` malý zmenšený septakord, ``+`` zvětšený; ``7``
diatonická septima, ``6``/``64`` obraty kvintakordu, ``65``/``43``/``42``
obraty septakordu (pozor: u jména akordu je ``C6`` přidaná sexta, u římské
číslice ``I6`` sextakord — přidaná sexta je ``Iadd6``). Lomítko u římské
číslice je vedlejší akord: ``V/V``, ``V7/iv``, ``viio7/V``.

Stupně v mollu se počítají od přirozeného mollu (``VI`` = Bb v d moll),
``V`` je durová dominanta a ``viio`` stojí na citlivém tónu.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import theory
from .theory import CHORDS, NOTE_TO_PC, SCALES, TheoryError

__all__ = ["HarmonyError", "Chord", "ChordSpan", "parse_chord", "parse_harmony",
           "roman_of", "key_signature", "pc_name", "cadence", "is_minor"]


class HarmonyError(ValueError):
    """Chyba v harmonickém plánu: neznámý akord, špatné délky v taktu."""


_MINOR_FAMILY = {"minor", "aeolian", "harmonic_minor", "melodic_minor",
                 "pentatonic_minor", "blues"}
# mody: o kolik půltónů leží tónika nad tónikou mateřské durové stupnice
_MODE_OFFSET = {"dorian": 2, "phrygian": 4, "lydian": 5, "mixolydian": 7,
                "locrian": 11}
# mateřská durová tónika -> počet křížků (+) / béček (-)
_SF = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3,
       10: -2, 5: -1}

_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
# C dur / a moll: vypůjčené akordy jsou béčkové, vedlejší dominanty křížkové
_NEUTRAL_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]

_LETTERS = "CDEFGAB"
_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII"]
_ROMAN_DEG = {n.lower(): i for i, n in enumerate(_NUMERALS)}

_NAME_RE = re.compile(
    r"^(?P<root>[A-G][#b]?)(?P<qual>[^/]*)(?:/(?P<bass>[A-G][#b]?))?$")
_ROMAN_RE = re.compile(
    r"^(?P<acc>[b#]?)(?P<num>VII|VI|IV|V|III|II|I|vii|vi|iv|v|iii|ii|i)"
    r"(?P<mark>[o°ø+]?)(?P<suf>[0-9a-z#]*)$")

_TRIAD_QUALITIES = {"", "maj", "M", "m", "min", "-", "dim", "aug", "+"}
_SEVENTH_QUALITIES = {"7", "maj7", "M7", "m7", "min7", "mmaj7", "dim7", "m7b5"}
# obraty: číslice -> (septakord?, index tónu v basu)
_FIGURES = {"6": (False, 1), "64": (False, 2), "7": (True, 0), "65": (True, 1),
            "43": (True, 2), "42": (True, 3), "2": (True, 3)}
# explicitní přípony: přípona -> (kvalita pro velkou číslici, pro malou)
_SUFFIX = {
    "maj7": ("maj7", "mmaj7"),
    "9": ("9", "m9"), "maj9": ("maj9", None),
    "11": ("11", "m11"), "13": ("13", None), "maj13": ("maj13", None),
    "add9": ("add9", "madd9"), "add6": ("6", "m6"),
    "sus2": ("sus2", "sus2"), "sus4": ("sus4", "sus4"), "sus": ("sus4", "sus4"),
    "7sus4": ("7sus4", "7sus4"), "5": ("5", "5"),
    "7b9": ("7b9", None), "7#9": ("7#9", None),
}
# kvalita -> přípona římské číslice (velikost písmene řeší tercie)
_ROMAN_SUFFIX = {
    "": "", "maj": "", "M": "", "m": "", "min": "", "-": "",
    "dim": "o", "dim7": "o7", "m7b5": "ø7", "aug": "+", "+": "+",
    "7": "7", "maj7": "maj7", "M7": "maj7", "m7": "7", "min7": "7",
    "mmaj7": "maj7", "6": "add6", "m6": "add6", "min6": "add6",
    "9": "9", "m9": "9", "min9": "9", "maj9": "maj9", "11": "11", "m11": "11",
    "13": "13", "maj13": "maj13", "add9": "add9", "madd9": "add9",
    "sus2": "sus2", "sus4": "sus4", "7sus4": "7sus4", "5": "5",
    "7b9": "7b9", "7#9": "7#9",
}


@dataclass(frozen=True)
class Chord:
    """Akord bez oktávy: základ, kvalita z `theory.CHORDS`, bas (pitch class)
    a popisek tak, jak ho uvidí autor (``"Dm/F"``)."""

    root: int
    quality: str
    bass: int
    label: str

    @property
    def intervals(self) -> list[int]:
        return [i % 12 for i in CHORDS[self.quality]]

    @property
    def pcs(self) -> frozenset:
        """Pitch classes akordu včetně lomeného basu."""
        return frozenset((self.root + i) % 12 for i in CHORDS[self.quality]) \
            | {self.bass}

    @property
    def third_pc(self) -> int | None:
        """Tercie akordu (velká má přednost), None u sus a power chordu."""
        ivs = self.intervals
        for third in (4, 3):
            if third in ivs:
                return (self.root + third) % 12
        return None

    @property
    def has_third(self) -> bool:
        return self.third_pc is not None


@dataclass
class ChordSpan:
    """Akord plánu na časové ose. `chord` je None pro ``_`` (bez akordu)."""

    start_beat: float
    dur_beats: float
    chord: Chord | None
    source: str = ""

    @property
    def end_beat(self) -> float:
        return self.start_beat + self.dur_beats


# --------------------------------------------------------------------------
# tónina
# --------------------------------------------------------------------------

def _tonic(key) -> int:
    try:
        return theory._root_pc(key)
    except TheoryError as e:
        raise HarmonyError(str(e)) from None


def _scale(scale: str) -> str:
    try:
        return theory._scale_key(scale)
    except TheoryError as e:
        raise HarmonyError(str(e)) from None


def is_minor(scale: str) -> bool:
    """Mollová rodina (přirozený, harmonický, melodický moll, mollová
    pentatonika, blues) — tam platí mollové konvence římských číslic."""
    return _scale(scale) in _MINOR_FAMILY


def _degrees(scale: str) -> list[int]:
    """Sedm stupňů, od kterých se počítají římské číslice."""
    k = _scale(scale)
    if k in _MINOR_FAMILY:
        return list(SCALES["minor"])
    iv = SCALES[k]
    return list(iv) if len(iv) == 7 else list(SCALES["major"])


def _diatonic(tonic: int, scale: str) -> set[int]:
    """Pitch classes, které se v tónině berou jako vlastní (v mollu i citlivý
    tón, v melodickém mollu i zvýšená sexta)."""
    pcs = {(tonic + d) % 12 for d in _degrees(scale)}
    if is_minor(scale):
        pcs.add((tonic + 11) % 12)
        if _scale(scale) == "melodic_minor":
            pcs.add((tonic + 9) % 12)
    return pcs


def key_signature(key, scale: str = "major") -> tuple[int, int]:
    """(sf, minor) jako v MIDI: sf = počet křížků (+) nebo béček (-)."""
    tonic = _tonic(key)
    k = _scale(scale)
    minor = 1 if k in _MINOR_FAMILY else 0
    if minor:
        parent = (tonic + 3) % 12
    else:
        parent = (tonic - _MODE_OFFSET.get(k, 0)) % 12
    sf = _SF[parent]
    name = str(key)
    # enharmonické dvojice: Gb/F#, Db/C#, Cb/B podle toho, jak je tónina psaná
    if parent == 6 and "b" in name[1:2]:
        sf = -6
    elif parent == 1 and "#" in name[1:2]:
        sf = 7
    elif parent == 11 and "b" in name[1:2]:
        sf = -7
    return sf, minor


def pc_name(pc: int, key="C", scale: str = "major") -> str:
    """Jméno tónu hláskované podle stupňů tóniny: Bb v F dur, C# (citlivý
    tón) v d moll, Cb v Gb dur. Nestupňové tóny: v mollu zvýšené (E v c moll,
    F#, A, B), jinak snížené (Db, Eb, Ab, Bb) kromě zvýšené kvarty (F# v C)."""
    pc = int(pc) % 12
    letter = str(key).strip()[:1].upper()
    if letter not in _LETTERS:
        sf, _ = key_signature(key, scale)
        table = _FLAT_NAMES if sf < 0 else (_SHARP_NAMES if sf > 0 else _NEUTRAL_NAMES)
        return table[pc]
    tonic = _tonic(key)
    degrees = _degrees(scale)
    rel = (pc - tonic) % 12
    if rel in degrees:
        deg = degrees.index(rel)
    else:
        lower = (rel - 1) % 12 in degrees
        upper = (rel + 1) % 12 in degrees
        if (_prefers_sharp(rel, scale) and lower) or not upper:
            deg = degrees.index((rel - 1) % 12)
        else:
            deg = degrees.index((rel + 1) % 12)
    name = _LETTERS[(_LETTERS.index(letter) + deg) % 7]
    diff = (pc - NOTE_TO_PC[name]) % 12
    diff = diff - 12 if diff > 6 else diff
    acc = {0: "", 1: "#", -1: "b"}.get(diff)
    if acc is None:
        # dvojitý posuvek (Ebb v Gb dur) je správně, ale nečitelný
        sf, _ = key_signature(key, scale)
        return (_FLAT_NAMES if sf < 0 else _SHARP_NAMES)[pc]
    return name + acc


def _prefers_sharp(rel: int, scale: str) -> bool:
    """Nestupňový tón: v mollu zvýšené stupně (III, IV, VI, VII), v duru jen
    zvýšená kvarta; ostatní se píšou snížené (bII, bIII, bVI, bVII)."""
    return rel in ((4, 6, 9, 11) if is_minor(scale) else (6,))


# --------------------------------------------------------------------------
# parsování akordů
# --------------------------------------------------------------------------

def _known_qualities() -> str:
    return ", ".join(q for q in CHORDS if q) + " (prázdná = dur)"


def _named(tok: str) -> Chord:
    m = _NAME_RE.match(tok)
    if not m:
        raise HarmonyError(f"nerozumím akordu {tok!r} — čekám jméno (Dm, Bbmaj7, "
                           f"C/E) nebo římskou číslici (i, VI, V7, V/V)")
    qual = m.group("qual")
    if qual not in CHORDS:
        raise HarmonyError(f"neznámá kvalita akordu {tok!r} ({qual!r}); "
                           f"známé: {_known_qualities()}")
    root = NOTE_TO_PC[m.group("root")]
    bass = NOTE_TO_PC[m.group("bass")] if m.group("bass") else root
    return Chord(root, qual, bass, tok)


def _roman_quality(tok, upper, mark, suf, root, diatonic, chromatic):
    """-> (kvalita, index tónu v basu)."""
    if suf in _FIGURES:
        seventh, inv = _FIGURES[suf]
    elif suf in _SUFFIX:
        if mark:
            raise HarmonyError(f"{tok!r}: značku {mark!r} nejde kombinovat "
                               f"s příponou {suf!r}")
        quality = _SUFFIX[suf][0 if upper else 1]
        if quality is None:
            raise HarmonyError(f"{tok!r}: přípona {suf!r} nejde s malou číslicí")
        return quality, 0
    elif suf == "":
        seventh, inv = False, 0
    else:
        raise HarmonyError(f"{tok!r}: neznámá přípona {suf!r}; známé: obraty "
                           f"{', '.join(_FIGURES)}, " + ", ".join(_SUFFIX))
    if mark == "ø":
        return "m7b5", inv
    if mark in ("o", "°"):
        return ("dim7" if seventh else "dim"), inv
    if mark == "+":
        if seventh:
            raise HarmonyError(f"{tok!r}: zvětšený septakord paprika nezná")
        return "aug", inv
    if not seventh:
        return ("" if upper else "m"), inv
    if not upper:
        return "m7", inv
    # velká číslice + 7: diatonická velká septima -> maj7, jinak dominantní 7
    if not chromatic and (root + 11) % 12 in diatonic:
        return "maj7", inv
    return "7", inv


def _roman_core(tok: str, tonic: int, scale: str) -> tuple[int, str, int]:
    """Římská číslice bez lomítka -> (základ, kvalita, bas)."""
    m = _ROMAN_RE.match(tok)
    if not m:
        raise HarmonyError(f"nerozumím akordu {tok!r} — čekám jméno (Dm, Bbmaj7, "
                           f"C/E) nebo římskou číslici (i, VI, V7, V/V)")
    num, acc, mark = m.group("num"), m.group("acc"), m.group("mark")
    deg = _ROMAN_DEG[num.lower()]
    shift = {"b": -1, "#": 1}.get(acc, 0)
    root = (tonic + _degrees(scale)[deg] + shift) % 12
    if is_minor(scale) and deg == 6 and mark in ("o", "°", "ø") and not acc:
        root = (tonic + 11) % 12          # viio v mollu stojí na citlivém tónu
    quality, inv = _roman_quality(tok, num.isupper(), mark, m.group("suf"), root,
                                  _diatonic(tonic, scale), bool(acc))
    ivs = CHORDS[quality]
    if inv >= len(ivs):
        raise HarmonyError(f"{tok!r}: akord {quality or 'dur'} nemá {inv}. obrat")
    return root, quality, (root + ivs[inv]) % 12


def _label(root: int, quality: str, bass: int, key, scale: str) -> str:
    name = pc_name(root, key, scale) + quality
    if bass != root:
        name += "/" + pc_name(bass, key, scale)
    return name


def parse_chord(token: str, key="C", scale: str = "major") -> Chord:
    """Jeden akord plánu: jméno (``Dm/F``) nebo římská číslice (``V7/V``)."""
    tok = str(token).strip()
    if not tok:
        raise HarmonyError("prázdný akord")
    tonic = _tonic(key)
    if tok[0] in "ABCDEFG":
        return _named(tok)
    head, slash, target = tok.partition("/")
    if slash:
        if not target:
            raise HarmonyError(f"{tok!r}: za lomítkem chybí cílový stupeň (V/V)")
        t_root, t_quality, _ = _roman_core(target, tonic, scale)
        t_scale = "major" if 4 in [i % 12 for i in CHORDS[t_quality]] else "minor"
        root, quality, bass = _roman_core(head, t_root, t_scale)
        # vedlejší akord se hláskuje v tónině cíle: viio/ii v C dur je C#dim
        spell_key, spell_scale = pc_name(t_root, key, scale), t_scale
    else:
        root, quality, bass = _roman_core(head, tonic, scale)
        spell_key, spell_scale = key, scale
    return Chord(root, quality, bass,
                 _label(root, quality, bass, spell_key, spell_scale))


def _split_dur(tok: str, bar_no: int) -> tuple[str, float | None]:
    name, sep, dur = tok.partition(":")
    if not sep:
        return name, None
    try:
        d = float(dur)
    except ValueError:
        raise HarmonyError(f"takt {bar_no}: špatná délka v {tok!r}") from None
    if d <= 0:
        raise HarmonyError(f"takt {bar_no}: délka musí být kladná ({tok!r})")
    return name, d


def _span(start, dur, name, key, scale, bar_no) -> ChordSpan:
    if name == "_":
        return ChordSpan(start, dur, None, name)
    try:
        return ChordSpan(start, dur, parse_chord(name, key, scale), name)
    except HarmonyError as e:
        raise HarmonyError(f"takt {bar_no}: {e}") from None


def parse_harmony(text: str, key="C", scale: str = "major",
                  beats_per_bar: float = 4.0, bar_offset: int = 0) -> list[ChordSpan]:
    """Naparsuje plán na `ChordSpan`y se startem od nuly (včetně ``_``).

    `bar_offset` se přičte k číslu taktu v chybové hlášce.
    """
    if not isinstance(text, str):
        raise HarmonyError(f"očekávám string, dostal jsem {type(text).__name__}")
    bpb = float(beats_per_bar)
    spans: list[ChordSpan] = []
    if "|" not in text:
        cursor = 0.0
        for i, tok in enumerate(text.split()):
            bar_no = int(cursor // bpb) + bar_offset + 1
            name, dur = _split_dur(tok, bar_no)
            d = bpb if dur is None else dur
            spans.append(_span(cursor, d, name, key, scale, bar_no))
            cursor += d
        return spans

    segments = text.split("|")
    if segments and not segments[0].strip():
        segments = segments[1:]
    if segments and not segments[-1].strip():
        segments = segments[:-1]
    for bar_i, segment in enumerate(segments):
        bar_no = bar_i + bar_offset + 1
        start = bar_i * bpb
        toks = segment.split()
        if not toks:
            raise HarmonyError(f"takt {bar_no}: prázdný takt — napiš akord nebo _")
        parsed = [_split_dur(t, bar_no) for t in toks]
        fixed = sum(d for _, d in parsed if d is not None)
        free = sum(1 for _, d in parsed if d is None)
        share = 0.0
        if free:
            share = (bpb - fixed) / free
            if share <= 1e-9:
                raise HarmonyError(
                    f"takt {bar_no}: akordy s délkou zaberou {fixed:g} beatů, "
                    f"na ostatní nezbývá místo (segment {segment.strip()!r})")
        elif abs(fixed - bpb) > 1e-6:
            raise HarmonyError(
                f"takt {bar_no}: délky akordů dávají {fixed:g} beatů, čekám "
                f"{bpb:g} (segment {segment.strip()!r})")
        cursor = start
        for name, dur in parsed:
            d = share if dur is None else dur
            spans.append(_span(cursor, d, name, key, scale, bar_no))
            cursor += d
    return spans


# --------------------------------------------------------------------------
# rozbor: římská číslice, kadence
# --------------------------------------------------------------------------

def _figure(chord: Chord) -> str | None:
    """Číslice obratu, None když bas není tón akordu nebo kvalita je jiná."""
    if chord.bass == chord.root:
        return ""
    ivs = CHORDS[chord.quality]
    rel = (chord.bass - chord.root) % 12
    pos = [i % 12 for i in ivs].index(rel) if rel in [i % 12 for i in ivs] else -1
    if chord.quality in _TRIAD_QUALITIES and pos in (1, 2):
        return {1: "6", 2: "64"}[pos]
    if chord.quality in _SEVENTH_QUALITIES and pos in (1, 2, 3):
        return {1: "65", 2: "43", 3: "42"}[pos]
    return None


def _numeral(chord: Chord, tonic: int, scale: str) -> str:
    degrees = _degrees(scale)
    rel = (chord.root - tonic) % 12
    acc = ""
    if rel in degrees:
        deg = degrees.index(rel)
    elif is_minor(scale) and rel == 11:
        deg = 6                            # citlivý tón v mollu: viio, ne #viio
    else:
        flat_ok = (rel + 1) % 12 in degrees
        sharp_ok = (rel - 1) % 12 in degrees
        if (_prefers_sharp(rel, scale) and sharp_ok) or not flat_ok:
            acc, deg = "#", degrees.index((rel - 1) % 12)
        else:
            acc, deg = "b", degrees.index((rel + 1) % 12)
    num = _NUMERALS[deg]
    ivs = chord.intervals
    if 3 in ivs and 4 not in ivs:
        num = num.lower()
    suffix = _ROMAN_SUFFIX.get(chord.quality, chord.quality)
    fig = _figure(chord)
    if fig:
        suffix = suffix.replace("7", "") + fig if "7" in suffix else suffix + fig
    return acc + num + suffix


def roman_of(chord: Chord, key="C", scale: str = "major") -> str:
    """Římská číslice akordu v tónině: ``i``, ``V7``, ``i6``, ``V/V``, ``bVII``."""
    tonic = _tonic(key)
    dia = _diatonic(tonic, scale)
    if chord.pcs <= dia:
        return _numeral(chord, tonic, scale)
    degrees = _degrees(scale)
    # vedlejší dominanta (o kvintu nad stupněm) nebo vedlejší zmenšený akord
    # (o půltón pod stupněm); cílem není tónika
    if chord.quality in ("", "maj", "M", "7"):
        target, head = (chord.root - 7) % 12, "V"
    elif chord.quality in ("dim", "dim7", "m7b5"):
        target, head = (chord.root + 1) % 12, "vii" + _ROMAN_SUFFIX[chord.quality]
    else:
        target = None
    if target is not None and target != tonic and (target - tonic) % 12 in degrees:
        if chord.quality == "7":
            head = "V7"
        fig = _figure(chord)
        if fig:
            head = head.replace("7", "") + fig if "7" in head else head + fig
        return f"{head}/{_target_numeral(degrees.index((target - tonic) % 12), scale)}"
    return _numeral(chord, tonic, scale)


def _target_numeral(deg: int, scale: str) -> str:
    """Římská číslice diatonického kvintakordu na stupni (cíl V/x)."""
    if is_minor(scale) and deg == 4:
        return "V"                        # V/V v mollu, ne V/v
    degrees = _degrees(scale)
    third = (degrees[(deg + 2) % 7] - degrees[deg]) % 12
    fifth = (degrees[(deg + 4) % 7] - degrees[deg]) % 12
    num = _NUMERALS[deg] if third == 4 else _NUMERALS[deg].lower()
    return num + ("o" if fifth == 6 else "")


def cadence(prev: Chord | None, last: Chord, key="C",
            scale: str = "major") -> str | None:
    """Druh kadence posledních dvou akordů úseku, None když žádná."""
    tonic = _tonic(key)
    rel_last = (last.root - tonic) % 12
    rel_prev = None if prev is None else (prev.root - tonic) % 12
    dominant = (prev is not None and rel_prev == 7
                and (tonic + 11) % 12 in prev.pcs)
    if rel_last == 0 and dominant:
        return "autentická"
    if rel_last == 0 and rel_prev == 5:
        return "plagální"
    if rel_last == 7 and (tonic + 11) % 12 in last.pcs:
        return "půlová"
    if dominant and rel_last in (8, 9):
        return "klamná"
    return None
