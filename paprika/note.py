"""Nota a parsing notačních řetězců typu ``"C#4:1.5"`` nebo ``"E5:1 G5:1 A5:2"``.

Gramatika jednoho tokenu::

    C#4:1.5        nota, oktáva, dvojtečka, délka v beatech
    C4             délka se vezme z `default_dur` (výchozí 1 beat)
    (C4 E4 G4):2   akord — všechny tóny naráz
    _:2            pauza
    ~:1            prodloužení předchozí noty o 1 beat
    C4:1@110       volitelná velocity za zavináčem

Svislítko ``|`` odděluje takty a slouží jako kontrola délek: když součet
délek v taktu nesedí na `beats_per_bar`, letí `NoteLengthError`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .theory import pitch as _pitch, note_name

__all__ = ["Note", "NoteLengthError", "NoteParseError", "parse", "DEFAULT_VELOCITY"]

DEFAULT_VELOCITY = 100

_NOTE_RE = re.compile(
    r"^(?P<name>[A-Ga-g][#b]?-?\d+)(?::(?P<dur>\d*\.?\d+))?(?:@(?P<vel>\d+))?$")
_CHORD_RE = re.compile(
    r"^\((?P<body>[^)]*)\)(?::(?P<dur>\d*\.?\d+))?(?:@(?P<vel>\d+))?$")
_REST_RE = re.compile(r"^[_.](?::(?P<dur>\d*\.?\d+))?$")
_TIE_RE = re.compile(r"^~(?::(?P<dur>\d*\.?\d+))?$")


class NoteParseError(ValueError):
    """Token, kterému parser nerozumí."""


class NoteLengthError(ValueError):
    """Součet délek v taktu neodpovídá `beats_per_bar`."""


@dataclass
class Note:
    """Jedna nota. `start_beat` je vždy relativní k začátku klipu."""

    pitch: int
    start_beat: float
    dur_beats: float
    velocity: int = DEFAULT_VELOCITY

    def copy(self) -> "Note":
        return Note(self.pitch, self.start_beat, self.dur_beats, self.velocity)

    @property
    def end_beat(self) -> float:
        return self.start_beat + self.dur_beats

    @property
    def name(self) -> str:
        return note_name(self.pitch)

    def __repr__(self) -> str:  # pragma: no cover - jen pro ladění
        return (f"Note({self.name} @{self.start_beat:g} "
                f"dur={self.dur_beats:g} vel={self.velocity})")


def _tokens(segment: str) -> list[str]:
    """Rozdělí takt na tokeny, závorkový akord drží pohromadě."""
    out, buf, depth = [], "", 0
    for ch in segment:
        if ch == "(":
            depth += 1
            buf += ch
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise NoteParseError("přebytečná ')'")
            buf += ch
        elif ch.isspace() and depth == 0:
            if buf:
                out.append(buf)
                buf = ""
        else:
            buf += ch
    if depth != 0:
        raise NoteParseError("nezavřená '('")
    if buf:
        out.append(buf)
    return out


def parse(text: str, beats_per_bar: float = 4.0, default_dur: float = 1.0,
          velocity: int = DEFAULT_VELOCITY, check_bars: bool | None = None,
          bar_offset: int = 0) -> list[Note]:
    """Naparsuje notační řetězec na seznam not se startem od nuly.

    `check_bars` zapíná kontrolu délek taktů; ve výchozím stavu se zapne,
    jen když je v řetězci alespoň jedno ``|`` (jinak není co kontrolovat).
    `bar_offset` se přičte k číslu taktu v chybové hlášce.
    """
    if not isinstance(text, str):
        raise NoteParseError(f"očekávám string, dostal jsem {type(text).__name__}")
    has_bars = "|" in text
    if check_bars is None:
        check_bars = has_bars

    segments = [s for s in text.split("|")]
    if has_bars:
        # Vedoucí a koncové svislítko je jen ozdoba: "|C4:4|" == "C4:4"
        if segments and not segments[0].strip():
            segments = segments[1:]
        if segments and not segments[-1].strip():
            segments = segments[:-1]

    notes: list[Note] = []
    cursor = 0.0
    last: list[Note] = []

    for bar_i, segment in enumerate(segments):
        bar_start = cursor
        for tok in _tokens(segment):
            m = _REST_RE.match(tok)
            if m:
                cursor += float(m.group("dur") or default_dur)
                last = []
                continue
            m = _TIE_RE.match(tok)
            if m:
                extra = float(m.group("dur") or default_dur)
                if not last:
                    raise NoteParseError(f"'~' bez předchozí noty: {tok!r}")
                for n in last:
                    n.dur_beats += extra
                cursor += extra
                continue
            m = _CHORD_RE.match(tok)
            if m:
                dur = float(m.group("dur") or default_dur)
                vel = int(m.group("vel") or velocity)
                members = m.group("body").split()
                if not members:
                    raise NoteParseError(f"prázdný akord: {tok!r}")
                last = []
                for name in members:
                    n = Note(_pitch(name), cursor, dur, vel)
                    notes.append(n)
                    last.append(n)
                cursor += dur
                continue
            m = _NOTE_RE.match(tok)
            if m:
                dur = float(m.group("dur") or default_dur)
                vel = int(m.group("vel") or velocity)
                n = Note(_pitch(m.group("name")), cursor, dur, vel)
                notes.append(n)
                last = [n]
                cursor += dur
                continue
            raise NoteParseError(f"nerozumím tokenu {tok!r}")

        if check_bars:
            length = cursor - bar_start
            if abs(length - beats_per_bar) > 1e-6:
                raise NoteLengthError(
                    f"takt {bar_i + bar_offset + 1}: délky dávají {length:g} "
                    f"beatů, čekám {beats_per_bar:g} "
                    f"(segment {segment.strip()!r})")
            cursor = bar_start + beats_per_bar

    return notes
