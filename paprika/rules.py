"""Pravidla hudební teorie nad datovým modelem — bez audia a bez renderu.

Čte noty stop, harmonický plán (`Song.harmony`) a sekce (`Song.section`)
a vrací `Warning_` stejně jako audio pravidla v `analyze.py`. Přísnost řídí
`Song.theory`:

- ``free`` (výchozí) — cinematic/pop: kontrolují se samostatné linky
  (jednohlasé stopy), oktávová zdvojení a pady jsou záměr; WARN jen pro
  jasné chyby (bas mimo akord, chromatický cizí tón na těžké době,
  zablácené tercie v hloubce), zbytek je INFO,
- ``strict`` — učebnice (chorál, kvartet): vícehlasé stopy se rozdělí na
  hlasy, hlásí se všechny paralely, křížení a rozestupy, vše jako WARN,
- ``off`` — teorie se nekontroluje.

Pravidla, která potřebují plán, se bez `harmony()` přeskočí. Každé pravidlo
hlásí nejvýš `CAP` nálezů, ať report zůstane čitelný.
"""

from __future__ import annotations

from bisect import bisect_right
from statistics import median

from . import harmony as _harmony
from . import theory
from .song import DrumTrack, Song

__all__ = ["Warning_", "theory_warnings", "harmony_lines", "EPS"]

EPS = 0.05          # beaty: tolerance pro humanize (posouvá starty o ±0.015)
OVERLAP = 0.1       # beaty: překryv not, od kterého je stopa vícehlasá
CAP = 3             # nálezů na pravidlo
LOW_THIRD = 43      # G2: tercie pod tímhle tónem zní zablácené
HELD_BASS = 0.5     # beaty: bas, který začal dřív, je prodleva
LOW_THIRD_BEATS = 1.0  # beaty: jak dlouho musí tercie v hloubce znít spolu
FIGURE_BEATS = 0.5  # beaty: medián délky not figurace (arpeggio, ostinato)


class Warning_:
    """Jedno hlášení. `level` je WARN nebo INFO, `bar` takt (když ho pravidlo
    zná) — podle něj `check --bars` filtruje."""

    __slots__ = ("level", "where", "text", "bar")

    def __init__(self, level: str, where: str, text: str, bar: int | None = None):
        self.level = level
        self.where = where
        self.text = text
        self.bar = bar

    def line(self) -> str:
        return f"{self.level:<4}  {self.where:<14} {self.text}"

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.level} {self.where}: {self.text}>"


# --------------------------------------------------------------------------
# pomocné
# --------------------------------------------------------------------------

def _bar(song: Song, beat: float) -> int:
    return int((beat + EPS) // song.beats_per_bar)


def _strong(song: Song, beat: float) -> bool:
    """Těžká doba: 1. doba taktu, ve 4/4 i 3., v 6/x i 4."""
    bpb = song.beats_per_bar
    pos = beat % bpb
    strong = [0.0]
    if abs(bpb - 4) < 1e-9:
        strong.append(2.0)
    elif abs(bpb - 6) < 1e-9:
        strong.append(3.0)
    return any(abs(pos - s) < EPS or abs(pos - s - bpb) < EPS for s in strong)


def _pitched(song: Song) -> list:
    return [t for t in song.tracks if not isinstance(t, DrumTrack)]


def _name(song: Song, pitch: int) -> str:
    return _harmony.pc_name(pitch % 12, song.key, song.scale) + str(pitch // 12 - 1)


def _chord_text(song: Song, span) -> str:
    roman = _harmony.roman_of(span.chord, song.key, song.scale)
    return f"{span.chord.label} ({roman})"


class _Line:
    """Jedna melodická linka: noty bez překryvu, seřazené podle času.

    `figuration` = rozložené akordy a ostinata (krátké tóny, kroky aspoň malá
    tercie): shodné tóny s melodií jsou náhoda, ne vedení hlasů."""

    def __init__(self, name: str, notes: list):
        self.name = name
        self.notes = notes
        self.starts = [n.start_beat for n in notes]
        self.median = median(n.pitch for n in notes)
        steps = [abs(b.pitch - a.pitch) for a, b in zip(notes, notes[1:])
                 if b.pitch != a.pitch]
        self.figuration = (bool(steps) and median(steps) >= 3
                           and median(n.dur_beats for n in notes) <= FIGURE_BEATS)

    def at(self, beat: float):
        """Nota, která zní v daném beatu (None v pauze)."""
        i = bisect_right(self.starts, beat) - 1
        if i >= 0 and self.notes[i].end_beat > beat:
            return self.notes[i]
        return None


def _monophonic(track, notes) -> bool:
    if track.mono:
        return True
    return all(b.start_beat >= a.end_beat - OVERLAP for a, b in zip(notes, notes[1:]))


def _split_voices(name: str, notes: list) -> list[_Line]:
    """Vícehlasá stopa -> hlasy podle pořadí výšky v souzvucích nejčastější
    velikosti (strict). Souzvuky jiné velikosti (arpeggio, melodie) vypadnou."""
    groups: list[list] = []
    for n in notes:
        if groups and abs(n.start_beat - groups[-1][0].start_beat) < EPS:
            groups[-1].append(n)
        else:
            groups.append([n])
    sizes = [len(g) for g in groups if len(g) >= 2]
    if len(sizes) < 2:
        return []
    k = max(set(sizes), key=sizes.count)
    voices: list[list] = [[] for _ in range(k)]
    for g in groups:
        if len(g) == k:
            for i, n in enumerate(sorted(g, key=lambda n: n.pitch)):
                voices[i].append(n)
    return [_Line(f"{name}:{i + 1}", v) for i, v in enumerate(voices) if len(v) >= 2]


def _lines(song: Song, strict: bool) -> tuple[list[_Line], dict]:
    """Linky skladby a {stopa: noty} všech melodických stop."""
    lines, by_track = [], {}
    for t in _pitched(song):
        notes = t.all_notes()
        if not notes:
            continue
        by_track[t.name] = notes
        if _monophonic(t, notes):
            lines.append(_Line(t.name, notes))
        elif strict:
            lines.extend(_split_voices(t.name, notes))
    return lines, by_track


# --------------------------------------------------------------------------
# pravidla bez plánu
# --------------------------------------------------------------------------

def _motions(a: _Line, b: _Line):
    """Po sobě jdoucí souzvuky dvou linek: [(beat, a1, b1, a2, b2)]."""
    times = sorted(set(a.starts) | set(b.starts))
    merged: list[float] = []
    for t in times:
        if not merged or t - merged[-1] >= EPS:
            merged.append(t)
    out, prev = [], None
    for t in merged:
        na, nb = a.at(t + EPS), b.at(t + EPS)
        if na is None or nb is None:
            out.append(None)              # přerušení: pauza v jedné z linek
            prev = None
            continue
        if prev is not None and (na is not prev[0] or nb is not prev[1]):
            out.append((t, prev[0], prev[1], na, nb))
        prev = (na, nb)
    return out


def _parallel_kind(a1, b1, a2, b2) -> str | None:
    da, db = a2.pitch - a1.pitch, b2.pitch - b1.pitch
    if da == 0 or db == 0 or (da > 0) != (db > 0):
        return None
    i1, i2 = abs(a1.pitch - b1.pitch) % 12, abs(a2.pitch - b2.pitch) % 12
    if i1 == 7 and i2 == 7:
        return "kvinty"
    if i1 == 0 and i2 == 0:
        return "oktávy"
    return None


def _parallels(song: Song, lines: list, strict: bool, warns: list) -> None:
    """Paralelní čisté kvinty a oktávy mezi linkami. Ve free režimu je běh
    tří a víc paralel po sobě záměrné zdvojení a nehlásí se, stejně jako
    paralely s figurací (arpeggio, ostinato)."""
    found = 0
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            if not strict and (a.figuration or b.figuration):
                continue
            run: list = []
            for m in _motions(a, b) + [None]:
                kind = None if m is None else _parallel_kind(*m[1:])
                if kind and run and run[0][0] == kind:
                    run.append((kind, m))
                    continue
                if run and (strict or len(run) <= 2):
                    _, (t, a1, b1, a2, b2) = run[0]
                    level = "WARN" if strict else "INFO"
                    warns.append(_W(level, a.name,
                        f"bar {_bar(song, t)}   paralelní {run[0][0]} s {b.name}: "
                        f"{_name(song, a1.pitch)}/{_name(song, b1.pitch)} → "
                        f"{_name(song, a2.pitch)}/{_name(song, b2.pitch)}",
                        _bar(song, t)))
                    found += 1
                    if found >= CAP:
                        return
                run = [(kind, m)] if kind else []


def _interval_text(semitones: int) -> str:
    names = {6: "tritón", 8: "malou sextu", 9: "velkou sextu",
             10: "malou septimu", 11: "velkou septimu"}
    return names.get(semitones, f"{semitones} půltónů")


def _leaps(song: Song, lines: list, strict: bool, warns: list) -> None:
    """Melodický skok ≥ malá sexta (kromě oktávy) bez pohybu opačným směrem
    a tritónový skok. Jen linky, které jsou melodií: medián výšky ≥ C4
    a medián kroku < 4 půltóny (arpeggia a basy se nehodnotí)."""
    found = 0
    level = "WARN" if strict else "INFO"
    for line in lines:
        ns = line.notes
        if len(ns) < 3 or line.median < 60:
            continue
        steps = [abs(b.pitch - a.pitch) for a, b in zip(ns, ns[1:]) if b.pitch != a.pitch]
        if not steps or median(steps) >= 4:
            continue
        for i in range(len(ns) - 1):
            a, b = ns[i], ns[i + 1]
            if b.start_beat - a.end_beat > 1.0:
                continue                      # přes pauzu: nová fráze
            d = b.pitch - a.pitch
            size = abs(d)
            if size == 6:
                text = "tritónový skok"
            elif size >= 8 and size != 12:
                c = ns[i + 2] if i + 2 < len(ns) else None
                if c is not None and c.start_beat - b.end_beat <= 1.0 \
                        and (c.pitch - b.pitch) * d < 0:
                    continue                  # skok se vrací opačným směrem
                text = f"skok o {_interval_text(size)} bez návratu opačným směrem"
            else:
                continue
            bar = _bar(song, b.start_beat)
            warns.append(_W(level, line.name,
                            f"bar {bar}   {text} {_name(song, a.pitch)} → "
                            f"{_name(song, b.pitch)}", bar))
            found += 1
            if found >= CAP:
                return


def _low_thirds(song: Song, by_track: dict, warns: list) -> None:
    """Dva tóny v terciové vzdálenosti, spodní pod G2, zní spolu aspoň
    `LOW_THIRD_BEATS` — v hloubce to splyne v bahno. Krátké tóny arpeggia
    (F1 C2 F2 A2 v levé ruce) se nehlásí."""
    low = sorted(((n, name) for name, notes in by_track.items()
                  for n in notes if n.pitch < LOW_THIRD + 4),
                 key=lambda x: x[0].start_beat)
    found, seen = 0, set()
    for i, (a, ta) in enumerate(low):
        for b, tb in low[i + 1:]:
            if b.start_beat >= a.end_beat:
                break
            lo, hi = sorted((a.pitch, b.pitch))
            together = min(a.end_beat, b.end_beat) - max(a.start_beat, b.start_beat)
            if hi - lo not in (3, 4) or lo >= LOW_THIRD or together < LOW_THIRD_BEATS:
                continue
            bar = _bar(song, max(a.start_beat, b.start_beat))
            if (bar, lo, hi) in seen:
                continue
            seen.add((bar, lo, hi))
            where = ta if a.pitch == lo else tb
            warns.append(_W("WARN", where,
                f"bar {bar}   tercie {_name(song, lo)}+{_name(song, hi)} pod G2 "
                f"zní zablácené — horní tón o oktávu výš nebo jen základ", bar))
            found += 1
            if found >= CAP:
                return


def _spacing(song: Song, by_track: dict, warns: list) -> None:
    """strict: sousední horní hlasy souzvuku dál než oktávu od sebe."""
    found = 0
    for name, notes in by_track.items():
        groups: list[list] = []
        for n in notes:
            if groups and abs(n.start_beat - groups[-1][0].start_beat) < EPS:
                groups[-1].append(n)
            else:
                groups.append([n])
        for g in groups:
            if len(g) < 3:
                continue
            ps = sorted(n.pitch for n in g)
            for lo, hi in zip(ps[1:], ps[2:]):        # bas (ps[0]) smí být dál
                if hi - lo > 12:
                    bar = _bar(song, g[0].start_beat)
                    warns.append(_W("WARN", name,
                        f"bar {bar}   horní hlasy {_name(song, lo)} a "
                        f"{_name(song, hi)} dál než oktávu od sebe", bar))
                    found += 1
                    break
            if found >= CAP:
                return


def _crossing(song: Song, lines: list, warns: list) -> None:
    """strict: sousední linky (podle mediánu výšky) se překříží."""
    ordered = sorted(lines, key=lambda line: line.median)
    found = 0
    for low, high in zip(ordered, ordered[1:]):
        for m in _motions(low, high):
            if m is None:
                continue
            t, _, _, a2, b2 = m
            if a2.pitch > b2.pitch:
                bar = _bar(song, t)
                warns.append(_W("WARN", low.name,
                    f"bar {bar}   křížení hlasů: {_name(song, a2.pitch)} nad "
                    f"{high.name} ({_name(song, b2.pitch)})", bar))
                found += 1
                break
        if found >= CAP:
            return


# --------------------------------------------------------------------------
# pravidla s harmonickým plánem
# --------------------------------------------------------------------------

def _bass_vs_plan(song: Song, by_track: dict, warns: list) -> None:
    """Nejnižší znějící tón na začátku akordu musí patřit do akordu.
    Bas, který zní už dřív (prodleva), se nehlásí."""
    allnotes = [(n, name) for name, notes in by_track.items() for n in notes]
    found = 0
    for span in song.plan:
        t = span.start_beat + EPS
        sounding = [(n, name) for n, name in allnotes
                    if n.start_beat <= t < n.end_beat]
        if not sounding:
            continue
        n, name = min(sounding, key=lambda x: x[0].pitch)
        if n.start_beat < span.start_beat - HELD_BASS:
            continue
        if n.pitch % 12 in span.chord.pcs:
            continue
        bar = _bar(song, span.start_beat)
        root = _harmony.pc_name(span.chord.bass, song.key, song.scale)
        warns.append(_W("WARN", name,
            f"bar {bar}   bas {_name(song, n.pitch)} nepatří do akordu "
            f"{_chord_text(song, span)} — plán chce v basu {root} "
            f"(nebo jiný tón akordu)", bar))
        found += 1
        if found >= CAP:
            return


def _coverage(song: Song, by_track: dict, strict: bool, warns: list) -> None:
    """V úseku akordu nezní jeho základ nebo tercie — plán není slyšet."""
    allnotes = [n for notes in by_track.values() for n in notes]
    found = 0
    for span in song.plan:
        pcs = {n.pitch % 12 for n in allnotes
               if n.start_beat < span.end_beat - EPS and n.end_beat > span.start_beat + EPS}
        if not pcs:
            continue
        missing = []
        if span.chord.root not in pcs:
            missing.append(f"základ {_harmony.pc_name(span.chord.root, song.key, song.scale)}")
        third = span.chord.third_pc
        if third is not None and third not in pcs:
            missing.append(f"tercie {_harmony.pc_name(third, song.key, song.scale)}")
        if not missing:
            continue
        bar = _bar(song, span.start_beat)
        warns.append(_W("WARN" if strict else "INFO", "harmonie",
                        f"bar {bar}   {_chord_text(song, span)} není slyšet — "
                        f"chybí {' a '.join(missing)}", bar))
        found += 1
        if found >= CAP:
            return


def _nonchord(song: Song, lines: list, strict: bool, warns: list) -> None:
    """Nota linky na těžké době mimo akord, která se nerozvede krokem.
    Průtah (stejná výška před ní) a appoggiatura (pokračuje krokem) projdou."""
    scale = set(song.scale_pcs())
    found = 0
    for line in lines:
        ns = line.notes
        for i, n in enumerate(ns):
            if not _strong(song, n.start_beat):
                continue
            span = song.chord_at(n.start_beat + EPS)
            if span is None or n.pitch % 12 in span.chord.pcs:
                continue
            prev = ns[i - 1] if i > 0 and ns[i - 1].end_beat >= n.start_beat - OVERLAP else None
            nxt = ns[i + 1] if i + 1 < len(ns) and ns[i + 1].start_beat <= n.end_beat + EPS else None
            if prev is not None and prev.pitch == n.pitch:
                continue
            if nxt is not None and 0 < abs(nxt.pitch - n.pitch) <= 2:
                continue
            chromatic = n.pitch % 12 not in scale
            level = "WARN" if strict or chromatic else "INFO"
            bar = _bar(song, n.start_beat)
            extra = ", navíc mimo stupnici" if chromatic else ""
            warns.append(_W(level, line.name,
                f"bar {bar}   {_name(song, n.pitch)} na těžké době mimo akord "
                f"{_chord_text(song, span)} a nerozvede se krokem{extra}", bar))
            found += 1
            if found >= CAP:
                return


def _leading_tone(song: Song, lines: list, strict: bool, warns: list) -> None:
    """Dominanta → tónika: nejvyšší linka s citlivým tónem jde na tóniku."""
    tonic = theory._root_pc(song.key)
    lt = (tonic + 11) % 12
    found = 0
    for prev, cur in zip(song.plan, song.plan[1:]):
        if abs(cur.start_beat - prev.end_beat) > EPS:
            continue
        if (prev.chord.root - tonic) % 12 != 7 or lt not in prev.chord.pcs:
            continue
        if (cur.chord.root - tonic) % 12 != 0:
            continue
        t = cur.start_beat
        moves = []
        for line in lines:
            a, b = line.at(t - EPS), line.at(t + EPS)
            if a is not None and b is not None and a is not b:
                moves.append((a, b, line))
        if not moves:
            continue
        a, b, line = max(moves, key=lambda m: m[0].pitch)
        if a.pitch % 12 != lt or b.pitch == a.pitch + 1:
            continue
        bar = _bar(song, t)
        warns.append(_W("WARN" if strict else "INFO", line.name,
            f"bar {bar}   citlivý tón {_name(song, a.pitch)} v "
            f"{prev.chord.label} → {cur.chord.label} nejde na tóniku "
            f"(nejvyšší hlas jde na {_name(song, b.pitch)})", bar))
        found += 1
        if found >= CAP:
            return


def _W(level: str, where: str, text: str, bar: int | None = None) -> Warning_:
    return Warning_(level, where, text, bar)


def theory_warnings(song: Song) -> list[Warning_]:
    """Všechna pravidla teorie podle `song.theory` (WARN i INFO, neseřazené)."""
    if song.theory == "off":
        return []
    strict = song.theory == "strict"
    lines, by_track = _lines(song, strict)
    warns: list[Warning_] = []
    _low_thirds(song, by_track, warns)
    _parallels(song, lines, strict, warns)
    _leaps(song, lines, strict, warns)
    if song.plan:
        _bass_vs_plan(song, by_track, warns)
        _coverage(song, by_track, strict, warns)
        _nonchord(song, lines, strict, warns)
        _leading_tone(song, lines, strict, warns)
    if strict:
        _spacing(song, by_track, warns)
        _crossing(song, lines, warns)
    return warns


# --------------------------------------------------------------------------
# souhrn harmonie pro `check`
# --------------------------------------------------------------------------

def _groups(song: Song) -> list[tuple[str, int, int]]:
    """(jméno, od, do) — sekce, nebo bez sekcí úseky po osmi taktech."""
    if song.sections:
        return [(s.name, s.start_bar, s.end_bar) for s in song.sections]
    end = max(song.bars, max((_bar(song, s.end_beat - EPS) + 1 for s in song.plan),
                             default=0))
    return [(f"{lo}–{min(lo + 8, end) - 1}", lo, min(lo + 8, end))
            for lo in range(0, end, 8)]


def harmony_lines(song: Song, bars=None) -> list[str]:
    """Řádky HARMONIE (římské číslice po taktech), AKORDY (legenda) a KADENCE."""
    if not song.plan:
        return ["HARMONIE (bez plánu — s.harmony(\"Dm | Bb | Gm | A\") zapne "
                "kontrolu basu, nevlastních tónů a kadencí)"]
    lo_b, hi_b = (bars if bars else (0, 10 ** 9))
    out, cads, legend = [], [], {}
    bpb = song.beats_per_bar
    for name, lo, hi in _groups(song):
        if hi <= lo_b or lo >= hi_b:
            continue
        cells = []
        for b in range(lo, hi):
            here = [s for s in song.plan
                    if b * bpb - EPS <= s.start_beat < (b + 1) * bpb - EPS]
            romans = []
            for s in here:
                r = _harmony.roman_of(s.chord, song.key, song.scale)
                legend.setdefault(r, s.chord.label)
                romans.append(r)
            cells.append(" ".join(romans) if romans else "%")
        label = name if song.sections else ""
        span = f"{lo}–{hi - 1}"
        head = f"{label} ({span})" if label else span
        out.append(f"HARMONIE {head:<12} " + " | ".join(cells))
        inside = [s for s in song.plan if lo * bpb - EPS <= s.start_beat < hi * bpb - EPS]
        if inside:
            last = inside[-1]
            prev = inside[-2] if len(inside) > 1 else None
            kind = _harmony.cadence(prev and prev.chord, last.chord, song.key, song.scale)
            r_last = _harmony.roman_of(last.chord, song.key, song.scale)
            r_prev = _harmony.roman_of(prev.chord, song.key, song.scale) if prev else "—"
            cads.append(f"{head}: {r_prev} → {r_last} {kind or 'bez kadence'}")
    if legend:
        items = [f"{r}={c}" for r, c in list(legend.items())[:16]]
        out.append("AKORDY   " + "  ".join(items))
    if cads:
        out.append("KADENCE  " + " · ".join(cads))
    return out
