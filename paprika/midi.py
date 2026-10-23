"""MIDI export a čtení (Standard MIDI File, formát 1) — ručně přes `struct`.

Export: stopa 0 drží tempo, takt, předznamenání, sekce (markery) a akordy
harmonického plánu (textové události); každá `Track` je vlastní MIDI stopa
s programem podle GM mapy. Bicí jdou na kanál 10 (index 9), sustain pedál
jako CC64. SFX stopy (riser, impact — zvuky, ne noty) a automatizace se do
MIDI nevejdou; `export` vrátí poznámky, co vynechal.

Čtení zná running status a skládá note-on/off do not — pro testy a pro
pozdější porovnání partitury se zvukem. Modul čte jen datový model, o
renderu nic neví.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

from . import harmony as _harmony
from .song import DrumTrack, HumanTrack, SfxTrack, Song

__all__ = ["PPQ", "GM_PROGRAMS", "GM_DRUMS", "MidiError", "MidiNote", "MidiTrack",
           "MidiData", "to_bytes", "export", "read"]

PPQ = 480                 # ticků na beat (čtvrťovou notu)
DRUM_CHANNEL = 9          # GM bicí = kanál 10
DRUM_TICKS = PPQ // 8     # délka úderu bicích
_MELODIC = [c for c in range(16) if c != DRUM_CHANNEL]

# GM program (0-127) podle jména nástroje/presetu; co tu není, nedostane
# program change a DAW mu nechá výchozí zvuk
GM_PROGRAMS = {
    # samplované nástroje (library.py)
    "grand_piano": 0, "upright_piano": 0, "violins": 48, "violas": 48,
    "celli": 48, "violin_solo": 40, "contrabass": 43, "concert_harp": 46,
    "flute": 73, "oboe": 68, "clarinet": 71, "bassoon": 70, "horn": 60,
    "trumpet": 56, "trombone": 57, "tuba": 58, "male_choir": 52,
    "female_choir": 52, "basso": 53, "tenor": 53, "soprano": 53, "alto": 53,
    "marimba": 12, "glockenspiel": 9, "xylophone": 13, "timpani": 47,
    # presety (presets.py)
    "epiano": 4, "felt_piano": 0, "lofi_keys": 4, "toy_piano": 8, "celesta": 8,
    "music_box": 10, "bell": 14, "organ": 16, "strings": 48, "strings_ens": 48,
    "strings_short": 48, "string_stab": 48, "pizz": 45, "cello_solo": 42,
    "harp": 46, "brass_section": 61, "horn_swell": 60, "choir_aah": 52,
    "choir_ooh": 53, "choir_mm": 53, "vox_chop": 54, "808": 38, "subbass": 38,
    "round_bass": 38, "reese": 39, "acid": 38, "lead": 81, "detuned_lead": 81,
    "supersaw": 81, "warmpad": 89, "air_pad": 89, "pwm_pad": 90,
    "glass_pad": 92, "shimmer_pad": 88, "timpani_tone": 47,
}
_PIZZ_PROGRAM = 45        # "violins.pizz" a spol.

# bicí paprika -> GM klávesa na kanálu 10
GM_DRUMS = {"kick": 36, "snare": 38, "hat": 42, "openhat": 46, "clap": 39,
            "tom": 45, "ride": 51, "rim": 37, "boom": 35, "shaker": 82,
            "perc": 75}


class MidiError(ValueError):
    """Soubor není Standard MIDI File nebo je poškozený."""


@dataclass
class MidiNote:
    start: int            # ticky
    dur: int
    pitch: int
    velocity: int
    channel: int


@dataclass
class MidiTrack:
    name: str
    notes: list = field(default_factory=list)
    # (tick, druh, data): "meta" (typ, bajty), "cc" (kanál, ovladač, hodnota),
    # "program" (kanál, program), "note_on"/"note_off" (kanál, výška, velocity)
    events: list = field(default_factory=list)


@dataclass
class MidiData:
    ppq: int
    fmt: int
    tracks: list


# --------------------------------------------------------------------------
# zápis
# --------------------------------------------------------------------------

def _vlq(n: int) -> bytes:
    """Variable-length quantity (delta čas, délka meta události)."""
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _meta(kind: int, payload: bytes) -> bytes:
    return bytes([0xFF, kind]) + _vlq(len(payload)) + payload


def _text(kind: int, text: str) -> bytes:
    return _meta(kind, str(text).encode("utf-8"))


def _tick(beat: float) -> int:
    return max(0, int(round(beat * PPQ)))


def _chunk(events: list) -> bytes:
    """[(tick, pořadí, bajty)] -> MTrk chunk s delta časy a End of Track."""
    data = bytearray()
    last = 0
    for tick, _, ev in sorted(events, key=lambda e: (e[0], e[1])):
        data += _vlq(tick - last) + ev
        last = tick
    data += _vlq(0) + _meta(0x2F, b"")
    return b"MTrk" + struct.pack(">I", len(data)) + bytes(data)


# pořadí událostí ve stejném ticku: meta, konec not, pedál nahoru, pedál dolů,
# začátek not — note-off musí předběhnout note-on stejné výšky
_O_META, _O_OFF, _O_UP, _O_DOWN, _O_ON = range(5)


def _time_signature(song: Song, notes: list) -> bytes:
    bpb = song.beats_per_bar
    if abs(bpb - round(bpb)) < 1e-9:
        nn, dd = int(round(bpb)), 2
    elif abs(bpb * 2 - round(bpb * 2)) < 1e-9:
        nn, dd = int(round(bpb * 2)), 3
    else:
        notes.append(f"takt {bpb:g} beatů nejde zapsat jako MIDI takt — zapsán 4/4")
        nn, dd = 4, 2
    return _meta(0x58, bytes([nn, dd, 24, 8]))


def _conductor(song: Song, notes: list) -> bytes:
    ev = [(0, _O_META, _text(0x03, song.title or "paprika"))]
    tempo = int(round(60_000_000 / song.bpm))
    ev.append((0, _O_META, _meta(0x51, tempo.to_bytes(3, "big"))))
    ev.append((0, _O_META, _time_signature(song, notes)))
    try:
        sf, minor = _harmony.key_signature(song.key, song.scale)
        ev.append((0, _O_META, _meta(0x59, struct.pack(">bB", sf, minor))))
    except _harmony.HarmonyError:
        notes.append(f"tónina {song.key} {song.scale} nemá MIDI předznamenání")
    for sec in song.sections:
        ev.append((_tick(sec.start_bar * song.beats_per_bar), _O_META,
                   _text(0x06, sec.name)))
    for span in song.plan:
        roman = _harmony.roman_of(span.chord, song.key, song.scale)
        ev.append((_tick(span.start_beat), _O_META,
                   _text(0x01, f"{span.chord.label} ({roman})")))
    return _chunk(ev)


def _program(track) -> int | None:
    name = str(getattr(track.patch, "name", "") or "")
    base, _, art = name.partition(".")
    if art == "pizz":
        return _PIZZ_PROGRAM
    return GM_PROGRAMS.get(base)


def _note_events(track, channel: int, notes_out: list) -> list:
    """Noty stopy jako události; překryv stejné výšky se zkrátí."""
    raw = []
    for n in track.all_notes():
        if not 0 <= n.pitch <= 127:
            notes_out.append(f"{track.name}: výška {n.pitch} je mimo MIDI rozsah, "
                             f"nota vynechána")
            continue
        raw.append([_tick(n.start_beat), _tick(n.end_beat), n.pitch,
                    max(1, min(127, int(n.velocity)))])
    raw.sort(key=lambda r: (r[0], r[2]))
    last: dict[int, list] = {}
    for r in raw:
        prev = last.get(r[2])
        if prev is not None and prev[1] > r[0]:
            prev[1] = r[0]            # note-off první by jinak utnul druhou
        last[r[2]] = r
    ev = []
    for start, end, pitch, vel in raw:
        if end <= start:
            continue                  # zdvojená nota ve stejném ticku
        ev.append((start, _O_ON, bytes([0x90 | channel, pitch, vel])))
        ev.append((end, _O_OFF, bytes([0x80 | channel, pitch, 0])))
    for down, up in getattr(track, "pedals", []):
        ev.append((_tick(down), _O_DOWN, bytes([0xB0 | channel, 64, 127])))
        ev.append((_tick(up), _O_UP, bytes([0xB0 | channel, 64, 0])))
    return ev


def _drum_events(track) -> list:
    ev = []
    for h in track.hits:
        key = GM_DRUMS.get(h.name, GM_DRUMS["perc"])
        t = _tick(h.start_beat)
        vel = max(1, min(127, int(h.velocity)))
        ev.append((t, _O_ON, bytes([0x90 | DRUM_CHANNEL, key, vel])))
        ev.append((t + DRUM_TICKS, _O_OFF, bytes([0x80 | DRUM_CHANNEL, key, 0])))
    return ev


def to_bytes(song: Song) -> tuple[bytes, list[str]]:
    """Celá skladba jako SMF formát 1 + poznámky, co se nevešlo."""
    notes: list[str] = []
    chunks = [_conductor(song, notes)]
    melodic = 0
    for t in song.tracks:
        if isinstance(t, HumanTrack):
            notes.append(f"{t.name}: lidské zvuky (dech, kroky, ...) jsou zvuk, "
                         f"ne noty — do MIDI se nevejdou")
            continue
        if isinstance(t, SfxTrack):
            notes.append(f"{t.name}: SFX stopa (riser, impact, ...) je zvuk, ne "
                         f"noty — do MIDI se nevejde")
            continue
        if getattr(t, "automations", None):
            notes.append(f"{t.name}: automatizace "
                         f"({', '.join(a.target for a in t.automations)}) se "
                         f"neexportuje")
        ev = [(0, _O_META, _text(0x03, t.name))]
        if isinstance(t, DrumTrack):
            ev.extend(_drum_events(t))
        else:
            channel = _MELODIC[melodic % len(_MELODIC)]
            if melodic == len(_MELODIC):
                notes.append("víc než 15 melodických stop — MIDI kanály se "
                             "opakují (stopy zůstávají oddělené)")
            melodic += 1
            program = _program(t)
            if program is not None:
                ev.append((0, _O_META, bytes([0xC0 | channel, program])))
            ev.extend(_note_events(t, channel, notes))
        chunks.append(_chunk(ev))
    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), PPQ)
    return header + b"".join(chunks), notes


def export(song: Song, path: str) -> list[str]:
    """Zapíše skladbu do MIDI souboru; vrátí poznámky, co se nevešlo."""
    data, notes = to_bytes(song)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return notes


# --------------------------------------------------------------------------
# čtení
# --------------------------------------------------------------------------

def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    while True:
        if pos >= len(data):
            raise MidiError("useknutá délka (variable-length quantity)")
        b = data[pos]
        pos += 1
        value = (value << 7) | (b & 0x7F)
        if not b & 0x80:
            return value, pos


_DATA_LEN = {0x80: 2, 0x90: 2, 0xA0: 2, 0xB0: 2, 0xC0: 1, 0xD0: 1, 0xE0: 2}


def _read_track(body: bytes) -> MidiTrack:
    track = MidiTrack(name="")
    pos, tick, status = 0, 0, None
    open_notes: dict[tuple, list] = {}
    while pos < len(body):
        delta, pos = _read_vlq(body, pos)
        tick += delta
        if pos >= len(body):
            raise MidiError("useknutá událost")
        b = body[pos]
        if b == 0xFF:
            kind = body[pos + 1]
            length, pos = _read_vlq(body, pos + 2)
            payload = body[pos:pos + length]
            pos += length
            track.events.append((tick, "meta", (kind, payload)))
            if kind == 0x03 and not track.name:
                track.name = payload.decode("utf-8", errors="replace")
            if kind == 0x2F:
                break
            continue
        if b in (0xF0, 0xF7):
            length, pos = _read_vlq(body, pos + 1)
            pos += length
            track.events.append((tick, "sysex", None))
            continue
        if b & 0x80:
            status = b
            pos += 1
        elif status is None:
            raise MidiError("data bez stavového bajtu (running status bez začátku)")
        kind, ch = status & 0xF0, status & 0x0F
        args = body[pos:pos + _DATA_LEN[kind]]
        pos += _DATA_LEN[kind]
        if kind == 0x90 and args[1] > 0:
            track.events.append((tick, "note_on", (ch, args[0], args[1])))
            open_notes.setdefault((ch, args[0]), []).append([tick, args[1]])
        elif kind in (0x80, 0x90):
            track.events.append((tick, "note_off", (ch, args[0], args[1])))
            stack = open_notes.get((ch, args[0]))
            if stack:
                start, vel = stack.pop(0)
                track.notes.append(MidiNote(start, tick - start, args[0], vel, ch))
        elif kind == 0xB0:
            track.events.append((tick, "cc", (ch, args[0], args[1])))
        elif kind == 0xC0:
            track.events.append((tick, "program", (ch, args[0])))
        else:
            track.events.append((tick, "other", (status, bytes(args))))
    track.notes.sort(key=lambda n: (n.start, n.pitch))
    return track


def read(src) -> MidiData:
    """Načte MIDI ze souboru (cesta) nebo z bajtů."""
    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
    else:
        with open(src, "rb") as f:
            data = f.read()
    if data[:4] != b"MThd" or len(data) < 14:
        raise MidiError("není to Standard MIDI File (chybí hlavička MThd)")
    hlen, fmt, ntrks, division = struct.unpack(">IHHH", data[4:14])
    pos = 8 + hlen
    tracks = []
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        (clen,) = struct.unpack(">I", data[pos + 4:pos + 8])
        body = data[pos + 8:pos + 8 + clen]
        pos += 8 + clen
        if cid == b"MTrk":
            tracks.append(_read_track(body))
    if len(tracks) != ntrks:
        raise MidiError(f"hlavička slibuje {ntrks} stop, v souboru je {len(tracks)}")
    return MidiData(division, fmt, tracks)
