"""SFZ: text nástroje -> seznam regionů.

Modul zná jen text a čísla, žádné audio. Umí podmnožinu SFZ, která stačí na
běžné knihovny (Salamander, VSCO, vlastní WAV): hlavičky `<control>
<global> <master> <group> <region>`, `#define`, `#include`, komentáře
a cesty ke vzorkům s mezerami.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

__all__ = ["Region", "SfzError", "parse", "load", "note_number"]


class SfzError(ValueError):
    """SFZ se nedá přečíst — zpráva říká soubor, opcode a hodnotu."""


@dataclass
class Region:
    """Jeden region: vzorek a kdy/jak se hraje. Časy v sekundách."""

    sample: str
    lokey: int = 0
    hikey: int = 127
    keycenter: int = 60
    lovel: int = 1
    hivel: int = 127
    seq_length: int = 1
    seq_position: int = 1
    tune: float = 0.0          # centy
    transpose: int = 0         # půltóny
    volume: float = 0.0        # dB
    pan: float = 0.0           # -100 (vlevo) .. 100 (vpravo)
    veltrack: float = 100.0    # % (amp_veltrack)
    attack: float = 0.0
    release: float = 0.001
    offset: int = 0            # vzorky od začátku souboru
    loop_mode: str = "no_loop"
    loop_start: int = 0
    loop_end: int = 0
    trigger: str = "attack"
    ignored: tuple = ()        # opcody, kterým parser nerozumí


HEADERS = ("control", "global", "master", "group", "region")
_ALIASES = {"loopmode": "loop_mode", "loopstart": "loop_start",
            "loopend": "loop_end"}
_INT = {"lovel": "lovel", "hivel": "hivel", "seq_length": "seq_length",
        "seq_position": "seq_position", "transpose": "transpose",
        "offset": "offset", "loop_start": "loop_start", "loop_end": "loop_end"}
_FLOAT = {"tune": "tune", "volume": "volume", "pan": "pan",
          "amp_veltrack": "veltrack", "ampeg_attack": "attack",
          "ampeg_release": "release"}
_NOTE_RE = re.compile(r"^([a-gA-G])([#b]?)(-?\d+)$")
_PCS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_TOKEN = re.compile(r"<(\w+)>|([A-Za-z0-9_]+)=")
_DEFINE = re.compile(r"^\s*#define\s+(\$\w+)\s+(.*?)\s*$")
_INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"\s*$')


def note_number(value) -> int:
    """'60', 'c4', 'F#3', 'eb2' -> MIDI číslo (c4 = 60)."""
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    m = _NOTE_RE.match(text)
    if not m:
        raise SfzError(f"neplatná nota {value!r} (čekám číslo nebo např. c4, f#3)")
    pc = _PCS[m.group(1).lower()] + {"#": 1, "b": -1, "": 0}[m.group(2)]
    return (int(m.group(3)) + 1) * 12 + pc


def _read(path: str) -> str:
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise SfzError(f"SFZ soubor se nedá otevřít: {path} ({e.strerror})") from e
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _expand(text: str, base_dir: str, depth: int = 0) -> str:
    """`#include` a `#define` -> čistý text."""
    if depth > 8:
        raise SfzError("#include je zanořený příliš hluboko (cyklus?)")
    defines: dict = {}
    out = []
    for line in text.splitlines():
        m = _DEFINE.match(line)
        if m:
            defines[m.group(1)] = m.group(2)
            continue
        m = _INCLUDE.match(line)
        if m:
            path = os.path.join(base_dir, m.group(1).replace("\\", "/"))
            out.append(_expand(_strip_comments(_read(path)), base_dir, depth + 1))
            continue
        for name in sorted(defines, key=len, reverse=True):
            line = line.replace(name, defines[name])
        out.append(line)
    return "\n".join(out)


def _tokens(text: str):
    """Text -> ('header', jméno, '') a ('op', jméno, hodnota).

    Hodnota sahá až k dalšímu `opcode=` nebo hlavičce na řádku, takže
    `sample=` smí obsahovat mezery.
    """
    for line in text.splitlines():
        found = list(_TOKEN.finditer(line))
        for i, m in enumerate(found):
            if m.group(1):
                yield "header", m.group(1).lower(), ""
                continue
            end = found[i + 1].start() if i + 1 < len(found) else len(line)
            yield "op", m.group(2).lower(), line[m.end():end].strip()


def _region(ops: dict, base_dir: str, default_path: str,
            source: str) -> Region | None:
    sample = ops.get("sample", "")
    if not sample or sample.startswith("*"):
        return None               # generátory (*sine, *silence) nepodporujeme
    rel = (default_path + sample).replace("\\", "/")
    r = Region(sample=os.path.normpath(os.path.join(base_dir, rel)))
    ignored = []
    # `key` nastaví i keycenter, takže musí přijít před pitch_keycenter
    for name, full in sorted(ops.items(), key=lambda kv: kv[0] != "key"):
        if name == "sample":
            continue
        # jen první slovo: skutečné knihovny mívají za hodnotou smetí
        # (Salamander: „pitch_keycenter=102 tune-27“)
        value = full.split()[0] if full.split() else ""
        try:
            if name == "key":
                r.lokey = r.hikey = r.keycenter = note_number(value)
            elif name == "pitch_keycenter":
                if value.lower() != "sample":
                    r.keycenter = note_number(value)
            elif name in ("lokey", "hikey"):
                setattr(r, name, note_number(value))
            elif name in _INT:
                setattr(r, _INT[name], int(float(value)))
            elif name in _FLOAT:
                setattr(r, _FLOAT[name], float(value))
            elif name in ("loop_mode", "trigger"):
                setattr(r, name, value.lower())
            else:
                ignored.append(name)
        except ValueError as e:
            raise SfzError(f"{source}: opcode {name}={full!r} má neplatnou "
                           f"hodnotu") from e
    r.ignored = tuple(sorted(ignored))
    return r


def parse(text: str, base_dir: str = ".", source: str = "<sfz>") -> list[Region]:
    """Text SFZ -> regiony s cestami ke vzorkům složenými s `base_dir`."""
    text = _expand(_strip_comments(text), base_dir)
    scopes: dict = {h: {} for h in HEADERS}
    current = None
    regions: list[Region] = []

    def close():
        if current != "region":
            return
        merged: dict = {}
        for h in ("global", "master", "group", "region"):
            merged.update(scopes[h])
        r = _region(merged, base_dir, scopes["control"].get("default_path", ""),
                    source)
        if r is not None:
            regions.append(r)

    resets = {"control": ("control",), "global": ("global", "master", "group"),
              "master": ("master", "group"), "group": ("group",),
              "region": ("region",)}
    for kind, name, value in _tokens(text):
        if kind == "header":
            close()
            current = name if name in HEADERS else None
            for h in resets.get(current, ()):
                scopes[h] = {}
            continue
        if current is not None:
            scopes[current][_ALIASES.get(name, name)] = value
    close()
    return regions


def load(path: str) -> list[Region]:
    """SFZ soubor -> regiony (cesty ke vzorkům relativně k souboru)."""
    base = os.path.dirname(os.path.abspath(path))
    return parse(_read(path), base, source=os.path.basename(path))
