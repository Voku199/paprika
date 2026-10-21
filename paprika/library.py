"""Katalog samplovaných nástrojů a cesty ke vzorkům.

Nástroj se jmenuje `jméno.artikulace` (`violins.pizz`); bez artikulace
platí první, výchozí. Vzorky leží mimo repo, viz `samples_root()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .sampler import Sampler

__all__ = ["Instrument", "INSTRUMENTS", "GROUPS", "SHORT_ARTS", "LOOP_ARTS", "SALAMANDER_SFZ",
           "SamplesMissing", "samples_root", "split_name", "is_sampled",
           "instrument_names", "is_installed", "instrument",
           "instrument_defaults", "instrument_fx", "status_lines"]

GROUPS = ("piano", "strings", "winds", "brass", "mallets", "percussion", "voices")
SHORT_ARTS = ("pizz", "spic", "stac", "hit")
LOOP_ARTS = ("roll",)          # vířivé artikulace bicích drží smyčkou
SALAMANDER_SFZ = ("salamander/SalamanderGrandPianoV3_44.1khz16bit/"
                  "SalamanderGrandPianoV3Retuned.sfz")
GRAND_GAIN_DB = -4.6  # max RMS C4v16 (-13.4 dBFS) -> -18 dBFS, změřeno 2026-09-24


class SamplesMissing(RuntimeError):
    """Nástroj potřebuje vzorky, které nejsou nainstalované."""


@dataclass
class Instrument:
    """Samplovaný nástroj. `arts` = artikulace -> složka ve VSCO repu."""

    name: str
    group: str
    arts: dict
    library: str = "vsco2"
    mono: bool = False            # sólový nástroj: stopa je mono (legato)
    one_shot: bool = False        # všechno dozní samo (harfa, mallets)
    release: float = 0.3          # s, zapisuje se do generovaného SFZ
    undamped_from: int | None = None
    fx: str = "section"
    gain_db: float = 0.0

    @property
    def default_art(self) -> str:
        return next(iter(self.arts))

    def is_one_shot(self, art: str) -> bool:
        return self.one_shot or art in SHORT_ARTS

    def loops(self, art: str) -> bool:
        """Výdrž, která smí běžet déle než nahrávka (smyčka)."""
        return ((self.group in ("strings", "winds", "brass", "voices")
                 and not self.is_one_shot(art)) or art in LOOP_ARTS)

    def sfz(self, art: str) -> str:
        """Cesta k SFZ relativně ke kořeni vzorků."""
        if self.library == "salamander":
            return SALAMANDER_SFZ
        if self.library == "vocalset":
            return f"vocalset/sfz/{self.name}.{art}.sfz"
        return f"vsco2/sfz/{self.name}.{art}.sfz"


def _arts(base: str, **arts) -> dict:
    return {art: f"{base}/{folder}" for art, folder in arts.items()}


_VOWELS = {"aah": "a", "oh": "o", "ooh": "u", "eh": "e"}

_LIST = [
    Instrument("grand_piano", "piano", {"sus": ""}, library="salamander",
               undamped_from=89, fx="piano", gain_db=GRAND_GAIN_DB),
    Instrument("upright_piano", "piano", {"sus": "Keys/Upright Nr1"},
               release=0.4, undamped_from=89, fx="piano"),
    Instrument("violins", "strings", _arts("Strings/Violin Section", sus="susVib",
                                           trem="Trem", pizz="Pizz", spic="Spic")),
    Instrument("violas", "strings", _arts("Strings/Viola Section", sus="susvib",
                                          trem="trem", pizz="pizz", spic="spic")),
    Instrument("celli", "strings", _arts("Strings/Cello Section", sus="susvib",
                                         trem="trem", pizz="pizzT", spic="spic")),
    Instrument("violin_solo", "strings",
               _arts("Strings/Solo Violin", sus="Arco Vib", trem="Trem",
                     pizz="Pizz", spic="spic"), mono=True, fx="solo"),
    Instrument("contrabass", "strings",
               _arts("Strings/Solo Contrabass", sus="SusVib", nv="SusNV",
                     trem="Trem", pizz="Pizz", spic="Spic"), mono=True, fx="solo"),
    Instrument("concert_harp", "strings", {"sus": "Strings/Harp"},
               one_shot=True, fx="solo"),
    Instrument("flute", "winds", _arts("Woodwinds/Flute", sus="susvib", nv="susNV",
                                       exp="expvib", stac="stac"),
               mono=True, fx="solo"),
    Instrument("oboe", "winds", _arts("Woodwinds/Oboe", sus="Vib", nv="Sus",
                                      stac="Stacc"), mono=True, fx="solo"),
    Instrument("clarinet", "winds", _arts("Woodwinds/Clarinet", sus="susLong",
                                          stac="stac"), mono=True, fx="solo"),
    Instrument("bassoon", "winds", _arts("Woodwinds/Bassoon", sus="sus", vib="vib",
                                         stac="stac"), mono=True, fx="solo"),
    Instrument("horn", "brass", _arts("Brass/F Horn", sus="sus", stac="stac",
                                      mute="mute"),
               mono=True, release=0.25, fx="solo"),
    Instrument("trumpet", "brass", _arts("Brass/Trumpet", sus="sus", vib="susvib",
                                         stac="stac", mute="straightM-sus"),
               mono=True, release=0.25, fx="solo"),
    Instrument("trombone", "brass", _arts("Brass/Tenor Trombone", sus="sus",
                                          vib="vib", stac="stac"),
               mono=True, release=0.25, fx="solo"),
    Instrument("tuba", "brass", _arts("Brass/Tuba", sus="sus", stac="stac"),
               mono=True, release=0.25, fx="solo"),
    # zpěv z VocalSet (CC BY 4.0), artikulace = samohláska; viz voices.py
    Instrument("male_choir", "voices", _VOWELS, library="vocalset", fx="choir"),
    Instrument("basso", "voices", _VOWELS, library="vocalset", mono=True, fx="voice"),
    Instrument("tenor", "voices", _VOWELS, library="vocalset", mono=True, fx="voice"),
    Instrument("female_choir", "voices", _VOWELS, library="vocalset", fx="choir"),
    Instrument("soprano", "voices", _VOWELS, library="vocalset", mono=True, fx="voice"),
    Instrument("alto", "voices", _VOWELS, library="vocalset", mono=True, fx="voice"),
    Instrument("marimba", "mallets", {"sus": "Percussion/Marimba"},
               one_shot=True, fx="mallets"),
    Instrument("glockenspiel", "mallets", {"sus": "Percussion/Glock"},
               one_shot=True, fx="mallets"),
    Instrument("xylophone", "mallets", {"sus": "Percussion/Xylo"},
               one_shot=True, fx="mallets"),
    # pět kotlů bez noty ve jméně souboru; výšky viz install.VSCO_NAMED_PITCH
    Instrument("timpani", "percussion", {"hit": "Percussion/Timpani",
                                         "roll": "Percussion/Timpani/Rolls"},
               release=1.2, fx="timpani"),
]
INSTRUMENTS = {i.name: i for i in _LIST}


def samples_root() -> str:
    """Kořen vzorků: `PAPRIKA_SAMPLES`, jinak LOCALAPPDATA, jinak ~/.paprika."""
    env = os.environ.get("PAPRIKA_SAMPLES")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return os.path.join(base, "paprika", "samples")
    return os.path.join(os.path.expanduser("~"), ".paprika", "samples")


def split_name(name: str) -> tuple[str, str | None]:
    base, _, art = str(name).strip().lower().partition(".")
    return base, (art or None)


def is_sampled(name) -> bool:
    return isinstance(name, str) and split_name(name)[0] in INSTRUMENTS


def instrument_names() -> list[str]:
    return list(INSTRUMENTS)


def _lookup(name: str) -> tuple[Instrument, str]:
    base, art = split_name(name)
    inst = INSTRUMENTS.get(base)
    if inst is None:
        raise KeyError(f"neznámý samplovaný nástroj {base!r}; k dispozici: "
                       f"{', '.join(INSTRUMENTS)}")
    art = art or inst.default_art
    if art not in inst.arts:
        raise KeyError(f"nástroj {base!r} nemá artikulaci {art!r}; k dispozici: "
                       f"{', '.join(inst.arts)}")
    return inst, art


def is_installed(name: str, root: str | None = None) -> bool:
    inst, art = _lookup(name)
    return os.path.isfile(os.path.join(root or samples_root(), inst.sfz(art)))


def instrument(name: str, root: str | None = None) -> Sampler:
    """`"violins.pizz"` -> `Sampler`. Chybějící vzorky -> `SamplesMissing`."""
    inst, art = _lookup(name)
    base = os.path.abspath(root or samples_root())
    path = os.path.join(base, inst.sfz(art))
    if not os.path.isfile(path):
        raise SamplesMissing(
            f"nástroj {inst.name!r} potřebuje vzorky skupiny {inst.group!r} — "
            f"nainstaluj: python -m paprika samples install {inst.group} "
            f"(cíl: {base})")
    label = inst.name if art == inst.default_art else f"{inst.name}.{art}"
    return Sampler(path, name=label, legato=not inst.is_one_shot(art),
                   undamped_from=inst.undamped_from, gain_db=inst.gain_db)


def instrument_defaults(name: str) -> dict:
    """Výchozí nastavení stopy (sólové nástroje jsou mono = legato)."""
    inst, _ = _lookup(name)
    return {"mono": inst.mono}


def instrument_fx(name: str) -> list:
    """Doporučený fx řetěz — vždy nové instance efektů."""
    from .fx_space import Hall
    inst, _ = _lookup(name)
    table = {
        "piano": lambda: [Hall("hall", mix=0.22)],
        "section": lambda: [Hall("hall", mix=0.3)],
        "solo": lambda: [Hall("hall", mix=0.25)],
        "mallets": lambda: [Hall("hall", mix=0.2)],
        "timpani": lambda: [Hall("hall", mix=0.32)],
        "choir": lambda: [Hall("cathedral", mix=0.38)],
        "voice": lambda: [Hall("hall", mix=0.3)],
    }
    return table[inst.fx]()


def status_lines(root: str | None = None) -> list[str]:
    """Řádky pro `python -m paprika samples`: cesta a stav skupin."""
    base = os.path.abspath(root or samples_root())
    lines = [f"vzorky: {base}  (jinam přes proměnnou PAPRIKA_SAMPLES)"]
    for g in GROUPS:
        insts = [i for i in INSTRUMENTS.values() if i.group == g]
        ok = all(is_installed(i.name, base) for i in insts)
        names = ", ".join(i.name if len(i.arts) == 1
                          else f"{i.name} ({' '.join(i.arts)})" for i in insts)
        lines.append(f"{g:<10} {'[ok]   ' if ok else '[chybí]'} {names}")
    lines.append("instalace: python -m paprika samples install <skupina|all>")
    return lines
