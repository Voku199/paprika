"""Instalace vzorků: stažení, rozbalení, vygenerování SFZ.

Jediný modul Paprika, který sahá na síť. Stahování pokračuje: hotové
soubory se správnou velikostí se přeskočí, rozstahované leží jako `.part`.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import posixpath
import re
import tarfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .library import GROUPS, INSTRUMENTS, SALAMANDER_SFZ, Instrument, samples_root
from .sampler import load_wav, to_float
from .sfz import note_number

__all__ = ["InstallError", "parse_sample_name", "plan_regions", "write_sfz",
           "download", "extract", "vsco_files", "install", "MANIFEST",
           "SALAMANDER_URL", "SALAMANDER_SIZE", "VSCO_SHA", "VSCO_OCTAVE_ZERO",
           "TARGET_RMS_DB"]

VSCO_SHA = "440300901dfe9275fd84e0b7763af1f8443ae62e"
VSCO_TREE = ("https://api.github.com/repos/sgossner/VSCO-2-CE/git/trees/"
             f"{VSCO_SHA}?recursive=1")
VSCO_RAW = f"https://raw.githubusercontent.com/sgossner/VSCO-2-CE/{VSCO_SHA}/"
SALAMANDER_URL = ("https://archive.org/download/SalamanderGrandPianoV3/"
                  "SalamanderGrandPianoV3_44.1khz16bit.tar.bz2")
SALAMANDER_SIZE = 488713261
MANIFEST = "paprika_samples.json"
CREDITS = """Vzorky nainstalované Paprikou
==============================

Salamander Grand Piano V3 (nástroj grand_piano)
  Autor: Alexander Holm. Licence: CC-BY 3.0
  https://creativecommons.org/licenses/by/3.0/
  POVINNÉ při použití (i ve hře): uveď v titulcích
  "Salamander Grand Piano V3 by Alexander Holm, licensed under CC-BY 3.0".

VSCO 2 Community Edition (smyčce, dechy, žestě, mallets, upright_piano)
  Versilian Studios LLC a přispěvatelé (Upright Piano: Simon Dalzell,
  Ivy Audio). Licence: CC0 1.0 — kredit není povinný, ale je vítaný.
  https://github.com/sgossner/VSCO-2-CE
"""
TARGET_RMS_DB = -18.0     # medián max RMS nejhlasitější vrstvy přes kořeny
PEAK_CEILING = 0.9        # žádný vzorek nejhlasitější vrstvy nad tuhle špičku
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


def _normalize_db(regions: list, base: str) -> float:
    """Hlasitost `<global> volume` pro vygenerované SFZ.

    Medián max RMS (100 ms okna) přes kořeny nejhlasitější vrstvy míří na
    `TARGET_RMS_DB`; zároveň žádný vzorek té vrstvy nesmí přesáhnout
    špičku `PEAK_CEILING` (krátké staccato má malé RMS a velkou špičku).
    """
    top = max(r["hivel"] for r in regions)
    first = {}
    for r in regions:
        if r["hivel"] == top:
            first.setdefault(r["pitch_keycenter"], r)
    levels, peaks = [], []
    for r in first.values():
        sr, data = load_wav(os.path.join(base, *r["sample"].split("/")))
        x = to_float(data[: sr * 4]).mean(axis=1)
        peaks.append(float(np.abs(x).max()) if x.size else 0.0)
        win = max(1, int(0.1 * sr))
        n = (x.size // win) * win
        if n:
            rms = float(np.sqrt(np.mean(x[:n].reshape(-1, win) ** 2, axis=1)).max())
            levels.append(20.0 * math.log10(max(rms, 1e-6)))
    volume = TARGET_RMS_DB - (float(np.median(levels)) if levels else TARGET_RMS_DB)
    loudest = max(peaks, default=0.0)
    if loudest > 0.0:
        volume = min(volume, 20.0 * math.log10(PEAK_CEILING / loudest))
    return volume


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
    volume = _normalize_db(regions, base)
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


# --------------------------------------------------------------------------
# síť
# --------------------------------------------------------------------------

def _urlopen(url: str, timeout: float = 60.0):
    req = urllib.request.Request(url, headers={"User-Agent": "paprika-sampler"})
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - pevné https URL


def download(url: str, dst: str, size: int | None = None) -> bool:
    """Stáhne `url` do `dst` přes `.part`. False = soubor už je hotový."""
    if os.path.isfile(dst) and (size is None or os.path.getsize(dst) == size):
        return False
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    part = dst + ".part"
    try:
        with _urlopen(url) as resp, open(part, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except OSError as e:              # URLError je podtřída OSError
        if os.path.exists(part):
            os.remove(part)
        raise InstallError(f"stažení selhalo: {url} ({e}) — spusť instalaci "
                           f"znovu, hotové soubory se přeskočí") from e
    got = os.path.getsize(part)
    if size is not None and got != size:
        os.remove(part)
        raise InstallError(f"stažený soubor má {got} B místo {size} B: {url} — "
                           f"spusť instalaci znovu")
    os.replace(part, dst)
    return True


def extract(archive: str, dest: str) -> None:
    """Rozbalí tar(.bz2) do `dest`; člen mimo `dest` nebo odkaz = chyba."""
    dest_abs = os.path.abspath(dest)
    os.makedirs(dest_abs, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        for m in tar:
            target = os.path.abspath(os.path.join(dest_abs, m.name))
            if (os.path.commonpath([dest_abs, target]) != dest_abs
                    or not (m.isfile() or m.isdir())):
                raise InstallError(f"archiv {archive} obsahuje nebezpečnou "
                                   f"položku: {m.name}")
            tar.extract(m, dest_abs, filter="data")


def _install_salamander(root: str, log) -> None:
    dest = os.path.join(root, "salamander")
    marker = os.path.join(dest, ".complete")
    if os.path.isfile(marker):
        log("piano: Salamander Grand Piano už je nainstalovaný")
        return
    archive = os.path.join(root, "_downloads", os.path.basename(SALAMANDER_URL))
    log("piano: stahuji Salamander Grand Piano (489 MB) ...")
    download(SALAMANDER_URL, archive, SALAMANDER_SIZE)
    log("piano: rozbaluji (~1.2 GB) ...")
    extract(archive, dest)
    if not os.path.isfile(os.path.join(root, SALAMANDER_SFZ)):
        raise InstallError(f"v archivu chybí {SALAMANDER_SFZ}")
    with open(marker, "w", encoding="utf-8") as f:
        f.write(datetime.date.today().isoformat() + "\n")


def vsco_files(folders) -> list[tuple[str, int]]:
    """WAV soubory VSCO repa přímo ve složkách `folders` (cesta, velikost)."""
    try:
        with _urlopen(VSCO_TREE) as resp:
            tree = json.load(resp)["tree"]
    except (OSError, ValueError, KeyError) as e:
        raise InstallError(f"seznam souborů VSCO 2 CE se nepodařilo získat ({e}) "
                           f"— zkus to znovu za chvíli (limit GitHub API)") from e
    want = set(folders)
    return [(x["path"], int(x["size"])) for x in tree
            if x.get("type") == "blob" and x["path"].lower().endswith(".wav")
            and posixpath.dirname(x["path"]) in want]


def _download_vsco(files, root: str, log) -> None:
    base = os.path.join(root, "vsco2")
    total = sum(s for _, s in files)
    step = max(1, len(files) // 10)

    def one(item):
        path, size = item
        download(VSCO_RAW + urllib.parse.quote(path),
                 os.path.join(base, *path.split("/")), size)
        return size

    done = 0
    with ThreadPoolExecutor(8) as ex:
        for i, size in enumerate(ex.map(one, files), 1):
            done += size
            if i % step == 0 or i == len(files):
                log(f"  {i}/{len(files)} souborů, "
                    f"{done / 1e6:.0f}/{total / 1e6:.0f} MB")


def _write_manifest(root: str, groups) -> None:
    path = os.path.join(root, MANIFEST)
    data: dict = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except ValueError:
            data = {}
    today = datetime.date.today().isoformat()
    data.setdefault("groups", {})
    for g in groups:
        data["groups"][g] = {"installed": today}
    data["vsco2_commit"] = VSCO_SHA
    data["salamander"] = SALAMANDER_URL
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def install(group: str, root: str | None = None, log=print) -> list[str]:
    """Nainstaluje skupinu (`piano`, `strings`, ..., `all`). Vrací nástroje."""
    groups = list(GROUPS) if group == "all" else [group]
    for g in groups:
        if g not in GROUPS:
            raise InstallError(f"neznámá skupina {g!r}; k dispozici: "
                               f"{', '.join(GROUPS)}, all")
    root = os.path.abspath(root or samples_root())
    os.makedirs(root, exist_ok=True)
    done: list[str] = []
    for g in groups:
        insts = [i for i in INSTRUMENTS.values() if i.group == g]
        if any(i.library == "salamander" for i in insts):
            _install_salamander(root, log)
        vsco = [i for i in insts if i.library == "vsco2"]
        if vsco:
            folders = sorted({f for i in vsco for f in i.arts.values()})
            log(f"{g}: zjišťuji soubory VSCO 2 CE ...")
            files = vsco_files(folders)
            log(f"{g}: {len(files)} souborů, "
                f"{sum(s for _, s in files) / 1e6:.0f} MB")
            _download_vsco(files, root, log)
            for inst in vsco:
                for art in inst.arts:
                    write_sfz(inst, art, root)
            log(f"{g}: SFZ vygenerováno")
        done += [i.name for i in insts]
    _write_manifest(root, groups)
    with open(os.path.join(root, "CREDITS.txt"), "w", encoding="utf-8") as f:
        f.write(CREDITS)
    return done
