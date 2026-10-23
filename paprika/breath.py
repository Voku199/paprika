"""Nádechy z VocalSet: nahrávky zpěváků -> vyříznuté nádechy -> model obálek.

Zpěvák se před každou frází nadechne a VocalSet ty nádechy nahrál spolu
se zpěvem: cvičení začínají nádechem a mezi dlouhými tóny je slyšet další.
Modul je najde, popíše a shrne do malého modelu, ze kterého `human.py`
syntetizuje nádechy s barvou skutečného zpěváka:

1. **Hledání.** Před každým nasazením tónu (YIN, znělý úsek aspoň 0.15 s)
   se hledá šum nad hladinou místnosti, který skončí nejvýš `MAX_LEAD` s
   před tónem — v pásmu `DETECT_BAND`, protože šum místnosti je hlavně
   dunění a brum a širokopásmová hlasitost by tichý nádech schovala. Rámce
   s tónovým spektrem (doznívání předchozího tónu, které YIN přehlédne) do
   nádechu nepatří. Spektrum šumu místnosti se změří z nejtišších míst
   souboru a od spektra kandidáta odečte; kandidát pak projde kontrolou:
   není to dunění (výkon pod 150 Hz), cvak (crest), tón (spektrální
   plochost) ani sykavka.
2. **Popis.** Nádech je šum, takže jeho realističnost je v časovém průběhu
   spektrální obálky, ne v jemné struktuře. Každý nádech se popíše obálkou
   hlasitosti (dB vůči maximu) a tvarem spektra v `BANDS` mel pásmech (dB
   vůči celkové hlasitosti rámce), obojí časově normalizované na `K` bodů.
3. **Model.** PCA přes nádechy zpěváka (a souhrnně přes muže a ženy):
   průměr, hlavní komponenty a jejich rozptyl. Náhodné koeficienty dají
   nekonečno nádechů, které se neopakují, ale zní jako ten zpěvák.
4. **Syntéza.** Bílý šum v STFT vynásobený obálkou z modelu, libovolná
   délka (obálky jsou v normalizovaném čase).

Nesahá na síť: čte WAV z `vocalset/FULL` (rozbalené `install.py`) a píše
`vocalset/breath/models.json`. Vzorky jsou CC BY 4.0, model z nich
odvozený nese stejnou atribuci.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.io import wavfile

from . import voices
from .library import SamplesMissing, samples_root

__all__ = ["Breath", "MODELS", "find_breaths", "describe", "fit", "build",
           "load", "resolve", "model_path", "synth", "stats_lines",
           "write_examples", "MissingModel"]

MODELS = "models.json"
VERSION = 1
SOURCE = ("VocalSet by J. Wilkins, P. Seetharaman, A. Wahl and B. Pardo, "
          "licensed under CC BY 4.0")

# --- hledání (orientační prahy pro studiové nahrávky VocalSet) -------------
MIN_SUNG = 0.15        # s: kratší znělý úsek není nasazení tónu
MIN_DUR = 0.12         # s: nejkratší nádech
MAX_DUR = 1.5          # s: nejdelší nádech
MAX_LEAD = 0.4         # s: nádech končí nejvýš tolik před nasazením
GUARD = 3              # rámce (30 ms) kolem zpěvu, kde se nádech nehledá
HOLES = 2              # rámce: díra uvnitř nádechu, která ho ještě nepřeruší
FLOOR_PCT = 10         # percentil hlasitosti rámců mimo zpěv = hladina místnosti
ABOVE_FLOOR_DB = 8.0   # nádech je aspoň o tolik nad hladinou místnosti
MAX_LOW = 0.35         # podíl výkonu pod 150 Hz: víc = dunění, ruch
MIN_CENTROID = 500.0   # Hz: tmavší je spíš ruch než dech
MAX_CENTROID = 9000.0  # Hz: světlejší je sykavka nebo šum elektroniky
MIN_FLATNESS = 0.05    # spektrální plochost 300 Hz–8 kHz: tón má méně
MAX_CREST_DB = 22.0    # peak/RMS: cvak, mlasknutí, ťuknutí do mikrofonu
# šum místnosti ve VocalSet je hlavně dunění pod 150 Hz (změřeno 2026-10-23
# na 41 souborech: o 14–25 dB na bin silnější než 150 Hz–8 kHz). Nádechy
# jsou nad ním o 20–30 dB v pásmu 150 Hz–8 kHz, ale jen o ~8 dB pod 150 Hz,
# a protože širokopásmovou hlasitost dělá dunění, stály jen 3–8 dB nad
# hladinou: starý detektor našel 66 nádechů z 1485 souborů
DETECT_BAND = (300.0, 16000.0)   # Hz: hlasitost pro hledání se měří v tomhle pásmu
NOISE_DB = 3.0         # rámce mimo zpěv do hladiny + tolik = šum místnosti
SUBTRACT_FLOOR = 0.05  # odečtení šumu nechá aspoň tento podíl výkonu (spektrální podlaha)
# rámec (20 ms) s plochostí 300 Hz–8 kHz pod tímhle je tón, ne dech: YIN
# doznívání tónu často nepozná a bez téhle kontroly začínal nádech ještě
# v předchozím tónu (změřeno na VocalSet male3, 2026-10-23: 40–60 ms
# s plochostí 0.00 a 86–96 % energie pod 500 Hz; nádechy mají 0.03–0.4)
TONAL_FLAT = 0.02

# --- popis a model ----------------------------------------------------------
BANDS = 28             # mel pásma
F_LO, F_HI = 150.0, 16000.0
ROLLOFF_DB = 12.0      # dB/okt nad posledním pásmem (syntéza nedrží šum až do 22 kHz)
K = 24                 # body normalizovaného času
FLOOR_DB = -40.0       # obálka hlasitosti se dole ořízne
COMPONENTS = 6
MIN_BREATHS = 8        # méně nádechů = zpěvák nemá vlastní model, platí souhrnný
VOWEL_MIN = 4          # nádechů na samohlásku, aby měla vlastní posun průměru
EXAMPLES = 4           # odkazů na skutečné nádechy v modelu (k poslechu)


class MissingModel(SamplesMissing):
    """Model dechu chybí (VocalSet není nainstalovaný nebo je z doby před H2)."""


@dataclass
class Breath:
    """Jeden nalezený nádech."""

    t0: float           # s
    t1: float           # s
    rel_db: float       # RMS nádechu vůči mediánu zpěvu v souboru
    centroid: float     # Hz
    flatness: float


# --------------------------------------------------------------------------
# hledání
# --------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Souvislé běhy True jako `(start, konec)`."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _close(mask: np.ndarray, gap: int) -> np.ndarray:
    """Zalepí díry kratší nebo rovné `gap` mezi běhy True."""
    out = np.asarray(mask, dtype=bool).copy()
    runs = _runs(out)
    for (_, e), (s, _) in zip(runs, runs[1:]):
        if s - e <= gap:
            out[e:s] = True
    return out


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    m = np.asarray(mask, dtype=np.float64)
    return np.convolve(m, np.ones(2 * k + 1), mode="same") > 0.5


def _tonal(frames: np.ndarray, sr: int) -> np.ndarray:
    """Které rámce mají tónové spektrum (harmonické, plochost < `TONAL_FLAT`)."""
    out = np.zeros(frames.shape[0], dtype=bool)
    if frames.size == 0:
        return out
    f = np.fft.rfftfreq(4096, 1.0 / sr)
    band = (f >= 300.0) & (f <= 8000.0)
    win = np.hanning(frames.shape[1])
    for s in range(0, frames.shape[0], 256):        # po blocích kvůli paměti
        p = np.abs(np.fft.rfft(frames[s:s + 256] * win, 4096, axis=1))[:, band] ** 2 + 1e-20
        flat = np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)
        out[s:s + 256] = flat < TONAL_FLAT
    return out


def _noise(x: np.ndarray, sr: int, quiet: np.ndarray, hs: int) -> dict | None:
    """Spektrum šumu místnosti z rámců `quiet` (souvislé úseky aspoň jedno
    okno): Welch pro `_check`, střední výkon STFT pro `describe` a rozptyl."""
    nfft, hop = _stft_size(sr)
    win = np.hanning(nfft)
    welch, stft, var, total = 0.0, 0.0, 0.0, 0
    for a, b in _runs(quiet):
        seg = x[a * hs:min(x.size, b * hs)]
        if seg.size < max(nfft, 1024):
            continue
        _, p = signal.welch(seg, sr, nperseg=1024)
        frames = sliding_window_view(seg, nfft)[::hop]
        welch = welch + p * seg.size
        stft = stft + (np.abs(np.fft.rfft(frames * win, axis=1)) ** 2).mean(axis=0) * seg.size
        var += float(np.sum(seg ** 2))
        total += seg.size
    if total == 0:
        return None
    return {"welch": welch / total, "stft": stft / total, "var": var / total}


def _check(seg: np.ndarray, sr: int, noise: dict | None = None) -> tuple[str | None, dict]:
    """Je úsek dech? Vrací (důvod odmítnutí nebo None, změřené hodnoty).
    S `noise` (z `_noise`) se nejdřív odečte spektrum šumu místnosti."""
    f, p = signal.welch(seg, sr, nperseg=min(1024, seg.size))
    if noise is not None and noise["welch"].size == p.size:
        p = np.maximum(p - noise["welch"], SUBTRACT_FLOOR * p)
    tot = float(p.sum())
    if tot <= 0.0:
        return "ticho", {}
    band = (f >= 300.0) & (f <= 8000.0)
    pb = p[band] + 1e-20
    info = {
        "centroid": float((f * p).sum() / tot),
        "flatness": float(np.exp(np.mean(np.log(pb))) / np.mean(pb)),
    }
    rms = float(np.sqrt(np.mean(seg ** 2)))
    crest = 20.0 * math.log10(float(np.abs(seg).max()) / max(rms, 1e-12))
    if float(p[f < 150.0].sum()) / tot > MAX_LOW:
        return "dunění", info
    if info["centroid"] < MIN_CENTROID:
        return "tmavý", info
    if info["centroid"] > MAX_CENTROID:
        return "sykavý", info
    if info["flatness"] < MIN_FLATNESS:
        return "tónový", info
    if crest > MAX_CREST_DB:
        return "cvak", info
    return None, info


def find_breaths(x: np.ndarray, sr: int) -> tuple[list[Breath], Counter]:
    """Nádechy před nasazeními tónů v mono nahrávce `x`.

    Vrací nalezené nádechy a počty odmítnutých kandidátů podle důvodu
    (`bez nádechu` = před nasazením není šum nad hladinou místnosti).
    """
    found, rejects, _ = _find(x, sr)
    return found, rejects


def _find(x: np.ndarray, sr: int) -> tuple[list[Breath], Counter, dict | None]:
    """`find_breaths` + spektrum šumu místnosti souboru (pro `describe`)."""
    x = np.asarray(x, dtype=np.float64)
    pitch, _, hop = voices.track_pitch(x, sr)
    rejects: Counter = Counter()
    nf = pitch.size
    if nf == 0:
        return [], rejects, None
    hs = max(1, int(round(hop * sr)))
    fw = max(1, int(0.02 * sr))

    def framed(sig):
        pad = np.zeros(nf * hs + fw)
        pad[:min(sig.size, pad.size)] = sig[:pad.size]
        return sliding_window_view(pad, fw)[::hs][:nf]

    frames = framed(x)
    full_db = 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)
    sos = signal.butter(4, [DETECT_BAND[0], min(DETECT_BAND[1], 0.45 * sr)],
                        btype="bandpass", fs=sr, output="sos")
    high = signal.sosfiltfilt(sos, x) if x.size > 64 else x
    rms_db = 20.0 * np.log10(np.sqrt(np.mean(framed(high) ** 2, axis=1)) + 1e-9)

    # zpěv: znělé běhy aspoň MIN_SUNG; krátké výpadky YIN uvnitř tónu zalepit
    voiced = _close(~np.isnan(pitch), 3)
    runs = [(a, b) for a, b in _runs(voiced) if (b - a) * hop >= MIN_SUNG]
    if not runs:
        return [], rejects, None
    sung = np.zeros(nf, dtype=bool)
    for a, b in runs:
        sung[a:b] = True
    near = _dilate(sung, GUARD)
    quiet = rms_db[~near]
    floor = float(np.percentile(quiet if quiet.size else rms_db, FLOOR_PCT))
    level = float(np.median(full_db[sung]))
    noise = _noise(x, sr, ~near & (rms_db <= floor + NOISE_DB), hs)
    active = (rms_db > floor + ABOVE_FLOOR_DB) & ~near
    cand = np.flatnonzero(active)
    active[cand[_tonal(frames[cand], sr)]] = False

    found: list[Breath] = []
    prev_end = 0
    lead = int(round(MAX_LEAD / hop))
    for a, b in runs:
        lo = prev_end
        prev_end = min(nf, b + GUARD)
        j = a - 1
        while j >= lo and not active[j]:
            j -= 1
        if j < lo or a - 1 - j > lead + GUARD:
            rejects["bez nádechu"] += 1
            continue
        first, misses = j, 0
        for m in range(j - 1, lo - 1, -1):
            if active[m]:
                first, misses = m, 0
            else:
                misses += 1
                if misses > HOLES:
                    break
        dur = (j + 1 - first) * hop
        if dur < MIN_DUR:
            rejects["krátký"] += 1
            continue
        if dur > MAX_DUR:
            rejects["dlouhý"] += 1
            continue
        s0, s1 = first * hs, min(x.size, (j + 1) * hs + fw)
        seg = x[s0:s1]
        why, info = _check(seg, sr, noise)
        if why:
            rejects[why] += 1
            continue
        power = float(np.mean(seg ** 2))
        if noise is not None:
            power = max(power - noise["var"], SUBTRACT_FLOOR * power)
        rel = 10.0 * math.log10(power + 1e-24) - level
        found.append(Breath(round(float(s0) / sr, 4), round(float(s1) / sr, 4), round(rel, 2),
                            round(info["centroid"], 1), round(info["flatness"], 4)))
    return found, rejects, noise


# --------------------------------------------------------------------------
# popis
# --------------------------------------------------------------------------

_BANKS: dict = {}


def _stft_size(sr: int) -> tuple[int, int]:
    """(nfft, hop): okno kolem 23 ms, překryv 75 %."""
    nfft = 1 << int(round(math.log2(0.023 * sr)))
    return nfft, nfft // 4


def _mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _bank(sr: int, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    """Trojúhelníková mel banka `(BANDS, bins)` s plochou 1 a středy pásem v Hz."""
    key = (sr, nfft)
    if key not in _BANKS:
        hi = min(F_HI, 0.47 * sr)
        edges = 700.0 * (10.0 ** (np.linspace(_mel(F_LO), _mel(hi), BANDS + 2) / 2595.0) - 1.0)
        f = np.fft.rfftfreq(nfft, 1.0 / sr)
        lo, mid, up = edges[:-2, None], edges[1:-1, None], edges[2:, None]
        fb = np.maximum(0.0, np.minimum((f - lo) / (mid - lo), (up - f) / (up - mid)))
        # plocha 1: pásmo měří hustotu výkonu na bin, ne součet — široká
        # horní pásma by jinak vyšla silnější a syntéza by zněla ostře
        fb /= np.maximum(fb.sum(axis=1, keepdims=True), 1e-12)
        _BANKS[key] = (fb, edges[1:-1].copy())
    return _BANKS[key]


def _power_frames(seg: np.ndarray, sr: int) -> np.ndarray:
    nfft, hop = _stft_size(sr)
    y = np.concatenate([np.zeros(nfft // 2), np.asarray(seg, dtype=np.float64),
                        np.zeros(nfft // 2 + nfft)])
    frames = sliding_window_view(y, nfft)[::hop]
    frames = frames[:max(1, (seg.size + hop - 1) // hop)]
    return np.abs(np.fft.rfft(frames * np.hanning(nfft), axis=1)) ** 2


def describe(seg: np.ndarray, sr: int, noise: dict | None = None) -> np.ndarray:
    """Vektor nádechu: obálka hlasitosti `K` (dB vůči maximu, ořez na
    `FLOOR_DB`) a tvar spektra `K x BANDS` (dB vůči hlasitosti rámce).

    Tvar spektra v čase k je průměrné spektrum rámců v okolí (výkon, takže
    hlasité rámce váží víc a tiché okraje nádechu barvu nezkreslí). S `noise`
    (z `_noise`) se od každého rámce odečte spektrum šumu místnosti.
    """
    power = _power_frames(seg, sr)
    nfft = power.shape[1] * 2 - 2
    fb, _ = _bank(sr, nfft)
    # hlasitost rámce jen od F_LO: pod ním model nic nepopisuje a v nahrávkách
    # je tam hlavně dunění místnosti
    above = np.fft.rfftfreq(nfft, 1.0 / sr) >= F_LO
    bandp = power @ fb.T
    frame = power[:, above].sum(axis=1)
    if noise is not None and noise["stft"].size == power.shape[1]:
        # odečítá se v pásmech, ne po binech: periodogram šumu v jednom binu
        # kolísá a odečtený průměr by v rámci nechal velký zbytek
        n = noise["stft"]
        bandp = np.maximum(bandp - (fb @ n)[None, :], SUBTRACT_FLOOR * bandp)
        frame = np.maximum(frame - float(n[above].sum()), SUBTRACT_FLOOR * frame)
    bandp = bandp + 1e-20
    frame = frame + 1e-20
    frame_db = 10.0 * np.log10(frame)
    u = np.linspace(0.0, 1.0, frame.size) if frame.size > 1 else np.zeros(1)
    uk = np.linspace(0.0, 1.0, K)
    energy = np.maximum(frame_db - frame_db.max(), FLOOR_DB)
    env = np.interp(uk, u, energy) if frame.size > 1 else np.zeros(K)
    # průměrné spektrum okolí (lineárně, pak dB): průměr logaritmů by pásma,
    # jejichž výkon po odečtení šumu kolísá, systematicky ztlumil
    g = np.exp(-((u[None, :] - uk[:, None]) / (1.5 / K)) ** 2)
    shape = 10.0 * np.log10(g @ bandp) - 10.0 * np.log10(g @ frame)[:, None]
    return np.concatenate([env, shape.ravel()])


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def _pct(values, qs=(10, 50, 90)) -> list:
    return [round(float(np.percentile(values, q)), 3) for q in qs]


def fit(vectors, meta: list[dict], name: str) -> dict:
    """PCA model z vektorů `describe` a metadat nádechů (`dur`, `rel_db`,
    `centroid`, `vowel`, `singer`, `file`, `t0`, `t1`)."""
    x = np.asarray(vectors, dtype=np.float64)
    mean = x.mean(axis=0)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    c = int(min(COMPONENTS, max(0, x.shape[0] - 1)))
    total = float((s ** 2).sum()) or 1.0
    vowels = {}
    for v in sorted({m["vowel"] for m in meta}):
        idx = [i for i, m in enumerate(meta) if m["vowel"] == v]
        if len(idx) >= VOWEL_MIN:
            vowels[v] = np.round(x[idx].mean(axis=0) - mean, 3).tolist()
    picked = sorted(meta, key=lambda m: abs(m["dur"] - float(np.median([q["dur"] for q in meta]))))
    return {
        "name": name,
        "singers": sorted({m["singer"] for m in meta}),
        "count": int(x.shape[0]),
        "k": K,
        "bands": BANDS,
        "mean": np.round(mean, 3).tolist(),
        "components": np.round(vt[:c], 5).tolist(),
        "sigma": np.round(s[:c] / math.sqrt(max(1, x.shape[0] - 1)), 4).tolist(),
        "explained": round(float((s[:c] ** 2).sum()) / total, 3),
        "vowels": vowels,
        "dur": _pct([m["dur"] for m in meta]),
        "rel_db": _pct([m["rel_db"] for m in meta]),
        "centroid": round(float(np.median([m["centroid"] for m in meta])), 1),
        "examples": [[m["file"], m["t0"], m["t1"]] for m in picked[:EXAMPLES]],
    }


def _gender(singer: str) -> str:
    return "female" if singer.startswith("female") else "male"


def model_path(root: str | None = None) -> str:
    return os.path.join(os.path.abspath(root or samples_root()), "vocalset",
                        "breath", MODELS)


def build(root: str, log=print, reuse: bool = True, singers=None) -> str:
    """`<root>/vocalset/FULL` -> `vocalset/breath/models.json`.

    Projde stejná cvičení jako `voices.py` (`voices.TAKES`). Modely jsou pro
    každého zpěváka s aspoň `MIN_BREATHS` nádechy a souhrnně `male`/`female`.
    S `reuse` a existujícím souborem nedělá nic.
    """
    out = model_path(root)
    if reuse and os.path.isfile(out):
        return out
    full = os.path.join(root, "vocalset", "FULL")
    log("voices: hledám nádechy ve VocalSet ...")
    vectors: dict = {}
    meta: dict = {}
    stats: dict = {}
    for singer in singers or voices.all_singers():
        st = {"files": 0, "breaths": 0, "rejected": Counter()}
        for take in voices.TAKES:
            folder = os.path.join(full, singer, *take.split("/"))
            if not os.path.isdir(folder):
                continue
            for fn in sorted(os.listdir(folder)):
                vowel = voices._vowel(fn)
                if not fn.lower().endswith(".wav") or vowel not in "aeiou":
                    continue
                sr, x = voices._load_mono(os.path.join(folder, fn))
                st["files"] += 1
                found, rejects, noise = _find(x, sr)
                st["rejected"].update(rejects)
                for b in found:
                    seg = x[int(b.t0 * sr):int(b.t1 * sr)]
                    vectors.setdefault(singer, []).append(describe(seg, sr, noise))
                    meta.setdefault(singer, []).append({
                        "singer": singer, "vowel": vowel, "dur": b.t1 - b.t0,
                        "rel_db": b.rel_db, "centroid": b.centroid,
                        "file": f"{singer}/{take}/{fn}", "t0": b.t0, "t1": b.t1})
                st["breaths"] += len(found)
        st["rejected"] = dict(st["rejected"])
        stats[singer] = st
        log(f"  {singer}: {st['breaths']} nádechů z {st['files']} souborů")
    models = {}
    for singer, vec in vectors.items():
        if len(vec) >= MIN_BREATHS:
            models[singer] = fit(vec, meta[singer], singer)
    for g in ("male", "female"):
        names = [s for s in vectors if _gender(s) == g]
        vec = [v for s in names for v in vectors[s]]
        if len(vec) >= MIN_BREATHS:
            models[g] = fit(vec, [m for s in names for m in meta[s]], g)
    for m in models.values():
        m.update(_verify(m))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"version": VERSION, "source": SOURCE, "models": models,
                   "stats": stats}, f, ensure_ascii=False)
    _CACHE.clear()
    log(f"voices: modely dechu {', '.join(models) or '— žádné'}")
    return out


_CACHE: dict = {}


def _read(root: str | None = None) -> dict:
    path = model_path(root)
    if not os.path.isfile(path):
        raise MissingModel(
            f"dech z dat potřebuje model z VocalSet ({path}) — nainstaluj nebo "
            f"obnov: python -m paprika samples install voices; bez dat použij "
            f"s.human(breath_model=None)")
    key = (path, os.path.getmtime(path))
    if key not in _CACHE:
        with open(path, encoding="utf-8") as f:
            _CACHE.clear()
            _CACHE[key] = json.load(f)
    return _CACHE[key]


def resolve(name: str) -> str:
    """Jméno modelu: zpěvák (`female2`), nástroj VocalSet (`soprano`,
    `male_choir.ooh`) nebo souhrnný `male` / `female`."""
    n = str(name).strip().lower()
    base = n.partition(".")[0]
    if base in voices.VOICES:
        vd = voices.VOICES[base]
        return vd.singers[0] if vd.stack == 1 else _gender(vd.singers[0])
    if n in ("male", "female") or n in voices.all_singers():
        return n
    raise KeyError(f"neznámý model dechu {name!r}; k dispozici: male, female, "
                   f"{', '.join(voices.VOICES)} nebo zpěvák "
                   f"({', '.join(voices.all_singers())})")


def load(name: str, root: str | None = None) -> dict:
    """Model jako slovník s numpy poli. Zpěvák s málo nádechy dostane
    souhrnný model svého pohlaví (je to pořád jeho skupina, ne syntéza)."""
    key = resolve(name)
    models = _read(root)["models"]
    m = models.get(key) or models.get(_gender(key))
    if m is None:
        raise MissingModel(f"model dechu {key!r} chybí — korpus nádechů je "
                           f"prázdný; zkontroluj: python -m paprika samples breaths")
    return _arrays(m)


# --------------------------------------------------------------------------
# syntéza
# --------------------------------------------------------------------------

def _vector(model: dict, rng, vowel: str | None) -> np.ndarray:
    model = _arrays(model)
    c = np.clip(rng.standard_normal(model["_sigma"].size), -2.0, 2.0) * model["_sigma"]
    v = model["_mean"] + model["_vowels"].get(vowel, 0.0)
    return v + c @ model["_comp"] if c.size else v


def synth(model: dict, n: int, sr: int, rng, vowel: str | None = None,
          tilt_db: float = 0.0) -> np.ndarray:
    """Nádech délky `n` vzorků z modelu: bílý šum v STFT × tvar spektra,
    pak obálka hlasitosti. `tilt_db` přidá tolik dB na oktávu nad 1 kHz
    (rychlejší proud = jasnější). Vrací mono float64 s RMS 1."""
    n = max(16, int(n))
    v = _vector(model, rng, vowel)
    k, b = int(model["k"]), int(model["bands"])
    env, shape = v[:k], v[k:].reshape(k, b)
    nfft, hop = _stft_size(sr)
    fb, centers = _bank(sr, nfft)
    nfr = n // hop + 4
    noise = rng.standard_normal((nfr + 3) * hop + nfft)
    win = np.hanning(nfft)
    spec = np.fft.rfft(sliding_window_view(noise, nfft)[::hop][:nfr] * win, axis=1)
    uk = np.linspace(0.0, 1.0, k)
    # střed rámce f leží ve výstupu na vzorku f * hop
    uf = np.clip(np.arange(nfr) * hop / n, 0.0, 1.0)
    frames_db = np.stack([np.interp(uf, uk, shape[:, i]) for i in range(b)], axis=1)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    logf = np.log(np.maximum(f, 1.0))
    gains_db = np.stack([np.interp(logf, np.log(centers), row) for row in frames_db])
    gains_db += float(tilt_db) * np.log2(np.maximum(f, 1000.0) / 1000.0)[None, :]
    gains_db -= ROLLOFF_DB * np.log2(np.maximum(f, centers[-1]) / centers[-1])[None, :]
    gains_db[:, f < 0.7 * F_LO] -= 30.0
    y_frames = np.fft.irfft(spec * 10.0 ** (gains_db / 20.0), nfft, axis=1) * win
    out = np.zeros(nfr * hop + nfft)
    for i in range(4):
        # rámce i, i+4, ... jsou od sebe celé okno: leží za sebou bez překryvu
        part = y_frames[i::4].ravel()
        out[i * hop:i * hop + part.size] += part
    y = out[nfft // 2:nfft // 2 + n]
    amp = 10.0 ** (np.interp(np.linspace(0.0, 1.0, n), uk, env) / 20.0)
    y = y * amp
    return y / max(1e-12, float(np.sqrt(np.mean(y ** 2))))


def _arrays(model: dict) -> dict:
    """Doplní do slovníku modelu numpy pole pro syntézu (jednou)."""
    if "_mean" not in model:
        model["_mean"] = np.asarray(model["mean"], dtype=np.float64)
        model["_comp"] = np.asarray(model["components"], dtype=np.float64).reshape(
            -1, model["_mean"].size)
        model["_sigma"] = np.asarray(model["sigma"], dtype=np.float64)
        model["_vowels"] = {v: np.asarray(d, dtype=np.float64)
                            for v, d in model["vowels"].items()}
    return model


def _verify(model: dict, sr: int = 44100, count: int = 12) -> dict:
    """Kontrola, že syntéza model reprodukuje: syntetizuje `count` nádechů
    mediánové délky, popíše je stejně jako korpus a změří odchylku
    průměrného tvaru spektra (LSD, dB), průměrné obálky (dB) a těžiště."""
    m = _arrays(dict(model))
    k = int(model["k"])
    rng = np.random.default_rng(7)
    n = int(model["dur"][1] * sr)
    ys = [synth(m, n, sr, rng) for _ in range(count)]
    vecs = np.array([describe(y, sr) for y in ys])
    got = vecs[:, k:].reshape(count, k, -1).mean(axis=(0, 1))
    want = m["_mean"][k:].reshape(k, -1).mean(axis=0)
    env = vecs[:, :k].mean(axis=0) - m["_mean"][:k]
    cents = []
    for y in ys:
        p = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(y.size, 1.0 / sr)
        cents.append(float((p * f).sum() / p.sum()))
    return {"lsd_db": round(float(np.sqrt(np.mean((got - want) ** 2))), 2),
            "env_db": round(float(np.sqrt(np.mean(env ** 2))), 2),
            "synth_centroid": round(float(np.median(cents)), 1)}


# --------------------------------------------------------------------------
# zpětná vazba (AI dech neslyší)
# --------------------------------------------------------------------------

def stats_lines(root: str | None = None) -> list[str]:
    """Řádky pro `python -m paprika samples breaths`."""
    data = _read(root)
    lines = [f"nádechy z VocalSet: {model_path(root)}",
             f"{'model':<9} {'nádechů':>7}  {'délka p10/p50/p90 s':<20} "
             f"{'pod zpěvem':>10}  {'těžiště':>7}  {'syntéza':>7}  {'PCA':>4}  "
             f"{'LSD':>5}  {'obálka':>6}"]
    for name, m in data["models"].items():
        d = m["dur"]
        lines.append(f"{name:<9} {m['count']:>7}  "
                     f"{d[0]:.2f} / {d[1]:.2f} / {d[2]:.2f}{'':<5} "
                     f"{m['rel_db'][1]:>7.0f} dB  {m['centroid']:>7.0f}  "
                     f"{m['synth_centroid']:>7.0f}  {100 * m['explained']:>3.0f}%  "
                     f"{m['lsd_db']:>5.2f}  {m['env_db']:>6.2f}")
    lines.append("(těžiště a syntéza v Hz; LSD a obálka = odchylka syntézy od "
                 "korpusu v dB, dobré je pod ~2 dB)")
    total: Counter = Counter()
    for st in data["stats"].values():
        total.update(st["rejected"])
    found = sum(st["breaths"] for st in data["stats"].values())
    files = sum(st["files"] for st in data["stats"].values())
    lines.append(f"souborů {files}, nádechů {found}; odmítnuto: " + (", ".join(
        f"{why} {n}" for why, n in total.most_common()) or "nic"))
    lines.append("bez modelu (málo nádechů, platí souhrnný): " + (", ".join(
        s for s in data["stats"] if s not in data["models"]) or "—"))
    return lines


def write_examples(out_dir: str, root: str | None = None, count: int = 3) -> list[str]:
    """Skutečné a syntetizované nádechy každého modelu jako WAV k poslechu
    (srovnané na stejnou hlasitost, -24 dBFS RMS). Skutečné mají horní
    propust na `F_LO`: pod ní model nic nepopisuje a po zesílení tichého
    nádechu by v nahrávce bylo slyšet hlavně dunění místnosti."""
    data = _read(root)
    full = os.path.join(os.path.abspath(root or samples_root()), "vocalset", "FULL")
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def save(name, sr, y):
        y = np.asarray(y, dtype=np.float64)
        y = y / max(1e-12, float(np.sqrt(np.mean(y ** 2)))) * 10.0 ** (-24.0 / 20.0)
        path = os.path.join(out_dir, name)
        wavfile.write(path, sr, (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16))
        written.append(path)

    for name in data["models"]:
        m = load(name, root)
        for i, (rel, t0, t1) in enumerate(m["examples"][:count], start=1):
            path = os.path.join(full, *rel.split("/"))
            if os.path.isfile(path):
                sr, x = voices._load_mono(path)
                sos = signal.butter(4, F_LO, btype="highpass", fs=sr, output="sos")
                x = signal.sosfiltfilt(sos, x)
                save(f"{name}_real_{i}.wav", sr, x[int(t0 * sr):int(t1 * sr)])
        rng = np.random.default_rng(1)
        for i in range(1, count + 1):
            save(f"{name}_synth_{i}.wav", 44100,
                 synth(m, int(m["dur"][1] * 44100), 44100, rng))
    return written
