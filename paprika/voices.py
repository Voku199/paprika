"""Zpěv z VocalSet: nahrávky zpěváků -> jednotlivé tóny -> SFZ nástroje.

VocalSet (CC BY 4.0) nejsou vzorky nástroje, ale cvičení: v jednom souboru
„dlouhých tónů“ zpěvák drží tři tóny za sebou (zhruba C3, C4, F4), stupnice
a arpeggia mají tón na půl vteřiny. Tenhle modul je rozřeže na stabilní tóny
(YIN výška + medián přes vibrato), každý tón uloží jako vlastní WAV se
změřeným naladěním a pod nejnižší zazpívaný tón dopočítá hlubší přes
TD-PSOLA — výška klesne, formanty (barva samohlásky) zůstanou, takže bas
nezní jako zpomalená páska.

Nesahá na síť; stažení a rozbalení dělá `install.py`. Čte jen WAV a píše
WAV + SFZ + `catalog.json`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, medfilt, resample_poly

__all__ = ["VoiceNote", "VOICES", "VOWELS", "TAKES", "track_pitch", "find_notes",
           "singer_range", "all_singers",
           "fix_long_tones", "spectral_octave",
           "psola", "extract_notes", "plan_voice", "write_voice_sfz", "build",
           "voice_quality", "CATALOG"]

CATALOG = "catalog.json"
NOTE_RMS_DB = -18.0        # každý vyříznutý tón se srovná na tuhle hlasitost
P_LAYER_DB = -7.0          # piano vrstva zní o tolik tišeji než forte
PRE_ROLL = 0.06            # s před stabilní výškou (nasazení)
TAIL_TRIM = 0.04           # s ubrané z konce (přechod na další tón)
MIN_NOTE = 0.45            # s nejkratší použitelný tón
MAX_SPREAD = 1.6           # půltóny p90-p10 uvnitř tónu (operní vibrato ~1)
PSOLA_STEPS = (-2, -4, -6)  # půltóny pod nejnižší zazpívaný tón
PSOLA_UP_STEPS = (2, 4)     # půltóny nad nejvyšší dlouhý tón (jen ženské hlasy)
PSOLA_MAX_SUB_DB = -33.0    # zazpívané tóny mají složku na f0/2 kolem -50 dB

# složka ve VocalSet -> (vrstva, styl); styl "any" smí sbor i sólo
TAKES = {
    "long_tones/straight": ("f", "straight"),
    "long_tones/forte": ("f", "vib"),
    "long_tones/pp": ("p", "any"),
    "scales/straight": ("f", "straight"),
    "arpeggios/straight": ("f", "straight"),
    "scales/slow_forte": ("f", "any"),
    "arpeggios/slow_forte": ("f", "any"),
    "scales/slow_piano": ("p", "any"),
    "arpeggios/slow_piano": ("p", "any"),
    "scales/vibrato": ("f", "vib"),
    "arpeggios/vibrato": ("f", "vib"),
}
VOWELS = {"aah": "a", "oh": "o", "ooh": "u", "eh": "e"}


@dataclass
class VoiceDef:
    """Nástroj ze zpěváků: kdo zpívá, jakým stylem, kolik hlasů naráz."""

    singers: tuple
    style: str          # "straight" (sbor, bez vibrata) nebo "vib" (sólo, opera)
    stack: int          # kolik zpěváků zní na jedné klávese (1 = sólo)
    spread: float = 0.0  # rozestavění sboru do stran (0..1)


# rozsah, který zpěvák opravdu zazpívá (MIDI); tón mimo je chyba měření
SINGER_RANGE = {"male9": (45, 80)}
DEFAULT_RANGE = (38, 70)
FEMALE_RANGE = (52, 86)
# dlouhé tóny VocalSet: každý zpěvák drží tytéž tři výšky (tenor o kvartu
# výš, zpěvačky o oktávu výš než muži - změřeno u všech devíti 2026-09-29)
LONG_TONES = {None: (48, 60, 65), "male9": (60, 65, 72), "female": (60, 72, 77)}
# ženský hlas má základní tón kolem 250-700 Hz: pod 2.5 kHz by zbyly jen
# tři harmonické, oktávu čte řada do 4 kHz
FEMALE_TOP_HZ = 4000.0


def _female(singer: str) -> bool:
    return singer.startswith("female")


def singer_range(singer: str) -> tuple:
    return SINGER_RANGE.get(singer, FEMALE_RANGE if _female(singer) else DEFAULT_RANGE)

# male8 má nejhlubší hlas sady (zazpívá F#2), male9 je tenor (drží C5);
# zpěvačky zpívají C4-F5 (některé do B5), PSOLA přidá tóny pod i nad
VOICES = {
    "male_choir": VoiceDef(("male1", "male2", "male3", "male4", "male5", "male6",
                            "male7", "male8", "male10", "male11"),
                           "straight", stack=6, spread=0.6),
    "basso": VoiceDef(("male8",), "vib", stack=1),
    "tenor": VoiceDef(("male9",), "vib", stack=1),
    "female_choir": VoiceDef(tuple(f"female{i}" for i in range(1, 10)),
                             "straight", stack=6, spread=0.6),
    # female2 zazpívá nejvýš (B5) a má nejvíc tónů; female7 má nejtmavší
    # barvu (těžiště spektra na C5 1.9 kHz proti 2.5-2.8 kHz u ostatních)
    "soprano": VoiceDef(("female2",), "vib", stack=1),
    "alto": VoiceDef(("female7",), "vib", stack=1),
}


@dataclass
class VoiceNote:
    """Jeden vyříznutý tón v katalogu."""

    file: str           # relativně k adresáři tónů
    singer: str
    vowel: str
    layer: str          # "f" / "p"
    style: str
    pitch: float        # změřená výška (MIDI, s desetinami)
    spread: float
    dur: float
    source: str
    psola: int = 0      # o kolik půltónů posunuto (0 = zazpívané)


# --------------------------------------------------------------------------
# výška a tóny
# --------------------------------------------------------------------------

def track_pitch(x: np.ndarray, sr: int, hop: float = 0.01, fmin: float = 60.0,
                fmax: float = 1000.0, frame: float = 0.04, thr: float = 0.12):
    """YIN: mono `x` -> (výška v MIDI po `hop` s, NaN = nezněl, rms, hop).

    Počítá na polovičním vzorkování; rozdílová funkce přes FFT, takže celý
    soubor je jedna maticová operace.
    """
    y = resample_poly(np.asarray(x, dtype=np.float64), 1, 2)
    fs = sr / 2.0
    fr = int(frame * fs)
    hp = max(1, int(hop * fs))
    tmax = int(fs / fmin)
    tmin = max(2, int(fs / fmax))
    width = fr + tmax
    n = (len(y) - width) // hp + 1
    if n <= 0:
        return np.full(0, np.nan), np.zeros(0), hp / fs
    out = np.full(n, np.nan)
    rms = np.zeros(n)
    size = 1 << int(math.ceil(math.log2(width + fr)))
    for b0 in range(0, n, 512):           # po blocích, ať matice nezabere GB
        idx = (np.arange(width)[None, :] + hp * np.arange(b0, min(n, b0 + 512))[:, None])
        f = y[idx]
        w = f[:, :fr]
        r = np.fft.irfft(np.fft.rfft(f, size) * np.conj(np.fft.rfft(w, size)),
                         size)[:, :tmax + 1]
        c = np.concatenate([np.zeros((f.shape[0], 1)), np.cumsum(f ** 2, axis=1)], axis=1)
        e = c[:, fr:fr + tmax + 1] - c[:, :tmax + 1]
        d = e[:, :1] + e - 2.0 * r
        d[:, 0] = 0.0
        cm = d[:, 1:] * np.arange(1, tmax + 1) / np.maximum(np.cumsum(d[:, 1:], axis=1), 1e-12)
        cm = np.concatenate([np.ones((f.shape[0], 1)), cm], axis=1)
        rr = np.sqrt(np.mean(w ** 2, axis=1))
        rms[b0:b0 + f.shape[0]] = rr
        below = cm[:, tmin:] < thr
        for i in np.where(below.any(axis=1) & (rr > 0.003))[0]:
            t = int(np.argmax(below[i])) + tmin
            while t + 1 <= tmax and cm[i, t + 1] < cm[i, t]:
                t += 1
            tf = float(t)
            if 1 <= t < tmax:
                a, m, g = cm[i, t - 1], cm[i, t], cm[i, t + 1]
                den = a - 2.0 * m + g
                if den != 0.0:
                    tf += 0.5 * (a - g) / den
            out[b0 + i] = 69.0 + 12.0 * math.log2(fs / tf / 440.0)
    return out, rms, hp / fs


def find_notes(pitch: np.ndarray, hop: float, min_len: float = MIN_NOTE,
               tol: float = 0.9) -> list:
    """Stabilní tóny v průběhu výšky -> `[(t0, t1, výška, rozptyl)]` v sekundách.

    Vibrato se nejdřív vyhladí mediánem přes 250 ms; tón končí, když se
    vyhlazená výška vzdálí od mediánu tónu o `tol` půltónu nebo zazní ticho.
    Rozptyl je p90-p10 surové výšky (šířka vibrata).
    """
    if pitch.size == 0:
        return []
    k = int(0.25 / hop) | 1
    sm = medfilt(np.where(np.isnan(pitch), -1000.0, pitch), k)
    sm[np.isnan(pitch) | (sm < 0)] = np.nan
    notes, i, n = [], 0, sm.size
    while i < n:
        if np.isnan(sm[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(sm[j]) and abs(sm[j] - np.median(sm[i:j + 1])) < tol:
            j += 1
        if (j - i) * hop >= min_len:
            core = pitch[i:j]
            core = core[~np.isnan(core)]
            notes.append((i * hop, j * hop, float(np.median(core)),
                          float(np.percentile(core, 90) - np.percentile(core, 10))))
        i = max(j, i + 1)
    return notes


def spectral_octave(x: np.ndarray, sr: int, pitch: float,
                    top_hz: float = 2500.0) -> float | None:
    """Oktáva tónu z řady harmonických; YIN dává jen přesnou výšku v oktávě.

    YIN se u zpěvu plete o oktávu oběma směry (slabý základní tón, široké
    vibrato, tiché zpívání) a u některých souborů VocalSet celých. Skutečný
    základní tón je nejvyšší z `pitch - 12`, `pitch`, `pitch + 12`, jehož
    násobky vysvětlí výrazné špičky spektra (do -35 dB, pod `top_hz`): o
    oktávu níž je vysvětlí taky, o oktávu výš jen sudé. `None` = spektrum
    nemá čitelnou řadu harmonických.
    """
    mid = x[x.size // 5: x.size * 4 // 5]
    if mid.size < 4096:
        return None
    n = 1 << int(math.log2(mid.size))
    mag = np.abs(np.fft.rfft(mid[:n] * np.hanning(n)))
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    mag[(fr < 50.0) | (fr > top_hz)] = 0.0
    top = float(mag.max())
    if top <= 0.0:
        return None
    peaks, m, w = [], mag.copy(), max(1, int(20.0 / (fr[1])))
    for _ in range(10):
        i = int(np.argmax(m))
        if m[i] < top * 10.0 ** (-35.0 / 20.0):
            break
        peaks.append((fr[i], m[i]))
        m[max(0, i - w):i + w] = 0.0
    if len(peaks) < 3:
        return None
    for cand in (pitch + 12.0, pitch, pitch - 12.0):
        f0 = 440.0 * 2.0 ** ((cand - 69) / 12)
        # vibrato roztáhne k-tou harmonickou k-krát víc než základní tón
        weight = sum(a for f, a in peaks if f >= 0.9 * f0 and
                     abs(f / f0 - round(f / f0)) < min(0.2, 0.05 * round(f / f0)))
        if weight >= 0.85 * sum(a for _, a in peaks):
            return cand
    return None


def fix_long_tones(pitches, singer: str) -> list:
    """Dlouhé tóny VocalSet jsou vždy tytéž tři výšky (`LONG_TONES`).

    Tón, který k žádné z nich nesedí (ani po oktávové kontrole spektrem),
    je chyba měření -> `None`.
    """
    want = LONG_TONES.get(singer) or LONG_TONES["female" if _female(singer) else None]
    return [p if min(abs(p - w) for w in want) <= 1.5 else None for p in pitches]


def subharmonic_db(x: np.ndarray, sr: int, pitch: float) -> float:
    """Síla složky na f0/2 a 1.5·f0 proti nejsilnější špičce (dB)."""
    mid = x[x.size // 5: x.size * 4 // 5]
    n = 1 << int(math.log2(max(mid.size, 2)))
    mag = np.abs(np.fft.rfft(mid[:n] * np.hanning(n)))
    df = sr / n
    top = float(mag[int(50.0 / df):int(3000.0 / df)].max(initial=0.0))
    if top <= 0.0:
        return 0.0

    def peak(f):
        return float(mag[int(f * 0.97 / df):int(f * 1.03 / df) + 1].max(initial=0.0))

    f0 = 440.0 * 2.0 ** ((pitch - 69) / 12)
    return 20.0 * math.log10(max(peak(f0 / 2), peak(1.5 * f0), 1e-12) / top)


def voice_quality(x: np.ndarray, sr: int, f0: float | None = None) -> dict:
    """Lokální jitter a shimmer (%) a HNR (dB) znělého úseku.

    Jako „voice report“ Praatu, zjednodušeně: periody se hledají
    porovnáním sousedních period (posun s největší korelací v ±20 % kolem
    odhadu, parabolické zpřesnění), takže filtr kolem F0 kolísání
    nevyhladí. Jitter = průměrný rozdíl sousedních period / průměrná
    perioda, shimmer totéž pro RMS period, HNR z průměrné korelace
    sousedních period r: 10·log10(r / (1 - r)). Lidský modální hlas má
    jitter ~0.2–1 %, shimmer ~1–4 %, HNR ~15–25 dB; syntéza bez
    mikro-variací má jitter i shimmer skoro nula a HNR přes 35 dB.
    """
    x = np.asarray(x, dtype=np.float64)
    if f0 is None:
        pitch, _, _ = track_pitch(x, sr)
        if np.isnan(pitch).all():
            raise ValueError("voice_quality: úsek nemá znělou výšku")
        f0 = 440.0 * 2.0 ** ((float(np.nanmedian(pitch)) - 69.0) / 12.0)
    per = sr / float(f0)
    w = int(round(per))
    lo, hi = int(0.8 * per), int(1.2 * per) + 1
    t = float(np.argmax(np.abs(x[:w + 1])))
    marks, rs, amps = [t], [], []
    while t + hi + w < x.size:
        i = int(t)
        a = x[i:i + w]
        lags = np.arange(lo, hi)
        segs = np.stack([x[i + L:i + L + w] for L in lags])
        num = segs @ a
        den = np.sqrt((segs ** 2).sum(axis=1) * float(a @ a)) + 1e-12
        c = num / den
        j = int(np.argmax(c))
        shift = float(lags[j])
        if 0 < j < c.size - 1:
            d = c[j - 1] - 2.0 * c[j] + c[j + 1]
            if d < 0:
                shift += 0.5 * (c[j - 1] - c[j + 1]) / d
        rs.append(float(c[j]))
        amps.append(float(np.sqrt(np.mean(a ** 2))))
        t += shift
        marks.append(t)
    if len(marks) < 4:
        raise ValueError("voice_quality: úsek je na měření moc krátký")
    periods = np.diff(marks)
    amps = np.asarray(amps)
    r = float(np.clip(np.mean(rs), 1e-6, 1.0 - 1e-6))
    return {"f0": float(f0),
            "jitter": float(np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100.0),
            "shimmer": float(np.mean(np.abs(np.diff(amps))) / np.mean(amps) * 100.0),
            "hnr": float(10.0 * math.log10(r / (1.0 - r)))}


# --------------------------------------------------------------------------
# TD-PSOLA
# --------------------------------------------------------------------------

def _marks(x: np.ndarray, sr: int, f0: np.ndarray) -> np.ndarray:
    """Značky period: vrcholy základní vlny, jedna na periodu."""
    med = float(np.nanmedian(f0))
    b, a = butter(2, min(0.45, 1.5 * med / (sr / 2.0)))
    lp = lfilter(b, a, x)
    n = x.size
    period = sr / f0
    t = int(np.argmax(lp[: int(period[0]) + 1]))
    marks = [t]
    while True:
        p = period[min(t, n - 1)]
        lo, hi = int(t + 0.75 * p), int(t + 1.25 * p)
        if hi >= n:
            break
        t = lo + int(np.argmax(lp[lo:hi]))
        marks.append(t)
    return np.asarray(marks)


def psola(x: np.ndarray, sr: int, pitch: np.ndarray, hop: float,
          semitones: float) -> np.ndarray:
    """Posune výšku mono `x` o `semitones`, délku a formanty nechá.

    `pitch` je průběh výšky v MIDI po `hop` s (z `track_pitch`). Zrna
    dlouhá dvě periody (Hann) se berou u nejbližší značky zdroje a skládají
    se s novou periodou; hlasitost se vrátí na RMS zdroje.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    p = pitch.copy()
    ok = ~np.isnan(p)
    if not ok.any():
        raise ValueError("psola: tón nemá změřenou výšku")
    p[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), p[ok])
    f0 = 440.0 * 2.0 ** ((np.interp(np.arange(n) / sr, np.arange(p.size) * hop, p) - 69) / 12)
    marks = _marks(x, sr, f0)
    ratio = 2.0 ** (semitones / 12.0)
    out = np.zeros(n + 4 * int(sr / f0.min()) + 4)
    t = float(marks[0])
    while t < n:
        k = int(np.clip(np.searchsorted(marks, t), 0, marks.size - 1))
        if k > 0 and abs(marks[k - 1] - t) < abs(marks[k] - t):
            k -= 1
        m = int(marks[k])
        half = int(sr / f0[min(m, n - 1)])
        lo, hi = max(0, m - half), min(n, m + half + 1)
        w0 = lo - (m - half)
        grain = x[lo:hi] * np.hanning(2 * half + 1)[w0:w0 + hi - lo]
        at = int(round(t)) - (m - lo)
        if at >= 0:
            out[at:at + grain.size] += grain
        t += sr / (f0[min(int(t), n - 1)] * ratio)
    out = out[:n]
    src = float(np.sqrt(np.mean(x ** 2)))
    got = float(np.sqrt(np.mean(out ** 2)))
    return out * (src / got if got > 0 else 1.0)


# --------------------------------------------------------------------------
# VocalSet -> tóny
# --------------------------------------------------------------------------

def _load_mono(path: str):
    sr, d = wavfile.read(path)
    x = d.astype(np.float64)
    if d.dtype == np.int16:
        x /= 32768.0
    elif d.dtype == np.int32:
        x /= 2147483648.0
    if x.ndim == 2:
        x = x.mean(axis=1)
    return int(sr), x


def _save(path: str, sr: int, x: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    y = np.clip(x, -1.0, 1.0)
    wavfile.write(path, sr, (y * 32767.0).astype(np.int16))


def _level(x: np.ndarray, db: float) -> np.ndarray:
    """Srovná tón na RMS `db` (měřeno ve středu tónu, bez nasazení a konce)."""
    core = x[x.size // 5: x.size * 4 // 5] if x.size > 10 else x
    rms = float(np.sqrt(np.mean(core ** 2))) if core.size else 0.0
    if rms <= 0.0:
        return x
    y = x * (10.0 ** (db / 20.0) / rms)
    peak = float(np.abs(y).max())
    return y * (0.95 / peak) if peak > 0.95 else y


def _vowel(fn: str) -> str:
    return os.path.splitext(fn)[0].rsplit("_", 1)[-1].lower()


def all_singers() -> list:
    return sorted({s for v in VOICES.values() for s in v.singers})


def extract_notes(full_dir: str, out_dir: str, singers=None, log=print,
                  keep=()) -> list:
    """Projde `FULL/<zpěvák>/<TAKES>` a uloží stabilní tóny do `out_dir`.

    Vrací katalog `VoiceNote` (a zapíše ho do `out_dir/catalog.json`), včetně
    tónů dopočítaných PSOLA pod nejnižší zazpívaný tón každého zpěváka (a u
    zpěvaček nad nejvyšší). `keep` jsou tóny z dřívějšího řezání, které se
    zapíšou do katalogu s novými - tak se doplní jen chybějící zpěváci.
    """
    singers = singers or all_singers()
    notes: list[VoiceNote] = []
    for singer in singers:
        count = 0
        for take, (layer, style) in TAKES.items():
            folder = os.path.join(full_dir, singer, *take.split("/"))
            if not os.path.isdir(folder):
                continue
            for fn in sorted(os.listdir(folder)):
                if not fn.lower().endswith(".wav") or _vowel(fn) not in "aeiou":
                    continue
                sr, x = _load_mono(os.path.join(folder, fn))
                pitch, _, hop = track_pitch(x, sr)
                lo, hi = singer_range(singer)
                top = FEMALE_TOP_HZ if _female(singer) else 2500.0
                found = []
                for t0, t1, p, spread in find_notes(pitch, hop):
                    a = max(0, int((t0 - PRE_ROLL) * sr))
                    b = int((t1 - TAIL_TRIM) * sr)
                    # oktávu určí řada harmonických; nečitelná nebo nejistá
                    # (jiná na výřezu s nasazením) = tón vynechat, je jich dost
                    q = spectral_octave(x[int(t0 * sr):int(t1 * sr)], sr, p, top)
                    if q is None or spectral_octave(x[a:b], sr, q, top) != q:
                        continue
                    found.append((a, b, q, spread))
                if take.startswith("long_tones"):
                    fixed = fix_long_tones([f[2] for f in found], singer)
                else:
                    fixed = [f[2] for f in found]
                for i, ((a, b, _, spread), p) in enumerate(zip(found, fixed)):
                    if spread > MAX_SPREAD or p is None or not lo <= p <= hi:
                        continue
                    y = _level(x[a:b], NOTE_RMS_DB)
                    base = f"{os.path.splitext(fn)[0]}_{i}"
                    rel = f"{singer}/{base}.wav"
                    _save(os.path.join(out_dir, *rel.split("/")), sr, y)
                    notes.append(VoiceNote(rel, singer, _vowel(fn), layer, style,
                                           round(p, 3), round(spread, 3),
                                           round((b - a) / sr, 3), f"{take}/{fn}"))
                    count += 1
        made = _deepen(notes, singer, out_dir)
        if _female(singer):
            made += _deepen(notes, singer, out_dir, PSOLA_UP_STEPS)
        notes += made
        log(f"  {singer}: {count} tónů")
    _write_catalog(out_dir, list(keep) + notes)
    return list(keep) + notes


def _deepen(notes: list, singer: str, out_dir: str, steps=PSOLA_STEPS) -> list:
    """PSOLA pod nejnižší dlouhý tón zpěváka, pro každou samohlásku a vrstvu.

    Kladné `steps` jdou naopak nad nejvyšší dlouhý tón (soprán nad F5).
    """
    made = []
    groups: dict = {}
    up = steps[0] > 0
    sign = -1.0 if up else 1.0
    for v in notes:
        if v.singer == singer and not v.psola and v.dur >= 1.0:
            key = (v.vowel, v.layer, v.style)
            if key not in groups or sign * v.pitch < sign * groups[key].pitch - 0.5 or (
                    abs(v.pitch - groups[key].pitch) <= 0.5 and v.dur > groups[key].dur):
                groups[key] = v
    for edge in groups.values():
        sr, x = _load_mono(os.path.join(out_dir, *edge.file.split("/")))
        pitch, _, hop = track_pitch(x, sr)
        if np.isnan(pitch).all():
            continue
        for step in steps:
            y = _level(psola(x, sr, pitch, hop, step), NOTE_RMS_DB)
            if subharmonic_db(y, sr, edge.pitch + step) > PSOLA_MAX_SUB_DB:
                continue                  # zrna se nesložila čistě: drnčí
            rel = edge.file[:-4] + (f"_psolaup{step}.wav" if up else f"_psola{-step}.wav")
            _save(os.path.join(out_dir, *rel.split("/")), sr, y)
            made.append(VoiceNote(rel, singer, edge.vowel, edge.layer, edge.style,
                                  round(edge.pitch + step, 3), edge.spread, edge.dur,
                                  edge.source, psola=step))
    return made


def _write_catalog(out_dir: str, notes: list) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, CATALOG), "w", encoding="utf-8") as f:
        json.dump([asdict(v) for v in notes], f, indent=1, ensure_ascii=False)


def read_catalog(out_dir: str) -> list:
    with open(os.path.join(out_dir, CATALOG), encoding="utf-8") as f:
        return [VoiceNote(**d) for d in json.load(f)]


# --------------------------------------------------------------------------
# katalog -> SFZ
# --------------------------------------------------------------------------

def _score(v: VoiceNote) -> float:
    """Čím vyšší, tím lepší vzorek: dlouhý, klidný, zazpívaný (ne PSOLA)."""
    return (min(v.dur, 4.0) - 1.5 * v.spread - 1.5 * abs(v.pitch - round(v.pitch))
            - (3.0 + 0.3 * abs(v.psola) if v.psola else 0.0)
            + (0.5 if "long_tones" in v.source else 0.0))


def plan_voice(name: str, vowel: str, notes: list) -> list[dict]:
    """Regiony nástroje `name` pro samohlásku: vrstvy p/f, klávesy, sbor."""
    vd = VOICES[name]
    styles = {vd.style, "any"}
    order = {s: i for i, s in enumerate(vd.singers)}
    regions = []
    layers = [(1, 63, "p"), (64, 127, "f")]
    for lovel, hivel, layer in layers:
        mine = [v for v in notes if v.singer in order and v.vowel == vowel
                and v.layer == layer and v.style in styles]
        if not mine:                       # chybí vrstva: vezmi tu druhou
            mine = [v for v in notes if v.singer in order and v.vowel == vowel
                    and v.style in styles]
        by_root: dict = {}
        for v in mine:
            by_root.setdefault(int(round(v.pitch)), {}).setdefault(v.singer, []).append(v)
        roots = sorted(by_root)
        for k, root in enumerate(roots):
            lokey = roots[0] - 5 if k == 0 else (roots[k - 1] + root) // 2 + 1
            hikey = roots[-1] + 4 if k == len(roots) - 1 else (root + roots[k + 1]) // 2
            best = sorted((max(vs, key=_score) for vs in by_root[root].values()),
                          key=_score, reverse=True)
            if vd.stack == 1:
                takes = sorted(by_root[root][best[0].singer], key=_score, reverse=True)[:2]
                chosen = [(t, 0.0) for t in takes]
                seq = len(chosen)
            else:
                picked = sorted(best[: vd.stack], key=lambda v: order[v.singer])
                span = len(picked) - 1
                chosen = [(v, (2.0 * i / span - 1.0) * vd.spread * 100.0 if span else 0.0)
                          for i, v in enumerate(picked)]
                seq = 1
            vol = (P_LAYER_DB if layer == "p" else 0.0) - (
                10.0 * math.log10(len(chosen)) if seq == 1 else 0.0)
            for pos, (v, pan) in enumerate(chosen, start=1):
                region = {"sample": v.file, "lokey": max(0, lokey), "hikey": min(127, hikey),
                          "pitch_keycenter": root,
                          "tune": round((root - v.pitch) * 100.0, 1),
                          "lovel": lovel, "hivel": hivel, "volume": round(vol, 2),
                          "pan": round(pan, 1)}
                if seq > 1:
                    region.update(seq_length=seq, seq_position=pos)
                regions.append(region)
    return regions


def write_voice_sfz(name: str, art: str, notes_dir: str, sfz_dir: str,
                    notes: list, release: float = 0.35) -> str:
    """Zapíše `<sfz_dir>/<name>.<art>.sfz`; smyčky uprostřed každého tónu."""
    regions = plan_voice(name, VOWELS[art], notes)
    if not regions:
        raise ValueError(f"pro {name}.{art} nejsou v katalogu žádné tóny")
    rel_notes = os.path.relpath(notes_dir, sfz_dir).replace("\\", "/") + "/"
    lines = ["// vygenerováno Paprikou z VocalSet (CC BY 4.0) — neupravuj, přepíše se",
             f"<control> default_path={rel_notes}",
             f"<global> amp_veltrack=40 ampeg_attack=0.05 ampeg_release={release:g}"]
    for r in regions:
        sr, data = wavfile.read(os.path.join(notes_dir, *r["sample"].split("/")), mmap=True)
        n = data.shape[0]
        r["loop_mode"] = "loop_continuous"
        r["loop_start"] = max(int(0.35 * n), int((PRE_ROLL + 0.1) * sr))
        r["loop_end"] = int(0.92 * n)
        del data
        lines.append("<region> " + " ".join(f"{k}={v}" for k, v in r.items()))
    path = os.path.join(sfz_dir, f"{name}.{art}.sfz")
    os.makedirs(sfz_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def build(root: str, log=print, reuse: bool = True) -> list[str]:
    """`<root>/vocalset/FULL` -> tóny v `vocalset/notes`, SFZ v `vocalset/sfz`.

    Když katalog tónů už existuje a `reuse`, jen přegeneruje SFZ.
    """
    base = os.path.join(root, "vocalset")
    notes_dir = os.path.join(base, "notes")
    if reuse and os.path.isfile(os.path.join(notes_dir, CATALOG)):
        notes = read_catalog(notes_dir)
        have = {v.singer for v in notes}
        missing = [s for s in all_singers() if s not in have]
        if missing:
            log(f"voices: doplňuji zpěváky {', '.join(missing)} ...")
            notes = extract_notes(os.path.join(base, "FULL"), notes_dir, missing,
                                  log=log, keep=notes)
    else:
        log("voices: řežu nahrávky VocalSet na tóny (pár minut) ...")
        notes = extract_notes(os.path.join(base, "FULL"), notes_dir, log=log)
    paths = []
    for name in VOICES:
        for art in VOWELS:
            paths.append(write_voice_sfz(name, art, notes_dir,
                                         os.path.join(base, "sfz"), notes))
    return paths
