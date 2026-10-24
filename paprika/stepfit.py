"""Krok z nahrávky -> model povrchu (F4). Čte audio, píše jen JSON.

Fyzikální povrchy ve `foley.SURFACES` jsou odhad z literatury. Tady se
z nahrávek skutečné chůze odhadne povrch, který zní jako ony:

1. **Segmentace** (`segment`): dlouhá nahrávka -> jednotlivé kroky.
   Obálka nad 150 Hz (dupání do mikrofonu a hluk pod tím by kroky slily)
   v 5ms oknech, vrcholy aspoň `ONSET_DB` nad šumem a `MIN_GAP` od sebe,
   nástup = první vzorek nad 10 % vrcholu, konec před dalším krokem nebo
   po návratu k šumu.
2. **Rezonance** (`modes`): matrix pencil (Hua & Sarkar 1990) na doznění
   po vrcholu -> `(f, τ, amp)` tlumených sinusů. Rezonance, které se
   opakují aspoň ve 40 % kroků (±4 %), jsou rezonance podložky; nejnižší
   výrazná pod 160 Hz je dupnutí. Zbytek (šum, zrna) modelem nejsou.
3. **Analýza syntézou** (`fit`): povrch `base` dostane naměřené
   rezonance, dupnutí, klik a pásmo úlomků, pak se `FIT_ITER`× vyrenderují
   referenční kroky a zisky vrstev se posunou podle podílu energie v pásmech
   (dupnutí <- sub+bass, rezonance <- 250 Hz–2 kHz, klik, tření a úlomky
   <- nad 2 kHz).
4. **Přenos obálky:** zbylý rozdíl podílů výkonu ve 24 mel pásmech (tlumeně,
   ±`EQ_MAX` dB, `EQ_ITER` kol) se uloží jako `Surface.eq` — tím se přenese i barva
   mikrofonu a místnosti nahrávky.

Výsledek jde do `<vzorky>/steps/surfaces.json` a `steps(surface="jméno")`
ho použije jako vestavěný povrch. Zisk se kalibruje na cíl povrchu `base`
(absolutní hlasitost nahrávky neznáme — záleží na mikrofonu).
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
import os

import numpy as np
from scipy import signal

from . import dsp, foley, gait
from . import metrics as _metrics

__all__ = ["segment", "modes", "describe", "fit", "save", "load_file", "cut",
           "from_files", "write_cut", "check_name",
           "noise_floor", "band_error", "shape_error",
           "compare_lines", "fit_lines", "status_lines", "FitError"]

MIN_GAP = 0.25        # s: dva kroky nejsou blíž
MAX_LEN = 0.8         # s: nejdelší okno kroku
ONSET_DB = 12.0       # dB: vrchol kroku nad šumem nahrávky (20. percentil obálky)
MERGE_DB = 10.0       # dB: o tolik slabší vrchol do MAX_LEN po kroku je jeho odval
HP_HZ = 150.0         # Hz: segmentace jen nad tímhle
HOP = 0.005           # s: krok obálky
PRE = 0.005           # s: okno začíná před nástupem
MAX_MODES = 12        # kolik rezonancí vrátí `modes`
SV_DB = 45.0          # dB: singulární čísla pod maximem - tohle jsou šum
PENCIL_SEC = 0.09     # s: délka doznění pro matrix pencil
PENCIL_DEC = 2        # decimace před matrix pencil (do 11 kHz při 44.1 kHz)
TONAL_TAU = 0.004     # s: kratší doznění je šum/klik, ne rezonance
CLUSTER = 0.04        # ±4 %: stejná rezonance v různých krocích
SUPPORT = 0.4         # rezonance podložky je aspoň ve 40 % kroků
FIT_MODES = 6         # kolik rezonancí si povrch nechá
MODE_FLOOR = 0.1      # rezonance slabší než −20 dB pod nejsilnější jsou zbytky kliku a šumu
FIT_ITER = 4          # kola analýzy syntézou
FIT_STEPS = 12        # referenčních kroků na kolo
EQ_MAX = 12.0         # dB: mez přenosu obálky
EQ_ITER = 5           # kola zpřesnění přenosu obálky
EQ_STEP = 0.7         # podíl rozdílu, který jedno kolo opraví
MIN_STEPS = 4         # méně kroků na fit nestačí
MAX_FIT_STEPS = 32    # rezonance se odhadnou nejvýš z tolika kroků (rovnoměrně vybraných)
BAND_FLOOR = 0.1      # %: podíl pásma pod tím se v chybě počítá jako tohle (−30 dB)
LOUD_DB = 30.0        # dB: tvar spektra a přenos obálky jen v pásmech do tolika pod maximem


class FitError(ValueError):
    """Nahrávky na odhad povrchu nestačí (málo kroků, ticho, ...)."""


# --------------------------------------------------------------------------
# nahrávka -> kroky
# --------------------------------------------------------------------------

def _mono(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a.mean(axis=1) if a.ndim == 2 else a


def load_file(path: str) -> tuple[np.ndarray, int]:
    """WAV/MP3 -> (mono float64, sr). MP3 potřebuje `imageio-ffmpeg`."""
    x, sr = _metrics.load_audio(path)
    return _mono(x), int(sr)


def segment(x, sr: int, min_gap: float = MIN_GAP, max_len: float = MAX_LEN
            ) -> list[tuple[int, int]]:
    """Kroky v nahrávce jako `(začátek, konec)` ve vzorcích."""
    m = _mono(x)
    hp = dsp.filt(m, "hp", HP_HZ, sr).astype(np.float64)
    hop = max(1, int(HOP * sr))
    n = hp.size // hop
    if n < 8:
        return []
    env = np.sqrt(np.mean(hp[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-12
    db = 20.0 * np.log10(env)
    floor = float(np.percentile(db, 20))
    peaks, _ = signal.find_peaks(db, height=floor + ONSET_DB,
                                 distance=max(1, int(min_gap / HOP)),
                                 prominence=ONSET_DB * 0.75)
    # slabší vrchol krátce po silném je odval špičky téhož kroku, ne další krok
    kept: list[int] = []
    for p in peaks:
        if (kept and (p - kept[-1]) * HOP < max_len
                and db[p] < db[kept[-1]] - MERGE_DB):
            continue
        kept.append(int(p))
    peaks = np.asarray(kept, dtype=int)
    out: list[tuple[int, int]] = []
    starts = []
    for p in peaks:
        lo = max(0, p - int(0.06 / HOP))
        seg = hp[lo * hop:(p + 1) * hop]
        thr = 0.1 * float(np.abs(seg).max())
        first = int(np.argmax(np.abs(seg) >= thr))
        starts.append(lo * hop + first)
    pre = int(PRE * sr)
    for i, (st, p) in enumerate(zip(starts, peaks)):
        end = min(m.size, st + int(max_len * sr))
        if i + 1 < len(starts):
            end = min(end, starts[i + 1] - pre)
        # konec za posledním místem nad šumem (odval špičky patří ke kroku)
        loud = np.where(db[p:max(p + 1, end // hop)] > floor + 6.0)[0]
        if loud.size:
            end = min(end, (p + int(loud[-1]) + 1) * hop + int(0.03 * sr))
        if end - st > int(0.03 * sr):
            out.append((max(0, st - pre), end))
    return out


def cut(x, sr: int) -> list[np.ndarray]:
    """Kroky z nahrávky jako samostatná okna (mono)."""
    m = _mono(x)
    return [m[a:b] for a, b in segment(m, sr)]


def from_files(paths) -> tuple[list[np.ndarray], int, list]:
    """Kroky ze souborů: (okna, sr, šum nahrávky po krocích). Všechny
    soubory musí mít stejnou vzorkovací frekvenci."""
    segs: list = []
    noises: list = []
    rate = None
    for path in paths:
        x, sr = load_file(path)
        if rate is not None and sr != rate:
            raise FitError(f"{os.path.basename(path)} má {sr} Hz, ostatní {rate} Hz — "
                           f"převzorkuj nahrávky na stejnou frekvenci")
        rate = sr
        found = segment(x, sr)
        nz = noise_floor(x, sr, found)
        segs += [x[a:b] for a, b in found]
        noises += [nz] * len(found)
    if rate is None:
        raise FitError("žádné nahrávky")
    return segs, rate, noises


def write_cut(path: str, out_dir: str) -> list[str]:
    """Rozřeže nahrávku na kroky `step_001.wav`, ... (16 bit) pro poslech nebo
    `samples human import --kind steps`."""
    from .render import write_wav
    x, sr = load_file(path)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out = []
    for i, seg in enumerate(cut(x, sr), 1):
        p = os.path.join(out_dir, f"{base}_step_{i:03d}.wav")
        write_wav(p, np.stack([seg, seg], axis=1).astype(np.float32), sr)
        out.append(p)
    return out


# --------------------------------------------------------------------------
# rezonance (matrix pencil)
# --------------------------------------------------------------------------

def modes(seg, sr: int, n_modes: int = MAX_MODES, fmin: float = 40.0,
          fmax: float = 12000.0) -> list[tuple[float, float, float]]:
    """Tlumené sinusy v doznění kroku: `[(Hz, τ s, amp), ...]` seřazené podle
    energie (amp² τ), amp vůči špičce kroku. Matrix pencil na `PENCIL_SEC`
    po vrcholu, decimované `PENCIL_DEC`×."""
    m = _mono(seg)
    if m.size < 64:
        return []
    peak = float(np.abs(m).max())
    if peak <= 0:
        return []
    start = int(np.argmax(np.abs(m))) + int(0.001 * sr)
    x = m[start:start + int(PENCIL_SEC * sr)]
    fs = float(sr)
    if PENCIL_DEC > 1 and x.size > 64 * PENCIL_DEC:
        x = signal.decimate(x, PENCIL_DEC, zero_phase=True)
        fs = sr / PENCIL_DEC
    n = x.size
    if n < 64:
        return []
    L = n // 3
    Y = np.lib.stride_tricks.sliding_window_view(x, L + 1)
    _, sv, vh = np.linalg.svd(Y, full_matrices=False)
    order = int(np.sum(sv > sv[0] * 10.0 ** (-SV_DB / 20.0)))
    order = int(np.clip(order, 2, 2 * n_modes + 12))
    v = vh[:order].conj().T
    z = np.linalg.eigvals(np.linalg.pinv(v[:-1]) @ v[1:])
    z = z[(np.abs(z) > 1e-6) & (np.abs(z) < 1.0)]
    if z.size == 0:
        return []
    van = z[None, :] ** np.arange(n)[:, None]
    h = np.linalg.lstsq(van, x.astype(np.complex128), rcond=None)[0]
    out = []
    for zi, hi in zip(z, h):
        f = float(np.angle(zi)) * fs / (2.0 * math.pi)
        if f <= 0 or not fmin <= f <= min(fmax, 0.45 * fs):
            continue
        tau = -1.0 / (fs * math.log(abs(zi)))
        if not 0.0005 <= tau <= 3.0:
            continue
        out.append((f, tau, float(2.0 * abs(hi) / peak)))
    out.sort(key=lambda m_: -(m_[2] ** 2) * m_[1])
    return out[:n_modes]


def _clusters(per_step: list[list], n_steps: int) -> list[tuple[float, float, float, int]]:
    """Rezonance napříč kroky -> `(f, τ, amp, podpora)` těch, které se
    opakují (±`CLUSTER`) aspoň v `SUPPORT` kroků."""
    items = sorted((f, tau, a, k) for k, ms in enumerate(per_step) for f, tau, a in ms)
    groups: list[list] = []
    for it in items:
        if groups and it[0] <= np.median([g[0] for g in groups[-1]]) * (1 + 2 * CLUSTER):
            groups[-1].append(it)
        else:
            groups.append([it])
    need = max(2, int(math.ceil(SUPPORT * n_steps)))
    out = []
    for g in groups:
        support = len({it[3] for it in g})
        if support >= need:
            out.append((float(np.median([it[0] for it in g])),
                        float(np.median([it[1] for it in g])),
                        float(np.median([it[2] for it in g])), support))
    return out


# --------------------------------------------------------------------------
# popis a fit
# --------------------------------------------------------------------------

def _ring(m: np.ndarray, sr: int, db: float = -30.0) -> float:
    w = max(1, int(0.005 * sr))
    e = np.convolve(np.abs(m), np.ones(w) / w, "same")
    idx = np.where(e > e.max() * 10 ** (db / 20.0))[0]
    return float((idx[-1] - idx[0]) / sr) if idx.size else 0.0


def _edges7(sr: int) -> np.ndarray:
    hi = [b[2] for b in _metrics.BANDS7]
    return np.array([0.0] + [h if h is not None else sr / 2.0 + 1.0 for h in hi])


def _flat_window(n: int, sr: int, edge: float = 0.002) -> np.ndarray:
    w = np.ones(n)
    k = min(n // 2, max(1, int(edge * sr)))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, k))
    w[:k] *= ramp
    w[n - k:] *= ramp[::-1]
    return w


def _energy(m: np.ndarray, sr: int, edges: np.ndarray) -> tuple[np.ndarray, float]:
    """Energie v pásmech `edges` (Parseval: nezávisí na délce FFT) a Σw² okna
    — podle něj se odečítá šum. Okno je ploché s 2ms náběhy: krok je
    přechodový jev a Hannovo okno by potlačilo právě klik paty na začátku
    a odval špičky na konci."""
    w = _flat_window(m.size, sr)
    size = 1 << int(np.ceil(np.log2(max(256, m.size))))
    p = np.abs(np.fft.rfft(m * w, size)) ** 2 * (2.0 / size)
    f = np.fft.rfftfreq(size, 1.0 / sr)
    e = np.array([p[(f >= a) & (f < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
    return e, float(np.sum(w * w))


def noise_floor(x, sr: int, segments) -> dict | None:
    """Šum nahrávky mimo kroky: energie na Σw² v mel pásmech a v 7 pásmech.
    `None`, když mezi kroky není aspoň 0.2 s ticha."""
    m = _mono(x)
    mask = np.ones(m.size, dtype=bool)
    guard = int(0.02 * sr)
    for a, b in segments:
        mask[max(0, a - guard):b + guard] = False
    # jen opravdu tiché úseky: obálka nad 150 Hz do 3 dB nad 20. percentilem
    hp = dsp.filt(m, "hp", HP_HZ, sr).astype(np.float64)
    hop = max(1, int(HOP * sr))
    n = hp.size // hop
    if n >= 8:
        db = 20.0 * np.log10(np.sqrt(np.mean(hp[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-12)
        noisy = np.repeat(db > float(np.percentile(db, 20)) + 3.0, hop)
        mask[:noisy.size] &= ~noisy
    idx = np.flatnonzero(mask)
    if idx.size < int(0.2 * sr):
        return None
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    chunk = 4096
    mel, b7, n = 0.0, 0.0, 0
    for r in runs:
        for i in range(0, r.size - chunk + 1, chunk):
            z = m[r[i]:r[i] + chunk]
            e1, w2 = _energy(z, sr, _metrics._mel_edges(sr, _metrics.STEP_BANDS))
            e2, _ = _energy(z, sr, _edges7(sr))
            mel, b7, n = mel + e1 / w2, b7 + e2 / w2, n + 1
    if n == 0:
        return None
    return {"mel": mel / n, "b7": b7 / n}


def _bands(m: np.ndarray, sr: int, edges: np.ndarray, noise) -> np.ndarray:
    e, w2 = _energy(m, sr, edges)
    if noise is not None:
        e = np.maximum(e - noise * w2, 0.01 * e)   # odečtený šum, zbytek ≥ 1 %
    return e


def describe(seg, sr: int, noise: dict | None = None) -> dict:
    """Popis kroku: podíly 7 pásem (%), podíly mel pásem (dB, bez odečteného
    průměru — korekce jednoho pásma neposouvá ostatní), těžiště, doznění.
    `noise` (z `noise_floor`) se odečte od energie v pásmech."""
    m = _mono(seg)
    mel = _bands(m, sr, _metrics._mel_edges(sr, _metrics.STEP_BANDS),
                 None if noise is None else noise["mel"])
    b7 = _bands(m, sr, _edges7(sr), None if noise is None else noise["b7"])
    names = [b[0] for b in _metrics.BANDS7]
    total7 = float(b7.sum()) + 1e-30
    centers = _metrics.mel_centers(sr)
    return {"bands": {k: float(100.0 * v / total7) for k, v in zip(names, b7)},
            "shape": 10.0 * np.log10(mel / (mel.sum() + 1e-30) + 1e-12),
            "centroid": float((mel * centers).sum() / (mel.sum() + 1e-30)),
            "ring": _ring(m, sr)}


def _median_desc(descs: list[dict]) -> dict:
    names = [b[0] for b in _metrics.BANDS7]
    return {"bands": {k: float(np.median([d["bands"][k] for d in descs])) for k in names},
            "shape": np.median([d["shape"] for d in descs], axis=0),
            "centroid": float(np.median([d["centroid"] for d in descs])),
            "ring": float(np.median([d["ring"] for d in descs]))}


def _synth(surf: foley.Surface, sr: int, shoe: str, k: int = FIT_STEPS) -> list[np.ndarray]:
    """Referenční kroky (75 kg, 1.4 m/s) z povrchu bez zisku, pevné varianty."""
    out = []
    for v in range(k):
        rng = np.random.default_rng(90_001 + v)
        y, _ = foley._render(sr, surf, v % 2, gait.REF_KG, gait.REF_SPEED,
                             1.0, 1.0, 1.0, 1.0, 1.0, shoe, None, rng)
        out.append(y)
    return out


def _group(bands: dict) -> tuple[float, float, float]:
    low = bands["sub"] + bands["bass"]
    mid = bands["lowmid"] + bands["mid"]
    high = bands["himid"] + bands["pres"] + bands["air"]
    return low, mid, high


def band_error(a: dict, b: dict) -> float:
    """RMS rozdíl podílů 7 pásem v dB; podíl pod `BAND_FLOOR` % (−30 dB) se
    počítá jako `BAND_FLOOR` — skoro prázdná pásma neslyšíš."""
    d = [10.0 * math.log10(max(a[k], BAND_FLOOR) / max(b[k], BAND_FLOOR)) for k in a]
    return float(np.sqrt(np.mean(np.square(d))))


def shape_error(real: np.ndarray, syn: np.ndarray) -> float:
    """RMS rozdíl podílů mel pásem v dB jen tam, kde nahrávka nebo syntéza má
    energii (do `LOUD_DB` pod maximem)."""
    real, syn = np.asarray(real), np.asarray(syn)
    loud = (real > real.max() - LOUD_DB) | (syn > syn.max() - LOUD_DB)
    return float(np.sqrt(np.mean((real[loud] - syn[loud]) ** 2)))


def _scale(surf: foley.Surface, body: float, modes_k: float, high: float) -> foley.Surface:
    deb = surf.debris
    if deb is not None:
        deb = deb[:4] + (deb[4] * high,) + deb[5:]
    fr = surf.friction
    return dataclasses.replace(
        surf,
        body=(surf.body[0], surf.body[1], surf.body[2] * body),
        modes=tuple((f, t, a * modes_k) for f, t, a in surf.modes),
        click=(surf.click[0], surf.click[1] * high), debris=deb,
        friction=(fr[0], fr[1], fr[2] * high))


def fit(segments, sr: int, base: str = "stone", shoe: str = "boot",
        noise=None) -> dict:
    """Povrch, který zní jako kroky `segments` (okna z `segment`/`cut`).

    Vrátí záznam pro `save`: `surface` (dict), `gain`, `base`, `steps`,
    `bands` (real / base / fit, %), `error_db` (base / fit), `modes`
    (nalezené rezonance s podporou) a `ring` (real / fit, s). `noise`
    (z `noise_floor` nahrávky, nebo seznam po krocích, když jsou z více
    nahrávek) se odečte od spektra kroků.
    """
    noises = noise if isinstance(noise, list) else [noise] * len(segments)
    pairs = [(_mono(s), n) for s, n in zip(segments, noises) if np.abs(_mono(s)).max() > 1e-5]
    segs = [p_[0] for p_ in pairs]
    if len(segs) < MIN_STEPS:
        raise FitError(f"na odhad povrchu je potřeba aspoň {MIN_STEPS} kroků, "
                       f"nahrávky jich daly {len(segs)} — delší nahrávka chůze, méně šumu")
    if base not in foley.SURFACES:
        raise FitError(f"základ {base!r} není vestavěný povrch: {', '.join(foley.SURFACES)}")
    base_surf = foley.SURFACES[base]
    real = _median_desc([describe(s, sr, n) for s, n in pairs])

    pick = (np.linspace(0, len(segs) - 1, MAX_FIT_STEPS).round().astype(int)
            if len(segs) > MAX_FIT_STEPS else np.arange(len(segs)))
    per_step = [modes(segs[i], sr) for i in pick]
    found = _clusters([[m_ for m_ in ms if m_[1] >= TONAL_TAU] for ms in per_step], len(pick))
    body = [c for c in found if c[0] < 160.0 and c[1] >= 0.008]
    tonal = [c for c in found if c[0] >= 160.0]
    if tonal:
        loud = max(c[2] for c in tonal)
        tonal = [c for c in tonal if c[2] >= MODE_FLOOR * loud]
    tonal.sort(key=lambda c: -c[3] * c[2] ** 2 * c[1])
    tonal = sorted(tonal[:FIT_MODES])
    # naměřená amplituda = rezonance × spektrum kontaktního pulzu; povrch
    # chce rezonanci samotnou, proto se dělí ziskem pulzu modelu (bota `shoe`)
    tau = foley.contact_ms(shoe, base_surf, gait.REF_SPEED) / 1000.0
    raw = [(f, t, a / max(0.02, float(foley.pulse_gain(f, tau)))) for f, t, a, _ in tonal]
    ref_amp = max([a for _, _, a in base_surf.modes] or [0.5])
    top = max([a for _, _, a in raw] or [1.0])
    new_modes = tuple((round(f, 1), round(t, 4), round(ref_amp * a / top, 4))
                      for f, t, a in raw)
    new_body = base_surf.body
    if body:
        b = max(body, key=lambda c: c[2])
        new_body = (round(b[0], 1), round(min(b[1], 0.15), 4), base_surf.body[2])

    early = []
    for s in segs:
        a = int(PRE * sr)
        w = s[a:a + int(0.01 * sr)]
        if w.size < 32:
            continue
        p = np.abs(np.fft.rfft(w * np.hanning(w.size), 4096)) ** 2
        f = np.fft.rfftfreq(4096, 1.0 / sr)
        sel = (f >= 1500) & (f <= 8000)
        early.append(float(f[sel][np.argmax(p[sel])]))
    click = base_surf.click
    if early:
        click = (round(float(np.median(early)), 1), click[1])
    deb = base_surf.debris
    if deb is not None:
        cents = []
        for s in segs:
            w = dsp.filt(s[int(0.02 * sr):int(0.25 * sr)], "hp", 1000.0, sr).astype(np.float64)
            if w.size > 64 and np.abs(w).max() > 0:
                p = np.abs(np.fft.rfft(w)) ** 2
                f = np.fft.rfftfreq(w.size, 1.0 / sr)
                cents.append(float((p * f).sum() / p.sum()))
        if cents:
            deb = (deb[0], deb[1], (round(float(np.clip(np.median(cents), 1500, 9000)), 1),
                                    deb[2][1])) + deb[3:]
    surf = dataclasses.replace(base_surf, modes=new_modes, body=new_body, click=click,
                               debris=deb, eq=None)
    if not surf.modes and not base_surf.modes:
        surf = dataclasses.replace(surf, modes=())

    base_desc = _median_desc([describe(y, sr) for y in _synth(base_surf, sr, shoe)])
    for _ in range(FIT_ITER):
        syn = _median_desc([describe(y, sr) for y in _synth(surf, sr, shoe)])
        rl, rm, rh = _group(real["bands"])
        sl, sm, sh = _group(syn["bands"])
        k = [float(np.clip((r / max(s_, 1e-3)) ** 0.5, 0.5, 2.0))
             for r, s_ in ((rl, sl), (rm, sm), (rh, sh))]
        surf = _scale(surf, *k)
    eq = np.zeros(len(real["shape"]))
    rs = np.asarray(real["shape"])
    for _ in range(EQ_ITER):
        syn = _median_desc([describe(y, sr) for y in _synth(surf, sr, shoe)])
        # bez vyhlazení přes sousední pásma: rezonance jsou úzké a průměr se
        # sousedy by korekci obrátil; tlumení EQ_STEP drží iteraci stabilní
        ss = np.asarray(syn["shape"])
        loud = (rs > rs.max() - LOUD_DB) | (ss > ss.max() - LOUD_DB)
        eq = np.clip(eq + EQ_STEP * np.where(loud, rs - ss, 0.0), -EQ_MAX, EQ_MAX)
        surf = dataclasses.replace(surf, eq=tuple(round(float(g), 2) for g in eq))
    fit_desc = _median_desc([describe(y, sr) for y in _synth(surf, sr, shoe)])
    target = foley.SURFACE_TARGET[base]
    peaks = [float(np.abs(y).max()) for y in _synth(surf, sr, shoe, 30)]
    gain = target / max(1e-9, float(np.median(peaks)))
    return {
        "surface": _surface_dict(surf),
        "gain": round(gain, 5),
        "base": base, "shoe": shoe, "steps": len(segs), "sr": sr,
        "bands": {"real": real["bands"], "base": base_desc["bands"], "fit": fit_desc["bands"]},
        "error_db": {"base": round(band_error(base_desc["bands"], real["bands"]), 2),
                     "fit": round(band_error(fit_desc["bands"], real["bands"]), 2)},
        "shape_db": {"base": round(shape_error(real["shape"], base_desc["shape"]), 2),
                     "fit": round(shape_error(real["shape"], fit_desc["shape"]), 2)},
        "noise": any(n is not None for n in noises),
        "centroid": {"real": round(real["centroid"]), "base": round(base_desc["centroid"]),
                     "fit": round(fit_desc["centroid"])},
        "ring": {"real": round(real["ring"], 3), "fit": round(fit_desc["ring"], 3)},
        "modes": [{"f": round(f, 1), "tau": round(t, 4), "support": s_}
                  for f, t, _, s_ in found],
    }


def _surface_dict(surf: foley.Surface) -> dict:
    d = dataclasses.asdict(surf)
    return json.loads(json.dumps(d))       # n-tice -> seznamy, čisté floaty


def _sources(paths) -> list[dict]:
    return [{"file": os.path.basename(p), "bytes": os.path.getsize(p)} for p in paths]


def check_name(name: str) -> None:
    """Jméno nafitovaného povrchu: písmena, číslice, _, ne vestavěný povrch."""
    if not name or not name.replace("_", "").isalnum():
        raise FitError(f"jméno povrchu {name!r}: jen písmena, číslice a _")
    if name in foley.SURFACES:
        raise FitError(f"{name!r} je vestavěný povrch — zvol jiné jméno (např. {name}_moje)")


def save(name: str, record: dict, sources=(), license: str | None = None,
         author: str | None = None) -> str:
    """Zapíše nafitovaný povrch do `foley.custom_path()` pod `name`."""
    check_name(name)
    path = foley.custom_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    rec = dict(record)
    rec["sources"] = _sources(sources)
    rec["license"] = license
    rec["author"] = author
    rec["created"] = datetime.date.today().isoformat()
    data[name] = rec
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------
# textový výstup
# --------------------------------------------------------------------------

def _band_rows(bands: dict, rows) -> list[str]:
    names = [b[0] for b in _metrics.BANDS7]
    lines = ["            " + "".join(f"{n:>8}" for n in names)]
    for label in rows:
        lines.append(f"  {label:<10}" + "".join(f"{bands[label][n]:7.1f}%" for n in names))
    return lines


def fit_lines(name: str, rec: dict) -> list[str]:
    """Textový výsledek fitu: pásma nahrávky / základu / fitu a chyby."""
    s = rec["surface"]
    lines = [f"POVRCH   {name}  (základ {rec['base']}, bota {rec['shoe']}, "
             f"{rec['steps']} kroků)"]
    lines += _band_rows(rec["bands"], ("real", "base", "fit"))
    lines.append(f"  chyba pásem   základ {rec['error_db']['base']:.1f} dB -> fit "
                 f"{rec['error_db']['fit']:.1f} dB   tvar spektra "
                 f"{rec['shape_db']['base']:.1f} -> {rec['shape_db']['fit']:.1f} dB")
    lines.append(f"  těžiště  real {rec['centroid']['real']} Hz   základ "
                 f"{rec['centroid']['base']} Hz   fit {rec['centroid']['fit']} Hz   "
                 f"doznění real {rec['ring']['real']:.2f} s / fit {rec['ring']['fit']:.2f} s")
    modes_txt = ", ".join(f"{m[0]:.0f} Hz/{1000 * m[1]:.0f} ms" for m in s["modes"]) or "žádné"
    lines.append(f"  rezonance     {modes_txt}")
    lines.append(f"  dupnutí       {s['body'][0]:.0f} Hz/{1000 * s['body'][1]:.0f} ms   "
                 f"klik {s['click'][0]:.0f} Hz")
    if s.get("eq"):
        eq = s["eq"]
        lines.append(f"  přenos obálky {min(eq):+.1f} .. {max(eq):+.1f} dB "
                     f"(24 pásem, ±{EQ_MAX:.0f} dB)")
    if rec["error_db"]["fit"] > rec["error_db"]["base"]:
        lines.append("  POZOR   fit je dál od nahrávky než základ — zkus jiný --base")
    return lines


def compare_lines(segments, sr: int, surface: str, shoe: str = "boot",
                  noise=None) -> list[str]:
    """Nahrávky vs povrch (vestavěný nebo nafitovaný) bez ukládání. `noise`
    jako u `fit`."""
    segs = [_mono(s) for s in segments]
    if not segs:
        raise FitError("v nahrávkách nejsou kroky")
    noises = noise if isinstance(noise, list) else [noise] * len(segs)
    real = _median_desc([describe(s, sr, n) for s, n in zip(segs, noises)])
    syn = _median_desc([describe(y, sr) for y in _synth(foley.lookup(surface), sr, shoe)])
    bands = {"real": real["bands"], surface: syn["bands"]}
    lines = [f"SROVNÁNÍ {len(segs)} kroků z nahrávky vs povrch {surface} (bota {shoe})"]
    lines += _band_rows(bands, ("real", surface))
    shape = shape_error(real["shape"], syn["shape"])
    lines.append(f"  chyba pásem {band_error(syn['bands'], real['bands']):.1f} dB   tvar "
                 f"spektra {shape:.1f} dB   těžiště real {real['centroid']:.0f} Hz / "
                 f"{surface} {syn['centroid']:.0f} Hz   doznění real {real['ring']:.2f} s / "
                 f"{surface} {syn['ring']:.2f} s")
    return lines


def status_lines() -> list[str]:
    lines = [f"vestavěné povrchy: {', '.join(foley.SURFACES)}"]
    custom = [n for n in foley.surface_names() if n not in foley.SURFACES]
    if not custom:
        lines.append(f"nafitované: žádné ({foley.custom_path()})")
        lines.append("  python -m paprika samples steps fit <wav...> --name <jméno> --base <povrch>")
        return lines
    lines.append(f"nafitované ({foley.custom_path()}):")
    data = foley._custom()
    for n in custom:
        r = data[n]
        lines.append(f"  {n:<16} základ {r['base']:<7} {r['steps']:>3} kroků   chyba "
                     f"{r['error_db']['base']:.1f} -> {r['error_db']['fit']:.1f} dB   "
                     f"licence {r.get('license') or '?'}")
    return lines
