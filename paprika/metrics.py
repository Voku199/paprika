"""Mix metriky hotového audia: pásma, stereo šířka, dynamika.

Čte jen buffer (nebo soubor), o skladbě neví nic. Používá ho report
(`analyze.py`) i `python -m paprika compare`, kterým AI porovná svůj render
s referenční nahrávkou — neslyší, ale vidí, o kolik má víc subu nebo užší
stereo než reference.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

__all__ = ["BANDS7", "TARGETS", "band_shares", "stereo_width", "correlation",
           "crest_db", "loudness_range", "centroid", "summary", "load_audio",
           "format_table", "STEP_WINDOW", "STEP_BANDS", "events", "step_stats"]

BANDS7 = (("sub", 0.0, 60.0), ("bass", 60.0, 250.0), ("lowmid", 250.0, 500.0),
          ("mid", 500.0, 2000.0), ("himid", 2000.0, 4000.0),
          ("pres", 4000.0, 8000.0), ("air", 8000.0, None))

# cílové rozsahy pro cinematic mix, odvozené z referencí (NieR, Requiem)
TARGETS = {"sub": (0.0, 12.0), "mids": (20.0, 100.0), "width": (10.0, 60.0)}

FRAME = 4096


def _mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float64)
    return a.mean(axis=1) if a.ndim == 2 else a


def _power_spectrum(audio: np.ndarray, sr: int):
    """Průměrné výkonové spektrum přes okna (max ~400 oken, ať je to rychlé)."""
    x = _mono(audio)
    if x.size < FRAME:
        x = np.pad(x, (0, FRAME - x.size))
    starts = np.arange(0, x.size - FRAME + 1, FRAME // 2)
    if starts.size > 400:
        starts = starts[np.linspace(0, starts.size - 1, 400).astype(int)]
    win = np.hanning(FRAME)
    frames = x[starts[:, None] + np.arange(FRAME)[None, :]] * win
    spec = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0)
    return spec, np.fft.rfftfreq(FRAME, 1.0 / sr)


def band_shares(audio: np.ndarray, sr: int) -> dict:
    """Podíl výkonu v sedmi pásmech v procentech (součet 100, u ticha nuly)."""
    spec, f = _power_spectrum(audio, sr)
    total = float(spec.sum())
    out = {}
    for name, lo, hi in BANDS7:
        mask = (f >= lo) & (f < hi) if hi is not None else (f >= lo)
        out[name] = float(spec[mask].sum() / total * 100.0) if total > 0 else 0.0
    return out


def stereo_width(audio: np.ndarray) -> float:
    """Podíl energie side signálu v % (0 = mono, 50 = nezávislé kanály)."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        return 0.0
    mid = (a[:, 0] + a[:, 1]) * 0.5
    side = (a[:, 0] - a[:, 1]) * 0.5
    e_m, e_s = float(np.sum(mid ** 2)), float(np.sum(side ** 2))
    return e_s / (e_m + e_s) * 100.0 if e_m + e_s > 0 else 0.0


def correlation(audio: np.ndarray) -> float | None:
    """Korelace L/R; None pro mono nebo ticho."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] < 2:
        return None
    denom = float(np.sqrt(np.sum(a[:, 0] ** 2) * np.sum(a[:, 1] ** 2)))
    return float(np.sum(a[:, 0] * a[:, 1]) / denom) if denom > 1e-12 else None


def crest_db(audio: np.ndarray) -> float:
    """Peak / RMS v dB — nízké číslo = překomprimovaný, „plochý“ mix."""
    x = _mono(audio)
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    return float(20 * np.log10(np.abs(x).max() / rms)) if rms > 1e-9 else 0.0


def loudness_range(audio: np.ndarray, sr: int) -> float:
    """Rozdíl mezi hlasitými a tichými místy (95. - 10. percentil 400ms RMS, dB)."""
    x = _mono(audio)
    blk = int(0.4 * sr)
    if x.size < 2 * blk:
        return 0.0
    n = x.size // blk
    rms = np.sqrt(np.mean(x[:n * blk].reshape(n, blk) ** 2, axis=1))
    rms = rms[rms > 1e-5]
    if rms.size < 2:
        return 0.0
    db = 20 * np.log10(rms)
    return float(np.percentile(db, 95) - np.percentile(db, 10))


def centroid(audio: np.ndarray, sr: int) -> float:
    """Spektrální těžiště v Hz — „jas“ mixu."""
    spec, f = _power_spectrum(audio, sr)
    total = float(spec.sum())
    return float((spec * f).sum() / total) if total > 0 else 0.0


def summary(audio: np.ndarray, sr: int) -> dict:
    bands = band_shares(audio, sr)
    return {
        "bands": bands,
        "mids": bands["lowmid"] + bands["mid"] + bands["himid"],
        "width": stereo_width(audio),
        "corr": correlation(audio),
        "centroid": centroid(audio, sr),
        "crest": crest_db(audio),
        "lra": loudness_range(audio, sr),
    }


# --- opakování krátkých zvuků (kroky) --------------------------------------
STEP_WINDOW = 0.2     # s: okno jednoho kroku od nástupu (zkrácené před dalším)
STEP_PRE = 0.005      # s: okno začíná kousek před nástupem (humanize, jitter)
STEP_LAG = 0.002      # s: podobnost tvaru hledá posun do ±2 ms
STEP_SIM_HZ = 1000.0  # Hz: podobnost tvaru vlny jen nad tímhle
STEP_COPY = 0.98      # podobnost, od které je událost kopií dřívější
STEP_BANDS = 24       # mel pásem pro tvar spektra
STEP_MAX = 64         # víc kroků se do párového srovnání nebere (rychlost)
STEP_SILENT = 1e-5    # okno tišší než tohle (RMS) se přeskočí


def _mel_edges(sr: int, n: int, lo: float = 100.0) -> np.ndarray:
    hi = min(16000.0, 0.45 * sr)
    mel = np.linspace(2595.0 * np.log10(1 + lo / 700.0),
                      2595.0 * np.log10(1 + hi / 700.0), n + 1)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def events(audio: np.ndarray, sr: int, onsets) -> list[dict]:
    """Popis každé události (kroku) od nástupu `onsets` (vzorky).

    Pro každé okno: `start`, `peak_db`, `rms_db`, `centroid` (Hz), `lr_db`
    (levý − pravý kanál), `shape` (tvar spektra v `STEP_BANDS` mel pásmech,
    dB s odečteným průměrem — jen barva, ne hlasitost) a `wave` (mono okno
    pro srovnání tvaru vlny). Tichá okna vynechá.
    """
    a = np.asarray(audio, dtype=np.float64)
    st = a if a.ndim == 2 else np.stack([a, a], axis=1)
    n = st.shape[0]
    pre, win = int(STEP_PRE * sr), int(STEP_WINDOW * sr)
    starts = sorted(int(o) for o in onsets)
    edges = _mel_edges(sr, STEP_BANDS)
    out = []
    for i, o in enumerate(starts):
        lo = o - pre
        hi = min(n, o + win)
        if i + 1 < len(starts):
            hi = min(hi, max(lo + int(0.02 * sr), starts[i + 1] - pre))
        if hi <= max(0, lo):
            continue
        # okno začíná vždy `pre` před nástupem (i u nástupu na začátku
        # bufferu), jinak by se dvě kopie téhož zvuku nesrovnaly
        seg = st[max(0, lo):hi]
        if lo < 0:
            seg = np.concatenate([np.zeros((-lo, 2)), seg])
        if seg.shape[0] < int(0.01 * sr):
            continue
        mono = seg.mean(axis=1)
        rms = float(np.sqrt(np.mean(mono ** 2)))
        if rms < STEP_SILENT:
            continue
        fft_n = 1 << int(np.ceil(np.log2(max(256, seg.shape[0]))))
        p = np.abs(np.fft.rfft(mono * np.hanning(mono.size), fft_n)) ** 2
        f = np.fft.rfftfreq(fft_n, 1.0 / sr)
        bands = np.array([p[(f >= a0) & (f < b0)].sum() + 1e-20
                          for a0, b0 in zip(edges[:-1], edges[1:])])
        shape = 10.0 * np.log10(bands / (edges[1:] - edges[:-1]))
        lr = (np.sqrt(np.mean(seg[:, 0] ** 2)) + 1e-12) / (np.sqrt(np.mean(seg[:, 1] ** 2)) + 1e-12)
        out.append({
            "start": o,
            "peak_db": 20.0 * np.log10(float(np.abs(mono).max()) + 1e-12),
            "rms_db": 20.0 * np.log10(rms + 1e-12),
            "centroid": float((p * f).sum() / max(1e-20, float(p.sum()))),
            "lr_db": 20.0 * float(np.log10(lr)),
            "shape": shape - shape.mean(),
            "wave": mono,
        })
    return out


def _similarity(a: np.ndarray, b: np.ndarray, lag: int, sr: int) -> float:
    """Maximum normované křížové korelace dvou oken v posunu ±`lag`, jen nad
    `STEP_SIM_HZ`: tam je náhodná část zvuku (klik, zrna). Tlumený sinus
    dupnutí je u každého kroku skoro stejný a kopii vzorku by neodlišil."""
    m = min(a.size, b.size)
    size = 1 << int(np.ceil(np.log2(2 * m)))
    fa, fb = np.fft.rfft(a[:m], size), np.fft.rfft(b[:m], size)
    keep = np.fft.rfftfreq(size, 1.0 / sr) >= STEP_SIM_HZ
    fa, fb = fa * keep, fb * keep
    na, nb = float(np.sum(np.abs(fa) ** 2)), float(np.sum(np.abs(fb) ** 2))
    if na <= 0 or nb <= 0:
        return 0.0
    c = np.fft.irfft(fa * np.conj(fb), size)
    c = np.concatenate([c[-lag:], c[:lag + 1]]) if lag > 0 else c[:1]
    # irfft(X·Y*) = korelace · size; Parseval pro rfft se zdvojením (bez DC a Nyquistu)
    return float(np.max(c) * size / (2.0 * np.sqrt(na * nb)))


def step_stats(audio: np.ndarray, sr: int, onsets) -> dict | None:
    """Jak moc se opakují kroky (nebo jiné krátké zvuky) v `audio`.

    `onsets` jsou nástupy ve vzorcích (z datového modelu). Vrátí `None`, když
    je méně než 3 slyšitelné události. Klíče: `n`; `sim_max` / `sim_med`
    (podobnost tvaru vlny nad 1 kHz, 0..1, všech párů — 1 = tentýž vzorek), `repeats`
    (kolik událostí je skoro kopií některé dřívější, podobnost ≥ `STEP_COPY`, a
    `repeat_of` = index té dřívější); `shape_adj` / `shape_all` (medián
    rozdílu barvy v dB mezi sousedy / všemi páry — sousedé blíž = kroky mají
    paměť, stejně = náhoda bez paměti); `level_sd` (dB rozptyl špiček),
    `centroid` (medián Hz), `centroid_cv`; `starts` (nástupy změřených
    událostí), `peak_db` a `lr_db` (L−P kanál) po událostech.
    """
    ev = events(audio, sr, onsets)
    if len(ev) < 3:
        return None
    ev = ev[:STEP_MAX]
    lag = int(STEP_LAG * sr)
    k = len(ev)
    sims = np.zeros((k, k))
    for i in range(k):
        for j in range(i):
            sims[i, j] = sims[j, i] = _similarity(ev[i]["wave"], ev[j]["wave"], lag, sr)
    pairs = sims[np.tril_indices(k, -1)]
    repeat_of = [int(np.argmax(sims[i, :i])) if i and sims[i, :i].max() >= STEP_COPY else None
                 for i in range(k)]
    shapes = np.array([e["shape"] for e in ev])
    dist = np.sqrt(np.mean((shapes[:, None, :] - shapes[None, :, :]) ** 2, axis=2))
    adj = np.array([dist[i, i + 1] for i in range(k - 1)])
    cents = np.array([e["centroid"] for e in ev])
    return {
        "n": k,
        "sim_max": float(pairs.max()),
        "sim_med": float(np.median(pairs)),
        "repeats": sum(r is not None for r in repeat_of),
        "repeat_of": repeat_of,
        "starts": [e["start"] for e in ev],
        "shape_adj": float(np.median(adj)),
        "shape_all": float(np.median(dist[np.tril_indices(k, -1)])),
        "level_sd": float(np.std([e["peak_db"] for e in ev])),
        "centroid": float(np.median(cents)),
        "centroid_cv": float(cents.std() / max(1e-9, cents.mean())),
        "lr_db": [e["lr_db"] for e in ev],
        "peak_db": [e["peak_db"] for e in ev],
    }


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Načte soubor jako stereo `(n, 2)` float32 (mono se zdvojí).

    .wav bez závislostí, ostatní formáty přes ffmpeg (`imageio-ffmpeg`).
    Chyby jsou `reference.ReferenceError` s českou hláškou.
    """
    from .reference import ReferenceError, _decode_to_wav, _read_wav
    if not os.path.isfile(path):
        raise ReferenceError(f"soubor neexistuje: {path}")
    if path.lower().endswith(".wav"):
        data, sr = _read_wav(path, mono=False)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "in.wav")
            _decode_to_wav(path, wav_path, channels=2)
            data, sr = _read_wav(wav_path, mono=False)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    return data[:, :2].astype(np.float32), sr


def format_table(rows: list[tuple[str, dict]]) -> str:
    """Tabulka metrik; druhý a další soubor mají i rozdíl proti prvnímu."""
    names = [b[0] for b in BANDS7]
    head = f"{'soubor':<28}" + "".join(f"{n:>7}" for n in names) + \
        f"{'středy':>8}{'šířka':>7}{'korel':>7}{'jas Hz':>8}{'crest':>7}{'LRA':>6}"
    lines = [head, "-" * len(head)]
    base = rows[0][1] if rows else None
    for i, (label, m) in enumerate(rows):
        corr = "—" if m["corr"] is None else f"{m['corr']:.2f}"
        lines.append(f"{label[:27]:<28}" + "".join(f"{m['bands'][n]:6.1f}%" for n in names)
                     + f"{m['mids']:7.1f}%{m['width']:6.1f}%{corr:>7}"
                     + f"{m['centroid']:8.0f}{m['crest']:6.1f}d{m['lra']:5.1f}d")
        if i > 0 and base is not None:
            diff = "".join(f"{m['bands'][n] - base['bands'][n]:+6.1f} " for n in names)
            lines.append(f"{'  rozdíl proti prvnímu':<28}{diff}"
                         f"{m['mids'] - base['mids']:+6.1f}  {m['width'] - base['width']:+5.1f}")
    lines.append("")
    sub_max = TARGETS["sub"][1]
    lines.append(f"cinematic cíl: sub ≤ {sub_max:g} %, středy (250 Hz–4 kHz) ≥ "
                 f"{TARGETS['mids'][0]:g} %, šířka ≥ {TARGETS['width'][0]:g} %")
    return "\n".join(lines)
