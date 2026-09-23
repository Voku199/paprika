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
           "format_table"]

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
