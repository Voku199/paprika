"""Analýza externí referenční nahrávky (mp3/wav): tempo, tónina, akordy po
taktech, dynamika. Cíl je dát AI čísla, podle kterých ručně napíše vlastní
skladbu v paprice — ne přesnou transkripci a ne source separation.

Modul je záměrně mimo hranici render enginu: `paprika/__init__.py` a
`render.py` o něm nic nevědí, takže `pip install -r requirements.txt`
(jen numpy) dál stačí na psaní a render skladeb. Dekódování mp3 potřebuje
navíc `imageio-ffmpeg` — lazy import, jen když se skutečně použije.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field

import numpy as np

from . import theory

__all__ = ["ReferenceReport", "load_reference_audio", "analyze_reference",
           "format_reference_report"]

FRAME = 4096
HOP = 1024
MIN_BPM, MAX_BPM = 60.0, 180.0

# Krumhansl-Schmuckler profily tonality (dur, moll), root = C.
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


class ReferenceError(Exception):
    """Chyba čtení/dekódování referenční nahrávky."""


@dataclass
class ReferenceReport:
    """Výsledek analýzy — jen čísla a odhady, žádná zvuková data."""

    duration_sec: float
    sample_rate: int
    bpm: float
    bpm_confidence: float
    key: str
    scale: str
    key_confidence: float
    bar_sec: float
    chords: list[str] = field(default_factory=list)
    bar_energy_db: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------
# načtení
# --------------------------------------------------------------------------

def _read_wav(path: str, mono: bool = True) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ReferenceError(f"nepodporovaná šířka vzorku ve wav: {width * 8} bit")
    if n_ch > 1:
        data = data.reshape(-1, n_ch)
        if mono:
            data = data.mean(axis=1)
    return data, sr


def _decode_to_wav(src_path: str, out_path: str, channels: int = 1) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as e:
        raise ReferenceError(
            "dekódování ne-wav souborů potřebuje 'imageio-ffmpeg' "
            "(pip install imageio-ffmpeg); .wav jde načíst i bez toho") from e
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-y", "-i", src_path, "-ac", str(channels), "-ar", "44100", out_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ReferenceError(
            f"ffmpeg selhal na {src_path!r}: {proc.stderr.strip()[-400:]}")


def load_reference_audio(path: str) -> tuple[np.ndarray, int]:
    """Načte mono float32 signál (-1..1) a sample rate z libovolného audio
    souboru, který zvládne ffmpeg (mp3, m4a, ogg, ...); .wav bez závislostí."""
    if not os.path.isfile(path):
        raise ReferenceError(f"soubor neexistuje: {path}")
    if path.lower().endswith(".wav"):
        return _read_wav(path)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "ref.wav")
        _decode_to_wav(path, wav_path)
        return _read_wav(wav_path)


# --------------------------------------------------------------------------
# DSP pomůcky
# --------------------------------------------------------------------------

def _frames(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """(n_frames, frame) matice překrývajících se oken."""
    if x.size < frame:
        x = np.concatenate([x, np.zeros(frame - x.size, dtype=x.dtype)])
    n = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _stft_mag(x: np.ndarray, sr: int, frame: int = FRAME, hop: int = HOP):
    """Magnitude STFT + odpovídající frekvence, počítané block-wise (numpy
    fancy indexing na `_frames`, žádný per-sample Python loop)."""
    win = np.hanning(frame)
    fr = _frames(x.astype(np.float32), frame, hop) * win
    spec = np.abs(np.fft.rfft(fr, axis=1))
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    return spec, freqs


def _onset_envelope(spec: np.ndarray) -> np.ndarray:
    """Spectral flux: součet kladných přírůstků magnitudy mezi po sobě
    jdoucími framy."""
    diff = np.diff(spec, axis=0)
    return np.maximum(diff, 0.0).sum(axis=1)


def _estimate_tempo(onset: np.ndarray, sr: int, hop: int) -> tuple[float, float]:
    """Autokorelace onset envelope -> BPM + hrubá jistota (0..1)."""
    if onset.size < 4 or not np.any(onset):
        return 120.0, 0.0
    onset = onset - onset.mean()
    frame_rate = sr / hop
    lag_lo = int(frame_rate * 60.0 / MAX_BPM)
    lag_hi = int(frame_rate * 60.0 / MIN_BPM)
    lag_hi = min(lag_hi, onset.size - 1)
    if lag_hi <= lag_lo:
        return 120.0, 0.0
    full = np.correlate(onset, onset, mode="full")
    ac = full[full.size // 2:]
    window = ac[lag_lo:lag_hi + 1]
    if window.size == 0 or not np.any(window):
        return 120.0, 0.0
    # Periodický signál má skoro stejně vysoké peaky i na násobcích pravé
    # periody (oktávová chyba). Radši nejrychlejší tempo, jehož peak je
    # blízko maxima v okně, než holý argmax, který sklouzává na půltempo.
    peak_val = float(window.max())
    threshold = 0.85 * peak_val
    candidates = np.where(window >= threshold)[0]
    best = int(candidates.min()) + lag_lo
    bpm = 60.0 * frame_rate / best
    conf = float(window[best - lag_lo] / (ac[0] + 1e-9))
    return float(bpm), float(np.clip(conf, 0.0, 1.0))


_PC_FREQS = 440.0 * 2.0 ** ((np.arange(12) - 9) / 12.0)  # A4=440, pc0=C


def _chroma(spec: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """(n_frames, 12) chroma matice — energii FFT binu přičte k nejbližší
    pitch class napříč oktávami (log-frequency mapping)."""
    valid = freqs > 20.0
    f = freqs[valid]
    mag = spec[:, valid] ** 2
    # nejbližší pitch class podle vzdálenosti v půltónech od C
    midi = 69 + 12 * np.log2(f / 440.0)
    pc = np.round(midi).astype(int) % 12
    out = np.zeros((spec.shape[0], 12))
    for p in range(12):
        sel = pc == p
        if np.any(sel):
            out[:, p] = mag[:, sel].sum(axis=1)
    return out


def _estimate_key(chroma_mean: np.ndarray) -> tuple[str, str, float]:
    """Krumhansl-Schmuckler: koreluje průměrný chroma s 24 rotovanými
    profily, vrátí nejlepší (kořen, dur/moll, korelace)."""
    v = chroma_mean - chroma_mean.mean()
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-9:
        return "C", "major", 0.0
    best = ("C", "major", -1.0)
    for root in range(12):
        for scale_name, profile in (("major", _MAJOR_PROFILE),
                                     ("minor", _MINOR_PROFILE)):
            p = np.roll(profile, root) - profile.mean()
            corr = float(np.dot(v, p) / (v_norm * np.linalg.norm(p) + 1e-9))
            if corr > best[2]:
                best = (theory.PC_TO_NOTE[root], scale_name, corr)
    return best


_TRIADS = {"maj": [0, 4, 7], "min": [0, 3, 7]}


def _match_chord(chroma_bar: np.ndarray) -> str:
    """Nejbližší dur/moll trojzvuk pro jeden takt chroma energie."""
    if not np.any(chroma_bar):
        return "?"
    v = chroma_bar / (chroma_bar.sum() + 1e-9)
    best_label, best_score = "?", -1.0
    for root in range(12):
        for quality, ivs in _TRIADS.items():
            template = np.zeros(12)
            for iv in ivs:
                template[(root + iv) % 12] = 1.0
            template /= template.sum()
            score = float(np.dot(v, template))
            if score > best_score:
                suffix = "" if quality == "maj" else "m"
                best_label, best_score = f"{theory.PC_TO_NOTE[root]}{suffix}", score
    return best_label


# --------------------------------------------------------------------------
# hlavní analýza
# --------------------------------------------------------------------------

def analyze_reference(samples: np.ndarray, sr: int) -> ReferenceReport:
    """Čistě numpy DSP nad mono signálem — žádné I/O, testovatelné na
    syntetických signálech."""
    if samples.size == 0:
        return ReferenceReport(duration_sec=0.0, sample_rate=sr, bpm=120.0,
                               bpm_confidence=0.0, key="C", scale="major",
                               key_confidence=0.0, bar_sec=2.0)
    duration = samples.size / sr if sr else 0.0
    spec, freqs = _stft_mag(samples, sr)
    frame_rate = sr / HOP

    onset = _onset_envelope(spec)
    bpm, bpm_conf = _estimate_tempo(onset, sr, HOP)

    chroma = _chroma(spec, freqs)
    chroma_mean = chroma.mean(axis=0) if chroma.size else np.zeros(12)
    key, scale_name, key_conf = _estimate_key(chroma_mean)

    bar_sec = 4.0 * 60.0 / bpm if bpm > 0 else 2.0
    frames_per_bar = max(1, int(round(bar_sec * frame_rate)))
    n_bars = max(1, int(np.ceil(chroma.shape[0] / frames_per_bar))) if chroma.size else 0

    rms = np.sqrt(np.mean(_frames(samples.astype(np.float32), FRAME, HOP) ** 2,
                          axis=1)) if samples.size >= FRAME else np.zeros(0)

    chords: list[str] = []
    bar_energy_db: list[float] = []
    for b in range(n_bars):
        lo, hi = b * frames_per_bar, min((b + 1) * frames_per_bar, chroma.shape[0])
        chords.append(_match_chord(chroma[lo:hi].sum(axis=0)) if hi > lo else "?")
        seg_rms = float(rms[lo:hi].mean()) if hi > lo and rms.size else 0.0
        bar_energy_db.append(-99.0 if seg_rms < 1e-9 else float(20 * np.log10(seg_rms)))

    return ReferenceReport(
        duration_sec=duration, sample_rate=sr, bpm=bpm, bpm_confidence=bpm_conf,
        key=key, scale=scale_name, key_confidence=key_conf, bar_sec=bar_sec,
        chords=chords, bar_energy_db=bar_energy_db)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _energy_bar_chars(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1.0
    ramp = " ▁▂▃▄▅▆▇█"
    out = []
    for v in values:
        level = int(round((v - lo) / span * (len(ramp) - 1)))
        out.append(ramp[max(0, min(len(ramp) - 1, level))])
    return "".join(out)


def format_reference_report(rep: ReferenceReport, name: str) -> str:
    lines = [f"REFERENCE  {name}"]
    lines.append(f"audio    {_mmss(rep.duration_sec)}  {rep.sample_rate} Hz  mono")
    lines.append("")
    conf = ("nízká" if rep.bpm_confidence < 0.15 else
            "střední" if rep.bpm_confidence < 0.35 else "vysoká")
    lines.append(f"TEMPO    ~{rep.bpm:.0f} BPM  (jistota: {conf})")
    kconf = ("nízká" if rep.key_confidence < 0.4 else
             "střední" if rep.key_confidence < 0.65 else "vysoká")
    lines.append(f"KEY      {rep.key} {rep.scale}  (korelace {rep.key_confidence:.2f}, "
                 f"jistota: {kconf})")
    lines.append("")

    if rep.bar_energy_db:
        lines.append(f"DYNAMIKA  (energie po taktech, ~{rep.bar_sec:.1f}s/takt)")
        lines.append("  " + _energy_bar_chars(rep.bar_energy_db))
        lines.append("")

    if rep.chords:
        lines.append("AKORDY   (hrubý odhad z chroma vektorů, po taktech)")
        per_row = 8
        for i in range(0, len(rep.chords), per_row):
            row = rep.chords[i:i + per_row]
            lines.append(f"  bar {i + 1:>3}-{min(i + per_row, len(rep.chords)):<3} "
                         + "  ".join(f"{c:<4}" for c in row))
        lines.append("")

    lines.append("INFO  akordy jsou hrubý odhad z chroma energie, ne přesná")
    lines.append("      transkripce — na hustém mixu (klavír+smyčce+bicí) čekej šum,")
    lines.append("      hlavně v hlasitých pasážích. Žádná separace nástrojů.")
    return "\n".join(lines)
