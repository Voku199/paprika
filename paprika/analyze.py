"""Zpětná vazba pro AI: mix report a hudební warningy.

Modul čte jen hotový buffer a datový model. O syntéze neví nic.
"""

from __future__ import annotations

import os

import numpy as np

from . import theory
from .song import DrumTrack, Song

__all__ = ["report", "warnings_for", "band_energy", "db", "Warning_"]

BANDS = ((0.0, 250.0), (250.0, 4000.0), (4000.0, 22050.0))
BAND_NAMES = ("low", "mid", "high")


class Warning_:
    """Jedno hlášení. `level` je WARN nebo INFO."""

    __slots__ = ("level", "where", "text")

    def __init__(self, level: str, where: str, text: str):
        self.level = level
        self.where = where
        self.text = text

    def line(self) -> str:
        return f"{self.level:<4}  {self.where:<14} {self.text}"

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.level} {self.where}: {self.text}>"


def db(x: float) -> float:
    """Lineární amplituda -> dB. Nula je -inf, tady zastropovaná na -99."""
    x = float(abs(x))
    return -99.0 if x < 1e-9 else float(20.0 * np.log10(x))


def _mono(buf):
    """Stereo buffer stopy -> mono pro analýzu (pravidla pracují s mono)."""
    buf = np.asarray(buf)
    return buf.mean(axis=1) if buf.ndim == 2 else buf


def _fmt_db(x: float) -> str:
    v = db(x)
    return "-inf" if v <= -98.9 else f"{v:.1f}"


def _mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


# --------------------------------------------------------------------------
# spektrum
# --------------------------------------------------------------------------

def band_energy(buf: np.ndarray, sr: int) -> np.ndarray:
    """Energie v pásmech low / mid / high pro jeden blok vzorků."""
    if buf.size < 16:
        return np.zeros(3)
    spec = np.abs(np.fft.rfft(buf * np.hanning(buf.size))) ** 2
    freqs = np.fft.rfftfreq(buf.size, 1.0 / sr)
    out = np.empty(3)
    for i, (lo, hi) in enumerate(BANDS):
        out[i] = spec[(freqs >= lo) & (freqs < hi)].sum()
    return out


def _per_bar_bands(buf: np.ndarray, sr: int, bar_samples: int, bars: int):
    """(bars, 3) matice energií po taktech."""
    out = np.zeros((max(bars, 1), 3))
    for b in range(bars):
        seg = buf[b * bar_samples:(b + 1) * bar_samples]
        if seg.size:
            out[b] = band_energy(seg, sr)
    return out


# --------------------------------------------------------------------------
# pravidla
# --------------------------------------------------------------------------

def _bar_of(song: Song, beat: float) -> int:
    return int(beat // song.beats_per_bar)


def _ranges(bars: list[int]) -> list[tuple[int, int]]:
    """[3,4,5,9] -> [(3,5), (9,9)]"""
    out: list[tuple[int, int]] = []
    for b in sorted(set(bars)):
        if out and b == out[-1][1] + 1:
            out[-1] = (out[-1][0], b)
        else:
            out.append((b, b))
    return out


def _out_of_scale(song: Song, warns: list) -> None:
    pcs = set(song.scale_pcs())
    for t in song.tracks:
        if t.chromatic or isinstance(t, DrumTrack):
            continue
        bad = [(n.pitch, _bar_of(song, n.start_beat))
               for n in t.all_notes() if n.pitch % 12 not in pcs]
        if not bad:
            continue
        for p, bar in bad[:4]:
            warns.append(Warning_(
                "WARN", t.name,
                f"bar {bar}   {theory.note_name(p)} mimo "
                f"{song.key} {song.scale}"))
        if len(bad) > 4:
            warns.append(Warning_(
                "INFO", t.name,
                f"a dalších {len(bad) - 4} not mimo stupnici "
                f"(chromatic=True to hlášení vypne)"))


def _levels(song: Song, audio: np.ndarray, raw_peak: float, warns: list) -> None:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    if raw_peak > 1.0:
        warns.append(Warning_("WARN", "mix",
                              f"clipping před limiterem, peak {db(raw_peak):.1f} "
                              f"dBFS — stáhni vol na stopách"))
    if db(peak) > -0.3:
        warns.append(Warning_("WARN", "mix", "peak nad -0.3 dBFS, mix je nalisovaný"))
    if audio.size and db(rms) < -30.0:
        warns.append(Warning_("WARN", "mix",
                              f"RMS {db(rms):.1f} dB — celé to zní moc potichu"))


SILENT_BAR_DB = -80.0     # takt tišší než tohle se do maskování nepočítá
MASKING_SHOWN = 4         # kolik nejdelších překryvů report vypíše jako WARN


def _masking(song: Song, buffers: dict, warns: list) -> None:
    sr = song.sample_rate
    bar_samples = int(round(song.beats_per_bar * song.seconds_per_beat * sr))
    if bar_samples <= 0:
        return
    bars = song.bars
    floor = 10.0 ** (SILENT_BAR_DB / 20.0)
    mats, active = {}, {}
    for t in song.tracks:
        buf = buffers.get(t.name)
        if buf is None or not np.any(buf):
            continue
        mats[t.name] = _per_bar_bands(buf, sr, bar_samples, bars)
        # numerický šum (třeba z konvoluce) v tichém taktu není hudba
        active[t.name] = [
            float(np.sqrt(np.mean(seg ** 2))) > floor if seg.size else False
            for seg in (buf[b * bar_samples:(b + 1) * bar_samples]
                        for b in range(bars))]

    found = []
    names = list(mats)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ma, mb = mats[a], mats[b]
            for band in range(3):
                hits = []
                for bar in range(bars):
                    if not (active[a][bar] and active[b][bar]):
                        continue
                    ta, tb = ma[bar].sum(), mb[bar].sum()
                    if ta <= 0 or tb <= 0:
                        continue
                    if ma[bar][band] / ta > 0.4 and mb[bar][band] / tb > 0.4:
                        hits.append(bar)
                # kratší překryv než tři takty nikoho netrápí
                ranges = [(lo, hi) for lo, hi in _ranges(hits) if hi - lo >= 2]
                if ranges:
                    total = sum(hi - lo + 1 for lo, hi in ranges)
                    found.append((total, a, b, band, ranges))

    found.sort(key=lambda f: -f[0])
    for total, a, b, band, ranges in found[:MASKING_SHOWN]:
        lo_hz, hi_hz = BANDS[band]
        spans = [f"{lo}-{hi}" for lo, hi in ranges]
        shown = ", ".join(spans[:4])
        if len(spans) > 4:
            shown += f" (+{len(spans) - 4})"
        warns.append(Warning_(
            "WARN", f"{a} + {b}",
            f"maskování {int(lo_hz)}-{int(hi_hz)} Hz, bars {shown}"))
    if len(found) > MASKING_SHOWN:
        warns.append(Warning_(
            "INFO", "mix",
            f"a dalších {len(found) - MASKING_SHOWN} maskování (vypsány "
            f"{MASKING_SHOWN} nejdelší) — HighPass/EQ nebo pan pomůže"))


def _robotic(song: Song, warns: list) -> None:
    for t in song.tracks:
        vels = ([h.velocity for h in t.hits] if isinstance(t, DrumTrack)
                else [n.velocity for n in t.all_notes()])
        if len(vels) >= 6 and float(np.std(vels)) == 0.0:
            warns.append(Warning_(
                "WARN", t.name,
                f"všechny velocity {vels[0]} — zní roboticky, zkus humanize()"))


def _dissonance(song: Song, warns: list) -> None:
    for t in song.tracks:
        if isinstance(t, DrumTrack):
            continue
        notes = t.all_notes()
        reported = 0
        for i, a in enumerate(notes):
            if a.pitch >= 48:
                continue
            for b in notes[i + 1:]:
                if b.start_beat >= a.end_beat:
                    break
                if abs(a.pitch - b.pitch) == 1 and b.pitch < 48:
                    warns.append(Warning_(
                        "WARN", t.name,
                        f"bar {_bar_of(song, a.start_beat)}   ostrá disonance "
                        f"{a.name}+{b.name} v base"))
                    reported += 1
                    break
            if reported >= 3:
                break


def _gaps(song: Song, warns: list) -> None:
    for t in song.tracks:
        starts = ([h.start_beat for h in t.hits] if isinstance(t, DrumTrack)
                  else [n.start_beat for n in t.all_notes()])
        if len(starts) < 2:
            continue
        used = sorted({_bar_of(song, s) for s in starts})
        for a, b in zip(used, used[1:]):
            if b - a - 1 >= 4:
                warns.append(Warning_(
                    "INFO", t.name, f"bars {a + 1}-{b - 1} ticho"))


def _correlation(buf: np.ndarray) -> float | None:
    """Korelace L/R stereo bufferu; None pro mono nebo ticho."""
    buf = np.asarray(buf, dtype=np.float64)
    if buf.ndim != 2 or buf.shape[1] != 2 or buf.shape[0] < 2:
        return None
    left, right = buf[:, 0], buf[:, 1]
    denom = float(np.sqrt(np.sum(left ** 2) * np.sum(right ** 2)))
    if denom <= 1e-12:
        return None
    return float(np.sum(left * right) / denom)


def _mono_compat(song: Song, stereo_buffers: dict, warns: list) -> None:
    """Stopa, jejíž levý a pravý kanál jdou proti sobě, se v mono vyruší."""
    reported = 0
    for t in song.tracks:
        buf = stereo_buffers.get(t.name)
        if buf is None:
            continue
        buf = np.asarray(buf)
        if buf.ndim != 2 or db(np.sqrt(np.mean(buf ** 2))) < -50.0:
            continue
        corr = _correlation(buf)
        if corr is not None and corr < 0.0:
            warns.append(Warning_(
                "WARN", t.name,
                f"fázový rozpad v mono (korelace {corr:.2f}) — sniž Width/Chorus"))
            reported += 1
            if reported >= 3:
                return


def _moving_patch(patch) -> bool:
    """Má patch něco, co zvuk hýbe v čase (modulace, round-robin, struna)?"""
    if (getattr(patch, "vibrato", None) or getattr(patch, "tremolo", None)
            or getattr(patch, "pitch_env", None) or getattr(patch, "drift", 0)
            or getattr(patch, "rr", 1) > 1 or getattr(patch, "vel_to_cutoff", 0)):
        return True
    for f in getattr(patch, "filters", []):
        if getattr(f, "lfo", None) is not None or getattr(f, "morph", None):
            return True
    for o in patch.osc:
        if getattr(o, "pwm", None) is not None:
            return True
        if type(o).__name__ in ("Pluck", "SuperSaw", "Additive"):
            return True
    return False


def _static_sound(song: Song, warns: list) -> None:
    """Dlouhá melodická stopa bez jakéhokoli pohybu zní „plastově“."""
    reported = 0
    for t in song.tracks:
        if isinstance(t, DrumTrack):
            continue
        notes = t.all_notes()
        if not notes:
            continue
        span = _bar_of(song, notes[-1].start_beat) - _bar_of(song, notes[0].start_beat) + 1
        if span <= 16:
            continue
        if (_moving_patch(t.patch) or getattr(t, "automations", None)
                or getattr(t, "_humanized", False)):
            continue
        warns.append(Warning_(
            "INFO", t.name,
            f"statický zvuk přes {span} taktů — přidej vibrato/LFO, "
            f"automate() nebo humanize()"))
        reported += 1
        if reported >= 3:
            return


def warnings_for(song: Song, audio: np.ndarray, buffers: dict,
                 raw_peak: float) -> list:
    """Všechna pravidla pohromadě, WARN před INFO."""
    warns: list = []
    stereo_buffers = dict(buffers or {})
    buffers = {k: _mono(v) for k, v in stereo_buffers.items()}
    _levels(song, audio, raw_peak, warns)
    _out_of_scale(song, warns)
    _masking(song, buffers, warns)
    _mono_compat(song, stereo_buffers, warns)
    _robotic(song, warns)
    _dissonance(song, warns)
    _gaps(song, warns)
    _static_sound(song, warns)
    warns.sort(key=lambda w: 0 if w.level == "WARN" else 1)
    return warns


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def report(song: Song, audio: np.ndarray, buffers: dict, raw_peak: float,
           path: str | None = None, render_seconds: float = 0.0,
           bars=None) -> str:
    sr = song.sample_rate
    dur = audio.shape[0] / sr if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio

    energy = band_energy(mono, sr)
    total = energy.sum()
    shares = (energy / total * 100.0) if total > 0 else np.zeros(3)
    # energie nad 4 kHz je i v dobře znějícím mixu jen pár procent výkonu
    balanced = shares[0] < 65 and shares[1] > 15 and shares[2] > 0.5

    name = os.path.basename(path) if path else (song.title or "bez souboru")
    n_notes = sum(len(t.hits) if isinstance(t, DrumTrack) else len(t)
                  for t in song.tracks)

    lines = [f"PAPRIKA  {name}"]
    span = (f"  jen bars {bars[0]}-{bars[1]}"
            if bars and tuple(bars) != (0, song.bars) else "")
    lines.append(
        f"render   {_mmss(dur)}  {sr} Hz  "
        f"{'stereo' if audio.ndim > 1 and audio.shape[1] == 2 else 'mono'}  "
        f"{len(song.tracks)} stop  {n_notes} not  "
        f"[{render_seconds:.1f}s]{span}")
    lines.append("")
    lines.append(
        f"MIX      peak {_fmt_db(peak)} dBFS   rms {_fmt_db(rms)} dB   "
        f"{'clipping!' if raw_peak > 1.0 else 'no clipping'}")
    lines.append(
        "BANDS    " + "  ".join(f"{BAND_NAMES[i]} {shares[i]:.0f}%"
                                for i in range(3)) +
        f"      [{'ok' if balanced else 'zkontroluj'}]")
    if audio.ndim == 2 and audio.shape[1] == 2 and audio.size:
        mid = (audio[:, 0] + audio[:, 1]) * 0.5
        side = (audio[:, 0] - audio[:, 1]) * 0.5
        e_mid, e_side = float(np.sum(mid ** 2)), float(np.sum(side ** 2))
        width = e_side / (e_mid + e_side) * 100.0 if e_mid + e_side > 0 else 0.0
        corr = _correlation(audio)
        lines.append(f"STEREO   šířka {width:.0f} %   korelace "
                     f"{'—' if corr is None else f'{corr:.2f}'}")

    parts = []
    for t in song.tracks:
        buf = buffers.get(t.name)
        level = float(np.sqrt(np.mean(buf ** 2))) * t.vol if buf is not None \
            and buf.size else 0.0
        parts.append(f"{t.name} {_fmt_db(level)} dB")
    lines.append("TRACKS   " + " | ".join(parts))
    lines.append("")

    warns = warnings_for(song, audio, buffers, raw_peak)
    if warns:
        lines.extend(w.line() for w in warns)
    else:
        lines.append("OK    žádné warningy")
    return "\n".join(lines)
