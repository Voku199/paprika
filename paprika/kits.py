"""Bicí sady: generátory syntetizovaných úderů.

Každá sada je slovník `jméno nástroje -> f(sr) -> mono float64` při velocity
127. Sada, která nástroj nemá, spadne na `classic`. Render a cache řeší
`drums.render_hit`; tenhle modul jen vyrábí zvuk.

| sada       | charakter                                                    |
|------------|--------------------------------------------------------------|
| classic    | zvuk v1 — neutrální elektronické bicí                        |
| trap       | punchy 808 kick s klikem, ostrý snare/clap, těsné hi-haty    |
| cinematic  | orchestrální basový buben, taiko (`b`), field snare, gong    |
| breakbeat  | punchy „akustický“ kick, praskavý snare, živé hi-haty (Moby) |
| industrial | zkreslené, kovové, bitcrushnuté                               |
| lofi       | měkké, tlumené, zašuměné                                     |
"""

from __future__ import annotations

import math

import numpy as np

from . import dsp
from .filters import onepole_hp, onepole_lp

__all__ = ["KITS", "kit_names"]

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------
# pomocné
# --------------------------------------------------------------------------

def _t(n: int, sr: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / sr


def _noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, n)


def _fade_out(buf: np.ndarray, sr: int, sec: float = 0.002) -> np.ndarray:
    n = min(int(sec * sr), buf.size // 2)
    if n > 0:
        ramp = np.linspace(1.0, 0.0, n)
        buf[-n:] *= ramp
        buf[:n] *= ramp[::-1]
    return buf


def _norm(x: np.ndarray, peak: float) -> np.ndarray:
    m = float(np.abs(x).max()) if x.size else 0.0
    return x / m * peak if m > 0 else x


def _finish(x: np.ndarray, sr: int, peak: float) -> np.ndarray:
    return _fade_out(_norm(np.asarray(x, dtype=np.float64), peak), sr)


def _sweep(sr: int, dur: float, f_hi: float, f_lo: float, sweep: float,
           decay: float, drive: float = 0.0) -> np.ndarray:
    """Sinus s klesající výškou (kopák, tom, taiko, 808)."""
    n = int(dur * sr)
    t = _t(n, sr)
    f = f_lo + (f_hi - f_lo) * np.exp(-t / sweep)
    y = np.sin(TWO_PI * np.cumsum(f) / sr) * np.exp(-t / decay)
    if drive > 0:
        y = np.tanh(drive * y) / np.tanh(drive)
    return y


def _env(n: int, sr: int, decay: float, attack: float = 0.0) -> np.ndarray:
    t = _t(n, sr)
    e = np.exp(-t / decay)
    if attack > 0:
        e *= np.clip(t / attack, 0.0, 1.0)
    return e


def _metal(sr: int, dur: float, freqs, decay: float, square: bool = True) -> np.ndarray:
    """Neharmonické parciály — činely, hi-haty, kovové údery."""
    n = int(dur * sr)
    t = _t(n, sr)
    out = np.zeros(n)
    for i, f in enumerate(freqs):
        s = np.sin(TWO_PI * f * t + 0.7 * i)
        out += np.sign(s) if square else s
    return out * np.exp(-t / decay) / max(1, len(freqs))


def _bp(x: np.ndarray, f: float, sr: int, q: float) -> np.ndarray:
    return dsp.filt(x, "bp", f, sr, q=q).astype(np.float64)


def _lp(x: np.ndarray, f: float, sr: int) -> np.ndarray:
    return dsp.filt(x, "lp", f, sr).astype(np.float64)


def _hp(x: np.ndarray, f: float, sr: int) -> np.ndarray:
    return dsp.filt(x, "hp", f, sr).astype(np.float64)


def _crush(x: np.ndarray, bits: int, ds: int) -> np.ndarray:
    held = np.repeat(x[::ds], ds)[:x.size]
    steps = 2.0 ** (bits - 1)
    return np.round(held * steps) / steps


def _claps(sr: int, dur: float, bursts, spacing: float, tail: float, seed: int) -> np.ndarray:
    n = int(dur * sr)
    out = np.zeros(n)
    for i in range(bursts):
        off = int(i * spacing * sr)
        seg = _noise(n - off, seed + i) * np.exp(-_t(n - off, sr) / 0.012)
        out[off:] += seg * (1.0 - 0.15 * i)
    out += _noise(n, seed + 40) * np.exp(-_t(n, sr) / tail) * 0.35
    return out


# --------------------------------------------------------------------------
# classic — zvuk v1 (beze změny) + nové nástroje
# --------------------------------------------------------------------------

def _kick(sr: int) -> np.ndarray:
    n = int(0.42 * sr)
    t = _t(n, sr)
    # pitch envelope 110 Hz -> 45 Hz, to dělá ten "dunk"
    f = 45.0 + 75.0 * np.exp(-t / 0.030)
    phase = TWO_PI * np.cumsum(f) / sr
    body = np.sin(phase) * np.exp(-t / 0.16)
    click = _noise(n, 7) * np.exp(-t / 0.004) * 0.25
    return _fade_out((body + click) * 0.80, sr)


def _snare(sr: int) -> np.ndarray:
    n = int(0.30 * sr)
    t = _t(n, sr)
    noise = _noise(n, 11) * np.exp(-t / 0.085)
    noise = onepole_hp(noise.astype(np.float32), 900.0, sr, poles=1)
    noise = onepole_lp(noise, 7500.0, sr, poles=1)
    tone = (np.sin(TWO_PI * 185.0 * t) * 0.6 +
            np.sin(TWO_PI * 330.0 * t) * 0.3) * np.exp(-t / 0.045)
    return _fade_out(noise.astype(np.float64) * 0.9 + tone * 0.5, sr)


def _hat(sr: int, decay: float = 0.045, seed: int = 21) -> np.ndarray:
    n = int(max(0.05, decay * 6) * sr)
    t = _t(n, sr)
    noise = _noise(n, seed) * np.exp(-t / decay)
    noise = onepole_hp(noise.astype(np.float32), 7000.0, sr, poles=2)
    return _fade_out(noise.astype(np.float64) * 0.55, sr)


def _openhat(sr: int) -> np.ndarray:
    return _hat(sr, decay=0.28, seed=22)


def _clap(sr: int) -> np.ndarray:
    n = int(0.35 * sr)
    t = _t(n, sr)
    out = np.zeros(n)
    # tři rychlé odrazy plus ocásek — tak zní tleskání skupiny lidí
    for i, delay in enumerate((0.0, 0.011, 0.021)):
        off = int(delay * sr)
        seg = _noise(n - off, 31 + i) * np.exp(-_t(n - off, sr) / 0.014)
        out[off:] += seg * (1.0 - 0.18 * i)
    out += _noise(n, 41) * np.exp(-t / 0.12) * 0.35
    out = onepole_hp(out.astype(np.float32), 900.0, sr, poles=1)
    out = onepole_lp(out, 6000.0, sr, poles=1)
    return _fade_out(out.astype(np.float64) * 0.6, sr)


def _tom(sr: int) -> np.ndarray:
    n = int(0.38 * sr)
    t = _t(n, sr)
    f = 120.0 + 90.0 * np.exp(-t / 0.05)
    phase = TWO_PI * np.cumsum(f) / sr
    body = np.sin(phase) * np.exp(-t / 0.18)
    noise = _noise(n, 51) * np.exp(-t / 0.02) * 0.15
    return _fade_out((body + noise) * 0.8, sr)


def _ride(sr: int) -> np.ndarray:
    n = int(0.9 * sr)
    t = _t(n, sr)
    metal = np.zeros(n)
    for k, f in enumerate((2100.0, 3170.0, 4210.0, 5390.0, 6700.0)):
        metal += np.sin(TWO_PI * f * t) * (0.6 ** k)
    metal *= np.exp(-t / 0.35)
    noise = _noise(n, 61) * np.exp(-t / 0.10) * 0.5
    out = onepole_hp((metal * 0.35 + noise).astype(np.float32), 3000.0, sr, poles=1)
    return _fade_out(out.astype(np.float64) * 0.45, sr)


def _classic_rim(sr):
    n = int(0.12 * sr)
    t = _t(n, sr)
    tone = np.sin(TWO_PI * 1700.0 * t) * np.exp(-t / 0.02)
    click = _hp(_noise(n, 71), 2000.0, sr) * np.exp(-t / 0.005)
    return _finish(tone + 0.6 * click, sr, 0.8)


def _classic_boom(sr):
    n = int(1.0 * sr)
    body = _sweep(sr, 1.0, 80.0, 40.0, 0.05, 0.6)
    thump = _lp(_noise(n, 72), 300.0, sr) * _env(n, sr, 0.05)
    return _finish(body + 0.5 * thump, sr, 0.9)


def _classic_shaker(sr):
    n = int(0.2 * sr)
    return _finish(_bp(_noise(n, 73), 6000.0, sr, 1.0) * _env(n, sr, 0.06, 0.01), sr, 0.55)


def _classic_perc(sr):
    n = int(0.4 * sr)
    t = _t(n, sr)
    y = (np.sign(np.sin(TWO_PI * 540.0 * t)) + np.sign(np.sin(TWO_PI * 800.0 * t)))
    y = _bp(y * _env(n, sr, 0.08), 900.0, sr, 1.2)
    return _finish(y, sr, 0.75)


CLASSIC = {
    "kick": _kick, "snare": _snare, "hat": lambda sr: _hat(sr),
    "openhat": _openhat, "clap": _clap, "tom": _tom, "ride": _ride,
    "rim": _classic_rim, "boom": _classic_boom, "shaker": _classic_shaker,
    "perc": _classic_perc,
}


# --------------------------------------------------------------------------
# trap
# --------------------------------------------------------------------------

_TRAP_METAL = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)


def _trap_kick(sr):
    # punchy: krátký sub ocas, výrazný klik — dlouhý 808 ocas dělá z mixu kaši
    n = int(0.7 * sr)
    body = _sweep(sr, 0.7, 200.0, 52.0, 0.02, 0.3, drive=2.5)
    click = _hp(_noise(n, 101), 3000.0, sr) * _env(n, sr, 0.004) * 0.5
    return _finish(body + click, sr, 0.92)


def _trap_snare(sr):
    n = int(0.35 * sr)
    t = _t(n, sr)
    noise = _hp(_noise(n, 102), 1500.0, sr) * _env(n, sr, 0.11)
    tone = np.sin(TWO_PI * 200.0 * t) * _env(n, sr, 0.05) * 0.5
    return _finish(noise + tone, sr, 0.9)


def _trap_clap(sr):
    y = _bp(_claps(sr, 0.45, 4, 0.008, 0.15, 103), 1200.0, sr, 0.8)
    return _finish(y, sr, 0.88)


def _trap_hat(sr, decay=0.03, dur=0.12):
    y = _hp(_metal(sr, dur, [f * 8 for f in _TRAP_METAL], decay), 7000.0, sr)
    return _finish(y, sr, 0.75)


def _trap_ride(sr):
    return _finish(_hp(_metal(sr, 1.4, [f * 6 for f in _TRAP_METAL], 0.6), 5000.0, sr), sr, 0.6)


def _trap_rim(sr):
    n = int(0.1 * sr)
    t = _t(n, sr)
    y = _bp(_noise(n, 104), 2500.0, sr, 3.0) * _env(n, sr, 0.02) \
        + np.sin(TWO_PI * 1800.0 * t) * _env(n, sr, 0.015)
    return _finish(y, sr, 0.85)


def _trap_perc(sr):
    n = int(0.15 * sr)
    y = _hp(_noise(n, 105), 2000.0, sr) * _env(n, sr, 0.012) \
        + _bp(_noise(n, 106), 3000.0, sr, 2.0) * _env(n, sr, 0.02)
    return _finish(y, sr, 0.8)


def _trap_shaker(sr):
    n = int(0.2 * sr)
    return _finish(_bp(_noise(n, 107), 7000.0, sr, 1.2) * _env(n, sr, 0.05, 0.012), sr, 0.6)


TRAP = {
    "kick": _trap_kick,
    "snare": _trap_snare,
    "clap": _trap_clap,
    "hat": lambda sr: _trap_hat(sr),
    "openhat": lambda sr: _trap_hat(sr, decay=0.25, dur=0.8),
    "tom": lambda sr: _finish(_sweep(sr, 0.7, 220.0, 110.0, 0.04, 0.3, drive=1.5), sr, 0.88),
    "ride": _trap_ride,
    "rim": _trap_rim,
    "boom": lambda sr: _finish(_sweep(sr, 2.2, 120.0, 42.0, 0.04, 1.4, drive=1.2), sr, 0.92),
    "shaker": _trap_shaker,
    "perc": _trap_perc,
}


# --------------------------------------------------------------------------
# cinematic
# --------------------------------------------------------------------------

def _cine_kick(sr):
    n = int(2.0 * sr)
    body = _sweep(sr, 2.0, 75.0, 48.0, 0.06, 0.9)
    thump = _lp(_noise(n, 201), 250.0, sr) * _env(n, sr, 0.12) * 0.5
    return _finish(body + thump, sr, 0.92)


def _cine_snare(sr):
    n = int(0.9 * sr)
    t = _t(n, sr)
    head = _bp(_noise(n, 202), 3500.0, sr, 0.6) * _env(n, sr, 0.22)
    wires = _hp(_noise(n, 203), 4000.0, sr) * _env(n, sr, 0.3) * 0.4
    tone = np.sin(TWO_PI * 180.0 * t) * _env(n, sr, 0.04)
    return _finish(head + wires + tone, sr, 0.88)


def _cine_hat(sr):
    n = int(0.1 * sr)
    y = _metal(sr, 0.1, (3390.0, 4870.0, 6120.0, 7730.0), 0.02, square=False)
    y = _hp(y + 0.3 * _noise(n, 204) * _env(n, sr, 0.015), 9000.0, sr)
    return _finish(y, sr, 0.6)


def _cine_gong(sr):
    n = int(3.5 * sr)
    metal = _metal(sr, 3.5, (211.0, 297.0, 413.0, 586.0, 742.0, 1003.0, 1391.0), 1.4,
                   square=False)
    wash = _hp(_noise(n, 205), 3000.0, sr) * _env(n, sr, 1.2, 0.05) * 0.5
    return _finish(metal + wash, sr, 0.7)


def _cine_clap(sr):
    n = int(0.5 * sr)
    t = _t(n, sr)
    y = _bp(_claps(sr, 0.5, 5, 0.012, 0.2, 206), 900.0, sr, 0.7)
    y += np.sin(TWO_PI * 90.0 * t) * _env(n, sr, 0.08)
    return _finish(y, sr, 0.88)


def _cine_tom(sr):
    n = int(1.2 * sr)
    y = _sweep(sr, 1.2, 140.0, 80.0, 0.05, 0.55) \
        + _lp(_noise(n, 207), 1000.0, sr) * _env(n, sr, 0.03) * 0.4
    return _finish(y, sr, 0.9)


def _cine_rim(sr):
    n = int(0.15 * sr)
    t = _t(n, sr)
    y = np.sin(TWO_PI * 950.0 * t) * _env(n, sr, 0.035) \
        + np.sin(TWO_PI * 1480.0 * t) * _env(n, sr, 0.02) * 0.6 \
        + _bp(_noise(n, 208), 1000.0, sr, 4.0) * _env(n, sr, 0.01)
    return _finish(y, sr, 0.85)


def _cine_taiko(sr):
    n = int(1.6 * sr)
    body = _sweep(sr, 1.6, 110.0, 62.0, 0.02, 0.7)
    skin = _bp(_noise(n, 209), 350.0, sr, 1.5) * _env(n, sr, 0.05) * 0.6
    y = np.tanh(1.5 * (body + skin)) / np.tanh(1.5)
    return _finish(y, sr, 0.95)


def _cine_anvil(sr):
    return _finish(_metal(sr, 1.8, (1180.0, 1760.0, 2630.0, 3910.0), 1.2, square=False),
                   sr, 0.7)


CINEMATIC = {
    "kick": _cine_kick,
    "snare": _cine_snare,
    "hat": _cine_hat,
    "openhat": _cine_gong,
    "clap": _cine_clap,
    "tom": _cine_tom,
    "ride": lambda sr: _finish(_hp(_metal(sr, 2.0, (2310.0, 3290.0, 4460.0, 5610.0), 1.0,
                                          square=False), 4000.0, sr), sr, 0.5),
    "rim": _cine_rim,
    "boom": _cine_taiko,
    "shaker": lambda sr: _finish(_bp(_noise(int(0.3 * sr), 210), 5000.0, sr, 0.8)
                                 * _env(int(0.3 * sr), sr, 0.09, 0.02), sr, 0.55),
    "perc": _cine_anvil,
}


# --------------------------------------------------------------------------
# breakbeat
# --------------------------------------------------------------------------

def _break_kick(sr):
    n = int(0.5 * sr)
    body = _sweep(sr, 0.5, 150.0, 58.0, 0.015, 0.22, drive=2.5)
    click = _bp(_noise(n, 301), 4000.0, sr, 1.0) * _env(n, sr, 0.004) * 0.5
    return _finish(body + click, sr, 0.92)


def _break_snare(sr):
    n = int(0.45 * sr)
    t = _t(n, sr)
    crack = _hp(_noise(n, 302), 400.0, sr) * _env(n, sr, 0.16)
    tone = (np.sin(TWO_PI * 220.0 * t) + 0.7 * np.sin(TWO_PI * 180.0 * t)) * _env(n, sr, 0.06)
    y = np.tanh(2.2 * (crack + 0.6 * tone))
    return _finish(y, sr, 0.9)


def _break_hat(sr, decay=0.06, dur=0.25, seed=303):
    n = int(dur * sr)
    t = _t(n, sr)
    body = _bp(_noise(n, seed), 8000.0, sr, 0.7) * _env(n, sr, decay)
    ring = _metal(sr, dur, (4020.0, 5830.0, 7210.0), decay * 1.2, square=False) * 0.5
    tick = np.sin(TWO_PI * 3000.0 * t) * _env(n, sr, 0.004) * 0.4
    return _finish(body + ring + tick, sr, 0.6)


def _break_conga(sr):
    n = int(0.5 * sr)
    y = _sweep(sr, 0.5, 380.0, 330.0, 0.02, 0.18) \
        + _bp(_noise(n, 304), 2000.0, sr, 2.0) * _env(n, sr, 0.01)
    return _finish(y, sr, 0.85)


BREAKBEAT = {
    "kick": _break_kick,
    "snare": _break_snare,
    "hat": lambda sr: _break_hat(sr),
    "openhat": lambda sr: _break_hat(sr, decay=0.35, dur=1.0, seed=305),
    "clap": lambda sr: _finish(_bp(_claps(sr, 0.35, 3, 0.01, 0.1, 306), 1800.0, sr, 0.9),
                               sr, 0.8),
    "tom": lambda sr: _finish(_sweep(sr, 0.6, 180.0, 130.0, 0.03, 0.25, drive=1.5), sr, 0.88),
    "ride": lambda sr: _finish(_metal(sr, 1.6, (2450.0, 3620.0, 5110.0), 0.8, square=False),
                               sr, 0.45),
    "rim": lambda sr: _finish(_bp(_noise(int(0.08 * sr), 307), 3000.0, sr, 5.0)
                              * _env(int(0.08 * sr), sr, 0.015), sr, 0.8),
    "boom": lambda sr: _finish(_sweep(sr, 1.0, 90.0, 50.0, 0.04, 0.5), sr, 0.9),
    "shaker": lambda sr: _finish(_bp(_noise(int(0.15 * sr), 308), 8000.0, sr, 1.0)
                                 * _env(int(0.15 * sr), sr, 0.04, 0.006), sr, 0.5),
    "perc": _break_conga,
}


# --------------------------------------------------------------------------
# industrial
# --------------------------------------------------------------------------

def _ind_kick(sr):
    y = _sweep(sr, 0.7, 200.0, 50.0, 0.02, 0.35, drive=8.0)
    return _finish(_crush(y, 8, 1), sr, 0.92)


def _ind_snare(sr):
    n = int(0.4 * sr)
    noise = _crush(_noise(n, 401) * _env(n, sr, 0.15), 6, 3)
    metal = _metal(sr, 0.4, (400.0, 1130.0), 0.1, square=False)
    y = np.tanh(3.0 * (noise + metal))
    return _finish(y, sr, 0.9)


def _ind_hat(sr, decay=0.04, dur=0.15, bits=5, seed=402):
    n = int(dur * sr)
    y = _crush(_hp(_noise(n, seed), 5000.0, sr) * _env(n, sr, decay), bits, 2)
    return _finish(y, sr, 0.6)


def _ind_clap(sr):
    n = int(0.25 * sr)
    gate = (_t(n, sr) < 0.06).astype(np.float64)
    y = np.tanh(4.0 * _bp(_noise(n, 403), 1500.0, sr, 0.8) * gate)
    return _finish(y, sr, 0.85)


def _ind_clank(sr):
    n = int(0.8 * sr)
    y = _metal(sr, 0.8, (420.0, 1030.0, 1780.0, 2710.0, 3570.0), 0.25, square=False) \
        + _noise(n, 404) * _env(n, sr, 0.01) * 0.5
    return _finish(np.tanh(2.0 * y), sr, 0.8)


def _ind_boom(sr):
    n = int(1.6 * sr)
    t = _t(n, sr)
    y = (np.sin(TWO_PI * 45.0 * t) + np.sin(TWO_PI * 47.0 * t)) * _env(n, sr, 1.0) \
        + dsp.colored_noise(n, "brown", 405).astype(np.float64) * _env(n, sr, 0.3)
    return _finish(np.tanh(3.0 * y), sr, 0.92)


INDUSTRIAL = {
    "kick": _ind_kick,
    "snare": _ind_snare,
    "hat": lambda sr: _ind_hat(sr),
    "openhat": lambda sr: _ind_hat(sr, decay=0.3, dur=1.0, bits=6, seed=406),
    "clap": _ind_clap,
    "tom": lambda sr: _finish(_sweep(sr, 0.7, 160.0, 70.0, 0.03, 0.3, drive=5.0), sr, 0.88),
    "ride": lambda sr: _finish(_metal(sr, 1.6, (312.0, 537.0, 841.0, 1290.0), 0.7,
                                      square=False), sr, 0.6),
    "rim": lambda sr: _finish(_metal(sr, 0.2, (2100.0, 3300.0), 0.04, square=False), sr, 0.75),
    "boom": _ind_boom,
    "shaker": lambda sr: _finish(_crush(_bp(_noise(int(0.15 * sr), 407), 6000.0, sr, 1.0)
                                        * _env(int(0.15 * sr), sr, 0.05), 4, 2), sr, 0.5),
    "perc": _ind_clank,
}


# --------------------------------------------------------------------------
# lofi
# --------------------------------------------------------------------------

def _lofi(x, sr, peak, lp=None, bits=10, ds=2):
    if lp:
        x = _lp(x, lp, sr)
    return _finish(_crush(_norm(x, 0.95), bits, ds), sr, peak)


def _lofi_snare(sr):
    n = int(0.3 * sr)
    t = _t(n, sr)
    noise = _hp(_lp(_noise(n, 501), 5000.0, sr), 600.0, sr) * _env(n, sr, 0.1)
    tone = np.sin(TWO_PI * 190.0 * t) * _env(n, sr, 0.05) * 0.7
    return _lofi(noise + tone, sr, 0.8)


def _lofi_hat(sr, decay=0.03, dur=0.12, seed=502):
    n = int(dur * sr)
    y = _lp(_hp(_noise(n, seed), 5000.0, sr), 9000.0, sr) * _env(n, sr, decay)
    return _lofi(y, sr, 0.45, bits=9)


def _lofi_bell(sr):
    n = int(0.8 * sr)
    t = _t(n, sr)
    y = (np.sin(TWO_PI * 880.0 * t) + 0.3 * np.sin(TWO_PI * 1760.0 * t)) * _env(n, sr, 0.3)
    return _lofi(y, sr, 0.6)


LOFI = {
    "kick": lambda sr: _lofi(_sweep(sr, 0.45, 110.0, 52.0, 0.03, 0.2), sr, 0.85, lp=2000.0),
    "snare": _lofi_snare,
    "hat": lambda sr: _lofi_hat(sr),
    "openhat": lambda sr: _lofi_hat(sr, decay=0.2, dur=0.7, seed=503),
    "clap": lambda sr: _lofi(_claps(sr, 0.35, 3, 0.012, 0.1, 504), sr, 0.7, lp=4000.0),
    "tom": lambda sr: _lofi(_sweep(sr, 0.6, 150.0, 100.0, 0.04, 0.25), sr, 0.8, lp=1500.0),
    "ride": lambda sr: _lofi(_metal(sr, 1.2, (2210.0, 3170.0, 4390.0), 0.5, square=False),
                             sr, 0.4, lp=7000.0),
    "rim": lambda sr: _lofi(_bp(_noise(int(0.1 * sr), 505), 1800.0, sr, 4.0)
                            * _env(int(0.1 * sr), sr, 0.02)
                            + np.sin(TWO_PI * 1200.0 * _t(int(0.1 * sr), sr))
                            * _env(int(0.1 * sr), sr, 0.02), sr, 0.7),
    "boom": lambda sr: _lofi(_sweep(sr, 1.0, 80.0, 45.0, 0.05, 0.6), sr, 0.85, lp=800.0),
    "shaker": lambda sr: _lofi(_bp(_noise(int(0.15 * sr), 506), 5000.0, sr, 1.0)
                               * _env(int(0.15 * sr), sr, 0.05), sr, 0.4),
    "perc": _lofi_bell,
}


KITS = {
    "classic": CLASSIC,
    "trap": TRAP,
    "cinematic": CINEMATIC,
    "breakbeat": BREAKBEAT,
    "industrial": INDUSTRIAL,
    "lofi": LOFI,
}


def kit_names() -> list[str]:
    return list(KITS)
