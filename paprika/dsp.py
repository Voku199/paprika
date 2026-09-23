"""Čisté DSP funkce nad numpy/scipy.

Modul nezná hudební teorii ani datový model — jen pole vzorků. Rekurzivní
filtry jedou přes `scipy.signal.lfilter` (v C), takže nikde není cyklus po
samplech. Filtr, jehož parametry se mění v čase, se počítá po blocích:
koeficienty platí pro střed bloku a stav filtru (`zi`) se předává dál, takže
na hranách bloků nelupe.

Všechny funkce berou `(n,)` nebo `(n, kanály)` a filtrují po ose 0.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

__all__ = ["KINDS", "LFO_SHAPES", "NOISE_COLORS", "biquad", "filt", "filt_varying",
           "res_to_q", "onepole", "lfo", "colored_noise", "karplus", "synth_ir",
           "blep_saw"]

KINDS = ("lp", "hp", "bp", "notch", "peak", "lowshelf", "highshelf")
Q_MIN, Q_MAX = 0.707, 12.0


def res_to_q(res):
    """Rezonance 0..1 -> Q 0.707 (Butterworth, bez špičky) .. 12 (skoro píská).

    Bere skalár i pole (automatizace rezonance)."""
    r = np.clip(np.asarray(res, dtype=np.float64), 0.0, 1.0)
    q = Q_MIN * (Q_MAX / Q_MIN) ** r
    return float(q) if q.ndim == 0 else q


def biquad(kind: str, cutoff: float, sr: int, q: float = 0.707,
           gain_db: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Koeficienty (b, a) podle RBJ Audio EQ Cookbook, normalizované na a0 = 1."""
    if kind not in KINDS:
        raise ValueError(f"neznámý typ filtru {kind!r}; k dispozici: "
                         f"{', '.join(KINDS)}")
    f = float(np.clip(cutoff, 10.0, sr * 0.49))
    q = max(1e-3, float(q))
    w0 = 2.0 * math.pi * f / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    big_a = 10.0 ** (gain_db / 40.0)
    if kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bp":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "notch":
        b = [1.0, -2 * cw, 1.0]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * big_a, -2 * cw, 1 - alpha * big_a]
        a = [1 + alpha / big_a, -2 * cw, 1 - alpha / big_a]
    else:
        A = big_a
        sq = 2.0 * math.sqrt(A) * alpha
        if kind == "lowshelf":
            b = [A * ((A + 1) - (A - 1) * cw + sq),
                 2 * A * ((A - 1) - (A + 1) * cw),
                 A * ((A + 1) - (A - 1) * cw - sq)]
            a = [(A + 1) + (A - 1) * cw + sq,
                 -2 * ((A - 1) + (A + 1) * cw),
                 (A + 1) + (A - 1) * cw - sq]
        else:
            b = [A * ((A + 1) + (A - 1) * cw + sq),
                 -2 * A * ((A - 1) + (A + 1) * cw),
                 A * ((A + 1) + (A - 1) * cw - sq)]
            a = [(A + 1) - (A - 1) * cw + sq,
                 2 * ((A - 1) - (A + 1) * cw),
                 (A + 1) - (A - 1) * cw - sq]
    b_arr = np.asarray(b, dtype=np.float64) / a[0]
    a_arr = np.asarray(a, dtype=np.float64) / a[0]
    return b_arr, a_arr


def filt(x, kind: str, cutoff: float, sr: int, q: float = 0.707,
         gain_db: float = 0.0, stages: int = 1) -> np.ndarray:
    """Biquad s konstantními koeficienty. `stages` řadí stejný filtr za sebe."""
    b, a = biquad(kind, cutoff, sr, q, gain_db)
    y = np.asarray(x, dtype=np.float64)
    if y.shape[0] == 0:
        return y.astype(np.float32)
    for _ in range(max(1, int(stages))):
        y = signal.lfilter(b, a, y, axis=0)
    return y.astype(np.float32)


def _zi(order: int, x: np.ndarray) -> np.ndarray:
    return np.zeros((order,) + tuple(x.shape[1:]), dtype=np.float64)


def filt_varying(x, kind: str, cutoff, sr: int, q=0.707, gain_db: float = 0.0,
                 stages: int = 1, block: int = 256) -> np.ndarray:
    """Biquad, jehož cutoff a/nebo Q se mění v čase (pole délky n)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if np.isscalar(cutoff) and np.isscalar(q):
        return filt(x, kind, float(cutoff), sr, float(q), gain_db, stages)
    if n == 0:
        return x.astype(np.float32)
    cut = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), (n,))
    qq = np.broadcast_to(np.asarray(q, dtype=np.float64), (n,))
    y = x
    for _ in range(max(1, int(stages))):
        out = np.empty_like(y)
        zi = _zi(2, y)
        for s in range(0, n, block):
            e = min(n, s + block)
            m = (s + e) // 2
            b, a = biquad(kind, cut[m], sr, qq[m], gain_db)
            out[s:e], zi = signal.lfilter(b, a, y[s:e], axis=0, zi=zi)
        y = out
    return y.astype(np.float32)


def _onepole_coef(cutoff: float, sr: int) -> float:
    c = float(np.clip(cutoff, 10.0, sr * 0.49))
    return 1.0 - math.exp(-2.0 * math.pi * c / sr)


def onepole(x, cutoff, sr: int, poles: int = 1, block: int = 128) -> np.ndarray:
    """Jednopólový lowpass (6 dB/okt na pól), konstantní i proměnný cutoff."""
    y = np.asarray(x, dtype=np.float64)
    n = y.shape[0]
    if n == 0:
        return y.astype(np.float32)
    for _ in range(max(1, int(poles))):
        if np.isscalar(cutoff):
            g = _onepole_coef(float(cutoff), sr)
            y = signal.lfilter([g], [1.0, g - 1.0], y, axis=0)
            continue
        cut = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), (n,))
        out = np.empty_like(y)
        zi = _zi(1, y)
        for s in range(0, n, block):
            e = min(n, s + block)
            g = _onepole_coef(cut[(s + e) // 2], sr)
            out[s:e], zi = signal.lfilter([g], [1.0, g - 1.0], y[s:e],
                                          axis=0, zi=zi)
        y = out
    return y.astype(np.float32)


# --------------------------------------------------------------------------
# modulace a zdroje
# --------------------------------------------------------------------------

LFO_SHAPES = ("sine", "tri", "saw", "square", "random")


def lfo(shape: str, rate_hz: float, n: int, sr: int, phase: float = 0.0,
        seed: int = 0) -> np.ndarray:
    """Nízkofrekvenční oscilátor v rozsahu -1..1. `phase` v periodách.

    `random` je sample & hold (nová náhodná hodnota každou periodu) vyhlazený
    jednopólem — plynulé bloudění, ne schody.
    """
    if shape not in LFO_SHAPES:
        raise ValueError(f"neznámý tvar LFO {shape!r}; k dispozici: "
                         f"{', '.join(LFO_SHAPES)}")
    cyc = float(rate_hz) * np.arange(n, dtype=np.float64) / sr + float(phase)
    ph = cyc % 1.0
    if shape == "sine":
        return np.sin(2.0 * np.pi * ph)
    if shape == "tri":
        return 4.0 * np.abs(ph - 0.5) - 1.0
    if shape == "saw":
        return 2.0 * ph - 1.0
    if shape == "square":
        return np.where(ph < 0.5, 1.0, -1.0)
    if n == 0:
        return np.zeros(0)
    idx = np.floor(cyc).astype(np.int64)
    idx -= idx.min()
    vals = np.random.default_rng(seed).uniform(-1.0, 1.0, int(idx.max()) + 1)
    smooth = onepole(vals[idx], max(0.5, float(rate_hz) * 4.0), sr)
    return np.clip(smooth.astype(np.float64), -1.0, 1.0)


NOISE_COLORS = ("white", "pink", "brown")

# Paul Kellet: aproximace 1/f filtru, -3 dB/okt v celém slyšitelném pásmu
_PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
_PINK_A = [1.0, -2.494956002, 2.017265875, -0.522189400]


def colored_noise(n: int, color: str = "white", seed: int = 0) -> np.ndarray:
    """Šum normalizovaný na peak 1. Pink = -3 dB/okt, brown = -6 dB/okt."""
    if color not in NOISE_COLORS:
        raise ValueError(f"neznámá barva šumu {color!r}; k dispozici: "
                         f"{', '.join(NOISE_COLORS)}")
    x = np.random.default_rng(seed).uniform(-1.0, 1.0, n)
    if color == "pink":
        x = signal.lfilter(_PINK_B, _PINK_A, x)
    elif color == "brown":
        x = signal.lfilter([1.0], [1.0, -0.995], x)
        x -= x.mean() if n else 0.0
    peak = float(np.abs(x).max()) if n else 0.0
    return (x / peak if peak > 0 else x).astype(np.float32)


def karplus(freq: float, n: int, sr: int, decay: float = 1.0,
            bright: float = 0.5, seed: int = 0) -> np.ndarray:
    """Karplus-Strong struna jako jeden IIR filtr (lfilter, žádný cyklus).

    Smyčka = zpoždění D vzorků + průměrovací LP (půl vzorku) + allpass
    prvního řádu na zlomkovou část periody, takže ladění sedí i nahoře.
    `decay` = doba doznění na -60 dB v sekundách, `bright` = jas úderu.
    """
    period = sr / max(20.0, float(freq))
    d = max(1, int(math.floor(period - 0.5)))
    frac = period - 0.5 - d
    c = (1.0 - frac) / (1.0 + frac)
    g = min(0.99999, 10.0 ** (-3.0 * (period / sr) / max(1e-3, float(decay))))
    # Y/X = (1 + c z^-1) / (1 + c z^-1 - g/2 (c + (1+c) z^-1 + z^-2) z^-D)
    a = np.zeros(d + 3)
    a[0] += 1.0
    a[1] += c
    a[d] -= g / 2.0 * c
    a[d + 1] -= g / 2.0 * (1.0 + c)
    a[d + 2] -= g / 2.0
    b = np.array([1.0, c])
    m = min(n, int(math.ceil(period)))
    exc = np.random.default_rng(seed).uniform(-1.0, 1.0, m)
    top = 300.0 + float(np.clip(bright, 0.0, 1.0)) * (sr * 0.45 - 300.0)
    exc = onepole(exc, top, sr).astype(np.float64)
    burst = np.zeros(n)
    burst[:m] = exc - exc.mean()
    y = signal.lfilter(b, a, burst)
    peak = float(np.abs(y).max()) if n else 0.0
    return (y / peak if peak > 0 else y).astype(np.float32)


def synth_ir(seconds: float, sr: int, predelay: float = 0.02, damp: float = 0.5,
             early: float = 0.3, seed: int = 0) -> np.ndarray:
    """Syntetická stereo impulzní odezva prostoru, tvar (m, 2).

    Dekorelovaný šum L/R s exponenciálním útlumem na -60 dB za `seconds`,
    výšky tlumené čím dál víc (`damp`), pár časných odrazů (`early`).
    Každý kanál má součet kvadrátů 1 — konvoluce tedy nemění výkon šumu.
    """
    seconds = max(0.05, float(seconds))
    m = int(seconds * 1.1 * sr)
    t = np.arange(m) / sr
    env = np.exp(-6.9078 * t / seconds)
    w = np.clip(t / seconds, 0.0, 1.0) ** 0.5 * float(np.clip(damp, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    out = np.zeros((m, 2))
    for ch in range(2):
        nz = rng.standard_normal(m)
        dark = onepole(nz, 1500.0, sr, poles=2).astype(np.float64)
        dark *= np.std(nz) / max(1e-9, float(np.std(dark)))
        tail = ((1.0 - w) * nz + w * dark) * env
        if early > 0:
            for k in range(8):
                i = int(rng.uniform(0.003, 0.08) * sr)
                if i < m:
                    tail[i] += rng.choice([-1.0, 1.0]) * early * 3.0 * (0.8 ** k)
        out[:, ch] = tail / np.sqrt(float((tail ** 2).sum()))
    pre = np.zeros((int(max(0.0, float(predelay)) * sr), 2))
    return np.concatenate([pre, out]).astype(np.float32)


def blep_saw(cycles, dt) -> np.ndarray:
    """Pila s polyBLEP korekcí skoku — levná a skoro bez aliasu.

    `cycles` je fáze v periodách (smí růst donekonečna), `dt` = f / sr
    (skalár nebo pole stejného tvaru).
    """
    t = np.asarray(cycles, dtype=np.float64) % 1.0
    dt = np.broadcast_to(np.clip(np.asarray(dt, dtype=np.float64), 1e-6, 0.5),
                         t.shape)
    y = 2.0 * t - 1.0
    lo = t < dt
    x = t[lo] / dt[lo]
    y[lo] -= x + x - x * x - 1.0
    hi = t > 1.0 - dt
    x = (t[hi] - 1.0) / dt[hi]
    y[hi] -= x * x + x + x + 1.0
    return y
