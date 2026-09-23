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

__all__ = ["KINDS", "biquad", "filt", "filt_varying", "res_to_q", "onepole"]

KINDS = ("lp", "hp", "bp", "notch", "peak", "lowshelf", "highshelf")
Q_MIN, Q_MAX = 0.707, 12.0


def res_to_q(res: float) -> float:
    """Rezonance 0..1 -> Q 0.707 (Butterworth, bez špičky) .. 12 (skoro píská)."""
    r = float(np.clip(res, 0.0, 1.0))
    return Q_MIN * (Q_MAX / Q_MIN) ** r


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


def onepole(x, cutoff, sr: int, poles: int = 1, block: int = 64) -> np.ndarray:
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
