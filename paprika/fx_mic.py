"""Blízký mikrofon: proximity efekt a šum místnosti.

Lidský zvuk nahraný zblízka (dech, mlasknutí, šustění) nezní jako čistý
syntetický šum ze dvou důvodů: směrový mikrofon blízko zdroje zvedne basy
a pod zvukem je vždycky tichý šum místnosti, nikdy digitální ticho. Oba
jevy jsou tady jako efekty stopy.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from . import dsp
from .fx import Fx, FxContext

__all__ = ["Proximity", "RoomTone", "MIC_PATTERNS", "SPEED_OF_SOUND"]

SPEED_OF_SOUND = 343.0    # m/s
# podíl tlakového gradientu v charakteristice mikrofonu
MIC_PATTERNS = {"cardioid": 0.5, "figure8": 1.0, "omni": 0.0}
_HOP = 0.01               # s: krok detektoru aktivity v RoomTone


class Proximity(Fx):
    """Proximity efekt směrového mikrofonu ve vzdálenosti `dist` metrů.

    Kulová vlna na mikrofonu s tlakovým gradientem: H(s) = 1 + k·c/(r·s),
    kde k = 1/2 pro kardioidu, 1 pro osmičku a 0 pro kouli (bez efektu).
    Vůči vzdálenosti `ref`, na kterou je mikrofon naladěný, je to shelf
    1. řádu (s + k·c/r) / (s + k·c/ref) s rohem f = k·c/(2π·r): kardioida
    na 5 cm zvedne 100 Hz o 14,6 dB, na 30 cm o 2,3 dB.

    `max_db` omezí zdvih basů (mikrofony mají vestavěný low-cut a ústa
    nejsou bodový zdroj), `hp` pod tím srazí sub (Butterworth 12 dB/okt,
    None = vypnuto). `dist` jde automatizovat (zdroj se přibližuje).
    """

    STEREO = True
    AUTOMATABLE = ("dist",)

    def __init__(self, dist: float = 0.15, pattern: str = "cardioid",
                 ref: float = 1.0, max_db: float | None = 12.0,
                 hp: float | None = 50.0, name: str | None = None):
        if pattern not in MIC_PATTERNS:
            raise ValueError(f"neznámá charakteristika mikrofonu {pattern!r}; "
                             f"k dispozici: {', '.join(MIC_PATTERNS)}")
        if float(dist) <= 0 or float(ref) <= 0:
            raise ValueError(f"Proximity: dist a ref musí být kladné (metry), "
                             f"dostal jsem dist={dist!r}, ref={ref!r}")
        self.dist = float(dist)
        self.pattern = pattern
        self.ref = float(ref)
        self.max_db = None if max_db is None else float(max_db)
        self.hp = None if hp is None else float(hp)
        self.name = name

    def coefs(self, dist: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """(b, a) shelfu pro danou vzdálenost; bilineární transformace."""
        k = MIC_PATTERNS[self.pattern]
        w1 = k * SPEED_OF_SOUND / max(1e-3, float(dist))
        w2 = k * SPEED_OF_SOUND / self.ref
        if self.max_db is not None and w1 > w2:
            w2 = max(w2, w1 / 10.0 ** (self.max_db / 20.0))
        return signal.bilinear([1.0, w1], [1.0, w2], sr)

    def process(self, buf, ctx: FxContext):
        y = np.asarray(buf, dtype=np.float64)
        n = y.shape[0]
        if n == 0:
            return y.astype(np.float32)
        if MIC_PATTERNS[self.pattern] > 0:
            dist = self.value("dist")
            if np.isscalar(dist) or np.ndim(dist) == 0:
                b, a = self.coefs(float(dist), ctx.sr)
                y = signal.lfilter(b, a, y, axis=0)
            else:
                # automatizace: koeficienty po blocích, stav filtru jde dál
                d = np.broadcast_to(np.asarray(dist, dtype=np.float64), (n,))
                out = np.empty_like(y)
                zi = np.zeros((1,) + y.shape[1:])
                for s in range(0, n, 256):
                    e = min(n, s + 256)
                    b, a = self.coefs(d[(s + e) // 2], ctx.sr)
                    out[s:e], zi = signal.lfilter(b, a, y[s:e], axis=0, zi=zi)
                y = out
        if self.hp:
            y = dsp.filt(y, "hp", self.hp, ctx.sr)
        return np.asarray(y, dtype=np.float32)


class RoomTone(Fx):
    """Šum místnosti pod blízkým zdrojem.

    Kde stopa zní (RMS v 10ms oknech nad `threshold` dBFS), objeví se tichý
    šum o RMS `level` dBFS. Otevře se `attack` sekund před zvukem a drží
    ještě `release` sekund po něm, jako u nahrávky stříhané s rezervou.
    Jinde zůstane ticho, takže se šum nesčítá v mixu přes celou skladbu.
    Levé a pravé kanály mají nezávislý šum (místnost je široká, zdroj ne).
    """

    STEREO = True
    AUTOMATABLE = ("level",)

    def __init__(self, level: float = -66.0, color: str = "pink",
                 attack: float = 0.08, release: float = 0.5,
                 threshold: float = -55.0, seed: int = 0,
                 name: str | None = None):
        if color not in dsp.NOISE_COLORS:
            raise ValueError(f"neznámá barva šumu {color!r}; k dispozici: "
                             f"{', '.join(dsp.NOISE_COLORS)}")
        self.level = float(level)
        self.color = color
        self.attack = max(0.0, float(attack))
        self.release = max(0.0, float(release))
        self.threshold = float(threshold)
        self.seed = int(seed)
        self.name = name

    def tail(self, ctx: FxContext) -> float:
        return self.release

    def gate(self, buf: np.ndarray, sr: int) -> np.ndarray:
        """Křivka 0..1 po samplech: kde má šum místnosti znít."""
        x = np.asarray(buf, dtype=np.float64)
        det = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
        n = det.size
        hop = max(1, int(_HOP * sr))
        nb = -(-n // hop)
        padded = np.zeros(nb * hop)
        padded[:n] = det
        rms = np.sqrt(np.mean(padded.reshape(nb, hop) ** 2, axis=1))
        active = (rms > 10.0 ** (self.threshold / 20.0)).astype(np.float64)
        if not active.any():
            return np.zeros(n)
        pre = int(math.ceil(self.attack / _HOP))
        post = int(math.ceil(self.release / _HOP))
        # blok i je otevřený, když je aktivita v blocích i-post .. i+pre
        hits = np.convolve(active, np.ones(pre + post + 1))[pre:pre + nb]
        g = np.repeat((hits > 0.5).astype(np.float64), hop)[:n]
        smooth = max(0.005, self.attack / 3.0)
        return np.clip(dsp.onepole(g, 1.0 / (2.0 * math.pi * smooth), sr), 0.0, 1.0)

    def process(self, buf, ctx: FxContext):
        buf = np.asarray(buf, dtype=np.float32)
        n = buf.shape[0]
        if n == 0:
            return buf
        g = self.gate(buf, ctx.sr)
        if not g.any():
            return buf
        chans = buf.shape[1] if buf.ndim == 2 else 1
        tone = np.zeros((n, chans))
        for c in range(chans):
            nz = dsp.colored_noise(n, self.color, self.seed * 7 + c).astype(np.float64)
            tone[:, c] = nz / max(1e-12, float(np.sqrt(np.mean(nz ** 2))))
        level = self.value("level")
        amp = 10.0 ** (np.asarray(level, dtype=np.float64) / 20.0)
        amp = amp if np.ndim(amp) == 0 else np.asarray(amp)[:n, None]
        tone *= g[:, None] * amp
        out = buf.astype(np.float64) + (tone if buf.ndim == 2 else tone[:, 0])
        return out.astype(np.float32)
