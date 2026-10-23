"""Dynamické efekty: Compressor, dynamická EQ a rytmický Gate.

Obojí počítá křivku zisku vektorově (detektor přes jednopólový filtr,
žádný cyklus po samplech) a násobí jí buffer. Stereo je linkované — oba
kanály dostanou stejný zisk, takže se stereo obraz nerozjede.
"""

from __future__ import annotations

import math

import numpy as np

from . import dsp
from .fx import Fx, FxContext, _col

__all__ = ["Compressor", "DynEQ", "Gate"]


class Compressor(Fx):
    """Feed-forward kompresor se soft knee.

    `threshold` v dBFS, `ratio` (4 = nad prahem jen čtvrtina nárůstu),
    `attack`/`release` v sekundách, `makeup` v dB. Na busu bicích drží
    groove pohromadě, na masteru „lepí“ mix.
    """

    STEREO = True

    AUTOMATABLE = ("threshold",)

    def __init__(self, threshold: float = -18.0, ratio: float = 4.0,
                 attack: float = 0.01, release: float = 0.15, makeup: float = 0.0,
                 knee: float = 6.0, name: str | None = None):
        self.threshold = float(threshold)
        self.ratio = max(1.0, float(ratio))
        self.attack = max(1e-4, float(attack))
        self.release = max(1e-3, float(release))
        self.makeup = float(makeup)
        self.knee = max(0.0, float(knee))
        self.name = name

    def gain(self, buf: np.ndarray, sr: int) -> np.ndarray:
        """Křivka zisku (lineární) pro každý sample."""
        det = np.abs(buf).max(axis=1) if buf.ndim == 2 else np.abs(buf)
        env = dsp.onepole(det, 1.0 / (2.0 * math.pi * self.release), sr)
        level = 20.0 * np.log10(np.maximum(env.astype(np.float64), 1e-9))
        over = level - np.asarray(self.value("threshold"), dtype=np.float64)
        k = self.knee
        if k > 0:
            over_k = np.where(over <= -k / 2, 0.0,
                              np.where(over >= k / 2, over, (over + k / 2) ** 2 / (2 * k)))
        else:
            over_k = np.maximum(over, 0.0)
        gain_db = -over_k * (1.0 - 1.0 / self.ratio) + self.makeup
        g = 10.0 ** (gain_db / 20.0)
        return dsp.onepole(g, 1.0 / (2.0 * math.pi * self.attack), sr)

    def process(self, buf, ctx: FxContext):
        buf = np.asarray(buf, dtype=np.float32)
        if buf.shape[0] == 0:
            return buf
        return (buf * _col(self.gain(buf, ctx.sr), buf)).astype(np.float32)


class DynEQ(Fx):
    """Dynamická EQ: zvon na `freq`, který stáhne pásmo jen ve chvíli, kdy
    je v něm moc energie (ostré nádechy, sykavky); jinak nedělá nic.

    Obálka pásma (`freq`, `q`) se měří v dBFS jako u kompresoru: nad
    `threshold` ubírá s `ratio`, nejvýš o `range_db`. `source="dech"` řídí
    EQ pásmo jiné stopy — hudba uhne ve 2–5 kHz jen tehdy, když zní dech.
    """

    STEREO = True
    AUTOMATABLE = ("threshold",)

    def __init__(self, freq: float = 5000.0, q: float = 1.5,
                 threshold: float = -30.0, ratio: float = 3.0,
                 range_db: float = 12.0, attack: float = 0.005,
                 release: float = 0.08, source: str | None = None,
                 name: str | None = None):
        self.freq = float(freq)
        self.q = max(0.1, float(q))
        self.threshold = float(threshold)
        self.ratio = max(1.0, float(ratio))
        self.range_db = max(0.0, float(range_db))
        self.attack = max(1e-4, float(attack))
        self.release = max(1e-3, float(release))
        self.source = None if source is None else str(source)
        self.name = name

    @property
    def needs_tracks(self) -> bool:
        return self.source is not None

    def gain_db(self, key: np.ndarray, sr: int) -> np.ndarray:
        """Zisk zvonu v dB (0 nebo záporný) pro každý sample klíče."""
        k = np.asarray(key, dtype=np.float64)
        k = k.mean(axis=1) if k.ndim == 2 else k
        band = dsp.filt(k, "bp", self.freq, sr, q=self.q).astype(np.float64)
        env = dsp.onepole(np.abs(band), 1.0 / (2.0 * math.pi * self.release), sr)
        level = 20.0 * np.log10(np.maximum(env.astype(np.float64), 1e-9))
        over = level - np.asarray(self.value("threshold"), dtype=np.float64)
        cut = np.clip(over * (1.0 - 1.0 / self.ratio), 0.0, self.range_db)
        return -dsp.onepole(cut, 1.0 / (2.0 * math.pi * self.attack), sr).astype(np.float64)

    def process(self, buf, ctx: FxContext):
        buf = np.asarray(buf, dtype=np.float32)
        n = buf.shape[0]
        if n == 0:
            return buf
        key = buf
        if self.source is not None:
            tracks = ctx.track_buffers or {}
            if self.source not in tracks:
                raise KeyError(
                    f"DynEQ: stopa {self.source!r} neexistuje nebo ještě není "
                    f"vyrenderovaná (k dispozici: {', '.join(sorted(tracks)) or '—'})")
            src = np.asarray(tracks[self.source], dtype=np.float32)[:n]
            key = np.zeros((n,) + src.shape[1:], dtype=np.float32)
            key[:src.shape[0]] = src
        g = self.gain_db(key, ctx.sr)
        if float(g.min()) > -0.01:
            return buf
        return dsp.filt_varying(buf, "peak", self.freq, ctx.sr, q=self.q,
                                gain_db=g, block=128)


class Gate(Fx):
    """Rytmický gate (trance gate / stutter) synchronní s tempem.

    `pattern`: `x` otevřeno, `.` nebo `_` zavřeno; opakuje se po krocích
    `rate` (zlomek celé noty). `depth` 1 = úplné ticho, 0.5 = na půl.
    Počítá se od začátku skladby, takže sedí i při renderu výřezu.
    """

    STEREO = True

    OPEN = "xX"
    CLOSED = "._"
    AUTOMATABLE = ("depth",)

    def __init__(self, pattern: str = "x.x.xx..", rate: float = 1 / 16,
                 depth: float = 1.0, smooth: float = 0.004, name: str | None = None):
        bad = [c for c in pattern if c not in self.OPEN + self.CLOSED]
        if bad or not pattern:
            raise ValueError(f"Gate pattern {pattern!r}: povolené znaky jsou "
                             f"'x' (otevřeno) a '.' nebo '_' (zavřeno)")
        self.pattern = pattern
        self.rate = float(rate)
        self.depth = float(np.clip(depth, 0.0, 1.0))
        self.smooth = max(1e-4, float(smooth))
        self.name = name

    def process(self, buf, ctx: FxContext):
        buf = np.asarray(buf, dtype=np.float32)
        n = buf.shape[0]
        if n == 0:
            return buf
        step = self.rate * ctx.beats_per_bar
        beats = ctx.beat_offset + np.arange(n, dtype=np.float64) / ctx.sr * (ctx.bpm / 60.0)
        mask = np.array([1.0 if c in self.OPEN else 0.0 for c in self.pattern])
        idx = np.floor(beats / step + 1e-9).astype(np.int64) % len(self.pattern)
        opened = dsp.onepole(mask[idx], 1.0 / (2.0 * math.pi * self.smooth), ctx.sr)
        depth = np.asarray(self.value("depth"), dtype=np.float32)
        gain = 1.0 - depth * (1.0 - opened)
        return (buf * _col(gain, buf)).astype(np.float32)
