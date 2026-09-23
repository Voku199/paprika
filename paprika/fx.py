"""Efekty na úrovni stopy: Reverb, Delay, LowPass, HighPass, Saturate, Sidechain.

Každý efekt má `process(buf, ctx)` a `tail(ctx)` — kolik sekund navíc si
`render()` musí nechat v bufferu, aby se ocásek vešel.

Buffer stopy je stereo `(n, 2)`. Efekt, který stereo neřeší, implementuje jen
`process_mono(x, ctx)` a základní třída ho zavolá na každý kanál zvlášť.
Efekt, který umí stereo (reverb, delay), přepíše `process` a přijme `(n,)`
i `(n, 2)`.

Automatizace: efekt vyjmenuje parametry v `AUTOMATABLE`, render mu před
zpracováním předá křivky přes `set_curve()` a efekt si hodnotu čte přes
`self.value(param)` — dostane skalár, nebo pole po samplech.

Zpětné vazby (comb, allpass, delay) jsou počítané po blocích délky zpoždění,
takže i přes rekurzi jde o vektorizované numpy operace, ne cyklus po samplech.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import dsp
from .patch import onepole_hp, onepole_lp

__all__ = ["Fx", "FxContext", "Reverb", "Delay", "LowPass", "HighPass",
           "Filter", "EQ", "Saturate", "Sidechain"]


@dataclass
class FxContext:
    """Co efekt potřebuje vědět o okolí."""

    sr: int = 44100
    bpm: float = 120.0
    beats_per_bar: float = 4.0
    track_buffers: dict | None = None
    beat_offset: float = 0.0

    def beats_to_sec(self, beats: float) -> float:
        return float(beats) * 60.0 / float(self.bpm)


def _col(v, buf: np.ndarray):
    """Hodnotu parametru (skalár nebo pole po samplech) připraví k násobení bufferu."""
    if np.isscalar(v) or np.ndim(v) == 0:
        return float(v)
    v = np.asarray(v, dtype=np.float32)[:buf.shape[0]]
    return v[:, None] if buf.ndim == 2 else v


class Fx:
    """Základ efektu.

    `STEREO = True` říká, že `process` umí `(n, 2)`. Efekt napsaný podle v1
    (přepsaný `process`, bez flagu) dostane v renderu každý kanál zvlášť.
    """

    AUTOMATABLE: tuple = ()
    STEREO = False

    def process(self, buf: np.ndarray, ctx: FxContext) -> np.ndarray:
        buf = np.asarray(buf, dtype=np.float32)
        if buf.ndim == 1:
            return np.asarray(self.process_mono(buf, ctx), dtype=np.float32)
        chans = [np.asarray(self.process_mono(buf[:, c], ctx), dtype=np.float32)
                 for c in range(buf.shape[1])]
        return np.stack(chans, axis=1)

    def process_mono(self, x: np.ndarray, ctx: FxContext) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__}: implementuj process_mono(x, ctx) "
            f"nebo process(buf, ctx)")

    def tail(self, ctx: FxContext) -> float:
        return 0.0

    @property
    def needs_tracks(self) -> bool:
        return False

    # -- automatizace -----------------------------------------------------
    @property
    def label(self) -> str:
        """Jméno pro automatizaci: `name=` z konstruktoru, jinak jméno třídy."""
        return getattr(self, "name", None) or type(self).__name__.lower()

    def set_curve(self, param: str, arr) -> None:
        self.__dict__.setdefault("_curves", {})[param] = np.asarray(arr)

    def clear_curves(self) -> None:
        self.__dict__.pop("_curves", None)

    def value(self, param: str):
        curves = self.__dict__.get("_curves")
        if curves and param in curves:
            return curves[param]
        return getattr(self, param)


def apply_fx(effect, buf: np.ndarray, ctx: FxContext) -> np.ndarray:
    """Prožene stereo buffer efektem; v1 efekt s vlastním `process` jede po kanálech."""
    if getattr(effect, "STEREO", False) or type(effect).process is Fx.process:
        return effect.process(buf, ctx)
    n = buf.shape[0]
    chans = []
    for c in range(buf.shape[1]):
        y = np.asarray(effect.process(np.ascontiguousarray(buf[:, c]), ctx),
                       dtype=np.float32).reshape(-1)[:n]
        chans.append(np.pad(y, (0, n - y.size)))
    return np.stack(chans, axis=1)


def _feedback_delay(x: np.ndarray, d: int, fb: float) -> np.ndarray:
    """y[n] = x[n] + fb * y[n-d], počítané po blocích délky d."""
    if d <= 0:
        return x.copy()
    y = x.astype(np.float64).copy()
    for start in range(d, y.shape[0], d):
        end = min(start + d, y.shape[0])
        y[start:end] += fb * y[start - d:start - d + (end - start)]
    return y


def _allpass(x: np.ndarray, d: int, g: float) -> np.ndarray:
    """y[n] = -x[n] + x[n-d] + g*y[n-d]."""
    if d <= 0:
        return x.copy()
    xd = np.concatenate([np.zeros(d), x.astype(np.float64)])[:x.size]
    y = (-x.astype(np.float64) + xd)
    for start in range(d, y.size, d):
        end = min(start + d, y.size)
        y[start:end] += g * y[start - d:start - d + (end - start)]
    return y


class Reverb(Fx):
    """Dozvuk místnosti. `room` 0..1 určuje velikost, `damp` tlumení výšek,
    `width` 0 = mono dozvuk, 1 = plně rozevřený.

    `algo="conv"` (výchozí): konvoluce se syntetickou stereo impulzní odezvou
    — přirozený prostor. `algo="freeverb"`: původní Schroeder/Freeverb z v1
    (8 combů + 4 allpassy), kovovější a „retro“. Oba vstup sečtou do mono
    a vrátí stereo dozvuk; hlasitost mokrého signálu je srovnaná.
    """

    STEREO = True

    COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
    ALLPASS = (556, 441, 341, 225)
    STEREO_SPREAD = 23
    ALGOS = ("conv", "freeverb")
    CONV_WET = 10.0 ** (-8.0 / 20.0)   # srovná mokrý signál s Freeverbem
    AUTOMATABLE = ("mix",)

    def __init__(self, room: float = 0.6, mix: float = 0.3, damp: float = 0.4,
                 width: float = 1.0, name: str | None = None, algo: str = "conv"):
        if algo not in self.ALGOS:
            raise ValueError(f"neznámý algoritmus reverbu {algo!r}; k dispozici: "
                             f"{', '.join(self.ALGOS)}")
        self.room = float(np.clip(room, 0.0, 1.0))
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.damp = float(np.clip(damp, 0.0, 1.0))
        self.width = float(np.clip(width, 0.0, 1.0))
        self.name = name
        self.algo = algo

    def _feedback(self) -> float:
        return 0.70 + 0.28 * self.room

    def _conv_decay(self) -> float:
        return 0.3 + 4.7 * self.room ** 2

    def _conv_predelay(self) -> float:
        return 0.01 + 0.03 * self.room

    def tail(self, ctx: FxContext) -> float:
        if self.algo == "conv":
            return self._conv_decay() * 1.1 + self._conv_predelay()
        # doba, za kterou feedback klesne pod -60 dB
        fb = self._feedback()
        loop_sec = (max(self.COMBS) + self.STEREO_SPREAD) / 44100.0
        n_loops = np.log(1e-3) / np.log(max(fb, 1e-6))
        return float(min(12.0, loop_sec * n_loops + 0.1))

    def _wet(self, x: np.ndarray, sr: int, spread: int) -> np.ndarray:
        scale = sr / 44100.0
        fb = self._feedback()
        wet = np.zeros(x.shape[0], dtype=np.float64)
        for d in self.COMBS:
            wet += _feedback_delay(x, int((d + spread) * scale), fb)
        wet /= len(self.COMBS)
        if self.damp > 0:
            cutoff = 12000.0 * (1.0 - self.damp) + 1200.0 * self.damp
            wet = onepole_lp(wet.astype(np.float32), cutoff, sr, poles=1)
        for d in self.ALLPASS:
            wet = _allpass(np.asarray(wet, dtype=np.float64),
                           int((d + spread) * scale), 0.5)
        return np.asarray(wet, dtype=np.float32) * 0.35

    def _conv_wet(self, src: np.ndarray, sr: int):
        ir = dsp.cached_ir(self._conv_decay(), sr, self._conv_predelay(),
                           self.damp, 0.3, 0)
        wl = dsp.convolve(src, ir[:, 0]) * self.CONV_WET
        wr = dsp.convolve(src, ir[:, 1]) * self.CONV_WET
        return wl.astype(np.float32), wr.astype(np.float32)

    def process(self, buf, ctx):
        buf = np.asarray(buf, dtype=np.float32)
        mix = self.value("mix")
        if (np.isscalar(mix) and mix <= 0.0) or buf.shape[0] == 0:
            return buf
        m = _col(mix, buf)
        if buf.ndim == 1:
            wet = (self._conv_wet(buf, ctx.sr)[0] if self.algo == "conv"
                   else self._wet(buf, ctx.sr, 0))
            return ((1.0 - m) * buf + m * wet).astype(np.float32)
        src = buf.mean(axis=1)
        if self.algo == "conv":
            wl, wr = self._conv_wet(src, ctx.sr)
        else:
            wl = self._wet(src, ctx.sr, 0)
            wr = self._wet(src, ctx.sr, self.STEREO_SPREAD)
        w1 = self.width / 2.0 + 0.5
        w2 = (1.0 - self.width) / 2.0
        wet = np.stack([wl * w1 + wr * w2, wr * w1 + wl * w2], axis=1)
        return ((1.0 - m) * buf + m * wet).astype(np.float32)


class Delay(Fx):
    """Tempo-synchronní echo. `time` je zlomek celé noty (1/8 = osminka).

    `tone` = cutoff lowpassu ve zpětné vazbě: každé další echo je tmavší,
    jako páskové echo. `ping_pong=True` střídá echa vlevo a vpravo.
    """

    STEREO = True

    AUTOMATABLE = ("mix",)

    def __init__(self, time: float = 0.25, fb: float = 0.3, mix: float = 0.25,
                 ping_pong: bool = False, tone: float | None = None,
                 name: str | None = None):
        self.time = float(time)
        self.fb = float(np.clip(fb, 0.0, 0.95))
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.ping_pong = bool(ping_pong)
        self.tone = float(tone) if tone else None
        self.name = name

    def _sec(self, ctx: FxContext) -> float:
        return ctx.beats_to_sec(self.time * ctx.beats_per_bar)

    def tail(self, ctx: FxContext) -> float:
        if self.fb <= 0:
            return self._sec(ctx)
        n_loops = np.log(1e-3) / np.log(max(self.fb, 1e-6))
        return float(min(15.0, self._sec(ctx) * (n_loops + 1)))

    def _repeats(self, src: np.ndarray, d: int, sr: int):
        """Jednotlivá echa: (zpoždění ve vzorcích, signál už zeslabený a ztmavený).

        Generátor — v paměti je vždy jen jedno echo, ne všechna naráz."""
        n = src.shape[0]
        if self.fb > 0:
            count = int(math.ceil(math.log(1e-3) / math.log(self.fb))) + 1
        else:
            count = 1
        count = max(1, min(count, n // max(1, d)))
        cur = src.astype(np.float64)
        for k in range(1, count + 1):
            if self.tone:
                cur = dsp.onepole(cur, self.tone, sr).astype(np.float64)
            yield k * d, cur * self.fb ** (k - 1)

    def process(self, buf, ctx):
        buf = np.asarray(buf, dtype=np.float32)
        mix = self.value("mix")
        if (np.isscalar(mix) and mix <= 0.0) or buf.shape[0] == 0:
            return buf
        d = int(round(self._sec(ctx) * ctx.sr))
        if d <= 0:
            return buf
        n = buf.shape[0]
        m = _col(mix, buf)
        stereo_pp = self.ping_pong and buf.ndim == 2
        if not self.tone and not stereo_pp:
            echo = _feedback_delay(buf, d, self.fb)
            pad = np.zeros((d,) + buf.shape[1:], dtype=np.float64)
            echo = np.concatenate([pad, echo])[:n]
            return (buf + m * echo).astype(np.float32)
        src = buf.mean(axis=1) if stereo_pp else buf
        echo = np.zeros(buf.shape, dtype=np.float64)
        for k, (delay, sig) in enumerate(self._repeats(src, d, ctx.sr)):
            if delay >= n:
                break
            part = sig[:n - delay]
            if stereo_pp:
                echo[delay:, k % 2] += part
            else:
                echo[delay:] += part
        return (buf + m * echo).astype(np.float32)


def _filter_process(fx, buf, ctx, kind: str):
    buf = np.asarray(buf, dtype=np.float32)
    cut = fx.value("cutoff")
    res = fx.value("res")
    static_res = np.isscalar(res) or np.ndim(res) == 0
    if fx.poles <= 1 and static_res and float(res) <= 0:
        low = dsp.onepole(buf, cut, ctx.sr, poles=1)
        return low if kind == "lp" else (buf - low).astype(np.float32)
    q = dsp.res_to_q(res)
    y = dsp.filt_varying(buf, kind, cut, ctx.sr, q=q, stages=max(1, fx.poles // 2))
    if kind not in ("lp", "hp"):
        return y
    comp = (dsp.Q_MIN / np.asarray(q)) ** 0.35
    return (y * _col(comp, y)).astype(np.float32)


class LowPass(Fx):
    """Lowpass na celé stopě. `res` 0..1 přidá rezonanci, `cutoff` jde
    automatizovat (filtr sweep do dropu)."""

    STEREO = True

    AUTOMATABLE = ("cutoff", "res")

    def __init__(self, cutoff: float = 2000.0, poles: int = 2, res: float = 0.0,
                 name: str | None = None):
        self.cutoff = float(cutoff)
        self.poles = int(poles)
        self.res = float(res)
        self.name = name

    def process(self, buf, ctx):
        return _filter_process(self, buf, ctx, "lp")


class HighPass(Fx):
    """Highpass na celé stopě — nejlevnější způsob, jak uvolnit basy v mixu."""

    STEREO = True

    AUTOMATABLE = ("cutoff", "res")

    def __init__(self, cutoff: float = 200.0, poles: int = 2, res: float = 0.0,
                 name: str | None = None):
        self.cutoff = float(cutoff)
        self.poles = int(poles)
        self.res = float(res)
        self.name = name

    def process(self, buf, ctx):
        return _filter_process(self, buf, ctx, "hp")


class Filter(Fx):
    """Filtr s volitelným typem: `lp`, `hp`, `bp` (pásmo) nebo `notch` (zářez).

    `bp` s vyšší rezonancí a automatizovaným cutoffem je klasický „wah“
    nebo telefonní hlas; `notch` vyřízne jednu frekvenci.
    """

    STEREO = True

    MODES = ("lp", "hp", "bp", "notch")
    AUTOMATABLE = ("cutoff", "res")

    def __init__(self, mode: str = "lp", cutoff: float = 1000.0, res: float = 0.0,
                 name: str | None = None):
        if mode not in self.MODES:
            raise ValueError(f"neznámý typ filtru {mode!r}; k dispozici: "
                             f"{', '.join(self.MODES)}")
        self.mode = mode
        self.cutoff = float(cutoff)
        self.res = float(res)
        self.poles = 2
        self.name = name

    def process(self, buf, ctx):
        return _filter_process(self, buf, ctx, self.mode)


class EQ(Fx):
    """Ekvalizér: `low=(freq, dB)` shelf, `mid=[(freq, dB, q), ...]` zvony,
    `high=(freq, dB)` shelf. Kladné dB zesílí, záporné utlumí."""

    STEREO = True

    def __init__(self, low=None, mid=(), high=None, name: str | None = None):
        self.low = tuple(low) if low else None
        self.mid = [tuple(m) for m in (mid or ())]
        self.high = tuple(high) if high else None
        self.name = name

    def process(self, buf, ctx):
        y = np.asarray(buf, dtype=np.float32)
        if self.low:
            y = dsp.filt(y, "lowshelf", self.low[0], ctx.sr, gain_db=self.low[1])
        for freq, gain, *q in self.mid:
            y = dsp.filt(y, "peak", freq, ctx.sr, q=q[0] if q else 1.0, gain_db=gain)
        if self.high:
            y = dsp.filt(y, "highshelf", self.high[0], ctx.sr, gain_db=self.high[1])
        return y


class Saturate(Fx):
    """Měkké zkreslení: tanh normalizovaný tak, aby nezvedal hlasitost."""

    STEREO = True

    AUTOMATABLE = ("drive", "mix")

    def __init__(self, drive: float = 2.0, mix: float = 1.0,
                 name: str | None = None):
        self.drive = max(1e-3, float(drive))
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.name = name

    def process(self, buf, ctx):
        buf = np.asarray(buf, dtype=np.float32)
        drive = np.maximum(1e-3, _col(self.value("drive"), buf))
        m = _col(self.value("mix"), buf)
        wet = np.tanh(buf * drive) / np.tanh(drive)
        return ((1.0 - m) * buf + m * wet).astype(np.float32)


class Sidechain(Fx):
    """Klasické pumpování: obálka jiné stopy (nebo busu) stahuje hlasitost téhle."""

    STEREO = True

    def __init__(self, source: str, amount: float = 0.6, attack: float = 0.005,
                 release: float = 0.18, name: str | None = None):
        self.source = str(source)
        self.amount = float(np.clip(amount, 0.0, 1.0))
        self.attack = float(attack)
        self.release = float(release)
        self.name = name

    @property
    def needs_tracks(self) -> bool:
        return True

    def process(self, buf, ctx):
        buf = np.asarray(buf, dtype=np.float32)
        tracks = ctx.track_buffers or {}
        src = tracks.get(self.source)
        if src is None:
            raise KeyError(
                f"Sidechain: stopa {self.source!r} neexistuje nebo ještě není "
                f"vyrenderovaná (k dispozici: {', '.join(sorted(tracks)) or '—'})")
        n = buf.shape[0]
        src = np.asarray(src, dtype=np.float32)
        env = np.abs(src).mean(axis=1) if src.ndim == 2 else np.abs(src)
        env = env[:n]
        if env.size < n:
            env = np.concatenate([env, np.zeros(n - env.size, dtype=np.float32)])
        env = onepole_lp(env, 1.0 / max(self.release, 1e-3) / 6.28, ctx.sr, poles=1)
        peak = float(env.max()) if env.size else 0.0
        if peak <= 1e-9:
            return buf
        gain = 1.0 - self.amount * (env / peak)
        atk = onepole_lp(gain.astype(np.float32),
                         1.0 / max(self.attack, 1e-3) / 6.28, ctx.sr, poles=1)
        return (buf * _col(atk, buf)).astype(np.float32)
