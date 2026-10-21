"""Sampler: skutečné nástroje z nahrávek (SFZ + WAV).

Nezná skladbu, beaty ani pedál — dostane výšku, velocity a délku v sekundách
a vrátí stereo hlas `(n, 2)`. Rozvrh not (pedál, legato) dělá `render.py`.
WAV se čte líně přes mmap, takže velká knihovna nestojí paměť.
"""

from __future__ import annotations

import gc
import math
import os
import warnings

import numpy as np
from scipy.io import wavfile

from . import sfz as _sfz

__all__ = ["Sampler", "SampleError", "load_wav", "to_float", "clear_cache",
           "layer_weights",
           "LEGATO_OFFSET", "LEGATO_XFADE", "TAIL_CAP", "UNDAMPED_RELEASE"]

LEGATO_OFFSET = 0.08      # s: legato nota začíná za nasazením smyčce / jazyku
LEGATO_XFADE = 0.08       # s: prolnutí dvou legato not
EDGE_FADE = 0.002         # s: fade na krajích hlasu, jinak to lupne
UNDAMPED_RELEASE = 2.0    # s: struny bez dusítek (nejvyšší oktávy piana)
TAIL_CAP = 8.0            # s: strop doznění pro délku bufferu
DYN_XFADE = 24.0          # velocity: šířka prolnutí dvou vrstev při `dyn`
LOOPING = ("loop_continuous", "loop_sustain")

_WAVS: dict = {}          # cesta -> (sr, data); data je mmap, žádná kopie


class SampleError(RuntimeError):
    """Vzorek nebo SFZ chybí či je vadný — zpráva obsahuje cestu."""


def load_wav(path: str):
    """WAV -> `(sr, data (n, kanály))`. Mmap, když to formát dovolí; cache."""
    hit = _WAVS.get(path)
    if hit is not None:
        return hit
    if not os.path.isfile(path):
        raise SampleError(f"vzorek neexistuje: {path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # neznámé chunky (smpl, LIST)
        try:
            try:
                sr, data = wavfile.read(path, mmap=True)
            except ValueError:                   # 24 bit mmap neumí
                sr, data = wavfile.read(path)
        except Exception as e:  # noqa: BLE001 - scipy hází různé typy
            raise SampleError(f"vzorek se nedá přečíst: {path} ({e})") from e
    if data.ndim == 1:
        data = data[:, None]
    _WAVS[path] = (int(sr), data)
    return _WAVS[path]


def clear_cache() -> None:
    """Zavře mmapy vzorků (Windows jinak drží soubory otevřené)."""
    _WAVS.clear()
    gc.collect()


def to_float(block: np.ndarray) -> np.ndarray:
    """Blok PCM -> float32 stereo `(m, 2)` v rozsahu -1..1."""
    if block.dtype == np.int16:
        out = block.astype(np.float32) / 32768.0
    elif block.dtype == np.int32:                # i 24 bit (zarovnaný nahoru)
        out = block.astype(np.float32) / 2147483648.0
    elif block.dtype == np.uint8:
        out = (block.astype(np.float32) - 128.0) / 128.0
    else:
        out = block.astype(np.float32)
    if out.shape[1] == 1:
        return np.repeat(out, 2, axis=1)
    return out[:, :2]


def _read(src: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Hodnoty `src` v neceločíselných pozicích (4bodový Hermite). Mimo = 0."""
    out = np.zeros((pos.size, 2), dtype=np.float32)
    n = src.shape[0]
    if pos.size == 0 or n == 0:
        return out
    idx = np.floor(pos).astype(np.int64)
    ok = (idx >= 0) & (idx < n)
    if not np.any(ok):
        return out
    lo = max(0, int(idx[ok].min()) - 1)
    hi = min(n, int(idx[ok].max()) + 3)
    pad = np.zeros((2, 2), np.float32)
    block = np.concatenate([pad, to_float(src[lo:hi]), pad])
    j = idx[ok] - lo + 2
    f = (pos[ok] - idx[ok]).astype(np.float32)[:, None]
    y0, y1, y2, y3 = block[j - 1], block[j], block[j + 1], block[j + 2]
    c1 = 0.5 * (y2 - y0)
    c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)
    out[ok] = ((c3 * f + c2) * f + c1) * f + y1
    return out


def _looped(data: np.ndarray, start: int, end: int, need: int,
            sr: int) -> np.ndarray:
    """Vzorek jako float, smyčka `[start, end)` opakovaná na ≥ `need` vzorků.

    Konec smyčky se prolne (rovný výkon) s tím, co jí předchází, takže skok
    zpět na `start` je plynulý.
    """
    x = to_float(data[:end])
    length = end - start
    xf = int(min(0.25 * length, 0.5 * sr, start))
    seg = x[start:end].copy()
    if xf > 0:
        t = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]
        seg[-xf:] = (seg[-xf:] * np.cos(t * np.pi / 2)
                     + x[start - xf:start] * np.sin(t * np.pi / 2))
    reps = max(1, math.ceil((need - start) / length) + 1)
    return np.concatenate([x[:start]] + [seg] * reps)


def _fall(x: np.ndarray) -> np.ndarray:
    """Exponenciální pokles 1 -> 0 na x ∈ [0, 1], přesně 0 v x = 1."""
    x = np.clip(x, 0.0, 1.0).astype(np.float64)
    y = (np.exp(-5.0 * x) - math.exp(-5.0)) / (1.0 - math.exp(-5.0))
    return y.astype(np.float32)


class Sampler:
    """Nahraný nástroj z SFZ. Ve stopě se používá místo `Patch`.

    `legato` říká, jestli artikulace umí legato (výdrže ano, pizz ne);
    `undamped_from` je nejnižší MIDI výška bez dusítek (piano ~89);
    `gain_db` srovná hlasitost knihovny s ostatními nástroji.
    """

    sampled = True

    def __init__(self, sfz_path: str, name: str | None = None,
                 legato: bool = False, undamped_from: int | None = None,
                 gain_db: float = 0.0):
        self.path = os.path.abspath(sfz_path)
        if not os.path.isfile(self.path):
            raise SampleError(f"SFZ neexistuje: {self.path}")
        try:
            regions = _sfz.load(self.path)
        except _sfz.SfzError as e:
            raise SampleError(str(e)) from e
        # hratelné klávesou: release regiony a regiony bez kláves (pedálový
        # šum spouštěný CC, lokey=-1) sampler nehraje
        self.regions = [r for r in regions
                        if r.trigger in ("attack", "first") and r.hikey >= 0
                        and r.lokey <= r.hikey]
        if not self.regions:
            raise SampleError(f"SFZ nemá žádné hratelné regiony: {self.path}")
        self.name = name or os.path.splitext(os.path.basename(self.path))[0]
        self.legato = bool(legato)
        self.undamped_from = undamped_from
        self.gain_db = float(gain_db)
        self._ids = {id(r): i for i, r in enumerate(self.regions)}
        self._mtime = os.path.getmtime(self.path)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Sampler {self.name} regions={len(self.regions)}>"

    @property
    def range(self) -> tuple[int, int]:
        """Nejnižší a nejvyšší klávesa, kterou knihovna pokrývá."""
        return (min(r.lokey for r in self.regions),
                max(r.hikey for r in self.regions))

    @property
    def tail(self) -> float:
        """Nejdelší doznění po konci noty (pro délku bufferu)."""
        rel = max(r.release for r in self.regions)
        if self.undamped_from is not None:
            rel = max(rel, UNDAMPED_RELEASE)
        if any(r.loop_mode == "one_shot" for r in self.regions):
            rel = TAIL_CAP
        return float(min(TAIL_CAP, rel))

    def key(self) -> tuple:
        return ("sampler", self.path, self._mtime, self.undamped_from,
                self.gain_db)

    def region_id(self, region) -> int:
        return self._ids[id(region)]

    def select(self, pitch: int, vel: int, rr_index: int = 0) -> list:
        """Regiony, které pro notu zazní.

        Round robin (`seq_length` > 1) vybírá podle `rr_index`; ostatní
        sedící regiony zní spolu (vrstvy). Mimo rozsah vezme region
        s nejbližším kořenem.
        """
        vel = max(1, min(127, int(vel)))
        by_vel = [r for r in self.regions if r.lovel <= vel <= r.hivel] or self.regions
        cands = [r for r in by_vel if r.lokey <= pitch <= r.hikey]
        if not cands:
            best = min(abs(r.keycenter - pitch) for r in by_vel)
            cands = [r for r in by_vel if abs(r.keycenter - pitch) == best]
        rr = [r for r in cands if r.seq_length > 1]
        if not rr:
            return cands
        pos = rr_index % rr[0].seq_length + 1
        chosen = [r for r in rr if r.seq_position == pos]
        if not chosen:          # díra v sekvenci: vezmi dostupné cyklicky
            ordered = sorted(rr, key=lambda r: (r.seq_position, r.sample))
            chosen = [ordered[rr_index % len(ordered)]]
        return chosen + [r for r in cands if r.seq_length <= 1]

    def layers(self, pitch: int, rr_index: int = 0) -> list:
        """Velocity vrstvy pro notu: `[(lovel, hivel, regiony)]` od nejtišší.

        Pro plynulou dynamiku (`dyn`): mezi sousedními vrstvami se prolíná.
        Vrstvy se stejnými regiony se slijí do jedné.
        """
        out = []
        for lo, hi in sorted({(r.lovel, r.hivel) for r in self.regions}):
            regs = self.select(pitch, (lo + hi) // 2, rr_index)
            if out and [id(r) for r in out[-1][2]] == [id(r) for r in regs]:
                out[-1] = (out[-1][0], hi, regs)
            else:
                out.append((lo, hi, regs))
        return out

    def gain_curve(self, region, vel: np.ndarray) -> np.ndarray:
        """`gain` po samplech: velocity jako pole (plynulá dynamika)."""
        v = np.clip(np.asarray(vel, dtype=np.float64), 1.0, 127.0) / 127.0
        db = (region.volume + self.gain_db
              + region.veltrack / 100.0 * 40.0 * np.log10(v))
        return (10.0 ** (db / 20.0)).astype(np.float32)

    def gain(self, region, vel: int) -> float:
        """Zisk noty: SFZ zákon velocity (`amp_veltrack`) + hlasitost."""
        v = max(1, min(127, int(vel))) / 127.0
        db = (region.volume + self.gain_db
              + region.veltrack / 100.0 * 40.0 * math.log10(v))
        return float(10.0 ** (db / 20.0))

    def voice(self, region, pitch: int, dur: float, sr: int,
              legato_in: bool = False, legato_out: bool = False) -> np.ndarray:
        """Jedna nota z regionu -> stereo `(n, 2)` float32, bez velocity zisku.

        Posun výšky a převzorkování na `sr` jsou jeden poměr čtení. Po gate
        (`dur`) přijde release; `one_shot` přehraje vzorek celý.
        """
        src_sr, data = load_wav(region.sample)
        rate = (2.0 ** ((pitch - region.keycenter + region.transpose) / 12.0
                        + region.tune / 1200.0) * src_sr / sr)
        start = region.offset + (int(LEGATO_OFFSET * src_sr) if legato_in else 0)
        start = min(start, max(0, data.shape[0] - 1))
        one_shot = region.loop_mode == "one_shot"
        looping = (region.loop_mode in LOOPING
                   and region.loop_end > region.loop_start)
        gate_n = max(1, int(round(dur * sr)))
        if legato_out:
            rel = LEGATO_XFADE
        elif self.undamped_from is not None and pitch >= self.undamped_from:
            rel = max(region.release, UNDAMPED_RELEASE)
        else:
            rel = region.release
        rel_n = max(1, int(round(rel * sr)))
        avail = int((data.shape[0] - start) / rate)
        if one_shot:
            n_out = avail
        else:
            n_out = gate_n + rel_n
            if not looping:
                n_out = min(n_out, avail)
        n_out = max(1, n_out)
        pos = start + np.arange(n_out, dtype=np.float64) * rate
        src = data
        if looping and pos[-1] >= region.loop_end - 1:
            end = min(region.loop_end, data.shape[0])
            src = _looped(data, min(region.loop_start, end - 1), end,
                          int(pos[-1]) + 4, src_sr)
        out = _read(src, pos)
        out *= self._envelope(n_out, gate_n, rel_n, sr, region.attack, one_shot,
                              legato_in, legato_out)[:, None]
        if region.pan:
            p = max(-1.0, min(1.0, region.pan / 100.0))
            out[:, 0] *= min(1.0, 1.0 - p)
            out[:, 1] *= min(1.0, 1.0 + p)
        return out

    @staticmethod
    def _envelope(n: int, gate_n: int, rel_n: int, sr: int, attack: float,
                  one_shot: bool, legato_in: bool, legato_out: bool) -> np.ndarray:
        env = np.ones(n, dtype=np.float32)
        a_n = min(n, int(max(attack, LEGATO_XFADE if legato_in else 0.0,
                             EDGE_FADE) * sr))
        if a_n > 0:
            x = np.linspace(0.0, 1.0, a_n, endpoint=False, dtype=np.float32)
            env[:a_n] = np.sin(x * np.pi / 2) if legato_in else x
        if not one_shot and n > gate_n:
            x = np.arange(n - gate_n, dtype=np.float32) / rel_n
            fall = (np.cos(np.clip(x, 0.0, 1.0) * np.pi / 2) if legato_out
                    else _fall(x))
            env[gate_n:] *= fall
        e_n = min(n, int(EDGE_FADE * sr))
        if e_n > 0:
            env[n - e_n:] *= np.linspace(1.0, 0.0, e_n, dtype=np.float32)
        return env


def layer_weights(vel: np.ndarray, bounds) -> np.ndarray:
    """Váhy vrstev `(počet vrstev, n)` pro velocity v čase.

    Uvnitř vrstvy zní jen ona; kolem hranice dvou vrstev (±`DYN_XFADE`/2,
    u úzkých vrstev méně) se prolnou equal-power. Pod nejnižší a nad
    nejvyšší vrstvou platí krajní vrstva.
    """
    e = np.asarray(vel, dtype=np.float64)
    k = len(bounds)
    w = np.zeros((k, e.size), dtype=np.float64)
    for i, (lo, hi) in enumerate(bounds):
        low = -np.inf if i == 0 else lo - 0.5
        high = np.inf if i == k - 1 else hi + 0.5
        w[i] = (e >= low) & (e < high)
    for i in range(k - 1):
        (lo0, hi0), (lo1, hi1) = bounds[i], bounds[i + 1]
        z = min(DYN_XFADE / 2, (hi0 - lo0 + 1) / 2, (hi1 - lo1 + 1) / 2)
        edge = (hi0 + lo1) / 2.0
        zone = (e >= edge - z) & (e < edge + z)
        if not zone.any():
            continue
        u = (e[zone] - (edge - z)) / (2 * z)
        w[i, zone] = np.cos(u * np.pi / 2)
        w[i + 1, zone] = np.sin(u * np.pi / 2)
    return w.astype(np.float32)
