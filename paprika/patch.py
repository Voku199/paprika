"""Syntéza jednoho hlasu: obálka a Patch.

Jedna nota = jeden `voice` buffer. Oscilátory žijí v `oscillators.py`,
filtry v `filters.py`; tenhle modul je re-exportuje, takže
`from paprika.patch import Saw, LP` funguje dál.
"""

from __future__ import annotations

import functools
import inspect
import math
from dataclasses import dataclass, field

import numpy as np

from scipy import signal

from . import dsp
from .filters import BP, HP, LP, Formant, onepole_hp, onepole_lp
from .modulation import LFO, Ensemble, PitchEnv
from .oscillators import (FM, Additive, Noise, Osc, Pluck, Pulse, Saw, Sine,
                          Square, SuperSaw, Tri)

__all__ = [
    "Osc", "Sine", "Saw", "Square", "Tri", "Noise", "FM",
    "SuperSaw", "Pulse", "Pluck", "Additive",
    "ADSR", "LP", "HP", "BP", "Formant", "LFO", "PitchEnv", "Ensemble",
    "Patch", "midi_to_freq", "onepole_lp", "onepole_hp",
]

FADE_SEC = 0.002  # povinný fade na krajích hlasu, jinak to lupe
TWO_PI = 2.0 * math.pi


def midi_to_freq(pitch: float) -> float:
    return 440.0 * 2.0 ** ((float(pitch) - 69.0) / 12.0)


# --------------------------------------------------------------------------
# obálka
# --------------------------------------------------------------------------

@dataclass
class ADSR:
    """Attack / decay / sustain / release. Časy v sekundách, sustain 0-1.

    `curve="exp"` (výchozí): decay a release klesají exponenciálně a attack
    je lehce konvexní — jako akustický nástroj nebo analogový syntezátor.
    `curve="lin"` jsou přímky z v1 (typický zvuk trackerů a čipů).
    """

    a: float = 0.01
    d: float = 0.1
    s: float = 0.7
    r: float = 0.2
    curve: str = "exp"

    CURVES = ("exp", "lin")

    def __post_init__(self):
        if self.curve not in self.CURVES:
            raise ValueError(f"neznámá křivka obálky {self.curve!r}; k dispozici: "
                             f"{', '.join(self.CURVES)}")

    @staticmethod
    def _fall(x: np.ndarray) -> np.ndarray:
        """Exponenciální pokles 1 -> 0 na x ∈ [0, 1], přesně 0 v x = 1."""
        y = (np.exp(-5.0 * x.astype(np.float64)) - np.exp(-5.0)) / (1.0 - np.exp(-5.0))
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    def render(self, gate_n: int, total_n: int, sr: int) -> np.ndarray:
        env = np.zeros(total_n, dtype=np.float32)
        if total_n <= 0:
            return env
        gate_n = max(1, min(int(gate_n), total_n))
        a_n = max(1, int(self.a * sr))
        d_n = max(1, int(self.d * sr))
        r_n = max(1, int(self.r * sr))
        exp = self.curve == "exp"

        seg = min(a_n, gate_n)
        x = np.linspace(0.0, 1.0, a_n, endpoint=False, dtype=np.float32)[:seg]
        env[:seg] = 1.0 - (1.0 - x) ** 2 if exp else x
        i = seg
        if i < gate_n:
            seg = min(d_n, gate_n - i)
            if exp:
                x = np.linspace(0.0, 1.0, d_n, endpoint=False, dtype=np.float32)[:seg]
                env[i:i + seg] = self.s + (1.0 - self.s) * self._fall(x)
            else:
                env[i:i + seg] = np.linspace(1.0, self.s, seg, endpoint=False,
                                             dtype=np.float32)
            i += seg
        if i < gate_n:
            env[i:gate_n] = self.s
        level = float(env[gate_n - 1])
        tail = total_n - gate_n
        if tail > 0:
            seg = min(r_n, tail)
            if exp:
                x = np.linspace(0.0, 1.0, seg, dtype=np.float32)
                env[gate_n:gate_n + seg] = level * self._fall(x)
            else:
                env[gate_n:gate_n + seg] = np.linspace(level, 0.0, seg, dtype=np.float32)
            # za releasem je ticho, což už v poli je
        return env

    def key(self) -> tuple:
        return (self.a, self.d, self.s, self.r, self.curve)


# --------------------------------------------------------------------------
# patch
# --------------------------------------------------------------------------

RR_CENTS = (0.0, 3.0, -3.0, 1.5, -1.5, 4.0, -4.0, 2.0)


@functools.lru_cache(maxsize=None)
def _accepted(cls, method: str, names: tuple) -> frozenset:
    """Která z `names` jako keyword metoda třídy přijme (v1 rozšíření je nemají)."""
    params = inspect.signature(getattr(cls, method)).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return frozenset(names)
    return frozenset(n for n in names if n in params)


def _call(obj, method: str, args: tuple, kwargs: dict):
    """Zavolá metodu jen s keywordy, které zná — oscilátor nebo filtr napsaný
    podle v1 (`render(freq, n, sr)`, `apply(x, sr, env)`) funguje dál."""
    ok = _accepted(type(obj), method, tuple(kwargs))
    return getattr(obj, method)(*args, **{k: v for k, v in kwargs.items() if k in ok})


def _as_list(flt) -> list:
    if flt is None:
        return []
    return list(flt) if isinstance(flt, (list, tuple)) else [flt]


def _glide_curve(path, glide_time: float, n: int, sr: int) -> np.ndarray:
    """Frekvenční křivka z bodů `((t_sec, freq), ...)`.

    Cílová výška se mění skokem v časech bodů; skutečná výška za ní klouže
    exponenciálně v log-frekvenci (τ = glide_time / 4, tedy ~98 % cesty za
    `glide_time`). Počítá se jedním `lfilter`, žádný cyklus po samplech.
    """
    target = np.empty(n, dtype=np.float64)
    # jen podle času: body se stejným časem (skok z předchozí noty) musí
    # zůstat v pořadí, v jakém přišly, i když cílová výška je nižší
    pts = sorted(path, key=lambda p: p[0])
    for i, (t, f) in enumerate(pts):
        start = 0 if i == 0 else min(n, max(0, int(round(t * sr))))
        end = n if i + 1 == len(pts) else min(n, max(0, int(round(pts[i + 1][0] * sr))))
        target[start:end] = math.log(max(1.0, float(f)))
    if glide_time <= 0:
        return np.exp(target)
    tau = glide_time / 4.0
    g = 1.0 - math.exp(-1.0 / (tau * sr))
    # start z výšky prvního bodu, i když cíl hned na začátku skočí jinam
    first = math.log(max(1.0, float(pts[0][1])))
    y, _ = signal.lfilter([g], [1.0, g - 1.0], target, zi=[first * (1.0 - g)])
    return np.exp(y)


@dataclass
class Patch:
    """Nástroj: sada oscilátorů, obálka, filtr (nebo seznam filtrů), modulace.

    Modulace (vše volitelné, vše v `key()`):

    - `vibrato` — `LFO`, hloubka v půltónech,
    - `tremolo` — `LFO`, hloubka jako podíl hlasitosti,
    - `pitch_env` — `PitchEnv`, nota začne jinde a dojede na svou výšku,
    - `vel_to_cutoff` — 0..1, silnější úhoz = jasnější zvuk, ne jen hlasitější,
    - `rr` — round-robin: kolik variant (fáze + pár centů) se střídá,
    - `drift` — pomalé náhodné kolísání ladění v centech (analogový feel),
    - `ensemble` — `Ensemble`: sekce několika rozladěných, rozmazaných
      a rozprostřených kopií místo jednoho hlasu (výstup je stereo).
    """

    osc: list = field(default_factory=lambda: [Sine()])
    adsr: ADSR = field(default_factory=ADSR)
    filter: object | None = None
    gain: float = 1.0
    name: str = "patch"
    vibrato: LFO | None = None
    tremolo: LFO | None = None
    pitch_env: PitchEnv | None = None
    vel_to_cutoff: float = 0.0
    rr: int = 1
    drift: float = 0.0
    ensemble: Ensemble | None = None

    @property
    def filters(self) -> list:
        return _as_list(self.filter)

    def key(self) -> tuple:
        flt = self.filter
        if isinstance(flt, (list, tuple)):
            fkey = tuple(f.key() for f in flt)
        else:
            fkey = flt.key() if flt is not None else None
        return (
            tuple(o.key() for o in self.osc),
            self.adsr.key(),
            fkey,
            self.gain,
            self.vibrato.key() if self.vibrato is not None else None,
            self.tremolo.key() if self.tremolo is not None else None,
            self.pitch_env.key() if self.pitch_env is not None else None,
            self.vel_to_cutoff,
            int(self.rr),
            self.drift,
            self.ensemble.key() if self.ensemble is not None else None,
        )

    def voice(self, freq: float, dur: float, vel: int = 100, sr: int = 44100,
              bpm: float = 120.0, rr_index: int = 0, tone_vel: int | None = None,
              glide_path=None, glide_time: float = 0.0) -> np.ndarray:
        """Jedna nota. `dur` je znějící délka v sekundách; buffer je delší
        o release obálky. Vrací `(n,)`, nebo `(n, 2)` u stereo oscilátoru.

        `glide_path` = `((t_sec, freq), ...)` pro glide/legato frázi,
        `rr_index` vybírá round-robin variantu, `tone_vel` je velocity jen pro
        jas filtru (`vel_to_cutoff`), když se hlasitost škáluje zvlášť.
        """
        gate_n = max(1, int(round(dur * sr)))
        total_n = gate_n + max(1, int(round(self.adsr.r * sr))) + 1

        semis = np.zeros(total_n)
        moved = False
        if self.pitch_env is not None:
            semis += self.pitch_env.curve(total_n, sr)
            moved = True
        if self.vibrato is not None:
            semis += self.vibrato.curve(total_n, sr, bpm)
            moved = True
        if self.drift:
            semis += self.drift / 100.0 * dsp.lfo("random", 0.7, total_n, sr,
                                                  seed=rr_index + 17)
            moved = True
        rr = max(1, int(self.rr))
        phase = 0.0
        if rr > 1:
            semis += RR_CENTS[rr_index % rr % len(RR_CENTS)] / 100.0
            phase = (rr_index % rr * 0.37) % 1.0
            moved = True

        f = (_glide_curve(glide_path, glide_time, total_n, sr) if glide_path
             else float(freq))
        if moved:
            f = np.broadcast_to(np.asarray(f, dtype=np.float64), (total_n,)) \
                * 2.0 ** (semis / 12.0)

        env = self.adsr.render(gate_n, total_n, sr)
        amp_env = env
        if self.tremolo is not None:
            trem = 1.0 - 0.5 * (self.tremolo.depth +
                                self.tremolo.curve(total_n, sr, bpm))
            amp_env = (env * trem).astype(np.float32)

        def core(freq_curve, osc_phase):
            """Oscilátory × obálka pro jednu kopii hlasu."""
            parts = [o.amp * np.asarray(_call(o, "render", (freq_curve, total_n, sr),
                                              {"phase": osc_phase}), dtype=np.float32)
                     for o in self.osc]
            is_stereo = any(p.ndim == 2 for p in parts)
            sig = np.zeros((total_n, 2) if is_stereo else total_n, dtype=np.float32)
            for p in parts:
                sig += p[:, None] if (is_stereo and p.ndim == 1) else p
            sig *= amp_env[:, None] if is_stereo else amp_env
            return sig

        if self.ensemble is None:
            buf = core(f, phase)
        else:
            # sekce: kopie se sečtou do sterea, filtry pak běží jednou na součtu
            layout = self.ensemble.layout(rr_index)
            extra = int(round(max(0.0, self.ensemble.smear) * sr))
            buf = np.zeros((total_n + extra, 2), dtype=np.float32)
            for i, (cents, delay, pan) in enumerate(layout):
                sig = core(f * 2.0 ** (cents / 1200.0),
                           (phase + 0.37 * (rr_index * len(layout) + i)) % 1.0)
                ang = (pan + 1.0) * math.pi / 4.0
                gl, gr = math.cos(ang) * math.sqrt(2.0), math.sin(ang) * math.sqrt(2.0)
                d = int(round(delay * sr))
                if sig.ndim == 1:
                    buf[d:d + total_n, 0] += sig * gl
                    buf[d:d + total_n, 1] += sig * gr
                else:
                    buf[d:d + total_n, 0] += sig[:, 0] * min(1.0, gl)
                    buf[d:d + total_n, 1] += sig[:, 1] * min(1.0, gr)
            buf /= math.sqrt(len(layout))
            env = np.concatenate([env, np.zeros(extra, dtype=np.float32)])

        tv = vel if tone_vel is None else tone_vel
        scale = (2.0 ** (self.vel_to_cutoff * 3.0 * (tv / 127.0 - 0.75))
                 if self.vel_to_cutoff else 1.0)
        base_f = float(freq) if not glide_path else float(glide_path[-1][1])
        for flt in self.filters:
            buf = _call(flt, "apply", (buf, sr, env),
                        {"scale": scale, "bpm": bpm, "freq": base_f})

        buf = np.asarray(buf, dtype=np.float32)
        buf *= self.gain * ((max(0, min(127, int(vel))) / 127.0) ** 1.6)
        _fade(buf, sr)
        return buf


def _fade(buf: np.ndarray, sr: int, sec: float = FADE_SEC) -> None:
    """Povinný fade-in/out na krajích hlasu. Mění buffer na místě."""
    n = min(int(sec * sr), buf.shape[0] // 2)
    if n <= 0:
        return
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if buf.ndim == 2:
        ramp = ramp[:, None]
    buf[:n] *= ramp
    buf[-n:] *= ramp[::-1]
