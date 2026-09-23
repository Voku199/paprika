"""Syntéza jednoho hlasu: obálka a Patch.

Jedna nota = jeden `voice` buffer. Oscilátory žijí v `oscillators.py`,
filtry v `filters.py`; tenhle modul je re-exportuje, takže
`from paprika.patch import Saw, LP` funguje dál.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from scipy import signal

from . import dsp
from .filters import BP, HP, LP, Formant, onepole_hp, onepole_lp
from .modulation import LFO, PitchEnv
from .oscillators import (FM, Additive, Noise, Osc, Pluck, Pulse, Saw, Sine,
                          Square, SuperSaw, Tri)

__all__ = [
    "Osc", "Sine", "Saw", "Square", "Tri", "Noise", "FM",
    "SuperSaw", "Pulse", "Pluck", "Additive",
    "ADSR", "LP", "HP", "BP", "Formant", "LFO", "PitchEnv",
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
    """Attack / decay / sustain / release. Časy v sekundách, sustain 0-1."""

    a: float = 0.01
    d: float = 0.1
    s: float = 0.7
    r: float = 0.2

    def render(self, gate_n: int, total_n: int, sr: int) -> np.ndarray:
        env = np.zeros(total_n, dtype=np.float32)
        if total_n <= 0:
            return env
        gate_n = max(1, min(int(gate_n), total_n))
        a_n = max(1, int(self.a * sr))
        d_n = max(1, int(self.d * sr))
        r_n = max(1, int(self.r * sr))

        i = 0
        seg = min(a_n, gate_n)
        env[:seg] = np.linspace(0.0, 1.0, seg, endpoint=False, dtype=np.float32)
        i = seg
        if i < gate_n:
            seg = min(d_n, gate_n - i)
            env[i:i + seg] = np.linspace(1.0, self.s, seg, endpoint=False,
                                         dtype=np.float32)
            i += seg
        if i < gate_n:
            env[i:gate_n] = self.s
            i = gate_n
        level = float(env[gate_n - 1]) if gate_n > 0 else 0.0
        tail = total_n - gate_n
        if tail > 0:
            seg = min(r_n, tail)
            env[gate_n:gate_n + seg] = np.linspace(level, 0.0, seg, dtype=np.float32)
            # za releasem je ticho, což už v poli je
        return env

    def key(self) -> tuple:
        return (self.a, self.d, self.s, self.r)


# --------------------------------------------------------------------------
# patch
# --------------------------------------------------------------------------

RR_CENTS = (0.0, 3.0, -3.0, 1.5, -1.5, 4.0, -4.0, 2.0)


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
    pts = sorted(path)
    for i, (t, f) in enumerate(pts):
        start = 0 if i == 0 else min(n, max(0, int(round(t * sr))))
        end = n if i + 1 == len(pts) else min(n, max(0, int(round(pts[i + 1][0] * sr))))
        target[start:end] = math.log(max(1.0, float(f)))
    if glide_time <= 0:
        return np.exp(target)
    tau = glide_time / 4.0
    g = 1.0 - math.exp(-1.0 / (tau * sr))
    y, _ = signal.lfilter([g], [1.0, g - 1.0], target, zi=[target[0] * (1.0 - g)])
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
    - `drift` — pomalé náhodné kolísání ladění v centech (analogový feel).
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
            f = np.broadcast_to(np.asarray(f, dtype=np.float64), (total_n,))                 * 2.0 ** (semis / 12.0)

        parts = [o.amp * np.asarray(o.render(f, total_n, sr, phase=phase),
                                    dtype=np.float32) for o in self.osc]
        stereo = any(p.ndim == 2 for p in parts)
        buf = np.zeros((total_n, 2) if stereo else total_n, dtype=np.float32)
        for p in parts:
            buf += p[:, None] if (stereo and p.ndim == 1) else p

        env = self.adsr.render(gate_n, total_n, sr)
        amp_env = env
        if self.tremolo is not None:
            trem = 1.0 - 0.5 * (self.tremolo.depth +
                                self.tremolo.curve(total_n, sr, bpm))
            amp_env = (env * trem).astype(np.float32)
        buf *= amp_env[:, None] if stereo else amp_env

        tv = vel if tone_vel is None else tone_vel
        scale = (2.0 ** (self.vel_to_cutoff * 3.0 * (tv / 127.0 - 0.75))
                 if self.vel_to_cutoff else 1.0)
        base_f = float(freq) if not glide_path else float(sorted(glide_path)[-1][1])
        for flt in self.filters:
            buf = flt.apply(buf, sr, env, scale=scale, bpm=bpm, freq=base_f)

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
