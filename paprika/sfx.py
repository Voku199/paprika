"""Cinematic SFX: riser, impact, braam, subdrop, reverse swell, whoosh.

Trailerové prvky, na kterých stojí build-upy a dropy. Všechno syntetizované,
stereo, s délkou v beatech (generátor dostane BPM), takže riser skončí
přesně na začátku taktu, na který míří.

Generátor vrací `(m, 2)` float64 při velocity 127; render a cache řeší
`render_sfx`, časování `SfxTrack` v `song.py`.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from . import dsp
from .drums import DrumHit
from .filters import Formant
from .oscillators import SuperSaw
from .patch import ADSR, midi_to_freq

__all__ = ["SfxHit", "SFX_NAMES", "SFX_CHARS", "RISER_KINDS", "render_sfx"]

TWO_PI = 2.0 * math.pi
SFX_NAMES = ("riser", "impact", "braam", "subdrop", "reverse_swell", "whoosh")
SFX_CHARS = {"riser": "R", "impact": "I", "braam": "B", "subdrop": "v",
             "reverse_swell": "<", "whoosh": "w"}
RISER_KINDS = ("noise", "tone", "both")
PEAK = 0.9


class SfxHit(DrumHit):
    """Jeden SFX: jako úder bicích, navíc parametry generátoru a délka v beatech."""

    __slots__ = ("params", "dur_beats")

    def __init__(self, name: str, start_beat: float, velocity: int,
                 params: tuple = (), dur_beats: float = 0.0):
        super().__init__(name, start_beat, velocity)
        self.params = tuple(params)
        self.dur_beats = float(dur_beats)


# --------------------------------------------------------------------------
# pomocné
# --------------------------------------------------------------------------

def _t(n: int, sr: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / sr


def _stereo(x: np.ndarray) -> np.ndarray:
    return np.repeat(x[:, None], 2, axis=1) if x.ndim == 1 else x


def _edges(x: np.ndarray, sr: int, sec: float = 0.005) -> np.ndarray:
    n = min(int(sec * sr), x.shape[0] // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n)[:, None]
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def _finish(x: np.ndarray, sr: int, peak: float = PEAK) -> np.ndarray:
    x = _stereo(np.asarray(x, dtype=np.float64))
    m = float(np.abs(x).max()) if x.size else 0.0
    if m > 0:
        x = x / m * peak
    return _edges(x, sr)


def _noise2(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, (n, 2))


# --------------------------------------------------------------------------
# generátory
# --------------------------------------------------------------------------

def _riser(sr, bpm, kind, beats):
    if kind not in RISER_KINDS:
        raise ValueError(f"neznámý druh riseru {kind!r}; k dispozici: "
                         f"{', '.join(RISER_KINDS)}")
    n = max(1, int(float(beats) * 60.0 / bpm * sr))
    u = np.linspace(0.0, 1.0, n)
    out = np.zeros((n, 2))
    if kind in ("noise", "both"):
        sweep = np.geomspace(300.0, 9000.0, n)
        out += dsp.filt_varying(_noise2(n, 11), "bp", sweep, sr, q=2.0) * (u ** 2)[:, None]
    if kind in ("tone", "both"):
        tone = SuperSaw(voices=5, detune=0.3, spread=0.8).render(
            np.geomspace(110.0, 880.0, n), n, sr)
        out += 0.5 * tone * (u ** 2)[:, None] / max(1e-9, float(np.abs(tone).max()))
    return _finish(out, sr, 0.85)


def _impact(sr, bpm, size):
    size = max(0.2, float(size))
    n = int((3.0 * size + 0.2) * sr)
    t = _t(n, sr)
    f = 40.0 + 50.0 * np.exp(-t / 0.016)
    sub = np.sin(TWO_PI * np.cumsum(f) / sr) * np.exp(-t / (1.2 * size))
    crack = dsp.filt(np.random.default_rng(21).uniform(-1, 1, n), "hp", 1000.0, sr) \
        * np.exp(-t / 0.06)
    burst = np.random.default_rng(22).uniform(-1, 1, n) * np.exp(-t / 0.05)
    ir = dsp.synth_ir(2.5 * size, sr, predelay=0.0, damp=0.7, seed=23)
    tail = np.stack([signal.fftconvolve(burst, ir[:, c])[:n] for c in range(2)], axis=1)
    x = _stereo(sub) + 0.5 * _stereo(crack) + 0.6 * tail / max(1e-9, np.abs(tail).max())
    return _finish(np.tanh(1.5 * x) / np.tanh(1.5), sr)


def _braam(sr, bpm, midi, beats):
    f = midi_to_freq(midi)
    gate = max(0.1, float(beats) * 60.0 / bpm)
    adsr = ADSR(0.06, 0.4, 0.8, 0.9)
    gate_n = int(gate * sr)
    n = gate_n + int(adsr.r * sr) + 1
    x = np.zeros((n, 2))
    for ratio, amp in ((1.0, 1.0), (0.5, 0.8), (1.5, 0.45)):
        x += amp * SuperSaw(voices=5, detune=0.15, spread=0.7, seed=int(ratio * 10)
                            ).render(f * ratio, n, sr)
    x *= adsr.render(gate_n, n, sr)[:, None]
    tt = _t(n, sr)
    cut = np.interp(tt, [0.0, 0.25, max(0.3, gate)], [150.0, 2500.0, 600.0])
    x = dsp.filt_varying(x, "lp", cut, sr, q=dsp.res_to_q(0.3)).astype(np.float64)
    vowel = Formant("a", voice="bass").apply(x.astype(np.float32), sr, freq=f)
    x = 0.5 * x + 0.5 * vowel / max(1e-9, float(np.abs(vowel).max())) * np.abs(x).max()
    return _finish(np.tanh(3.0 * x / max(1e-9, np.abs(x).max())), sr)


def _subdrop(sr, bpm):
    n = int(2.0 * sr)
    t = _t(n, sr)
    f = 28.0 + 37.0 * np.exp(-t / 0.4)
    y = np.sin(TWO_PI * np.cumsum(f) / sr) * np.exp(-t / 1.5)
    return _finish(np.tanh(2.0 * y), sr)


def _reverse_swell(sr, bpm, beats):
    n = max(1, int(float(beats) * 60.0 / bpm * sr))
    t = _t(n, sr)
    rng = np.random.default_rng(31)
    metal = sum(np.sin(TWO_PI * fr * t + rng.uniform(0, TWO_PI))
                for fr in rng.uniform(4000.0, 9000.0, 8))
    wash = dsp.filt(rng.uniform(-1, 1, n), "hp", 3000.0, sr)
    hit = (0.3 * metal + wash) * np.exp(-t / 0.7)
    ir = dsp.synth_ir(1.5, sr, predelay=0.0, damp=0.4, seed=32)
    wet = np.stack([signal.fftconvolve(hit, ir[:, c])[:n] for c in range(2)], axis=1)
    x = _stereo(hit) * 0.6 + wet * (np.abs(hit).max() / max(1e-9, np.abs(wet).max()))
    return _finish(x[::-1].copy(), sr, 0.8)


def _whoosh(sr, bpm, beats):
    n = max(1, int(float(beats) * 60.0 / bpm * sr))
    u = np.linspace(0.0, 1.0, n)
    src = dsp.colored_noise(n, "pink", 41).astype(np.float64)
    cut = np.interp(u, [0.0, 0.6, 1.0], [400.0, 4000.0, 800.0])
    y = dsp.filt_varying(src, "bp", cut, sr, q=1.5).astype(np.float64) * np.sin(np.pi * u)
    ang = u * np.pi / 2.0
    return _finish(np.stack([y * np.cos(ang), y * np.sin(ang)], axis=1), sr, 0.85)


_GENERATORS = {
    "riser": _riser,
    "impact": _impact,
    "braam": _braam,
    "subdrop": _subdrop,
    "reverse_swell": _reverse_swell,
    "whoosh": _whoosh,
}

_CACHE: dict = {}


def render_sfx(name: str, params: tuple, sr: int, bpm: float) -> np.ndarray:
    """Vyrenderuje (a nacachuje) jeden SFX při velocity 127, tvar `(m, 2)`."""
    if name not in _GENERATORS:
        raise KeyError(f"neznámý SFX {name!r}; k dispozici: {', '.join(SFX_NAMES)}")
    key = (name, tuple(params), sr, float(bpm))
    if key not in _CACHE:
        _CACHE[key] = np.asarray(_GENERATORS[name](sr, float(bpm), *params),
                                 dtype=np.float32)
    return _CACHE[key]
