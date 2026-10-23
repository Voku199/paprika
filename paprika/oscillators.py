"""Oscilátory: jeden periodický (nebo šumový) průběh na dané frekvenci.

`ratio` je poměr k základní frekvenci noty, takže `Sine(2.01)` je mírně
rozladěná oktáva. `render(freq, n, sr, phase)` bere frekvenci jako číslo,
nebo jako pole délky `n` (frekvenční křivka v Hz) — na tom stojí vibrato,
glide i pitch envelope. `phase` je posun v periodách (round-robin varianty).

Mono oscilátor vrací `(n,)`, stereo (`SuperSaw` se `spread > 0`) `(n, 2)`.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from . import dsp

__all__ = ["Osc", "Sine", "Saw", "Square", "Tri", "Noise", "FM",
           "SuperSaw", "Pulse", "Pluck", "Additive", "Glottal", "glottal",
           "lf_timing", "RD_RANGE"]

TWO_PI = 2.0 * math.pi
MAX_HARMONICS = 64


def _harmonic_sum(cycles: np.ndarray, ks, weights) -> np.ndarray:
    """Σ w_k · sin(2π k · cycles) bez volání sin pro každou harmonickou.

    sin(kx) je imaginární část z^k, kde z = e^{ix}; z^k se počítá postupným
    násobením (komplexní násobení je několikrát levnější než sin).
    Předpoklad: `ks` rostou po stejném kroku (1, 2, 3… nebo 1, 3, 5…).
    """
    ks = np.asarray(ks, dtype=np.int64)
    out = np.zeros(cycles.shape[0], dtype=np.float64)
    if ks.size == 0:
        return out
    x = TWO_PI * np.asarray(cycles, dtype=np.float64)
    step = np.exp(1j * x * (ks[1] - ks[0])) if ks.size > 1 else None
    zk = np.exp(1j * x * ks[0])
    for i, w in enumerate(weights):
        out += w * zk.imag
        if step is not None and i + 1 < ks.size:
            zk *= step
    return out


class Osc:
    """Základ oscilátoru. `ratio` je poměr k základní frekvenci noty."""

    def __init__(self, ratio: float = 1.0, amp: float = 1.0, phase: float = 0.0):
        self.ratio = float(ratio)
        self.amp = float(amp)
        self.phase = float(phase)

    @property
    def stereo(self) -> bool:
        return False

    def _cycles(self, freq, n: int, sr: int, phase: float = 0.0) -> np.ndarray:
        """Fáze v periodách. Pro křivku je to integrál frekvence."""
        ph = self.phase + float(phase)
        if np.isscalar(freq):
            return float(freq) * self.ratio * np.arange(n, dtype=np.float64) / sr + ph
        f = np.asarray(freq, dtype=np.float64)[:n] * self.ratio
        return np.cumsum(f) / sr - f[0] / sr + ph

    def _fmax(self, freq) -> float:
        return float(np.max(freq)) * self.ratio

    def _harmonics(self, freq, sr: int, cap: int = MAX_HARMONICS) -> int:
        """Kolik harmonických se vejde pod Nyquist u nejvyšší frekvence."""
        return max(1, min(cap, int(sr * 0.5 / self._fmax(freq))))

    def render(self, freq, n: int, sr: int, phase: float = 0.0) -> np.ndarray:
        raise NotImplementedError

    def key(self) -> tuple:
        return (type(self).__name__, self.ratio, self.amp, self.phase)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}({self.ratio:g}, amp={self.amp:g})"


class Sine(Osc):
    def render(self, freq, n, sr, phase=0.0):
        return np.sin(TWO_PI * self._cycles(freq, n, sr, phase)).astype(np.float32)


class Saw(Osc):
    """Pilový průběh, band-limited přes sčítání harmonických (bez aliasu)."""

    def render(self, freq, n, sr, phase=0.0):
        if self._fmax(freq) <= 0:
            return np.zeros(n, dtype=np.float32)
        ks = np.arange(1, self._harmonics(freq, sr) + 1)
        out = _harmonic_sum(self._cycles(freq, n, sr, phase), ks, -1.0 / ks)
        out *= 2.0 / math.pi
        return out.astype(np.float32)


class Square(Osc):
    """Obdélník, band-limited (jen liché harmonické)."""

    def __init__(self, ratio=1.0, amp=1.0, phase=0.0, width: float = 0.5):
        super().__init__(ratio, amp, phase)
        self.width = float(width)

    def render(self, freq, n, sr, phase=0.0):
        if self._fmax(freq) <= 0:
            return np.zeros(n, dtype=np.float32)
        ks = np.arange(1, self._harmonics(freq, sr) + 1, 2)
        out = _harmonic_sum(self._cycles(freq, n, sr, phase), ks, 1.0 / ks)
        out *= 4.0 / math.pi
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def key(self):
        return super().key() + (self.width,)


class Tri(Osc):
    def render(self, freq, n, sr, phase=0.0):
        if self._fmax(freq) <= 0:
            return np.zeros(n, dtype=np.float32)
        ks = np.arange(1, self._harmonics(freq, sr, cap=32) + 1, 2)
        signs = np.where(np.arange(ks.size) % 2 == 0, 1.0, -1.0)
        out = _harmonic_sum(self._cycles(freq, n, sr, phase), ks, signs / (ks * ks))
        out *= 8.0 / (math.pi ** 2)
        return out.astype(np.float32)


class Noise(Osc):
    """Šum. `seed` drží zvuk deterministický, `color` white/pink/brown."""

    def __init__(self, ratio=1.0, amp=1.0, seed: int = 12345,
                 color: str = "white"):
        super().__init__(ratio, amp, 0.0)
        self.seed = int(seed)
        if color not in dsp.NOISE_COLORS:
            raise ValueError(f"neznámá barva šumu {color!r}; k dispozici: "
                             f"{', '.join(dsp.NOISE_COLORS)}")
        self.color = color

    def render(self, freq, n, sr, phase=0.0):
        seed = self.seed + int(round(float(phase) * 1000))
        if self.color == "white":
            rng = np.random.default_rng(seed)
            return rng.uniform(-1.0, 1.0, n).astype(np.float32)
        return dsp.colored_noise(n, self.color, seed)

    def key(self):
        return super().key() + (self.seed, self.color)


class FM(Osc):
    """Dvouoperátorové FM: nosič + modulátor s vlastním decay.

    Základ zvuku typu Rhodes (`epiano`) a zvonu (`bell`).
    """

    def __init__(self, ratio=1.0, amp=1.0, mod: float = 2.0, index: float = 3.0,
                 decay: float = 0.6, phase: float = 0.0):
        super().__init__(ratio, amp, phase)
        self.mod = float(mod)
        self.index = float(index)
        self.decay = float(decay)

    def render(self, freq, n, sr, phase=0.0):
        t = np.arange(n, dtype=np.float64) / sr
        cyc = self._cycles(freq, n, sr, phase)
        idx_env = np.exp(-t / self.decay) if self.decay > 0 else np.ones(n)
        modulator = np.sin(TWO_PI * self.mod * (cyc - self.phase - float(phase)))
        out = np.sin(TWO_PI * cyc + self.index * idx_env * modulator)
        return out.astype(np.float32)

    def key(self):
        return super().key() + (self.mod, self.index, self.decay)


# --------------------------------------------------------------------------
# v2
# --------------------------------------------------------------------------

def _dt(freq, ratio: float, n: int, sr: int) -> np.ndarray:
    """f / sr pro polyBLEP — skalár rozkopírovaný, nebo křivka."""
    f = np.asarray(freq, dtype=np.float64) * ratio
    return np.broadcast_to(f / sr, (n,)) if f.ndim == 0 else f[:n] / sr


class SuperSaw(Osc):
    """Unison pil: `voices` hlasů rozladěných v rozsahu `detune` půltónů,
    rozprostřených do stran podle `spread` (0 = mono, 1 = plně L/R).

    Každý hlas má náhodnou startovní fázi (podle `seed`), takže to nezní jako
    jedna fázující pila. Pily jsou polyBLEP — levné i při sedmi hlasech.
    """

    def __init__(self, ratio=1.0, amp=1.0, voices: int = 7, detune: float = 0.25,
                 spread: float = 0.8, seed: int = 0):
        super().__init__(ratio, amp, 0.0)
        self.voices = max(1, int(voices))
        self.detune = float(detune)
        self.spread = float(np.clip(spread, 0.0, 1.0))
        self.seed = int(seed)

    @property
    def stereo(self) -> bool:
        return self.spread > 0

    def render(self, freq, n, sr, phase=0.0):
        v = self.voices
        offs = (np.linspace(-self.detune / 2, self.detune / 2, v) if v > 1
                else np.zeros(1))
        pans = np.linspace(-self.spread, self.spread, v) if v > 1 else np.zeros(1)
        starts = np.random.default_rng(self.seed).uniform(0.0, 1.0, v)
        base = self._cycles(freq, n, sr, phase)
        dt0 = _dt(freq, self.ratio, n, sr)
        left = np.zeros(n)
        right = np.zeros(n)
        for off, pan, st in zip(offs, pans, starts):
            r = 2.0 ** (off / 12.0)
            y = dsp.blep_saw(base * r + st, dt0 * r)
            ang = (pan + 1.0) * math.pi / 4.0
            left += y * math.cos(ang)
            right += y * math.sin(ang)
        norm = 1.0 / math.sqrt(v)
        if not self.stereo:
            # všechny hlasy uprostřed: levý kanál je součet × cos(π/4)
            return (left * norm / math.cos(math.pi / 4)).astype(np.float32)
        return (np.stack([left, right], axis=1) * norm).astype(np.float32)

    def key(self):
        return super().key() + (self.voices, self.detune, self.spread, self.seed)


class Pulse(Osc):
    """Obdélník s proměnnou šířkou (rozdíl dvou polyBLEP pil).

    `pwm` je `LFO` — hloubka v jednotkách šířky, typicky 0.1–0.4.
    """

    def __init__(self, ratio=1.0, amp=1.0, width: float = 0.5, pwm=None):
        super().__init__(ratio, amp, 0.0)
        self.width = float(width)
        self.pwm = pwm

    def render(self, freq, n, sr, phase=0.0):
        cyc = self._cycles(freq, n, sr, phase)
        dt = _dt(freq, self.ratio, n, sr)
        w = np.full(n, self.width)
        if self.pwm is not None:
            w = w + self.pwm.curve(n, sr)
        w = np.clip(w, 0.02, 0.98)
        y = dsp.blep_saw(cyc, dt) - dsp.blep_saw(cyc + w, dt)
        wm = float(np.mean(w))
        return (y / (2.0 * max(wm, 1.0 - wm))).astype(np.float32)

    def key(self):
        pwm = self.pwm.key() if self.pwm is not None else None
        return super().key() + (self.width, pwm)


class Pluck(Osc):
    """Karplus-Strong struna: pizzicato, harfa, kytara, koto.

    Ladí se podle frekvence na začátku noty (glide struna neumí).
    """

    def __init__(self, ratio=1.0, amp=1.0, decay: float = 1.0,
                 bright: float = 0.5, seed: int = 0):
        super().__init__(ratio, amp, 0.0)
        self.decay = float(decay)
        self.bright = float(bright)
        self.seed = int(seed)

    def render(self, freq, n, sr, phase=0.0):
        f0 = float(freq if np.isscalar(freq) else np.asarray(freq)[0]) * self.ratio
        seed = self.seed + int(round(float(phase) * 1000))
        return dsp.karplus(f0, n, sr, self.decay, self.bright, seed)

    def key(self):
        return super().key() + (self.decay, self.bright, self.seed)


class Additive(Osc):
    """Součet parciálů `(ratio, amp, decay_s)`; decay 0 = bez útlumu.

    `inharm` natahuje vyšší parciály jako u struny piana:
    f_k = f · r · sqrt(1 + inharm · r²). Normalizováno součtem amplitud.
    """

    def __init__(self, partials=((1.0, 1.0, 0.0),), inharm: float = 0.0,
                 ratio=1.0, amp=1.0):
        super().__init__(ratio, amp, 0.0)
        self.partials = tuple((float(r), float(a), float(d)) for r, a, d in partials)
        self.inharm = float(inharm)

    def render(self, freq, n, sr, phase=0.0):
        cyc = self._cycles(freq, n, sr, phase)
        fmax = self._fmax(freq)
        t = np.arange(n, dtype=np.float64) / sr
        out = np.zeros(n, dtype=np.float64)
        total = 0.0
        for k, (r, a, d) in enumerate(self.partials):
            r_eff = r * math.sqrt(1.0 + self.inharm * r * r)
            if fmax * r_eff >= sr * 0.5:
                continue
            part = np.sin(TWO_PI * (cyc * r_eff + 0.13 * k)) * a
            if d > 0:
                part *= np.exp(-t / d)
            out += part
            total += abs(a)
        if total > 0:
            out /= total
        return out.astype(np.float32)

    def key(self):
        return super().key() + (self.partials, self.inharm)


# --------------------------------------------------------------------------
# hlasivky
# --------------------------------------------------------------------------

RD_RANGE = (0.3, 2.7)       # tlačený .. dyšný hlas (Fant 1995)
_RD_STEP = 0.1
_TABLE = 2048               # vzorků na periodu v tabulce pulzu
_TABLES: dict = {}


def lf_timing(rd: float) -> tuple[float, float, float]:
    """Rd -> (tp, te, ta) LF modelu v podílech periody (Fant 1995).

    tp = vrchol toku, te = okamžik uzávěru (největší záporná derivace),
    ta = délka návratové fáze. Rd 1 ≈ modální hlas (tp 0.48, te 0.65,
    ta 0.04), 0.3 tlačený (krátké otevření, ostrý uzávěr), 2.7 dyšný.
    """
    rd = float(np.clip(rd, *RD_RANGE))
    rap = (-1.0 + 4.8 * rd) / 100.0
    rkp = (22.4 + 11.8 * rd) / 100.0
    rgp = 0.25 * rkp / (0.11 * rd / (0.5 + 1.2 * rkp) - rap)
    tp = 1.0 / (2.0 * rgp)
    return tp, tp * (1.0 + rkp), rap


def _pulse_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Rd mřížka, derivace toku, tok) — jedna perioda na řádek.

    Tok: nárůst 0.5(1 - cos) do tp, pokles čtvrtvlnou kosinu do te, pak
    zavřeno. Návratovou fázi LF modelu dělá jednopólový lowpass s časovou
    konstantou ta (Fant: návratová fáze ≈ filtr 1. řádu, fa = 1 / 2π ta).
    Derivace (budicí signál, už s vyzařováním rty) má nulový průměr a RMS 1.
    """
    if not _TABLES:
        grid = np.round(np.arange(RD_RANGE[0], RD_RANGE[1] + 1e-9, _RD_STEP), 2)
        u = np.arange(_TABLE) / _TABLE
        exc, flow = [], []
        for rd in grid:
            tp, te, ta = lf_timing(rd)
            g = np.where(u < tp, 0.5 * (1.0 - np.cos(np.pi * u / tp)),
                         np.where(u < te, np.cos(0.5 * np.pi * (u - tp) / (te - tp)), 0.0))
            d = np.diff(g, append=g[0]) * _TABLE
            a = math.exp(-1.0 / (max(ta, 1e-3) * _TABLE))
            two = signal.lfilter([1.0 - a], [1.0, -a], np.tile(d, 3))[2 * _TABLE:]
            two -= two.mean()
            exc.append(two / max(1e-12, float(np.sqrt(np.mean(two ** 2)))))
            flow.append(g)
        _TABLES["grid"] = grid
        _TABLES["exc"] = np.array(exc)
        _TABLES["flow"] = np.array(flow)
    return _TABLES["grid"], _TABLES["exc"], _TABLES["flow"]


def _lookup(table: np.ndarray, row: np.ndarray, w: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Hodnota tabulky na fázi `u` (0..1), interpolace mezi řádky (Rd) i vzorky."""
    pos = u * _TABLE
    j = np.floor(pos).astype(np.int64) % _TABLE
    fr = pos - np.floor(pos)
    j1 = (j + 1) % _TABLE
    a = table[row, j] * (1.0 - fr) + table[row, j1] * fr
    b = table[row + 1, j] * (1.0 - fr) + table[row + 1, j1] * fr
    return a * (1.0 - w) + b * w


def glottal(freq, n: int, sr: int, rd=1.0, jitter: float = 0.008,
            shimmer: float = 0.025, aspiration: float = 0.08, leak: float = 0.0,
            doubling: float = 0.0, seed: int = 0, phase: float = 0.0) -> np.ndarray:
    """Budicí signál hlasivek (derivace toku glotis) délky `n`.

    `freq` a `rd` jsou čísla nebo křivky po samplech. `jitter` a `shimmer`
    jsou směrodatné odchylky délky a amplitudy periody (podíl; výchozí
    hodnoty dají po traktu lokální jitter ~0.35 % a shimmer ~2 %, měřeno
    `voices.voice_quality`). `aspiration` je
    poměr RMS šumu k RMS pulzu; šum je modulovaný tokem glotis (Gobl 2006),
    `leak` přidá šum i v zavřené fázi (neúplné dovření). `doubling` střídá
    amplitudu sudých a lichých period (námaha, chrapot). Počítá se na
    dvojnásobném vzorkování a decimuje, ostrý uzávěr tak nealiasuje.
    """
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    grid, exc, flow = _pulse_tables()
    over = 2
    nx = n * over
    srx = sr * over
    t_src = np.arange(n)
    t_dst = np.arange(nx) / over
    f = (np.full(nx, float(freq)) if np.ndim(freq) == 0
         else np.interp(t_dst, t_src, np.asarray(freq, dtype=np.float64)[:n]))
    rng = np.random.default_rng(int(seed) * 7919 + int(round(float(phase) * 1000)))
    cyc0 = np.cumsum(f) / srx + float(phase)
    k0 = np.floor(cyc0).astype(np.int64)
    span = int(k0.max() - k0.min()) + 3
    dper = 1.0 + float(jitter) * rng.standard_normal(span)
    fj = f / np.clip(dper[k0 - k0.min()], 0.5, 1.5)
    cyc = np.cumsum(fj) / srx - fj[0] / srx + float(phase)
    k = np.floor(cyc).astype(np.int64)
    u = cyc - k
    rdv = np.clip(np.broadcast_to(np.asarray(rd, dtype=np.float64), (n,)), *RD_RANGE)
    pos = (np.interp(t_dst, t_src, rdv) - grid[0]) / _RD_STEP
    row = np.clip(np.floor(pos).astype(np.int64), 0, grid.size - 2)
    w = np.clip(pos - row, 0.0, 1.0)
    e = _lookup(exc, row, w, u)
    kk = k - k.min()
    amp = 1.0 + float(shimmer) * rng.standard_normal(int(kk.max()) + 1)
    if doubling:
        amp *= np.where(np.arange(amp.size) % 2 == 0, 1.0 + doubling, 1.0 - doubling)
    e *= np.clip(amp, 0.1, 3.0)[kk]
    if aspiration > 0 or leak > 0:
        g = _lookup(flow, row, w, u)
        noise = dsp.onepole(rng.standard_normal(nx), 3000.0, srx).astype(np.float64)
        noise /= max(1e-12, float(np.sqrt(np.mean(noise ** 2))))
        mod = float(leak) + g
        mod /= max(1e-12, float(np.sqrt(np.mean(mod ** 2))))
        e = e + float(aspiration) * noise * mod
    y = signal.resample_poly(e, 1, over)[:n]
    return y.astype(np.float32)


class Glottal(Osc):
    """Hlasivky: puls toku glotis podle LF modelu (parametr `rd`: 0.3
    tlačený, 1 modální, 2.7 dyšný) s jitterem, shimmerem a aspiračním
    šumem modulovaným tokem — zdroj pro `Tract` nebo `Formant`.

    Výstup je derivace toku (vyzařování rty je v ní), RMS kolem 1. Každá
    round-robin varianta (jiná `phase`) má jiný jitter i šum.
    """

    def __init__(self, ratio=1.0, amp=1.0, rd: float = 1.0, jitter: float = 0.008,
                 shimmer: float = 0.025, aspiration: float = 0.08, leak: float = 0.0,
                 doubling: float = 0.0, seed: int = 0):
        super().__init__(ratio, amp, 0.0)
        lo, hi = RD_RANGE
        if not lo <= float(rd) <= hi:
            raise ValueError(f"Glottal: rd musí být {lo}..{hi} (tlačený..dyšný), ne {rd!r}")
        self.rd = float(rd)
        self.jitter = max(0.0, float(jitter))
        self.shimmer = max(0.0, float(shimmer))
        self.aspiration = max(0.0, float(aspiration))
        self.leak = max(0.0, float(leak))
        self.doubling = float(np.clip(doubling, 0.0, 0.9))
        self.seed = int(seed)

    def render(self, freq, n, sr, phase=0.0):
        f = (float(freq) * self.ratio if np.isscalar(freq)
             else np.asarray(freq, dtype=np.float64)[:n] * self.ratio)
        return glottal(f, n, sr, self.rd, self.jitter, self.shimmer, self.aspiration,
                       self.leak, self.doubling, self.seed, self.phase + float(phase))

    def key(self):
        return super().key() + (self.rd, self.jitter, self.shimmer, self.aspiration,
                                self.leak, self.doubling, self.seed)
