"""Filtry uvnitř patche (LP, HP, BP, Formant) a jednopólové pomocné funkce.

`onepole_lp/hp` používají i bicí a efekty; počítají se přes `dsp.onepole`
(IIR v scipy), takže ani proměnný cutoff nemá cyklus po samplech.

Všechny patch filtry mají `apply(x, sr, env=None, scale=1.0, bpm=120.0,
freq=440.0)`: `env` je obálka hlasu (0..1), `scale` násobí cutoff
(velocity → jas), `freq` je výška noty (formanty podle hlasového rozsahu).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from . import dsp

__all__ = ["LP", "HP", "BP", "Formant", "FORMANTS", "VOWELS", "VOICE_TYPES",
           "Tract", "tube_formants", "vowel_areas", "VOICE_LENGTH",
           "onepole_lp", "onepole_hp"]


def onepole_lp(x: np.ndarray, cutoff, sr: int, poles: int = 2) -> np.ndarray:
    """Lowpass s konstantním nebo proměnným (pole) cutoffem."""
    if x.size == 0:
        return x
    return dsp.onepole(x, cutoff, sr, poles=max(1, poles))


def onepole_hp(x: np.ndarray, cutoff, sr: int, poles: int = 2) -> np.ndarray:
    """Highpass jako signál minus jeho lowpass."""
    return (x - onepole_lp(x, cutoff, sr, poles)).astype(np.float32)


def _cutoff_curve(cutoff, env_amt, env, scale, lfo, n, sr, bpm):
    """Cutoff jako skalár (nic se nehýbe) nebo pole po samplech."""
    base = float(cutoff) * float(scale)
    curve = base
    if env_amt and env is not None:
        top = min(sr * 0.49, base * (1.0 + 7.0 * env_amt))
        curve = base + (top - base) * np.asarray(env, dtype=np.float64)[:n]
    if lfo is not None:
        curve = curve * 2.0 ** lfo.curve(n, sr, bpm)
    if np.isscalar(curve):
        return float(min(curve, sr * 0.49))
    return np.clip(curve, 10.0, sr * 0.49)


def _lfo_key(lfo):
    return lfo.key() if lfo is not None else None


@dataclass
class LP:
    """Lowpass v patchi. `env` posune cutoff nahoru podle obálky.

    `res` 0..1 přidá rezonanční špičku (acid, growl), `lfo` hýbe cutoffem
    (hloubka v oktávách). S `res=0` a bez `lfo` je to jednopólový filtr
    z v1 — staré songy zní stejně.
    """

    cutoff: float = 3000.0
    env: float = 0.0
    poles: int = 2
    res: float = 0.0
    lfo: object | None = None

    _kind = "lp"

    def apply(self, x, sr, env=None, scale=1.0, bpm=120.0, freq=440.0):
        n = x.shape[0]
        cut = _cutoff_curve(self.cutoff, self.env, env, scale, self.lfo, n, sr, bpm)
        if self.res <= 0 and self.lfo is None:
            low = onepole_lp(x, cut, sr, self.poles)
            return low if self._kind == "lp" else (x - low).astype(np.float32)
        q = dsp.res_to_q(self.res)
        y = dsp.filt_varying(x, self._kind, cut, sr, q=q,
                             stages=max(1, self.poles // 2))
        # rezonanční špička by jinak s `res` rostla o až +25 dB; mírná
        # kompenzace drží hlasitost patche zhruba stejnou jako bez rezonance
        return (y * (dsp.Q_MIN / q) ** 0.35).astype(np.float32)

    def key(self):
        return (type(self).__name__, self.cutoff, self.env, self.poles, self.res,
                _lfo_key(self.lfo))


@dataclass
class HP(LP):
    """Highpass v patchi."""

    cutoff: float = 200.0

    _kind = "hp"


@dataclass
class BP:
    """Pásmová propust: nechá projít jen okolí `cutoff`. Užší s vyšším `res`."""

    cutoff: float = 1000.0
    res: float = 0.3
    env: float = 0.0
    lfo: object | None = None

    def apply(self, x, sr, env=None, scale=1.0, bpm=120.0, freq=440.0):
        cut = _cutoff_curve(self.cutoff, self.env, env, scale, self.lfo,
                            x.shape[0], sr, bpm)
        return dsp.filt_varying(x, "bp", cut, sr, q=dsp.res_to_q(self.res))

    def key(self):
        return ("BP", self.cutoff, self.res, self.env, _lfo_key(self.lfo))


# --------------------------------------------------------------------------
# formanty
# --------------------------------------------------------------------------

VOWELS = ("a", "e", "i", "o", "u")
FORMANT_MODES = ("parallel", "cascade")
VOICE_TYPES = ("auto", "bass", "tenor", "alto", "soprano")

# (frekvence Hz, zisk dB, šířka pásma Hz) pěti formantů — tabulka z Csound
# manuálu („Formant values“).
FORMANTS = {
    "soprano": {
        "a": ((800, 1150, 2900, 3900, 4950), (0, -6, -32, -20, -50), (80, 90, 120, 130, 140)),
        "e": ((350, 2000, 2800, 3600, 4950), (0, -20, -15, -40, -56), (60, 100, 120, 150, 200)),
        "i": ((270, 2140, 2950, 3900, 4950), (0, -12, -26, -26, -44), (60, 90, 100, 120, 120)),
        "o": ((450, 800, 2830, 3800, 4950), (0, -11, -22, -22, -50), (70, 80, 100, 130, 135)),
        "u": ((325, 700, 2700, 3800, 4950), (0, -16, -35, -40, -60), (50, 60, 170, 180, 200)),
    },
    "alto": {
        "a": ((800, 1150, 2800, 3500, 4950), (0, -4, -20, -36, -60), (80, 90, 120, 130, 140)),
        "e": ((400, 1600, 2700, 3300, 4950), (0, -24, -30, -35, -60), (60, 80, 120, 150, 200)),
        "i": ((350, 1700, 2700, 3700, 4950), (0, -20, -30, -36, -60), (50, 100, 120, 150, 200)),
        "o": ((450, 800, 2830, 3500, 4950), (0, -9, -16, -28, -55), (70, 80, 100, 130, 135)),
        "u": ((325, 700, 2530, 3500, 4950), (0, -12, -30, -40, -64), (50, 60, 170, 180, 200)),
    },
    "tenor": {
        "a": ((650, 1080, 2650, 2900, 3250), (0, -6, -7, -8, -22), (80, 90, 120, 130, 140)),
        "e": ((400, 1700, 2600, 3200, 3580), (0, -14, -12, -14, -20), (70, 80, 100, 120, 120)),
        "i": ((290, 1870, 2800, 3250, 3540), (0, -15, -18, -20, -30), (40, 90, 100, 120, 120)),
        "o": ((400, 800, 2600, 2800, 3000), (0, -10, -12, -12, -26), (40, 80, 100, 120, 120)),
        "u": ((350, 600, 2700, 2900, 3300), (0, -20, -17, -14, -26), (40, 60, 100, 120, 120)),
    },
    "bass": {
        "a": ((600, 1040, 2250, 2450, 2750), (0, -7, -9, -9, -20), (60, 70, 110, 120, 130)),
        "e": ((400, 1620, 2400, 2800, 3100), (0, -12, -9, -12, -18), (40, 80, 100, 120, 120)),
        "i": ((250, 1750, 2600, 3050, 3340), (0, -30, -16, -22, -28), (60, 90, 100, 120, 120)),
        "o": ((400, 750, 2400, 2600, 2900), (0, -11, -21, -20, -40), (40, 80, 100, 120, 120)),
        "u": ((350, 600, 2400, 2675, 2950), (0, -20, -32, -28, -36), (40, 80, 100, 120, 120)),
    },
}


def voice_for_pitch(freq: float) -> str:
    """Hlasový rozsah podle výšky noty (MIDI < 48 bas, < 57 tenor, < 65 alt)."""
    midi = 69.0 + 12.0 * math.log2(max(1.0, float(freq)) / 440.0)
    if midi < 48:
        return "bass"
    if midi < 57:
        return "tenor"
    if midi < 65:
        return "alto"
    return "soprano"


@dataclass
class Formant:
    """Samohláska: pět paralelních pásmových filtrů na formantech.

    Pilu nebo pulz udělá „óó“ / „áá“ — sbor, vocal chop, braam. `voice="auto"`
    vybere tabulku podle výšky noty. `morph=("a", "o", 0.3)` přejde z jedné
    samohlásky do druhé za 0.3 s. `gain` dorovná hlasitost po filtraci.
    `mode="cascade"` řadí rezonátory za sebe (údolí mezi formanty a sklon
    spektra jako u skutečného traktu), `vary` náhodně posune formanty mezi
    notami (podíl). Výchozí paralelní režim zní stejně jako dřív.
    """

    vowel: str = "a"
    voice: str = "auto"
    morph: tuple | None = None
    gain: float = 3.0
    mode: str = "parallel"
    vary: float = 0.0

    def __post_init__(self):
        vowels = [self.vowel] + (list(self.morph[:2]) if self.morph else [])
        for v in vowels:
            if v not in VOWELS:
                raise ValueError(f"neznámá samohláska {v!r}; k dispozici: "
                                 f"{', '.join(VOWELS)}")
        if self.voice not in VOICE_TYPES:
            raise ValueError(f"neznámý hlas {self.voice!r}; k dispozici: "
                             f"{', '.join(VOICE_TYPES)}")
        if self.mode not in FORMANT_MODES:
            raise ValueError(f"neznámý režim formantů {self.mode!r}; k dispozici: "
                             f"{', '.join(FORMANT_MODES)}")

    def _cascade(self, x, sr, table, shift):
        """Kaskáda rezonátorů na frekvencích a šířkách tabulky (zisky dá kaskáda)."""
        if self.morph is None:
            f, _, bw = table[self.vowel]
            return dsp.cascade(x, [(fi * shift, bi) for fi, bi in zip(f, bw)], sr)
        v1, v2, sec = self.morph
        u = np.clip(np.arange(x.shape[0]) / sr / max(1e-3, float(sec)), 0.0, 1.0)
        pairs = []
        for (f1, _, b1), (f2, _, b2) in [(table[v1], table[v2])]:
            for i in range(len(f1)):
                pairs.append(((f1[i] + (f2[i] - f1[i]) * u) * shift,
                              b1[i] + (b2[i] - b1[i]) * u))
        return dsp.cascade(x, pairs, sr)

    def apply(self, x, sr, env=None, scale=1.0, bpm=120.0, freq=440.0):
        voice = voice_for_pitch(freq) if self.voice == "auto" else self.voice
        table = FORMANTS[voice]
        x = np.asarray(x, dtype=np.float32)
        shift = 1.0
        if self.vary:
            rng = np.random.default_rng(int(round(float(freq) * 100)) + 3)
            shift = float(np.clip(1.0 + self.vary * rng.standard_normal(), 0.85, 1.15))
        if self.mode == "cascade":
            # kaskáda zesílí podle samohlásky a výšky 3–27x; nota se srovná na
            # polovinu RMS vstupu (gain/3), aby hlasitost na samohlásce nezávisela
            y = self._cascade(x, sr, table, shift)
            got = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
            want = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) if x.size else 0.0
            y = y * (0.5 * want / got if got > 0 else 0.0)
            return (y * self.gain / 3.0).astype(np.float32)
        out = np.zeros(x.shape, dtype=np.float64)
        if self.morph is None:
            for f, db, bw in zip(*table[self.vowel]):
                f = f * shift
                if f >= sr * 0.45:
                    continue
                out += dsp.filt(x, "bp", f, sr, q=f / bw) * 10.0 ** (db / 20.0)
        else:
            v1, v2, sec = self.morph
            n = x.shape[0]
            u = np.clip(np.arange(n) / sr / max(1e-3, float(sec)), 0.0, 1.0)
            for (f1, db1, bw1), (f2, db2, bw2) in zip(zip(*table[v1]), zip(*table[v2])):
                f = (f1 + (f2 - f1) * u) * shift
                q = f / (bw1 + (bw2 - bw1) * u)
                amp = 10.0 ** ((db1 + (db2 - db1) * u) / 20.0)
                y = dsp.filt_varying(x, "bp", f, sr, q=q)
                out += y * (amp[:, None] if y.ndim == 2 else amp)
        return (out * self.gain).astype(np.float32)

    def key(self):
        return ("Formant", self.vowel, self.voice, self.morph, self.gain, self.mode,
                self.vary)


# --------------------------------------------------------------------------
# hlasový trakt jako trubice
# --------------------------------------------------------------------------

SPEED_CM = 35000.0          # rychlost zvuku v teplém vlhkém traktu, cm/s
TUBE_SECTIONS = 44          # jako Pink Trombone; 17.5 cm = 0.4 cm na sekci
LIPS_REFLECTION = 0.95      # ztráty vyzařováním rty (1 = bez ztrát)
# hlasové rozsahy -> délka traktu v cm (muži ~17–18, ženy ~14.5–15)
VOICE_LENGTH = {"bass": 18.0, "tenor": 17.0, "alto": 15.0, "soprano": 14.5}


def _step_up(k: np.ndarray) -> np.ndarray:
    """Odrazové koeficienty -> polynom D(z) (Levinsonova rekurze)."""
    d = np.array([1.0])
    for km in k:
        dp = np.concatenate([d, [0.0]])
        d = dp + km * dp[::-1]
    return d


@functools.lru_cache(maxsize=512)
def tube_formants(areas: tuple, length: float, count: int = 6) -> tuple:
    """Formanty trubice o daných plochách (cm², od glotis ke rtům) a délce
    (cm) jako `((F, B), ...)` v Hz.

    Bezztrátové trubice (Rabiner & Schafer): plochy -> odrazové koeficienty
    -> D(z) -> kořeny. Počítá se na vlastní vzorkovací frekvenci, při které
    sekce přesně vyplní délku, takže formanty nezávisí na `sr` renderu.
    Šířky pásem: ztráty na rtech + stěny a tření (30 Hz + 4 % frekvence,
    empiricky, ať F1 ~70 Hz a F3 ~150 Hz jako u skutečného hlasu).
    Uniformní 17.5 cm trubice dá 500, 1500, 2500 Hz.
    """
    a = np.asarray(areas, dtype=np.float64)
    x = np.linspace(0.0, 1.0, a.size)
    sec = np.interp(np.linspace(0.0, 1.0, TUBE_SECTIONS), x, a)
    k = list((sec[1:] - sec[:-1]) / (sec[1:] + sec[:-1])) + [LIPS_REFLECTION]
    fs = TUBE_SECTIONS * SPEED_CM / (2.0 * float(length))
    z = np.roots(_step_up(np.asarray(k)))
    z = z[np.imag(z) > 1e-9]
    f = np.angle(z) * fs / (2.0 * math.pi)
    bw = -np.log(np.abs(z)) * fs / math.pi
    keep = f > 60.0
    order = np.argsort(f[keep])
    out = [(float(fi), float(bi + 30.0 + 0.04 * fi))
           for fi, bi in zip(f[keep][order], bw[keep][order])]
    return tuple(out[:count])


# Parametrický tvar traktu pro samohlásky: (místo zúžení 0..1 od glotis,
# plocha zúžení cm², plocha rtů cm², plocha traktu cm², podíl délky rtů).
# Fitováno na F1–F3 tabulky `tenor` při délce 17 cm (2026-10-23): a, e, i
# do 0.5 %, o do 3.2 %, u do 6 % — jedno zúžení plus rty „u“ přesně neumí.
VOWEL_TUBES = {
    "a": (0.205, 0.1, 9.34, 2.856, 0.098),
    "e": (0.53, 0.596, 4.295, 4.086, 0.057),
    "i": (0.71, 0.29, 2.661, 4.058, 0.059),
    "o": (0.307, 0.2, 0.554, 2.987, 0.128),
    "u": (0.305, 0.1, 0.489, 4.155, 0.144),
}
_AREA_POINTS = 40
# nosní větev pro „m“ (zavřené rty): murmur ~270 Hz, antirezonance ~1 kHz
NASAL_POLES = ((270.0, 100.0), (2100.0, 300.0), (3200.0, 400.0))
NASAL_ZERO = 1050.0


@functools.lru_cache(maxsize=32)
def vowel_areas(vowel: str) -> tuple:
    """Plochy traktu (cm², od glotis ke rtům) pro samohlásku."""
    if vowel not in VOWEL_TUBES:
        raise ValueError(f"neznámá samohláska {vowel!r}; k dispozici: {', '.join(VOWELS)}")
    xc, ac, lip, a0, lf = VOWEL_TUBES[vowel]
    x = np.linspace(0.0, 1.0, _AREA_POINTS)
    a = a0 - (a0 - ac) * np.exp(-((x - xc) / 0.2) ** 2)
    a[x > 1.0 - lf] = lip
    return tuple(np.round(np.clip(a, 0.05, 12.0), 4))


@dataclass
class Tract:
    """Hlasový trakt jako trubice: tvar (samohláska nebo plochy) -> formanty
    s fyzikálními šířkami pásem -> kaskáda rezonátorů (Klatt).

    Na rozdíl od `Formant` (pět paralelních pásmových propustí z tabulky)
    vzniknou údolí mezi formanty a spektrální sklon samy a přechod mezi
    samohláskami jde přes plochy, ne přes lineární posun formantů.

    - `vowel` a e i o u, nebo `areas` = vlastní plochy v cm² od glotis ke rtům,
    - `voice` bass/tenor/alto/soprano/auto (podle výšky noty) určí délku
      traktu, `length` (cm) ji přepíše,
    - `morph=("a", "u", 0.4)` přejde za 0.4 s z jedné samohlásky do druhé,
    - `nasal` 0..1 přimíchá nosní větev (1 = „m“ se zavřenými rty),
    - `vary` náhodně posune formanty mezi notami (podíl, 0.03 = ±3 %:
      čelist a hrtan nejsou nikdy ve stejné poloze),
    - `gain` dorovná hlasitost (výchozí 0.5 dá z `Glottal` zhruba jeho RMS;
      samohlásky si nechají přirozený rozdíl hlasitosti, „a“ zní silněji než „i“),
    - `norm=True` srovná každou notu na RMS vstupu × `gain` — pro nástroje,
      kde by harmonická trefená do F1 dělala skoky hlasitosti mezi notami.

    Zdroj čeká derivaci toku (`Glottal`), vyzařování rty je už v ní.
    """

    vowel: str = "a"
    voice: str = "auto"
    length: float | None = None
    areas: tuple | None = None
    morph: tuple | None = None
    nasal: float = 0.0
    vary: float = 0.0
    gain: float = 0.5
    norm: bool = False

    def __post_init__(self):
        vowels = [self.vowel] + (list(self.morph[:2]) if self.morph else [])
        for v in vowels:
            if v not in VOWEL_TUBES:
                raise ValueError(f"neznámá samohláska {v!r}; k dispozici: "
                                 f"{', '.join(VOWELS)}")
        if self.voice not in VOICE_TYPES:
            raise ValueError(f"neznámý hlas {self.voice!r}; k dispozici: "
                             f"{', '.join(VOICE_TYPES)}")
        if self.areas is not None:
            self.areas = tuple(float(a) for a in self.areas)
            if len(self.areas) < 2 or min(self.areas) <= 0:
                raise ValueError("Tract: areas musí být aspoň dvě kladné plochy (cm²)")
        self.nasal = float(np.clip(self.nasal, 0.0, 1.0))

    def _length(self, freq: float) -> float:
        if self.length:
            return float(self.length)
        voice = voice_for_pitch(freq) if self.voice == "auto" else self.voice
        return VOICE_LENGTH[voice]

    def _shape(self, vowel: str) -> np.ndarray:
        return np.asarray(self.areas if self.areas is not None else vowel_areas(vowel))

    def formants(self, freq: float = 440.0) -> tuple:
        """Formanty `((F, B), ...)` výchozího tvaru pro notu dané výšky."""
        return tube_formants(tuple(self._shape(self.vowel)), self._length(freq))

    def apply(self, x, sr, env=None, scale=1.0, bpm=120.0, freq=440.0):
        x = np.asarray(x, dtype=np.float64)
        n = x.shape[0]
        length = self._length(freq)
        shift = 1.0
        if self.vary:
            rng = np.random.default_rng(int(round(float(freq) * 100)) + 7 * len(self.vowel))
            shift = float(np.clip(1.0 + self.vary * rng.standard_normal(), 0.85, 1.15))
        if self.morph is None:
            formants = [(f * shift, b) for f, b in
                        tube_formants(tuple(self._shape(self.vowel)), length)]
            y = dsp.cascade(x, formants, sr)
        else:
            v1, v2, sec = self.morph
            a1, a2 = np.log(self._shape(v1)), np.log(self._shape(v2))
            steps = np.linspace(0.0, 1.0, 9)
            table = [tube_formants(tuple(np.round(np.exp(a1 + (a2 - a1) * w), 4)), length)
                     for w in steps]
            k = min(len(t) for t in table)
            u = np.clip(np.arange(n) / sr / max(1e-3, float(sec)), 0.0, 1.0)
            formants = [(np.interp(u, steps, [t[i][0] for t in table]) * shift,
                         np.interp(u, steps, [t[i][1] for t in table])) for i in range(k)]
            y = dsp.cascade(x, formants, sr)
        if self.nasal > 0:
            nose = dsp.cascade(x, NASAL_POLES, sr, scale=VOICE_LENGTH["tenor"] / length)
            nose = dsp.filt(nose, "notch", NASAL_ZERO * VOICE_LENGTH["tenor"] / length,
                            sr, q=1.5).astype(np.float64)
            y = math.sqrt(1.0 - self.nasal) * y + math.sqrt(self.nasal) * nose
        if self.norm:
            got = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
            want = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
            y = y * (want / got if got > 0 else 0.0)
        return (y * self.gain).astype(np.float32)

    def key(self):
        return ("Tract", self.vowel, self.voice, self.length, self.areas, self.morph,
                self.nasal, self.vary, self.gain, self.norm)
