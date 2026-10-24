"""Lidské zvuky: dech, výdech, mlasknutí, srdce, šustění oblečení, kroky.

Nic tu není nahraná smyčka. Zvuky vycházejí z fyziky, aby se daly
přizpůsobit frázi a nikdy se neopakovaly:

- **Dech** je proud vzduchu. Objem za čas dává průtok U(t). Slyšet je až
  turbulence nad prahem laminárního proudu a její hlasitost roste zhruba
  s U³ (Stevens). Engine exponent stlačuje (`BREATH_EXP`), aby dech zůstal
  v rozsahu mixu. Rychlý nádech (catch breath) je proto hlasitější a
  jasnější než pomalý hluboký, stejně jako ve skutečnosti.
- **Trakt** (ústa, nos) je kaskáda rezonátorů na formantech samohlásky
  (Klatt), s rozšířenými pásmy, protože glotis je otevřená. Přední
  turbulence na zubech a rtech je zvlášť přimíchaný vysoký šum.
- **Pohyb** (oblečení) sleduje profil minimálního trhnutí (Flash & Hogan).
  Šustění je proud mikro-impulzů, jejichž četnost roste s rychlostí pohybu
  (PhISEM, Perry Cook).
- **Kroky a srdce** jsou tlumené rezonance buzené krátkým pulzem.

Nádech může místo fyzikálního traktu vzít barvu a průběh z modelu
skutečných nádechů zpěváků VocalSet (`breath.py`, parametr `model`);
hlasitost a jas pak dál řídí stejná fyzika proudu.

Každý zvuk má `variant`, ze které se losují parametry (délka traktu, tvar
průtoku, zrna šustění). Dva nádechy za sebou tak nejsou tentýž vzorek.
Generátor vrací `(m, 2)` float64 při velocity 127. Render a cache řeší
`sfx.render_sfx`, časování `HumanTrack` v `song.py`. Tady jsou i čistě
modelové pomocníky `breath_phrases` a `air_seconds`, které používá
`HumanTrack.breaths()` a pravidla reportu.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from . import dsp, gait
from .filters import FORMANTS, VOWELS, Tract
from .oscillators import glottal

__all__ = ["HUMAN_NAMES", "HUMAN_CHARS", "BREATH_KINDS", "EXHALE_KINDS",
           "HUMAN_VOICES", "FABRICS", "MOTIONS", "SURFACES", "MAX_VARIANTS",
           "BREATH_SEC", "EXHALE_SEC", "GENERATORS", "breath_phrases",
           "air_seconds", "voice_of", "vowel_of", "VOICE_GESTURES",
           "GESTURE_SEC", "GESTURE_CENTER", "pitch_curve"]

TWO_PI = 2.0 * math.pi

HUMAN_NAMES = ("breath", "exhale", "lip_smack", "heartbeat", "cloth", "step",
               "hum", "sigh", "effort", "laugh", "clip")
HUMAN_CHARS = {"breath": "^", "exhale": "~", "lip_smack": "'", "heartbeat": "@",
               "cloth": "%", "step": "x", "hum": "m", "sigh": "\\", "effort": "!",
               "laugh": "h", "clip": "&"}
VOICE_GESTURES = ("hum", "sigh", "effort", "laugh")
BREATH_KINDS = ("mouth", "nose", "catch")
EXHALE_KINDS = ("mouth", "nose", "sigh")
HUMAN_VOICES = ("male", "female")
MOTIONS = ("swing", "fidget")
MAX_VARIANTS = 8

# výchozí délky v sekundách; HumanTrack je převede na beaty podle tempa
BREATH_SEC = {"mouth": 0.45, "nose": 0.6, "catch": 0.2}
EXHALE_SEC = {"mouth": 0.6, "nose": 0.8, "sigh": 1.4}

# --- fyzika dechu (orientační hodnoty, viz research/2026-10-23) ------------
LUNG_L = 1.5          # l: objem hlubokého nádechu (depth = 1)
FLOW_QUIET = 0.3      # l/s: pod tímhle průtokem je proud laminární, skoro neslyšet
FLOW_REF = 3.0        # l/s: měřítko průtoku pro hlasitost a jas
BREATH_EXP = 1.5      # turbulence ~U^3; stlačeno, aby dech zůstal v mixu
BREATH_RMS = 0.08     # RMS referenčního nádechu (ústa, depth 0.6, 0.45 s) při vel 127
CEILING = 0.95
BW_OPEN = 4.0         # otevřená glotis rozšíří pásma formantů (dech, šepot)
VOICE_SCALE = {"male": 1.0, "female": 1.17}   # kratší trakt = vyšší formanty
NOSE_GAIN = 0.7       # nosem proteče méně vzduchu, dech je tišší
# nosní dutina: (Hz, šířka pásma Hz); antirezonance kolem 1.1 kHz
NASAL = ((480.0, 300.0), (1650.0, 450.0), (2700.0, 600.0))
NASAL_ZERO = 1100.0
# vyšší póly traktu nad F5 (Klattova „higher pole correction“), aby spektrum
# nad 4 kHz nepadalo jako kaskáda pěti rezonátorů
HIGH_POLES = ((4500.0, 900.0), (5800.0, 1200.0))
# přední turbulence (zuby, rty / nosní dírky): (podíl, střed pásma Hz)
FRONT_MOUTH = (1.0, 1200.0)
FRONT_NOSE = (0.5, 800.0)
FRONT_SPEED_REF = 1.0  # rychlost proudu (průtok / FLOW_REF), při které platí podíl
BRIGHT_LO, BRIGHT_SPAN = 400.0, 800.0  # Hz: jas zdroje = LO + SPAN * rychlost^0.8
# Jas, šířky formantů a přední turbulence jsou zkalibrované na 10 skutečných
# nádechů VocalSet (female7, female8, male3; samples breaths --examples,
# 2026-10-23): těžiště 2.2–2.4 kHz u obou pohlaví, hlavní energie 1–2 kHz.
# Původní odhad (3.7–3.9 kHz) byl o 1.5 kHz jasnější než skutečný dech.
RADIATION = 1.0        # vyzařování rty 1 - a z^-1 (1 = +6 dB/okt, 0 = rovně)
DATA_TILT = 3.0       # dB/okt nad 1 kHz na zdvojnásobení rychlosti proudu (model z dat)

# --- zásoba vzduchu pro frázi (pravidlo „chybí nádech“) --------------------
AIR_L = 2.5           # l: použitelný objem u školeného zpěváka/hráče
MIN_REST_SEC = 0.15   # s: kratší pauza mezi notami není místo na nádech

# --- oblečení a kroky --------------------------------------------------------
FABRICS = {
    # četnost zrn při plné rychlosti /s, (hlavní pásmo Hz, Q), (tělo Hz, Q), šum
    "cotton": (1500.0, (2600.0, 0.9), (900.0, 0.8), 0.25),
    "nylon": (2600.0, (5200.0, 0.8), (2200.0, 0.9), 0.55),
    "wool": (700.0, (1400.0, 0.8), (500.0, 0.9), 0.12),
}
SURFACES = {
    # dupnutí (Hz, τ s, amp), klik (Hz, τ s, amp), rezonance (Hz, τ, amp) nebo
    # None, zrna (četnost /s, pásmo Hz, délka s)
    "stone": ((90.0, 0.025, 0.45), (3200.0, 0.006, 1.6), None, None),
    "wood": ((130.0, 0.045, 0.55), (2000.0, 0.005, 1.6), (420.0, 0.06, 0.3), None),
    "gravel": ((80.0, 0.03, 0.4), (2500.0, 0.004, 0.6), None, (2500.0, 3500.0, 0.07)),
    "dirt": ((70.0, 0.04, 0.6), (1500.0, 0.004, 0.5), None, (600.0, 1500.0, 0.05)),
}
# pevný zisk kroku z plánu chůze: referenční chodec (75 kg, 1.4 m/s) má
# medián špičky 0.45, aby běh (~0.7) i těžší chodec měli kam růst pod CEILING
STEP_GAIN = {"stone": 0.335, "wood": 0.393, "gravel": 0.499, "dirt": 0.623}


# --------------------------------------------------------------------------
# model: fráze a vzduch (bez audia)
# --------------------------------------------------------------------------

def breath_phrases(notes, min_rest: float) -> list[tuple[float, float, list]]:
    """Noty stopy rozdělené na fráze `(start, konec, noty)` v beatech.

    Fráze končí tam, kde je mezi koncem dosud znějících not a další notou
    pauza aspoň `min_rest` beatů — tam se dá nadechnout.
    """
    out: list = []
    for n in sorted(notes, key=lambda n: n.start_beat):
        end = n.start_beat + n.dur_beats
        if out and n.start_beat - out[-1][1] < min_rest:
            start, last, group = out[-1]
            group.append(n)
            out[-1] = (start, max(last, end), group)
        else:
            out.append((n.start_beat, end, [n]))
    return out


def air_seconds(velocity: float) -> float:
    """Jak dlouho vystačí jeden nádech při dané dynamice (orientačně).

    Průtok roste s hlasitostí: ~0.08 l/s v pp, ~0.25 l/s ve ff. Použitelný
    objem `AIR_L` dává zhruba 20 s v mp a 10 s ve ff.
    """
    v = float(np.clip(velocity, 1, 127)) / 127.0
    return AIR_L / (0.08 + 0.17 * v * v)


def voice_of(track) -> str:
    """Mužský/ženský trakt podle jména nástroje stopy (výchozí "male")."""
    if track is None:
        return "male"
    label = f"{getattr(track, 'name', '')} {getattr(getattr(track, 'patch', None), 'name', '')}"
    label = label.lower()
    if any(w in label for w in ("female", "soprano", "alto", "mezzo")):
        return "female"
    return "male"


def vowel_of(track) -> str:
    """Samohláska, přes kterou se dýchá: z artikulace sboru, jinak "a"."""
    label = str(getattr(getattr(track, "patch", None), "name", "") or "").lower()
    label += " " + str(getattr(track, "name", "") or "").lower()
    for key, v in (("ooh", "u"), ("aah", "a"), ("oh", "o"), ("eh", "e"),
                   ("mm", "u")):
        if key in label:
            return v
    return "a"


# --------------------------------------------------------------------------
# pomocné DSP
# --------------------------------------------------------------------------

def _rng(kind_seed: int, variant: int) -> np.random.Generator:
    return np.random.default_rng(kind_seed * 7919 + int(variant))


def _stereo(x: np.ndarray, left: float = 1.0, right: float = 1.0) -> np.ndarray:
    return np.stack([x * left, x * right], axis=1)


def _edges(x: np.ndarray, sr: int, sec: float = 0.004) -> np.ndarray:
    n = min(int(sec * sr), x.shape[0] // 2)
    if n > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, n))
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def _ceiling(x: np.ndarray) -> np.ndarray:
    peak = float(np.abs(x).max()) if x.size else 0.0
    return x * (CEILING / peak) if peak > CEILING else x


def _peak(x: np.ndarray, level: float) -> np.ndarray:
    peak = float(np.abs(x).max()) if x.size else 0.0
    return x * (level / peak) if peak > 0 else x


def _min_jerk(u: np.ndarray) -> np.ndarray:
    """Rychlost pohybu s minimálním trhnutím na 0..1, maximum 1."""
    u = np.clip(u, 0.0, 1.0)
    return 16.0 * u * u * (1.0 - u) * (1.0 - u)


# --------------------------------------------------------------------------
# dech
# --------------------------------------------------------------------------

def _inhale_profile(n: int, kind: str, rng) -> np.ndarray:
    u = (np.arange(n) + 0.5) / max(1, n)
    if kind == "catch":
        a, b = rng.uniform(0.8, 1.1), rng.uniform(2.1, 2.7)
    else:
        a, b = rng.uniform(1.2, 1.6), rng.uniform(1.8, 2.4)
    p = u ** a * (1.0 - u) ** b
    return p / max(1e-12, float(p.max()))


def _exhale_profile(n: int, sr: int, kind: str, rng) -> np.ndarray:
    t = np.arange(n) / sr
    dur = n / sr
    rise = rng.uniform(0.04, 0.08)
    decay = dur * (0.45 if kind == "sigh" else 0.35) * rng.uniform(0.85, 1.15)
    p = (1.0 - np.exp(-t / rise)) * np.exp(-t / decay)
    tail = max(1, int(0.12 * n))
    p[-tail:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, math.pi, tail))
    return p / max(1e-12, float(p.max()))


def _loudness(flow: np.ndarray) -> np.ndarray:
    """Hlasitost turbulence z průtoku (l/s): práh laminárního proudu, pak
    mocnina. Fyzikálně ~U^3, tady stlačeno na `BREATH_EXP`."""
    return (np.maximum(0.0, flow - FLOW_QUIET) / FLOW_REF) ** BREATH_EXP


def _flow(profile: np.ndarray, sec: float, depth: float) -> np.ndarray:
    """Průtok v l/s: profil škálovaný tak, aby proteklo `depth · LUNG_L`."""
    mean = float(profile.mean()) if profile.size else 1.0
    return profile * (float(depth) * LUNG_L / max(1e-6, sec * mean))


def _ref_rms() -> float:
    """RMS obálky referenčního nádechu — kalibrace `BREATH_RMS`."""
    sr = 8000
    n = int(BREATH_SEC["mouth"] * sr)
    u = (np.arange(n) + 0.5) / n
    p = u ** 1.4 * (1.0 - u) ** 2.1
    p /= p.max()
    env = _loudness(_flow(p, BREATH_SEC["mouth"], 0.6))
    return float(np.sqrt(np.mean(env ** 2)))


_REF_RMS = _ref_rms()


def _airflow(sr: int, sec: float, depth: float, kind: str, vowel: str, voice: str,
             variant: int, inhale: bool) -> np.ndarray:
    """Společné jádro nádechu a výdechu: průtok → šum → trakt → hlasitost."""
    if vowel not in VOWELS:
        raise ValueError(f"neznámá samohláska {vowel!r}; k dispozici: {', '.join(VOWELS)}")
    if voice not in HUMAN_VOICES:
        raise ValueError(f"neznámý hlas {voice!r}; k dispozici: {', '.join(HUMAN_VOICES)}")
    n = max(16, int(round(float(sec) * sr)))
    rng = _rng(11 if inhale else 13, variant)
    profile = (_inhale_profile(n, kind, rng) if inhale
               else _exhale_profile(n, sr, kind, rng))
    flow = _flow(profile, n / sr, depth)
    env = _loudness(flow)
    speed = np.clip(flow / FLOW_REF, 0.0, 1.6)

    # zdroj u glotis, filtrovaný celým traktem (jako Klattovo „h“); jas
    # roste s rychlostí proudu
    src = rng.standard_normal(n) * env
    bright = dsp.filt_varying(src, "lp", BRIGHT_LO + BRIGHT_SPAN * speed ** 0.8,
                              sr).astype(np.float64)
    tract_len = rng.uniform(0.96, 1.04) * VOICE_SCALE[voice]
    if kind == "nose":
        body = dsp.cascade(bright, NASAL, sr, scale=tract_len)
        body = dsp.filt(body, "notch", NASAL_ZERO * tract_len, sr, q=1.4).astype(np.float64)
        front_k, front_f = FRONT_NOSE
    else:
        f, _, bw = FORMANTS["tenor"][vowel]
        formants = [(fi, bi * BW_OPEN) for fi, bi in zip(f, bw)] + list(HIGH_POLES)
        scale = tract_len
        if kind == "sigh":
            # při vzdechu se čelist přivírá: formanty klesnou asi o desetinu
            scale = tract_len * np.linspace(1.0, 0.9, n)
        body = dsp.cascade(bright, formants, sr, scale=scale)
        front_k, front_f = FRONT_MOUTH
    body = signal.lfilter([1.0, -RADIATION], [1.0], body)   # vyzařování rty
    # turbulence u zubů a rtů vyzařuje skoro nefiltrovaná: široké pásmo,
    # jehož střed roste s rychlostí proudu (Strouhal: f ~ v/d)
    front_src = rng.standard_normal(n) * env * np.sqrt(speed)
    front = dsp.filt_varying(front_src, "bp", front_f * (0.6 + 0.6 * speed ** 0.8),
                             sr, q=0.6).astype(np.float64)
    front = dsp.filt(front, "lp", 9000.0, sr).astype(np.float64)
    body /= max(1e-12, float(np.sqrt(np.mean(body ** 2))))
    front /= max(1e-12, float(np.sqrt(np.mean(front ** 2))))
    # podíl přední turbulence roste s rychlostí proudu (váženo energií)
    energy = env * env
    speed_eff = float((energy * speed).sum() / max(1e-12, float(energy.sum())))
    y = body + front_k * math.sqrt(speed_eff / FRONT_SPEED_REF) * front
    # hlasitost určuje jen průtok, ne barva filtru
    y *= float(np.sqrt(np.mean(env ** 2))) / max(1e-12, float(np.sqrt(np.mean(y ** 2))))
    y *= BREATH_RMS / _REF_RMS
    if kind == "nose":
        y *= NOSE_GAIN
    return _stereo(_edges(_ceiling(y), sr))


def _breath(sr, bpm, kind, beats, depth, vowel, voice, model, variant):
    if kind not in BREATH_KINDS:
        raise ValueError(f"neznámý druh nádechu {kind!r}; k dispozici: "
                         f"{', '.join(BREATH_KINDS)}")
    sec = float(beats) * 60.0 / float(bpm)
    if model is not None and kind != "nose":
        return _breath_data(sr, sec, depth, kind, vowel, model, variant)
    return _airflow(sr, sec, depth, kind, vowel, voice, variant, inhale=True)


def _mean_speed(flow: np.ndarray) -> float:
    """Rychlost proudu vážená energií turbulence (jako `_airflow`)."""
    energy = _loudness(flow) ** 2
    speed = np.clip(flow / FLOW_REF, 0.0, 1.6)
    return float((energy * speed).sum() / max(1e-12, float(energy.sum())))


def _breath_data(sr: int, sec: float, depth: float, kind: str, vowel: str,
                 model: str, variant: int) -> np.ndarray:
    """Nádech z modelu VocalSet (`breath.py`): barva a průběh skutečného
    zpěváka, hlasitost a jas podle stejné fyziky jako `_airflow`.

    Model zná přirozené nádechy zpěváka (mediánová délka). Kratší nebo
    hlubší nádech, než je jeho zvyk, je rychlejší proud: zní hlasitěji
    (kalibrace jako u fyzikálního dechu) a o `DATA_TILT` dB/okt na každé
    zdvojnásobení rychlosti jasněji.
    """
    from . import breath as _data
    m = _data.load(model)
    n = max(16, int(round(float(sec) * sr)))
    rng = _rng(37, variant)
    flow = _flow(_inhale_profile(n, kind, rng), n / sr, depth)
    env = _loudness(flow)
    nat = max(0.05, float(m["dur"][1]))
    u = (np.arange(256) + 0.5) / 256
    ref = u ** 1.4 * (1.0 - u) ** 2.1
    natural = _mean_speed(_flow(ref / ref.max(), nat, 0.6))
    tilt = float(np.clip(DATA_TILT * math.log2(max(1e-6, _mean_speed(flow))
                                                / max(1e-6, natural)), -4.0, 4.0))
    y = _data.synth(m, n, sr, rng, vowel=vowel, tilt_db=tilt)
    y *= float(np.sqrt(np.mean(env ** 2))) * BREATH_RMS / _REF_RMS
    return _stereo(_edges(_ceiling(y), sr))


def _exhale(sr, bpm, kind, beats, depth, vowel, voice, variant):
    if kind not in EXHALE_KINDS:
        raise ValueError(f"neznámý druh výdechu {kind!r}; k dispozici: "
                         f"{', '.join(EXHALE_KINDS)}")
    sec = float(beats) * 60.0 / float(bpm)
    return _airflow(sr, sec, depth, kind, vowel, voice, variant, inhale=False)


# --------------------------------------------------------------------------
# ústa a tělo
# --------------------------------------------------------------------------

def _lip_smack(sr, bpm, variant):
    """Mlasknutí: 2–4 drobné cvaky, jak se odlepují vlhké rty (5–20 ms)."""
    rng = _rng(17, variant)
    n = int(0.06 * sr)
    x = np.zeros(n)
    t0 = 0.002
    burst = int(0.0015 * sr)
    tb = np.arange(burst) / sr
    for i in range(2 + int(variant) % 3):
        start = int(t0 * sr)
        if start + burst >= n:
            break
        amp = 0.6 ** i * rng.uniform(0.8, 1.2)
        x[start:start + burst] += amp * rng.standard_normal(burst) * np.exp(-tb / 0.0004)
        t0 += rng.uniform(0.003, 0.009)
    click = dsp.filt(x, "bp", rng.uniform(2200.0, 3200.0), sr, q=0.9).astype(np.float64)
    b, a = dsp.resonator(rng.uniform(1000.0, 1400.0), 400.0, sr)     # dutina úst
    y = click + 0.4 * signal.lfilter(b, a, x)
    return _stereo(_edges(_peak(y, 0.5), sr, 0.001))


def _damped(freqs, n: int, sr: int, rng) -> np.ndarray:
    t = np.arange(n) / sr
    out = np.zeros(n)
    for f, tau, amp in freqs:
        f = f * rng.uniform(0.96, 1.04)
        out += amp * np.sin(TWO_PI * f * t) * np.exp(-t / tau)
    atk = min(n, int(0.006 * sr))
    out[:atk] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, atk))
    return out


def _heartbeat(sr, bpm, rate, variant):
    """Jeden tep: „lub“ (S1) a o systolu později kratší vyšší „dub“ (S2).

    Systola se zrychlením tepu zkracuje méně než celá perioda (~√T).
    Saturace přidá harmonické, aby tep byl slyšet i bez subbasu.
    """
    rng = _rng(19, variant)
    systole = 0.35 * math.sqrt(60.0 / max(20.0, float(rate)))
    n = int((systole + 0.3) * sr)
    s1 = _damped(((62.0, 0.045, 1.0), (100.0, 0.03, 0.6), (160.0, 0.018, 0.35)),
                 n, sr, rng)
    s2_n = n - int(systole * sr)
    s2 = _damped(((78.0, 0.025, 1.0), (130.0, 0.018, 0.6), (210.0, 0.012, 0.3)),
                 s2_n, sr, rng) * rng.uniform(0.55, 0.75)
    y = s1
    y[n - s2_n:] += s2
    t = np.arange(n) / sr
    thump = dsp.filt(rng.standard_normal(n) * np.exp(-t / 0.02), "lp", 250.0, sr)
    y = y + 0.3 * thump / max(1e-9, float(np.abs(thump).max()))
    y = np.tanh(2.2 * y / max(1e-9, float(np.abs(y).max()))) / math.tanh(2.2)
    y = dsp.filt(y, "hp", 35.0, sr).astype(np.float64)
    return _stereo(_edges(_peak(y, 0.85), sr))


def _motion(n: int, motion: str, rng) -> np.ndarray:
    u = (np.arange(n) + 0.5) / max(1, n)
    if motion == "swing":
        return _min_jerk(u)
    v = np.zeros(n)
    for _ in range(3):
        w = rng.uniform(0.25, 0.45)
        c = rng.uniform(w / 2, 1.0 - w / 2)
        v += rng.uniform(0.5, 1.0) * _min_jerk((u - (c - w / 2)) / w)
    return v / max(1e-12, float(v.max()))


def _cloth(sr, bpm, fabric, beats, motion, variant):
    """Šustění: zrna s četností úměrnou rychlosti pohybu + tření (šum ~v²)."""
    if fabric not in FABRICS:
        raise ValueError(f"neznámá látka {fabric!r}; k dispozici: {', '.join(FABRICS)}")
    if motion not in MOTIONS:
        raise ValueError(f"neznámý pohyb {motion!r}; k dispozici: {', '.join(MOTIONS)}")
    rate, (band_f, band_q), (body_f, body_q), hiss_k = FABRICS[fabric]
    rng = _rng(23, variant)
    n = max(16, int(float(beats) * 60.0 / float(bpm) * sr))
    v = _motion(n, motion, rng)
    events = rng.random(n) < rate * v / sr
    grains = np.where(events, v ** 1.5 * rng.lognormal(0.0, 0.6, n), 0.0)
    band_f *= rng.uniform(0.9, 1.1)
    crackle = (dsp.filt(grains, "bp", band_f, sr, q=band_q).astype(np.float64)
               + 0.5 * dsp.filt(grains, "bp", body_f, sr, q=body_q).astype(np.float64))
    hiss = dsp.filt(rng.standard_normal(n) * v * v, "bp", band_f, sr, q=0.5).astype(np.float64)
    hiss *= hiss_k * float(np.abs(crackle).max()) / max(1e-12, float(np.abs(hiss).max()))
    return _stereo(_edges(_peak(crackle + hiss, 0.6), sr))


def _strike(n: int, sr: int, surface, strength: float, rng, pitch: float = 1.0,
            bright: float = 1.0, decay: float = 1.0) -> np.ndarray:
    (thud_f, thud_tau, thud_a), (click_f, click_tau, click_a), res, grains = surface
    t = np.arange(n) / sr
    y = (thud_a * np.sin(TWO_PI * thud_f * rng.uniform(0.95, 1.05) * pitch * t)
         * np.exp(-t / (thud_tau * decay)))
    click = rng.standard_normal(n) * np.exp(-t / click_tau)
    y += click_a * dsp.filt(click, "bp", min(click_f * rng.uniform(0.9, 1.1) * bright,
                                             0.4 * sr), sr, q=1.2)
    if res is not None:
        rf, rtau, ra = res
        y += ra * np.sin(TWO_PI * rf * rng.uniform(0.97, 1.03) * pitch * t) * np.exp(-t / rtau)
    if grains is not None:
        g_rate, g_f, g_len = grains
        dens = g_rate * np.exp(-t / (g_len / 3.0))
        hits = np.where(rng.random(n) < dens / sr, rng.lognormal(0.0, 0.5, n), 0.0)
        y += 0.8 * dsp.filt(hits, "bp", g_f, sr, q=0.8)
    atk = min(n, int(0.002 * sr))
    y[:atk] *= np.linspace(0.0, 1.0, atk)
    return strength * y


def _step(sr, bpm, surface, foot, *rest):
    """Krok: pata a o 35–70 ms později špička, levá a pravá noha se liší.

    `rest` je buď jen `(variant,)` (pevný krok, jako dřív), nebo z plánu
    chůze `gait.plan`: `(weight, speed, level, gap, ratio, bright, pitch,
    variant)` — váha a rychlost dávají fyziku (`gait.physics`), ostatní
    jsou korelované násobky kolem 1. Hlasitost pak není normalizovaná:
    těžší a rychlejší chodec je hlasitější.
    """
    if surface not in SURFACES:
        raise ValueError(f"neznámý povrch {surface!r}; k dispozici: {', '.join(SURFACES)}")
    spec = SURFACES[surface]
    n = int(0.35 * sr)
    if len(rest) == 1:
        rng = _rng(29, rest[0])
        y = _strike(n, sr, spec, 1.0, rng)
        gap = int(rng.uniform(0.035, 0.07) * sr)
        y[gap:] += _strike(n - gap, sr, spec, rng.uniform(0.5, 0.7), rng)
        y = _peak(y, 0.8)
    elif len(rest) == 8:
        weight, speed, level, gap_k, ratio_k, bright, pitch, variant = rest
        if not gait.MIN_KG <= float(weight) <= gait.MAX_KG:
            raise ValueError(f"váha chodce {weight!r} kg mimo "
                             f"{gait.MIN_KG:.0f}–{gait.MAX_KG:.0f}")
        if not gait.MIN_SPEED <= float(speed) <= gait.MAX_SPEED:
            raise ValueError(f"rychlost {speed!r} m/s mimo "
                             f"{gait.MIN_SPEED}–{gait.MAX_SPEED}")
        ph = gait.physics(weight, speed)
        rng = _rng(29, variant)
        tone = dict(pitch=ph.pitch * float(pitch), bright=ph.bright * float(bright),
                    decay=ph.decay)
        y = _strike(n, sr, spec, 1.0, rng, **tone)
        gap = int(np.clip(ph.gap * float(gap_k) * rng.uniform(0.9, 1.1), 0.005, 0.2) * sr)
        y[gap:] += _strike(n - gap, sr, spec,
                           ph.ratio * float(ratio_k) * rng.uniform(0.9, 1.1), rng, **tone)
        y = _ceiling(y * (STEP_GAIN[surface] * ph.level * float(level)))
    else:
        raise ValueError(f"step: čekám (variant,) nebo 8 parametrů chůze, ne {rest!r}")
    left, right = (1.0, 0.85) if int(foot) % 2 == 0 else (0.85, 1.0)
    return _stereo(_edges(y, sr, 0.001), left, right)


# --------------------------------------------------------------------------
# hlas: hlasivky (oscillators.glottal) -> trakt (filters.Tract)
# --------------------------------------------------------------------------

# výchozí délky gest v sekundách a pohodlná výška (MIDI), kolem které
# HumanTrack hledá tón akordu; smích je výrazně nad mluvenou výškou
# (Bachorowski et al. 2001: muži ~280 Hz, ženy ~420 Hz)
GESTURE_SEC = {"hum": 1.5, "sigh": 1.3, "effort": 0.3}
GESTURE_CENTER = {"hum": {"male": 48, "female": 57}, "sigh": {"male": 52, "female": 62},
                  "effort": {"male": 50, "female": 60}, "laugh": {"male": 60, "female": 67}}
VOICE_RMS = {"hum": 0.06, "sigh": 0.07, "effort": 0.14, "laugh": 0.1}   # při vel 127
TRACT_VOICE = {"male": "tenor", "female": "alto"}
# přestřel výšky po změně cíle: tlumený systém 2. řádu (Saitou 2007)
OVERSHOOT_HZ, OVERSHOOT_DAMP = 4.5, 0.55
DRIFT_TAU = 0.35       # s: bloudění výšky (OU)
FINE_HZ = 10.0         # jemné chvění nad touhle frekvencí


def pitch_curve(n: int, sr: int, points, rng, drift: float = 6.0, fine: float = 2.5,
                settle: bool = True) -> np.ndarray:
    """MIDI výška hlasu po samplech.

    `points` = `[(podíl délky 0..1, MIDI), ...]` je zamýšlená melodie (i
    glissanda). Na ni: přestřel a dojezd po změně cíle (2. řád, `settle`),
    bloudění kolem cíle o `drift` centů (Ornstein–Uhlenbeck, τ 0.35 s) a
    jemné chvění nad 10 Hz o `fine` centů (Saitou: fine fluctuation).
    """
    u = (np.arange(n) + 0.5) / max(1, n)
    target = np.interp(u, [p[0] for p in points], [float(p[1]) for p in points])
    y = target
    if settle and n > 1:
        w0 = 2.0 * math.pi * OVERSHOOT_HZ
        b, a = signal.bilinear([w0 * w0], [1.0, 2.0 * OVERSHOOT_DAMP * w0, w0 * w0], sr)
        y = signal.lfilter(b, a, target - target[0]) + target[0]
    y = y + float(drift) / 100.0 * dsp.ar1(n, DRIFT_TAU, sr, seed=int(rng.integers(1 << 30)))
    if fine:
        f = dsp.filt(rng.standard_normal(n), "hp", FINE_HZ, sr).astype(np.float64)
        y = y + float(fine) / 100.0 * f / max(1e-12, float(np.sqrt(np.mean(f ** 2))))
    return y


def _voiced(sr: int, midi: np.ndarray, rng, voice: str, vowel: str = "a", rd=1.0,
            aspiration: float = 0.08, morph=None, nasal: float = 0.0,
            jitter: float = 0.008, shimmer: float = 0.025,
            doubling: float = 0.0) -> np.ndarray:
    """Hlasivky -> trakt pro danou výškovou křivku (MIDI po samplech)."""
    f0 = 440.0 * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)
    src = glottal(f0, f0.size, sr, rd=rd, jitter=jitter, shimmer=shimmer,
                  aspiration=aspiration, doubling=doubling,
                  seed=int(rng.integers(1 << 30))).astype(np.float64)
    tract = Tract(vowel, voice=TRACT_VOICE[voice], morph=morph, nasal=nasal,
                  vary=0.02)
    return tract.apply(src, sr, freq=float(np.median(f0))).astype(np.float64)


def _level(y: np.ndarray, name: str) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    return y * (VOICE_RMS[name] / rms) if rms > 0 else y


def _ramp(n: int, sr: int, attack: float, release: float) -> np.ndarray:
    env = np.ones(n)
    a, r = min(n // 2, int(attack * sr)), min(n // 2, int(release * sr))
    if a > 0:
        env[:a] = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, a))
    if r > 0:
        env[-r:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, math.pi, r))
    return env


def _check_voice(voice: str, vowel: str = "a") -> None:
    if voice not in HUMAN_VOICES:
        raise ValueError(f"neznámý hlas {voice!r}; k dispozici: {', '.join(HUMAN_VOICES)}")
    if vowel not in VOWELS:
        raise ValueError(f"neznámá samohláska {vowel!r}; k dispozici: {', '.join(VOWELS)}")


def _hum(sr, bpm, midi, beats, voice, variant):
    """Bručení „mmm“ se zavřenými rty: nosní murmur, měkký nástup s malým
    nájezdem zdola, na konci výška lehce klesne."""
    _check_voice(voice)
    rng = _rng(41, variant)
    n = max(64, int(float(beats) * 60.0 / float(bpm) * sr))
    m = float(midi)
    pitch = pitch_curve(n, sr, [(0.0, m - 0.4), (0.12, m), (0.85, m), (1.0, m - 0.5)], rng)
    y = _voiced(sr, pitch, rng, voice, "u", rd=rng.uniform(1.0, 1.3),
                aspiration=0.04, nasal=1.0)
    y *= _ramp(n, sr, 0.08, 0.18) * (1.0 - 0.15 * np.linspace(0.0, 1.0, n))
    return _stereo(_edges(_ceiling(_level(y, "hum")), sr))


def _sigh(sr, bpm, midi, drop, beats, voice, variant):
    """Vzdech s hlasem: „h“ na začátku, dyšný hlas klesající o `drop`
    půltónů, čelist se přivírá (a -> o), síla dojde jako výdech."""
    _check_voice(voice)
    rng = _rng(43, variant)
    n = max(64, int(float(beats) * 60.0 / float(bpm) * sr))
    m = float(midi)
    pitch = pitch_curve(n, sr, [(0.0, m), (0.12, m + 0.3), (1.0, m - float(drop))], rng,
                        drift=8.0)
    rd = np.linspace(1.2, 2.0, n)
    sec = n / sr
    y = _voiced(sr, pitch, rng, voice, "a", rd=rd, aspiration=0.15,
                morph=("a", "o", sec))
    t = np.arange(n) / sr
    voicing = np.clip((t - 0.05) / 0.08, 0.0, 1.0) * np.exp(-t / (0.45 * sec))
    air = _airflow(sr, sec, 0.7, "mouth", "a", voice, variant, inhale=False)[:, 0]
    y = y * voicing / max(1e-12, float(np.sqrt(np.mean((y * voicing) ** 2))))
    air = air / max(1e-12, float(np.sqrt(np.mean(air ** 2))))
    # vzduch převáží jen na začátku („h“); HNR vzdechu ~10 dB = dyšný, ale znělý
    out = (y + 0.3 * air[:n]) * _ramp(n, sr, 0.01, 0.2)
    return _stereo(_edges(_ceiling(_level(out, "sigh")), sr))


def _effort(sr, bpm, midi, beats, vowel, voice, variant):
    """Námaha („hm!“, zvednutí, úder): tlačený hlas (Rd 0.35), rázové
    nasazení, hrubost (jitter, shimmer, zdvojení periody), výška padá."""
    _check_voice(voice, vowel)
    rng = _rng(47, variant)
    n = max(64, int(float(beats) * 60.0 / float(bpm) * sr))
    m = float(midi)
    pitch = pitch_curve(n, sr, [(0.0, m + 1.0), (0.2, m), (1.0, m - 3.0)], rng,
                        drift=15.0, fine=6.0, settle=False)
    y = _voiced(sr, pitch, rng, voice, vowel, rd=0.35, aspiration=0.1,
                jitter=0.025, shimmer=0.08, doubling=rng.uniform(0.2, 0.35))
    t = np.arange(n) / sr
    env = np.minimum(1.0, t / 0.004) * (1.0 - 0.4 * np.clip(t / (0.6 * n / sr), 0.0, 1.0))
    y = y * env * _ramp(n, sr, 0.0, 0.06)
    return _stereo(_edges(_ceiling(_level(y, "effort")), sr, 0.001))


def _laugh(sr, bpm, midi, calls, rate, vowel, voice, variant):
    """Smích: série „ha“ (`calls` za sebou, `rate` za sekundu), každé
    s přídechem „h“; výška i síla během série klesají, hlas dyšní."""
    _check_voice(voice, vowel)
    calls = max(1, int(calls))
    rate = float(np.clip(rate, 2.0, 8.0))
    rng = _rng(53, variant)
    period = 1.0 / rate
    starts = np.cumsum(np.r_[0.0, period * (1.0 + 0.08 * rng.standard_normal(calls - 1))])
    sec = float(starts[-1]) + period + 0.12
    n = int(sec * sr)
    t = np.arange(n) / sr
    m = float(midi)
    points = []
    for i, s0 in enumerate(starts):
        top = m - 0.7 * i
        points += [(s0 / sec, top + 0.4), (min(1.0, (s0 + 0.09) / sec), top - 1.0)]
    points.sort()
    pitch = pitch_curve(n, sr, points, rng, drift=10.0, settle=False)
    gate = np.zeros(n)
    breath_gate = np.zeros(n)
    for i, s0 in enumerate(starts):
        v_len = 0.09 * rng.uniform(0.85, 1.15)
        on = s0 + 0.035
        w = np.clip((t - on) / v_len, 0.0, 1.0)
        call = np.where((t >= on) & (t < on + v_len), np.sin(np.pi * w) ** 0.6, 0.0)
        gate += call * 0.85 ** i
        h = np.clip((t - s0) / 0.035, 0.0, 1.0)
        breath_gate += np.where((t >= s0) & (t < on + 0.02), np.sin(np.pi * h) ** 2, 0.0) * 0.85 ** i
    rd = np.linspace(1.2, 1.9, n)
    y = _voiced(sr, pitch, rng, voice, vowel, rd=rd, aspiration=0.15) * gate
    noise = dsp.cascade(rng.standard_normal(n), [(f, b * 3.0) for f, b in
                        zip(FORMANTS["tenor"][vowel][0][:4], FORMANTS["tenor"][vowel][2][:4])], sr)
    noise = noise / max(1e-12, float(np.sqrt(np.mean(noise ** 2)))) * breath_gate
    y = y / max(1e-12, float(np.sqrt(np.mean(y ** 2)))) + 0.4 * noise
    return _stereo(_edges(_ceiling(_level(y, "laugh")), sr))


# --------------------------------------------------------------------------
# klipy z korpusu (corpus.py, fáze H4)
# --------------------------------------------------------------------------

CLIP_RMS = 0.08        # RMS aktivní části klipu při vel 127 (korpus má -20 dBFS)
CLIP_SHIFT = 6.0       # půltónů: víc PSOLA neposune, zní to pak uměle


def _clip(sr, bpm, kind, cid, midi, beats, variant):
    """Klip z korpusu: volitelně posunutý PSOLA na tón `midi` (oktávu vybere
    nejblíž vlastní výšce klipu, formanty zůstanou), převzorkovaný na `sr`
    a zkrácený na `beats` s 30 ms doběhem."""
    from . import corpus, voices
    entry = corpus.clip_entry(cid)
    csr, x = corpus.read_clip(cid)
    x = np.array(x, dtype=np.float64)
    if midi is not None and entry.get("pitch") is not None:
        own = float(entry["pitch"])
        target = float(midi) + 12.0 * round((own - float(midi)) / 12.0)
        step = float(np.clip(target - own, -CLIP_SHIFT, CLIP_SHIFT))
        pitch, _, hop = voices.track_pitch(x, csr)
        if abs(step) > 0.01 and int(np.sum(~np.isnan(pitch))) >= 5:
            x = voices.psola(x, csr, pitch, hop, step)
    if csr != sr:
        g = math.gcd(int(sr), int(csr))
        x = signal.resample_poly(x, int(sr) // g, int(csr) // g)
    if beats:
        n = max(16, int(float(beats) * 60.0 / float(bpm) * sr))
        if x.size > n:
            x = x[:n].copy()
            tail = min(n // 2, int(0.03 * sr))
            x[-tail:] *= np.linspace(1.0, 0.0, tail)
    y = x * (CLIP_RMS / 10.0 ** (-20.0 / 20.0))
    return _stereo(_edges(_ceiling(y), sr, 0.002))


GENERATORS = {
    "breath": _breath,
    "exhale": _exhale,
    "lip_smack": _lip_smack,
    "heartbeat": _heartbeat,
    "cloth": _cloth,
    "step": _step,
    "hum": _hum,
    "sigh": _sigh,
    "effort": _effort,
    "laugh": _laugh,
    "clip": _clip,
}
