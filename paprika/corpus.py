"""Korpus lidských zvuků: klipy z neuronových modelů nebo vlastních nahrávek.

Smích, kašel, lapání po dechu nebo šustění oblečení umí neuronové modely
(TTS s neverbálními tagy jako Dia nebo Chatterbox, text -> audio jako Stable
Audio) líp než procedurální syntéza. Paprika je ale nespouští: jádro zůstává
numpy + scipy a render deterministický. Modely běží jednou, offline, ve
workeru `tools/human_worker.py` (vlastní venv s torch). Worker zapíše WAV
a ke každému JSON se zdrojem (backend, model, prompt, seed, licence) a tento
modul klipy:

1. **prosije** — délka podle druhu, ticho, clipping, podíl znělých rámců
   (YIN), u smíchu rytmus impulzů; když worker přidal štítky klasifikátoru
   (PANNs/AudioSet) nebo přepis řeči (Whisper), vyřadí i klipy, které nejsou
   tím druhem, nebo v nich někdo mluví; nakonec duplicity,
2. **ořeže** ticho na krajích a **srovná** hlasitost (RMS −20 dBFS),
3. **uloží** do `<vzorky>/human/<druh>/<id>.wav` a zapíše `manifest.json`
   a `CREDITS-human.txt` (licence a atribuce každého klipu).

Bez licence se klip nepřijme — u vlastních nahrávek ji musí zadat autor.
`HumanTrack.clip()` pak bere z korpusu varianty; chybí-li, letí
`SamplesMissing` s příkazem, žádný tichý návrat k syntéze.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import re
from collections import Counter

import numpy as np
from scipy import signal
from scipy.io import wavfile

from . import voices
from .library import SamplesMissing, samples_root

__all__ = ["KINDS", "CORPUS_KINDS", "MANIFEST", "MissingCorpus", "corpus_root", "analyze_clip",
           "check", "prepare", "ingest", "clips", "read_clip", "status_lines",
           "request", "credit_line", "clip_entry", "PROMPTS", "BACKENDS",
           "VOCAL_KINDS"]

MANIFEST = "manifest.json"
CREDITS = "CREDITS-human.txt"
VERSION = 1
TARGET_RMS_DB = -20.0     # aktivní část klipu
PEAK_CEILING = 0.9
TRIM_DB = 35.0            # ticho na krajích: rámce víc než tolik pod nejhlasitějším
SILENT_DB = -50.0         # klip, jehož nejhlasitější rámec je pod tímhle, je ticho
DUPLICATE = 0.98          # korelace otisku nad tímhle (a délka do 2 %) = duplicita
DUP_DUR = 0.02
FP_GRID = np.arange(20.0, 8000.0, 20.0)
LABEL_MIN = 0.2           # klasifikátor: nejnižší skóre správného druhu
SPEECH_OVER = 0.5         # klasifikátor: řeč nad tímhle a nad druhem = mluví

# druh -> (délka s), (podíl znělých rámců), rytmus impulzů za s (jen smích),
# štítky AudioSet, které mu odpovídají
KINDS = {
    "laugh": {"dur": (0.4, 5.0), "voiced": (0.1, 0.95), "bursts": (2.0, 9.0),
              "labels": ("Laughter", "Giggle", "Chuckle, chortle", "Belly laugh",
                         "Snicker", "Baby laughter")},
    "cough": {"dur": (0.15, 2.5), "voiced": (0.0, 0.6), "labels": ("Cough",)},
    "throat": {"dur": (0.2, 2.0), "voiced": (0.0, 0.8), "labels": ("Throat clearing",)},
    "sigh": {"dur": (0.4, 3.5), "voiced": (0.0, 0.95), "labels": ("Sigh",)},
    "gasp": {"dur": (0.1, 1.5), "voiced": (0.0, 0.5), "labels": ("Gasp",)},
    "sniff": {"dur": (0.08, 1.2), "voiced": (0.0, 0.25), "labels": ("Sniff",)},
    "scream": {"dur": (0.3, 4.0), "voiced": (0.3, 1.0), "labels": ("Screaming", "Shout",
                                                                   "Yell")},
    "hum": {"dur": (0.5, 8.0), "voiced": (0.5, 1.0), "labels": ("Humming",)},
    "cloth": {"dur": (0.2, 6.0), "voiced": (0.0, 0.15), "labels": ("Rustle",)},
    # tlumené rezonance podlahy YIN čte jako výšku: znělost kroky neodliší
    "steps": {"dur": (0.3, 10.0), "voiced": (0.0, 0.95), "labels": ("Walk, footsteps",)},
}
CORPUS_KINDS = tuple(KINDS)
# druhy, které dělá hlas (interpret nemůže zároveň zpívat nebo foukat)
VOCAL_KINDS = ("laugh", "cough", "throat", "sigh", "gasp", "sniff", "scream", "hum")
# citoslovce smíchu a vzdechů, která v přepisu řeči nevadí
_FILLERS = re.compile(r"^(h+[aeiou]+h*|[aeiou]*h+|hm+|mm+|uh+|ah+|oh+|eh+|ugh)$")

# výchozí prompty pro worker podle backendu; Dia a Chatterbox dostávají
# neverbální tagy, Stable Audio textový popis zvuku
PROMPTS = {
    "dia": {"laugh": "[S1] (laughs)", "cough": "[S1] (coughs)",
            "throat": "[S1] (clears throat)", "sigh": "[S1] (sighs)",
            "gasp": "[S1] (gasps)", "sniff": "[S1] (sniffs)",
            "scream": "[S1] (screams)", "hum": "[S1] (humming)"},
    "chatterbox": {"laugh": "[laugh]", "cough": "[cough]"},
    "stable_audio": {
        "laugh": "one person laughing, close microphone, dry studio recording, no music",
        "cough": "one person coughing, close microphone, dry studio recording",
        "sigh": "a tired person sighing, breathy exhale, close microphone, dry",
        "gasp": "a person gasping in surprise, short sharp inhale, close microphone",
        "sniff": "a person sniffing through the nose, close microphone, dry",
        "cloth": "clothing rustle, cotton jacket movement, close microphone, dry foley",
        "steps": "footsteps on a wooden floor, slow walk, close microphone, dry foley",
    },
    "paprika": {"laugh": "laugh", "sigh": "sigh", "hum": "hum", "cloth": "cloth",
                "steps": "steps"},
}
BACKENDS = tuple(PROMPTS)


class MissingCorpus(SamplesMissing):
    """Korpus lidských zvuků (nebo daný druh) není naplněný."""


# --------------------------------------------------------------------------
# analýza a síto
# --------------------------------------------------------------------------

def corpus_root(root: str | None = None) -> str:
    return os.path.join(os.path.abspath(root or samples_root()), "human")


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


def _frames_db(x: np.ndarray, sr: int, hop: float = 0.01) -> tuple[np.ndarray, int]:
    h = max(1, int(hop * sr))
    nb = max(1, x.size // h)
    seg = x[:nb * h].reshape(nb, h)
    return 20.0 * np.log10(np.sqrt(np.mean(seg ** 2, axis=1)) + 1e-9), h


def _fingerprint(x: np.ndarray, sr: int) -> np.ndarray:
    """Otisk klipu pro duplicity: jemné průměrné spektrum do 8 kHz (vidí
    jednotlivé harmonické, takže dvě bručení v jiné výšce se liší) na pevné
    mřížce 20 Hz a průběh hlasitosti v 32 bodech; obojí z-normalizované."""
    nper = 1 << int(round(math.log2(0.093 * sr)))
    f, p = signal.welch(x, sr, nperseg=min(nper, x.size))
    spec = np.interp(FP_GRID, f, 10.0 * np.log10(p + 1e-12))
    db, _ = _frames_db(x, sr)
    env = np.interp(np.linspace(0.0, 1.0, 32), np.linspace(0.0, 1.0, db.size), db)

    def z(v):
        return (v - v.mean()) / max(1e-9, float(v.std()))
    return np.concatenate([z(spec), z(env)])


def analyze_clip(x: np.ndarray, sr: int) -> dict:
    """Vlastnosti klipu: délka, hlasitost, clipping, znělost, výška (medián
    znělých rámců v MIDI, None u neznělého klipu), rytmus impulzů za s."""
    x = _mono(x)
    db, h = _frames_db(x, sr)
    peak_db = float(db.max()) if db.size else -180.0
    active = db > peak_db - TRIM_DB
    info = {"dur": x.size / sr, "peak": float(np.abs(x).max()) if x.size else 0.0,
            "peak_db": peak_db, "clipped": int(np.sum(np.abs(x) >= 0.999)),
            "voiced": 0.0, "pitch": None, "bursts": 0.0}
    if x.size < int(0.05 * sr) or peak_db < SILENT_DB:
        return info
    pitch, _, hop = voices.track_pitch(x, sr)
    if pitch.size:
        idx = np.minimum((np.arange(pitch.size) * hop * sr / h).astype(int), active.size - 1)
        act = active[idx]
        voiced = ~np.isnan(pitch) & act
        info["voiced"] = float(voiced.sum() / max(1, act.sum()))
        if voiced.sum() >= 5 and info["voiced"] >= 0.2:
            info["pitch"] = round(float(np.median(pitch[voiced])), 2)
    env = signal.lfilter([0.3], [1.0, -0.7], db)
    peaks, _ = signal.find_peaks(env, distance=max(1, int(0.1 / 0.01)), prominence=6.0)
    info["bursts"] = len(peaks) / info["dur"]
    return info


def check(info: dict, kind: str, meta: dict | None = None) -> str | None:
    """Důvod, proč klip do korpusu nepatří, nebo None."""
    spec = KINDS[kind]
    meta = meta or {}
    if info["peak_db"] < SILENT_DB:
        return "ticho"
    if info["clipped"] > 3:
        return "clipping"
    lo, hi = spec["dur"]
    if not lo <= info["dur"] <= hi:
        return "krátký" if info["dur"] < lo else "dlouhý"
    vlo, vhi = spec["voiced"]
    if not vlo <= info["voiced"] <= vhi:
        return "málo znělý" if info["voiced"] < vlo else "moc znělý"
    if "bursts" in spec:
        blo, bhi = spec["bursts"]
        if not blo <= info["bursts"] <= bhi:
            return "rytmus"
    labels = meta.get("labels")
    if labels:
        own = max((float(labels.get(name, 0.0)) for name in spec["labels"]), default=0.0)
        speech = float(labels.get("Speech", 0.0))
        if speech > SPEECH_OVER and speech > own:
            return "řeč (klasifikátor)"
        if own < LABEL_MIN:
            return "jiný zvuk (klasifikátor)"
    text = meta.get("transcript")
    if text:
        words = [w for w in re.findall(r"[^\W\d_]+", str(text).lower()) if not _FILLERS.match(w)]
        if len(words) >= 2:
            return "řeč (přepis)"
    return None


def prepare(x: np.ndarray, sr: int) -> np.ndarray:
    """Ořez ticha na krajích (20 ms rezerva), 5 ms náběh a doběh, RMS aktivní
    části na −20 dBFS se stropem špičky 0.9."""
    x = _mono(x)
    db, h = _frames_db(x, sr)
    active = np.flatnonzero(db > db.max() - TRIM_DB)
    margin = int(0.02 * sr)
    a = max(0, active[0] * h - margin)
    b = min(x.size, (active[-1] + 1) * h + margin)
    y = x[a:b].copy()
    fade = min(int(0.005 * sr), y.size // 4)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    db, h = _frames_db(y, sr)
    act = db > db.max() - TRIM_DB
    seg = y[:act.size * h].reshape(act.size, h)[act]
    rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
    if rms > 0:
        y *= 10.0 ** (TARGET_RMS_DB / 20.0) / rms
    peak = float(np.abs(y).max()) if y.size else 0.0
    return y * (PEAK_CEILING / peak) if peak > PEAK_CEILING else y


# --------------------------------------------------------------------------
# manifest a import
# --------------------------------------------------------------------------

_CACHE: dict = {}


def _read_manifest(root: str | None = None) -> dict:
    path = os.path.join(corpus_root(root), MANIFEST)
    if not os.path.isfile(path):
        return {"version": VERSION, "clips": []}
    key = (path, os.path.getmtime(path))
    if key not in _CACHE:
        with open(path, encoding="utf-8") as f:
            _CACHE.clear()
            _CACHE[key] = json.load(f)
    return _CACHE[key]


def _write(base: str, data: dict) -> None:
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, MANIFEST), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    lines = ["Lidské zvuky v korpusu Paprika (human/)",
             "========================================", "",
             "Při použití ve hře uveď u každého zdroje jeho licenci a autora.", ""]
    groups: dict = {}
    for c in data["clips"]:
        groups.setdefault((c["license"], c["attribution"]), []).append(c)
    for (lic, attr), cs in sorted(groups.items()):
        kinds = ", ".join(sorted({c["kind"] for c in cs}))
        lines += [f"{attr}", f"  Licence: {lic}", f"  Klipů: {len(cs)} ({kinds})", ""]
    with open(os.path.join(base, CREDITS), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    _CACHE.clear()


def _load_wav(path: str) -> tuple[int, np.ndarray]:
    sr, d = wavfile.read(path)
    x = d.astype(np.float64)
    if d.dtype == np.int16:
        x /= 32768.0
    elif d.dtype == np.int32:
        x /= 2147483648.0
    return int(sr), _mono(x)


def ingest(folder: str, root: str | None = None, kind: str | None = None,
           license: str | None = None, attribution: str | None = None,
           log=print) -> dict:
    """Prosije WAV klipy ze `folder` a přijaté uloží do korpusu.

    Ke klipu `x.wav` patří `x.json` od workeru (`kind`, `backend`, `model`,
    `prompt`, `seed`, `license`, `attribution`, volitelně `labels` a
    `transcript`). Vlastní nahrávky bez JSON potřebují `kind` a `license`
    (a `attribution`) z argumentů. Vrací `{"accepted": [id], "rejected":
    {soubor: důvod}}`.
    """
    base = corpus_root(root)
    data = json.loads(json.dumps(_read_manifest(root)))     # kopie
    prints = {}
    for c in data["clips"]:
        prints.setdefault(c["kind"], []).append((np.asarray(c["print"]), c["dur"]))
    known = {c["id"] for c in data["clips"]}
    accepted, rejected = [], {}
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(".wav"):
            continue
        path = os.path.join(folder, fn)
        side = os.path.splitext(path)[0] + ".json"
        meta = {}
        if os.path.isfile(side):
            with open(side, encoding="utf-8") as f:
                meta = json.load(f)
        k = meta.get("kind") or kind
        lic = meta.get("license") or license
        attr = meta.get("attribution") or attribution or lic
        if k not in KINDS:
            rejected[fn] = f"neznámý druh {k!r} (k dispozici: {', '.join(KINDS)})"
            continue
        if not lic:
            rejected[fn] = "chybí licence (--license)"
            continue
        try:
            sr, x = _load_wav(path)
        except ValueError as e:
            rejected[fn] = f"nečitelný WAV ({e})"
            continue
        raw = analyze_clip(x, sr)
        if raw["peak_db"] < SILENT_DB:
            rejected[fn] = "ticho"
            continue
        if raw["clipped"] > 3:
            rejected[fn] = "clipping"
            continue
        # délka, znělost a rytmus až po ořezu ticha (TTS přidává ticho na konce)
        y = prepare(x, sr)
        info = analyze_clip(y, sr)
        why = check(info, k, meta)
        if why:
            rejected[fn] = why
            continue
        cid = hashlib.sha1(np.round(y * 32767).astype(np.int16).tobytes()).hexdigest()[:16]
        fp = _fingerprint(y, sr)
        dur = y.size / sr
        if cid in known or any(abs(d - dur) <= DUP_DUR * dur
                               and float(np.dot(fp, q)) / fp.size > DUPLICATE
                               for q, d in prints.get(k, [])):
            rejected[fn] = "duplicitní"
            continue
        rel = f"{k}/{cid}.wav"
        os.makedirs(os.path.join(base, k), exist_ok=True)
        wavfile.write(os.path.join(base, *rel.split("/")), sr,
                      (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16))
        data["clips"].append({
            "id": cid, "kind": k, "file": rel, "sr": sr, "dur": round(y.size / sr, 3),
            "pitch": info["pitch"] if k in VOCAL_KINDS else None,
            "voiced": round(info["voiced"], 3),
            "license": lic, "attribution": attr,
            "source": {key: meta.get(key) for key in ("backend", "model", "prompt", "seed")}
            | {"file": fn},
            "added": datetime.date.today().isoformat(),
            "print": np.round(fp, 2).tolist()})
        known.add(cid)
        prints.setdefault(k, []).append((fp, dur))
        accepted.append(cid)
    _write(base, data)
    log(f"korpus: přijato {len(accepted)}, odmítnuto {len(rejected)}"
        + ("" if not rejected else " — " + ", ".join(
            f"{why} {n}" for why, n in Counter(rejected.values()).most_common())))
    return {"accepted": accepted, "rejected": rejected}


def clips(kind: str, root: str | None = None) -> list[dict]:
    """Klipy daného druhu seřazené podle id; prázdný druh -> `MissingCorpus`."""
    if kind not in KINDS:
        raise ValueError(f"neznámý druh lidského zvuku {kind!r}; k dispozici: "
                         f"{', '.join(KINDS)}")
    found = sorted((c for c in _read_manifest(root)["clips"] if c["kind"] == kind),
                   key=lambda c: c["id"])
    if not found:
        raise MissingCorpus(
            f"korpus lidských zvuků nemá žádný klip druhu {kind!r} ({corpus_root(root)}) — "
            f"vygeneruj ho workerem a naimportuj: python -m paprika samples human "
            f"request {kind} ...; python tools/human_worker.py ...; python -m paprika "
            f"samples human import <složka>")
    return found


_AUDIO: dict = {}


def read_clip(cid: str, root: str | None = None) -> tuple[int, np.ndarray]:
    """Audio klipu podle id (mono float64), s cache v procesu."""
    for c in _read_manifest(root)["clips"]:
        if c["id"] == cid:
            path = os.path.join(corpus_root(root), *c["file"].split("/"))
            if path not in _AUDIO:
                if not os.path.isfile(path):
                    raise MissingCorpus(f"klip {cid} chybí na disku ({path}) — obnov korpus "
                                        f"importem")
                _AUDIO[path] = _load_wav(path)
            return _AUDIO[path]
    raise MissingCorpus(f"klip {cid} v korpusu {corpus_root(root)} není — skladba "
                        f"čeká jiný korpus; naimportuj ho znovu")


def status_lines(root: str | None = None) -> list[str]:
    data = _read_manifest(root)
    lines = [f"korpus lidských zvuků: {corpus_root(root)}"]
    by_kind = Counter(c["kind"] for c in data["clips"])
    for k in KINDS:
        n = by_kind.get(k, 0)
        lic = sorted({c["license"] for c in data["clips"] if c["kind"] == k})
        lines.append(f"{k:<8} {n:>4} klipů  {', '.join(lic) if lic else '—'}")
    lines.append("naplnění: python -m paprika samples human request <druh> --backend dia; "
                 "python tools/human_worker.py <request.json> --out <složka>; "
                 "python -m paprika samples human import <složka>")
    return lines


def request(kinds, backend: str = "dia", count: int = 8, seed: int = 0,
            duration: float | None = None, model: str | None = None) -> dict:
    """Zadání pro worker: pro každý druh `count` generací s různými seedy."""
    if backend not in PROMPTS:
        raise ValueError(f"neznámý backend {backend!r}; k dispozici: {', '.join(BACKENDS)}")
    jobs = []
    for k in kinds:
        if k not in KINDS:
            raise ValueError(f"neznámý druh {k!r}; k dispozici: {', '.join(KINDS)}")
        if k not in PROMPTS[backend]:
            raise ValueError(f"backend {backend!r} druh {k!r} neumí; zkus "
                             + ", ".join(b for b in BACKENDS if k in PROMPTS[b]))
        for i in range(int(count)):
            job = {"kind": k, "backend": backend, "prompt": PROMPTS[backend][k],
                   "seed": int(seed) + i}
            if model:
                job["model"] = model
            if duration:
                job["duration"] = float(duration)
            jobs.append(job)
    return {"version": VERSION, "jobs": jobs}


def clip_entry(cid: str, root: str | None = None) -> dict:
    """Záznam manifestu pro id klipu."""
    for c in _read_manifest(root)["clips"]:
        if c["id"] == cid:
            return c
    raise MissingCorpus(f"klip {cid} v korpusu {corpus_root(root)} není — skladba "
                        f"čeká jiný korpus; naimportuj ho znovu")


def credit_line(clip: dict) -> str:
    return f"{clip['attribution']} ({clip['license']})"

