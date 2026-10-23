"""Worker pro korpus lidských zvuků (fáze H4): spustí neuronové modely a
zapíše WAV + JSON, které pak prosije a naimportuje Paprika.

Běží MIMO jádro Paprika, ideálně ve vlastním venv s torch a s GPU:

    python -m venv .venv-human
    .venv-human/bin/pip install numpy scipy torch
    .venv-human/bin/pip install git+https://github.com/nari-labs/dia.git   # --backend dia
    .venv-human/bin/pip install chatterbox-tts                             # --backend chatterbox
    .venv-human/bin/pip install diffusers transformers accelerate          # --backend stable_audio
    .venv-human/bin/pip install panns-inference librosa                    # --classify (AudioSet)
    .venv-human/bin/pip install openai-whisper                             # --asr (přepis řeči)

    python -m paprika samples human request laugh cough --backend dia --count 12 --out req.json
    .venv-human/bin/python tools/human_worker.py req.json --out incoming/ --classify --asr
    python -m paprika samples human import incoming/

Backend `paprika` nic dalšího nepotřebuje: klipy dělá vestavěná syntéza
(`paprika.human`), takže jde ověřit celý protokol bez GPU. Adaptéry `dia`,
`chatterbox` a `stable_audio` jsou napsané podle README projektů a v cloudové
session, kde vznikly, nebyly spuštěné (bez GPU a bez přístupu k Hugging
Face) — první běh u sebe zkontroluj. Licence modelů ověř v jejich
repozitářích; worker je k JSON zapíše tak, jak jsou v `LICENSES`.

Každá generace má klíč (hash backendu, modelu, promptu, seedu a délky):
když WAV a JSON s tím klíčem už ve výstupní složce jsou, přeskočí se.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import random
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER_VERSION = 1

# backend -> (atribuce, licence); generovaný klip nese obojí do korpusu
LICENSES = {
    "paprika": ("Paprika, vestavěná syntéza", "bez omezení"),
    "dia": ("Generováno modelem Nari Labs Dia (nari-labs/Dia-1.6B)",
            "Apache-2.0 (ověř v repozitáři nari-labs/dia)"),
    "chatterbox": ("Generováno modelem Resemble AI Chatterbox",
                   "MIT; výstup nese neslyšitelný vodoznak PerTh"),
    "stable_audio": ("Generováno modelem Stability AI Stable Audio Open",
                     "Stability AI Community License (komerčně zdarma pod 1 mil. $ "
                     "ročního obratu)"),
}
DEFAULT_MODEL = {"paprika": "paprika.human", "dia": "nari-labs/Dia-1.6B",
                 "chatterbox": "turbo", "stable_audio": "stabilityai/stable-audio-open-1.0"}
INSTALL = {"dia": "pip install git+https://github.com/nari-labs/dia.git",
           "chatterbox": "pip install chatterbox-tts",
           "stable_audio": "pip install diffusers transformers accelerate",
           "panns": "pip install panns-inference librosa",
           "whisper": "pip install openai-whisper"}
NEGATIVE = "music, singing, speech, words, low quality, heavy reverb, noise"
# štítky AudioSet, které se do JSON zapíšou vždy (síto podle nich rozhoduje)
KEEP_LABELS = ("Speech", "Laughter", "Giggle", "Chuckle, chortle", "Belly laugh",
               "Snicker", "Baby laughter", "Cough", "Throat clearing", "Sigh",
               "Gasp", "Sniff", "Screaming", "Shout", "Yell", "Humming", "Rustle",
               "Walk, footsteps", "Breathing", "Music")

_MODELS: dict = {}


class WorkerError(RuntimeError):
    pass


def _need(name: str):
    try:
        return __import__(name)
    except ImportError:
        raise WorkerError(f"chybí balík pro {name!r} — nainstaluj ve venv workeru: "
                          f"{INSTALL.get(name, 'pip install ' + name)}")


def _cached(key: str, make):
    if key not in _MODELS:
        _MODELS[key] = make()
    return _MODELS[key]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# backendy: (job, seed, device) -> (audio (n,) nebo (n, k), sr)
# --------------------------------------------------------------------------

def gen_paprika(job: dict, seed: int, device: str):
    """Vestavěná syntéza Paprika — test protokolu bez GPU a bez modelů."""
    sys.path.insert(0, ROOT)
    from paprika import human
    sr, bpm = 44100, 120.0
    rng = np.random.default_rng(seed)
    kind = job["kind"]
    voice = "female" if seed % 2 else "male"
    if kind == "laugh":
        y = human._laugh(sr, bpm, 60.0 + rng.uniform(-3, 3), int(rng.integers(3, 7)),
                         rng.uniform(3.8, 5.2), "a", voice, seed)
    elif kind == "sigh":
        y = human._sigh(sr, bpm, 55.0 + rng.uniform(-3, 5), rng.uniform(3, 6),
                        rng.uniform(1.8, 3.0), voice, seed)
    elif kind == "hum":
        y = human._hum(sr, bpm, 50.0 + rng.uniform(-2, 7), rng.uniform(2.0, 5.0), voice, seed)
    elif kind == "cloth":
        y = human._cloth(sr, bpm, ("cotton", "nylon", "wool")[seed % 3], rng.uniform(1.0, 3.0),
                         ("swing", "fidget")[seed % 2], seed)
    elif kind == "steps":
        steps = [human._step(sr, bpm, "wood", i % 2, seed * 8 + i) for i in range(4)]
        gap = np.zeros((int(0.5 * sr) - steps[0].shape[0], 2))
        y = np.concatenate([np.concatenate([s, gap]) for s in steps])
    else:
        raise WorkerError(f"backend paprika neumí druh {kind!r}")
    return np.asarray(y, dtype=np.float32)[:, 0], sr


def gen_dia(job: dict, seed: int, device: str):
    """Nari Labs Dia: text se značkou mluvčího a neverbálním tagem, 44.1 kHz."""
    _need("dia")
    from dia.model import Dia
    name = job.get("model") or DEFAULT_MODEL["dia"]
    dtype = "float16" if device == "cuda" else "float32"
    model = _cached(f"dia:{name}", lambda: Dia.from_pretrained(name, compute_dtype=dtype))
    _seed_all(seed)
    audio = model.generate(job["prompt"], use_torch_compile=False, verbose=False)
    return np.asarray(audio, dtype=np.float32), 44100


def gen_chatterbox(job: dict, seed: int, device: str):
    """Resemble AI Chatterbox (Turbo umí tagy [laugh], [cough], [chuckle])."""
    _need("chatterbox")
    which = job.get("model") or DEFAULT_MODEL["chatterbox"]
    if which == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
    else:
        from chatterbox.tts import ChatterboxTTS as Model
    model = _cached(f"chatterbox:{which}", lambda: Model.from_pretrained(device=device))
    _seed_all(seed)
    kw = {"audio_prompt_path": job["voice"]} if job.get("voice") else {}
    wav = model.generate(job["prompt"], **kw)
    return wav.squeeze().detach().cpu().numpy().astype(np.float32), int(model.sr)


def gen_stable_audio(job: dict, seed: int, device: str):
    """Stable Audio Open přes diffusers: text -> zvuk (stereo 44.1 kHz)."""
    torch = _need("torch")
    _need("diffusers")
    from diffusers import StableAudioPipeline
    name = job.get("model") or DEFAULT_MODEL["stable_audio"]
    dtype = torch.float16 if device == "cuda" else torch.float32

    def make():
        return StableAudioPipeline.from_pretrained(name, torch_dtype=dtype).to(device)

    pipe = _cached(f"stable_audio:{name}", make)
    gen = torch.Generator(device).manual_seed(seed)
    out = pipe(job["prompt"], negative_prompt=NEGATIVE,
               num_inference_steps=int(job.get("steps", 100)),
               audio_end_in_s=float(job.get("duration", 3.0)),
               num_waveforms_per_prompt=1, generator=gen).audios
    audio = out[0].T.float().cpu().numpy()
    return audio.astype(np.float32), int(pipe.vae.sampling_rate)


BACKENDS = {"paprika": gen_paprika, "dia": gen_dia, "chatterbox": gen_chatterbox,
            "stable_audio": gen_stable_audio}


# --------------------------------------------------------------------------
# uši: klasifikátor AudioSet a přepis řeči
# --------------------------------------------------------------------------

def _mono_at(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    x = audio.mean(axis=1) if audio.ndim == 2 else audio
    g = np.gcd(int(sr), int(target))
    return resample_poly(x.astype(np.float64), target // g, sr // g).astype(np.float32)


def classify(audio: np.ndarray, sr: int, device: str) -> dict:
    """Skóre AudioSet (PANNs CNN14): deset nejvyšších + štítky důležité pro síto."""
    _need("panns_inference")
    from panns_inference import AudioTagging, labels
    at = _cached("panns", lambda: AudioTagging(checkpoint_path=None, device=device))
    scores, _ = at.inference(_mono_at(audio, sr, 32000)[None, :])
    scores = np.asarray(scores)[0]
    keep = set(np.argsort(scores)[::-1][:10]) | {i for i, n in enumerate(labels)
                                                if n in KEEP_LABELS}
    return {labels[i]: round(float(scores[i]), 4) for i in sorted(keep)}


def transcribe(audio: np.ndarray, sr: int, device: str) -> str:
    """Přepis Whisperem — síto podle něj pozná klip, ve kterém někdo mluví."""
    whisper = _need("whisper")
    model = _cached("whisper", lambda: whisper.load_model("base", device=device))
    result = model.transcribe(_mono_at(audio, sr, 16000), language="en",
                              fp16=(device == "cuda"))
    return str(result.get("text", "")).strip()


# --------------------------------------------------------------------------
# běh
# --------------------------------------------------------------------------

def job_key(job: dict) -> str:
    fields = {k: job.get(k) for k in ("backend", "model", "prompt", "seed", "duration",
                                      "steps", "voice", "kind")}
    return hashlib.sha1(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def run(request: dict, out: str, device: str = "auto", do_classify: bool = False,
        do_asr: bool = False, log=print) -> dict:
    """Zpracuje zadání; vrací počty `{"made", "cached", "failed"}`."""
    os.makedirs(out, exist_ok=True)
    dev = _device(device)
    counts = {"made": 0, "cached": 0, "failed": 0}
    for job in request.get("jobs", []):
        backend = job.get("backend")
        if backend not in BACKENDS:
            raise WorkerError(f"neznámý backend {backend!r}; k dispozici: "
                              f"{', '.join(BACKENDS)}")
        kind = str(job.get("kind", ""))
        if not kind.isidentifier():
            raise WorkerError(f"neplatný druh {kind!r} v zadání")
        key = job_key(job)
        base = os.path.join(out, f"{kind}_{backend}_{key}")
        if os.path.isfile(base + ".wav") and os.path.isfile(base + ".json"):
            counts["cached"] += 1
            continue
        audio, sr = BACKENDS[backend](job, int(job.get("seed", 0)), dev)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            counts["failed"] += 1
            log(f"  {os.path.basename(base)}: prázdný nebo neplatný výstup")
            continue
        attribution, lic = LICENSES[backend]
        meta = {"kind": job["kind"], "backend": backend,
                "model": job.get("model") or DEFAULT_MODEL[backend],
                "prompt": job.get("prompt"), "seed": job.get("seed"),
                "license": lic, "attribution": attribution,
                "generated": datetime.date.today().isoformat(),
                "worker": WORKER_VERSION}
        if do_classify:
            meta["labels"] = classify(audio, sr, dev)
        if do_asr:
            meta["transcript"] = transcribe(audio, sr, dev)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        wavfile.write(base + ".wav", int(sr), pcm)
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        counts["made"] += 1
    log(f"worker: vytvořeno {counts['made']}, z cache {counts['cached']}, "
        f"selhalo {counts['failed']} -> {out}")
    return counts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generuje lidské zvuky pro korpus Paprika.")
    p.add_argument("request", help="JSON ze `python -m paprika samples human request`")
    p.add_argument("--out", required=True, help="složka pro WAV + JSON")
    p.add_argument("--device", default="auto", help="cuda, cpu, mps nebo auto")
    p.add_argument("--classify", action="store_true", help="štítky AudioSet (PANNs)")
    p.add_argument("--asr", action="store_true", help="přepis řeči (Whisper)")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass
    try:
        with open(args.request, encoding="utf-8") as f:
            request = json.load(f)
        run(request, args.out, args.device, args.classify, args.asr)
    except WorkerError as e:
        print(f"chyba: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
