"""Malé syntetické knihovny vzorků pro testy (sinusovky, žádná síť)."""

import os

import numpy as np
from scipy.io import wavfile


def midi_freq(p):
    return 440.0 * 2.0 ** ((p - 69) / 12.0)


def sine_wav(path, freq, sr=44100, seconds=1.0, amp=0.5, stereo=True):
    t = np.arange(int(sr * seconds)) / sr
    x = amp * np.sin(2 * np.pi * freq * t)
    if stereo:
        x = np.stack([x, x], axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wavfile.write(path, sr, (x * 32767).astype(np.int16))
    return path


def make_sfz_library(root, sr=48000, seconds=1.0):
    """Kořeny C4/E4/G#4, dvě velocity vrstvy, u C4 dva round robiny.

    Vzorky leží ve složce s mezerou ve jménu. Vrací cestu k SFZ.
    """
    lines = ["<global> amp_veltrack=100 ampeg_release=0.2"]
    for root_p, lo, hi in ((60, 58, 62), (64, 63, 66), (68, 67, 70)):
        for layer, (lovel, hivel, amp) in enumerate(((1, 63, 0.2), (64, 127, 0.8)), 1):
            rrs = (1, 2) if root_p == 60 else (1,)
            for rr in rrs:
                name = f"s_{root_p}_v{layer}_rr{rr}.wav"
                sine_wav(os.path.join(root, "wav dir", name), midi_freq(root_p),
                         sr, seconds, amp * (1.0 - 0.1 * (rr - 1)))
                lines.append(
                    f"<region> sample=wav dir/{name} lokey={lo} hikey={hi} "
                    f"pitch_keycenter={root_p} lovel={lovel} hivel={hivel} "
                    f"seq_length={len(rrs)} seq_position={rr}")
    path = os.path.join(root, "test.sfz")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


VSCO_VIOLINS = {
    "Strings/Violin Section/susVib": "VlnEns_susVib_{n}_v{v}.wav",
    "Strings/Violin Section/Trem": "VlnEns_Trem_{n}_v{v}.wav",
    "Strings/Violin Section/Pizz": "VlnEns_Pizz_{n}_v{v}_rr{r}.wav",
    "Strings/Violin Section/Spic": "VlnEns_Spic_{n}_v{v}_rr{r}.wav",
}


def make_vsco_violins(root, notes=("A2", "C3", "E3")):
    """Složky VSCO pro `violins` se jmény jako ve skutečném repu.

    VSCO jmenuje s C3 = střední C, takže A2 zní jako MIDI 57.
    """
    pcs = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    for folder, pattern in VSCO_VIOLINS.items():
        for n in notes:
            pitch = (int(n[-1]) + 1) * 12 + pcs[n[0]] + 12
            for v in (1, 2):
                for r in (1, 2):
                    name = pattern.format(n=n, v=v, r=r)
                    sine_wav(os.path.join(root, "vsco2", *folder.split("/"), name),
                             midi_freq(pitch), 44100, 0.6, 0.15 * v)


# --------------------------------------------------------------------------
# VocalSet s nádechy (breath.py): tóny, před každým nádech, šum místnosti
# --------------------------------------------------------------------------

def sung_tone(midi, sr=44100, seconds=1.2, amp=0.4, formants=((700, 110), (1150, 120))):
    """Zazpívaný tón: harmonické přes formanty (YIN ho pozná jako znělý)."""
    from scipy.signal import lfilter
    f0 = midi_freq(midi)
    t = np.arange(int(sr * seconds)) / sr
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 30) if k * f0 < sr / 2.2)
    for fc, bw in formants:
        r = np.exp(-np.pi * bw / sr)
        th = 2 * np.pi * fc / sr
        x = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)
    ramp = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.03)
    return x / np.abs(x).max() * amp * ramp


def breath_sound(sr, seconds, center, seed):
    """Nádech pro fixture: šum v pásmu `center` se zvonovitou obálkou."""
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    u = (np.arange(n) + 0.5) / n
    env = u ** 1.4 * (1 - u) ** 2.1
    b, a = butter(2, [center / 1.8 / (sr / 2), min(0.95, center * 1.8 / (sr / 2))], "band")
    x = lfilter(b, a, rng.standard_normal(n)) * env
    return x / np.sqrt(np.mean(x ** 2)) * 0.02


SINGER_BREATH_HZ = {"male1": 1300.0, "male3": 1500.0, "female2": 3200.0}


def fake_vocalset_breaths(full, sr=44100, singers=("male1", "female2"), vowels="aeiou",
                          distractors=True):
    """Dlouhé tóny jako ve VocalSet: 3 tóny za sebou, před každým nádech.

    Nádechy mají barvu podle zpěváka (`SINGER_BREATH_HZ`), aby šlo ověřit,
    že ji model zachytí. Vrací {cesta: [(t0, t1) nádechů]}. Soubor
    `*_distract_a.wav` má před prvními dvěma tóny cvakání a dunění, za
    posledním výdech — nádech je jen před třetím tónem.
    """
    truths = {}

    def write(singer, name, parts, seed):
        x = np.concatenate(parts)
        x = x + 0.001 * np.random.default_rng(seed).standard_normal(x.size)
        path = os.path.join(full, singer, "long_tones", "straight", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        wavfile.write(path, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))
        return path

    gap, rest = np.zeros(int(0.06 * sr)), np.zeros(int(0.25 * sr))
    for si, singer in enumerate(singers):
        center = SINGER_BREATH_HZ[singer]
        lift = 12 if singer.startswith("female") else 0
        for vi, v in enumerate(vowels):
            rng = np.random.default_rng(100 * si + vi)
            parts, truth, t = [np.zeros(int(0.3 * sr))], [], 0.3
            for tone in (48, 55, 60):
                b = breath_sound(sr, rng.uniform(0.3, 0.55), center, int(rng.integers(1 << 30)))
                truth.append((t, t + b.size / sr))
                parts += [b, gap, sung_tone(tone + lift, sr), rest]
                t += (b.size + gap.size) / sr + 1.2 + rest.size / sr
            truths[write(singer, f"{singer[0]}_long_straight_{v}.wav", parts, si * 10 + vi)] = truth
        if distractors:
            rng = np.random.default_rng(999 + si)
            clicks = np.zeros(int(0.3 * sr))
            for k in range(3):
                i = int((0.05 + 0.08 * k) * sr)
                clicks[i:i + 40] = 0.3 * rng.standard_normal(40)
            tt = np.arange(int(0.3 * sr)) / sr
            rumble = 0.05 * np.sin(2 * np.pi * 40 * tt) * np.exp(-tt / 0.1)
            b = breath_sound(sr, 0.4, center, 7)
            parts = [np.zeros(int(0.3 * sr)), clicks, gap, sung_tone(48 + lift, sr), rest,
                     rumble, gap, sung_tone(55 + lift, sr), rest]
            t0 = sum(p.size for p in parts) / sr
            parts += [b, gap, sung_tone(60 + lift, sr), np.zeros(int(0.1 * sr)),
                      breath_sound(sr, 0.5, center, 8), np.zeros(int(0.4 * sr))]
            truths[write(singer, f"{singer[0]}_long_distract_a.wav", parts, 77 + si)] = \
                [(t0, t0 + b.size / sr)]
    return truths
