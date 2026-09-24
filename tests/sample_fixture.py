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
