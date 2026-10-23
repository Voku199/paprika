"""Obrázky a MP3 ukázky pro README z hotových ukázek v examples/.

    pip install matplotlib imageio-ffmpeg     # jen pro tenhle skript
    python tools/readme_media.py

Chybějící WAV v output/ nejdřív vyrenderuje (potřebuje nainstalované vzorky).
Výstup: docs/images/banner.png, docs/images/scena.png, docs/audio/*.mp3.
"""

from __future__ import annotations

import os
import subprocess
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from paprika.cli import load_song  # noqa: E402
from paprika.song import DrumTrack  # noqa: E402

EXAMPLES = ["01_klavir", "02_smycce", "03_sbor", "04_filmova_scena", "05_kontrola_teorie"]
IMAGES = os.path.join(ROOT, "docs", "images")
AUDIO = os.path.join(ROOT, "docs", "audio")

BG = "#16110f"
INK = "#f3e9e2"
MUTED = "#a8968c"
PAPRIKA = LinearSegmentedColormap.from_list(
    "paprika", ["#16110f", "#5a1410", "#b3261e", "#e8572a", "#f6a04d", "#fbe3b0"])
TRACK_COLORS = ["#f6a04d", "#e8572a", "#fbe3b0", "#d9607a", "#c9a227", "#e0805f",
                "#b98ad6", "#8fbf9f", "#7fb2d6"]


def wav_path(name: str) -> str:
    return os.path.join(ROOT, "output", name + ".wav")


def ensure_rendered(name: str) -> None:
    if not os.path.exists(wav_path(name)):
        subprocess.run([sys.executable, "-m", "paprika", "render",
                        os.path.join("examples", name + ".py")], cwd=ROOT, check=True)


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return x.reshape(-1, ch).mean(axis=1), sr


def envelope(x: np.ndarray, n: int) -> np.ndarray:
    hop = max(1, len(x) // n)
    frames = x[: hop * n].reshape(n, hop)
    return np.sqrt((frames ** 2).mean(axis=1))


def banner() -> None:
    x, _ = read_wav(wav_path("04_filmova_scena"))
    env = envelope(x, 900)
    env = env / env.max()
    fig = plt.figure(figsize=(16, 5), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    t = np.linspace(0.04, 0.96, len(env))
    amp = 0.12 + 0.55 * env ** 0.8
    for i in range(len(t)):
        ax.plot([t[i], t[i]], [-amp[i] * 0.55 - 0.42, amp[i] * 0.55 - 0.42],
                color=PAPRIKA(0.35 + 0.6 * env[i]), lw=1.1, alpha=0.9,
                solid_capstyle="round")

    ax.text(0.05, 0.55, "paprika", color=INK, fontsize=88, fontweight="bold",
            family="DejaVu Serif", va="center")
    ax.text(0.052, 0.18, "Hudba napsaná v Pythonu — křídlo, smyčce, sbor, žestě. Bez DAW.",
            color=MUTED, fontsize=21, va="center")
    fig.savefig(os.path.join(IMAGES, "banner.png"), facecolor=BG)
    plt.close(fig)


def scena() -> None:
    name = "04_filmova_scena"
    song, _ = load_song(os.path.join(ROOT, "examples", name + ".py"))
    x, sr = read_wav(wav_path(name))
    bpb = song.beats_per_bar
    beat_s = 60.0 / song.bpm

    tracks = [t for t in song.tracks if not isinstance(t, DrumTrack) and t.all_notes()]
    fig, (roll, spec) = plt.subplots(
        2, 1, figsize=(16, 9), dpi=100, facecolor=BG, sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.08})
    for ax in (roll, spec):
        ax.set_facecolor(BG)
        for side in ax.spines.values():
            side.set_color("#3a2e29")
        ax.tick_params(colors=MUTED, labelsize=10)

    # piano roll z datového modelu — přesně to, co je napsané v souboru
    labels = []
    for i, t in enumerate(tracks):
        color = TRACK_COLORS[i % len(TRACK_COLORS)]
        notes = t.all_notes()
        for n in notes:
            roll.add_patch(plt.Rectangle(
                (n.start_beat / bpb, n.pitch - 0.4), n.dur_beats / bpb, 0.8,
                color=color, alpha=0.45 + 0.55 * n.velocity / 127, lw=0))
        labels.append([float(np.median([n.pitch for n in notes])), t.name, color])
    # popisky vpravo bez překryvu: aspoň 2.6 půltónu od sebe
    labels.sort()
    for a, b in zip(labels, labels[1:]):
        b[0] = max(b[0], a[0] + 2.6)
    for y, name, color in labels:
        roll.text(song.bars + 0.15, y, name.replace("_", " "), color=color,
                  fontsize=10, va="center")
    lo = min(n.pitch for t in tracks for n in t.all_notes())
    hi = max(n.pitch for t in tracks for n in t.all_notes())
    roll.set_ylim(lo - 2, hi + 2)
    roll.set_xlim(0, song.bars + 2.2)
    roll.set_ylabel("výška (MIDI)", color=MUTED)
    roll.set_title("examples/04_filmova_scena.py — noty z kódu (nahoře) a vyrenderovaný zvuk (dole)",
                   color=INK, fontsize=14, loc="left", pad=12)
    for b in range(0, song.bars + 1, 4):
        for ax in (roll, spec):
            ax.axvline(b, color="#3a2e29", lw=0.8, zorder=0)

    # spektrogram renderu, osa x v taktech
    n_fft, hop = 2048, 512
    win = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * win
    mag = np.abs(np.fft.rfft(frames, axis=1)).T
    db = 20 * np.log10(mag + 1e-6)
    db = np.clip(db - db.max(), -80, 0)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    keep = freqs <= 8000
    bars_axis = np.arange(db.shape[1]) * hop / sr / (beat_s * bpb)
    spec.pcolormesh(bars_axis, freqs[keep], db[keep], cmap=PAPRIKA, shading="auto",
                    rasterized=True)
    spec.set_yscale("symlog", linthresh=200)
    spec.set_ylim(40, 8000)
    spec.set_yticks([100, 250, 500, 1000, 2000, 4000, 8000])
    spec.set_yticklabels(["100", "250", "500", "1k", "2k", "4k", "8k"])
    spec.set_ylabel("frekvence (Hz)", color=MUTED)
    spec.set_xlabel("takt", color=MUTED)
    spec.set_xticks(range(0, song.bars + 1, 2))
    fig.savefig(os.path.join(IMAGES, "scena.png"), facecolor=BG, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def mp3s() -> None:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for name in EXAMPLES:
        out = os.path.join(AUDIO, name + ".mp3")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", wav_path(name),
                        "-codec:a", "libmp3lame", "-b:a", "160k", out], check=True)


def main() -> None:
    os.makedirs(IMAGES, exist_ok=True)
    os.makedirs(AUDIO, exist_ok=True)
    for name in EXAMPLES:
        ensure_rendered(name)
    banner()
    scena()
    mp3s()
    print("hotovo:", IMAGES, AUDIO)


if __name__ == "__main__":
    main()
