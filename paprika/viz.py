"""ASCII náhled: piano roll a obálka mixu.

AI zvuk neslyší, takže tohle je jediný způsob, jak zkontrolovat strukturu
skladby bez člověka ve smyčce.
"""

from __future__ import annotations

import numpy as np

from . import theory
from .drums import DRUM_CHARS
from .sfx import SFX_CHARS
from .song import DrumTrack, SfxTrack, Song

__all__ = ["piano_roll", "waveform", "LABEL_WIDTH"]

LABEL_WIDTH = 7
_NAME_TO_CHAR = {v: k for k, v in DRUM_CHARS.items()}


def _note_char(pitch: int) -> str:
    """C4 -> 'C', C#4 -> 'c'. Malé písmeno znamená křížek."""
    name = theory.note_name(pitch)
    return name[0].lower() if "#" in name else name[0]


def piano_roll(song: Song, bars=None, res: int = 8) -> str:
    """Timeline skladby po krocích. `res` je počet znaků na takt.

    Onset noty je písmeno tónu (malé = křížek), držení je '-', ticho '.'.
    Bicí kreslí svůj vlastní znak, akcent velkým písmenem.
    """
    res = max(1, int(res))
    if bars is None:
        lo, hi = 0, min(song.bars, 8)
    else:
        lo, hi = bars
        lo, hi = int(lo), min(int(hi), song.bars)
    if hi <= lo:
        hi = lo + 1
    n_bars = hi - lo
    step = song.beats_per_bar / res
    total = n_bars * res

    def idx(beat: float) -> int:
        """Konec noty: krok, do kterého ještě spadá."""
        return int(np.floor((beat - lo * song.beats_per_bar) / step + 1e-9))

    def onset(beat: float) -> int:
        """Začátek noty: nejbližší krok, aby humanize nerozhodila obrázek."""
        return int(np.floor((beat - lo * song.beats_per_bar) / step + 0.5))

    header = " " * LABEL_WIDTH
    for b in range(lo, hi):
        header += "|" + str(b).ljust(res)[:res]
    header += "|"

    rows = [header]
    for t in song.tracks:
        cells = ["."] * total
        if isinstance(t, SfxTrack):
            # SFX: symbol na začátku, '-' přes délku; symbol nepřepíše jiný
            for h in t.hits:
                i = onset(h.start_beat)
                end = idx(h.start_beat + max(0.0, h.dur_beats) - 1e-9)
                for j in range(max(i + 1, 0), min(end + 1, total)):
                    if cells[j] == ".":
                        cells[j] = "-"
                if 0 <= i < total and cells[i] in ".-":
                    cells[i] = SFX_CHARS.get(h.name, "*")
        elif isinstance(t, DrumTrack):
            for h in t.hits:
                i = onset(h.start_beat)
                if 0 <= i < total:
                    ch = _NAME_TO_CHAR.get(h.name, "x")
                    cells[i] = ch.upper() if h.velocity >= 110 else ch
        else:
            for note in t.all_notes():
                i = onset(note.start_beat)
                end = idx(note.start_beat + note.dur_beats - 1e-9)
                if end < 0 or i >= total:
                    continue
                for j in range(max(i + 1, 0), min(end + 1, total)):
                    if cells[j] == ".":
                        cells[j] = "-"
                if 0 <= i < total:
                    cells[i] = _note_char(note.pitch)

        line = t.name[:LABEL_WIDTH - 1].ljust(LABEL_WIDTH)
        for b in range(n_bars):
            line += "|" + "".join(cells[b * res:(b + 1) * res])
        rows.append(line + "|")

    unit = "znak" if res == 1 else ("znaky" if res < 5 else "znaků")
    rows.append("")
    rows.append(f"{song.bpm:g} BPM  {song.key} {song.scale}  "
                f"takty {lo}-{hi}  ({res} {unit} na takt, "
                f"malé písmeno = křížek)")
    return "\n".join(rows)


def waveform(audio: np.ndarray, sr: int, width: int = 72, height: int = 9,
             bpm: float | None = None, beats_per_bar: float = 4.0) -> str:
    """Obálka hotového mixu — vidět, kde je tiché intro a kde nastupuje refrén."""
    if audio.size == 0:
        return "WAVE   (prázdný buffer)"
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    width = max(8, int(width))
    height = max(3, int(height))
    edges = np.linspace(0, mono.size, width + 1).astype(int)
    env = np.array([
        float(np.sqrt(np.mean(mono[a:b] ** 2))) if b > a else 0.0
        for a, b in zip(edges[:-1], edges[1:])])
    peak = env.max()
    norm = env / peak if peak > 0 else env

    rows = []
    for r in range(height, 0, -1):
        threshold = r / height
        line = "".join("#" if v >= threshold else
                       ("-" if v >= threshold - 0.5 / height else " ")
                       for v in norm)
        rows.append(f"       |{line}|")

    dur = mono.size / sr
    axis = " " * 8
    if bpm:
        bar_sec = beats_per_bar * 60.0 / bpm
        n_bars = max(1, int(dur / bar_sec))
        marks = [" "] * width
        stride = max(1, n_bars // 8)
        for b in range(0, n_bars + 1, stride):
            i = int(b * bar_sec / dur * width)
            if 0 <= i < width:
                for k, ch in enumerate(str(b)):
                    if i + k < width:
                        marks[i + k] = ch
        axis += "".join(marks)
    rows.append(axis)
    rows.insert(0, f"WAVE   {dur:.1f} s   peak rms {peak:.3f}")
    return "\n".join(rows)
