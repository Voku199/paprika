"""Render: stopy -> stereo buffer -> WAV.

Modul nezná hudební teorii. Bere hotový datový model a dělá z něj vzorky.
"""

from __future__ import annotations

import os
import wave

import numpy as np

from . import drums as _drums
from .fx import FxContext
from .patch import midi_to_freq
from .song import DrumTrack, Song, Track

__all__ = ["render_song", "render_track", "mix", "limiter", "write_wav",
           "RenderResult"]

LIMIT_DB = -1.0
LIMIT_THRESHOLD = 10.0 ** (LIMIT_DB / 20.0)
CEILING = 0.999  # tvrdý strop, kterého se limiter jen asymptoticky dotýká


class RenderResult:
    """Co z renderu vypadne. `audio` je (n, 2) float32 v rozsahu -1..1."""

    def __init__(self, song: Song, audio: np.ndarray, tracks: dict,
                 raw_peak: float, path: str | None, report: str = "",
                 seconds: float = 0.0):
        self.song = song
        self.audio = audio
        self.tracks = tracks          # jméno -> mono buffer po fx, před vol/pan
        self.raw_peak = raw_peak      # peak před limiterem
        self.path = path
        self.report = report
        self.seconds = seconds

    @property
    def sample_rate(self) -> int:
        return self.song.sample_rate

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RenderResult {self.seconds:.1f}s peak={self.raw_peak:.3f}>"


# --------------------------------------------------------------------------
# stopy
# --------------------------------------------------------------------------

def _voice_cache_key(track: Track, freq: float, dur: float) -> tuple:
    return (track.patch.key(), round(freq, 4), round(dur, 5))


def _fit(buf: np.ndarray, n: int) -> np.ndarray:
    """Výstup efektu srovná na stereo `(n, 2)` float32."""
    buf = np.asarray(buf, dtype=np.float32)
    if buf.ndim == 1:
        buf = np.repeat(buf[:, None], 2, axis=1)
    buf = buf[:n]
    if buf.shape[0] < n:
        buf = np.concatenate([buf, np.zeros((n - buf.shape[0], 2), np.float32)])
    return buf


def _add(buf: np.ndarray, i: int, sample: np.ndarray, amp: float) -> None:
    """Přičte mono nebo stereo sample do stereo bufferu od indexu `i`."""
    n = buf.shape[0]
    end = min(n, i + sample.shape[0])
    if end <= i:
        return
    part = sample[:end - i] * amp
    buf[i:end] += part[:, None] if part.ndim == 1 else part


def render_track(track: Track, n: int, ctx: FxContext, cache: dict | None = None,
                 beat_offset: float = 0.0, apply_fx: bool = True) -> np.ndarray:
    """Součet hlasů jedné stopy do stereo bufferu `(n, 2)` (bez vol/pan)."""
    sr = ctx.sr
    spb = 60.0 / ctx.bpm
    buf = np.zeros((n, 2), dtype=np.float32)
    cache = {} if cache is None else cache

    if isinstance(track, DrumTrack):
        for hit in track.hits:
            start = hit.start_beat - beat_offset
            if start < 0:
                continue
            i = int(round(start * spb * sr))
            if i >= n:
                continue
            sample = _drums.render_hit(hit.name, sr)
            amp = (max(1, min(127, hit.velocity)) / 127.0) ** 1.4
            _add(buf, i, sample, amp)
    else:
        for note in track.all_notes():
            start = note.start_beat - beat_offset
            if start < 0:
                continue
            i = int(round(start * spb * sr))
            if i >= n:
                continue
            freq = midi_to_freq(note.pitch)
            dur = max(0.01, note.dur_beats * spb)
            key = _voice_cache_key(track, freq, dur)
            voice = cache.get(key)
            if voice is None:
                voice = track.patch.voice(freq, dur, 127, sr, bpm=ctx.bpm)
                cache[key] = voice
            amp = (max(1, min(127, note.velocity)) / 127.0) ** 1.6
            _add(buf, i, voice, amp)

    if apply_fx:
        for effect in track.fx_chain:
            buf = _fit(effect.process(buf, ctx), n)
    return buf


def _pan_gains(pan: float) -> tuple[float, float]:
    """Equal-power pan: součet výkonů zůstává konstantní."""
    angle = (float(np.clip(pan, -1.0, 1.0)) + 1.0) * 0.25 * np.pi
    return float(np.cos(angle)), float(np.sin(angle))


def mix(track_buffers: dict, tracks: list, n: int) -> np.ndarray:
    """Stopy -> stereo (n, 2) s equal-power panem (balance) a hlasitostí."""
    out = np.zeros((n, 2), dtype=np.float32)
    for t in tracks:
        buf = track_buffers.get(t.name)
        if buf is None:
            continue
        buf = _fit(buf, n)
        left, right = _pan_gains(t.pan)
        out[:, 0] += buf[:, 0] * (t.vol * left)
        out[:, 1] += buf[:, 1] * (t.vol * right)
    return out


def limiter(audio: np.ndarray, threshold: float = LIMIT_THRESHOLD,
            ceiling: float = CEILING) -> np.ndarray:
    """Měkký limiter: nad prahem tanh, takže se nikdy neclipuje tvrdě.

    Ať je vstup jakkoli hlasitý, výstup zůstane pod `ceiling`.
    """
    mag = np.abs(audio)
    over = mag > threshold
    if not np.any(over):
        return audio.astype(np.float32)
    out = audio.copy()
    head = ceiling - threshold
    excess = (mag[over] - threshold) / head
    out[over] = np.sign(audio[over]) * (threshold + head * np.tanh(excess))
    return out.astype(np.float32)


def write_wav(path: str, audio: np.ndarray, sr: int) -> str:
    """16-bit PCM stereo přes stdlib `wave`."""
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    data = np.clip(audio, -1.0, 1.0)
    pcm = (data * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2 if pcm.ndim > 1 and pcm.shape[1] == 2 else 1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


# --------------------------------------------------------------------------
# celá skladba
# --------------------------------------------------------------------------

def _tail_seconds(song: Song, ctx: FxContext) -> float:
    """Nejdelší ocásek: release nástrojů plus tail efektů."""
    tail = 0.5
    for t in song.tracks:
        fx_tail = sum(f.tail(ctx) for f in t.fx_chain)
        release = getattr(getattr(t.patch, "adsr", None), "r", 0.3) or 0.0
        tail = max(tail, fx_tail + release + 0.3)
    return float(min(20.0, tail))


def render_song(song: Song, path: str | None = None, report: bool = True,
                bars=None, waveform: bool = False,
                print_report: bool = True) -> RenderResult:
    """Vyrenderuje skladbu, volitelně zapíše WAV a vypíše analýzu."""
    import time
    t0 = time.time()

    sr = song.sample_rate
    ctx = FxContext(sr=sr, bpm=song.bpm, beats_per_bar=song.beats_per_bar)

    if bars is None:
        lo_bar, hi_bar = 0, song.bars
    else:
        lo_bar, hi_bar = bars
        hi_bar = min(int(hi_bar), song.bars) if hi_bar else song.bars
    beat_offset = lo_bar * song.beats_per_bar
    span_beats = max(0.0, (hi_bar - lo_bar) * song.beats_per_bar)

    tail = _tail_seconds(song, ctx)
    n = int(round(span_beats * song.seconds_per_beat * sr)) + int(tail * sr)
    n = max(n, int(0.1 * sr))

    cache: dict = {}
    buffers: dict = {}

    # dvě fáze: nejdřív stopy bez sidechainu, pak ty, které cizí stopu potřebují
    deferred = []
    for t in song.tracks:
        if any(getattr(f, "needs_tracks", False) for f in t.fx_chain):
            deferred.append(t)
        else:
            buffers[t.name] = render_track(t, n, ctx, cache, beat_offset)
    ctx.track_buffers = buffers
    for t in deferred:
        buffers[t.name] = render_track(t, n, ctx, cache, beat_offset)

    stereo = mix(buffers, song.tracks, n)
    raw_peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    audio = limiter(stereo)

    # ořez ticha na konci, ale nikdy pod délku skladby
    min_n = int(round(span_beats * song.seconds_per_beat * sr))
    audio, n_used = _trim(audio, min_n)
    for name in buffers:
        buffers[name] = buffers[name][:n_used]

    out_path = None
    if path:
        out_path = write_wav(path, audio, sr)
        song.rendered_path = out_path

    text = ""
    if report:
        from .analyze import report as make_report
        text = make_report(song, audio, buffers, raw_peak,
                           path=out_path, render_seconds=time.time() - t0,
                           bars=(lo_bar, hi_bar))
        if waveform:
            from .viz import waveform as draw_waveform
            text += "\n\n" + draw_waveform(audio, sr, bpm=song.bpm,
                                           beats_per_bar=song.beats_per_bar)
        if print_report:
            print(text)

    return RenderResult(song, audio, buffers, raw_peak, out_path, text,
                        seconds=audio.shape[0] / sr)


def _trim(audio: np.ndarray, min_n: int, floor: float = 1e-4):
    """Ořízne koncové ticho, ale nikdy nezkrátí pod délku skladby."""
    if audio.size == 0:
        return audio, 0
    mag = np.max(np.abs(audio), axis=1)
    nz = np.nonzero(mag > floor)[0]
    end = int(nz[-1]) + 1 if nz.size else min_n
    end = max(end, min_n, 1)
    end = min(end, audio.shape[0])
    return audio[:end], end
