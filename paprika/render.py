"""Render: stopy -> stereo buffer -> WAV.

Modul nezná hudební teorii. Bere hotový datový model a dělá z něj vzorky.
"""

from __future__ import annotations

import os
import wave

import numpy as np

from . import drums as _drums
from .automation import resolve as _resolve_automation
from .fx import FxContext
from .patch import midi_to_freq
from . import sfx as _sfx
from .song import Bus, DrumTrack, SfxTrack, Song, Track

__all__ = ["render_song", "render_track", "mix", "limiter", "write_wav",
           "RenderResult"]

LIMIT_DB = -1.0
LIMIT_THRESHOLD = 10.0 ** (LIMIT_DB / 20.0)
CEILING = 0.999  # tvrdý strop, kterého se limiter jen asymptoticky dotýká


class RenderResult:
    """Co z renderu vypadne. `audio` je (n, 2) float32 v rozsahu -1..1."""

    def __init__(self, song: Song, audio: np.ndarray, tracks: dict,
                 raw_peak: float, path: str | None, report: str = "",
                 seconds: float = 0.0, buses: dict | None = None):
        self.song = song
        self.audio = audio
        self.tracks = tracks          # jméno -> stereo buffer po fx, před vol/pan
        self.buses = buses or {}      # jméno -> stereo buffer po fx busu, před vol
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

SFX_TAIL = 4.0            # sekundy: nejdelší doznění SFX (impact, braam)
LEGATO_GAP = 1.0 / 64.0   # beaty: menší mezera mezi notami je ještě legato
DEFAULT_SLIDE = 0.125     # beaty: délka glidu u `>` noty, když stopa nemá svůj


def _phrases(track: Track) -> list[tuple]:
    """Noty stopy -> fráze, z nichž každá je jeden syntetizovaný hlas.

    Fráze = `(start_beat, dur_beats, velocity, path, glide_beats)`, kde
    `path` je `((rel_beat, pitch), ...)`. Mono stopa slije navazující noty
    do jedné fráze; polyfonní stopa dá každé notě vlastní frázi a nota
    s `>` klouže z výšky předchozí noty.
    """
    notes = track.all_notes()
    if getattr(track, "mono", False):
        groups: list[dict] = []
        for n in notes:
            cur = groups[-1] if groups else None
            if cur is not None and n.start_beat <= cur["end"] + LEGATO_GAP:
                cur["path"].append((n.start_beat - cur["start"], n.pitch))
                cur["end"] = n.end_beat
                cur["slide"] = cur["slide"] or n.glide
                continue
            groups.append({"start": n.start_beat, "end": n.end_beat,
                           "vel": n.velocity, "path": [(0.0, n.pitch)],
                           "slide": False})
        out = []
        for c in groups:
            glide = track.glide or (DEFAULT_SLIDE if c["slide"] else 0.0)
            out.append((c["start"], c["end"] - c["start"], c["vel"],
                        tuple(c["path"]), glide if len(c["path"]) > 1 else 0.0))
        return out

    out = []
    prev = None
    for n in notes:
        if n.glide and prev is not None and prev.pitch != n.pitch:
            path = ((0.0, prev.pitch), (0.0, n.pitch))
            glide = track.glide or DEFAULT_SLIDE
        else:
            path = ((0.0, n.pitch),)
            glide = 0.0
        out.append((n.start_beat, n.dur_beats, n.velocity, path, glide))
        prev = n
    return out


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

    if isinstance(track, SfxTrack):
        for hit in track.hits:
            start = hit.start_beat - beat_offset
            if start < 0:
                continue
            i = int(round(start * spb * sr))
            if i >= n:
                continue
            sample = _sfx.render_sfx(hit.name, hit.params, sr, ctx.bpm)
            _add(buf, i, sample, (max(1, min(127, hit.velocity)) / 127.0) ** 1.4)
    elif isinstance(track, DrumTrack):
        for hit in track.hits:
            start = hit.start_beat - beat_offset
            if start < 0:
                continue
            i = int(round(start * spb * sr))
            if i >= n:
                continue
            sample = _drums.render_hit(hit.name, sr, getattr(track, "kit", "classic"))
            amp = (max(1, min(127, hit.velocity)) / 127.0) ** 1.4
            _add(buf, i, sample, amp)
    else:
        patch = track.patch
        pkey = patch.key()
        rr = max(1, int(getattr(patch, "rr", 1)))
        for idx, (start_b, dur_b, vel, path, glide_b) in enumerate(_phrases(track)):
            start = start_b - beat_offset
            if start < 0:
                continue
            i = int(round(start * spb * sr))
            if i >= n:
                continue
            freq = midi_to_freq(path[0][1])
            dur = max(0.01, dur_b * spb)
            gpath = (tuple((round(rel * spb, 6), round(midi_to_freq(p), 4))
                           for rel, p in path) if len(path) > 1 else None)
            gtime = round(glide_b * spb, 6)
            rr_index = idx % rr
            # jas podle velocity: kvantováno po 16, aby cache pořád fungovala
            tone = (127 if not getattr(patch, "vel_to_cutoff", 0)
                    else int(min(127, max(16, round(vel / 16) * 16))))
            key = (pkey, round(freq, 4), round(dur, 5), rr_index, tone, gpath,
                   gtime, ctx.bpm)
            voice = cache.get(key)
            if voice is None:
                voice = patch.voice(freq, dur, 127, sr, bpm=ctx.bpm,
                                    rr_index=rr_index, tone_vel=tone,
                                    glide_path=gpath, glide_time=gtime)
                cache[key] = voice
            amp = (max(1, min(127, vel)) / 127.0) ** 1.6
            _add(buf, i, voice, amp)

    if apply_fx:
        for effect in track.fx_chain:
            buf = _fit(effect.process(buf, ctx), n)
    return buf


def _pan_gains(pan):
    """Equal-power pan: součet výkonů zůstává konstantní. Skalár i pole."""
    angle = (np.clip(pan, -1.0, 1.0) + 1.0) * 0.25 * np.pi
    if np.ndim(angle) == 0:
        return float(np.cos(angle)), float(np.sin(angle))
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


def _apply_gain(buf: np.ndarray, vol, pan) -> np.ndarray:
    """Stereo buffer × hlasitost × balance (obojí skalár nebo pole)."""
    left, right = _pan_gains(pan)
    out = np.empty_like(buf)
    out[:, 0] = buf[:, 0] * (np.asarray(vol, dtype=np.float32) * left)
    out[:, 1] = buf[:, 1] * (np.asarray(vol, dtype=np.float32) * right)
    return out


def mix(track_buffers: dict, tracks: list, n: int,
        gains: dict | None = None) -> np.ndarray:
    """Stopy -> stereo (n, 2) s equal-power panem (balance) a hlasitostí.

    `gains` = {jméno: (vol, pan)} přepíše statické `t.vol`/`t.pan`
    (automatizace — hodnoty smí být pole po samplech).
    """
    out = np.zeros((n, 2), dtype=np.float32)
    for t in tracks:
        buf = track_buffers.get(t.name)
        if buf is None:
            continue
        vol, pan = (gains or {}).get(t.name, (t.vol, t.pan))
        out += _apply_gain(_fit(buf, n), vol, pan)
    return out


def _automated(owner, n: int, ctx: FxContext, beat_offset: float):
    """Připraví automatizaci: nastaví křivky efektům, vrátí (vol, pan, fx).

    Po zpracování je potřeba zavolat `clear_curves()` na vrácených efektech.
    """
    vol_a, pan_a, fx_autos = _resolve_automation(
        owner.name, getattr(owner, "automations", []), owner.fx_chain)
    args = (n, ctx.sr, ctx.bpm, ctx.beats_per_bar, beat_offset)
    for fx, param, a in fx_autos:
        fx.set_curve(param, a.sample(*args))
    vol = vol_a.sample(*args).astype(np.float32) if vol_a else owner.vol
    pan = pan_a.sample(*args) if pan_a else owner.pan
    return vol, pan, [fx for fx, _, _ in fx_autos]


def _apply_balance(buf: np.ndarray, vol, pan) -> np.ndarray:
    """Hlasitost × balance pro bus/master: pan 0 = beze změny obou kanálů."""
    pan = np.clip(np.asarray(pan, dtype=np.float32), -1.0, 1.0)
    vol = np.asarray(vol, dtype=np.float32)
    out = np.empty_like(buf)
    out[:, 0] = buf[:, 0] * vol * np.minimum(1.0, 1.0 - pan)
    out[:, 1] = buf[:, 1] * vol * np.minimum(1.0, 1.0 + pan)
    return out


def _run_chain(chain, buf: np.ndarray, ctx: FxContext) -> np.ndarray:
    n = buf.shape[0]
    for effect in chain:
        buf = _fit(effect.process(buf, ctx), n)
    return buf


def _render_bus(bus, inp: np.ndarray, ctx: FxContext, beat_offset: float):
    """Fx řetěz busu s automatizací. Vrací (buffer po fx, vol, pan)."""
    vol, pan, touched = _automated(bus, inp.shape[0], ctx, beat_offset)
    try:
        out = _run_chain(bus.fx_chain, inp, ctx)
    finally:
        for fx in touched:
            fx.clear_curves()
    return out, vol, pan


def _deps(obj, song: Song, known: set) -> set:
    """Jména stop/busů, které musí být hotové dřív než `obj`."""
    out = {f.source for f in obj.fx_chain if getattr(f, "needs_tracks", False)}
    if isinstance(obj, Bus):
        out |= {t.name for t in obj.members()}
    # neznámý zdroj Sidechainu nahlásí sám Sidechain, se seznamem stop
    return out & known


def _render_automated(t, n, ctx, cache, beat_offset, gains: dict) -> np.ndarray:
    vol, pan, touched = _automated(t, n, ctx, beat_offset)
    try:
        buf = render_track(t, n, ctx, cache, beat_offset)
    finally:
        for fx in touched:
            fx.clear_curves()
    gains[t.name] = (vol, pan)
    return buf


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
        if isinstance(t, SfxTrack) and t.hits:
            release = max(release, SFX_TAIL)
        tail = max(tail, fx_tail + release + 0.3)
    bus_tail = max((sum(f.tail(ctx) for f in b.fx_chain) for b in song.buses),
                   default=0.0)
    tail += bus_tail + sum(f.tail(ctx) for f in song.master.fx_chain)
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
    ctx.beat_offset = beat_offset
    span_beats = max(0.0, (hi_bar - lo_bar) * song.beats_per_bar)

    tail = _tail_seconds(song, ctx)
    n = int(round(span_beats * song.seconds_per_beat * sr)) + int(tail * sr)
    n = max(n, int(0.1 * sr))

    cache: dict = {}
    buffers: dict = {}      # stopy po fx, před vol/pan
    bus_out: dict = {}      # busy po fx, před vol/pan
    gains: dict = {}

    # render podle závislostí: stopa čeká na zdroje svých Sidechainů, bus na
    # své stopy; když se v kole nic nepohne, je v závislostech cyklus
    known = {t.name for t in song.tracks} | {b.name for b in song.buses}
    pending = list(song.tracks) + list(song.buses)
    ready: dict = {}
    ctx.track_buffers = ready
    while pending:
        progressed = False
        for obj in list(pending):
            if not _deps(obj, song, known) <= ready.keys():
                continue
            if isinstance(obj, Bus):
                inp = np.zeros((n, 2), dtype=np.float32)
                for t in obj.members():
                    inp += _apply_gain(buffers[t.name], *gains[t.name])
                out, vol, pan = _render_bus(obj, inp, ctx, beat_offset)
                bus_out[obj.name] = out
                gains[obj.name] = (vol, pan)
                ready[obj.name] = out
            else:
                buffers[obj.name] = _render_automated(obj, n, ctx, cache,
                                                      beat_offset, gains)
                ready[obj.name] = buffers[obj.name]
            pending.remove(obj)
            progressed = True
        if not progressed:
            names = ", ".join(o.name for o in pending)
            raise RuntimeError(
                f"cyklická závislost renderu: {names} — Sidechain nesmí brát "
                f"zdroj z busu, do kterého stopa sama patří")

    direct = [t for t in song.tracks if t.bus is None]
    stereo = mix(buffers, direct, n, gains)
    for b in song.buses:
        stereo += _apply_balance(bus_out[b.name], *gains[b.name])
    master_fx, m_vol, m_pan = _render_bus(song.master, stereo, ctx, beat_offset)
    stereo = _apply_balance(master_fx, m_vol, m_pan)
    raw_peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    audio = limiter(stereo)

    # ořez ticha na konci, ale nikdy pod délku skladby
    min_n = int(round(span_beats * song.seconds_per_beat * sr))
    audio, n_used = _trim(audio, min_n)
    for name in buffers:
        buffers[name] = buffers[name][:n_used]
    for name in bus_out:
        bus_out[name] = bus_out[name][:n_used]

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
                        seconds=audio.shape[0] / sr, buses=bus_out)


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
