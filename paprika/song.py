"""Datový model skladby: Song, Track, DrumTrack, Clip.

Noty žijí v beatech, ne v sekundách. Převod na samply se děje až v `render()`,
takže změna BPM je jeden parametr a nic se nepřepisuje.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import drums as _drums
from . import note as _note
from . import theory
from .note import Note
from .patch import Patch
from .presets import preset

__all__ = ["Song", "Track", "DrumTrack", "Clip", "suppress_render"]

# Když skladbu importuje CLI, `Song.render()` uvnitř souboru se má chovat jako
# no-op — jinak by se renderovalo dvakrát. CLI si tenhle přepínač zapíná samo.
_SUPPRESS_RENDER = False


class suppress_render:
    """Context manager: `s.render(...)` uvnitř importovaného modulu neběží."""

    def __enter__(self):
        global _SUPPRESS_RENDER
        self._prev = _SUPPRESS_RENDER
        _SUPPRESS_RENDER = True
        return self

    def __exit__(self, *exc):
        global _SUPPRESS_RENDER
        _SUPPRESS_RENDER = self._prev
        return False


@dataclass
class Clip:
    """Skupina not umístěná na timeline. Noty mají start relativní ke clipu."""

    start_beat: float
    notes: list = field(default_factory=list)

    def absolute(self):
        for n in self.notes:
            yield n, self.start_beat + n.start_beat


class Track:
    """Jedna stopa: nástroj, klipy, efektový řetěz, hlasitost a pan."""

    def __init__(self, song: "Song", patch: Patch, name: str, vol: float = 0.8,
                 pan: float = 0.0, chromatic: bool = False, mono: bool = False,
                 glide: float = 0.0):
        self.song = song
        self.patch = patch
        self.name = name
        self.vol = float(vol)
        self.pan = float(max(-1.0, min(1.0, pan)))
        self.chromatic = bool(chromatic)
        self.clips: list[Clip] = []
        self.fx_chain: list = []
        # mono legato: překrývající se noty splynou do jednoho hlasu a výška
        # mezi nimi klouže `glide` beatů (808, acid, sólové linky)
        self.mono = bool(mono)
        self.glide = float(glide)
        self.automations: list = []

    # -- pomocné ----------------------------------------------------------
    @property
    def bpb(self) -> float:
        return self.song.beats_per_bar

    def _bar_to_beat(self, bar: float) -> float:
        return float(bar) * self.bpb

    def _add(self, start_beat: float, notes: list[Note]) -> Clip:
        clip = Clip(float(start_beat), notes)
        self.clips.append(clip)
        return clip

    def all_notes(self) -> list[Note]:
        """Noty s absolutním startem, seřazené podle času."""
        out = []
        for clip in self.clips:
            for n, start in clip.absolute():
                out.append(Note(n.pitch, start, n.dur_beats, n.velocity, n.glide))
        out.sort(key=lambda n: (n.start_beat, n.pitch))
        return out

    def __len__(self) -> int:
        return sum(len(c.notes) for c in self.clips)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Track {self.name} notes={len(self)} vol={self.vol:g}>"

    # -- zápis not --------------------------------------------------------
    def note(self, pitch, at: float = 0.0, dur: float = 1.0, vel: int = 100):
        """Jedna nota. `at` je v beatech od začátku skladby."""
        p = theory.pitch(pitch)
        self._add(float(at), [Note(p, 0.0, float(dur), int(vel))])
        return self

    def notes(self, text: str, bar: float = 0, beat: float = 0.0,
              vel: int = 100, dur: float = 1.0):
        """Notační řetězec od daného taktu."""
        parsed = _note.parse(text, beats_per_bar=self.bpb, default_dur=dur,
                             velocity=vel, bar_offset=int(bar))
        self._add(self._bar_to_beat(bar) + beat, parsed)
        return self

    def melody(self, text: str, bar: float = 0, beat: float = 0.0,
               vel: int = 100, dur: float = 1.0):
        """Totéž co `notes()`, jen čitelnější jméno pro melodické linky."""
        return self.notes(text, bar=bar, beat=beat, vel=vel, dur=dur)

    def chord(self, name, bar: float = 0, dur: float = None, beat: float = 0.0,
              inv: int = 0, oct: int = 0, octave: int = 4, vel: int = 100):
        """Akord podle jména ('Cmaj7') nebo seznamu pitchů."""
        if dur is None:
            dur = self.bpb
        pitches = (theory.chord(name, octave=octave, inv=inv, oct=oct)
                   if isinstance(name, str)
                   else theory.invert([theory.pitch(p) for p in name], inv))
        notes = [Note(p, 0.0, float(dur), int(vel)) for p in pitches]
        self._add(self._bar_to_beat(bar) + beat, notes)
        return self

    def arp(self, name, bar: float = 0, pattern: str = "up", rate: float = 1 / 8,
            octaves: int = 1, bars: float = 1, beat: float = 0.0,
            inv: int = 0, oct: int = 0, octave: int = 4, vel: int = 100,
            gate: float = 0.9, seed: int | None = None):
        """Arpeggio přes `bars` taktů. `rate` je zlomek celé noty."""
        pitches = (theory.chord(name, octave=octave, inv=inv, oct=oct)
                   if isinstance(name, str)
                   else theory.invert([theory.pitch(p) for p in name], inv))
        step = float(rate) * self.bpb
        if step <= 0:
            raise ValueError("rate musí být kladné")
        steps = int(round(self.bpb * bars / step))
        seq = theory.arp(pitches, pattern=pattern, steps=steps,
                         octaves=octaves, seed=seed)
        notes = [Note(p, i * step, step * gate, int(vel))
                 for i, p in enumerate(seq)]
        self._add(self._bar_to_beat(bar) + beat, notes)
        return self

    # -- úpravy -----------------------------------------------------------
    def fx(self, *effects):
        """Připojí efekty na konec řetězu."""
        self.fx_chain.extend(effects)
        return self

    def automate(self, target: str, points, curve: str = "lin"):
        """Změna parametru v čase: `vol`, `pan` nebo `<efekt>.<param>`.

        `points` jsou `(takt, hodnota)`; křivka `lin`, `exp` (frekvence),
        `s` (plynulý náběh) nebo `step`. Příklad filtr sweepu do dropu::

            t.fx(LowPass(300)).automate("lowpass.cutoff",
                                        [(16, 300), (24, 9000)], "exp")
        """
        from .automation import Automation
        self.automations.append(Automation(target, points, curve))
        return self

    def humanize(self, timing: float = 0.01, vel: float = 0.1,
                 seed: int | None = None):
        """Rozhýbe časování (v beatech) a dynamiku (relativně) všech not."""
        rng = random.Random(seed)
        for clip in self.clips:
            for n in clip.notes:
                if timing:
                    n.start_beat = max(0.0, n.start_beat +
                                       rng.uniform(-timing, timing))
                if vel:
                    scaled = n.velocity * (1.0 + rng.uniform(-vel, vel))
                    n.velocity = int(max(1, min(127, round(scaled))))
        return self

    def clear(self, bars=None):
        """Smaže noty. `bars=(od, do)` smaže jen daný rozsah taktů."""
        if bars is None:
            self.clips = []
            return self
        lo, hi = (bars if isinstance(bars, (tuple, list)) else (bars, bars + 1))
        lo_b, hi_b = self._bar_to_beat(lo), self._bar_to_beat(hi)
        for clip in self.clips:
            clip.notes = [n for n in clip.notes
                          if not (lo_b <= clip.start_beat + n.start_beat < hi_b)]
        self.clips = [c for c in self.clips if c.notes]
        return self

    def transpose(self, semitones: int):
        """Posune celou stopu o půltóny."""
        for clip in self.clips:
            for n in clip.notes:
                n.pitch = int(n.pitch + semitones)
        return self

    def repeat(self, src_bars, times: int = 1, every: float | None = None):
        """Zkopíruje rozsah taktů `times`-krát dál po timeline."""
        lo, hi = src_bars
        span = every if every is not None else (hi - lo)
        lo_b, hi_b = self._bar_to_beat(lo), self._bar_to_beat(hi)
        source = [(n, clip.start_beat + n.start_beat)
                  for clip in list(self.clips) for n in clip.notes
                  if lo_b <= clip.start_beat + n.start_beat < hi_b]
        for i in range(1, int(times) + 1):
            shift = self._bar_to_beat(span) * i
            self._add(0.0, [Note(n.pitch, start + shift, n.dur_beats, n.velocity,
                                 n.glide) for n, start in source])
        return self


class DrumTrack(Track):
    """Stopa bicích. Místo pitchů drží jména nástrojů z pattern stringu."""

    def __init__(self, song: "Song", name: str = "drums", vol: float = 0.6,
                 pan: float = 0.0):
        super().__init__(song, Patch(name="drums"), name, vol, pan, chromatic=True)
        self.hits: list[_drums.DrumHit] = []

    def __len__(self) -> int:
        return len(self.hits)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DrumTrack {self.name} hits={len(self.hits)}>"

    def pattern(self, text: str, bars=(0, 1), rate: float = 1 / 16,
                vel: float = 1.0):
        """Zopakuje pattern přes rozsah taktů. Jeden znak = jeden krok."""
        lo, hi = (bars if isinstance(bars, (tuple, list)) else (bars, bars + 1))
        step = float(rate) * self.bpb
        if step <= 0:
            raise ValueError("rate musí být kladné")
        steps_per_bar = int(round(self.bpb / step))
        raw = _drums.parse_pattern(text, step_beats=step, start_beat=0.0)
        if not raw:
            return self
        length_steps = _drums.pattern_steps(text)
        # kratší pattern se dopočítá pauzami do konce taktu, delší běží dál
        span_bars = max(1, -(-length_steps // steps_per_bar))
        span_beats = span_bars * self.bpb
        bar = float(lo)
        while bar < hi:
            base = self._bar_to_beat(bar)
            for h in raw:
                start = base + h.start_beat
                if start >= self._bar_to_beat(hi):
                    continue
                self.hits.append(_drums.DrumHit(
                    h.name, start, int(max(1, min(127, round(h.velocity * vel))))))
            bar += span_bars if span_beats else 1
        return self

    def hit(self, name: str, at: float, vel: int = 100):
        """Jeden úder v beatech od začátku skladby."""
        if name not in _drums.drum_names():
            name = _drums.DRUM_CHARS.get(str(name).lower()[:1], name)
        self.hits.append(_drums.DrumHit(name, float(at), int(vel)))
        return self

    def all_notes(self) -> list[Note]:
        """Bicí jako noty — pro vizualizaci a analýzu délek."""
        return []

    def humanize(self, timing: float = 0.008, vel: float = 0.1,
                 seed: int | None = None):
        rng = random.Random(seed)
        for h in self.hits:
            if timing:
                h.start_beat = max(0.0, h.start_beat + rng.uniform(-timing, timing))
            if vel:
                h.velocity = int(max(1, min(127, round(
                    h.velocity * (1.0 + rng.uniform(-vel, vel))))))
        return self

    def clear(self, bars=None):
        if bars is None:
            self.hits = []
            return self
        lo, hi = (bars if isinstance(bars, (tuple, list)) else (bars, bars + 1))
        lo_b, hi_b = self._bar_to_beat(lo), self._bar_to_beat(hi)
        self.hits = [h for h in self.hits if not (lo_b <= h.start_beat < hi_b)]
        return self


class Song:
    """Celá skladba. Čistý datový objekt — nezná CLI ani soubory."""

    def __init__(self, bpm: float = 120, key: str = "C", scale: str = "major",
                 bars: int = 16, beats_per_bar: float = 4,
                 sample_rate: int = 44100, title: str | None = None):
        self.bpm = float(bpm)
        self.key = str(key)
        self.scale = str(scale)
        self.bars = int(bars)
        self.beats_per_bar = float(beats_per_bar)
        self.sample_rate = int(sample_rate)
        self.title = title
        self.tracks: list[Track] = []
        self.rendered_path: str | None = None

    # -- vlastnosti -------------------------------------------------------
    @property
    def total_beats(self) -> float:
        return self.bars * self.beats_per_bar

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def duration(self) -> float:
        """Délka v sekundách bez ocásků efektů."""
        return self.total_beats * self.seconds_per_beat

    def beat_to_sample(self, beat: float) -> int:
        return int(round(beat * self.seconds_per_beat * self.sample_rate))

    def scale_pcs(self) -> list[int]:
        return theory.scale_pcs(self.key, self.scale)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<Song {self.bpm:g}bpm {self.key} {self.scale} "
                f"{self.bars} bars, {len(self.tracks)} tracks>")

    # -- stavba -----------------------------------------------------------
    def _unique(self, name: str) -> str:
        taken = {t.name for t in self.tracks}
        if name not in taken:
            return name
        i = 2
        while f"{name}{i}" in taken:
            i += 1
        return f"{name}{i}"

    def track(self, instrument="epiano", vol: float = 0.8, pan: float = 0.0,
              chromatic: bool = False, name: str | None = None,
              mono: bool | None = None, glide: float | None = None,
              fx=None) -> Track:
        """Přidá stopu. `instrument` je jméno presetu nebo `Patch`.

        `mono=True` zapne mono legato, `glide` je délka sklouznutí v beatech.
        `fx` je seznam efektů, které se rovnou připojí.
        """
        if isinstance(instrument, str):
            patch = preset(instrument)
            label = name or instrument
        elif isinstance(instrument, Patch):
            patch = instrument
            label = name or instrument.name
        else:
            raise TypeError("instrument musí být jméno presetu nebo Patch")
        t = Track(self, patch, self._unique(label), vol, pan, chromatic,
                  mono=bool(mono) if mono is not None else False,
                  glide=float(glide) if glide is not None else 0.0)
        if fx:
            t.fx(*fx)
        self.tracks.append(t)
        return t

    def drums(self, vol: float = 0.6, pan: float = 0.0,
              name: str = "drums") -> DrumTrack:
        """Přidá stopu bicích."""
        t = DrumTrack(self, self._unique(name), vol, pan)
        self.tracks.append(t)
        return t

    def get(self, name: str) -> Track:
        for t in self.tracks:
            if t.name == name:
                return t
        raise KeyError(f"stopa {name!r} neexistuje")

    def note_count(self) -> int:
        return sum(len(t) for t in self.tracks)

    # -- výstup -----------------------------------------------------------
    def render(self, path: str | None = None, report: bool = True,
               bars=None, waveform: bool = False):
        """Vyrenderuje skladbu do WAV. Při importu z CLI je to no-op."""
        if _SUPPRESS_RENDER:
            self.rendered_path = path
            return None
        from .render import render_song
        return render_song(self, path, report=report, bars=bars,
                           waveform=waveform)

    def show(self, bars=None, res: int = 8) -> str:
        from .viz import piano_roll
        return piano_roll(self, bars=bars, res=res)
