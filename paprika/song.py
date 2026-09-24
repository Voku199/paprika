"""Datový model skladby: Song, Track, DrumTrack, Clip.

Noty žijí v beatech, ne v sekundách. Převod na samply se děje až v `render()`,
takže změna BPM je jeden parametr a nic se nepřepisuje.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from . import drums as _drums
from . import note as _note
from . import theory
from .note import Note
from .library import (instrument as _sampled_instrument, instrument_defaults,
                      instrument_fx, is_sampled)
from .patch import Patch
from .presets import preset
from .sampler import Sampler

__all__ = ["Song", "Track", "DrumTrack", "SfxTrack", "Bus", "Clip",
           "suppress_render"]

# Když skladbu importuje CLI, `Song.render()` uvnitř souboru se má chovat jako
# no-op — jinak by se renderovalo dvakrát. CLI si tenhle přepínač zapíná samo.
_SUPPRESS_RENDER = False

PEDAL_LIFT = 1.0 / 16.0   # beaty: zvednutí nohy při přešlápnutí pedálu


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

    def __init__(self, song: "Song", patch: Patch | Sampler, name: str, vol: float = 0.8,
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
        self.bus = None
        # sustain pedál: (down_beat, up_beat); noty končící při sešlápnutí
        # zní do uvolnění (render._pedaled)
        self.pedals: list = []

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

    def pedal(self, bars, every: float | None = None):
        """Sustain pedál (piano): nota, která skončí při sešlápnutí, zní do
        uvolnění. `bars=(od, do)` v taktech; `every` = přešlápnutí po tolika
        taktech (1 = každý takt, jak to dělá pianista při změně harmonie)::

            piano.pedal(bars=(0, 8), every=1)

        Pedál se sešlápne `PEDAL_LIFT` beatu po začátku úseku, takže nota
        končící přesně na hranici se nezachytí (noha je zrovna nahoře).
        """
        lo, hi = float(bars[0]), float(bars[1])
        if hi <= lo:
            raise ValueError(f"pedal: bars=(od, do) musí mít do > od, "
                             f"dostal jsem {bars!r}")
        step = float(every) if every else hi - lo
        if step <= 0:
            raise ValueError("pedal: every musí být kladné")
        b = lo
        while b < hi - 1e-9:
            e = min(hi, b + step)
            self.pedals.append((self._bar_to_beat(b) + PEDAL_LIFT,
                                self._bar_to_beat(e)))
            b = e
        self.pedals.sort()
        return self

    def to(self, bus):
        """Pošle stopu do busu (objekt `Bus` nebo jeho jméno).

        `to(song.master)` / `to("master")` vrátí stopu rovnou do mixu.
        """
        if isinstance(bus, str):
            bus = self.song.master if bus == "master" else self.song.get_bus(bus)
        if bus is self.song.master:
            self.bus = None
            return self
        if bus not in self.song.buses:
            raise ValueError(f"bus {getattr(bus, 'name', bus)!r} nepatří téhle "
                             f"skladbě — vytvoř ho přes song.bus()")
        self.bus = bus
        return self

    def humanize(self, timing: float = 0.01, vel: float = 0.1,
                 seed: int | None = None):
        """Rozhýbe časování (v beatech) a dynamiku (relativně) všech not."""
        rng = random.Random(seed)
        self._humanized = True
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
                 pan: float = 0.0, kit: str = "classic"):
        from .kits import kit_names
        if kit not in kit_names():
            raise KeyError(f"neznámá bicí sada {kit!r}; k dispozici: "
                           f"{', '.join(kit_names())}")
        super().__init__(song, Patch(name="drums"), name, vol, pan, chromatic=True)
        self.kit = kit
        self.hits: list[_drums.DrumHit] = []

    def __len__(self) -> int:
        return len(self.hits)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DrumTrack {self.name} hits={len(self.hits)}>"

    def pattern(self, text: str, bars=(0, 1), rate: float = 1 / 16,
                vel: float = 1.0, swing: float = 0.0):
        """Zopakuje pattern přes rozsah taktů. Jeden znak = jeden krok.

        `swing` 0..1 posune každý sudý krok (2., 4., ...) o tenhle podíl
        kroku dozadu — shuffle / groove místo strojové mřížky.
        """
        lo, hi = (bars if isinstance(bars, (tuple, list)) else (bars, bars + 1))
        step = float(rate) * self.bpb
        if step <= 0:
            raise ValueError("rate musí být kladné")
        steps_per_bar = int(round(self.bpb / step))
        raw = _drums.parse_pattern(text, step_beats=step, start_beat=0.0)
        if not raw:
            return self
        if swing:
            for h in raw:
                if int(round(h.start_beat / step)) % 2 == 1:
                    h.start_beat += float(swing) * step
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

    def _resolve(self, name: str) -> str:
        if name in _drums.drum_names():
            return name
        full = _drums.DRUM_CHARS.get(str(name).lower()) if len(str(name)) == 1 else None
        if full is None:
            raise ValueError(f"neznámý bicí nástroj {name!r}; k dispozici: "
                             f"{', '.join(_drums.drum_names())} nebo znaky "
                             f"{''.join(sorted(_drums.DRUM_CHARS))}")
        return full

    def roll(self, name: str, bar: float = 0, beat: float = 0.0,
             beats: float = 1.0, rate: float = 1 / 32, vel=(60, 110)):
        """Víření: rychlé údery po `rate` přes `beats` beatů s rampou velocity.

        Trap hi-hat roll: `roll("h", bar=7, beat=3, beats=1, rate=1/32)`.
        """
        full = self._resolve(name)
        step = float(rate) * self.bpb
        if step <= 0:
            raise ValueError("rate musí být kladné")
        count = max(1, int(round(float(beats) / step)))
        lo, hi = (vel, vel) if np.isscalar(vel) else vel
        start = self._bar_to_beat(bar) + beat
        for i, v in enumerate(np.linspace(lo, hi, count)):
            self.hits.append(_drums.DrumHit(full, start + i * step,
                                            int(max(1, min(127, round(v))))))
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
        self._humanized = True
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


class SfxTrack(DrumTrack):
    """Stopa cinematic efektů: riser, impact, braam, subdrop, reverse swell,
    whoosh. Dědí z `DrumTrack`, takže ji analýza i vizualizace berou jako
    stopu úderů. Délky jsou v taktech/beatech a sedí na hrany taktů.
    """

    def __init__(self, song: "Song", name: str = "sfx", vol: float = 0.6,
                 pan: float = 0.0):
        super().__init__(song, name, vol, pan)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<SfxTrack {self.name} hits={len(self.hits)}>"

    def _put(self, name, start_beat, vel, params=(), dur_beats=0.0):
        from .sfx import SfxHit
        self.hits.append(SfxHit(name, float(start_beat), int(vel), params, dur_beats))
        return self

    def riser(self, to_bar: float, bars: float = 2, kind: str = "noise",
              vel: int = 110):
        """Nájezd, který skončí přesně na začátku taktu `to_bar`."""
        from .sfx import RISER_KINDS
        if kind not in RISER_KINDS:
            raise ValueError(f"neznámý druh riseru {kind!r}; k dispozici: "
                             f"{', '.join(RISER_KINDS)}")
        beats = float(bars) * self.bpb
        return self._put("riser", self._bar_to_beat(to_bar) - beats, vel,
                         (kind, beats), beats)

    def impact(self, bar: float, beat: float = 0.0, size: float = 1.0,
               vel: int = 115):
        """Trailerová rána: sub boom + crack + dlouhý ocas."""
        return self._put("impact", self._bar_to_beat(bar) + beat, vel, (float(size),))

    def braam(self, bar: float, pitch="D2", bars: float = 2, vel: int = 110):
        """Žesťová „braam“ rána (Inception) na daném tónu."""
        midi = theory.pitch(pitch)
        beats = float(bars) * self.bpb
        return self._put("braam", self._bar_to_beat(bar), vel, (int(midi), beats), beats)

    def subdrop(self, bar: float, beat: float = 0.0, vel: int = 110):
        """Klesající sub (65 → 28 Hz) — pod drop nebo impact."""
        return self._put("subdrop", self._bar_to_beat(bar) + beat, vel)

    def reverse_swell(self, to_bar: float, beats: float = 2.0, vel: int = 100):
        """Obrácený činel/dozvuk, který vyvrcholí na začátku `to_bar`."""
        return self._put("reverse_swell", self._bar_to_beat(to_bar) - float(beats),
                         vel, (float(beats),), float(beats))

    def whoosh(self, bar: float, beats: float = 1.0, beat: float = 0.0,
               vel: int = 100):
        """Průlet šumu zleva doprava."""
        return self._put("whoosh", self._bar_to_beat(bar) + beat, vel,
                         (float(beats),), float(beats))

    def humanize(self, timing: float = 0.0, vel: float = 0.1,
                 seed: int | None = None):
        """Rozhýbe jen velocity — časování SFX musí sedět na hrany taktů."""
        return super().humanize(timing=0.0, vel=vel, seed=seed)


class Bus:
    """Sběrnice: sečte své stopy, prožene je vlastním fx řetězem a pošle dál.

    Typicky společná komprese bicích nebo reverb na skupinu. `pan` je tady
    balance (stereo obraz busu zůstane), ne equal-power pan jako u stopy.
    `Song.master` je taky `Bus` — poslední řetěz před limiterem.
    """

    def __init__(self, song: "Song", name: str, vol: float = 1.0,
                 pan: float = 0.0):
        self.song = song
        self.name = name
        self.vol = float(vol)
        self.pan = float(max(-1.0, min(1.0, pan)))
        self.fx_chain: list = []
        self.automations: list = []

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Bus {self.name} fx={len(self.fx_chain)} vol={self.vol:g}>"

    def fx(self, *effects):
        """Připojí efekty na konec řetězu busu."""
        self.fx_chain.extend(effects)
        return self

    def automate(self, target: str, points, curve: str = "lin"):
        """Jako `Track.automate`: `vol`, `pan` nebo `<efekt>.<param>`."""
        from .automation import Automation
        self.automations.append(Automation(target, points, curve))
        return self

    def members(self) -> list:
        return [t for t in self.song.tracks if t.bus is self]


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
        self.buses: list[Bus] = []
        self.master = Bus(self, "master")
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
        taken = ({t.name for t in self.tracks} | {b.name for b in self.buses}
                 | {"master"})
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
        """Přidá stopu. `instrument` je preset, samplovaný nástroj
        (`"grand_piano"`, `"violins.pizz"`), `Patch` nebo `Sampler`.

        `mono=True` zapne mono legato, `glide` je délka sklouznutí v beatech
        (když je nezadáš, vezmou se z `PRESET_DEFAULTS`, např. 808 je mono;
        sólové nahrané nástroje jsou mono = legato).
        `fx` je seznam efektů, nebo `"default"` = doporučený řetěz nástroje.
        """
        from .presets import PRESET_DEFAULTS, preset_fx
        defaults = {}
        if isinstance(instrument, str) and is_sampled(instrument):
            patch = _sampled_instrument(instrument)
            label = name or instrument.strip().lower().replace(".", "_")
            defaults = instrument_defaults(instrument)
        elif isinstance(instrument, str):
            patch = preset(instrument)
            label = name or instrument
            defaults = PRESET_DEFAULTS.get(str(instrument).lower(), {})
        elif isinstance(instrument, (Patch, Sampler)):
            patch = instrument
            label = name or instrument.name
        else:
            raise TypeError("instrument musí být jméno presetu nebo samplovaného "
                            "nástroje, Patch nebo Sampler")
        if mono is None:
            mono = defaults.get("mono", False)
        if glide is None:
            glide = defaults.get("glide", 0.0)
        t = Track(self, patch, self._unique(label), vol, pan, chromatic,
                  mono=bool(mono), glide=float(glide))
        if isinstance(fx, str):
            if fx != "default" or not isinstance(instrument, str):
                raise ValueError("fx='default' funguje jen se jménem nástroje; "
                                 "jinak předej seznam efektů")
            t.fx(*(instrument_fx(instrument) if is_sampled(instrument)
                   else preset_fx(instrument)))
        elif fx:
            t.fx(*fx)
        self.tracks.append(t)
        return t

    def drums(self, vol: float = 0.6, pan: float = 0.0, name: str = "drums",
              kit: str = "classic") -> DrumTrack:
        """Přidá stopu bicích. `kit`: classic, trap, cinematic, breakbeat,
        industrial, lofi."""
        t = DrumTrack(self, self._unique(name), vol, pan, kit=kit)
        self.tracks.append(t)
        return t

    def sfx(self, vol: float = 0.6, pan: float = 0.0, name: str = "sfx") -> SfxTrack:
        """Přidá stopu cinematic efektů (riser, impact, braam, ...)."""
        t = SfxTrack(self, self._unique(name), vol, pan)
        self.tracks.append(t)
        return t

    def bus(self, name: str = "bus", vol: float = 1.0, pan: float = 0.0) -> Bus:
        """Přidá sběrnici. Stopy do ní pošleš přes `track.to(bus)`."""
        b = Bus(self, self._unique(name), vol, pan)
        self.buses.append(b)
        return b

    def get_bus(self, name: str) -> Bus:
        for b in self.buses:
            if b.name == name:
                return b
        raise KeyError(f"bus {name!r} neexistuje (k dispozici: "
                       f"{', '.join(b.name for b in self.buses) or '—'})")

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
