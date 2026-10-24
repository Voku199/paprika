"""Paprika — psaní a render skladeb z Pythonu, bez DAW a VST.

Typické použití::

    from paprika import Song, Hall, LowPass

    s = Song(bpm=96, key="D", scale="minor", bars=16)
    choir = s.track("choir_ooh", fx="default")          # sbor + cathedral hall
    bass = s.track("808", fx="default")                  # mono, glide, saturace
    dr = s.drums(kit="trap")
    fx = s.sfx()

    s.harmony("i | i | VI | V")                          # plán pro `check`
    choir.chord("Dm", bar=0, dur=8)
    bass.notes("D1:3 >F1:1 | D1:4", bar=8)
    dr.pattern("k......k..k.....", bars=(8, 16)).roll("h", bar=11, beat=3)
    fx.riser(to_bar=8, bars=2).impact(8)
    s.render("output/skica.wav")

Skutečné nástroje (koncertní křídlo, smyčce, dechy, žestě) hraje sampler
z volně licencovaných nahrávek: `python -m paprika samples install all`,
pak `s.track("grand_piano")`, `s.track("violins.pizz")`.

Paprika není generátor hudby: melodii, harmonii a strukturu píše autor.
Knihovna dává teorii, syntézu, efekty, mix a zpětnou vazbu.
"""

from . import dsp
from .analyze import (band_energy, check_report, model_warnings, report,
                      warnings_for)
from .automation import Automation
from .drums import DRUM_CHARS, drum_names, parse_pattern
from .fx import (EQ, Delay, Filter, Fx, HighPass, LowPass, Reverb, Saturate,
                 Sidechain)
from .fx_dyn import Compressor, DynEQ, Gate
from .fx_mic import Proximity, RoomTone
from .fx_lofi import BitCrush, Tape
from .fx_space import Chorus, Hall, Width
from .harmony import HarmonyError, parse_chord, roman_of
from .kits import kit_names
from .library import (INSTRUMENTS, SamplesMissing, instrument, instrument_names,
                      samples_root)
from .mastering import MASTERING_STYLES, mastering
from .midi import export as export_midi
from .midi import read as read_midi
from .note import DEFAULT_VELOCITY, Note, NoteLengthError, NoteParseError, parse
from .patch import (ADSR, BP, FM, HP, LFO, LP, Additive, Ensemble, Formant, Glottal,
                    Noise, Osc, Patch, PitchEnv, Pluck, Pulse, Saw, Sine, Square,
                    SuperSaw, Tract, Tri, midi_to_freq)
from .presets import (PRESET_TAGS, PRESETS, preset, preset_fx, preset_names,
                      presets_by_tag, tag_names)
from .render import RenderResult, limiter, mix, render_song, write_wav
from .sampler import SampleError, Sampler
from .corpus import CORPUS_KINDS, MissingCorpus
from .human import HUMAN_NAMES
from .sfx import SFX_NAMES
from .song import Bus, Clip, DrumTrack, HumanTrack, Section, SfxTrack, Song, Track
from .theory import (CHORDS, SCALES, TheoryError, arp, chord, in_scale, invert,
                     note_name, pitch, scale, scale_pcs)
from .viz import piano_roll, waveform

__version__ = "2.8.0"

__all__ = [
    # model
    "Song", "Track", "DrumTrack", "SfxTrack", "HumanTrack", "Bus", "Clip",
    "Section", "Note",
    "Automation",
    # teorie
    "scale", "scale_pcs", "chord", "arp", "invert", "pitch", "note_name",
    "in_scale", "SCALES", "CHORDS", "TheoryError",
    "HarmonyError", "parse_chord", "roman_of",
    # zvuk
    "Patch", "Osc", "Sine", "Saw", "Square", "Tri", "Noise", "FM",
    "SuperSaw", "Pulse", "Pluck", "Additive", "Glottal",
    "ADSR", "LP", "HP", "BP", "Formant", "Tract", "LFO", "PitchEnv", "Ensemble",
    "midi_to_freq",
    "preset", "preset_names", "preset_fx", "presets_by_tag", "tag_names",
    "PRESETS", "PRESET_TAGS",
    "Sampler", "SampleError", "SamplesMissing", "MissingCorpus", "CORPUS_KINDS",
    "instrument",
    "instrument_names", "samples_root", "INSTRUMENTS",
    # efekty
    "Fx", "Reverb", "Hall", "Delay", "LowPass", "HighPass", "Filter", "EQ",
    "Saturate", "Sidechain", "Compressor", "DynEQ", "Gate", "Chorus", "Width",
    "Proximity", "RoomTone",
    "BitCrush", "Tape", "mastering", "MASTERING_STYLES",
    # render a zpětná vazba
    "render_song", "RenderResult", "mix", "limiter", "write_wav",
    "report", "warnings_for", "model_warnings", "check_report", "band_energy",
    "piano_roll", "waveform", "export_midi", "read_midi",
    # noty, bicí, sfx
    "parse", "NoteLengthError", "NoteParseError", "DEFAULT_VELOCITY",
    "parse_pattern", "drum_names", "DRUM_CHARS", "kit_names", "SFX_NAMES",
    "HUMAN_NAMES",
    "__version__",
]
