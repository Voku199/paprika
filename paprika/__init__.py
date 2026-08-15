"""Paprika — psaní a render skladeb z Pythonu, bez DAW, VST a samplů.

Typické použití::

    from paprika import Song, Reverb

    s = Song(bpm=72, key="C", scale="major", bars=8)
    piano = s.track("epiano", vol=0.8)
    piano.fx(Reverb(room=0.7, mix=0.3))
    piano.melody("E5:1 G5:1 A5:2 | G5:1 E5:1 D5:2 | C5:4 | C5:4", bar=0)
    s.render("output/skica.wav")

Paprika není generátor hudby: melodii, harmonii a strukturu píše autor.
Knihovna dává teorii, syntézu, efekty, mix a zpětnou vazbu.
"""

from .analyze import band_energy, report, warnings_for
from .drums import DRUM_CHARS, drum_names, parse_pattern
from .fx import Delay, Fx, HighPass, LowPass, Reverb, Saturate, Sidechain
from .note import DEFAULT_VELOCITY, Note, NoteLengthError, NoteParseError, parse
from .patch import (ADSR, FM, HP, LP, Noise, Osc, Patch, Saw, Sine, Square,
                    Tri, midi_to_freq)
from .presets import PRESETS, preset, preset_names
from .render import RenderResult, limiter, mix, render_song, write_wav
from .song import Clip, DrumTrack, Song, Track
from .theory import (CHORDS, SCALES, TheoryError, arp, chord, in_scale, invert,
                     note_name, pitch, scale, scale_pcs)
from .viz import piano_roll, waveform

__version__ = "1.0.0"

__all__ = [
    # model
    "Song", "Track", "DrumTrack", "Clip", "Note",
    # teorie
    "scale", "scale_pcs", "chord", "arp", "invert", "pitch", "note_name",
    "in_scale", "SCALES", "CHORDS", "TheoryError",
    # zvuk
    "Patch", "Osc", "Sine", "Saw", "Square", "Tri", "Noise", "FM",
    "ADSR", "LP", "HP", "midi_to_freq",
    "preset", "preset_names", "PRESETS",
    # efekty
    "Fx", "Reverb", "Delay", "LowPass", "HighPass", "Saturate", "Sidechain",
    # render a zpětná vazba
    "render_song", "RenderResult", "mix", "limiter", "write_wav",
    "report", "warnings_for", "band_energy", "piano_roll", "waveform",
    # noty a bicí
    "parse", "NoteLengthError", "NoteParseError", "DEFAULT_VELOCITY",
    "parse_pattern", "drum_names", "DRUM_CHARS",
    "__version__",
]
