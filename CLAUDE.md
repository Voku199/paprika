# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Paprika is a pure-Python music library: write a song as a Python module, render
it to WAV with hand-written synthesis — or, for real acoustic instruments, with
the SFZ sampler. Dependencies are `numpy` and `scipy` (scipy for `lfilter` IIR
filters, FFT convolution and `scipy.io.wavfile` sample reading; output WAV via
stdlib `wave`; oscillators, envelopes, effects, limiter all written by hand).

Current goal (engine v2, 2026-09): varied instrumental Arcane-style soundtrack
for a Unity game. Specs live in `docs/superpowers/specs/`; engine v2 and the
SFZ sampler (`2026-09-24-sampler-design.md`, pulled forward because synthesis
still sounded "chiptune") are done; next sub-projects are style packs +
`/paprika` skill and game export (loops, stems, stingers). The AI-engine
roadmap (`docs/superpowers/research/2026-10-21-ai-engine-pruzkum.md`) phase 1
— harmony plan, theory checks, MIDI export — is done
(`docs/superpowers/plans/2026-10-21-teorie-midi.md`). Human sounds (breath,
vocal gestures, body/foley, mic simulation) have a research proposal, not yet
approved: `docs/superpowers/research/2026-10-23-lidske-zvuky-pruzkum.md`.

The library is *not* a music generator — melody, harmony and structure are
written by the author (usually the AI). Paprika supplies theory, synthesis, FX,
mix and **textual feedback**, because the AI cannot hear the output. The
report + ASCII piano roll are the only feedback loop; treat them as the primary
verification tool after every change to a song.

`README.md` documents the user-facing API (note notation, track methods,
presets, drum chars, warning rules). Read it before writing a song file.
`docs/superpowers/specs/2026-08-12-paprika-design.md` records the v1 design,
`2026-09-23-engine-v2-design.md` the v2 engine. `songs/v2_showcase.py`
exercises every v2 feature.

## Commands

```bash
pip install -r requirements.txt                          # numpy + scipy

python -m paprika render songs/romantic.py               # WAV + report
python -m paprika render songs/x.py --bars 0:8           # fast preview slice
python -m paprika render songs/x.py --waveform --play    # + envelope, then open
python -m paprika show   songs/x.py --bars 12:16 --res 16  # ASCII roll, no render
python -m paprika check  songs/x.py --bars 8:16          # theory + model rules, no render
python -m paprika midi   songs/x.py                      # output/x.mid
python -m paprika play   output/x.wav
python -m paprika presets
python -m paprika compare ref.mp3 output/x.wav          # mix metrics vs a reference
python -m paprika samples                               # sampler: path, groups, instruments
python -m paprika samples install all                   # ~2.3 GB, resumable

python -m pytest tests/                                  # ~520 tests, ~75 s
PAPRIKA_SLOW=1 python -m pytest tests/test_compat.py     # renders every song, ~2 min
PAPRIKA_SLOW=1 python -m pytest tests/test_samples_real.py  # installed samples: level + tuning
python -m pytest tests/test_render.py::TestLimiter::test_tames_overs
python -m unittest discover tests                        # pytest not required
```

Rendering a full song is slow (tens of seconds); iterate with `--bars` slices
and `show` before a full render.

## Song file contract

A song is a plain Python module under `songs/`. The CLI imports it and looks for
**exactly one module-level `Song` instance** — zero or two is a `CliError`. So:
build helper `Song`s inside functions, never at module level.

During that import `song.suppress_render` makes `s.render(...)` a no-op (global
`_SUPPRESS_RENDER` flag), so a trailing `s.render("output/x.wav")` in the file is
optional and never double-renders. Output path falls back to
`<repo>/output/<song-basename>.wav`, derived from the *parent of the parent* of
the song path.

Write the harmony first: `s.harmony("i | VI | iv | V")` (and `s.section()`),
then the parts, then `python -m paprika check` — it reports wrong bass notes,
unresolved non-chord tones, parallels and cadences per bar in about a second.
The plan, sections and `Song(theory=...)` are metadata: they never change the
sound (render stays bit-identical). A chord tone from the plan is not "out of
scale", so `chromatic=True` is only for genuinely chromatic lines.

## Architecture

Layer boundaries are deliberately sharp — keep them:

- `theory.py` — MIDI ints only (C4 = 60). Knows nothing about audio.
- `harmony.py` — harmony plan text (chord names *and* Roman numerals relative to
  key/scale) -> `ChordSpan`s; `roman_of`, `key_signature`, `pc_name` (spelling
  by scale degree), `cadence`. Pitch classes only, knows no `Song`.
- `note.py` — parses notation strings into `Note` (pitch, start_beat, dur_beats, velocity, glide).
- `song.py` — `Song` / `Track` / `DrumTrack` / `SfxTrack` / `Bus` / `Clip`. Pure data model, no files, no CLI.
- `dsp.py` — pure array DSP (biquads, time-varying filters, LFO, noise, Karplus-Strong, synthetic IR, polyBLEP). No model, no theory.
- `oscillators.py`, `filters.py`, `modulation.py` — building blocks of a patch (`modulation.py` also holds `Ensemble`).
- `patch.py` — `ADSR` + `Patch.voice()`: one phrase → one mono `(n,)` or stereo `(n, 2)` buffer; re-exports the building blocks.
- `presets.py`, `kits.py`, `sfx.py` — instruments, drum kits, cinematic SFX; all synthesized, no samples.
- `sfz.py` — SFZ text -> `Region`s. Text only, no audio.
- `sampler.py` — `Sampler`: region selection (key/vel/round robin), pitch shift +
  resample in one Hermite read, loops, legato, release. Knows no beats or pedal.
- `library.py` — catalog of sampled instruments (`name.articulation`), sample root.
- `install.py` — the only module touching the network: Salamander + VSCO 2 CE +
  VocalSet download, safe tar/zip extraction, SFZ generation from VSCO file names.
- `voices.py` — VocalSet singing exercises -> single notes -> SFZ (`male_choir`,
  `basso`, `tenor`, `female_choir`, `soprano` = female2, `alto` = female7; the
  women sing an octave above the men, long tones C4/C5/F5, and get two PSOLA
  notes above their highest long tone too). YIN gives the pitch within the octave; the octave comes from
  the harmonic series (`spectral_octave`) because YIN misses by an octave both ways
  on sung notes, sometimes for whole files. Notes whose octave is unreadable are
  dropped. Below the lowest sung note, TD-PSOLA adds three deeper notes (formants
  kept); rough ones (subharmonic > -33 dB) are dropped. `catalog.json` caches the
  notes so SFZ can be regenerated without re-cutting; singers added to `VOICES`
  later are cut and appended to the existing catalog by `build()`.
- `fx.py`, `fx_space.py`, `fx_dyn.py`, `fx_lofi.py` — track/bus/master effects.
- `automation.py` — automation curves and target resolution.
- `metrics.py` — mix metrics of a finished buffer/file (7 bands, width, crest, LRA); used by the report and `compare`. Reads only audio.
- `mastering.py` — ready-made master chains built from existing effects.
- `render.py` — model → stereo buffers → buses → master → WAV. Knows no music theory.
- `rules.py` — music-theory rules over the data model only (no audio): bass vs
  plan, chord coverage, non-chord tones on strong beats, parallels, leading
  tone, melodic leaps, low thirds; strict adds spacing/crossing. Owns
  `Warning_` (with optional `bar`); `analyze` re-exports it.
- `analyze.py` — reads only the finished buffer + the data model;
  `model_warnings` (no audio) feeds `check_report` and `warnings_for`.
- `midi.py` — hand-written SMF format 1 writer + reader (running status). Reads
  only the model.
- `viz.py`, `cli.py` — ASCII output, argparse.

**Everything is in beats, never seconds.** Conversion to samples happens once, in
`render_song`, so changing BPM is a single parameter. `Clip.start_beat` is
absolute on the timeline; `Note.start_beat` is relative to its clip.

### Render pipeline (`render_song`)

1. `_tail_seconds` — buffer length = song span + max(fx tail + ADSR release)
   + bus/master fx tails. Any effect that rings out **must** report
   `tail(ctx)` or it gets truncated.
2. Dependency loop over tracks and buses: a track waits for its `Sidechain`
   sources, a bus for its member tracks. No progress in a round →
   `RuntimeError` naming the cycle. `ctx.track_buffers` holds finished tracks
   (post-fx, pre-vol) and buses.
3. Per track: `_phrases` (mono legato / `>` glide) → `Patch.voice` → stereo
   `(n, 2)` buffer → fx chain with automation curves set via
   `fx.set_curve()` and cleared afterwards.
4. Tracks get vol/pan (equal-power pan, automatable); bus members are summed
   into the bus, bus fx run, bus vol/pan is a **balance** law (pan 0 = unity).
5. Master fx + master automation, then `limiter` — tanh soft limiter above
   −1 dBFS; `raw_peak` is captured *before* it so `analyze` can report clipping.
6. `_trim` trailing silence, but never below the song's own length.

Everything stereo is `(n, 2)`. A mono-only effect implements
`process_mono(x, ctx)`; the base `Fx.process` runs it per channel. Effects
that handle stereo override `process` and must accept `(n,)` too.

### Voice cache

`render_track` caches synthesized voices by `(patch.key(), freq, dur, rr_index,
tone_vel, glide_path, glide_time, bpm)`. Any new `Osc`, filter, modulation, or
`Patch` field **must** be included in the corresponding `key()`, otherwise two
different sounds collide in the cache and one is silently rendered as the
other (`tests/test_modulation.py::test_every_field_changes_key` guards this).

`Patch(ensemble=...)` renders N detuned, time-smeared, panned copies of the
oscillator+envelope core and runs the patch filters once on their stereo sum;
the voice is longer by `smear`. ADSR defaults to `curve="exp"`; `Reverb`
defaults to convolution (`algo="freeverb"` is v1).

"Retro / 16-bit" feel was measured against `songs_mp3/` references: it came
from mono sources, too much sub (< 60 Hz), thin mids, linear envelopes and
fast arpeggios. Check new songs with `compare` — the cinematic target is
sub ≤ 12 %, mids ≥ 20 %, width ≥ 10 % (`metrics.TARGETS`).

### Sampler (real instruments)

A track's `patch` is either a `Patch` or a `Sampler`; `render_track` branches on
`isinstance(track.patch, Sampler)` (after the Drum/Sfx branches) into
`_render_sampled`. Scheduling lives in `render.py`: `_pedaled` applies
`Track.pedals` (also used for poly synth tracks in `_phrases`), and
`_sampled_events` marks legato pairs (mono track or `>` note, and
`Sampler.legato`). Sampled voices are cached by `(sampler.key(), region id,
pitch, dur, legato_in, legato_out, sr)`; velocity is applied afterwards as
`Sampler.gain` (SFZ `amp_veltrack` law) so humanized velocities still hit the
cache. `_tail_seconds` uses `Sampler.tail`.

`dyn` automation (sampled tracks only; `automation.resolve(..., sampled=True)`
skips it, a synth track raises `KeyError`) is continuous dynamics like a
modwheel: `_dyn_voice` multiplies note velocity by the curve per sample,
renders every velocity layer with non-zero weight (`Sampler.layers`,
`sampler.layer_weights`: equal-power crossfade `DYN_XFADE` wide around layer
boundaries) and applies `Sampler.gain_curve`. Voices still come from the same
cache; without `dyn` the old single-layer path is untouched (bit-identical).

Samples live **outside the repo** (`PAPRIKA_SAMPLES`, else
`%LOCALAPPDATA%\paprika\samples`, else `~/.paprika/samples`). WAVs are read
lazily via `scipy.io.wavfile.read(mmap=True)` and cached per process in
`sampler._WAVS`; on Windows the mmaps keep files open, so tests call
`sampler.clear_cache()` before deleting temp dirs. Tests build tiny synthetic
libraries with `tests/sample_fixture.py` — no network.

VSCO 2 CE names most folders with C3 = middle C (pitch from name + 12); the
exceptions are listed in `install.VSCO_OCTAVE_ZERO` (measured by pitch
detection). Files without a note in the name (timpani `Timpani3_Hit_...`) get
their pitch from `install.VSCO_NAMED_PITCH` (measured, fractional MIDI): the
root is rounded and the rest goes into the SFZ `tune` opcode. Articulations in
`library.LOOP_ARTS` (`roll`) loop like sustains even outside the
strings/winds/brass groups. Generated SFZ are normalized: median max-RMS of the top layer to
−18 dBFS with a 0.9 peak ceiling. Sampled instrument names must not collide
with preset names (`test_library.test_catalog`). A missing library raises
`SamplesMissing` with the install command — never fall back to synthesis
silently (the AI cannot hear the difference).

### DrumTrack is a special case

`DrumTrack` stores `DrumHit`s in `.hits`, not notes — its `all_notes()` returns
`[]` on purpose. Any code walking tracks (analysis, viz, render) must branch on
`isinstance(t, DrumTrack)`. Drum hits are rendered at velocity 127 once and
cached in `drums._CACHE` keyed `(kit, name, sr)`, then scaled per hit.
`SfxTrack` subclasses `DrumTrack` (hits are `SfxHit` with `params`,
`dur_beats`); check `SfxTrack` **before** `DrumTrack` where they differ.

Mono tracks (`mono=True`): touching notes of a different pitch (or marked `>`)
merge into one voice; same pitch without `>` retriggers (808 patterns).

## Extending

- **New preset** — add a `_name() -> Patch` factory in `presets.py`, register
  it in `PRESETS` and `_TAGGED`; optional `PRESET_DEFAULTS` (mono/glide) and
  `_default_fx`. `preset()` returns a fresh instance each call, so per-track
  tweaks don't leak. `tests/test_presets.py` checks level (< 1.0 at vel 127,
  55–880 Hz) and that every new preset differs spectrally from the others.
- **New sampled instrument** — an `Instrument` in `library._LIST` mapping each
  articulation to a VSCO repo folder (short arts `pizz/spic/stac` are one-shot,
  string/wind/brass sustains get loops). Verify the folder's octave naming by
  pitch detection (exceptions go in `install.VSCO_OCTAVE_ZERO`), rerun
  `python -m paprika samples install <group>`, then
  `PAPRIKA_SLOW=1 python -m pytest tests/test_samples_real.py`. Any SFZ file
  also works directly: `s.track(Sampler(path, legato=True))`.
- **New drum kit** — a dict of `name -> f(sr)` in `kits.py`, registered in
  `KITS`; missing instruments fall back to `classic`.
- **New oscillator / filter** — subclass `Osc` (`render(freq, n, sr, phase)`,
  `freq` scalar or per-sample curve) or match the patch-filter shape
  `apply(x, sr, env=None, scale=1.0, bpm=120.0, freq=440.0)` accepting `(n,)`
  and `(n, 2)`; implement `key()`, export from `__init__.py`.
- **New effect** — subclass `Fx` with `process_mono(x, ctx)` (or a stereo-aware
  `process(buf, ctx)`) and `tail(ctx)`; list automatable params in
  `AUTOMATABLE` and read them via `self.value(param)`; accept `name=`.
  Feedback structures are computed **block-wise by delay length** (see
  `_feedback_delay`, `_allpass`) so recursion stays vectorized; IIR filters go
  through scipy `lfilter` (block-wise with carried `zi` when time-varying, see
  `dsp.filt_varying`, `dsp.onepole`). Do not write per-sample Python loops.
- **New warning rule** — a `_rule(song, ..., warns)` function appending
  `Warning_("WARN"|"INFO", where, text, bar)`: audio rules in `analyze.py`
  wired into `warnings_for`, model-only rules into `model_warnings` (so
  `check` runs them without a render), music-theory rules in `rules.py`
  wired into `theory_warnings`. Rules must cap their output (existing ones stop
  after 3–4 hits) so the report stays readable. Ignore near-silent bars
  (convolution leaves ~1e-17 noise). Theory rules must stay quiet on real songs
  in `theory="free"` (doublings, arpeggios/ostinatos and pads are intentional;
  tolerate humanize with `rules.EPS`) — check `songs/` before adding a WARN.
- `__init__.py` re-exports the public API explicitly; anything user-facing goes
  in both the imports and `__all__`.

## Conventions

- **Czech** for docstrings, comments, error messages and report text. Keep it —
  errors are user-facing. `cli._utf8_console()` reconfigures stdout because the
  Windows console defaults to cp125x and the report has diacritics.
- Errors are precise and actionable: `NoteLengthError` names the bar,
  `Sidechain` lists available tracks, `preset()` lists valid names.
- Track methods return `self` — chaining is the expected style.
- Randomness takes a `seed` (`humanize`, `arp(pattern="random")`, `Noise`) so
  renders are reproducible.
- Buffers are `float32` mono until `mix`; stereo is `(n, 2)`.
- Tests are `unittest` classes bootstrapping `sys.path` to the project root
  (no `__init__.py` in `tests/`); pytest just collects them.
