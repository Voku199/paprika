# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Paprika is a pure-Python music library: write a song as a Python module, render
it to WAV with hand-written synthesis. Dependencies are `numpy` and `scipy`
(scipy only for `lfilter` IIR filters and FFT convolution; WAV via stdlib
`wave`; oscillators, envelopes, effects, limiter all written by hand).

Current goal (engine v2, 2026-09): varied instrumental Arcane-style soundtrack
for a Unity game. Specs live in `docs/superpowers/specs/`; engine v2 is done,
next sub-projects are style packs + `/paprika` skill, game export (loops,
stems, stingers), SFZ sampler.

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
python -m paprika play   output/x.wav
python -m paprika presets

python -m pytest tests/                                  # ~285 tests, ~15 s
PAPRIKA_SLOW=1 python -m pytest tests/test_compat.py     # renders every song, ~1 min
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

## Architecture

Layer boundaries are deliberately sharp — keep them:

- `theory.py` — MIDI ints only (C4 = 60). Knows nothing about audio.
- `note.py` — parses notation strings into `Note` (pitch, start_beat, dur_beats, velocity, glide).
- `song.py` — `Song` / `Track` / `DrumTrack` / `SfxTrack` / `Bus` / `Clip`. Pure data model, no files, no CLI.
- `dsp.py` — pure array DSP (biquads, time-varying filters, LFO, noise, Karplus-Strong, synthetic IR, polyBLEP). No model, no theory.
- `oscillators.py`, `filters.py`, `modulation.py` — building blocks of a patch.
- `patch.py` — `ADSR` + `Patch.voice()`: one phrase → one mono `(n,)` or stereo `(n, 2)` buffer; re-exports the building blocks.
- `presets.py`, `kits.py`, `sfx.py` — instruments, drum kits, cinematic SFX; all synthesized, no samples.
- `fx.py`, `fx_space.py`, `fx_dyn.py`, `fx_lofi.py` — track/bus/master effects.
- `automation.py` — automation curves and target resolution.
- `render.py` — model → stereo buffers → buses → master → WAV. Knows no music theory.
- `analyze.py` — reads only the finished buffer + the data model.
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
- **New warning rule** — a `_rule(song, ..., warns)` function in `analyze.py`
  appending `Warning_("WARN"|"INFO", where, text)`, wired into `warnings_for`.
  Rules must cap their output (existing ones stop after 3–4 hits) so the report
  stays readable. Ignore near-silent bars (convolution leaves ~1e-17 noise).
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
