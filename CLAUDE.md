# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Paprika is a pure-Python music library: write a song as a Python module, render
it to WAV with hand-written synthesis. Only dependency is `numpy` (WAV via stdlib
`wave`; oscillators, filters, reverb, limiter all written by hand).

The library is *not* a music generator — melody, harmony and structure are
written by the author (usually the AI). Paprika supplies theory, synthesis, FX,
mix and **textual feedback**, because the AI cannot hear the output. The
report + ASCII piano roll are the only feedback loop; treat them as the primary
verification tool after every change to a song.

`README.md` documents the user-facing API (note notation, track methods,
presets, drum chars, warning rules). Read it before writing a song file.
`docs/superpowers/specs/2026-08-12-paprika-design.md` records the design
decisions and why v1 stops where it does.

## Commands

```bash
pip install -r requirements.txt                          # numpy only

python -m paprika render songs/romantic.py               # WAV + report
python -m paprika render songs/x.py --bars 0:8           # fast preview slice
python -m paprika render songs/x.py --waveform --play    # + envelope, then open
python -m paprika show   songs/x.py --bars 12:16 --res 16  # ASCII roll, no render
python -m paprika play   output/x.wav
python -m paprika presets

python -m pytest tests/                                  # 101 tests, ~6 s
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
- `note.py` — parses notation strings into `Note` (pitch, start_beat, dur_beats, velocity).
- `song.py` — `Song` / `Track` / `DrumTrack` / `Clip`. Pure data model, no files, no CLI.
- `patch.py` — one note → one mono `voice` buffer (oscillators, ADSR, filters).
- `presets.py`, `drums.py` — instruments; drums are synthesized, no samples.
- `fx.py` — track-level effects.
- `render.py` — model → stereo buffer → WAV. Knows no music theory.
- `analyze.py` — reads only the finished buffer + the data model.
- `viz.py`, `cli.py` — ASCII output, argparse.

**Everything is in beats, never seconds.** Conversion to samples happens once, in
`render_song`, so changing BPM is a single parameter. `Clip.start_beat` is
absolute on the timeline; `Note.start_beat` is relative to its clip.

### Render pipeline (`render_song`)

1. `_tail_seconds` — buffer length = song span + max(fx tail + ADSR release).
   Any effect that rings out **must** report `tail(ctx)` or it gets truncated.
2. Two passes over tracks: normal tracks first, then tracks whose fx chain has
   `needs_tracks = True` (only `Sidechain`), which read `ctx.track_buffers`.
3. `mix` — equal-power pan + vol → `(n, 2) float32`.
4. `limiter` — tanh soft limiter above −1 dBFS; `raw_peak` is captured *before*
   it so `analyze` can still report clipping.
5. `_trim` trailing silence, but never below the song's own length.

### Voice cache

`render_track` caches synthesized voices by `(patch.key(), freq, dur)`. Any new
`Osc`, filter, or `Patch` field **must** be included in the corresponding
`key()`, otherwise two different sounds collide in the cache and one is silently
rendered as the other.

### DrumTrack is a special case

`DrumTrack` stores `DrumHit`s in `.hits`, not notes — its `all_notes()` returns
`[]` on purpose. Any code walking tracks (analysis, viz, render) must branch on
`isinstance(t, DrumTrack)`. Drum hits are rendered at velocity 127 once and
cached in `drums._CACHE`, then scaled per hit.

## Extending

- **New preset** — add a `_name() -> Patch` factory in `presets.py` and register
  it in `PRESETS`. `preset()` returns a fresh instance each call, so per-track
  tweaks don't leak.
- **New oscillator / filter** — subclass `Osc` (or match the `LP`/`HP` shape:
  `apply(x, sr, env)`), implement `key()`, export from `__init__.py`.
- **New effect** — subclass `Fx` with `process(buf, ctx)` and `tail(ctx)`.
  Feedback structures are computed **block-wise by delay length** (see
  `_feedback_delay`, `_allpass`) so recursion stays vectorized. Do not write
  per-sample Python loops; the only accepted one is the variable-cutoff branch
  of `onepole_lp`.
- **New warning rule** — a `_rule(song, ..., warns)` function in `analyze.py`
  appending `Warning_("WARN"|"INFO", where, text)`, wired into `warnings_for`.
  Rules must cap their output (existing ones stop after 3–4 hits) so the report
  stays readable.
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
