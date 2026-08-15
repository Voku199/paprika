"""CLI: ``python -m paprika render|show|play``.

Skladba je normální Python modul. `render` ho naimportuje a najde na úrovni
modulu právě jednu instanci `Song` — tu vyrenderuje.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys

from .song import Song, suppress_render

__all__ = ["main", "load_song", "default_output"]


class CliError(Exception):
    """Chyba, kterou má uživatel vidět jako hlášku, ne jako traceback."""


def load_song(path: str) -> tuple[Song, object]:
    """Naimportuje soubor skladby a vrátí jedinou `Song` v jeho namespace."""
    if not os.path.isfile(path):
        raise CliError(f"soubor neexistuje: {path}")
    name = "paprika_song_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    if spec is None or spec.loader is None:
        raise CliError(f"{path} se nedá naimportovat jako Python modul")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        with suppress_render():
            spec.loader.exec_module(module)
    finally:
        if added and root in sys.path:
            sys.path.remove(root)

    found = [(k, v) for k, v in vars(module).items()
             if isinstance(v, Song) and not k.startswith("__")]
    if not found:
        raise CliError(f"{path}: na úrovni modulu není žádná instance Song")
    if len(found) > 1:
        names = ", ".join(k for k, _ in found)
        raise CliError(f"{path}: našel jsem víc skladeb ({names}) — "
                       f"nech v modulu jen jednu Song")
    return found[0][1], module


def default_output(song_path: str, song: Song) -> str:
    """songs/romantic.py -> output/romantic.wav"""
    if song.rendered_path:
        return song.rendered_path
    base = os.path.splitext(os.path.basename(song_path))[0]
    root = os.path.dirname(os.path.dirname(os.path.abspath(song_path)))
    return os.path.join(root, "output", base + ".wav")


def _parse_bars(text: str | None):
    if not text:
        return None
    try:
        lo, _, hi = text.partition(":")
        return int(lo), int(hi) if hi else None
    except ValueError:
        raise CliError(f"--bars čekám ve tvaru 0:8, dostal jsem {text!r}")


def _open_file(path: str) -> None:
    if not os.path.isfile(path):
        raise CliError(f"soubor neexistuje: {path}")
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606 - záměrné otevření systémovým přehrávačem
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m paprika",
        description="Paprika — psaní a render skladeb bez DAW.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="vyrenderuje skladbu do WAV a vypíše report")
    r.add_argument("song", help="cesta k songs/*.py")
    r.add_argument("-o", "--out", help="výstupní WAV (jinak output/<jméno>.wav)")
    r.add_argument("--bars", help="rozsah taktů, např. 0:8")
    r.add_argument("--waveform", action="store_true", help="vykreslí obálku mixu")
    r.add_argument("--no-report", action="store_true", help="jen render, bez analýzy")
    r.add_argument("--play", action="store_true", help="po renderu rovnou přehraje")

    s = sub.add_parser("show", help="ASCII timeline bez renderu")
    s.add_argument("song", help="cesta k songs/*.py")
    s.add_argument("--bars", help="rozsah taktů, např. 0:4")
    s.add_argument("--res", type=int, default=8, help="znaků na takt (výchozí 8)")

    pl = sub.add_parser("play", help="otevře WAV systémovým přehrávačem")
    pl.add_argument("wav", help="cesta k .wav")

    sub.add_parser("presets", help="vypíše dostupné presety a bicí")
    return p


def _utf8_console() -> None:
    """Report obsahuje diakritiku; na Windows konzoli je výchozí kódování cp125x."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass


def main(argv=None) -> int:
    _utf8_console()
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "render":
            song, _ = load_song(args.song)
            out = args.out or default_output(args.song, song)
            song.render(out, report=not args.no_report,
                        bars=_parse_bars(args.bars), waveform=args.waveform)
            print(f"\n-> {out}")
            if args.play:
                _open_file(out)
        elif args.cmd == "show":
            song, _ = load_song(args.song)
            print(song.show(bars=_parse_bars(args.bars), res=args.res))
        elif args.cmd == "play":
            _open_file(args.wav)
        elif args.cmd == "presets":
            from .drums import DRUM_CHARS
            from .presets import preset_names
            print("presety:  " + ", ".join(preset_names()))
            print("bicí:     " + ", ".join(f"{k} = {v}"
                                           for k, v in DRUM_CHARS.items()))
    except CliError as e:
        print(f"chyba: {e}", file=sys.stderr)
        return 2
    return 0
