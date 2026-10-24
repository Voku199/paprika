"""CLI: ``python -m paprika render|show|check|midi|play``.

Skladba je normální Python modul. `render` ho naimportuje a najde na úrovni
modulu právě jednu instanci `Song` — tu vyrenderuje.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys

from .library import SamplesMissing
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

    c = sub.add_parser(
        "check",
        help="hudební teorie a pravidla nad notami bez renderu (vteřina)")
    c.add_argument("song", help="cesta k songs/*.py")
    c.add_argument("--bars", help="rozsah taktů, např. 0:8")

    m = sub.add_parser(
        "midi", help="zapíše skladbu do MIDI (stopy, tempo, sekce, akordy plánu)")
    m.add_argument("song", help="cesta k songs/*.py")
    m.add_argument("-o", "--out", help="výstupní .mid (jinak output/<jméno>.mid)")

    pl = sub.add_parser("play", help="otevře WAV systémovým přehrávačem")
    pl.add_argument("wav", help="cesta k .wav")

    ar = sub.add_parser(
        "analyze-ref",
        help="analyzuje externí nahrávku (mp3/wav) — tempo, key, akordy, dynamika")
    ar.add_argument("audio", help="cesta k mp3/wav souboru")

    cp = sub.add_parser(
        "compare",
        help="porovná mix metriky souborů (render vs. reference mp3/wav)")
    cp.add_argument("files", nargs="+", help="wav/mp3 soubory; první je základ")

    sm = sub.add_parser(
        "samples",
        help="skutečné nástroje (piano, smyčce, ...): stav a instalace vzorků")
    smsub = sm.add_subparsers(dest="samples_cmd")
    si = smsub.add_parser("install", help="stáhne a připraví vzorky skupiny")
    si.add_argument("group", help="piano, strings, winds, brass, mallets, voices nebo all")
    sb = smsub.add_parser(
        "breaths", help="modely nádechů z VocalSet: statistiky, ukázky k poslechu")
    sb.add_argument("--rebuild", action="store_true",
                    help="znovu najde nádechy v nahrávkách a přepočítá modely")
    sb.add_argument("--examples", metavar="SLOŽKA",
                    help="uloží skutečné a syntetizované nádechy jako WAV")
    sh = smsub.add_parser(
        "human", help="korpus lidských zvuků (smích, kašel, ...): stav, zadání, import")
    shsub = sh.add_subparsers(dest="human_cmd")
    hr = shsub.add_parser("request", help="zapíše zadání pro tools/human_worker.py")
    hr.add_argument("kinds", nargs="+", help="druhy: laugh, cough, sigh, ...")
    hr.add_argument("--backend", default="dia",
                    help="dia, chatterbox, stable_audio nebo paprika (výchozí dia)")
    hr.add_argument("--count", type=int, default=8, help="generací na druh (výchozí 8)")
    hr.add_argument("--seed", type=int, default=0, help="první seed (výchozí 0)")
    hr.add_argument("--duration", type=float, help="délka v s (stable_audio)")
    hr.add_argument("--model", help="jiný model backendu")
    hr.add_argument("--out", default="human_request.json",
                    help="soubor zadání (výchozí human_request.json)")
    hi = shsub.add_parser("import", help="prosije WAV ze složky a přijaté uloží do korpusu")
    hi.add_argument("folder", help="složka s WAV (a JSON od workeru)")
    hi.add_argument("--kind", help="druh pro WAV bez JSON (vlastní nahrávky)")
    hi.add_argument("--license", help="licence pro WAV bez JSON, např. CC0 nebo vlastní")
    hi.add_argument("--author", help="autor / zdroj pro atribuci")

    st = smsub.add_parser(
        "steps", help="povrchy kroků z vlastních nahrávek chůze: fit, srovnání, řez")
    stsub = st.add_subparsers(dest="steps_cmd")
    sf = stsub.add_parser("fit", help="odhadne povrch z nahrávek a uloží ho pod jménem")
    sf.add_argument("files", nargs="+", help="WAV/MP3 s chůzí (aspoň 4 kroky)")
    sf.add_argument("--name", required=True, help="jméno nového povrchu (pro steps(surface=))")
    sf.add_argument("--base", default="stone",
                    help="vestavěný povrch, ze kterého fit vyjde (výchozí stone)")
    sf.add_argument("--shoe", default="boot", help="bota v nahrávce (výchozí boot)")
    sf.add_argument("--license", help="licence nahrávek, např. CC0 nebo vlastní")
    sf.add_argument("--author", help="autor / zdroj nahrávek")
    sc = stsub.add_parser("compare", help="porovná nahrávky s povrchem (nic neukládá)")
    sc.add_argument("files", nargs="+", help="WAV/MP3 s chůzí")
    sc.add_argument("--surface", required=True, help="vestavěný nebo nafitovaný povrch")
    sc.add_argument("--shoe", default="boot", help="bota (výchozí boot)")
    sx = stsub.add_parser("cut", help="rozřeže nahrávku chůze na kroky (WAV)")
    sx.add_argument("file", help="WAV/MP3 s chůzí")
    sx.add_argument("--out", required=True, help="cílová složka")

    pr = sub.add_parser("presets", help="vypíše presety, bicí sady a SFX")
    pr.add_argument("--tag", help="jen presety s tagem (orchestral, choir, ...)")
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
        elif args.cmd == "check":
            from .analyze import check_report
            song, _ = load_song(args.song)
            bars = _parse_bars(args.bars)
            if bars and bars[1] is None:
                bars = (bars[0], song.bars)
            print(check_report(song, bars=bars, name=os.path.basename(args.song)))
        elif args.cmd == "midi":
            from .midi import export
            song, _ = load_song(args.song)
            out = args.out or os.path.splitext(default_output(args.song, song))[0] + ".mid"
            for note in export(song, out):
                print(f"INFO  {note}")
            print(f"-> {out}")
        elif args.cmd == "play":
            _open_file(args.wav)
        elif args.cmd == "analyze-ref":
            from .reference import (ReferenceError, analyze_reference,
                                    format_reference_report, load_reference_audio)
            try:
                samples, sr = load_reference_audio(args.audio)
                rep = analyze_reference(samples, sr)
            except ReferenceError as e:
                raise CliError(str(e))
            print(format_reference_report(rep, os.path.basename(args.audio)))
        elif args.cmd == "compare":
            from .metrics import format_table, load_audio, summary
            from .reference import ReferenceError
            rows = []
            for path in args.files:
                try:
                    audio, sr = load_audio(path)
                except ReferenceError as e:
                    raise CliError(str(e))
                rows.append((os.path.basename(path), summary(audio, sr)))
            print(format_table(rows))
        elif args.cmd == "samples":
            if getattr(args, "samples_cmd", None) == "install":
                from . import install as _install
                try:
                    names = _install.install(args.group)
                except _install.InstallError as e:
                    raise CliError(str(e))
                print(f"hotovo: {', '.join(names)}")
            elif getattr(args, "samples_cmd", None) == "breaths":
                from . import breath
                from .library import samples_root
                if args.rebuild:
                    full = os.path.join(samples_root(), "vocalset", "FULL")
                    if not os.path.isdir(full):
                        raise CliError(f"nahrávky VocalSet nejsou rozbalené ({full}) — "
                                       f"python -m paprika samples install voices")
                    breath.build(samples_root(), reuse=False)
                try:
                    print("\n".join(breath.stats_lines()))
                    if args.examples:
                        paths = breath.write_examples(args.examples)
                        print(f"-> {len(paths)} WAV v {args.examples} "
                              f"(*_real_* = nahrávka, *_synth_* = syntéza z modelu)")
                except SamplesMissing as e:
                    raise CliError(str(e))
            elif getattr(args, "samples_cmd", None) == "human":
                from . import corpus
                cmd = getattr(args, "human_cmd", None)
                if cmd == "request":
                    try:
                        req = corpus.request(args.kinds, args.backend, args.count, args.seed,
                                             args.duration, args.model)
                    except ValueError as e:
                        raise CliError(str(e))
                    with open(args.out, "w", encoding="utf-8") as f:
                        json.dump(req, f, ensure_ascii=False, indent=1)
                    print(f"{len(req['jobs'])} generací -> {args.out}\n"
                          f"dál: python tools/human_worker.py {args.out} --out <složka> "
                          f"[--classify --asr]; python -m paprika samples human import "
                          f"<složka>")
                elif cmd == "import":
                    if not os.path.isdir(args.folder):
                        raise CliError(f"složka {args.folder} neexistuje")
                    if args.kind and args.kind not in corpus.KINDS:
                        raise CliError(f"neznámý druh {args.kind!r}; k dispozici: "
                                       f"{', '.join(corpus.KINDS)}")
                    res = corpus.ingest(args.folder, kind=args.kind, license=args.license,
                                        attribution=args.author)
                    rej = sorted(res["rejected"].items())
                    for fn, why in rej[:20]:
                        print(f"  odmítnuto {fn}: {why}")
                    if len(rej) > 20:
                        print(f"  ... a {len(rej) - 20} dalších")
                else:
                    print("\n".join(corpus.status_lines()))
            elif getattr(args, "samples_cmd", None) == "steps":
                from . import stepfit
                cmd = getattr(args, "steps_cmd", None)
                try:
                    if cmd == "fit":
                        stepfit.check_name(args.name)
                        segs, sr, noises = stepfit.from_files(args.files)
                        print(f"{len(segs)} kroků z {len(args.files)} nahrávek, fit od "
                              f"{args.base} ...")
                        rec = stepfit.fit(segs, sr, base=args.base, shoe=args.shoe,
                                          noise=noises)
                        path = stepfit.save(args.name, rec, args.files, args.license,
                                            args.author)
                        print("\n".join(stepfit.fit_lines(args.name, rec)))
                        print(f"-> {path}\n   použití: steps(..., surface=\"{args.name}\")")
                    elif cmd == "compare":
                        segs, sr, noises = stepfit.from_files(args.files)
                        print("\n".join(stepfit.compare_lines(segs, sr, args.surface,
                                                              args.shoe, noises)))
                    elif cmd == "cut":
                        paths = stepfit.write_cut(args.file, args.out)
                        print(f"-> {len(paths)} kroků v {args.out}")
                    else:
                        print("\n".join(stepfit.status_lines()))
                except (stepfit.FitError, ValueError, OSError) as e:
                    raise CliError(str(e))
            else:
                from .library import status_lines
                print("\n".join(status_lines()))
        elif args.cmd == "presets":
            from .drums import DRUM_CHARS
            from .kits import kit_names
            from .presets import presets_by_tag, tag_names
            from .sfx import SFX_NAMES
            tags = [args.tag] if args.tag else tag_names()
            try:
                for tag in tags:
                    print(f"{tag + ':':<12}" + ", ".join(presets_by_tag(tag)))
            except KeyError as e:
                raise CliError(str(e).strip("'\""))
            if not args.tag:
                print("")
                print("bicí sady:  " + ", ".join(kit_names()))
                print("bicí znaky: " + ", ".join(f"{k} = {v}"
                                                 for k, v in DRUM_CHARS.items()))
                print("sfx:        " + ", ".join(SFX_NAMES))
                from .library import INSTRUMENTS, is_installed
                print("sampled:    " + ", ".join(
                    n + ("" if is_installed(n) else "*") for n in INSTRUMENTS)
                    + "   (* = nenainstalováno: python -m paprika samples)")
    except CliError as e:
        print(f"chyba: {e}", file=sys.stderr)
        return 2
    except SamplesMissing as e:          # chybějící vzorky nebo korpus: postup, ne traceback
        print(f"chyba: {e}", file=sys.stderr)
        return 2
    return 0
