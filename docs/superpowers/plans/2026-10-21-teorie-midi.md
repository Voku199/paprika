# Fáze 1 — kontrola hudební teorie a MIDI export — implementační plán

> Steps use checkbox (`- [ ]`) syntax for tracking. Spec se podle rozhodnutí
> uživatele nepíše; návrh byl schválen v chatu 2026-10-21 (volby: přísnost
> „podle stylu“, notace plánu „obojí“). Kontext: `docs/superpowers/research/
> 2026-10-21-ai-engine-pruzkum.md`, kap. 5 a 7.

**Goal:** Paprika umí ověřit hudební teorii skladby bez renderu (harmonický
plán, vedení hlasů, nevlastní tóny, kadence) a zapsat skladbu do MIDI, aby
šla otevřít v DAW a později porovnat partituru se zvukem.

**Architecture:** `harmony.py` (jen MIDI čísla, žádný model) parsuje
harmonický plán ze jmen akordů i římských číslic a umí opačný převod.
`Song` drží plán (`harmony()`), sekce (`section()`) a režim (`theory=`).
`rules.py` čte jen datový model a vrací `Warning_`; `analyze.py` je skládá
s audio pravidly do reportu a nabízí `check_report` bez audia. `midi.py`
zapisuje a čte Standard MIDI File ručně přes `struct`. CLI dostane `check`
a `midi`, `show` řádek s akordy.

**Tech Stack:** Python 3, stdlib (`struct`, `re`, `dataclasses`), numpy jen
tam, kde už je; testy `unittest`.

## Global Constraints

- Žádná nová závislost. MIDI zápis i čtení ručně.
- Hranice vrstev: `harmony.py` nezná `Song` ani audio; `rules.py` nezná
  audio ani render; `midi.py` čte jen model; render se nemění (zvuk
  bit-identický).
- Čeština v docstrings, komentářích, chybách a reportu.
- Pravidla omezují výstup (3–4 nálezy na pravidlo), ignorují bicí a SFX
  stopy (`isinstance(t, DrumTrack)`).
- `theory="free"` (výchozí) nesmí na existujících skladbách (`songs/`)
  zaplavit report; WARN jen pro jasné chyby (špatný bas, chromatický cizí
  tón na těžké době, zablácené tercie v hloubce).
- Track/Song metody pro stavbu vrací `self` (kromě `Song.track()` apod.,
  které vrací novou stopu).
- Bez `harmony()` se pravidla, která plán potřebují, tiše přeskočí.

## Review Focus

- Římské číslice v mollu: stupně z přirozeného mollu (`VI` = Bb v d moll),
  `V` = durový akord, `viio`/`viiø7` v mollu staví na citlivém tónu.
- Kolize notací: `/` u jména akordu = bas (`Dm/F`), u římské číslice =
  vedlejší akord (`V/V`); `6` u římské číslice = obrat (sextakord), u jména
  akordu = přidaná sexta (`C6`).
- Humanize posouvá starty o ±0.015 beatu: těžké doby a začátky akordů se
  porovnávají s tolerancí, jinak se pravidla rozbijí na každé lidské skladbě.
- Paralely ve free režimu: běh ≥ 3 paralelních pohybů je záměrné zdvojení,
  nehlásí se.
- MIDI: noty stejné výšky, které se překrývají, se zkrátí (jinak note-off
  první utne druhou); note-off před note-on ve stejném ticku.
- Bicí na kanálu 10 (index 9), melodické stopy kanál 9 přeskočí.

---

### Task 1: `harmony.py` — plán akordů

**Files:** Create `paprika/harmony.py`, `tests/test_harmony.py`

**Interfaces (produces):**
- `HarmonyError(ValueError)`
- `Chord` (frozen dataclass): `root` (pc), `quality` (klíč `theory.CHORDS`),
  `bass` (pc), `label` (jak se píše, např. `"Dm/F"`); property `pcs`
  (frozenset pc), `has_third`, `third_pc`.
- `ChordSpan` (dataclass): `start_beat`, `dur_beats`, `chord`, `source`
  (token, jak ho autor napsal); property `end_beat`.
- `parse_chord(token, key, scale) -> Chord` — jméno (`Dm`, `Bbmaj7`, `C/E`)
  i římská číslice (`i`, `VI`, `bVII`, `V7`, `viio7`, `iiø7`, `IV6`, `I64`,
  `V65`, `V7/V`, `viio7/V`, `Isus4`, `Iadd9`, `I5`).
- `parse_harmony(text, key, scale, beats_per_bar=4, bar_offset=0)
  -> list[ChordSpan]` (starty od 0) — `|` odděluje takty; `Gm:2` explicitní
  délka; tokeny bez délky si dělí zbytek taktu; bez `|` je každý token jeden
  takt; `_` = bez akordu.
- `roman_of(chord, key, scale) -> str` — `i`, `VI`, `V7`, `i6`, `V65`,
  `V/V`, `bVII`, `viio7`, `Iadd6` …
- `key_signature(key, scale) -> (sf, minor)` — pro MIDI a hláskování.
- `pc_name(pc, key, scale) -> str` — béčka v béčkových tóninách.
- `cadence(prev, last, key, scale) -> str | None` — `autentická`,
  `plagální`, `půlová`, `klamná`.

- [ ] Testy: jména (kvality z `CHORDS`, lomený bas, neznámá kvalita →
  chyba se seznamem), římské v C dur / d moll / D dorian, sedmičky diatonicky
  (`I7`=Cmaj7, `V7`=G7, `ii7`=Dm7), obraty (`i6` bas F v d moll, `V65` bas
  C# v d moll), vedlejší (`V/V` v C = D dur, `V7/iv` v d moll = D7),
  `viio7` v d moll = C#dim7, délky v taktu (`Gm:2 A:2`, `Gm A` → po 2,
  chyba se jménem taktu), bez `|`, `_`, `roman_of` ↔ `parse_chord` pro
  diatonické akordy, `key_signature` (C=0, F=-1, d moll=(-1, 1), A dur=3),
  `pc_name` (Bb v F dur, A# v E dur), `cadence` (V→I, IV→I, →V, V→vi).
- [ ] Implementace, testy zelené.

### Task 2: model — `Song.harmony`, `section`, `theory`

**Files:** Modify `paprika/song.py`; tests v `tests/test_rules.py`

**Interfaces (produces):**
- `Song(..., theory="free")` — `"free" | "strict" | "off"`, jinak
  `ValueError` se seznamem.
- `Song.plan: list[ChordSpan]` (absolutní beaty, seřazené),
  `Song.harmony(text, bar=0) -> Song` (nový úsek přepíše překrývající se
  starý), `Song.chord_at(beat) -> ChordSpan | None`.
- `Section(name, start_bar, end_bar)`, `Song.sections`,
  `Song.section(name, bars=(lo, hi)) -> Song`.

- [ ] Testy: posun o `bar`, přepis překrývajícího úseku, `chord_at` na
  hranách, neplatný `theory`, sekce `hi <= lo` → chyba.
- [ ] Implementace.

### Task 3: `rules.py` — pravidla teorie

**Files:** Create `paprika/rules.py`; modify `paprika/analyze.py`;
create `tests/test_rules.py`

**Interfaces (produces):**
- `Warning_` se přesune do `rules.py` (+ volitelný `bar`), `analyze`
  ho re-exportuje (veřejná cesta `paprika.analyze.Warning_` zůstává).
- `theory_warnings(song) -> list[Warning_]`
- `harmony_lines(song, bars=None) -> list[str]` — řádky `HARMONIE` a
  `KADENCE` pro `check`.
- `analyze.model_warnings(song)` — pravidla bez audia (stávající
  `_out_of_scale`, `_robotic`, `_dissonance`, `_gaps`, `_static_sound`,
  `_sample_range` + teorie); `warnings_for` ho používá.
- `analyze.check_report(song, bars=None, name=None) -> str`.

Linky: jednohlasá stopa (`mono=True` nebo bez překryvu not delšího než
0.1 beatu) = jedna linka. `strict` navíc rozdělí vícehlasé stopy na hlasy
podle pořadí výšky v souzvuku.

Pravidla (free / strict):
1. Stupnice podle harmonie (`_out_of_scale`): tón z aktuálního akordu plánu
   není mimo stupnici.
2. `_bass_vs_plan` — nejnižší znějící tón na začátku akordu (nota začala
   nejdřív 0.5 beatu před ním) není tón akordu: WARN / WARN. Prodleva
   (bas drží z dřívějška) se nehlásí.
3. `_coverage` — v úseku akordu nezní základ nebo tercie: INFO / WARN.
4. `_nonchord` — nota linky na těžké době (1. a 3. doba ve 4/4, jinak 1.)
   mimo akord, nerozvedená krokem (≤ 2 půltóny): WARN když je i mimo
   stupnici, jinak INFO / vždy WARN. Průtah (stejná výška před ní) a
   appoggiatura (pokračuje krokem) projdou.
5. `_parallels` — paralelní čisté kvinty a oktávy (i složené) mezi linkami,
   oba hlasy se pohnou stejným směrem: izolované (běh ≤ 2) INFO, běh ≥ 3
   nehlásí / všechny WARN.
6. `_leading_tone` — dominanta → tónika, nejvyšší linka s citlivým tónem
   nejde na tóniku: INFO / WARN.
7. `_leaps` — melodická linka (medián výšky ≥ C4, medián kroku < 4
   půltóny): skok ≥ malá sexta (kromě oktávy) bez pohybu opačným směrem,
   tritónový skok: INFO / WARN.
8. `_low_thirds` — dva tóny ve vzdálenosti tercie, spodní pod G2, znějí
   spolu ≥ 0.5 beatu: WARN / WARN.
9. jen strict: `_spacing` (sousední horní hlasy > oktáva) a `_crossing`
   (linky překříží své pořadí podle mediánu): WARN.

- [ ] Testy pro každé pravidlo (pozitivní i negativní případ), free vs.
  strict, `theory="off"`, omezení výstupu, C# nad A v d moll bez
  `chromatic=True` projde, `check_report` obsahuje HARMONIE/KADENCE.
- [ ] Implementace, celé `tests/test_analyze*.py` zelené.

### Task 4: `midi.py` — export a čtení

**Files:** Create `paprika/midi.py`, `tests/test_midi.py`

**Interfaces (produces):**
- `PPQ = 480`, `GM_PROGRAMS`, `GM_DRUMS`
- `to_bytes(song) -> tuple[bytes, list[str]]` (data, poznámky co se
  nevešlo), `export(song, path) -> list[str]`
- `read(path_or_bytes) -> MidiData` (`ppq`, `fmt`, `tracks`:
  `MidiTrack(name, notes: list[MidiNote], events)`), `MidiNote(start, dur,
  pitch, velocity, channel)`; čtečka zná running status.

Obsah: stopa 0 = jméno, tempo, takt (4/4; půlbeaty → x/8), předznamenání,
markery sekcí, textové události akordů (`"Dm (i)"`); každá `Track` = MIDI
stopa se jménem, program change podle GM mapy, noty, CC64 z `pedals`;
`DrumTrack` na kanálu 9 podle `GM_DRUMS`; `SfxTrack` a automatizace se
nevyexportují (poznámka).

- [ ] Testy: hlavička (`MThd`, formát 1, PPQ), round-trip not (start,
  délka, výška, velocity), tempo a takt, předznamenání d moll, markery a
  akordy, bicí na kanálu 9 se správnými čísly, CC64, překryv stejné výšky,
  deterministické bajty, poznámka o SFX, čtení running status.
- [ ] Implementace.

### Task 5: CLI, `show`, veřejné API

**Files:** Modify `paprika/cli.py`, `paprika/viz.py`, `paprika/__init__.py`;
tests v `tests/test_rules.py` a `tests/test_midi.py`

- [ ] `python -m paprika check songs/x.py [--bars 0:8]` → `check_report`.
- [ ] `python -m paprika midi songs/x.py [-o out.mid]` → `output/<jméno>.mid`.
- [ ] `piano_roll`: když má skladba plán, řádek `chords` nad stopami
  (jméno akordu od jeho začátku, ořízne se na šířku úseku).
- [ ] Exporty: `HarmonyError`, `export_midi`, `read_midi`.
- [ ] Testy CLI přes `main([...])` a dočasný soubor skladby.

### Task 6: ukázka, dokumentace, ověření

- [ ] `songs/divadlo.py`: `s.harmony(...)` z `PROG` + sekce, bez plošného
  `chromatic=True`; `check` přes náhradní presety (vzorky nejsou nutné pro
  ověření pravidel) — žádný falešný WARN.
- [ ] README: harmonický plán, kontrola teorie, MIDI; CLAUDE.md: příkazy,
  moduly, počet testů.
- [ ] `python -m pytest tests/` zelené; commit + push.
