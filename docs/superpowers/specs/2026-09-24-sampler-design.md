# Paprika — sampler skutečných nástrojů (podprojekt 4, posunutý dopředu)

**Datum:** 2026-09-24
**Stav:** implementováno 2026-09-24 (schváleno: „Ano, souhlasím … stáhni to a provozni to“)
**Navazuje na:** `2026-09-23-engine-v2-design.md`, `2026-09-23-realism-pass-design.md`

## Proč

Po realism passu uživatel pořád slyší „bitovou hudbu“ a chce skutečné
nástroje: koncertní piano jako v divadle, housle, další orchestr. Syntéza
(i s ensemble, exp ADSR, konvolucí) akustické nástroje věrohodně nenapodobí —
to by bylo fyzikální modelování na úrovni Pianoteq/SWAM. Jediná cesta ke
„skutečnému pianu“ jsou nahrávky skutečného piana. Původní plán
(hybrid: syntéza + volitelný SFZ sampler) to už obsahoval jako podprojekt 4;
posouvá se před style packy a export.

## Rozhodnutí

| Rozhodnutí | Volba | Proč |
|---|---|---|
| Zdroj zvuku | Sampler přehrávající SFZ + WAV | Standardní formát, i vlastní knihovny uživatele |
| Piano | Salamander Grand Piano V3, 44.1 kHz/16 bit WAV (archive.org, 489 MB) | Yamaha C5 křídlo, 16 dynamik; CC-BY 3.0 |
| Orchestr | VSCO 2 CE (GitHub, commit `440300901dfe`), jen vybrané složky (~1.6 GB) | CC0, WAV 44.1 kHz, smyčce/dechy/žestě/mallets/pianino |
| Závislosti | žádné nové (stdlib `urllib`, `tarfile`; `scipy.io.wavfile`) | Knihovna zůstává numpy + scipy |
| Umístění vzorků | `%LOCALAPPDATA%\paprika\samples` (jinde `~/.paprika/samples`), přepíše env `PAPRIKA_SAMPLES` | Ne do repa na OneDrive |
| Chybějící knihovna | Přesná chyba s instalačním příkazem, žádný tichý fallback | AI neslyší — tichá náhrada syntézou by ji oklamala |
| API | `s.track("grand_piano")`, `s.track("violins.pizz")` = nástroj.artikulace | Stejný vstup jako presety |
| Mapování VSCO | SFZ se generuje při instalaci z jmen souborů | Repo VSCO nemá SFZ; runtime má jen jednu cestu (SFZ) |
| Licence | `CREDITS.txt` ve složce vzorků + sekce v README | CC-BY vyžaduje kredit „Alexander Holm“ ve hře |

Mimo rozsah (další krok): orchestrální bicí (tympány, činely, gong) jako
samplovaná bicí sada, sbor, release samply kladívek a rezonance strun
Salamanderu, WAV `smpl` smyčky.

## Moduly

```
paprika/sfz.py       text SFZ -> [Region]           (žádné audio)
paprika/sampler.py   Sampler: regiony + vzorky -> hlas noty
paprika/library.py   katalog nástrojů, cesty, instrument(name) -> Sampler
paprika/install.py   stažení, rozbalení, generování SFZ, manifest
```

Hranice: `sfz.py` zná jen text; `sampler.py` zná audio a regiony, ne noty
skladby; `render.py` rozvrhuje noty (pedál, legato) a volá sampler;
`library.py` zná jména a cesty; síť je jen v `install.py`.

### `sfz.py`

`parse(text, base_dir) -> list[Region]`, `load(path) -> list[Region]`.

- Hlavičky `<control>`, `<global>`, `<master>`, `<group>`, `<region>`;
  dědění global → master → group → region.
- `#define $X hodnota` (substituce), komentáře `//` a `/* */`.
- `sample=` smí obsahovat mezery (hodnota končí před dalším `opcode=`
  nebo koncem řádku); `\` i `/`; `default_path` z `<control>`.
- Klávesy číslem i jménem (`c4` = 60, `f#3`, `eb2`).
- Podporované opcody: `sample, key, lokey, hikey, pitch_keycenter, lovel,
  hivel, seq_length, seq_position, tune, transpose, volume, pan,
  amp_veltrack, ampeg_attack, ampeg_release, offset, loop_mode, loop_start,
  loop_end, trigger`. Ostatní se ignorují (seznam v `Region.ignored` pro
  ladění, do reportu ne).
- `Region` je dataclass s výchozími hodnotami SFZ (lokey 0, hikey 127,
  lovel 1, hivel 127, keycenter = key nebo 60, veltrack 100, release 0.001 s,
  loop_mode `no_loop`, trigger `attack`).

### `sampler.py`

```python
Sampler(sfz_path, name=None, legato=False, undamped_from=None, gain_db=0.0)
```

- Načte regiony (jen `trigger` attack/first; release/legato regiony
  ignoruje). WAV čte `scipy.io.wavfile.read(mmap=True)` líně při prvním
  použití regionu, fallback bez mmap (24 bit). Mono vzorek → stereo.
- `select(pitch, vel, rr_index) -> Region`: kandidáti s klávesou a velocity
  v rozsahu; když žádný nesedí klávesou, vezme region s nejbližším
  keycenter (a stejnou velocity vrstvou). Víc kandidátů (round robin) →
  seřadit podle `seq_position`, vybrat `rr_index % počet`.
- `range -> (lo, hi)`: nejnižší `lokey` a nejvyšší `hikey` regionů.
- `voice(region, pitch, dur, sr, legato_in=False, legato_out=False)
  -> (n, 2) float32`:
  - poměr `2^((pitch - keycenter + transpose)/12 + tune/1200) * sr_vzorku/sr`
    (převzorkování na sr skladby je součástí posunu výšky);
  - čtení 4bodovou Hermitovou interpolací (vektorově, bez smyček);
  - `one_shot` (pizz, stac, harfa, mallets): přehraje vzorek celý, gate
    ignoruje;
  - jinak gate = `dur`, pak release `ampeg_release` (exponenciálně);
    u piana noty ≥ `undamped_from` (bez dusítek) doznívají s release 2 s;
  - region se smyčkou (`loop_continuous`/`loop_sustain` + `loop_start/end`):
    nota delší než vzorek pokračuje smyčkou s crossfadem min(25 % smyčky,
    0.5 s), rovný výkon; bez smyčky vzorek prostě dozní (piano);
  - `legato_in`: začne `LEGATO_OFFSET` = 0.08 s ve vzorku (přeskočí nasazení
    smyčce/jazyku) s fade-in `LEGATO_XFADE` = 0.08 s; `legato_out`:
    release = `LEGATO_XFADE`;
  - 2 ms fade na krajích (proti lupnutí).
- `gain(region, vel)`: SFZ zákon — `dB = volume + gain_db +
  veltrack/100 · 40·log10(vel/127)`; `pan` regionu.
- `key()` pro cache hlasů: `("sampler", abspath, mtime sfz, legato,
  undamped_from, gain_db)`. Hlas se cachuje podle
  `(key, id regionu, pitch, dur, legato_in, legato_out, sr)`, velocity
  se aplikuje až jako zisk (cache funguje i s humanize).
- `tail`: max release + (u one_shot) délka nejdelšího vzorku, strop 8 s.
- `sampled = True` — pro analýzu (nahraný nástroj není „statický zvuk“).

### `library.py`

- `samples_root()`: env `PAPRIKA_SAMPLES`, jinak `%LOCALAPPDATA%\paprika\samples`,
  jinak `~/.paprika/samples`.
- Katalog `INSTRUMENTS`: jméno → `Instrument(group, arts, default_art,
  mono, legato_arts, undamped_from, tags, fx)`; `arts` mapuje artikulaci na
  relativní cestu SFZ. Nástroje:

| nástroj | skupina | artikulace (první = výchozí) |
|---|---|---|
| `grand_piano` | piano | sus |
| `upright_piano` | piano | sus |
| `violins`, `violas`, `celli` | strings | sus, trem, pizz, spic |
| `violin_solo` | strings | sus, trem, pizz, spic |
| `contrabass` | strings | sus, nv, trem, pizz, spic |
| `concert_harp` | strings | sus |
| `flute` | winds | sus, nv, exp, stac |
| `oboe` | winds | sus, nv, stac |
| `clarinet` | winds | sus, stac |
| `bassoon` | winds | sus, vib, stac |
| `horn` | brass | sus, stac, mute |
| `trumpet` | brass | sus, vib, stac, mute |
| `trombone` | brass | sus, vib, stac |
| `tuba` | brass | sus, stac |
| `marimba`, `glockenspiel`, `xylophone` | mallets | sus |

- Sólové nástroje (`violin_solo`, `contrabass`, dechy, žestě) mají
  výchozí `mono=True` → legato; sekce a klávesy polyfonní.
- `instrument(name) -> Sampler`: `"violins.pizz"`, bez artikulace výchozí.
  Neznámá artikulace → `KeyError` se seznamem. Nenainstalovaná skupina →
  `SamplesMissing` (podtřída `RuntimeError`) s textem
  `nástroj 'violins' potřebuje vzorky skupiny 'strings' — nainstaluj:
  python -m paprika samples install strings (cíl: <cesta>)`.
- `is_sampled(name)`, `instrument_names()`, `instrument_fx(name)` (piano →
  `Hall("hall", mix=0.22)`, sekce → `Hall("hall", mix=0.3)`, sóla →
  `Hall("hall", mix=0.25)`, mallets → `Hall("hall", mix=0.2)`),
  `installed_groups()`.
- Jméno stopy: tečka → podtržítko (`violins_pizz`).

### `install.py`

- `install(group, root=None, log=print)`; `group` ∈ piano, strings, winds,
  brass, mallets, all.
- **Salamander**: stáhne tar.bz2 do `<root>/_downloads/` (pokračuje,
  když soubor existuje a má správnou velikost), rozbalí `tarfile` do
  `<root>/salamander/` (jen členy uvnitř cíle — ochrana proti `../`),
  SFZ je `SalamanderGrandPianoV3Retuned.sfz` z archivu (doladěný do
  rovnoměrného ladění, ladí s orchestrem).
- **VSCO**: jeden dotaz na GitHub trees API (připnutý commit) → seznam
  souborů ve složkách skupiny → stažení přes `raw.githubusercontent.com`
  (8 vláken, `.part` + přejmenování, existující soubor se správnou
  velikostí se přeskočí) do `<root>/vsco2/<cesta z repa>`.
- **Generování SFZ** z jmen souborů (`<root>/vsco2/sfz/<nástroj>.<art>.sfz`):
  - tokeny oddělené `_`: nota `^[A-G][#b]?-?\d$`, vrstva `v\d+` nebo
    dynamika (`ppp pp p mp mf f ff fff`, `soft medium loud`), round robin
    `rr\d+`/`RR\d+` nebo koncové číslo (`_1`, `_01`);
  - na vrstvu: kořeny seřazené, hranice kláves v půli mezi sousedními
    kořeny, krajní rozšířené o 2 půltóny;
  - vrstvy rozdělí 1–127 rovnoměrně; `amp_veltrack` 100 pro jednu vrstvu,
    40 pro víc (nahrávky už nesou dynamiku);
  - oktáva: VSCO pojmenovává většinu složek s C3 = střední C, takže
    výška z jména + 12; výjimky (C4 = 60) jsou `Keys/Upright Nr1`,
    `Strings/Harp`, `Strings/Solo Violin/Arco Vib`, `Strings/Solo Violin/Pizz`
    (změřeno detekcí výšky všech složek 2026-09-24);
  - `loop_mode=one_shot` pro pizz, spic, stac, harfu, mallets;
  - výdrže (smyčce, dechy, žestě kromě one_shot): `loop_mode=loop_continuous`,
    `loop_start` = 45 %, `loop_end` = 90 % délky vzorku;
    `ampeg_release` 0.3 s (smyčce, dechy), 0.25 s žestě, 0.4 s pianino;
  - normalizace: `<global> volume` tak, aby medián max RMS (100 ms okna)
    nejhlasitější vrstvy přes všechny kořeny byl −18 dBFS a žádný vzorek
    té vrstvy nepřesáhl špičku 0.9 (jeden prostřední kořen se na
    skutečných vzorcích ukázal nespolehlivý — staccato žestí až 2.8).
    Salamander má pevný `gain_db` −4.6 dB (C4v16 na −18 dBFS).
- Manifest `<root>/paprika_samples.json` (skupiny, zdroj, commit, datum) a
  `<root>/CREDITS.txt` s licencemi a povinným kreditem.
- Chyby sítě → `InstallError` s URL a radou („spusť znovu, pokračuje“).

## Render

- `render_track`: větev `isinstance(track.patch, Sampler)` →
  `_render_sampled`: noty → události `(start, dur, vel, pitch, rr, legato_in,
  legato_out)`:
  - **pedál** (`Track.pedal`): nota, která končí při sešlápnutém pedálu,
    zní do jeho uvolnění; znovu udeřená stejná výška utne předchozí hlas
    na novém úderu (dál doznívá svým release);
  - **legato**: mono stopa + artikulace v `legato_arts` + nota začíná
    nejpozději `LEGATO_GAP` po konci předchozí → předchozí `legato_out`
    (končí v `start + LEGATO_XFADE`), nová `legato_in`; `>` v notaci
    u sampleru znamená totéž;
  - rr index = pořadí noty ve stopě.
- Pedál funguje i u syntetizovaných polyfonních stop (prodlouží noty
  v `_phrases`).
- `_tail_seconds`: u Sampleru `sampler.tail`.
- `Song.track()` přijme jméno samplovaného nástroje (před presety) nebo
  instanci `Sampler`; `fx="default"` → `instrument_fx`.

### `Track.pedal`

```python
t.pedal(bars=(0, 8), every=1)   # sešlápnuto 0–8, přešlápnuto každý takt
t.pedal(bars=(8, 10))           # jedno dlouhé sešlápnutí
```

Uloženo jako `(down_beat, up_beat)`; `every` rozdělí rozsah a mezi úseky
nechá mezeru 1/16 beatu (zvednutí nohy). Vrací `self`.

## Zpětná vazba

- `_moving_patch`: `sampled` → pohyblivý (žádné „statický zvuk“).
- Nové pravidlo `mimo rozsah` (WARN, max 3): nota samplované stopy mimo
  `Sampler.range` → takt, nota, rozsah nástroje.
- CLI `python -m paprika samples` — cesta, skupiny (nainstalováno / chybí,
  velikost), nástroje a artikulace; `samples install <skupina>`.
- `presets` vypíše i řádek `sampled:` se stavem instalace.

## Ukázka

`songs/divadlo.py`: klidný divadelní kus v d moll — sólo křídlo s pedálem
(úvod), pak celli + violy + violins (sus), sólové housle legato melodie,
harfa, lesní roh; `Hall`, mastering „cinematic“. Ověřit `compare` proti
`songs_mp3/City Ruins` (piano + smyčce): středy ≥ 25 %, sub ≤ 12 %,
šířka ≥ 12 %.

## Testy (offline, bez sítě)

- `test_sfz.py`: dědění hlaviček, `#define`, komentáře, mezery v `sample`,
  jména not, `default_path`, neznámé opcody.
- `test_sampler.py` (fixture: sinusové WAV v kořenech C4/E4/G#4, 2 vrstvy,
  2 rr, 48 kHz vzorky do 44.1 kHz skladby): výběr podle klávesy/velocity/rr,
  frekvence po posunu (FFT, < 0.5 %), převzorkování, délka gate+release,
  one_shot ignoruje gate, smyčka prodlouží dlouhou notu bez skoku,
  legato fade, zákon zisku, stereo, chybějící WAV → chyba s cestou.
- `test_library.py`: rozlišení jmen a artikulací, chyba s instalačním
  příkazem, `PAPRIKA_SAMPLES`, parsování reálných jmen VSCO souborů,
  generování SFZ z falešného seznamu (souvislé klávesy, velocity 1–127),
  `Song.track("violins.pizz")` s falešnou knihovnou vyrenderuje zvuk.
- `test_install.py`: stahování s mockem `urlopen` (přeskočí hotové,
  `.part`), rozbalení malého tar.bz2 (i odmítnutí `../`), manifest, CREDITS.
- Render: pedál prodlouží noty (sampler i syntéza), legato události,
  `tail`, pravidlo `mimo rozsah`, `samples` CLI výpis.
