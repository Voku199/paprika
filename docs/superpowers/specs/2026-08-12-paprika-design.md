# Paprika — design

**Datum:** 2026-08-12
**Stav:** návrh k odsouhlasení
**Verze:** v1

## Co to je

Paprika je konzolová Python knihovna, která umožňuje AI napsat a vyrenderovat
písničku bez DAW, bez VST pluginů a bez samplů. Uživatel zadá prompt
("chci romantickou skladbu na piano"), AI napíše soubor `songs/*.py`, spustí
`python -m paprika render` a dostane hotový `.wav` plus textovou analýzu, podle
které pozná, co se povedlo a co ne.

Paprika není generátor hudby. Melodii, harmonii a strukturu skládá AI. Paprika
poskytuje nástroje: hudební teorii, syntézu zvuku, efekty, mix a zpětnou vazbu.

## Rozhodnutí a jejich důvody

| Rozhodnutí | Volba | Proč |
|---|---|---|
| Výstup | WAV z vlastní syntézy | Soběstačné, nulové externí závislosti, žádné 150MB soundfonty |
| Rozhraní | Python API + CLI | Plná síla Pythonu (cykly, funkce, generátory), levné na tokeny |
| Hudební teorie | Střední — teorie ano, kompozice ne | Knihovna zná akordy a stupnice; melodii píše AI, takže každá skladba zní jinak |
| Zvukový engine | Presety + skládání vlastního patche | 8 presetů pokryje většinu, `Patch(...)` pokryje zbytek |
| Zpětná vazba | Analýza + ASCII náhled | AI zvuk neslyší; bez toho nelze iterovat bez člověka ve smyčce |
| Rozsah v1 | Plné jádro, MCP později | MCP obal má smysl až nad ověřeným jádrem |

**Vědomé omezení:** čistá syntéza zní synteticky. Synthwave, lo-fi, chiptune,
ambient a elektronické piano sednou. Koncertní křídlo nebo živé smyčce ne.

## Závislosti

`numpy` — a nic dalšího.

WAV zápis přes stdlib `wave`. Filtry, reverb, oscilátory a limiter jsou napsané
ručně. Instalace je `pip install numpy`.

## Struktura projektu

```
paprika/
  paprika/
    __init__.py     veřejné API (Song, Patch, Sine, Reverb, chord, scale, ...)
    song.py         Song, Track, Clip, timeline, převod beat -> sample
    note.py         Note, parsing "C#4:1.5" a "E5:1 G5:1 A5:2"
    theory.py       scale(), chord(), arp(), inverze, humanize()
    patch.py        Patch, oscilátory, ADSR, LP/HP filtr
    presets.py      8 hotových nástrojů
    drums.py        syntetizované bicí + pattern stringy
    fx.py           Reverb, Delay, LowPass, Saturate, Sidechain
    render.py       stopy -> stereo buffer -> WAV
    analyze.py      mix report + hudební warningy
    viz.py          ASCII piano roll + waveform
    cli.py          python -m paprika render|show|play
  songs/            skladby napsané AI
  output/           vyrenderované .wav
  tests/            unittest
  requirements.txt  numpy
  README.md
```

Hranice mezi moduly: `render.py` neví nic o hudební teorii, `theory.py` neví nic
o zvuku, `analyze.py` čte jen hotový buffer a datový model. Každý modul jde
testovat samostatně.

## Datový model

```
Song   { bpm, key, scale, bars, beats_per_bar=4, sample_rate=44100, tracks[] }
Track  { patch, clips[], fx[], vol, pan, chromatic }
Clip   { start_beat, notes[] }
Note   { pitch (MIDI 0-127), start_beat, dur_beats, velocity (0-127) }
```

**Klíčové rozhodnutí:** noty žijí v beatech, ne v sekundách. Převod na samply se
děje až v `render()`. Změna BPM je jeden parametr; nic se nepřepisuje.

`chromatic=True` na stopě vypne warning o notách mimo stupnici (pro záměrné
chromatické průchody).

## Datový tok

```
songs/romantic.py
        |
        v
Song -> Track -> Clip -> Note
        |
        | render()
        v
per Note:   Patch.voice(freq, dur, vel) -> np.float32 mono
per Track:  součet hlasů -> fx řetěz -> vol/pan
mix:        součet stop -> stereo -> limiter
        |
        +--> output/romantic.wav (16-bit PCM)
        +--> analyze.report()  -> konzole
        +--> viz.piano_roll()  -> konzole
```

---

## Část 1 — veřejné API

Takhle vypadá skladba, kterou napíše AI:

```python
from paprika import Song, Patch, Sine, ADSR, LP, Reverb, Delay

s = Song(bpm=72, key="C", scale="major", bars=32)

piano = s.track("epiano", vol=0.8)
piano.fx(Reverb(room=0.75, mix=0.35), Delay(1/8, fb=0.25, mix=0.15))

pads = s.track("warmpad", vol=0.4, pan=-0.2)
bass = s.track("subbass", vol=0.7)
dr   = s.drums(vol=0.55)

prog = ["Cmaj7", "Am7", "Fmaj7", "G7"]

# intro — arpeggia
for bar, ch in enumerate(prog * 2):
    piano.arp(ch, bar=bar, pattern="up", rate=1/8, octaves=2)
    pads.chord(ch, bar=bar, dur=4, inv=1)

# verse — melodii píše AI ručně
piano.melody("E5:1 G5:1 A5:2 | G5:1 E5:1 D5:2 | C5:4", bar=8)
bass.notes("C2:4 | A1:4 | F1:4 | G1:4", bar=8)
dr.pattern("k...s...k...s..h", bars=(16, 32))

piano.humanize(timing=0.015, vel=0.12)

s.render("output/romantic.wav")
```

### Notace not

`"C#4:1.5"` = nota, oktáva, dvojtečka, délka v beatech. Svislítko `|` odděluje
takty a slouží jako kontrola — pokud délky v taktu nesedí na `Song.beats_per_bar`,
Paprika vyhodí `NoteLengthError` s číslem taktu. Podtržítko `_` je pauza,
`~` prodlouží předchozí notu. Akord se píše závorkou: `(C4 E4 G4):2`.

### Metody stopy

| Metoda | Co dělá |
|---|---|
| `.note(pitch, at, dur, vel)` | jedna nota, `at` v beatech od začátku skladby |
| `.notes(str, bar=)` | řetězec not od daného taktu |
| `.melody(str, bar=)` | totéž, jen čitelnější jméno pro melodické linky |
| `.chord(name, bar, dur, inv=0, oct=0)` | akord podle jména |
| `.arp(name, bar, pattern, rate, octaves)` | arpeggio; pattern `up`/`down`/`updown`/`random` |
| `.fx(*efekty)` | připojí efektový řetěz |
| `.humanize(timing, vel)` | rozhýbe časování a dynamiku |
| `.clear(bars=)` | smaže rozsah taktů |

`s.drums()` vrací speciální stopu s `.pattern(str, bars=(od, do))`.

### Presety

`epiano`, `warmpad`, `subbass`, `pluck`, `lead`, `strings`, `bell`, `organ`.

Bicí: `k` kick, `s` snare, `h` hi-hat, `c` clap, `t` tom. Všechny syntetizované —
kick je sinus s pitch-envelope, snare je noise plus tón, hat je filtrovaný noise
s krátkým decay.

**Rozlišení pattern stringu:** jeden znak = jedna šestnáctina. Tečka je pauza.
Šestnáct znaků tedy tvoří jeden takt ve 4/4. Delší string se rozdělí na víc taktů,
kratší se dopočítá pauzami do konce taktu. Velké písmeno znamená akcent (vyšší
velocity), takže `"k...s...k...s..K"` má poslední kopák hlasitější. Rozlišení jde
změnit parametrem: `.pattern(str, bars=(0, 8), rate=1/8)`.

### Vlastní patch

```python
moje = s.track(Patch(
    osc=[Sine(1.0), Sine(2.01, amp=0.35), Sine(7.0, amp=0.08)],
    adsr=ADSR(0.004, 0.35, 0.4, 1.4),
    filter=LP(2200, env=0.6)))
```

Oscilátory: `Sine`, `Saw`, `Square`, `Tri`, `Noise`. První argument je poměr
k základní frekvenci noty, takže `Sine(2.01)` je mírně rozladěná oktáva —
základní trik pro živější zvuk.

---

## Část 2 — DSP engine

### Syntéza hlasu

Jedna nota = jeden `voice` buffer. Postup:

1. Frekvence z MIDI pitch: `440 * 2**((pitch - 69) / 12)`
2. Každý oscilátor generuje vektorizovaně přes `np.arange` — žádný Python cyklus
   po samplech
3. Součet oscilátorů vážený jejich `amp`
4. ADSR envelope, vygenerovaná jako pole stejné délky
5. Filtr, pokud je v patchi. Envelope na cutoff (`env=`) posune cutoff nahoru
   v attacku a zpět dolů
6. Násobení velocity, převedenou nelineárně (`(vel/127) ** 1.6` — zní přirozeněji
   než lineární)
7. **Povinný 2 ms fade-in a fade-out** na každý hlas. Bez toho lupou hrany.

Délka bufferu je délka noty plus release ADSR — nota tedy přesahuje svůj
`dur_beats`, což je správně a odpovídá tomu, jak zní skutečný nástroj.

### Filtry

Jednopólový IIR lowpass a highpass (`y[n] = y[n-1] + a * (x[n] - y[n-1])`),
aplikované ve dvou průchodech pro 12 dB/okt. Implementace přes `scipy.signal.lfilter`
není možná (nechceme scipy), takže cyklus po samplech v numpy — u filtru se tomu
nevyhneme. Pro rozumné délky je to přijatelné; pokud se ukáže jako úzké hrdlo,
řešením je `np.convolve` s FIR aproximací.

### Reverb

**Freeverb / Schroeder** — 8 paralelních comb filtrů s různým zpožděním, pak
4 sériové allpass filtry. Žádná konvoluce, žádný impulse response soubor. Rychlé,
znělé, a `room` mapuje na feedback comb filtrů.

Reverb prodlužuje stopu o svůj tail; `render()` proto alokuje buffer s rezervou
a na konci ořízne podle nejdelšího tailu.

### Ostatní efekty

- `Delay(time, fb, mix)` — `time` v beatech (tedy tempo-synchronní), feedback smyčka
- `Saturate(drive)` — `np.tanh(x * drive) / np.tanh(drive)`
- `LowPass(cutoff)` / `HighPass(cutoff)` — stejné filtry jako v patchi, na úrovni stopy
- `Sidechain(source_track, amount)` — envelope follower z jiné stopy (typicky bicí)
  násobí tuto stopu; klasické "pumpování"

### Mix a limiter

Stopy se sečtou s vahou `vol`. Pan je equal-power (`cos`/`sin` křivka), ne lineární.
Na konci soft limiter: `tanh` nad prahem −1 dBFS, takže se nikdy neclipuje tvrdě.
Zápis do 16-bit PCM přes stdlib `wave`.

---

## Část 3 — zpětná vazba

### Report po renderu

```
PAPRIKA  romantic.wav
render   1:47  44100 Hz  stereo  4 stopy  312 not

MIX      peak -1.1 dBFS   rms -17.4 dB   no clipping
BANDS    low 31%  mid 54%  high 15%      [ok]
TRACKS   piano -14 dB | pads -21 dB | bass -16 dB | drums -18 dB

WARN  piano  bar 12   D#5 mimo C major
WARN  bass + piano   maskování 60-140 Hz, bars 5-8
WARN  pads   všechny velocity 100 — zní roboticky, zkus humanize()
INFO  pads   bars 20-24 ticho
```

Pravidla warningů:

| Pravidlo | Podmínka |
|---|---|
| nota mimo stupnici | pitch není ve `Song.scale`, a stopa nemá `chromatic=True` |
| clipping | jakýkoliv sample před limiterem přesáhne 1.0 |
| špatná hlasitost | peak nad −0.3 dBFS nebo RMS pod −30 dB |
| frekvenční maskování | dvě stopy mají obě přes 40 % energie ve stejném pásmu ve stejném čase |
| robotická dynamika | směrodatná odchylka velocity ve stopě je nula |
| ostrá disonance | malá sekunda mezi současně znějícími notami pod MIDI 48 |
| díra v aranži | stopa mlčí 4+ takty uprostřed skladby |

Pásmová analýza počítá energii přes `np.fft.rfft` v oknech po jednom taktu,
sečtenou do tří pásem: pod 250 Hz, 250 Hz — 4 kHz, nad 4 kHz.

### ASCII náhled

```
$ python -m paprika show songs/romantic.py --bars 0:4
       |1       |2       |3       |4       |
piano  |C4-E4-G4|A3-C4-E4|F3-A3-C4|G3-B3-D4|
pads   |C3======|A2======|F2======|G2======|
bass   |C1------|A1------|F1------|G1------|
drums  |k..hs..h|k..hs..h|k..hs..h|k..hs.sK|
```

Šířka sloupce v `show` se přizpůsobí — výchozí je 8 znaků na takt (osminky), s
`--res 16` se zobrazí šestnáctiny. Nejde tedy o stejné rozlišení jako u pattern
stringu, který je vždy v šestnáctinách, pokud se neurčí `rate`.

`show` nerenderuje audio — jen naparsuje skladbu a vykreslí timeline. Rychlé,
použitelné pro kontrolu struktury před renderem.

`render --waveform` navíc vykreslí obálku hotového mixu, aby byla vidět dynamika
skladby (kde je tiché intro, kde nastupuje refrén).

## CLI

```bash
python -m paprika render songs/romantic.py              # wav + report
python -m paprika render songs/romantic.py --bars 0:8   # rychlý preview
python -m paprika render songs/romantic.py --waveform
python -m paprika show   songs/romantic.py --bars 0:4   # jen ASCII, bez renderu
python -m paprika play   output/romantic.wav            # otevře systémovým přehrávačem
```

Skladba je normální Python modul. `render` ho naimportuje a najde na úrovni modulu
právě jednu instanci `Song` — tu vyrenderuje. Když modul žádnou nemá nebo jich má
víc, CLI skončí chybou a vypíše nalezená jména. Volání `s.render(...)` přímo
v souboru je tedy volitelné; CLI funguje i bez něj a výstupní cestu si odvodí
z názvu skladby (`songs/romantic.py` -> `output/romantic.wav`).

## Testy

`unittest` v `tests/`, spouštěné z kořene projektu (`python -m pytest tests/`
i `python -m unittest discover tests`).

| Soubor | Co ověřuje |
|---|---|
| `test_theory.py` | `chord("Cmaj7") == [60, 64, 67, 71]`, inverze, stupnice, arp patterny |
| `test_note.py` | parsing `"C#4:1.5"` -> pitch 61 / dur 1.5; pauzy, prodloužení, akordy v závorce; `NoteLengthError` při špatné délce taktu |
| `test_timing.py` | nota na bar=4, beat=2 při 120 BPM začíná na očekávaném samplu |
| `test_patch.py` | ADSR má správnou délku a sustain hodnotu; žádné NaN; fade-in/out na krajích |
| `test_fx.py` | reverb neprodlouží signál víc než o svůj tail; delay s fb<1 konverguje k nule; saturate zůstane v rozsahu |
| `test_render.py` | výstup nemá NaN ani inf, peak <= 1.0, délka odpovídá počtu taktů a BPM |
| `test_analyze.py` | podstrčená skladba s notou mimo stupnici vygeneruje odpovídající warning; čistá skladba nevygeneruje žádný |

## Rizika

**Rychlost renderu.** Vektorizovaná generace hlasů je rychlá, ale IIR filtry
vyžadují cyklus po samplech. Mitigace: filtr je volitelný a aplikuje se na hlas,
ne na každý sample mixu. Pokud dvouminutová skladba trvá přes ~30 s, přejdeme
u filtrů na FIR aproximaci přes `np.convolve`.

**Kvalita zvuku presetů.** Jestli `epiano` nezní dobře, celý projekt selže bez
ohledu na architekturu. Mitigace: `epiano` staví na FM (nosič plus modulátor
s rychlým decay), což je ověřený způsob, jak dostat Rhodes-like zvuk ze dvou
sinusoid. Preset ladíme jako první a posloucháme, než se postaví zbytek.

**Lupání na hranách not.** Řešeno povinným 2 ms fade na každém hlasu — není to
volitelné nastavení, je to součást generace hlasu.

## Co v1 neobsahuje

MCP server, MP3 export, automatizační křivky parametrů v čase, změny tempa
uprostřed skladby, import/export MIDI, stereo šířka a chorus efekty.

MCP server je v2: tenký obal nad hotovým jádrem, který drží `Song` v paměti mezi
voláními a vystaví `paprika_new_project`, `paprika_add_track`, `paprika_write_notes`,
`paprika_render`. Architektura je na to připravená tím, že `Song` je čistý datový
objekt bez vazby na CLI.
