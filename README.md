# Paprika

Konzolová Python knihovna, která umožňuje napsat a vyrenderovat písničku bez
DAW, bez VST pluginů a bez samplů. Napíšeš soubor `songs/*.py`, spustíš
`python -m paprika render` a dostaneš hotový `.wav` plus textovou analýzu,
podle které poznáš, co se povedlo a co ne.

Paprika není generátor hudby. Melodii, harmonii a strukturu skládáš ty (nebo
AI). Paprika dává nástroje: hudební teorii, syntézu zvuku, efekty, mix
a zpětnou vazbu.

**Vědomé omezení:** čistá syntéza zní synteticky. Synthwave, lo-fi, chiptune,
ambient a elektronické piano sednou. Koncertní křídlo nebo živé smyčce ne.

## Instalace

```bash
pip install -r requirements.txt   # numpy, a nic dalšího
```

WAV zápis jede přes stdlib `wave`. Filtry, reverb, oscilátory i limiter jsou
napsané ručně v numpy.

Volitelně, jen pro `analyze-ref` na ne-wav souborech (mp3, m4a, ...):

```bash
pip install imageio-ffmpeg
```

## Rychlý start

```python
from paprika import Song, Reverb, Delay

s = Song(bpm=72, key="C", scale="major", bars=8)

piano = s.track("epiano", vol=0.7)
piano.fx(Reverb(room=0.75, mix=0.35), Delay(1/8, fb=0.25, mix=0.15))
bass = s.track("subbass", vol=0.45)
dr = s.drums(vol=0.7)

for bar, ch in enumerate(["Cmaj7", "Am7", "Fmaj7", "G7"] * 2):
    piano.arp(ch, bar=bar, pattern="up", rate=1/8, octaves=2)

bass.notes("C2:4 | A1:4 | F1:4 | G1:4 | C2:4 | A1:4 | F1:4 | G1:4", bar=0)
dr.pattern("k...s...k...s..h", bars=(4, 8))
piano.humanize(timing=0.015, vel=0.12)

s.render("output/skica.wav")
```

```bash
python -m paprika render songs/romantic.py              # wav + report
python -m paprika render songs/romantic.py --bars 0:8   # rychlý preview
python -m paprika render songs/romantic.py --waveform   # + obálka mixu
python -m paprika show   songs/romantic.py --bars 0:4   # jen ASCII, bez renderu
python -m paprika play   output/romantic.wav            # systémový přehrávač
python -m paprika presets                               # seznam nástrojů
python -m paprika analyze-ref referencni.mp3             # tempo/key/akordy/dynamika
```

Skladba je normální Python modul. `render` ho naimportuje a najde na úrovni
modulu právě jednu instanci `Song` — tu vyrenderuje. Volání `s.render(...)`
přímo v souboru je volitelné; CLI funguje i bez něj a výstupní cestu odvodí
z názvu skladby (`songs/romantic.py` → `output/romantic.wav`).

## Notace not

```
C#4:1.5        nota, oktáva, dvojtečka, délka v beatech
(C4 E4 G4):2   akord — celý naráz
_:2            pauza
~:1            prodlouží předchozí notu
C4:1@110       volitelná velocity
|              oddělovač taktů a zároveň kontrola délek
```

Když součet délek v taktu nesedí na `Song.beats_per_bar`, letí
`NoteLengthError` s číslem taktu — překlep v rytmu se pozná hned, ne až
podle zvuku.

## Metody stopy

| Metoda | Co dělá |
|---|---|
| `.note(pitch, at, dur, vel)` | jedna nota, `at` v beatech od začátku skladby |
| `.notes(str, bar=)` | řetězec not od daného taktu |
| `.melody(str, bar=)` | totéž, čitelnější jméno pro melodické linky |
| `.chord(name, bar, dur, inv=0, oct=0)` | akord podle jména nebo seznamu pitchů |
| `.arp(name, bar, pattern, rate, octaves)` | arpeggio; `up`/`down`/`updown`/`downup`/`random` |
| `.fx(*efekty)` | připojí efektový řetěz |
| `.humanize(timing, vel)` | rozhýbe časování (v beatech) a dynamiku (relativně) |
| `.clear(bars=)` | smaže rozsah taktů |
| `.transpose(semitones)` | posune celou stopu |
| `.repeat(bars, times)` | zkopíruje rozsah taktů dál po timeline |

`s.drums()` vrací stopu s `.pattern(str, bars=(od, do), rate=1/16)`.

## Nástroje

Presety: `epiano`, `warmpad`, `subbass`, `pluck`, `lead`, `strings`, `bell`,
`organ`.

Bicí: `k` kick, `s` snare, `h` hi-hat, `o` open hat, `c` clap, `t` tom,
`r` ride. Všechny syntetizované. V pattern stringu je jeden znak jeden krok
(výchozí šestnáctina), tečka nebo podtržítko je pauza a **velké písmeno
znamená akcent** — `"k...s...k...s..K"` má poslední kopák hlasitější.
Víc řádků v jednom stringu se navrství na sebe.

Vlastní patch:

```python
from paprika import Patch, Sine, ADSR, LP

moje = s.track(Patch(
    osc=[Sine(1.0), Sine(2.01, amp=0.35), Sine(7.0, amp=0.08)],
    adsr=ADSR(0.004, 0.35, 0.4, 1.4),
    filter=LP(2200, env=0.6)))
```

Oscilátory: `Sine`, `Saw`, `Square`, `Tri`, `Noise`, `FM`. První argument je
poměr k základní frekvenci noty, takže `Sine(2.01)` je mírně rozladěná
oktáva — základní trik pro živější zvuk.

Efekty: `Reverb(room, mix, damp)`, `Delay(time, fb, mix)` (tempo-synchronní,
`time` je zlomek celé noty), `LowPass`, `HighPass`, `Saturate(drive)`,
`Sidechain(source_track, amount)`.

## Zpětná vazba

Po renderu se vypíše report:

```
PAPRIKA  romantic.wav
render   1:50  44100 Hz  stereo  4 stop  396 not  [11.8s]

MIX      peak -0.6 dBFS   rms -20.4 dB   no clipping
BANDS    low 51%  mid 48%  high 1%      [zkontroluj]
TRACKS   piano -20.5 dB | pads -28.7 dB | bass -21.0 dB | drums -33.0 dB

WARN  piano + pads   maskování 250-4000 Hz, bars 1-3, 5-7
```

Pravidla warningů:

| Pravidlo | Podmínka |
|---|---|
| nota mimo stupnici | pitch není ve `Song.scale` a stopa nemá `chromatic=True` |
| clipping | jakýkoliv sample před limiterem přesáhne 1.0 |
| špatná hlasitost | peak nad −0.3 dBFS nebo RMS pod −30 dB |
| frekvenční maskování | dvě stopy mají obě přes 40 % energie ve stejném pásmu a čase |
| robotická dynamika | směrodatná odchylka velocity ve stopě je nula |
| ostrá disonance | malá sekunda mezi současně znějícími notami pod MIDI 48 |
| díra v aranži | stopa mlčí 4+ takty uprostřed skladby |

ASCII náhled (`show`) kreslí timeline bez renderu — rychlá kontrola struktury:

```
$ python -m paprika show songs/romantic.py --bars 12:16
       |12      |13      |14      |15      |
piano  |E-G-C---|B-G-E---|D-------|C-------|
pads   |C-------|A-------|F-------|G-------|
bass   |C-------|A-------|F-------|G-------|
drums  |k.h.s.h.|k.h.s.hh|k.h.s.hh|k.h.s.h.|
```

Onset noty je písmeno tónu (malé písmeno = křížek), držení `-`, ticho `.`.
Bicí kreslí svůj znak, akcent velkým písmenem. `--res 16` přepne na
šestnáctiny.

## Referenční analýza

`analyze-ref` čte externí nahrávku (mp3/wav — cokoliv, co zvládne ffmpeg) a
vypíše hrubý odhad tempa, tóniny, akordů po taktech a dynamiky, aby bylo
podle čeho ručně napsat vlastní skladbu ve stejné náladě:

```
$ python -m paprika analyze-ref kyomei.mp3
REFERENCE  kyomei.mp3
audio    3:35  44100 Hz  mono

TEMPO    ~92 BPM  (jistota: vysoká)
KEY      D minor  (korelace 0.87, jistota: vysoká)

DYNAMIKA  (energie po taktech, ~2.6s/takt)
  ████████████████████████████████████████████████████████████▇▁

AKORDY   (hrubý odhad z chroma vektorů, po taktech)
  bar   1-8   Fm    F     F     F     Dm    Dm    Dm    Dm
  ...
```

Detekce je čistě numpy (STFT, spectral-flux tempo, Krumhansl-Schmuckler
key-fit, chroma template matching na akordy) — žádné ML, žádná separace
nástrojů. Na polyfonním mixu (klavír+smyčce+bicí najednou) jsou akordy jen
orientační, hlavně v hlasitých pasážích. Note-by-note transkripce se
nedělá — to spolehlivě nejde bez ML.

`.wav` čte stdlib `wave` bez závislostí navíc. Ostatní formáty (mp3, m4a, ...)
potřebují `pip install imageio-ffmpeg` (lazy import, jen když se použije).

## Struktura

```
paprika/
  __init__.py   veřejné API
  song.py       Song, Track, DrumTrack, Clip, timeline
  note.py       Note a parsing notačních řetězců
  theory.py     stupnice, akordy, obraty, arpeggia
  patch.py      oscilátory, ADSR, filtry, Patch.voice()
  presets.py    8 hotových nástrojů
  drums.py      syntetizované bicí + pattern stringy
  fx.py         Reverb, Delay, LowPass, HighPass, Saturate, Sidechain
  render.py     stopy -> stereo buffer -> WAV
  analyze.py    mix report + hudební warningy
  reference.py  analýza externí nahrávky (tempo/key/akordy/dynamika)
  viz.py        ASCII piano roll + obálka mixu
  cli.py        python -m paprika render|show|play|presets|analyze-ref
```

Hranice jsou naschvál ostré: `render.py` neví nic o hudební teorii,
`theory.py` neví nic o zvuku, `analyze.py` čte jen hotový buffer a datový
model. Noty žijí v beatech, ne v sekundách — změna BPM je jeden parametr.

## Testy

```bash
python -m pytest tests/
python -m unittest discover tests
```

## Co v1 neobsahuje

MCP server, MP3 export, automatizační křivky parametrů v čase, změny tempa
uprostřed skladby, import/export MIDI, stereo šířku a chorus.
