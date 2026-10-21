# Paprika

Konzolová Python knihovna, která umožňuje napsat a vyrenderovat písničku bez
DAW a bez VST pluginů. Napíšeš soubor `songs/*.py`, spustíš
`python -m paprika render` a dostaneš hotový `.wav` plus textovou analýzu,
podle které poznáš, co se povedlo a co ne.

Paprika není generátor hudby. Melodii, harmonii a strukturu skládáš ty (nebo
AI). Paprika dává nástroje: hudební teorii, syntézu zvuku, efekty, mix
a zpětnou vazbu.

**Engine v2** (stereo, modulace, glide, automatizace, sběrnice, konvoluční
hall, sbor, bicí sady, cinematic SFX, 37 presetů) míří na temný cinematic /
dark-pop soundtrack ve stylu Arcane. **Sampler** (v2.1) k tomu přidává
skutečné nástroje z nahrávek — koncertní křídlo, smyčce, dechy, žestě,
mallets — viz [Skutečné nástroje](#skutečné-nástroje-sampler).

## Instalace

```bash
pip install -r requirements.txt   # numpy + scipy
```

WAV zápis jede přes stdlib `wave`. Oscilátory, obálky, efekty i limiter jsou
napsané ručně; scipy dává jen rychlé IIR filtry (`lfilter`) a FFT konvoluci.

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
python -m paprika check  songs/divadlo.py               # harmonie a teorie, bez renderu
python -m paprika midi   songs/divadlo.py               # output/divadlo.mid
python -m paprika play   output/romantic.wav            # systémový přehrávač
python -m paprika presets                               # seznam nástrojů
python -m paprika analyze-ref referencni.mp3             # tempo/key/akordy/dynamika
python -m paprika compare ref.mp3 output/song.wav        # mix metriky proti referenci
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
>G1:1          glide — nota doklouže z výšky předchozí noty (808, slide)
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
| `.automate(cíl, [(takt, hodnota)], curve)` | změna v čase: `vol`, `pan`, `<efekt>.<param>`; křivky `lin`, `exp`, `s`, `step` |
| `.to(bus)` | pošle stopu do sběrnice |

`s.track(nástroj, vol, pan, mono=, glide=, fx=)` — `mono=True` slije
navazující noty do jednoho hlasu (legato), `glide` je sklouznutí v beatech;
opakovaná nota stejné výšky se znovu udeří. `fx="default"` připojí
doporučený řetěz presetu (808 → saturace, sbor → katedrála).

`s.drums(kit="trap")` vrací stopu s `.pattern(str, bars=(od, do), rate=1/16,
swing=0)` a `.roll("h", bar, beat, beats, rate=1/32, vel=(60, 110))`.

`s.sfx()` vrací stopu cinematic efektů: `.riser(to_bar, bars, kind)` skončí
přesně na začátku taktu, `.impact(bar)`, `.braam(bar, "D2", bars)`,
`.subdrop(bar)`, `.reverse_swell(to_bar, beats)`, `.whoosh(bar, beats)`.

`s.bus("drums")` je sběrnice s vlastním fx řetězem a automatizací;
`s.master` je poslední řetěz před limiterem.

```python
bus = s.bus("drums").fx(Compressor(-16, 4))
kick.to(bus); hats.to(bus)
pad.fx(Sidechain("drums", amount=0.4))                  # pumpuje od busu
lead.fx(LowPass(300, res=0.5)).automate("lowpass.cutoff",
                                        [(8, 300), (16, 8000)], "exp")
s.master.automate("vol", [(0, 1), (15.875, 0), (16, 1)], "step")  # díra před dropem
```

## Skutečné nástroje (sampler)

Syntéza neumí znít jako piano v divadle nebo housle. Na to má Paprika
sampler, který přehrává nahrávky skutečných nástrojů (SFZ + WAV, žádná
závislost navíc). Vzorky se stáhnou jednou, mimo repo:

```bash
python -m paprika samples                  # stav, cesta, nástroje a artikulace
python -m paprika samples install all      # ~4.5 GB; nebo piano|strings|winds|brass|mallets|percussion|voices
```

Vzorky leží v `%LOCALAPPDATA%\paprika\samples` (jinde `~/.paprika/samples`),
jinam přes proměnnou `PAPRIKA_SAMPLES`. Instalace pokračuje, kde skončila.

| Skupina | Nástroje (`jméno.artikulace`, první je výchozí) |
|---|---|
| piano | `grand_piano` (Yamaha C5, 16 dynamik), `upright_piano` |
| strings | `violins`, `violas`, `celli` (`sus trem pizz spic`), `violin_solo` (`sus trem pizz spic`), `contrabass` (`sus nv trem pizz spic`), `concert_harp` |
| winds | `flute` (`sus nv exp stac`), `oboe` (`sus nv stac`), `clarinet` (`sus stac`), `bassoon` (`sus vib stac`) |
| brass | `horn` (`sus stac mute`), `trumpet` (`sus vib stac mute`), `trombone` (`sus vib stac`), `tuba` (`sus stac`) |
| mallets | `marimba`, `glockenspiel`, `xylophone` |
| percussion | `timpani` (`hit roll`) — pět kotlů, úder dozní sám, víření drží smyčkou |
| voices | `male_choir`, `basso`, `tenor` (`aah oh ooh eh`) |

```python
piano = s.track("grand_piano", fx="default")        # Hall
piano.notes("(D2 A2 F3):4", bar=0).pedal(bars=(0, 8), every=1)
violin = s.track("violin_solo", fx="default")       # sólo = mono = legato
violin.melody("A4:2 >D5:1 E5:1 | F5:4", bar=8)
s.track("celli.pizz").notes("D3:1 _:1 A2:1 _:1", bar=8)
```

- **Velocity mění barvu**, ne jen hlasitost (dynamické vrstvy nahrávky):
  piano 16 vrstev, smyčce 2–3. Piš dynamiku 40–115, ne všude 100.
  Pozor na hranici vrstev (u dvou vrstev 64): humanize kolem ní nahodile
  přepíná nahrávky — piš buď zřetelně pod, nebo nad.
- **Plynulá dynamika** `t.automate("dyn", [(takt, faktor), ...])` — jako
  modwheel ve filmových knihovnách: faktor násobí velocity not v čase a
  vrstvy nahrávky se prolnou (±12 velocity kolem hranice), takže jeden
  dlouhý tón umí crescendo i decrescendo. 1.0 = beze změny. Jen
  samplované stopy::

      violins.note("D5", at=0, dur=16, vel=110)
      violins.automate("dyn", [(0, 0.3), (3, 1.0), (4, 0.6)], "s")

- **Pedál** `t.pedal(bars=(od, do), every=1)` — nota, která skončí při
  sešlápnutí, zní do uvolnění; `every` přešlápne se změnou harmonie.
  Funguje i na syntetizovaných stopách.
- **Legato** — sólové nástroje (`violin_solo`, `contrabass`, dechy, žestě)
  jsou mono: navazující noty se prolnou bez nového nasazení smyčce.
  Na polyfonní stopě legato zapne `>` před notou nebo `mono=True`.
- **Round robin** — opakovaná nota vezme jinou nahrávku, nezní jako kulomet.
- **Rozsah** — nota mimo rozsah nahrávek se natáhne a zní nepřirozeně;
  report hlásí `mimo rozsah` s taktem a rozsahem nástroje.
- **Vlastní knihovny**: `s.track(Sampler("cesta/k/nastroji.sfz", legato=True))`.
- Ukázka: `songs/divadlo.py` (křídlo + smyčce + harfa + roh).

**Zpěv (skupina `voices`)** je z VocalSet: mužský sbor (šest zpěváků na
klávesu, rozestavění do stran, bez vibrata), `basso` (nejhlubší hlas sady,
operní vibrato, E2–F4) a `tenor` (C3–C5); ženský sbor `female_choir` (šest
zpěvaček ze devíti na klávesu), `soprano` (F#3–A5, zpívá nejvýš) a `alto`
(F#3–G5, nejtmavší barva). Artikulace je samohláska, velocity přepíná
piano / forte nahrávky. Instalace nahrávky rozřeže na jednotlivé tóny,
oktávu každého ověří řadou harmonických a pod nejnižší zazpívaný tón
dopočítá tři hlubší přes TD-PSOLA (výška klesne, samohláska zůstane),
zpěvačkám i dva nad nejvyšší dlouhý tón (soprán nad A5 už ladí nepřesně):

```python
choir = s.track("male_choir.ooh", fx="default")       # katedrála
choir.chord(["D2", "A2", "F3"], bar=0, dur=8)
basso = s.track("basso.aah", fx="default")            # sólo = mono = legato
basso.melody("D3:2 >A2:2 | F3:3 E3:1", bar=4)
```

**Licence:** VocalSet je **CC BY 4.0** — ve hře uveď „Vocals from VocalSet
by J. Wilkins, P. Seetharaman, A. Wahl and B. Pardo, licensed under CC BY 4.0“.
VSCO 2 CE je CC0. Salamander Grand Piano je **CC-BY 3.0** —
ve hře uveď v titulcích „Salamander Grand Piano V3 by Alexander Holm,
licensed under CC-BY 3.0“. Znění je i v `CREDITS.txt` ve složce vzorků.

## Nástroje

Presety podle tagů (`python -m paprika presets --tag choir`):

| Tag | Presety |
|---|---|
| classic | `epiano`, `warmpad`, `subbass`, `pluck`, `lead`, `strings`, `bell`, `organ` |
| orchestral | `strings_ens`, `strings_short`, `pizz`, `cello_solo`, `brass_section`, `horn_swell`, `harp`, `timpani_tone` |
| choir | `choir_ooh`, `choir_aah`, `choir_mm`, `vox_chop` |
| keys | `felt_piano`, `toy_piano`, `music_box`, `celesta` |
| darkpop | `808`, `reese`, `dark_pluck`, `detuned_lead` |
| electronic | `supersaw`, `acid`, `string_stab`, `pwm_pad`, `air_pad` |
| lofi | `lofi_keys`, `glass_pad`, `shimmer_pad`, `round_bass` |

Bicí: `k` kick, `s` snare, `h` hi-hat, `o` open hat, `c` clap, `t` tom,
`r` ride, `x` rim, `b` boom (taiko), `y` shaker, `p` perc. Jak zní, určuje
sada: `classic`, `trap`, `cinematic`, `breakbeat`, `industrial`, `lofi`.
V pattern stringu je jeden znak jeden krok (výchozí šestnáctina), tečka nebo
podtržítko je pauza a **velké písmeno znamená akcent** —
`"k...s...k...s..K"` má poslední kopák hlasitější. Víc řádků v jednom
stringu se navrství na sebe.

Vlastní patch:

```python
from paprika import Patch, Sine, ADSR, LP

moje = s.track(Patch(
    osc=[Sine(1.0), Sine(2.01, amp=0.35), Sine(7.0, amp=0.08)],
    adsr=ADSR(0.004, 0.35, 0.4, 1.4),
    filter=LP(2200, env=0.6)))
```

Oscilátory: `Sine`, `Saw`, `Square`, `Tri`, `Noise(color=white|pink|brown)`,
`FM`, `SuperSaw(voices, detune, spread)` (unison do stran), `Pulse(width,
pwm=LFO)`, `Pluck(decay, bright)` (Karplus-Strong struna), `Additive(partials,
inharm)` (piano, music box, zvony). První argument je poměr k základní
frekvenci noty, takže `Sine(2.01)` je mírně rozladěná oktáva.

Filtry v patchi: `LP`/`HP(cutoff, env, poles, res, lfo)` (rezonance, LFO na
cutoff), `BP`, `Formant(vowel, voice, morph)` — samohlásky a–u pro sbor;
`filter` smí být seznam.

Modulace v patchi: `vibrato=LFO(5, 0.1, delay=0.3)` (půltóny, nástup se
zpožděním), `tremolo=LFO`, `pitch_env=PitchEnv(semi, time)`,
`vel_to_cutoff` (silnější úhoz = jasnější), `rr` (round-robin varianty),
`drift` (analogové kolísání v centech), `ensemble=Ensemble(voices, detune,
spread, smear)` — sekce místo sólisty: kopie hlasu rozladěné o ±`detune`
centů, s nástupem rozmazaným o 0..`smear` s a rozprostřené do stran.

`ADSR(a, d, s, r, curve="exp")` — decay a release klesají exponenciálně jako
u akustického nástroje; `curve="lin"` jsou přímky z v1 (zní „trackerově“).

Efekty: `Reverb(room, mix, damp, width, algo="conv")` (konvoluční dozvuk;
`algo="freeverb"` = kovovější Freeverb z v1),
`Hall(preset, size, mix)` (konvoluce: room / plate / hall / cathedral),
`Delay(time, fb, mix, ping_pong, tone)`, `LowPass`/`HighPass(cutoff, poles,
res)`, `Filter(mode, cutoff, res)`, `EQ(low, mid, high)`, `Saturate(drive)`,
`Compressor(threshold, ratio, attack, release, makeup)`, `Gate(pattern,
rate, depth)` (trance gate), `Chorus`, `Width(amount)`, `BitCrush(bits,
downsample)`, `Tape(wow, flutter, drive, hiss)`, `Sidechain(zdroj, amount)`
— zdroj je stopa nebo bus. Každý efekt bere `name=` pro automatizaci.

## Harmonický plán a kontrola teorie

Plán říká, jaký akord má v kterém taktu znít. Zvuk neovlivní; podle něj
`check` (a report po renderu) hlídá bas, nevlastní tóny, vedení hlasů
a kadence:

```python
s = Song(bpm=70, key="D", scale="minor", bars=16)   # theory="free" je výchozí
s.harmony("Dm | Bb | Gm | Gm:2 A:2")                # jména akordů
s.harmony("i | VI | iv | V7/V", bar=4)               # římské číslice vůči key/scale
s.section("A", bars=(0, 8)).section("B", bars=(8, 16))
```

```bash
python -m paprika check songs/divadlo.py              # za vteřinu, bez renderu
python -m paprika check songs/divadlo.py --bars 8:16
```

- `|` odděluje takty a hlídá délky (`HarmonyError` s číslem taktu), `Gm:2`
  je akord na dva beaty, akordy bez délky si dělí takt rovným dílem
  (`Gm A` = po půl taktu), bez `|` je každý akord jeden takt, `_` = bez
  akordu.
- Jméno akordu je základ + kvalita z `CHORDS`, lomítko je bas: `Dm/F`.
- Římská číslice: velká = dur, malá = moll, `o` zmenšený, `ø7` malý
  zmenšený septakord, `+` zvětšený, `7` diatonická septima (`I7` = Cmaj7,
  `V7` = G7), `6`/`64`/`65`/`43`/`42` obraty, `add6` přidaná sexta (u jména
  je to `C6`), lomítko = vedlejší akord (`V/V`, `V7/iv`, `viio7/V`),
  `b`/`#` posune základ (`bVII`). V mollu se stupně počítají od
  přirozeného mollu (`VI` = Bb v d moll), `V` je durová dominanta a `viio`
  stojí na citlivém tónu.
- Další `harmony()` přes stejné takty plán přepíše — jde psát po sekcích.
- `show` kreslí nad stopami řádek `chords`.

```
PAPRIKA CHECK  divadlo.py
skladba  70 BPM  D minor  32 taktů  9 stop  teorie: free

HARMONIE A (0–7)      i | VI | iv | V | i | III | iv V | i
AKORDY   i=Dm  VI=Bb  iv=Gm  V=A  III=F
KADENCE  A (0–7): V → i autentická · ...

INFO  celli          bar 22   paralelní kvinty s horn: F2/C4 → G2/D4
```

`Song(theory=...)` volí přísnost. `free` (výchozí) je pro cinematic a pop:
kontroluje samostatné linky (jednohlasé stopy), oktávová zdvojení,
arpeggia, ostinata a pady bere jako záměr. `strict` je učebnice (chorál,
kvartet): rozdělí vícehlasé stopy na hlasy a hlásí všechno jako WARN.
`off` teorii vypne.

| Pravidlo | free | strict |
|---|---|---|
| bas na začátku akordu nepatří do akordu (prodleva se nehlásí) | WARN | WARN |
| v úseku akordu nezní jeho základ nebo tercie | INFO | WARN |
| nota linky na těžké době mimo akord, nerozvede se krokem (průtah a appoggiatura projdou) | WARN, když je mimo stupnici, jinak INFO | WARN |
| paralelní kvinty a oktávy mezi linkami | INFO, jen izolované; 3+ po sobě a figurace = záměr | WARN |
| citlivý tón v nejvyšším hlase ve V → I nejde na tóniku | INFO | WARN |
| melodie: skok ≥ malá sexta bez návratu, tritón | INFO | WARN |
| tercie pod G2 držená aspoň beat | WARN | WARN |
| sousední horní hlasy dál než oktávu, křížení linek | — | WARN |

Nota, která je v akordu plánu, není „mimo stupnici“ — C# v dominantě A
v d moll je v pořádku i bez `chromatic=True`.

## MIDI export

```bash
python -m paprika midi songs/divadlo.py               # vedle WAV: output/divadlo.mid
python -m paprika midi songs/x.py -o hotove/x.mid
```

```python
from paprika import export_midi, read_midi
notes = export_midi(s, "output/x.mid")   # vrátí, co se do MIDI nevešlo
mid = read_midi("output/x.mid")          # stopy, noty v tickách, události
```

Standard MIDI File formát 1, 480 ticků na beat. Stopa 0 nese tempo, takt,
předznamenání, sekce jako markery a akordy plánu jako text (`"Dm (i)"`).
Každá stopa je vlastní MIDI stopa se jménem a GM programem (piano, smyčce,
sbor, ...), sustain pedál jde jako CC64, bicí na kanál 10 (kick 36, snare
38, hat 42, ...). Noty mají humanize časování i velocity. SFX stopy (zvuky,
ne noty) a automatizace se nevyexportují — export to vypíše.

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
| nota mimo stupnici | pitch není ve `Song.scale` ani v akordu plánu a stopa nemá `chromatic=True` |
| clipping | jakýkoliv sample před limiterem přesáhne 1.0 |
| špatná hlasitost | peak nad −0.3 dBFS nebo RMS pod −30 dB |
| frekvenční maskování | dvě stopy mají obě přes 40 % energie ve stejném pásmu a čase |
| robotická dynamika | směrodatná odchylka velocity ve stopě je nula |
| ostrá disonance | malá sekunda mezi současně znějícími notami pod MIDI 48 |
| díra v aranži | stopa mlčí 4+ takty uprostřed skladby |
| mono kompatibilita | korelace L/R stopy klesne pod 0 — v mono se vyruší |
| statický zvuk | melodická stopa přes 16+ taktů bez modulace, automatizace i humanize |
| teorie | pravidla z [Harmonický plán a kontrola teorie](#harmonický-plán-a-kontrola-teorie) |

| přetížený sub | víc než 15 % výkonu pod 60 Hz (reference 3–7 %) |
| úzké stereo | šířka mixu pod 8 % u skladby delší než 8 taktů |
| tenké středy | méně než 15 % výkonu v 250 Hz–4 kHz |

Maskování vypisuje jen čtyři nejdelší překryvy a tiché takty (pod -80 dBFS)
ignoruje. `BANDS` ukazuje sedm pásem (sub, bass, lowmid, mid, himid, pres,
air), `STEREO` šířku mixu a korelaci kanálů — obojí s `[ok]` proti
cinematic cíli (sub ≤ 12 %, středy ≥ 20 %, šířka ≥ 10 %).

`compare` vypíše tytéž metriky pro render i referenční nahrávku vedle sebe
a rozdíl proti prvnímu souboru:

```
$ python -m paprika compare songs_mp3/requiem.mp3 output/v2_showcase.wav
soubor                          sub   bass lowmid    mid  himid ...  šířka
requiem.mp3                    3.5%  41.8%  22.1%  27.9%   4.5% ...  29.4%
v2_showcase.wav               11.0%  26.9%  36.2%  24.1%   0.9% ...  19.2%
  rozdíl proti prvnímu        +7.5  -14.9  +14.1   -3.8   -3.6
```

## Jak se vyhnout „čipovému“ zvuku

Retro / 16bitový dojem dělá strojová přesnost a úzký, basově přetížený mix.
Co pomáhá (měřeno proti referencím):

- **Nahrané nástroje** — piano, smyčce, dechy a žestě ze sampleru
  (`grand_piano`, `violins`, ...) místo syntetických presetů; syntéza ať
  dělá pady, basy a elektroniku.
- **Sekce místo sólisty** — smyčce, sbor, žesťe, pady přes `Ensemble`
  (presety `strings_ens`, `choir_*`, `brass_section`, `warmpad`, … ho mají).
- **Žádná rychlá arpeggia** 1/16 na pluck/square — to je podpis čipové
  hudby. Rozložené akordy ve čtvrtích/osminách, legato linky, držené akordy.
- **HighPass na všem kromě basů a kopáku** (80–180 Hz) — sub nechat 808
  a kopáku, a i ty tak, aby sub zůstal pod ~12 %.
- **Prostor** — `Hall`/`Reverb` na melodické stopy, pan ±0.1–0.3.
- **Mastering** — `s.master.fx(*mastering("cinematic"))` (také `darkpop`,
  `lofi`): HP 30 Hz, jemné EQ, glue kompresor, lehká páska.
- **Humanize** na každé stopě, vibrato se zpožděním na sólech.
- Ověř: `python -m paprika compare reference.mp3 output/song.wav`.

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
  __init__.py     veřejné API
  song.py         Song, Track, DrumTrack, SfxTrack, Bus, Clip, timeline
  note.py         Note a parsing notačních řetězců (včetně glide `>`)
  theory.py       stupnice, akordy, obraty, arpeggia
  harmony.py      harmonický plán: jména akordů, římské číslice, kadence
  dsp.py          biquad filtry, LFO, šum, Karplus-Strong, syntetická IR
  oscillators.py  Sine … FM, SuperSaw, Pulse, Pluck, Additive
  filters.py      LP, HP, BP, Formant
  modulation.py   LFO, PitchEnv, Ensemble
  patch.py        ADSR, Patch.voice() (modulace, glide, round-robin)
  automation.py   křivky automatizace a jejich cíle
  metrics.py      mix metriky (7 pásem, šířka, crest, LRA) pro report a compare
  mastering.py    hotové master řetězy (cinematic, darkpop, lofi)
  presets.py      37 nástrojů s tagy, výchozí fx
  drums.py        pattern stringy, render úderů
  kits.py         bicí sady (classic, trap, cinematic, ...)
  sfx.py          riser, impact, braam, subdrop, reverse swell, whoosh
  fx.py           Reverb, Delay, LowPass, HighPass, Filter, EQ, Saturate, Sidechain
  fx_space.py     Hall, Chorus, Width
  fx_dyn.py       Compressor, Gate
  fx_lofi.py      BitCrush, Tape
  sfz.py          SFZ text -> regiony (hlavičky, #define, #include)
  sampler.py      Sampler: vrstvy, round robin, smyčky, legato, mmap WAV
  library.py      katalog samplovaných nástrojů, cesta ke vzorkům
  install.py      stažení Salamander + VSCO 2 CE + VocalSet, generování SFZ
  voices.py       VocalSet -> tóny (YIN, oktáva z harmonických, TD-PSOLA) -> SFZ
  render.py       stopy -> busy -> master -> stereo buffer -> WAV
  rules.py        pravidla teorie nad notami (bas, nevlastní tóny, paralely, ...)
  analyze.py      mix report + hudební warningy, check report bez renderu
  midi.py         MIDI export a čtení (SMF formát 1)
  reference.py    analýza externí nahrávky (tempo/key/akordy/dynamika)
  viz.py          ASCII piano roll + obálka mixu
  cli.py          python -m paprika render|show|check|midi|play|presets|samples|analyze-ref|compare
```

Hranice jsou naschvál ostré: `render.py` neví nic o hudební teorii,
`theory.py` a `harmony.py` nevědí nic o zvuku, `rules.py` a `midi.py` čtou
jen datový model, `analyze.py` jen hotový buffer a datový model. Noty žijí v beatech, ne v sekundách — změna BPM je jeden parametr.

## Testy

```bash
python -m pytest tests/
python -m unittest discover tests
PAPRIKA_SLOW=1 python -m pytest tests/test_compat.py        # vyrenderuje všechny songy
PAPRIKA_SLOW=1 python -m pytest tests/test_samples_real.py  # nainstalované vzorky: úroveň, ladění
```

Ostatní testy sampleru běží bez sítě nad malými syntetickými WAV.

`songs/v2_showcase.py` je poslechová ukázka všech novinek v2.

## Co v2 neobsahuje

Export do hry (smyčky, stemy, vrstvy intenzity,
stingery), žánrové style packy a Claude Code skill `/paprika` — to jsou další
podprojekty. Dál chybí zpěv s textem, import MIDI do skladby, změny tempa
uprostřed skladby a automatizace na úrovni jednotlivé noty.
