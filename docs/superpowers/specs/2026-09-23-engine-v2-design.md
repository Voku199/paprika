# Paprika — zvukový engine v2

**Datum:** 2026-09-23
**Stav:** schváleno (sekce 1–2 v chatu, 3–4 podle doporučení — uživatel delegoval)
**Navazuje na:** `2026-08-12-paprika-design.md` (v1)

## Proč

Paprika má sloužit jako zdroj instrumentálního soundtracku pro Unity hru ve
stylu Arcane (reference: Woodkid — To Ashes and Blood, Stromae & Pomme — Ma
Meilleure Ennemie, Sub Urban — Cradles, Moby — Extreme Ways, SEAPOOL — mu).
Dnešní skladby znějí všechny podobně a synteticky. Příčiny v kódu:

1. 8 statických presetů; `subbass` je v 16/16 songů, jediná bicí sada.
2. Nota je v čase zmrazená — žádné LFO, vibrato, pitch envelope, glide.
3. Filtr je jednopólový bez rezonance; žádný formantový/pásmový filtr.
4. Stopa je mono až do mixu — žádný unison, chorus, stereo šíře.
5. Žádná automatizace (sweep, swell) → nejde build-up a drop.
6. Žádné cinematic prvky (riser, impact, braam, sub drop), žádný sbor.

## Celkový plán (4 podprojekty)

| # | Podprojekt | Stav |
|---|---|---|
| 1 | **Zvukový engine v2** (tento dokument) — 1a jádro, 1b obsah | tento spec |
| 2 | Style packy + Claude Code skill `/paprika` (prompt → song) + paměť použitých voleb proti opakování | další spec |
| 3 | Export do hry: loop mód, stemy, vrstvy intenzity, stingery, manifest JSON, `MusicPlayer.cs` pro Unity | další spec |
| 4 | Sampler SFZ (VSCO2 CE, Sonatina, vlastní WAV) | další spec |

Rozhodnutí pro celý projekt: spouštěč = prompt → Claude; zvuk = hybrid
(syntéza + volitelný sampler); **instrumentální, bez zpěvu**; engine hry =
Unity; závislosti `numpy` + **`scipy`**.

## Rozhodnutí a důvody

| Rozhodnutí | Volba | Proč |
|---|---|---|
| IIR filtry | `scipy.signal.lfilter` | Rezonance, formanty, EQ a Karplus-Strong potřebují rekurzi; numpy ji nevektorizuje. lfilter běží v C. |
| Časově proměnný filtr | bloky po 256 samplech, stav `zi` předávaný mezi bloky | Žádný per-sample Python cyklus, žádné lupance na hranách bloků. |
| Stereo | buffer stopy vždy `(n, 2)` float32 | Unison, chorus, šířka, stereo reverb. Mono downmix jen pro analýzu. |
| Staré efekty | `Fx.process_mono` + výchozí `process` po kanálech | Mono efekty se nemusí přepisovat. |
| Automatizace | na úrovni stopy/efektu, ne per-voice | Voice cache zůstává funkční. |
| Zpětná kompatibilita | všech 16 songů v `songs/` musí vyrenderovat | Zvuk se smí změnit (stereo reverb je širší), API ne. |

## 1a — jádro

### Stereo cesta

- `Patch.voice()` vrací `(n,)` mono, nebo `(n, 2)` stereo (když některý
  oscilátor je stereo, např. `SuperSaw` se `spread > 0`).
- `render_track` sčítá do `(n, 2)`; mono hlas se rozkopíruje do obou kanálů.
- `Fx.process(buf (n,2), ctx) -> (n,2)`. Základní třída volá
  `process_mono(x, ctx)` na každý kanál zvlášť, pokud podtřída
  nepřepíše `process`. Stávající efekty přejdou na `process_mono`.
- `mix()` bere stereo buffery; pan je equal-power balance (L/R gain) nad
  stereo signálem.
- `RenderResult.tracks` drží stereo buffery; `analyze` si dělá mono downmix
  (`buf.mean(axis=1)`), takže pravidla zůstávají beze změny.

### DSP modul `paprika/dsp.py` (nový)

Čisté funkce, žádná hudební teorie:

- `biquad(kind, cutoff, q, gain_db, sr) -> (b, a)` — RBJ cookbook:
  `lp`, `hp`, `bp`, `notch`, `peak`, `lowshelf`, `highshelf`.
- `filt(x, kind, cutoff, q=0.707, gain_db=0, sr, zi=None)` — konstantní
  koeficienty přes `lfilter`.
- `filt_varying(x, kind, cutoff_curve, q_curve, sr, block=256)` — po blocích,
  stav předávaný přes `lfilter_zi` / výstupní `zf`. Koeficienty se počítají pro
  střed bloku.
- `res_to_q(res)` — `res` 0..1 → Q 0.707..~12 (exponenciálně).
- `lfo(shape, rate_hz, n, sr, phase=0, seed=0) -> ndarray` v rozsahu −1..1;
  tvary `sine|tri|saw|square|random` (random = sample&hold vyhlazený).
- `karplus(freq, n, sr, decay, bright, seed)` — Karplus-Strong přes `lfilter`
  (feedback comb s průměrovacím LP ve smyčce; zlomkové zpoždění
  allpassem prvního řádu).
- `synth_ir(seconds, sr, predelay, damp, early, seed) -> (n, 2)` —
  syntetická impulzní odezva: dekorelovaný šum L/R, exponenciální útlum na
  −60 dB za `seconds`, časné odrazy, LP tlumení rostoucí s časem.

`patch.onepole_lp/hp` zůstanou (používá je `drums` a testy), ale per-sample
větev s proměnným cutoffem se přepíše na `dsp.filt_varying`.

### Oscilátory (`patch.py`)

- Podpis `Osc.render(freq, n, sr)` přijímá `freq` jako skalár **nebo**
  pole délky `n` (frekvenční křivka v Hz). Fáze = `cumsum(freq*ratio)/sr`.
- Band-limited `Saw/Square/Tri`: počet harmonických podle `max(freq)`.
- Nové:
  - `SuperSaw(ratio, amp, voices=7, detune=0.25, spread=0.8, seed)` —
    unison, detune v půltónech (celkový rozsah), `spread` 0 = mono, 1 = plně
    do stran. Výstup stereo.
  - `Pulse(ratio, amp, width=0.5, pwm=None)` — PWM přes LFO; band-limited
    jako rozdíl dvou saw.
  - `Noise(..., color="white|pink|brown")` — pink/brown přes `lfilter`.
  - `Pluck(ratio, amp, decay=1.0, bright=0.5)` — Karplus-Strong
    (pizzicato, harfa, kytara, koto).
  - `Additive(partials=[(ratio, amp, decay)...], inharm=0.0)` — piano,
    music box, celesta, zvony s nezávislým útlumem každé harmonické.
- Každé nové pole je v `key()`.

### Modulace v `Patch`

Nová volitelná pole (všechna v `Patch.key()`):

| Pole | Typ | Význam |
|---|---|---|
| `vibrato` | `LFO` | hloubka v půltónech; `delay` = nástup za X s po začátku noty |
| `tremolo` | `LFO` | hloubka 0..1 amplitudy |
| `pitch_env` | `PitchEnv(semi, time, curve="exp")` | start o `semi` výš/níž, dojede na 0 za `time` s |
| `vel_to_cutoff` | float 0..1 | velocity posouvá cutoff filtru (jas podle síly úhozu) |
| `rr` | int | round-robin: N variant (fáze + ±3 centy), cyklicky podle pořadí noty |
| `drift` | float centy | pomalé náhodné kolísání ladění (analogový feel) |

`LFO(rate=5.0, depth=0.1, shape="sine", sync=None, delay=0.0, fade=0.0)` —
`sync` (zlomek celé noty) má přednost před `rate` a přepočítá se podle BPM
v renderu. Tempo se proto dostane do `Patch.voice(..., bpm=)`.

Filtry v patchi: `LP/HP(cutoff, env, poles, res=0.0, lfo=None)` a nové
`BP`, `Formant(vowel, ...)`. `Patch.filter` smí být i seznam filtrů
(řetězení). `res=0` a `lfo=None` zachová přesně dnešní chování pro staré
songy (jednopólová cesta).

`Formant(vowel="a|e|i|o|u"|mix, voice="auto|bass|tenor|alto|soprano",
morph=None)`: 3–5 paralelních `bp` biquadů na formantech samohlásky; při
`voice="auto"` se tabulka vybere podle výšky noty. `morph=("o", "a", time)`
přejde mezi samohláskami (časově proměnný filtr).

### Glide / mono legato

- `Song.track(..., mono=False, glide=0.0)`; `glide` v beatech.
- V mono režimu se noty, které se překrývají nebo navazují (mezera ≤ 1/64
  beatu), slijí do jednoho hlasu: frekvenční křivka drží pitch první noty
  a na startu každé další noty exponenciálně doklouže za `glide`.
- Notace: prefix `>` na tokenu (`C2:1 >G1:1`) vynutí glide na tuto notu
  i u stopy bez `mono=True` (lokální slide). `Note` dostane pole
  `glide: bool = False`.
- Cache klíč mono-frází: (patch key, tuple((pitch, dur, start_rel)...),
  glide).

### Automatizace

```python
t.automate("vol", [(0, 0.0), (4, 0.8)])
t.automate("lowpass.cutoff", [(16, 300), (24, 9000)], curve="exp")
t.automate("hall.mix", [(30, 0.2), (32, 0.7)], curve="s")
```

- Body jsou `(bar, value)`; `bar` smí být float (`16.5` = půlka taktu).
  Před prvním bodem platí jeho hodnota, za posledním taky.
- Křivky: `lin`, `exp` (geometrická, pro frekvence), `s` (smoothstep),
  `step`.
- Cíl: `vol`, `pan`, nebo `<fx>.<param>`. `<fx>` je `Fx.name`
  (výchozí = jméno třídy malými písmeny, u duplicit `lowpass2`…), nebo
  explicitní `name=` v konstruktoru efektu.
- Efekt deklaruje `AUTOMATABLE = ("cutoff", "res", ...)`. Neznámý cíl →
  `KeyError` se seznamem dostupných cílů na stopě.
- Render převede body na per-sample pole (beats → samples jednou, s
  `beat_offset` pro `--bars` výřez) a předá je efektu přes
  `fx.set_curve(param, array)`. Efekt při zpracování volá
  `self.value(param, n)` → skalár nebo pole.
- Automatizace je i na `Bus` a `Master`.

### Sběrnice a master

```python
drums_bus = s.bus("drums", vol=0.9).fx(Compressor(-18, 4), Saturate(1.5))
kick.to(drums_bus); snare.to(drums_bus)
s.master.fx(EQ(low=(80, 1.5)), Compressor(-10, 2, attack=0.03))
```

- `Bus(name, vol, pan)` má `fx_chain`, `automate`, `to(other_bus)` není
  (jedna úroveň, YAGNI).
- Pořadí renderu: stopy bez závislostí → stopy se Sidechainem → busy →
  mix busů a stop, které do busu nejdou → master fx → limiter.
- `Sidechain(source)` smí odkazovat na stopu i bus (bus až po jeho
  vyrenderování; Sidechain na busu smí odkazovat jen na stopy).
- `s.master` existuje vždy, výchozí řetěz je prázdný.

### Efekty (`fx.py`)

Upravené:
- `LowPass/HighPass(cutoff, poles=2, res=0.0)` — biquad, automatizovatelné
  `cutoff`, `res`. `res=0` → Q 0.707 (Butterworth). Pozn.: zvuk
  starých songů se mírně změní (strmější filtr), API ne.
- `Reverb` — stereo (pravý kanál s posunutými délkami combů o 23 samplů,
  jako Freeverb), `width`. Automatizovatelné `mix`.
- `Delay(time, fb, mix, ping_pong=False, tone=None)` — `tone` = LP ve
  zpětné vazbě (tmavnoucí echa). Automatizovatelné `mix`, `fb`.
- `Saturate` — automatizovatelné `drive`, `mix`.
- `Sidechain` — zdroj stopa nebo bus.

Nové:

| Efekt | Parametry | Automatizovatelné |
|---|---|---|
| `Filter(mode, cutoff, res)` | `lp/hp/bp/notch` | cutoff, res |
| `EQ(low=(f, dB), mid=[(f, dB, q)...], high=(f, dB))` | shelvy + peaky | — |
| `Compressor(threshold_db, ratio, attack, release, makeup_db, knee_db)` | feed-forward, RMS detektor přes onepole | threshold |
| `Chorus(rate, depth_ms, mix, voices=2)` | modulované zpoždění přes `np.interp`, L/R opačná fáze LFO | mix |
| `Width(amount)` | M/S: 0 = mono, 1 = beze změny, 2 = extra široké | amount |
| `Hall(preset="hall", size, decay, predelay, mix, damp)` | konvoluce se `synth_ir`; presety `room/plate/hall/cathedral` | mix |
| `BitCrush(bits, downsample, mix)` | lo-fi | mix |
| `Tape(wow, flutter, drive, hiss)` | modulované zpoždění + saturace + šum | — |
| `Gate(pattern="x.x.xx..", rate=1/16, depth, smooth)` | rytmický trance gate / stutter synchronní s tempem | depth |

IR z `Hall` se cachuje podle `(preset params, sr)`.

## 1b — obsah

### Presety

Stávajících 8 presetů zůstává beze změny (kompatibilita). Přibude ~26 nových,
každý s tagy (`PRESET_TAGS`), `python -m paprika presets --tag cinematic`:

| Tag | Presety |
|---|---|
| orchestral | `strings_ens` (supersaw + vibrato + formant tvarování), `strings_short` (staccato), `pizz` (Pluck), `cello_solo`, `brass_section`, `horn_swell`, `harp` (Pluck), `timpani_tone` |
| choir | `choir_ooh`, `choir_aah`, `choir_mm`, `vox_chop` (krátký formantový pluck) |
| keys | `felt_piano` (Additive + inharmonicita + úder kladívka), `toy_piano`, `music_box`, `celesta` |
| darkpop / trap | `808` (sine + pitch env + mono + glide + saturace), `reese` (rozladěné saw + LFO filtr), `dark_pluck`, `detuned_lead` |
| electronic | `supersaw`, `acid` (rezonanční LP + env), `string_stab` (Moby), `pwm_pad`, `air_pad` (šum + formant) |
| lofi / dreamy | `lofi_keys`, `glass_pad`, `shimmer_pad`, `round_bass` |

Presety mohou vracet i doporučený fx řetěz: `preset_fx(name) -> list[Fx]`
(např. `808` → `Saturate(2)`), který `Song.track(name, fx="default")`
připojí. Výchozí `fx=None` = nic se nepřipojí (kompatibilita).

### Bicí sady

`s.drums(kit="classic")` — `classic` je dnešní zvuk (kompatibilita).
Nové sady: `trap`, `cinematic`, `breakbeat`, `industrial`, `lofi`.

Znaky: stávající `k s h o c t r` + nové `x` (rim/click), `b` (boom — taiko
nebo velký low hit), `y` (shaker), `p` (perc — kit-specifický, např. kovový
cink v `industrial`, snap v `trap`). Sada, která znak nemá, spadne na
`classic`.

| Sada | Charakter |
|---|---|
| trap | 808 kick s dlouhým ocasem, ostrý snare/clap, těsné hi-haty |
| cinematic | taiko (`b`), orchestrální basový buben (`k`), snare s víc šumem, gong-like crash (`o`), low toms |
| breakbeat | punchy „akustický“ kick, praskavý snare, živé hi-haty (Moby / Amen feel) |
| industrial | zkreslený kick, kovové údery, noise snare |
| lofi | měkký kick, tlumený snare, zašuměné hi-haty (bitcrush už v generátoru) |

Nové metody `DrumTrack`:
- `roll(name, bar, beat=0, beats=1, rate=1/32, vel=(60, 110))` — trap
  hi-hat roll / snare roll s velocity rampou.
- `pattern(..., swing=0.0)` — posun každého sudého kroku o `swing`×krok.

Cache: `(kit, name, sr)`.

### Cinematic SFX stopa

`sfx = s.sfx(vol=0.6)` → `SfxTrack(DrumTrack)` (dědí, takže všechny
větve `isinstance(t, DrumTrack)` v analyze/viz/render fungují). Hit nese
`params`; generátor dostane i BPM, takže délky sedí na takty.

| Metoda | Zvuk |
|---|---|
| `riser(to_bar, bars=2, kind="noise"|"tone"|"both")` | končí přesně na začátku `to_bar` |
| `impact(bar, beat=0, size=1.0)` | sub boom + šumový crack + dlouhý ocas |
| `braam(bar, pitch="D2", bars=2)` | Inception-style žesťová rána (supersaw + formant + distorce) |
| `subdrop(bar, beat=0)` | sine 60→25 Hz |
| `reverse_swell(to_bar, beats=2)` | reverzní činel / reverb, končí na `to_bar` |
| `whoosh(bar, beats=1)` | filtrovaný šum s panem zleva doprava |

Pauza před dropem se dělá automatizací `vol` na busu/masteru, ne SFX.

### Analýza

- Nové pravidlo `_mono_compat`: WARN, když stereo korelace stopy v mixu
  klesne pod 0 (fázový rozpad v mono). Max 3 hlášení.
- Nové pravidlo `_static_sound`: INFO pro melodickou stopu delší než 16
  taktů, jejíž patch nemá žádnou modulaci (vibrato/LFO/pitch env/rr) a stopa
  nemá automatizaci ani `humanize`. Max 3 hlášení. Cílí přímo na
  „zní to plasticky“.
- Report: řádek `STEREO  šířka X %  korelace Y` pro celý mix.
- `show` kreslí `SfxTrack` symboly (`R` riser, `I` impact, `B` braam, `v`
  subdrop, `<` reverse swell, `w` whoosh).

## Testy

Každá nová jednotka má unittest (styl projektu: třídy, `sys.path` bootstrap):

- `dsp`: frekvenční odezva LP/HP na cutoffu (−3 dB ±1), rezonanční špička
  roste s `res`, `filt_varying` bez skoků na hranách bloků (max rozdíl
  sousedních samplů u sinu), `lfo` rozsah a perioda, `karplus` základní
  frekvence ±1 %, `synth_ir` útlum na −60 dB v toleranci.
- Oscilátory: frekvenční křivka (glide) — nulové průchody na konci odpovídají
  cílové výšce; SuperSaw stereo a L≠R; band-limit bez energie nad Nyquistem.
- Patch: vibrato mění okamžitou frekvenci o ±depth; `rr` dává různé buffery
  a cache je rozliší; všechna nová pole v `key()` (test: změna každého pole →
  jiný klíč).
- Automatizace: hodnoty křivek na bodech a mezi nimi; `vol` automate ztiší
  začátek; neznámý cíl → `KeyError` se seznamem.
- Bus: součet stop, fx na busu, Sidechain z busu.
- Efekty: kompresor snižuje peak nad prahem, Chorus/Width mění korelaci,
  Gate následuje pattern, Hall tail ≤ `tail()`.
- Presety: každý nový preset je konečný, pod 1.0, a **páry presetů se
  spektrálně liší** (vzdálenost průměrných log-spekter nad prahem) —
  automatická kontrola „nezní to stejně“.
- Bicí: každá sada × každý znak renderuje konečný, omezený zvuk; sady se
  spektrálně liší; `roll` počet úderů a rampa; `swing` posun.
- SFX: `riser` končí na hraně taktu ±1 blok; `impact` peak na začátku.
- Kompatibilita: všech 16 songů v `songs/` se načte a vyrenderuje
  `--bars 0:2` bez výjimky a bez NaN (smoke test, označený pomalý).
- Výkon: render `songs/iron_requiem.py` nesmí být víc než 1.5× pomalejší
  než v1 (měřeno v plánu, ne v unittestu).

## Ukázka

`songs/v2_showcase.py` — 24 taktů, které projdou všemi novinkami (orchestrální
intro se sborem a Hall → trap build s rollem, risem a filtr sweepem → drop
s 808 glide, braam, impact). Slouží jako poslechový test pro uživatele.

## Co v2 neobsahuje

Sampler (podprojekt 4), exporty do hry (3), style packy a skill (2), MIDI,
změny tempa uprostřed skladby, per-note automatizace (MPE), zpěv.
