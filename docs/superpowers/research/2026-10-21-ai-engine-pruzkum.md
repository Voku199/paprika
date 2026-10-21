# Paprika jako AI audio engine — průzkum trhu a technologií

**Datum:** 2026-10-21
**Typ:** průzkum + návrh architektury (ne schválený spec)
**Navazuje na:** `specs/2026-09-23-engine-v2-design.md` (4 podprojekty),
`specs/2026-09-24-sampler-design.md`

Otázky, na které dokument odpovídá:

1. Jak má AI „slyšet“ vlastní výstup (audio feedback loop)?
2. Jak postavit „FL Studio pro AI“ — stopy, MIDI, nástroje, efekty — místo
   generování celého audia naráz?
3. Jak dostat operní výstřižky z VocalSetu na úroveň realistického zpěvu?
4. Jak vynutit hudební teorii tak, aby výsledek zněl organicky?

Zdroje jsou na konci. U čísel z preprintů a z blogů platí, že jsou
**self-reported** a nebyla nezávisle ověřena. Licence je před použitím
potřeba zkontrolovat přímo v repozitáři.

---

## TL;DR — osm doporučení

1. **Nedělat z paprika end-to-end generátor.** Na tom trhu vládne Suno
   (ocenění 5,4 mld. $, červen 2026) a open-source ACE-Step 1.5 / HeartMuLa.
   Paprika má jinou, volnou pozici: *ověřitelná skladba pro AI agenta*
   (symbolika → deterministický render → měřitelná zpětná vazba), čistý
   původ zvuku (CC0/CC BY vzorky + vlastní syntéza) a export do hry.
2. **„Uši“ stavět hlavně jako ověření záměru, ne jako slepé poslouchání.**
   Paprika zná partituru a má stopy před mixem (`RenderResult.tracks`). Nejsilnější
   signál je *rozdíl mezi tím, co je napsané, a tím, co zní*: přepis každé
   stopy zpět na noty, chroma vs. plánované akordy, beat tracking vs. mřížka,
   LUFS a maskování po stopách. Audio-LLM jsou jen doplněk: v dlouhém
   audiu mají otevřené modely podle neověřeného komunitního benchmarku
   ~39–57 % přesnosti.
3. **Uši ve třech vrstvách:** L1 měření a přepis (CPU, sekundy) → L2 embeddingy
   a skóre kvality (CLAP / MuQ-MuLan / CLaMP 3, Audiobox Aesthetics, FAD vs.
   reference) → L3 kritik audio-LLM (Qwen3-Omni, Audio Flamingo Next,
   Music Flamingo) s textovou kritikou. Každá vrstva vrací strukturovaný JSON
   s čísly taktů, ne prózu.
4. **Jádro zůstává numpy + scipy.** Těžké neuronové věci (torch) běží jako
   volitelné *workery* v odděleném procesu nebo venv a s enginem si vyměňují
   soubory (WAV/MIDI/JSON) s cache podle hashe. Chybějící model hodí chybu
   ve stylu `SamplesMissing`, nikdy tichý fallback.
5. **Teorie jako kompilátor.** Nová vrstva `plan` (forma, sekce, harmonický
   plán v římských číslicích, energetická křivka). AI napíše plán, pak party,
   a validátor hlídá party proti plánu: vedení hlasů, nevlastní tóny, kadence,
   rozsahy, registry. Hlásí to stejně přesně jako dnes `NoteLengthError`.
   K tomu korpusový „otisk“ (podle Libretta), který ukáže, kde se skladba
   statisticky odchyluje od skutečné hudby.
6. **Hlasy po žebříku:** (A) bezslovný sbor ze stávajících vzorků s morfem
   samohlásek a WORLD resyntézou, (B) partitura → zpěv přes DiffSinger nebo
   SoulX-Singer, (C) barva hlasu přes Seed-VC/RVC, (D) celé písňové modely
   jen jako generátor nápadů. Bark a Diff-SVC ne.
7. **Interop místo vlastního GUI:** export MIDI, stemů, DAWproject a REAPER
   `.rpp`, volitelný hosting VST3 přes Pedalboard nebo DawDreamer a MCP server
   paprika, aby ji mohl řídit kterýkoli agent.
8. **Smyčka generuj → změř → oprav** s pevným počtem kol a pamětí nejlepšího
   kandidáta. Je to přesně podprojekt 2 (`/paprika` skill) rozšířený o
   příkaz `paprika critique`.

---

## 0. Výchozí stav (co paprika umí a co chybí)

| Oblast | Dnes | Chybí pro „AI DAW“ |
|---|---|---|
| Skladba | Python DSL, `Song/Track/Clip/Bus`, notace, akordy, arpy, humanize | plán formy a harmonie, motivy a jejich transformace, MIDI import/export |
| Zvuk | 37 presetů (syntéza), SFZ sampler (Salamander, VSCO 2 CE), VocalSet sbor s PSOLA | VST3 nástroje, neuronové hlasy, texty |
| Mix | Fx řetězce, sběrnice, automatizace, limiter, mastering | optimalizace parametrů podle cíle (LUFS, maskování, pásma) |
| Zpětná vazba | report (úrovně, maskování, disonance v basu, mono kompatibilita, rozsahy), ASCII roll, `compare`, `analyze-ref` (tempo/tónina/akordy) | přepis audia → noty, ověření záměru, embeddingy, skóre kvality, kritika v přirozeném jazyce, strojově čitelný (JSON) report |
| Export | WAV | stemy, MIDI, smyčky, DAW projekty (podprojekt 3) |

Klíčová strukturální výhoda: render zná **partituru i stopy před mixem**.
Většina výzkumu „AI, která poslouchá“ pracuje s hotovým mixem bez
partitury. Paprika umí víc: porovnat zamýšlené se znějícím po jednotlivých
stopách.

---

## 1. Trh — kdo co dělá a kde je mezera

### 1.1 Krajina (říjen 2026)

| Segment | Hráči | Co z toho plyne pro paprika |
|---|---|---|
| End-to-end písně (uzavřené) | **Suno** (v5; Series D 400 mil. $ při ocenění 5,4 mld. $ v červnu 2026, ~300 mil. $ ARR; dohoda s WMG), **Udio** (dohody s UMG a WMG, přechod do „walled garden“, stahování vypnuté), Google **Lyria 3** / Lyria RealTime, ElevenLabs Music, Stable Audio 2.5 | Kvalita „hotové písně“ je komodita. Konkurovat zde nemá smysl. Právní riziko (Sony v červenci 2026 znovu žaluje Udio) je argument pro čistý původ zvuku. |
| End-to-end (open weights) | **ACE-Step 1.5** (leden 2026, < 4 GB VRAM, LoRA z pár písní), **HeartMuLa** (leden 2026, Apache 2.0), **YuE** (ICLR 2026), **LeVo 2** (Tencent), DiffRhythm 2, SongBloom, Muse | Použitelné jako *nástroj uvnitř stopy* (textury, vokální nápady), ne jako skladatel. Chybí kontrola na úrovni not. |
| Generativní DAW | **Suno Studio 2.0** (13. 8. 2026: MIDI, wavetable synth, automatizace, export stemů; jen tarif Premier, bez VST/AU), Magenta RealTime 2 (červen 2026, open weights, MIDI vstup, ~200 ms latence) | Trh se posouvá od „celé písně“ ke stopám a MIDI. To potvrzuje směr paprika. Suno Studio je ale uzavřené a AI v něm nemá uši. |
| Mosty LLM → DAW (MCP) | `ableton-mcp` (nejzralejší), `reaper-mcp` (~130 nástrojů), `flstudio-mcp` (file-based IPC se skriptováním FL), DAWZY (ReaScript akce), Doseedo (konverze projektů) | Agent ovládá DAW, ale **neslyší**. Uši nikdo nedodává. To je mezera. |
| Výzkum agentů | **Libretto** (červen 2026: měřitelná symbolika + smyčka generuj–změř–oprav), ComposerX (multiagentní kompozice), LLM2Fx / LLM2Fx-Tools (LLM nastaví EQ a reverb z textu), InstructFX2FX (DAFx26: LLM vybere efekty, optimalizace řízená CLAPem doladí parametry), Audio-Oscar (multiagentní zvukové scény s refinementem), „Ears/Brain/Mouth“ co-creation agent | Přesně skládačka, kterou paprika potřebuje. Nikdo ji nemá jako celek nad deterministickým enginem. |
| Hodnocení hudby | SongEval, Audiobox Aesthetics, MuseCritic (reward s textovou kritikou), TuneJury, SongBench, rubrikové optimalizace | Skóre lze použít jako uši v L2. Pozor: proxy reward při přeoptimalizaci snižuje diverzitu. |

### 1.2 Pozice paprika

Nikdo nedělá kombinaci **(a)** symbolika psaná agentem, **(b)**
deterministický a reprodukovatelný render, **(c)** zpětná vazba, která
ověřuje záměr po stopách, **(d)** zvuk s čistým původem a **(e)** výstup
pro hru (stemy, smyčky, stingery). Suno Studio má (a) jen částečně,
MCP mosty mají (a) bez (c), open modely mají jen audio.

Doporučená formulace: *„Paprika je kompilátor hudby pro AI. Agent píše
zdrojový kód skladby, engine ho přeloží na zvuk a vrátí chyby a
varování.“* „FL Studio pro AI“ je dobrá metafora pro UX, ale pro AI je
důležitější **zpětná vazba typu kompilátor** než GUI.

---

## 2. Audio vnímání („AI Hearing“)

### 2.1 Co které zmíněné technologie umí a neumí

| Technologie | Role | Hodí se pro paprika? |
|---|---|---|
| **Whisper** | ASR řeči | Jen na **srozumitelnost textu** ve zpěvu: přepsat vyrenderovaný vokál a spočítat WER proti zamýšlenému textu. Hudbu nerozumí. |
| **CLAP** (LAION-CLAP, MS-CLAP), **MuQ-MuLan**, **CLaMP 3** | společný embedding text ↔ audio (CLaMP 3 i partitura) | Ano, L2: „odpovídá sekce briefu *dark cinematic, Arcane*?“ a vzdálenost k referencím. V benchmarku MUUNRiver vede MuQ-MuLan nad CLaMP 3 (0,45 vs. 0,36 mAP na T2). CLaMP 3 umí i MIDI/ABC, takže porovná partituru se zvukem. |
| **EnCodec / DAC / Mimi** | neuronové kodeky (tokenizace audia) | Pro uši ne. Jsou to vnitřnosti generativních modelů. Jako embedding je lepší CLAP/MuQ. Relevantní jen při vlastním tréninku generátoru. |
| **CLAP-based reward models** | skóre podobnosti textu a audia | Ano, ale jen jako jeden signál z několika. Rubrikový přístup (2026) ukazuje, že optimalizace na jediné metrice nezlepší ostatní. |
| **Audiobox Aesthetics** (Meta) | 4 osy: Content Enjoyment, Content Usefulness, Production Complexity, Production Quality | Ano, L2. CC BY 4.0, rychlé. Dobrý „technický posluchač“. |
| **SongEval** | estetika celé písně (5 os) | Pro výzkum. Licence CC BY-NC-SA, takže ne do komerčního pipeline hry. |
| **Audio-LLM** (Qwen3-Omni, Audio Flamingo Next, Music Flamingo) | volná kritika, Q&A nad audiem | L3 jako „druhý názor“. Qwen3-Omni je Apache 2.0 (včetně varianty Captioner). Music Flamingo umí teorii (akordy, harmonie, chain-of-thought), ale licence je jen pro nekomerční výzkum. |

### 2.2 Doporučená architektura uší (L0–L3)

```
        render (numpy)                       ┌─────────── L3 kritik (volitelné, GPU/API)
  Song ──► stopy (n,2) + mix ──┬─► L0 DSP    │  Qwen3-Omni / AF-Next / Gemini:
  (partitura, plán)            │   (dnes)    │  „sekce B: smyčce maskují melodii,
        │                      ├─► L1 OVĚŘENÍ ZÁMĚRU (CPU, s)       chybí kontrast“
        │                      │   přepis stop → noty  ──┐   └─► soft signál, ověřit L1
        └──────────────────────┼──► diff partitura ↔ zvuk│
                               │   chroma ↔ plán akordů  │
                               │   beat ↔ mřížka, LUFS   │
                               ├─► L2 EMBEDDINGY A SKÓRE (CPU/GPU, s)
                               │   CLAP/MuQ ↔ brief, FAD/KAD ↔ reference,
                               │   Audiobox Aesthetics po sekcích
                               ▼
                     critique.json  (problém, takty, závažnost, návrh opravy)
                               ▼
                     agent (Claude Code + /paprika skill) upraví songs/x.py
```

**L1 — ověření záměru** (největší přínos, nejnižší cena):

| Kontrola | Jak | Knihovna |
|---|---|---|
| Hraje stopa noty, které jsou napsané? (ladění vzorků, chybná oktáva, utopené noty) | přepis *stopy před mixem* → noty, porovnání s partiturou (onset ±30 ms, výška ±50 centů), F1 po taktech | **Basic Pitch** (Apache 2.0, ONNX/CPU) na monofonní a jednoduché stopy, **MuScriptor** (červenec 2026, multi-instrument, kód MIT, váhy CC BY-NC) jako referenční přepis mixu, RMVPE/torchcrepe/PESTO na zpěv |
| Zní plánovaný akord? | chroma (CQT) po dobách vs. pitch-class set z plánu; skóre shody po taktech | vlastní numpy (už máme `_chroma` v `reference.py`), volitelně librosa |
| Je groove slyšet? | beat/downbeat tracking na mixu vs. mřížka BPM, odchylka a stabilita | **Beat This!** (CPJKU), allin1 (beaty + struktura) |
| Struktura | segmentace mixu (intro/verse/chorus) vs. plán formy | allin1, SongFormer |
| Hlasitost a maskování | LUFS (BS.1770) po stopách a sekcích, maskování po kritických pásmech, mezi stopami | pyloudnorm, vlastní numpy (rozšíření `_masking`) |
| Tonalita | key detection vs. `Song.key` po sekcích | vlastní (Krumhansl už je v `reference.py`) |

Důležitý detail: L1 ověřuje **engine i skladbu zároveň**. Chyby jako
oktávové posuny ve VocalSetu (`voices.spectral_octave`) nebo špatné
naladění VSCO složek by L1 našla automaticky, ne až při poslechu.

**L2 — embeddingy a skóre:** CLAP/MuQ-MuLan skóre sekce vs. textový brief
ze style packu, FAD/KAD proti sadě referencí ze `songs_mp3/` (fadtk) a
Audiobox Aesthetics po sekcích (Production Quality hlídá „chiptune“ zvuk).
Výstup musí být relativní („sekce C má PQ o 0,8 nižší než reference“),
absolutní čísla AI neumí interpretovat.

**L3 — kritik:** audio-LLM dostane sekci, brief a L1/L2 čísla a vrátí
nejvýš tři konkrétní problémy s takty. Výsledky L3 se berou jako hypotézy.
Do `critique.json` se dostanou jen ty, které potvrdí L1 nebo L2, nebo se
označí jako `INFO`. Důvod: HumMusQA (2026) ukazuje, že audio-LLM odpovídají
přes jednomodální zkratky, a v dlouhém audiu mají open modely ~39–57 %
(komunitní, nerecenzovaný benchmark ze září 2026).

### 2.3 „V reálném čase“ — realistická očekávání

Paprika renderuje offline, takže „reálný čas“ znamená **krátkou smyčku**,
ne streamování:

| Vrstva | Latence na 8taktový výřez | HW |
|---|---|---|
| L0 + L1 (bez neuronových sítí) | < 1 s | CPU |
| L1 s Basic Pitch / Beat This | 1–5 s | CPU |
| L2 (CLAP, Audiobox) | 1–3 s | CPU/GPU |
| L3 (30B MoE audio-LLM lokálně / API) | 10–60 s | GPU 24 GB+ nebo API |

Cíl: **jedno kolo agentní smyčky do 30 s na výřez** (`--bars`). To
odpovídá dnešnímu doporučení iterovat výřezy a `show`. Skutečný živý
režim (Magenta RT 2, ~200 ms) je jiný produkt a nepotřebujeme ho.

---

## 3. Architektura enginu — „FL Studio pro AI“

### 3.1 Princip

**Symbolika jako zdroj pravdy, neuronové modely jako nástroje, všechno
měřit.** End-to-end generování celé skladby se nepoužívá jako skladatel,
protože nedává kontrolu nad notami, nejde ověřit proti záměru a není
reprodukovatelné.

Mapování pojmů z FL Studia:

| FL Studio | Paprika dnes | Paprika cíl |
|---|---|---|
| Channel rack | `Track(patch)` | `Track(instrument)` s backendy: `Patch` · `Sampler` · **`Plugin`** (VST3) · **`NeuralVoice`** · **`AudioClip`** (vygenerovaná textura) |
| Piano roll | notace + ASCII `show` | + MIDI import/export, motivy a transformace |
| Playlist / patterny | `Clip` | + `plan.Form` / `Section` (forma, energie, intenzitní vrstvy pro hru) |
| Mixer | `Bus`, `Fx`, automatizace | + mixážní agent (optimalizace na cíle) |
| Pluginy | ruční DSP | + VST3 přes Pedalboard nebo DawDreamer (volitelné) |
| Uši producenta | report | **L0–L3 + `critique.json`** |

### 3.2 Vrstvy

```
 ┌───────────────────────────────────────────────────────────────────┐
 │ AGENT  Claude Code + /paprika skill + paměť voleb (podprojekt 2)  │
 │        smyčka: plán → party → render výřezu → critique → oprava   │
 └──────────────┬───────────────────────────────▲────────────────────┘
        píše .py│ (DSL)                         │ critique.json + report
 ┌──────────────▼──────────────┐   ┌────────────┴────────────────────┐
 │ MODEL  song.py + plan.py    │   │ UŠI  ears/ (L1 verify, L2 embed, │
 │  Form/Section/HarmonyPlan   │   │       L3 critic) — workery       │
 │  Motif, Track, Clip, Bus    │   └────────────▲────────────────────┘
 ├─────────────────────────────┤                │ WAV stopy + MIDI
 │ TEORIE  theory, harmony,    │                │
 │  rules (validátor), corpus  │                │
 ├─────────────────────────────┤                │
 │ RENDER  render.py (beaty →  │────────────────┘
 │  samply jednou), cache      │
 │  backendy nástrojů:         │──► worker: VST3 (Pedalboard/DawDreamer)
 │  Patch | Sampler | Plugin | │──► worker: NeuralVoice (DiffSinger/SoulX/Seed-VC)
 │  NeuralVoice | AudioClip    │──► worker: textury (ACE-Step 1.5 / MusicGen-Stem)
 ├─────────────────────────────┤
 │ EXPORT  WAV, stemy, MIDI,   │
 │  DAWproject, .rpp, smyčky,  │
 │  manifest pro Unity         │
 └─────────────────────────────┘
```

Pravidla pro novou vrstvu (v duchu `CLAUDE.md`):

- **Hranice zůstávají ostré.** `theory`/`harmony` nezná audio, `ears` čte
  jen buffery a datový model (jako `analyze`), `render` nezná teorii.
- **Workery** (`paprika/workers/`) mají vlastní `requirements-*.txt`
  (torch, onnxruntime) a běží v odděleném procesu nebo venv. Rozhraní je
  `stdin/stdout` JSON + cesty k WAV/MIDI. Výsledek se cachuje na disk podle
  `(backend.key(), vstup_hash)`, podobně jako voice cache. Bez workeru hodí
  `ModelMissing` s instalačním příkazem (stejně jako `SamplesMissing`).
- **Všechno v beatech.** Neuronový backend dostane noty v beatech a BPM,
  vrátí audio, render ho umístí. Generované textury se zarovnávají na
  mřížku a ověří v L1 (tónina a tempo), než se pustí do mixu.
- **Reprodukovatelnost:** každý generativní backend bere `seed` a jeho
  výstup je v cache. Druhý render je bit-identický.

### 3.3 Backendy nástrojů

| Backend | Co | Technologie | Poznámka |
|---|---|---|---|
| `Patch` | syntéza | dnešní numpy | beze změny |
| `Sampler` | SFZ | dnešní | + morf samohlásek pro sbor (viz kap. 4) |
| `Plugin` | VST3 nástroje a efekty | **Pedalboard** (Spotify; VST3/AU, offline render s MIDI) nebo **DawDreamer** (JUCE + Faust, automatizace, render grafu) | obě GPL-3.0, proto jen volitelný worker. Free syntezátory: Surge XT (má Python binding `surgepy`), Vital, Dexed, OB-Xd. Headless Linux: jen pluginy s Linux buildem. |
| `NeuralVoice` | zpěv z partitury | DiffSinger / SoulX-Singer (+ Seed-VC/RVC) | kap. 4 |
| `AudioClip(gen=…)` | textury, ambient, riser | ACE-Step 1.5 (LoRA na vlastní styl), MusicGen-Stem (stem k existujícímu kontextu), Stable Audio Open | jen tam, kde syntéza nestačí. Vždy přes L1 kontrolu tóniny a tempa. Licence vah ověřit (MusicGen je NC). |

### 3.4 Interop a ovládání

- **MIDI import/export** (vlastní SMF writer ~150 řádků nebo `mido`).
  Je to předpoklad pro L1 diff, pro symbolické modely a pro DAW.
- **Stemy + DAWproject** (otevřený XML formát Bitwig/PreSonus, čte ho
  Bitwig, Studio One a Cubase) a **REAPER `.rpp`** (textový formát).
  Člověk pak otevře AI skladbu ve skutečném DAW. FL `.flp` je proprietární
  (PyFLP umí jen část), proto jen MIDI + stemy.
- **MCP server `paprika mcp`** s nástroji `render_slice`, `show`, `listen`,
  `critique`, `compare`, `presets`, `instruments`. Paprika pak řídí
  libovolný agent (Claude Desktop, jiné IDE), nejen Claude Code přes CLI.
  Volitelně lze přes `reaper-mcp` renderovat ve skutečném REAPERu s VST.

### 3.5 Agentní smyčka (generuj → změř → oprav)

Podle Libretta a InstructFX2FX:

1. **Brief → plán:** agent vybere style pack a napíše `Form` + `HarmonyPlan`.
   Validátor teorie musí projít bez `WARN`.
2. **Party po sekcích:** agent píše noty proti plánu. Po každé sekci
   `render --bars a:b` + L1. Opravy, dokud nejsou chyby záměru nulové.
3. **Kandidáti:** u kritických míst (refrén, drop) agent napíše 2–3 varianty.
   Vybere se podle L2 skóre a korpusového otisku. Nejlepší kandidát se
   drží v paměti.
4. **Mix:** mixážní agent (3.6) optimalizuje parametry na cíle.
5. **L3 kritika celé skladby** → nejvýš 3 opravy → konec. Smyčka má pevný
   rozpočet kol, aby agent nehonil šum v metrikách.

### 3.6 Mixážní agent

Paprika efekty nejsou diferencovatelné, ale parametrů je málo, takže stačí
black-box optimalizace (`scipy.optimize` Nelder-Mead nebo jednoduché CMA-ES
v numpy) na cíle, které už částečně existují: `metrics.TARGETS` (sub ≤ 12 %,
středy ≥ 20 %, šíře ≥ 10 %), LUFS po stopách, maskování melodie < práh,
crest a LRA podle žánru. LLM (podle LLM2Fx) jen navrhne *výchozí* hodnoty a
které efekty použít. Optimalizér je doladí s měřením v cyklu.
Diferencovatelná cesta (dasp-pytorch, Diff-MST) dává smysl až při vlastním
tréninku.

---

## 4. Hlasy a syntéza zpěvu

### 4.1 Dnešní stav a jeho strop

`voices.py` řeže VocalSet cvičení na tóny (YIN + spektrální oktáva),
PSOLA doplní hlubší tóny a vznikne SFZ (`male_choir`, `soprano`, …).
Strop: jen 4 samohlásky bez souhlásek a textu, vibrato zapečené ve vzorku,
legato bez přirozeného přechodu výšky, malý rozsah na zpěváka a žádná
čeština.

Pozor na rozsah: spec engine v2 rozhodl **„instrumentální, bez zpěvu“**
pro hru ve stylu Arcane. Bezslovný sbor (A) tam patří. Zpěv s textem (B–C)
je změna rozsahu projektu a měl by mít vlastní spec.

### 4.2 Žebřík kvality

| Stupeň | Co | Technologie | Náročnost | Výsledek |
|---|---|---|---|---|
| **A — bezslovný sbor** (doporučeno hned) | morf samohlásek jako automatizace (`vowel` křivka, crossfade vrstev aah/oh/ooh/eh jako dnešní `dyn`), WORLD resyntéza místo PSOLA (pitch shift se zachováním formantů, syntetické vibrato a portamento), sbor = ansámbl více zpěváků s rozptylem časování a ladění | `pyworld` (WORLD vokodér), numpy | bez torch, zapadá do sampleru | cinematic „Aah/Ooh“ sbory a sóla bez textu. Přesně to, co soundtrack typu Arcane používá. |
| **B — partitura → zpěv s textem** | noty + text + tempo → zpěv | **DiffSinger** (openvpi, ekosystém OpenUtau, ONNX export, vlastní jazyky přes slovník), **SoulX-Singer** (Apache 2.0, zero-shot, vstup MIDI partitura nebo F0, zh/en/kantonština), VocalRender (Apache 2.0, score-native, zatím jen čínština, ~9,5 GB), TCSinger 2 (výzkum) | worker s torch/ONNX, GPU vhodné | skutečný zpěv s textem v angličtině. Čeština jen s vlastními daty. |
| **C — barva hlasu** | vyrenderovaný hlas (B nebo i A) převést na cílový timbre, např. operní soprán | **Seed-VC** (zero-shot z krátké reference, režim zpěvu s F0), **RVC/Applio** (model trénovaný na ~10+ min hlasu jednoho zpěváka) | worker, GPU vhodné | operní barva při zachování přesných not z partitury. Referenci lze vzít z VocalSetu (CC BY 4.0). |
| **D — generátor nápadů** | celá píseň s vokálem → separace → přepis → noty do partitury | ACE-Step 1.5 / HeartMuLa / YuE → Mel-RoFormer (separace) → přepis | těžké | inspirace pro melodii a frázování. Audio se přímo nepoužije. |

**Nedoporučuji:** **Bark** je primárně řečový model s nekontrolovatelnou
výškou a rytmem, takže partituru nedodrží. **Diff-SVC / so-vits-svc** jsou
archivované a nahradily je RVC, Seed-VC a DDSP-SVC. **Vlastní vokodér**
dává smysl jen při tréninku vlastního SVS modelu. Jinak stačí vokodér, který
model přináší (NSF-HiFiGAN u DiffSingeru, případně BigVGAN v2 nebo Vocos).

### 4.3 Čeština

Hotový český DiffSinger hlas ani G2P pro DiffSinger se nenašel. Cesta:
(1) česká G2P tabulka (čeština je foneticky pravidelná, stačí pravidla:
spodoba znělosti, ě/di/ti/ni, ř, dvojhlásky), (2) nahrát ~1–2 h zpěvu
jednoho zpěváka s texty, (3) zarovnat (SOFA nebo Montreal Forced Aligner),
(4) dotrénovat DiffSinger, (5) barvu doladit přes Seed-VC. Je to samostatný
projekt na týdny. Do té doby fungují latinské a vymyšlené „sborové“ texty
(Arcane-like) nebo angličtina.

### 4.4 Ověření zpěvu (uši pro hlas)

- **Přesnost výšky:** RMVPE/torchcrepe na vyrenderovaném hlasu vs.
  partitura, medián odchylky v centech na notu, varování nad ±30 c (bez
  vibrata).
- **Srozumitelnost textu:** Whisper → WER proti textu. Varování po frázích.
- **Kvalita:** Audiobox Production Quality hlasu solo, detekce chrapotu
  (subharmonické složky, už je `subharmonic_db`).

---

## 5. Hudební teorie a realističnost

### 5.1 Princip: tvrdá pravidla jako typová kontrola, styl jako statistika

| Typ | Příklady | Kam |
|---|---|---|
| **Tvrdá pravidla** (`WARN`, takt, návrh opravy) | paralelní kvinty a oktávy mezi krajními hlasy, křížení a rozestupy hlasů, nerozvedený citlivý tón, zdvojený citlivý tón, tercie pod ~C3 (zobecnění dnešní `_dissonance`), rozsahy a idiomatika nástrojů (už `_sample_range`), nota melodie mimo akord na těžké době bez přípravy, kadence na konci sekce | nový `rules.py` (podle vzoru `analyze`) |
| **Plán vs. party** | bas mimo základní tón nebo obrat plánovaného akordu, chroma sekce neodpovídá plánu (L1), modulace bez pivotu | `plan.py` + validátor |
| **Statistický otisk stylu** (`INFO`) | hustota not, podíl skoků, rozsah a kontura melodie, poměr akordových a nevlastních tónů, harmonický rytmus, opakování a variace motivů, synkopy | `corpus.py`: percentily z korpusu podle žánru (jako Libretto, 29 dimenzí) |

### 5.2 Nové objekty v modelu

- **`Form` / `Section`** — název, délka, energie 0–1 (řídí hustotu,
  dynamiku a intenzitní vrstvy pro hru), funkce (intro, A, B, drop, outro).
- **`HarmonyPlan`** — akordy v římských číslicích vůči tónině
  (`"i | VI | III | VII"`), harmonický rytmus, kadence a modulace. Generuje
  `theory.chord` voicingy a slouží jako cíl pro L1 chroma.
- **`Motif`** — krátký motiv a transformace (transpozice v rámci stupnice,
  inverze, retrográd, augmentace a diminuce, sekvence, fragmentace).
  Organičnost vzniká z **vývoje motivu**, ne z nových not v každém taktu.
  Report pak umí říct „refrén nepoužívá motiv z verše“.
- **Voicing solver** — nejbližší vedení hlasů mezi akordy jako malá
  optimalizační úloha (beam search v numpy, volitelně OR-Tools CP-SAT pro
  SATB). Návrh pro AI, ne autor: AI ho volá jako `pad.voicings(plan)`.

### 5.3 Rytmus a výraz (proč to zní „roboticky“)

- **Groove šablony** místo náhodného `humanize`: mikrotiming a velocity
  podle role nástroje a metrické váhy (Groove MIDI Dataset, extrahované
  šablony jako data, ne model).
- **Frázová dynamika:** oblouk crescenda a decrescenda na frázi, rubato
  před kadencí, artikulace podle délky noty (legato, staccato) volí
  `Sampler` artikulaci automaticky.
- **Výrazové modely** (volitelný worker): Pianist Transformer (výstup
  převádí na tempovou křivku zpět do mřížky), RenderBox (text + MIDI →
  výrazový přednes), CFE+P. Pro orchestr open model zatím chybí, takže
  pravidla výše jsou hlavní cesta.

### 5.4 Symbolické modely jako „session hráči“

Neuronové symbolické modely mohou **navrhovat** výplně, protimelodie a bicí
breaky, vždy uvnitř plánu a vždy přes validátor. Agent je vybírá, ne
přebírá: Anticipatory Music Transformer (infilling, Apache 2.0), MIDI-LLM
(Llama 3.2 1B, text → multitrack MIDI, ISMIR 2026), text2midi, Aria
(piano). Používat až po MIDI exportu a rules. Hlavní skladatel zůstává LLM
agent, jak říká `CLAUDE.md` („not a music generator“).

### 5.5 Data pro otisky a pravidla

Chordonomicon (~666 tis. akordových průběhů), Hooktheory TheoryTab
(melodie + harmonie s funkcemi), PDMX (public-domain MusicXML), Lakh MIDI,
GigaMIDI, Aria-MIDI, Groove MIDI, MAESTRO. Pro testy jako orákulum
`music21` (rozbor římských číslic, vedení hlasů).

---

## 6. Doporučený tech stack

| Vrstva | Doporučení | Alternativa | Licence (ověřit) | Závislost |
|---|---|---|---|---|
| Jádro, render, DSP | numpy + scipy (beze změny) | — | — | core |
| MIDI | vlastní SMF writer/reader | `mido`, `symusic`, `pretty_midi` | MIT | core / volitelná |
| Teorie, otisk | vlastní `harmony.py`, `rules.py`, `corpus.py` | `music21` (testy), OR-Tools CP-SAT | BSD / Apache | core / dev |
| L1 přepis | Basic Pitch (ONNX) | MuScriptor (NC váhy), YourMT3+, Transkun (piano) | Apache 2.0 | worker `ears` |
| L1 beat a struktura | Beat This! | allin1, madmom | MIT | worker `ears` |
| L1 pitch zpěvu | RMVPE / torchcrepe | PESTO, CREPE | MIT | worker `ears` |
| L1 hlasitost | pyloudnorm | vlastní BS.1770 | MIT | core-lite |
| L2 embedding | MuQ-MuLan nebo LAION-CLAP | CLaMP 3 (i MIDI) | MuQ váhy NC, CLAP CC0, CLaMP 3 MIT | worker `ears` |
| L2 kvalita | Audiobox Aesthetics | SongEval (NC), MuseCritic | CC BY 4.0 | worker `ears` |
| L2 vzdálenost k referencím | fadtk (FAD, KAD) | — | MIT | worker `ears` |
| L3 kritik | Qwen3-Omni-30B-A3B (Instruct/Captioner) | Audio Flamingo Next, Music Flamingo (NC), Gemini API | Apache 2.0 | worker `critic` (GPU/API) |
| Separace (pro reference, stupeň D) | Mel-/BS-RoFormer přes `audio-separator` | Demucs v4 | MIT | worker |
| VST3 hosting | Pedalboard | DawDreamer | GPL-3.0 | worker `plugins` |
| Zpěv z partitury | DiffSinger (openvpi) | SoulX-Singer, VocalRender | Apache 2.0 | worker `voice` |
| Barva hlasu | Seed-VC | RVC / Applio | GPL-3.0? / MIT | worker `voice` |
| WORLD resyntéza | pyworld | — | MIT | core-lite |
| Textury a stemy | ACE-Step 1.5 | MusicGen-Stem, Stable Audio Open | MIT/Apache? / NC / komunitní | worker `gen` |
| Interop | DAWproject, REAPER `.rpp`, MCP server | `reaper-mcp`, `ableton-mcp` | MIT | core |

Licenční pravidlo pro hru (komerční výstup): **nekomerční (NC) modely jen
v dev nástrojích, nikdy v pipeline, jejíž výstup jde do hry** (týká se
Music Flamingo, SongEval, MuScriptor vah, MusicGen vah). GPL a AGPL
knihovny (Pedalboard, DawDreamer, Seed-VC, essentia) držet ve workerech
mimo jádro. Repo zatím nemá `LICENSE` — rozhodnout dřív, než se cokoli
z toho začne distribuovat.

---

## 7. Roadmapa

Pořadí je dané poměrem přínos/cena. Fáze 1–2 nepotřebují GPU.

| Fáze | Obsah | Hotovo, když… |
|---|---|---|
| **1 — Kompilátor teorie** (bez nových závislostí) | `plan.py` (Form, Section, HarmonyPlan), `rules.py` (vedení hlasů, nevlastní tóny, kadence, registry), `Motif`, `render --json` (strojový report), MIDI export | `v2_showcase` projde s 0 `WARN`; testy pravidel na učebnicových příkladech (music21 jako orákulum) |
| **2 — Uši I: ověření záměru** | `paprika listen`: přepis stop (Basic Pitch), diff partitura ↔ zvuk, chroma vs. plán, Beat This, LUFS po stopách; `critique.json` | ≥ 95 % not každé samplované stopy se přepíše s výškou ±50 c; uměle rozladěný vzorek v testu L1 najde |
| **3 — Sbor A** | morf samohlásek, WORLD resyntéza, ansámbl zpěváků | sbor projde L1 (ladění) a v Audiobox PQ není horší než VSCO smyčce |
| **4 — Uši II: embeddingy a kritik** | L2 (MuQ/CLAP, Audiobox, FAD vs. `songs_mp3/`), L3 worker (Qwen3-Omni), `paprika critique` | FAD k referencím klesá napříč verzemi skladby; L3 tvrzení se do `WARN` dostanou jen s potvrzením L1/L2 |
| **5 — Agent a style packy** (podprojekt 2) | `/paprika` skill se smyčkou generuj → změř → oprav, rozpočet kol, paměť voleb, mixážní optimalizér | prompt → skladba bez ručního zásahu. 3 skladby ze stejného packu se liší v otisku. |
| **6 — Export do hry** (podprojekt 3) | stemy, smyčky, stingery, intenzitní vrstvy z `Section.energy`, DAWproject a `.rpp`, manifest pro Unity | Unity přehraje smyčku bez švu, intenzita se přepíná na hranici taktu |
| **7 — Pluginy a neuronové nástroje** | `Plugin` (Pedalboard), `AudioClip(gen=…)` (ACE-Step), MCP server | render s VST3 je reprodukovatelný (cache), vygenerovaná textura projde L1 kontrolou tóniny a tempa |
| **8 — Zpěv s textem** (vlastní spec) | DiffSinger nebo SoulX worker, Seed-VC, Whisper WER, česká G2P | anglický text s WER < 15 %, výška ±30 c; čeština až s vlastním datasetem |

---

## 8. Rizika a otevřená rozhodnutí

1. **Rozsah:** zpěv s textem je v rozporu s rozhodnutím v2 („bez zpěvu“).
   Doporučení: A (sbor) teď, B–C jako samostatný podprojekt po exportu do hry.
2. **Závislosti:** torch a modely v řádu GB nesmí rozbít `pip install -r
   requirements.txt`. Řešení jsou workery v oddělených venv a stejná filozofie
   jako u vzorků (mimo repo, `PAPRIKA_MODELS`).
3. **Licence:** NC modely mimo herní pipeline, GPL ve workerech, chybí
   `LICENSE` repozitáře.
4. **Goodhart:** optimalizace na jednu metriku (CLAP, Audiobox) snižuje
   diverzitu a vede k „průměrné“ hudbě. Proto víc nezávislých metrik,
   rubriky, korpusový otisk jako *pásmo* (ne cíl) a pevný rozpočet kol.
5. **Spolehlivost audio-LLM:** používat jako hypotézy, potvrzovat měřením.
6. **Windows:** vývoj běží i na Windows (cp125x konzole, `%LOCALAPPDATA%`).
   Workery s torch a VST3 hosting je tam potřeba otestovat zvlášť, mmap
   a otevřené soubory platí i pro modely.
7. **Výpočet:** L3 lokálně potřebuje GPU s ~24 GB+ nebo API. Fáze 1–3
   běží na CPU.

---

## Zdroje

Trh
- [Variety — Suno raises $400M at $5.4B valuation (2026)](https://variety.com/2026/digital/news/ai-music-suno-funding-round-400-million-5-4-billion-valuation-1236765727/)
- [Music Ally — Suno primer: valuation, lawsuits (2026-07)](https://musically.com/2026/07/23/suno-ai-music-generator-primer-data-breach-valuation-lawsuits/)
- [Music Business Worldwide — Suno: three licensing deals, $775M+ raised](https://www.musicbusinessworldwide.com/with-a-new-cfo-three-licensing-deals-and-775m-raised-whats-stopping-5-4b-valued-suno-from-an-ipo/)
- [ValueAddVC — Suno $300M ARR](https://valueaddvc.com/blog/suno-ai-valuation-2026-5-4b-round-300m-arr-and-the-business-model-behind-ai-music)
- [MarketScreener — WMG settles with Suno](https://uk.marketscreener.com/news/warner-music-group-settles-copyright-case-with-suno-for-licensed-ai-music-ce7d5ed2db8bff22)
- [Dubspot — Suno Studio 2.0 (MIDI, synth, stems)](https://blog.dubspot.com/suno-studio-2-0)
- [RouteNote — Suno Studio 2.0 launch](https://routenote.com/radar/suno-studio-2-0-launches-with-midi-support-ai-chat-and-advanced-stem-separation/)
- [MusicTech — Suno Studio review](https://musictech.com/reviews/digital-audio-workstations/suno-studio-review/)
- [ChatForest — MCP servers for music production](https://chatforest.com/reviews/music-audio-production-mcp-servers/)
- [Sonic Field — The MCP turn in audio](https://sonicfield.org/ai-agents-enter-the-studio-the-mcp-turn-in-audio)
- [flstudio-mcp (Glama)](https://glama.ai/mcp/servers/rosasynthesiz/flstudio-mcp)

Uši, hodnocení, audio-LLM
- [Audio Flamingo Next (2026)](https://arxiv.org/html/2604.10905v1)
- [Music Flamingo — NVIDIA](https://research.nvidia.com/labs/adlr/MF) · [HF karta](https://huggingface.co/nvidia/music-flamingo-hf)
- [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) · [Simon Willison o Qwen3-Omni (Apache 2.0)](https://simonwillison.net/2025/Sep/22/qwen)
- [HumMusQA — music QA benchmark (2026)](https://arxiv.org/pdf/2603.27877)
- [MusTBench — temporal grounding v music LLM](https://arxiv.org/pdf/2605.29300)
- [Long-audio understanding (HF, 2026-09)](https://huggingface.co/datasets/SEBK4C/long-audio-understanding-sept-2026)
- [CMI-Bench](https://arxiv.org/pdf/2506.12285)
- [SongEval](https://arxiv.org/pdf/2505.10793) · [README/licence](https://raw.githubusercontent.com/ASLP-lab/SongEval/main/README.md)
- [Audiobox Aesthetics (PyPI)](https://pypistats.org/packages/audiobox-aesthetics) · [přehled skóre](https://www.emergentmind.com/topics/audio-aesthetic-scores)
- [MuseCritic — song rewards z textové kritiky (2026)](https://arxiv.org/abs/2608.11755)
- [Rubric-Based Optimization for Text-to-Music (2026)](https://arxiv.org/html/2610.03589)
- [TuneJury](https://arxiv.org/pdf/2606.17006)
- [CLaMP 3](https://arxiv.org/pdf/2502.10362) · [MUUNRiver-Bench (MuQ-MuLan vs. CLaMP 3)](https://arxiv.org/pdf/2609.16090)
- [MuScriptor — open multi-instrument transcription](https://www.alphaxiv.org/abs/2607.08168v2) · [HF karta (licence)](https://huggingface.co/muscriptor/muscriptor-large) · [AI Weekly](https://aiweekly.co/alerts/kyutai-releases-muscriptor-for-multi-instrument-transcription)
- [Mel-Band RoFormer](https://arxiv.org/pdf/2310.01809)
- [music-transcription na GitHubu (omnizart, Transkun, GAME)](https://github.com/topics/music-transcription)

Agenti, mix, architektura
- [Libretto: Giving LLM Agents a Sense of Musical Structure (2026)](https://arxiv.org/pdf/2606.22708) · [HF papers](https://huggingface.co/papers/2606.22708)
- [ComposerX](https://ismir2024program.ismir.net/poster_237.html)
- [Helping Music Co-Creation Agents „Listen“ Well (2026)](https://arxiv.org/pdf/2608.04378)
- [Audio-Oscar (2026)](https://arxiv.org/pdf/2606.07397)
- [DAWZY](https://arxiv.org/pdf/2512.03289)
- [LLM2Fx (Sony)](https://ai.sony/publications/can-large-language-models-predict-audio-effects-parameters-from-natural-language) · [LLM2Fx-Tools](https://arxiv.org/pdf/2512.01559) · [InstructFX2FX, DAFx26](https://www.dafx.de/paper-archive/2026/papers/DAFx26_demo_74.pdf)
- [MixAssist](https://arxiv.org/pdf/2507.06329)
- [DawDreamer (paper)](https://arxiv.org/abs/2111.09931) · [PyPI](https://pypi.org/project/dawdreamer/0.6.9/)
- [Agentic Artifact Creation — survey (2026)](https://arxiv.org/pdf/2608.28122)

Generování (audio i symbolické)
- [ACE-Step 1.5](https://arxiv.org/abs/2602.00744) · [projekt](https://ace-step.github.io/ace-step-v1.5.github.io/)
- [HeartMuLa / heartlib](https://github.com/HeartMuLa/heartlib)
- [LeVo 2](https://arxiv.org/pdf/2606.30642) · [SongBench](https://arxiv.org/pdf/2604.25937) · [WanSong](https://arxiv.org/pdf/2607.14749)
- [Stemphonic (ICASSP 2026)](https://www.alphaxiv.org/abs/2602.09891) · [MusicGen-Stem](https://arxiv.org/html/2501.01757v2) · [StemGen](https://arxiv.org/abs/2312.08723)
- [Magenta RealTime 2](https://mer.vin/2026/06/magenta-realtime-2-open-local-live-music-model-with-midi-and-200ms-latency/)
- [MIDI-LLM](https://arxiv.org/html/2511.03942) · [MI-MIDI (interpretabilita text2midi vs. MIDI-LLM)](https://arxiv.org/abs/2608.06638) · [MIDI-LLaMA](https://arxiv.org/pdf/2601.21740)
- [CSyMR-Bench (LLM + music21 nástroje)](https://arxiv.org/html/2601.11556v1) · [ABC-Eval](https://arxiv.org/html/2509.23350v1)
- [Pianist Transformer](https://arxiv.org/pdf/2512.02652) · [RenderBox](https://arxiv.org/pdf/2502.07711)

Zpěv
- [SoulX-Singer](https://arxiv.org/abs/2602.07803v1) · [HF karta](https://huggingface.co/williamchangtw/SoulX-Singer/blob/main/README.md)
- [VocalRender](https://arxiv.org/pdf/2607.27768) · [GitHub](https://github.com/pymaster17/VocalRender)
- [TCSinger 2](https://arxiv.org/pdf/2505.14910) · [YingMusic-Singer](https://arxiv.org/pdf/2512.04779) · [CoMelSinger](https://arxiv.org/pdf/2509.19883)
- [Seed-VC](https://arxiv.org/html/2411.09943v1) · [R2-SVC](https://arxiv.org/pdf/2510.20677)
- [OpenUtau — DiffSinger singers](https://github.com/openutau/OpenUtau/wiki/Singer-Settings-and-Requirements) · [DiffSinger OpenUtau fork](https://github.com/Metal-Studioo/DiffSingerOpenUtau)
- [NNSVS](https://arxiv.org/pdf/2210.15987) · [Synthetic Singers — přehled SVS (2026)](https://arxiv.org/pdf/2601.13910)
