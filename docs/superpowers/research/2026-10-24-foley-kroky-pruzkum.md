# Realistické kroky a foley v paprika — chůze, běh, povrchy

**Datum:** 2026-10-24
**Typ:** průzkum + návrh implementace (ne schválený spec)
**Stav:** fáze F1 implementována (2026-10-24): `paprika/gait.py`
(kadence z rychlosti, 1/f časování od mřížky, korelované parametry kroků,
stálé rozdíly nohou, fyzika rychlosti a váhy), `HumanTrack.steps(pace=,
speed=, weight=, jitter=)`, nové parametry `human._step`, testy
`tests/test_gait.py`. Fáze F2 implementována (2026-10-24): řádek `KROKY`
v reportu a v `check`, `gait.irregularity`, `metrics.step_stats`,
pravidla metronomu, smyčky, stejně silných kroků a kopií (sekce 6).
Fáze F3 implementována (2026-10-24): `paprika/foley.py` — krok z vrstev
(dopad Hertzovým pulzem, tření, úlomky řízené reakční silou, bláto, sníh,
výbava), 8 parametrických povrchů, 4 boty, přechody povrchů,
`steps(shoe=, gear=, cloth=, blend=)`, navazování chůze, ukázka
`songs/kroky.py` (sekce 7). F4–F5 nejsou začaté.

Zdroje níž jsou citované z paměti, ne ověřené hledáním na webu v té
session. Před citací v jiném dokumentu je ověř (název, rok, licenci).

## 1. Proč kroky zněly „z kódu“

Stav před F1 (`human._step`, `HumanTrack.steps`):

- **Jedna šablona.** Sinusové dupnutí + šumový klik, po 35–70 ms totéž
  oslabené (špička). Mění se jen ±5–10 % frekvence a fáze šumu; tvar
  a spektrum jsou pořád stejné. Každý krok je normalizovaný na špičku 0.8.
- **Nezávislá náhoda.** Každý krok losuje z vlastního `_rng(29, variant)`.
  Člověk ale nechodí náhodně: sousední kroky se podobají, jedna noha je
  trvale těžší.
- **Metronom.** `at += every` — intervaly přesně stejné. Lidská chůze má
  variační koeficient intervalu ~2–4 % s dlouhou pamětí (1/f).
- **Bez rychlosti a váhy.** `vel` mění jen hlasitost; reálně rychlost mění
  kontakt, odstup paty a špičky, spektrum i podíl úlomků.
- **Chybí vrstvy:** oblečení svázané s fází kroku, výbava (klíče, zbroj),
  boty, úlomky po dosednutí (kromě gravel/dirt), povrchy tráva, sníh, bláto.
- **Žádná zpětná vazba.** Nic neměří, jestli se kroky opakují — a AI je
  neslyší.

## 2. Co dělají herní enginy

| Technika (Wwise / FMOD) | Co dělá | V paprice |
|---|---|---|
| Switch / Blend container, parametr `Surface` | povrch z raycastu vybírá sadu | `surface=`; plynulý přechod je F3 |
| Random container „avoid last N“ | nikdy dvakrát stejný sample | u syntézy místo toho korelace (F1) |
| Randomizace pitch ±1–3 st, gain ±2 dB, LPF | levná základní variace | F1: korelovaně, přes fyzikální parametry |
| RTPC (rychlost -> pitch, LPF, mix vrstev) | parametry hry řídí zvuk spojitě | F1: `speed` jako křivka, `weight` |
| Vrstvy foot × shoe × surface × cloth × gear | každá vrstva má vlastní variaci | F3 |
| Animační eventy (heel / toe) | zvuk sedí na fázi kroku | F1 plánuje kroky; fáze pro vrstvy je F3 |
| Wwise SoundSeed Impact | modální resyntéza dopadu ze sampleu | F4 |
| FMOD scatterer, granulární instrumenty | úlomky jako oblak zrn | Poissonova zrna (už v `_cloth`) |

Akademické modely: Cook, *Modeling Bill's Gait* (2002) a PhISEM / STK
`Shakers` („Crunch“); Fontana & Bresin (2003) crumpling pro štěrk a sníh;
Sound Design Toolkit (Delle Monache, Polotti, Rocchesso — impact, friction,
rolling, crumpling, walking; open source, ověřit licenci); Turchet (2016)
syntéza kroků s typem boty a stylem chůze; Farnell, *Designing Sound*
(2010). Biomechanika: reakční síla podložky (GRF) má při chůzi dva vrcholy
(~1.1–1.3× tělesné váhy), při běhu jeden (2–3×) a kontakt ~0.15–0.3 s;
variabilita intervalu kroku má dlouhou paměť (Hausdorff et al. 1995, DFA
α ≈ 0.8–1).

## 3. Povrchy a AI

- **Parametrický povrch** místo tabulky: tvrdost (doba kontaktu, cutoff),
  modální banka `(f, τ, amp)`, crumpling (četnost, velikost zrn, práh
  energie), vlhkost (bláto: sání, tmavé tělo), hloubka (sníh, písek:
  pomalý nástup, vrzání 2–4 kHz). Povrchy jdou interpolovat.
- **Sample -> model:** modální odhad (matrix pencil / ESPRIT / Prony) z
  prvních 100–300 ms reálného kroku; přenos spektrální obálky (LPC,
  kepstrum) z povrchu X na excitaci; řezání dlouhých nahrávek chůze
  detekcí onsetů do `corpus.py`.
- **Generativní AI** (AudioLDM 2, Stable Audio Open — už v
  `tools/human_worker.py`, MMAudio, Diff-Foley, FoleyCrafter; DCASE 2023
  Task 7 měl třídu *Footstep*): dávají 5–10s textury, ne řízené jednotlivé
  kroky. Hodí se jako zásobárna materiálu pro vrstvu úlomků, ne na rytmus.
  CLAP (LAION / Microsoft) jako zero-shot síto v `corpus.py` vedle PANNs.
  Nic z toho tu zatím neběželo.

## 4. Fáze

| Fáze | Obsah | Stav |
|---|---|---|
| F1 | plánovač chůze: kadence, 1/f jitter, OU korelace, nohy, váha a rychlost | hotovo |
| F2 | metriky opakování kroků v reportu (podobnost párů, nepravidelnost intervalů, síla, těžiště, rozdíl nohou) | hotovo |
| F3 | vrstvy (Hertzův dopad, GRF, úlomky řízené dGRF/dt, odval, oblečení na fázi kroku, výbava), povrchy grass / snow / mud / metal, přechody povrchů | hotovo |
| F4 (další) | sample -> model z uživatelových nahrávek (modální odhad, přenos obálky) | — |
| F5 | AI textury přes `human_worker.py` + síto s CLAP | — |

## 5. F1 — co je implementované

`paprika/gait.py` je čistý model (nezná audio), audio dělá `human._step`.

**Časování.** Mřížka jde po intervalech `1 / cadence(speed(beat))`
(`cadence(v) = 1.2 + 0.5 v` kroku/s: 1.9 při 1.4 m/s, 2.95 při 3.5 m/s),
nebo pevně `every` beatů. Odchylka kroku `k` od mřížky je
`e_k = c · x_k · T_k`, kde `x` je šum se spektrem 1/f^0.8 (`pink`) a `c`
je zvolené tak, aby směrodatná odchylka `diff(e) / T` byla přesně `jitter`
(výchozí 0.025). Odchylka je od mřížky, ne kumulativní: kumulovaný 1/f
interval by po 100 krocích ujel o ~2 kroky a rozešel se s hudbou. Ořez
`MAX_SHIFT` = 30 % intervalu. Noha navíc nese stálý posun (`Walker`,
σ 0.8 % intervalu).

**Korelace.** Pět parametrů kroku (síla, odstup pata–špička, poměr síly
špičky, jas kliku, výška dupnutí) je AR(1) přes kroky:
`z_k = 0.6 z_{k-1} + 0.8 ε_k`, násobek `exp(σ z)` se σ 0.12 / 0.15 / 0.12 /
0.06 / 0.025. Jas je z poloviny svázaný se silou (silnější došlap je
jasnější).

**Nohy.** `Walker(seed)` vylosuje pro levou `+d/2` a pravou `−d/2`
logaritmického rozdílu síly (σ 0.06), odstupu (0.08), jasu (0.04), výšky
(0.015) a časování. Seed je seed stopy, takže všechna volání `steps()` na
jedné stopě jsou jeden člověk; šum se mezi voláními liší.

**Fyzika** (`physics(weight, speed)`):

| | chůze (< 2.2 m/s) | běh (≥ 2.2 m/s) |
|---|---|---|
| kontakt | `max(0.3, 0.7 − 0.1 v)` s | `max(0.15, 0.34 − 0.05 (v − 2.2))` s |
| pata -> špička | 12 % kontaktu | 8 % kontaktu |
| síla špičky | 0.6 | 0.35 (jeden dopad) |
| GRF / váha | `1.1 + 0.15 (v − 1.4)` | `2.3 + 0.15 (v − 3.5)` |

Hlasitost `((kg / 75) · GRF / GRF_ref)^0.6`, výška dupnutí `(75 / kg)^0.25`,
jas kliku `1 + 0.12 (v − 1.4)` (0.85–1.6), doznění dupnutí
`(kg/75)^0.3 · (kontakt / kontakt_ref)^0.3`. `STEP_GAIN` je pevný na povrch
(medián špičky referenčního chodce 0.45, změřený na 30–60 variantách), běh
3.2 m/s dává ~0.7 a strop `CEILING` 0.95 zůstává rezervou.

**Kompatibilita.** Bez `pace` / `speed` / `weight` / `jitter` je hit
`(surface, foot, variant)` a zvuk bit-identický (kontrolní součty v
`test_legacy_step_unchanged`). Hit chůze má
`(surface, foot, weight, speed, level, gap, ratio, bright, pitch, variant)`
s floaty zaokrouhlenými na 4 místa (stabilní klíč cache); `variant` je
neomezené celé číslo, aby se šum kliku nikdy neopakoval po 8 krocích.

**Ověřeno testy** (`tests/test_gait.py`): CV intervalu = `jitter` ±0.006,
kroky po 400 beatech stále u mřížky, 1/f spektrum, lag-1 korelace síly
0.25–0.9 a žádné dva stejné hity, stálý poměr nohou nezávislý na šumu,
rozptyl rozdílu nohou mezi chodci = `BIAS`, zrychlení zhustí kroky, těžší
a rychlejší = hlasitější, běh jasnější než pomalá chůze, meze a chybové
hlášky, report nehlásí „všechny velocity stejné“.

**Neověřeno:** nikdo to neslyšel. Konstanty fyziky jsou orientační
z literatury, ne zkalibrované na nahrávkách. Až budou k dispozici
reálné kroky (F4 nebo uživatelovy nahrávky), stejně jako u dechů
porovnat těžiště, poměr pata/špička a CV intervalů a konstanty upravit.

## 6. F2 — kontrola kroků bez poslechu

AI kroky neslyší, takže F1 bez měření nejde ověřit ani ladit. F2 přidává
do reportu řádek `KROKY` a čtyři pravidla.

**Nepravidelnost** (`gait.irregularity`, bez audia): kroky se rozdělí na
souvislé chůze (`walks`: zastavení = interval delší než 1.8× medián),
interval se vydělí klouzavým mediánem 7 kroků a vezme se rozptyl. Pro
stálé tempo je to variační koeficient (`jitter` 0.01 / 0.025 / 0.05 dá
0.012 / 0.026 / 0.051), rampa 1 -> 4 m/s bez jitteru dá 0.4 % místo 18 %
prostého CV. Chůze s `jitter=0` má ~1 % ze stálého časového posunu nohou
(`Walker`), pevné `steps()` přesně 0.

**Podobnost** (`metrics.step_stats`, z audia): okno 200 ms od 5 ms před
nástupem každého kroku (nástupy z modelu; u nástupu na začátku bufferu
doplněné nulami, jinak se kopie posunou o 5 ms a neshodují). Podobnost
dvou kroků je maximum normované křížové korelace v posunu ±2 ms **jen nad
1 kHz**. Přes celé pásmo to nešlo: tlumený sinus dupnutí (hlína 70 Hz,
amplituda 0.6) je u každého kroku skoro stejný a unikátní běh po hlíně
vycházel na 0.994 — jako kopie. Nad 1 kHz je jen náhodná část (klik,
zrna):

| 48–64 kroků, mikrofon zapnutý | podobnost max | kopie |
|---|---|---|
| pevné `steps()`, všechny 4 povrchy | 1.00 | 40/48 (= 48 − 8 variant) |
| `pace="walk"`, 4 povrchy | 0.21–0.42 | 0 |
| `pace="run"`, 4 povrchy | 0.22–0.48 | 0 |

Kopie = podobnost ≥ 0.98 s některým dřívějším krokem.

**Síla** (rozptyl špiček v dB): pevné kroky 0.3–0.4 dB (kámen, dřevo; štěrk
a hlína ~1.1 dB díky náhodným zrnům), chůze 1.1–1.7 dB.

**Pravidla** (`analyze.py`, všechna jen INFO kromě posledního):

- metronom — 8+ kroků v souvislé chůzi, nepravidelnost < 0.5 % (model);
- smyčka — krok má stejné parametry jako některý dřívější, kromě
  bezprostředně předchozího, to hlásí existující WARN „kulometný efekt“
  (model);
- stejně silné kroky — 8+ kroků, rozptyl špiček < 0.5 dB (audio);
- kopie, kterou model nevysvětlí — dva kroky s *různými* parametry zní
  jako tentýž vzorek (WARN; hlídá generátor, v normálních songách mlčí).

Na `songs/dech.py` (12 pevných kroků s humanize) hlásí jen smyčku 4/12
a metronom ne: humanize dá nepravidelnost 0.7 %.

**Co se neosvědčilo:** rozdíl barvy (24 mel pásem) mezi sousedy oproti
všem párům měl ukázat „paměť“ korelovaných kroků. Rozdíl je ale malý
(chůze 0.96–0.98, pevné kroky 1.0–1.08), protože OU proces mění jen
několik násobků kolem 1, zatímco šum kliku je pokaždé nový. Metrika
zůstala v `step_stats` (`shape_adj`, `shape_all`), pravidlo na ní není.
Korelace z F1 je tedy slyšet hlavně v síle a odstupu pata–špička, ne
v barvě; silnější barevná paměť přijde s vrstvami a povrchy v F3.

## 7. F3 — krok z vrstev

`paprika/foley.py`. Pevné kroky (`human.SURFACES`, 3 parametry) zůstaly
bit-identické; chůze z F1 dostala nový zvuk — F1 tabulku a `STEP_GAIN`
nahradily parametrické povrchy (F1 byla jen na této větvi, nevydaná).

**Řídicí průběh.** Reakční síla podložky `grf(t, kontakt, běh)`: chůze
= dva gaussovské vrcholy (25 % a 75 % kontaktu) + údolí, okno
`sin(π t / T)^0.5`; běh = `sin^1.5`; obojí s rázem paty (12 ms, šířka
6 ms, 0.5) a normalizované na 1. Úlomky berou kladnou derivaci
normalizovanou na 1 a samotnou sílu.

**Dopad.** Půlsinusový pulz s jednotkovou plochou trvá
`τ = bota_ms · (1 + 8 (1 − tvrdost)) · (v / 1.4)^−0.2`
(boot 0.5, sneaker 1.2, heels 0.2, bare 2.5 ms; tvrdost kámen a kov 1,
dřevo 0.85, štěrk 0.6, hlína 0.4, sníh 0.2, tráva 0.15, bláto 0.05).
Místo konvoluce stačí analytické spektrum pulzu
`|cos(π f τ) / (1 − (2 f τ)²)|` (limita π/4 na f = 1/2τ): každá
rezonance (tlumený sinus) dostane tenhle zisk na své frekvenci a začne
o τ/2 později. Rezonance podložky mají u každého kroku amplitudu
lognormal(0, 0.35) a frekvenci ±3 % (jiné místo dopadu), dupnutí nohy
±5 % a výšku podle váhy. Špička dopadne po odstupu z F1 s 1.5× delším
pulzem.

*Klik hrany.* S celým pulzem neměl boot na kameni žádný klik: zisk na
3.2 kHz při τ = 0.5 ms je 0.033 (−30 dB), 95 % energie kroku bylo pod
200 Hz. Klik teď dělá hrana podrážky s kontaktem `CLICK_EDGE` = 0.2 τ
(boot na kameni 0.1 ms -> zisk 0.92; bosá noha 0.5 ms -> 0.03, tichá).
Současně kleslo dupnutí tvrdých podlah (kámen 0.9 -> 0.5, dřevo 0.8 ->
0.6, kov 0.6 -> 0.4). Kámen: těžiště 340 -> 850 Hz, nad 4 kHz 3 -> 9 %.

**Tření.** Smyk paty `grip · (v/1.4)^0.5`, τ 12 ms; odval špičky
`0.6 · scuff · min(v/1.4, 2)^1.5` kolem konce kontaktu. Původně lineární
smyk a odval ~v^1.5 bez meze přerostly při běhu dopad 1.6× (kámen, běh
3.2 m/s: tření 1.62 vs dopad 1.03) a běh narážel na strop.

**Úlomky.** Bernoulliho zrna s četností `rate · dF⁺ + sustain · F`
(+ 8 % `rate` dosedání po odlepení s τ = tail/3), amplituda
`lognormal(0, 0.6) · √(0.3 + F)`, dvě pásmová propust (hlavní pásmo
× jas, tělo). Sníh: vrzání = zrna při zatěžování přes úzkou propust
1.3 kHz (Q 6). Bláto: mlasknutí (šum 500 Hz, τ 30 ms), sání po odlepení
(sinus klesající 420–600 -> 130–190 Hz za 35–60 ms), Poisson(3) bublinek
400–1500 Hz. Výbava: Poisson(srážek) po dopadu se zpožděním 10–40 ms
+ polovina při odrazu, každá srážka dva mody ze sady (klíče 2.9–8.8 kHz,
řetěz 3.5–6.6 kHz, zbroj 0.65–3.3 kHz), síla ~ úroveň · v^1.2, ve středu.
Oblečení: hity `cloth` (existující generátor) od 10 % intervalu
po kroku na 80 % intervalu, velocity · (0.45 + 0.2 v/1.4).

**Kalibrace.** `SURFACE_GAIN` = cíl / medián špičky 40 variant
referenčního chodce (75 kg, 1.4 m/s, boot, bez výbavy, 44.1 kHz). Cíle
`SURFACE_TARGET`: kámen a dřevo 0.45, kov 0.5, štěrk 0.45, hlína 0.35,
tráva 0.3, sníh a bláto 0.4. Výsledek (medián):

| Povrch | Špička | Běh 3.2 | 110 kg běh 3.5 | Těžiště | > 4 kHz | < 200 Hz | −30 dB |
|---|---|---|---|---|---|---|---|
| stone | 0.45 | 0.85 | 0.95 | 848 Hz | 9 % | 85 % | 0.57 s |
| wood | 0.43 | 0.72 | 0.93 | 265 Hz | 0 % | 69 % | 0.56 s |
| metal | 0.49 | 0.80 | 0.95 | 767 Hz | 1 % | 3 % | 0.76 s |
| gravel | 0.46 | 0.70 | 0.94 | 1037 Hz | 10 % | 78 % | 0.56 s |
| dirt | 0.34 | 0.57 | 0.74 | 245 Hz | 1 % | 93 % | 0.56 s |
| grass | 0.32 | 0.46 | 0.56 | 1562 Hz | 17 % | 79 % | 0.53 s |
| snow | 0.46 | 0.62 | 0.68 | 2708 Hz | 22 % | 25 % | 0.53 s |
| mud | 0.40 | 0.67 | 0.86 | 127 Hz | 0 % | 87 % | 0.70 s |

Strop 0.95 zasáhne jen těžký sprinter na tvrdé podlaze (pak se ten krok
stáhne, hlasitost dál neroste). Boty na dřevě: heels 0.58, boot 0.43,
sneaker 0.22, bare 0.11. Výbava na hlíně: špička 0.40–0.42 (bez ní 0.34),
klíče zvednou podíl nad 2.5 kHz ~4×.

**F2 na F3** (8 taktů chůze, mikrofon zapnutý): žádný povrch nemá kopii;
podobnost max 0.11–0.42, kov 0.94 (pevné rezonance plechu, mění se jen
jejich amplitudy — skutečný rošt zní taky pokaždé stejně vysoko), rozptyl
síly 0.9–2.6 dB.

**Co F2 našla.** `songs/kroky.py` měla nepravidelnost 7.6 %: druhé
`steps()` (běh od taktu 10) začalo na mřížce 0.23 beatu po posledním
kroku chůze a stejnou nohou — zakopnutí. Teď `steps()` začínající méně
než 1.5 intervalu po posledním kroku stopy naváže (o interval později,
druhou nohou) a rychlost převede za `CONTINUE_STEPS` = 3 kroky (skok
z intervalu 1.05 na 0.72 beatu dával 5 %). Metrika nepravidelnosti
přitom přešla z klouzavého mediánu na lokální přímku (rozběh 1.4 ->
3.4 m/s za 2 takty mediánu ujížděl). Výsledek: 2.6 %, nohy se střídají.

**Neověřeno poslechem.** Všechno výš jsou měření spektra a obálky, ne
poslech. Rozložení energie (kámen 85 % pod 200 Hz, tráva jen 17 % nad
4 kHz) je odhad z fyziky, ne srovnání se skutečnými nahrávkami kroků —
to je F4. Až budou reference, porovnat pásma, těžiště a poměr dupnutí
ku kliku po površích a upravit `Surface`.
