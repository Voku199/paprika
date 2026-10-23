# Lidské zvuky v paprika — dech, hlasová mikro-gesta, tělo a pohyb

**Datum:** 2026-10-23
**Typ:** průzkum + návrh implementace (ne schválený spec)
**Stav:** fáze H1 implementována (2026-10-23, verze 2.3.0): `dsp.ar1`,
`dsp.pink_walk`, `humanize(corr=)`, `Proximity`, `RoomTone`, `DynEQ`,
`s.human()` / `HumanTrack` (`human.py`), pravidla dechu v reportu,
ukázka `songs/dech.py`. Fáze H2 implementována (2026-10-23, verze 2.4.0):
`breath.py` — nádechy z VocalSet, model obálek (mel + PCA), syntéza,
`s.human(breath_model="auto")`, `python -m paprika samples breaths`.
H2 je ověřená jen na syntetických nahrávkách (síť cloudové session
nepustila zenodo.org); prahy hledání je potřeba zkontrolovat na skutečném
korpusu. Fáze H3 implementována (2026-10-23, verze 2.5.0): `Glottal`
(LF/Rd, jitter, shimmer, aspirace modulovaná tokem), `Tract` (plochy ->
formanty -> kaskáda, 5parametrová trubice fitovaná na formanty),
`Formant(mode="cascade", vary=)`, znělá gesta hum / sigh / effort / laugh
s výškou z harmonického plánu, `voices.voice_quality`, pravidlo „sterilní
hlas“, preset `choir_tract`, ukázka `songs/hlas.py`. Fáze H4
implementována (2026-10-23, verze 2.6.0) jinak, než navrhuje tabulka níž:
místo SFZ je korpus klipů `<vzorky>/human/` s `manifest.json`
a `CREDITS-human.txt` (`corpus.py`), protože smích nebo kašel nejsou tóny
na klaviatuře; `HumanTrack.clip(druh, takt, pitch=, beats=)` bere varianty
a hlasové klipy posouvá PSOLA na tón akordu. Worker
`tools/human_worker.py` (Dia, Chatterbox, Stable Audio Open přes diffusers,
PANNs, Whisper) píše WAV + JSON se zdrojem a licencí; síto v jádru
(délka, ticho, clipping, znělost, rytmus smíchu, štítky, přepis,
duplicity). Neuronové backendy jsou napsané podle README projektů a nebyly
spuštěné (cloud bez GPU, huggingface.co blokované); celý protokol je
ověřený backendem `paprika` (vestavěná syntéza). Orpheus a Stable Audio
3.0 Small zapojené nejsou; další model je funkce `gen_<jméno>` ve workeru
(`BACKENDS`) a prompty v `corpus.PROMPTS`.

**Ověření na skutečných datech (2026-10-23):** z výstupu
`samples breaths --examples` (10 skutečných nádechů, female7, female8,
male3) vyšlo: (1) skutečné nádechy mají těžiště 2.2–2.4 kHz a hlavní
energii v 1–2 kHz u obou pohlaví, fyzikální dech z H1 byl o 1.5 kHz
jasnější — jas, šířky formantů a přední turbulence jsou podle nich
překalibrované (chyba v pásmech 11.5 -> 4.7 dB); (2) dva nádechy male3
začínaly 40–60 ms doznívání předchozího tónu, které YIN přehlédl —
hledání teď vyřazuje rámce s tónovým spektrem (`TONAL_FLAT`).

**Celý VocalSet (2026-10-23, zenodo.org už dostupné):** starý detektor
našel 66 nádechů v 1485 souborech (odmítnuto „bez nádechu“ 5803, „krátký“
809) a vlastní model měli jen female7, female8 a male3. Příčina: hluk
studia je hlavně dunění pod 150 Hz (na 41 souborech o 14–25 dB na bin
silnější než v 150 Hz–8 kHz), nádechy jsou nad hlukem o 20–30 dB v pásmu
150 Hz–8 kHz, ale jen o ~8 dB pod 150 Hz — širokopásmově tedy stály jen
3–8 dB nad hladinou a práh +8 dB je neviděl. Oprava: hledání v pásmu
300 Hz–16 kHz (pásmo 1–8 kHz dalo podobný počet, ale víc dunění a horší
shodu syntézy), odečtení spektra šumu místnosti z nejtišších úseků
souboru (v `_check` i po mel pásmech v `describe`) a průměrování tvaru
spektra ve výkonu místo v dB. Výsledek: 208 nádechů, vlastní model má
11 zpěváků (female3, 5, 6, 7, 8, male3, 4, 5, 6, 9, 11), LSD syntézy
0.9–2.0 dB (dřív 2.2–3.4), těžiště korpusu a syntézy se v pásmu
150 Hz–16 kHz shodují (1.4–2.4 kHz, souhrnně ~1.9 kHz; nad 16 kHz zbývá
po odečtení jen šum elektroniky, který těžiště korpusu dřív zvedal
o ~450 Hz). Nádechy jsou 33–57 dB pod zpěvem; u female1, 2, 4, 9 a male1,
2, 7, 8, 10 v nahrávkách skoro nejsou (female1 má navíc stálý šum
300 Hz–1 kHz). Fyzikální dech z H1 má těžiště ~2.4 kHz, tedy na jasném
konci korpusu. Při ověřování se ukázalo, že `install.VOCALSET_SIZE` byla
špatně a instalace hlasů vždy skončila chybou velikosti.
**Navazuje na:** `research/2026-10-21-ai-engine-pruzkum.md` (§4 hlasy, §5.3
výraz), `specs/2026-09-23-realism-pass-design.md`, `paprika/sfx.py`,
`paprika/voices.py`

Otázky, na které dokument odpovídá:

1. Které akustické parametry musí být proměnlivé a jak, aby lidský zvuk
   nezněl strojově? Jak simulovat nedokonalost hlasového traktu a těla?
2. Které DSP a AI přístupy fungují pro lidské zvuky nejlépe? Jak jim dodat
   hloubku konvolučním prostorem, dynamickou EQ a simulací mikrofonu?
3. Jak lidské zvuky navázat na tempo, tóninu, dynamiku a frázování skladby?

Fyziologická čísla jsou **orientační rozsahy** z literatury, ne cílové
hodnoty. Dva vzorce (proximity efekt a trubice hlasového traktu) jsou
**ověřené výpočtem** v numpy/scipy (§1.5, §2.4). Licence modelů je potřeba
před použitím zkontrolovat přímo v repozitáři. U několika se je nepodařilo
ověřit z primárního zdroje a u každé z nich to v textu stojí.

---

## TL;DR — devět doporučení

1. **Lidské zvuky odvozovat z fyziky a z frází, ne je rozhazovat jako
   samply po mřížce.** Nádech je proud vzduchu s objemem a časem. Hlasitost,
   jas a délka nádechu plynou z toho, kolik vzduchu potřebuje další fráze a
   kolik času na nádech zbývá. Paprika zná partituru, takže to umí spočítat.
   DAW to nezná.
2. **Strojový dojem dnes vzniká na třech místech kódu:**
   - `render_sfx` cachuje podle `(name, params, sr, bpm)`, takže každé
     opakování téhož efektu je bit-identické („kulometný efekt“).
   - `humanize` bere nezávislý rovnoměrný šum. Lidské odchylky jsou ale
     dlouhodobě korelované a posluchači je preferují (Hennig 2011).
   - `Formant` jsou statické paralelní pásmové propusti bez jitteru, bez
     pohybu šířek pásem a bez šumu svázaného s periodou.
3. **Variace mají tři časové škály:** perioda (jitter, shimmer, aspirační
   šum v rytmu glotis), nota nebo gesto (overshoot, preparation, vývoj
   vibrata, tvar proudu) a fráze (drift, únava, hloubka nádechu). Každá
   škála potřebuje jiný druh šumu: bílý, OU/AR(1) a 1/f. Všechny jdou přes
   `lfilter`, takže nevznikne žádná smyčka po vzorcích.
4. **Neznělé lidské zvuky** (dech, šepot, vzdech bez hlasu, šustění) **jsou
   šum násobený časově proměnnou spektrální obálkou.** Realističnost leží v
   trajektoriích obálky, ne v jemné struktuře. Nejlepší poměr ceny a
   výsledku dává tento postup: obálky naučit z reálných nahrávek (LPC nebo
   mel + PCA) a šum generovat. Vznikne nekonečno neopakujících se variant,
   deterministicky a jen s numpy.
5. **Data jsou už stažená.** Zip VocalSet (CC BY 4.0) zůstává po instalaci v
   `_downloads`, ale `install.extract_vocalset` z něj rozbaluje jen
   `voices.TAKES`. Článek uvádí 17 technik, mezi nimi dyšný zpěv, vocal fry,
   zpěv na nádech, lip trill a mluvené ukázky. Mezi frázemi cvičení by měly
   být nádechy týchž zpěváků, ze kterých jsou `soprano`, `tenor` a sbory.
   Dech by pak měl stejnou barvu a stejný mikrofon. Názvy složek je potřeba
   ověřit přes `zipfile.namelist()`.
6. **Znělá gesta** (hm, vzdech s hlasem, sten, námaha, smích) **dělat přes
   zdroj a filtr.** Zdrojem je glotální puls LF/Rd s jitterem a aspiračním
   šumem modulovaným tokem glotis. Filtrem je trubice traktu (plochy →
   odrazy → all-pole) s nosní větví. Uniformní trubice 17,5 cm dá podle
   výpočtu 490/1470/2450 Hz. Klatt a Pink Trombone slouží jako referenční
   modely, implementace bude vlastní a vektorová.
7. **Prostor dělá půlku realističnosti.** Patří sem blízký mikrofon
   (proximity efekt je filtr 1. řádu), zbytkový šum místnosti místo
   digitálního ticha, poměr přímého a odraženého zvuku a dynamická EQ v
   pásmu 3–8 kHz. Lidský zvuk blízko a suchý proti hudbě v sále působí
   intimně. Soundtracky tenhle trik používají běžně.
8. **Neuronové modely používat jen offline jako generátor korpusu, nikdy v
   renderu.** Na smích, vzdech a lapání po dechu poslouží TTS s neverbálními
   tagy (Dia, Chatterbox, Orpheus, CosyVoice), na foley Stable Audio 3.0
   Small SFX. Výsledky se prosejí ušima (klasifikátor PANNs/AudioSet +
   pravidla) a uloží jako SFZ mimo repo stejně jako VocalSet.
   HunyuanVideo-Foley nepoužívat, licence Tencent Hunyuan nepokrývá EU.
9. **AI neslyší, proto report potřebuje nová pravidla:** nádech přes noty
   téhož interpreta (fyzikálně nemožné), fráze bez nádechu delší než zásoba
   vzduchu, opakovaný identický efekt, „sterilní hlas“ (jitter ≈ 0 a příliš
   vysoké HNR) a nádech hlasitější než fráze.

---

## 0. Výchozí stav — co dnes v kódu brání lidskému zvuku

| Místo | Dnes | Proč to zní strojově | Náprava |
|---|---|---|---|
| `sfx.render_sfx` | klíč cache `(name, params, sr, bpm)` | dvakrát stejný SFX = stejný vzorek. U dechu a kroků je to slyšet okamžitě | `variant` v klíči i v parametrech generátoru (§1.4) |
| `Track.humanize`, `DrumTrack.humanize` | `rng.uniform(-t, t)` nezávisle pro každou notu | bílý šum v časování, lidské odchylky jsou korelované | `humanize(..., corr=...)` s AR(1)/1/f (§1.3) |
| `SfxTrack.humanize` | časování vynulované | u riseru správně. Dech ale přesně na mřížce sedět **nesmí** | dech se kotví ke konci (k nástupu fráze), konec s jitterem (§3.1) |
| `Patch.drift` | jedno náhodné LFO 0,7 Hz | jedna časová škála, chybí jitter, fine fluctuation a overshoot | glotální zdroj a vícevrstvý model F0 (§1.2, §2.2) |
| `filters.Formant` | 5 paralelních BP z tabulky Csound, lineární morf | paralelní BP nemá správná údolí mezi formanty ani sklon spektra. Výšky i šířky formantů jsou pevné | kaskádní nebo trubicový trakt, perturbace formantů ±2–5 % (§1.5, §2.2) |
| `voices.py` | jen `TAKES` (long tones, scales, arpeggios) | dech a dyšné techniky leží nevyužité | korpus dechů a gest (§2.3) |
| render | digitální ticho mezi zvuky | žádná nahrávka nemá nulový šum, ticho zní „sterilně“ | `RoomTone` pod blízkými zdroji (§2.4) |

Na co se dá navázat: `Ensemble` (rozmazání nástupu, rozladění),
`ADSR(curve="exp")`, konvoluční `Reverb` a `Hall`, `filt_varying` (časově
proměnný biquad po blocích se `zi`), `colored_noise`, YIN a PSOLA ve
`voices.py`, harmonický plán (`Song.chord_at`), sekce a automatizace `dyn`.

---

## 1. Akustika a mikro-variace

### 1.1 Mapa proměnných

| Parametr | Fyziologický původ | Lidský rozsah (orientačně) | Časová škála | Model |
|---|---|---|---|---|
| **Jitter** (kolísání délky periody) | nepravidelné kmity hlasivek | lokálně 0,2–1 % (Praat bere > 1,04 % jako patologické) | perioda | gaussovský posun okamžité frekvence pro každou periodu |
| **Shimmer** (kolísání amplitudy periody) | totéž | lokálně 1–3,5 % (práh Praat 3,81 %) | perioda | násobek amplitudy pro každou periodu |
| **Fine fluctuation** | mikrotřes svalů | jednotky centů, nad 10 Hz (Saitou) | 10–100 ms | horní propust na bílém šumu |
| **Drift, bloudění F0** | regulace výšky sluchem | ±5–20 c u školeného zpěváka, u laika víc | 0,3–3 s | OU proces (návrat k cíli) + 1/f |
| **Overshoot, preparation** | dynamika svalů při změně výšky | přestřelení a „nájezd“ o desítky centů | 50–200 ms | tlumený systém 2. řádu (Saitou 2007/2009) |
| **Vibrato** | modulace napětí hlasivek | 5–7 Hz; v opeře ±50–100 c, ve sboru a popu méně. Nastupuje se zpožděním, rychlost i hloubka v notě kolísají zhruba o ±10 % | nota | LFO s OU šumem na rychlosti i hloubce |
| **Dyšnost** (Rd, aspirace, HNR) | glotis se úplně nedovře | Rd ≈ 0,3–0,6 tlačený hlas, ~1 modální, 1,5–2,7 dyšný (Fant 1995). HNR modálního hlasu kolem 20 dB | fráze, gesto | Rd a hladina aspiračního šumu jako křivky |
| **Formanty** | délka a tvar traktu, čelist, jazyk | F_n ≈ (2n−1)·c/4L; L ≈ 17,5 cm u mužů, ~14,5 cm u žen. Mezi provedeními náhodně ±2–5 % | gesto | perturbace délky traktu + koartikulace k další samohlásce |
| **Sklon spektra** | rychlost uzavírání glotis | zdroj kolem −12 dB/okt, podle síly o několik dB víc či míň | fráze | Rd nebo dvoupólový LP na zdroji |
| **Subharmonie, period doubling** | nepravidelné kmitání (námaha, fry) | fry 20–70 Hz, nepravidelné pulzy | gesto | střídání amplitudy a periody každé druhé periody; pulzní řetěz s velkým jitterem |
| **Mikro-časování** | motorika, dýchání | jednotky až desítky ms, dlouhodobě korelované | nota → fráze | AR(1)/1/f přes indexy not |
| **Proud vzduchu (dech)** | plíce, objem za čas | klidný nádech 0,3–0,8 s; ve zpěvu „catch breath“ ~0,15–0,3 s | gesto | profil průtoku U(t) → turbulence (§1.5) |

Princip: **každá veličina má svou časovou škálu a svůj druh šumu.** Strojový
dojem nevzniká tím, že chybí náhoda. Vzniká tím, že je náhoda na špatné
škále (bílý šum tam, kde má být pomalé bloudění), nebo je všude stejná.

### 1.2 Model F0 pro znělá gesta (všechno vektorově)

```
F0(t) = F0_cíl(t)                       # z noty nebo z akordu plánu (§3.2)
      · 2^((overshoot(t) + vibrato(t) + drift(t) + fine(t)) / 1200)
      · (1 + jitter[perioda(t)])
```

- `overshoot` je odezva tlumeného systému 2. řádu na skok cíle: `lfilter`
  s koeficienty z ω a ζ (bilineární transformace). Parametry uvádí Saitou
  2009.
- `drift` je Ornstein–Uhlenbeck, tedy AR(1), viz náčrt níže. Na dlouhém tónu
  k němu patří pomalý pokles o jednotky centů (únava, ke konci fráze ubývá
  vzduch).
- `fine` vznikne z bílého šumu přes `dsp.filt(..., "hp", 10 Hz)`,
  normalizovaného na 2–5 c.
- `jitter`: z fáze bez jitteru (`floor(cumsum(f)/sr)`) se zjistí, kolikátá
  perioda zrovna zní. Každá perioda dostane gaussovské `δ_k` (σ ≈ 0,5 %) a
  fáze se integruje znovu. Shimmer funguje stejně, index periody se bere z
  výsledné fáze.

```python
def ar1(n: int, tau: float, sr: float, seed: int = 0) -> np.ndarray:
    """Ornstein-Uhlenbeck (AR(1)) s časovou konstantou `tau` s a rozptylem 1."""
    a = math.exp(-1.0 / (tau * sr))
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    zi = [a * rng.standard_normal()]          # start ve stacionárním stavu
    return signal.lfilter([math.sqrt(1 - a * a)], [1.0, -a], w, zi=zi)[0]


def pink_walk(n, sr, tau_min=0.05, octaves=6, seed=0):
    """≈1/f mezi 1/(2π·tau_max) a 1/(2π·tau_min): součet AR(1) po oktávách."""
    return sum(ar1(n, tau_min * 2 ** k, sr, seed + k)
               for k in range(octaves)) / math.sqrt(octaves)
```

Pro řídké veličiny (časování not, hladina kroků) poslouží stejný AR(1), jen
přes **index události** místo vzorků (`sr = 1`, `tau` v počtu not).

### 1.3 Humanize s pamětí

Hennig et al. (2011) ukázali, že odchylky lidských hráčů od metronomu jsou
dlouhodobě korelované přes stovky až tisíce úderů. V poslechovém testu
hodnotilo 39 sborových zpěváků lépe verzi s takto korelovanými odchylkami
než verzi s nekorelovanými, jaké generují běžné funkce „humanize“. Návrh
API, zpětně kompatibilní:

```python
piano.humanize(timing=0.015, vel=0.12, corr=0.85, seed=3)
# corr=0 -> dnešní chování (bit-identický render); corr>0 -> AR(1) přes indexy not:
# dev_k = corr * dev_{k-1} + sqrt(1 - corr^2) * e_k
```

Dvě jemnosti dělají znatelný rozdíl:

1. Odchylka akordu je společná všem jeho tónům plus malý rozptyl každého
   tónu. `Ensemble.smear` to řeší uvnitř hlasu, ne mezi notami.
2. Velocity a timing jsou slabě korelované: silnější úhoz přichází spíš o
   chlup dřív.

### 1.4 Varianty místo kopií (proti kulometnému efektu)

Každý lidský SFX dostane `variant` (index výskytu na stopě nebo `seed`).
Generátor z něj losuje **parametry**, ne jen šum: délku ±15 %, formanty
±4 %, hladinu ±2 dB, sklon ±2 dB, tvar průtoku a poměr nos/ústa. `variant`
patří do klíče cache `render_sfx`. Počet variant má strop (třeba 8 jako
`Patch.rr`), aby cache dál fungovala, a stejná varianta nesmí zaznít
dvakrát po sobě (stejná logika jako round-robin v sampleru). Determinismus
zůstává: stejný song dá stejné varianty.

### 1.5 Nedokonalost traktu a těla — fyzika, ze které se dá počítat

**Turbulence (dech, šepot, sykavky).** Proud je slyšet až nad kritickým
Reynoldsovým číslem: Re = v·d/ν, kde ν vzduchu ≈ 1,5·10⁻⁵ m²/s a přechod
leží řádově kolem 2000. Amplituda turbulentního zdroje roste zhruba jako
U³·A^(−5/2) (Stevens), po dosazení Bernoulliho rovnice jako Δp^1,5·√A.
Z toho plynou tři pravidla, která se dají přímo naprogramovat:

- **Obálka zvuku není obálka proudu.** Platí `zvuk(t) ∝ max(0, U(t) − U_c)^3`.
  Klidný dech je proto skoro neslyšitelný a zrychlený nádech náhle
  „naskočí“. Tenhle tvar lineární ADSR na šumu nikdy nedá.
- **Stejný objem za kratší čas zní hlasitěji a jasněji.** Rychlý nádech mezi
  krátkými frázemi (catch breath) musí znít ostřeji než hluboký pomalý
  nádech na začátku sekce. Těžiště spektra zdroje roste s rychlostí proudu
  (Strouhalovo škálování f ∝ v/d). Fyzikálně jde o +18 dB na každé
  zdvojnásobení průtoku. V enginu je potřeba exponent stlačit (~1,5–2) a
  hlídat úroveň vůči frázi (§4).
- **Výdech je pasivní, nádech aktivní.** Při výdechu se plíce elasticky
  vracejí a průtok klesá zhruba exponenciálně (časová konstanta R·C řádově
  v desetinách sekundy). Nádech má tvar zešikmeného zvonu.

**Trakt.** Vokální trakt je trubice proměnného průřezu. Výpočet:

1. Z ploch A_i se spočítají odrazové koeficienty
   k_i = (A_{i+1} − A_i)/(A_{i+1} + A_i).
2. Levinsonova „step-up“ rekurze z nich sestaví polynom D(z) (Rabiner &
   Schafer, model bezztrátových trubic). Glotis má r_G = 1, rty r_L < 1,
   to jsou ztráty vyzařováním.
3. Při `fs` = 44,1 kHz má jedna sekce délku c/(2·fs) ≈ 3,9 mm, na 17,5 cm
   tedy vyjde 45 sekcí. Pink Trombone používá 44.

**Ověřeno výpočtem:**

- Uniformní trubice dá formanty 490 / 1470 / 2450 / 3430 Hz, přesně podle
  teorie (2n−1)·c/4L.
- Dvojtrubice s úzkým hltanem a širokými ústy dá F1 ≈ 760 Hz a F2 ≈ 1220 Hz,
  což odpovídá /a/.
- Samotné ztráty na rtech dávají příliš úzké formanty (9–50 Hz místo
  50–150 Hz). Šířky pásem proto doladit roztažením `a_i ← a_i·γ^i`
  (γ ≈ 0,995–0,999).

Nedokonalosti, které stojí za simulaci:

- **Délka traktu** se mezi provedeními mírně mění (čelist, hrtan stoupá při
  vysokém tónu). Stačí škálovat všechny formanty gesta náhodně o ±2–4 %.
- **Koartikulace**: nádech se dělá přes tvar úst příští samohlásky (před
  `soprano.ooh` nádech přes „u“) a konec gesta přechází do další samohlásky.
- **Nosní větev**: dech nosem a „hm“ odpovídají paralelní nosní trubici.
  Ta přidá nízkou rezonanci (~250–300 Hz) a antirezonanci (notch, u /m/
  kolem 1 kHz). V enginu stačí `notch` biquad s pohyblivou frekvencí.
- **Subglotální vazba**: při otevřené glotis (nádech, dyšný hlas) se přidají
  slabé rezonance průdušnice. U dospělých leží první kolem 600 Hz a druhá
  kolem 1,4–1,5 kHz. Je to jemný detail, ale odlišuje „vzduch v těle“ od
  filtrovaného šumu.
- **Poddajné stěny traktu** lehce zvednou F1 a rozšíří jeho pásmo. V praxi
  to pokryje γ.

**Tělo a pohyb.**

- Tělo je silně tlumený rezonátor s Q řádově v jednotkách. Úder do hrudi,
  krok i srdce se dají složit z několika tlumených sinusovek (modální
  syntéza) buzených krátkým pulzem.
- Lidský pohyb sleduje profil minimálního trhnutí (Flash & Hogan 1985):
  poloha x(τ) = 10τ³ − 15τ⁴ + 6τ⁵, rychlost ∝ 30τ²(1−τ)².
- Šustění oblečení je Poissonův proud mikro-impulzů. **Četnost impulzů
  roste s rychlostí pohybu** a amplituda jako v^α. Proud jde přes filtr
  látky: nylon je jasný (2–8 kHz), vlna a kůže tmavší. Skřípění kůže je
  stick-slip, tedy pulzní řetěz. Jde o PhISEM Perryho Cooka: stochastické
  srážky plus rezonance.
- Krok má dvě pod-události (pata, špička) vzdálené desítky ms. Levá a pravá
  noha se liší hladinou a panem. Povrch určuje rezonance a hustota zrn
  (Fontana & Bresin, „crumpling“).

---

## 2. Syntéza a zpracování

### 2.1 Srovnání přístupů

| Přístup | Nejlepší na | Realističnost | Kontrola (tempo, výška, délka) | Determinismus a CPU | Do paprika | Licence |
|---|---|---|---|---|---|---|
| **Parametrický zdroj a filtr** (LF/Rd + aspirace + formanty, Klatt) | znělá gesta, šepot, „hm“, námaha | střední až dobrá, s mikro-variacemi | úplná | ano, levné | ano, numpy | vlastní |
| **Artikulační model** (trubice, Pink Trombone, VocalTractLab) | plynulé přechody tvarů, dech přes samohlásku, gesta „ah→mm“ | dobrá na přechody | úplná, intuitivní (plochy) | ano, levné, pokud all-pole po blocích | ano (§1.5) | fork `pink-trombone-mod` je MIT, originál neověřen. VocalTractLab GPL-3, jen jako reference |
| **Analýza a resyntéza obálek** (LPC nebo mel z reálných nahrávek, PCA, nový šum) | **dech, výdech, sniff, šepot, šustění** | velmi dobrá u neznělých zvuků | délka, hloubka, jas, tvar | ano, levné | ano, model je malý JSON | podle dat (VocalSet CC BY 4.0) |
| **Statistiky textur** (McDermott & Simoncelli 2011) | šustění, dav, déšť na oblečení | dobrá u stacionárních textur | slabší (délka, hustota) | ano, syntéza pomalejší | ano, offline | vlastní |
| **Granulární nebo konkatenační korpus** (+ WSOLA/PSOLA) | reálné dechy a gesta natažené na mezeru (±20 %) | nejvyšší, je to nahrávka | délka v mezích, výška přes PSOLA | ano | ano, jako sampler | podle dat |
| **WORLD** (`pyworld`) | úprava hotových vokálů: dyšnost přes aperiodicitu, posun formantů, jitter | velmi dobrá | F0, obálka a aperiodicita zvlášť | ano | volitelná závislost | WORLD BSD, pyworld MIT |
| **DDSP** (harmonický + filtrovaný šum) | naučený timbre z ~10 min dat | dobrá | F0 a hlasitost jako vstup | inference v numpy možná, trénink v torch | worker | Apache 2.0 |
| **TTS s neverbálními tagy** (Dia, Chatterbox, Orpheus, CosyVoice 2) | smích, vzdech, kašel, lapání po dechu | dobrá až výborná, ale nestabilní | slabá (délka, výška, tempo) | ne, proto generovat jednou a uložit | worker + korpus | Dia Apache 2.0 (neověřeno), Chatterbox MIT (výstup nese vodoznak PerTh), ostatní ověřit |
| **Text → audio SFX** (Stable Audio 3.0 Small SFX, 459 M parametrů, květen 2026) | foley: oblečení, kroky, předměty | dobrá | slabá | ne, proto korpus | worker + korpus | Stability AI Community License: komerčně zdarma pod ~1 mil. $ ročního obratu. Trénováno podle Stability jen na licencovaných datech |
| **Video → audio foley** (HunyuanVideo-Foley, MMAudio, FoleyCrafter, Foley-Omni) | synchronizace se záběrem | dobrá | žádná hudební | ne | ne | Hunyuan **mimo EU**; MMAudio a Foley-Omni ověřit |
| **Neuronové vokodéry** (Vocos, BigVGAN v2) | jen při editaci mel-spektrogramů | — | — | — | jen ve workeru | MIT |

Závěr pro paprika: **parametrické jádro + obálky naučené z dat + volitelný
korpus z neuronových modelů.** Neznělé zvuky stojí na resyntéze obálek,
znělá gesta na zdroji a filtru, „hrané“ zvuky (smích, kašel) na korpusu.
Render zůstává numpy + scipy a deterministický. TTS modely podle benchmarků
z roku 2026 (NVV-SuperBench, výzva NVVSpeech na ISCSLP 2026) pořád neumějí
spolehlivě umístit neverbální zvuk na přesné místo. I to mluví pro korpus a
vlastní časování.

### 2.2 Glotální zdroj a trakt (znělá gesta)

- **`Glottal(Osc)`** v `oscillators.py`:
  - Puls toku glotis je uložený jako tabulka LF pulzu pro několik hodnot Rd
    (předpočítat, mezi nimi interpolovat). Čte se fází přes `np.interp`
    jako u ostatních oscilátorů.
  - Antialiasing: dvojnásobné převzorkování (`resample_poly`) nebo tabulky
    po oktávách.
  - Parametry: `rd` (číslo nebo křivka), `jitter`, `shimmer`, `aspiration`
    (hladina šumu) a `leak` (šum i v uzavřené fázi, tedy neúplné dovření).
  - **Aspirační šum je modulovaný průtokem glotis**, obálka šumu kopíruje
    LF puls (Gobl 2006). Právě to odlišuje dyšný hlas od syntetického
    šumění přes tón.
  - Jako `Osc` ho mohou použít i presety (sbor, „vocal“ pady), takže se
    zlepší i hudba, nejen SFX. Všechny parametry patří do `key()`.
- **`Tract`** ve `filters.py`:
  - Vstupem je samohláska (tabulky ploch, např. MRI funkce ploch podle
    Storyho) nebo přímo plochy.
  - Uvnitř: plochy → D(z) → kořeny → formanty (F, B) → kaskáda rezonátorů.
  - Změna tvaru v čase se interpoluje v doméně (F, B), ne v koeficientech
    D(z). Filtr vysokého řádu v přímé formě při změně koeficientů lupe.
    Kaskáda biquadů přes stávající `filt_varying` po blocích je stabilní a
    otestovaná.
  - `nasal=0..1` přidá nosní pól a antirezonanci, `radiation=True` první
    diferenci (vyzařování rty, +6 dB/okt).
- **`Formant`** dostane `mode="cascade"` a `jitter` (náhodná perturbace
  formantů na notu), aby šly stávající songy zlepšit bez přepisu. Výchozí
  režim zůstane paralelní a render bit-identický.

### 2.3 Dech z dat (neznělé zvuky)

Postup modulu `breath.py`. Analýza běží offline jako ve `voices.py`, render
jen čte hotový model.

1. **Korpus.** Z VocalSet vyříznout nádechy mezi frázemi cvičení: úseky bez
   výšky (YIN vrací NaN) s energií nad šumem pozadí, dlouhé 0,15–1,5 s,
   těsně před nástupem tónu. Doplnit technikou zpěvu na nádech a dyšnými
   tóny, pokud v archivu jsou (pravděpodobně
   `FULL/<zpěvák>/<kontext>/<technika>/`, ověřit). Volitelně přidat CC0
   dechy z Freesound a vlastní nahrávky.
2. **Analýza.** Rámce 20 ms s krokem 5 ms: LPC řádu ~24 nebo ~30
   kepstrálních koeficientů → spektrální obálka v log-mel pásmech, k tomu
   energie a centroid. Délku normalizovat na 0..1 (time-warping k průtokové
   křivce).
3. **Model.** PCA přes celé časově normalizované trajektorie obálek. Průměr
   + 4–8 komponent pokryje většinu variability. Uložit jako
   `breath_model.json` vedle vzorků (s atribucí CC BY 4.0), zvlášť pro
   každého zpěváka a typ (ústa, nos, catch, vzdech).
4. **Syntéza.** Z `variant` se vylosují koeficienty komponent, délka a
   hloubka přijdou z frázování (§3.1). Vznikne trajektorie obálky, kterou
   se filtruje nový šum (násobení v STFT nebo LPC po blocích). Hlasitost
   určí průtok (§1.5). Výsledek je realistický jako nahrávka, ale nikdy se
   neopakuje a má přesně potřebnou délku.

Stejně se dá udělat sniff, výdech a šepot („h“). Na šustění stačí textura.

### 2.4 Prostor, mikrofon, dynamická EQ

**Proximity efekt.** Směrový mikrofon blízko zdroje zvedá basy. Pro
kardioidu na ose a kulovou vlnu platí H(s) = 1 + c/(2·r·s). Vůči vzdálenosti
`r_ref`, na kterou je mikrofon kalibrovaný:

```
H(s) = (s + c/(2r)) / (s + c/(2·r_ref))      # osmička: c/r místo c/(2r), koule: žádný efekt
roh (+3 dB): f_c = c / (4π r)               # 5 cm → 546 Hz, 30 cm → 91 Hz
```

Je to filtr 1. řádu (low shelf), který jde bilineární transformací rovnou
do `lfilter`. **Ověřeno výpočtem:** `signal.bilinear` se shoduje s
analytickým vzorcem na 0,1 dB. Zesílení vůči 1 m:

| vzdálenost | 50 Hz | 100 Hz | 200 Hz | 500 Hz | 1 kHz |
|---|---|---|---|---|---|
| 3 cm | +24,1 dB | +18,9 | +13,3 | +6,3 | +2,6 |
| 5 cm | +19,7 | +14,6 | +9,2 | +3,4 | +1,1 |
| 15 cm | +10,4 | +6,0 | +2,5 | +0,5 | +0,1 |
| 30 cm | +5,2 | +2,3 | +0,7 | +0,1 | 0 |

Ústa nejsou bodový zdroj a reálné mikrofony mají vestavěný low-cut.
Zesílení je proto potřeba omezit na ~+12 dB a pod ~60 Hz ho srazit HP
filtrem. Jinak mix spadne mimo cíl `sub ≤ 12 %` z `metrics.TARGETS`.

```python
class Proximity(Fx):
    """Blízký mikrofon: basy rostou, když je zdroj blíž než `ref` metrů."""
    AUTOMATABLE = ("dist",)

    def __init__(self, dist=0.1, pattern="cardioid", ref=1.0, name=None): ...

    def process(self, buf, ctx):
        k = {"cardioid": 0.5, "figure8": 1.0}[self.pattern]
        b, a = signal.bilinear([1.0, k * C / self.dist], [1.0, k * C / self.ref], ctx.sr)
        return signal.lfilter(b, a, buf, axis=0)
```

Automatizovaný `dist` znamená koeficienty po blocích se `zi`, stejně jako
`dsp.onepole`.

**Vzdálenost a místnost.**

- Přímý zvuk klesá o 6 dB na každé zdvojnásobení vzdálenosti, dozvuk se se
  vzdáleností skoro nemění. Rozhoduje proto poměr přímého a odraženého
  zvuku. Kritická vzdálenost je r_c ≈ 0,057·√(V/T60) m; místnost 200 m³
  s T60 = 1 s ji má kolem 0,8 m.
- Útlum vzduchem je slyšet až v desítkách metrů (vysoké frekvence). Pro
  dech je zanedbatelný, pro vzdálené kroky a výkřiky v hale ne.
- **Blízký lidský zvuk = suchý signál + proximity + krátká malá místnost**
  (IR 0,2–0,4 s se silnými časnými odrazy, `synth_ir(early=...)`). Hudba
  zůstane v `Hall` a vznikne intimita. „Vzdálený“ dech v hale je jen
  dozvuk a ztratí se.
- Reálné IR nabízí OpenAIR (licence po jednotlivých IR). Vlastní IR jde
  spočítat v `pyroomacoustics` (MIT, metoda zrcadlových zdrojů, směrovost
  mikrofonu).

**Šum místnosti místo ticha.** Pod blízkými zdroji poběží
`RoomTone(level=-66, color="pink")` s velmi pomalým náběhem a odezněním
kolem gesta. Nesmí to být stálá vrstva, ta by se v mixu sčítala. Pozor:
pravidla reportu, která ignorují „skoro tiché takty“ (`_gaps`, šum ~1e-17 z
konvoluce), musí s touto hladinou počítat.

**Dynamická EQ.** Energie dechu leží hlavně v 2–8 kHz. V hustém mixu dech
buď zmizí, nebo škrábe.

- `DynEQ(freq, q, threshold, ratio, sidechain=None)` je peak biquad, jehož
  zisk řídí obálka pásmového signálu (vlastního nebo z jiné stopy).
- Implementace: obálka jako v `Compressor.gain`, zisk po blocích. Do
  `filt_varying` je potřeba doplnit proměnné `gain_db`.
- Typické použití: zkrotit 4–6 kHz u hlasitých nádechů, nebo sidechain
  hudby na hlas (−2 až −3 dB v 2–5 kHz jen ve chvíli, kdy dech zní).
- Klasický střih „nádech před dropem“: hudba na jednu dobu ztichne a dech
  zní sám, blízko a suše.

**Mikrofon a předzesilovač.** Mírný „presence“ zdvih (+2–3 dB kolem 5 kHz,
typický pro velkomembránové mikrofony) a lehká saturace (`Saturate` s nízkým
`drive`). Vlastní šum mikrofonu pokryje `RoomTone`.

---

## 3. Napojení na hudební kontext

### 3.1 Tempo a časování: dech patří k frázi, ne k mřížce

- **Kotva na konci.** Nádech končí na nástupu fráze (stejná sémantika jako
  `riser(to_bar)`), jen s jitterem ±10–20 ms a mezerou 0–40 ms podle
  artikulace (§3.4).
- **Délka z mezery.** `délka = min(ideální(hloubka), mezera − rezerva)`. Do
  mezery kratší než ~120 ms se plný nádech nevejde. Pak přijde buď catch
  breath (krátký, ostrý), nebo žádný, a pravidlo zkontroluje zásobu vzduchu
  (§4).
- **Hloubka z příští fráze.** Objem nádechu roste s délkou a hlasitostí
  fráze, která po něm následuje. U řeči objem nádechu škáluje s délkou
  následující promluvy (Winkworth et al. 1995). Zásoba vzduchu:
  `T_max ≈ V_použitelný / U_průměr(dyn)`. Orientačně V ≈ 2–3 l u školeného
  zpěváka, U ≈ 0,1 l/s v p až 0,3 l/s ve ff (dyšný hlas víc). Konstanty se
  povedou v tabulce pro každý nástroj (hlas, flétna, žestě).
- **Automatické dechy.** `track.breaths(into=sfx, min_gap=..., style="singer")`
  projde fráze stopy (mezery mezi notami, pomlky, hranice `section`) a
  vloží dechy. Netýká se jen hlasu: **flétna, hoboj, lesní roh i žestě
  dýchají slyšitelně.** Dech mezi frázemi dodá samplovaným VSCO stopám
  lidskost, aniž by se změnila hudba.
- **Sekce dýchají střídavě.** Sbor a smyčcová nebo dechová sekce
  (`Ensemble`, `male_choir`) nemají společný nádech („staggered
  breathing“). Slyšitelný společný nádech celé sekce je dramatické gesto,
  tedy výjimka (unisono nádech před dropem).
- **Rytmus těla v tempu.** Kroky jdou v tempu skladby (chůze kolem 2 Hz,
  tedy ~120 BPM, MacDougall & Moore 2005). Dech při námaze je zamčený na
  takt, např. nádech a výdech na 2 doby. Srdce tepe v sub-násobku tempa (v
  klidu 60–80 BPM, s napětím sekce roste). Všechno s korelovaným jitterem
  (§1.3), jinak to zní jako sekvencer.

### 3.2 Tónina a výška

- **Znělá gesta berou výšku z plánu.** `song.chord_at(beat)` dá akord a z
  něj se vybere tón nejbližší pohodlné výšce mluvčího (muž ~100–130 Hz,
  žena ~190–230 Hz). Stabilní část gesta přistane do ±30 c od tónu akordu,
  glissanda jsou volná.
  - Vzdech s hlasem klesá z tónu akordu o 3–7 půltónů na nižší tón akordu
    nebo stupnice.
  - „Hm“ souhlasu sedí na tónice nebo kvintě.
  - Smích má impulzy ~4–5 za sekundu, výšku výrazně nad mluvenou a během
    série klesá (Bachorowski et al. 2001). Kolem tónů akordu se drží jen
    volně, aby nezněl zpívaně.
- **Neznělé zvuky nemají výšku, ale můžou mít „tónový šum“.** Úzké
  rezonance (Q > ~8, pískání nosem, vítr přes zuby) zní jako tón. Buď držet
  Q nízko, nebo nejsilnější vrchol posunout na tón akordu (stačí ±1 půltón).
- **Teorie gesta nehodnotí jako melodii.** `theory_warnings` je ignoruje;
  kontroluje se jen to, kam přistane stabilní část (INFO).

### 3.3 Dynamika

- Hladina a jas nádechu se řídí dynamikou příští fráze (velocity, křivka
  `dyn`). Před forte přijde hlubší a slyšitelnější nádech, před pp tichý
  nosní.
- V přirozené nahrávce je dech zhruba 15–30 dB pod vrcholem fráze. Hlasitější
  dech je záměrně intimní (ASMR) efekt a musí pak zůstat blízko a suchý.
- Dech potřebuje volné pásmo 2–5 kHz (DynEQ na hudbě). Jinak ho report
  označí jako maskovaný (rozšíření `_masking`).
- Únava a napětí fungují jako křivka sekce. S rostoucí energií jsou nádechy
  kratší, rychlejší a hlasitější, gesta dyšnější (vyšší Rd) a námaha má víc
  subharmonií.

### 3.4 Artikulace a frázování

| Nástup fráze | Dech nebo gesto před ním |
|---|---|
| tvrdý (akcent, marcato) | krátký ostrý nádech + glotální uzávěr: 20–50 ms ticha, pak tvrdý nástup |
| měkký (legato, pp) | nádech plynule přejde do nástupu s překryvem 0–30 ms. „H-nástup“: aspirace 30–80 ms před hlasem |
| po dlouhé pomlce | hluboký nádech, případně mlasknutí rtů 50–200 ms před nástupem (2–3 drobné impulzy 1–6 kHz) |
| konec fráze | uvolnění: krátký výdech, u hlasu dozvuk vzduchu s klesající výškou. Na konci dlouhé fráze slyšitelně ubývá síla i výška (dochází vzduch) |

**Gramatika gest:** přípravná gesta (nádech, mlasknutí), držící (námaha
během dlouhé fráze) a závěrečná (výdech, vzdech). Každé gesto musí mít
příčinu ve frázi. Náhodně rozesetá gesta zní hůř než žádná (uncanny valley).

**Koartikulace:** nádech končí v tvaru příští samohlásky. Artikulace
(`soprano.aah` nebo `.ooh`) je známá, takže formanty na konci nádechu se
rovnají formantům té samohlásky (morf `Tract`/`Formant` k cíli).

### 3.5 Návrh API

```python
sfx = s.sfx("dech", vol=0.5).fx(Proximity(dist=0.12), RoomTone(-66))
sfx.breath(to_bar=9, beats=0.75, depth=0.8, kind="mouth")   # skončí na začátku taktu 9
sfx.breath(to_bar=13, beat=2.5, kind="catch")
sfx.gesture("sigh", bar=24, pitch="chord", beats=2)         # výška z plánu
sfx.gesture("lip_smack", to_bar=9, beat=-0.2)
sfx.foley("cloth", bar=4, beats=1, motion="swing")          # profil minimálního trhnutí
sfx.foley("steps", bars=(16, 24), every=1, surface="stone") # v tempu, s korelovaným jitterem

flute.breaths(into=sfx, style="flute", min_gap=0.25)        # automaticky z frází
choir.breaths(into=sfx, style="section")                    # střídavě, tišeji
```

`SfxTrack.humanize` u lidských zvuků timing nenuluje. Kotva na konci
zůstane, jitter je ±10–20 ms.

---

## 4. Uši pro lidské zvuky (AI je neslyší)

Pravidla bez audia patří do `model_warnings`, pravidla nad audiem do
`warnings_for`. Každé má strop 3–4 hlášení.

| Pravidlo | Úroveň | Podmínka |
|---|---|---|
| `dech přes noty` | WARN | nádech přiřazený stopě se překrývá s jejími notami. Jeden interpret nemůže zároveň dýchat a zpívat nebo foukat. Neplatí pro sekce |
| `chybí nádech` | WARN | fráze bez mezery je delší, než dovolí zásoba vzduchu nástroje při její dynamice (§3.1) |
| `kulometný efekt` | WARN | dva po sobě jdoucí stejné lidské SFX mají stejnou variantu (stejný klíč cache) |
| `sterilní hlas` | WARN | znělé gesto má jitter < 0,05 % a HNR > 35 dB (měřeno YIN z `voices.track_pitch`) |
| `dech hlasitější než fráze` | INFO | RMS nádechu je méně než 12 dB pod frází |
| `dech na mřížce` | INFO | všechny dechy leží přesně na hraně doby (bez jitteru) |
| `maskovaný dech` | INFO | hudba je během nádechu v 2–5 kHz o víc než X dB silnější |

Měřitelné „posluchačské“ signály:

1. **Klasifikátor AudioSet** (PANNs, kód MIT, běží na CPU). AudioSet má
   třídy Breathing, Gasp, Sigh, Laughter, Sniff, Whispering, Humming,
   Walk/footsteps a Heart sounds. Syntetický vzdech musí vyjít jako vzdech.
   Pokles skóre po změně kódu je regrese.
2. **Praat/parselmouth** (GPL-3, jen jako externí měřicí nástroj, ne
   závislost jádra) měří jitter, shimmer, HNR a formanty gest. Výsledky se
   porovnají s lidskými rozsahy z §1.1.
3. **Síto pro neuronový korpus** používá obojí. Generace, které klasifikátor
   neoznačí správně nebo mají artefakty (clipping, ticho, opakování), se
   zahodí.

---

## 5. Implementační plán

| Fáze | Obsah | Závislosti | Akceptace |
|---|---|---|---|
| **H1 — mikro-variace a mikrofon** (dny) | `dsp.ar1`, `dsp.pink_walk`; `humanize(corr=)`; `variant` v `SfxHit` a v klíči cache; `Proximity`, `RoomTone`, `DynEQ` (+ proměnné `gain_db` ve `filt_varying`); procedurální `breath`, `exhale`, `catch`, `lip_smack`, `heartbeat`, `cloth`, `steps` v novém `human.py` (vrstva jako `sfx.py`); pravidla z §4 | numpy, scipy | `corr=0` dá bit-identický render; AR(1) má rozptyl 1 a autokorelaci e^(−1/(τ·sr)); Proximity sedí s vzorcem na ±0,1 dB; varianty se liší (korelace < 0,5) a jsou deterministické; dech nikdy nepřekryje noty své stopy |
| **H2 — dech z dat** (1–2 týdny) | korpus nádechů z VocalSet + techniky z archivu; analýza v `breath.py` (LPC/mel obálky, PCA) → `breath_model.json` pro každého zpěváka; syntéza z modelu; `track.breaths()` | numpy, scipy | délka přesně podle mezery; spektrální obálka syntézy v toleranci vůči korpusu (log-spektrální vzdálenost); PANNs „Breathing“ nad prahem |
| **H3 — zdroj a trakt** (1–2 týdny) | `Glottal(Osc)` (LF/Rd tabulky, jitter, shimmer, aspirace modulovaná tokem); `Tract` (plochy → formanty, nosní větev, vyzařování); `Formant(mode="cascade", jitter=)`; gesta `hum`, `sigh` s hlasem, `effort`, `laugh` | numpy, scipy | uniformní trubice dá 490/1470/2450 Hz ±1 %; změřený jitter, shimmer a HNR v lidských rozsazích; nová pole `Osc` a filtrů v `key()` (`test_every_field_changes_key`) |
| **H4 — neuronový korpus** (volitelné) | worker v odděleném venv s Dia/Chatterbox/Orpheus a Stable Audio 3.0 Small SFX; generace → síto (PANNs + pravidla) → ořez, normalizace hlasitosti (jako `voices.NOTE_RMS_DB`) → SFZ mimo repo; cache podle hashe promptu | torch ve workeru | chybějící korpus hodí `SamplesMissing` s příkazem instalace, žádný tichý fallback; licence a atribuce v `CREDITS.txt` |

Proč v tomhle pořadí:

- **H1** zlepší i stávající hudbu (humanize s pamětí, mikrofon, varianty) a
  nenese žádné riziko.
- **H2** dá nejrealističtější dech za nejnižší cenu.
- **H3** je nejvíc práce, ale s nejširším dosahem: zlepší i sbory a „vocal“
  pady.
- **H4** jen tam, kde procedurální cesta nestačí (smích, kašel).

---

## 6. Rizika a otevřená rozhodnutí

- **Rozsah projektu.** Spec engine v2 rozhodl „instrumentální, bez zpěvu“.
  Dechy dechových nástrojů a foley do něj zapadají. Hlasová gesta (vzdech,
  smích) jsou spíš herní SFX a patří k podprojektu exportu pro hru
  (stingery). Hranici je potřeba rozhodnout.
- **Uncanny valley.** Skoro lidský dech ve špatnou chvíli je horší než
  žádný. Výchozí stav proto: vypnuto, tišší než „správně“ a vždy navázané
  na frázi.
- **Licence.**
  - HunyuanVideo-Foley: licence Tencent Hunyuan vylučuje EU.
  - VocalSound: CC BY-SA. Share-alike se může vztahovat i na model obálek z
    něj odvozený. Navíc jde o crowdsourcované nahrávky v 16 kHz, takže jen
    na analýzu a ověřování.
  - Chatterbox vkládá do výstupu vodoznak.
  - Freesound má licenci u každého souboru zvlášť; preferovat CC0.
  - NonverbalTTS kombinuje zdroje s různými licencemi.
  - Praat/parselmouth a VocalTractLab jsou GPL, proto jen jako externí
    nástroje.
  - Licence Dia, MMAudio, Foley-Omni a původního Pink Trombone se nepodařilo
    ověřit z primárního zdroje.
- **Determinismus.** Neuronové generování běží jen offline, výsledek se
  uloží a zahashuje. Render zůstává bit-identický.
- **Výkon.** Žádné smyčky po vzorcích: trubice jako all-pole nebo formanty po
  blocích, jitter přes index periody, šumy přes `lfilter`. Kaskádu
  rezonátorů pro všech 45 sekcí traktu nepočítat, stačí 5–6 formantů z
  kořenů.
- **Měření není poslech.** Pravidla a klasifikátor odhalí hrubé chyby, ne
  jemnou kvalitu. Konečné rozhodnutí o „lidskosti“ potřebuje lidský poslech
  nebo kritika L3 z předchozího průzkumu.

---

## Zdroje

Primární zdroje jsou označené (P), sekundární shrnutí (S). Kde chybí odkaz,
jde o standardní literaturu dohledatelnou podle názvu.

**Mikro-variace a hlas**

- (P) Hennig et al. (2011): The Nature and Perception of Fluctuations in
  Human Musical Rhythms, PLoS ONE — <https://doi.org/10.1371/journal.pone.0026457>,
  shrnutí <https://www.psych.uni-goettingen.de/de/cognition/publications/hennigetal2011>
- (P) Saitou et al.: model F0 zpěvu (overshoot, vibrato, preparation, fine
  fluctuation) — WASPAA 2007
  <https://staff.aist.go.jp/m.goto/PAPER/WASPAA2007saitou.pdf>, Interspeech
  2004 <https://www.isca-archive.org/interspeech_2004/saitou04_interspeech.pdf>,
  NCMMSC 2009 (parametry 2. řádu)
  <https://staff.aist.go.jp/m.goto/PAPER/NCMMSC2009saitou.pdf>
- (P) Gobl (2006): Modelling aspiration noise during phonation using the LF
  voice source model, Interspeech —
  <https://www.isca-archive.org/interspeech_2006/gobl06_interspeech.html>
- (P) Lu: Turbulence noise modeling for the breathy singing voices, CCRMA —
  <https://ccrma.stanford.edu/~jos/mus423h/Turbulence_noise_modeling_breathy.html>
- (P) Boersma (1991): Synthesis of speech sounds from a multi-mass model of
  the lungs, vocal tract, and glottis —
  <https://www.fon.hum.uva.nl/paul/papers/Proc15Boersma.pdf>
- (P) Fant (1995): The LF-model revisited (parametr Rd), STL-QPSR 36
- (P) Stevens (1998): Acoustic Phonetics, MIT Press (turbulentní zdroj,
  subglotální rezonance)
- (P) Rabiner & Schafer (1978): Digital Processing of Speech Signals (model
  bezztrátových trubic, odrazové koeficienty)
- (P) Story, Titze & Hoffman (1996): Vocal tract area functions from
  magnetic resonance imaging, JASA
- (P) Klatt (1980): Software for a cascade/parallel formant synthesizer, JASA
- (P) Bachorowski, Smoski & Owren (2001): The acoustic features of human
  laughter, JASA
- (P) Winkworth et al. (1995): Variability and consistency in speech
  breathing during reading, JSHR
- (P) Praat manuál, prahy jitteru a shimmeru —
  <https://www.fon.hum.uva.nl/praat/manual/Voice_2__Jitter.html>

**Tělo, pohyb, textury**

- (P) Cook: PhISEM a PhOLISE (fyzikálně informované foley), GameSoundCon
  2014 — <https://www.cs.princeton.edu/~prc/ChucKU/PerryCookPhysicalModelingGameSoundCon2014.pdf>,
  Cook & Lakatos WASPAA 2003
  <https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/CookWASPAAFinal03.pdf>,
  opkódy PhISEM v Csoundu <https://csound.com/manual/opcodes/crunch/>
- (P) Fontana & Bresin (2003): Physics-based sound synthesis and control:
  crushing, walking and running by crumpling sounds
- (P) Flash & Hogan (1985): The coordination of arm movements (minimum
  jerk), J. Neuroscience
- (P) McDermott & Simoncelli (2011): Sound texture perception via statistics
  of the auditory periphery, Neuron
- (P) MacDougall & Moore (2005): Marching to the beat of the same drummer,
  J. Sports Sciences

**Data**

- (P) VocalSet (CC BY 4.0) — <https://zenodo.org/records/1442513>, článek
  <https://ismir2018.ismir.net/doc/pdfs/114_Paper.pdf>
- (P) VocalSound (CC BY-SA; smích, vzdech, kašel, odkašlání, kýchnutí, sniff;
  21 024 nahrávek) — <https://sls.csail.mit.edu/downloads/vocalsound>,
  <https://arxiv.org/abs/2205.03433>
- (P) NonverbalTTS (17 h, 10 typů neverbálních zvuků) —
  <https://arxiv.org/abs/2507.13155>
- OpenAIR (impulzní odezvy, licence po souborech) —
  <https://www.openair.hosted.york.ac.uk/>

**Modely a knihovny**

- (P) Stable Audio 3.0 (Small SFX 459 M, open weights) —
  <https://stability.ai/news-updates/meet-stable-audio-3-the-model-family-built-for-artistic-experimentation-with-open-weight-models>,
  licence <https://stability.ai/license>, (S)
  <https://musicbusinessworldwide.com/stability-ai-launches-new-audio-models-that-can-generate-6-minute-music-tracks>
- (P) Licence HunyuanVideo (vyloučení EU) —
  <https://huggingface.co/tencent/HunyuanVideo/commit/a4ed25a99d7eb9cf21e12f5fd8052d266f2105fc>;
  (S) Foley varianta mimo EU/UK/Jižní Koreu —
  <https://www.creativeainews.com/articles/how-to-add-sound-effects-to-ai-video-2026/>
- (P) MMAudio, předtrénované modely (záznam Illinois Databank) —
  <https://databank.illinois.edu/datasets/IDB-1068600>; licence repa neověřena
- (P) Foley-Omni — <https://arxiv.org/abs/2606.03672>; (S) MIT podle blogu
  <https://studio.aifilms.ai/blog/foley-omni-video-soundtrack-generation>
- (P) FoleyCrafter — <https://github.com/open-mmlab/FoleyCrafter>
- (S) Přehled open-source TTS 2026 (Dia, Chatterbox, Orpheus, CosyVoice) —
  <https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models>;
  Chatterbox (MIT, vodoznak PerTh) —
  <https://www.linuxlinks.com/chatterbox-family-text-to-speech-models/>
- (P) NVV-SuperBench — <https://arxiv.org/abs/2604.16211>; řízené generování
  neverbálních zvuků — <https://arxiv.org/abs/2609.14231>
- (P) Pink Trombone (modulární fork, MIT) —
  <https://github.com/chdh/pink-trombone-mod>
- (P) VocalTractLab (GPL) — <https://www.vocaltractlab.de/>
- (P) pyworld — <https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder>
- (P) DDSP — <https://github.com/magenta/ddsp>
- (P) Vocos — <https://github.com/gemelo-ai/vocos>; BigVGAN —
  <https://github.com/NVIDIA/BigVGAN>
- (P) pyroomacoustics — <https://github.com/LCAV/pyroomacoustics>
- (P) PANNs (AudioSet tagging) — <https://github.com/qiuqiangkong/audioset_tagging_cnn>
- (P) Parselmouth (Praat v Pythonu, GPL) —
  <https://github.com/YannickJadoul/Parselmouth>
