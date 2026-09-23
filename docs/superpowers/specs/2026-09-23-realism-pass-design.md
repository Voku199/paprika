# Paprika — realism pass (v2.1)

**Datum:** 2026-09-23
**Stav:** schváleno delegací („udělej, co doporučuješ“)
**Navazuje na:** `2026-09-23-engine-v2-design.md`

## Proč

Uživatel po poslechu v2: „pořád je tam retro feeling, 16bitová pixelová
hudba“. Měření proti referencím v `songs_mp3/` (NieR City Ruins, Requiem of
Silence):

| metrika | reference | showcase v2 | iron_requiem (v1 song) |
|---|---|---|---|
| sub < 60 Hz | 3–7 % | 34 % | 40 % |
| středy 250 Hz–2 kHz | 30–50 % | 34 % | 3 % |
| stereo šířka (S/(M+S)) | 13–30 % | 8.5 % | 0.1 % |

Příčiny „čipového“ zvuku v kódu:

1. Každý nástroj je jeden hlas uprostřed — žádná sekce, žádná šířka u zdroje.
2. Strojová přesnost: tóny akordu startují ve stejném vzorku, přesně naladěné.
3. Lineární ADSR (decay/release jako přímka) — typické pro trackery a čipy.
4. Freeverb (kovový) ve starých songech.
5. Přetížený sub (subbass, 808, sub SFX) a tenké středy.
6. Report neumí říct, jak daleko je mix od reference.

## Rozhodnutí

| Rozhodnutí | Volba | Proč |
|---|---|---|
| Sekce nástrojů | `Ensemble` v patchi: N kopií hlasu, rozladění, časové rozmazání nástupu, rozprostření do stran | Nejsilnější rozdíl mezi „čipem“ a orchestrem/sborem; zároveň řeší šířku u zdroje |
| Obálky | `ADSR(curve="exp")` jako výchozí: decay a release exponenciálně, attack lehce konvexní | Analogové/akustické chování; lineární zůstane přes `curve="lin"` |
| Reverb | `Reverb` přejde na konvoluci se syntetickou IR (room → délka), Freeverb dostupný jako `Reverb(algo="freeverb")` | Staré songy dostanou lepší prostor bez úprav; API stejné |
| Staré presety | Melodické v1 presety dostanou drift, round-robin a (pady/smyčce) ensemble; `subbass` přesune energii z pod-60 Hz výš (víc 2. harmonické) | Staré songy zní méně čipově bez přepisu |
| Zpětná vazba | Report: pásma v 7 pásmech + cílové rozsahy; nová pravidla `sub přetížený`, `úzké stereo`, `tenké středy`; CLI `python -m paprika compare song.wav ref.mp3` | AI neslyší — potřebuje číslo, kterým se přiblíží referenci |
| Mastering | `s.master.fx(*mastering("cinematic"))` — HP 30 Hz, jemné EQ, glue kompresor, lehká páska | Jeden řádek na „produkovaný“ zvuk |

Zpětná kompatibilita API zůstává. Zvuk starých songů se **záměrně** změní
(to je cíl).

## Ensemble

```python
Ensemble(voices=4, detune=10.0, spread=0.8, smear=0.02, seed=0)
```

- `voices` kopií hlasu; kopie `i` má rozladění rovnoměrně v ±`detune` centů
  (plus malý náhodný posun), náhodnou fázi (round-robin index), zpoždění
  nástupu náhodně 0..`smear` s a pan rovnoměrně v ±`spread`.
- Výstup je vždy stereo, hlasitost normalizovaná `1/sqrt(voices)`.
- Pole v `Patch.key()`; stejná nota se stejným `rr_index` dá stejný výsledek
  (deterministické přes `seed`).
- Buffer hlasu se prodlouží o `smear`.

## Obálka

`ADSR(a, d, s, r, curve="exp")`: attack `1 - (1-x)^2` (rychlejší nástup,
měkký konec), decay `s + (1-s)·e^(-5x)`, release `level·e^(-5x)` s nulou na
konci (poslední vzorek 0). `curve="lin"` = chování v1. `curve` je v `key()`.

## Reverb

`Reverb(room, mix, damp, width, algo="conv")`: decay = `0.3 + 4.7·room²` s,
damp z parametru, predelay 0.01 + 0.03·room s; stereo IR z `dsp.synth_ir`
(cache), `width` míchá L/R IR. `tail()` = délka IR. `algo="freeverb"`
= původní algoritmus.

## Zpětná vazba

- Report: řádek `BANDS` se sedmi pásmy (sub, bass, lowmid, mid, himid,
  pres, air) a značkou `[ok]`/`[málo]`/`[moc]` podle cílových rozsahů
  „cinematic“ (odvozených z referencí: sub ≤ 12 %, středy 250 Hz–4 kHz ≥ 20 %,
  šířka ≥ 10 %).
- Pravidla (WARN, max 1 hlášení každé): `sub přetížený` (sub > 15 %),
  `úzké stereo` (šířka < 8 % a skladba delší než 8 taktů),
  `tenké středy` (250 Hz–4 kHz < 15 %).
- `python -m paprika compare A B [C ...]`: tabulka metrik (7 pásem, centroid,
  šířka, korelace, crest, LRA) pro každý soubor + rozdíl proti prvnímu.
  Metriky žijí v `paprika/metrics.py` (čte jen audio).

## Mastering

`mastering(style)` v `paprika/mastering.py` vrací nový seznam efektů:
`cinematic` (HighPass 30 Hz, EQ low-shelf −2 dB @ 90 Hz, high-shelf +1.5 dB
@ 8 kHz, Compressor −14 dB 2:1 30 ms, Tape wow 0.05 drive 1.2), `darkpop`
(víc drive, kompresor 3:1), `lofi` (Tape wow 0.4, hiss, BitCrush 12 bit,
LowPass 9 kHz). Neznámý styl → `KeyError` se seznamem.

## Presety

- v2 presety se sekcí dostanou `Ensemble`: `strings_ens` (6 hlasů), `choir_*`
  (5), `brass_section` (3), `horn_swell` (2), `pwm_pad`, `shimmer_pad`,
  `string_stab` (3).
- v1 melodické presety (`epiano`, `warmpad`, `pluck`, `lead`, `strings`,
  `bell`, `organ`): `drift` 3–5 centů, `rr` 2–3; `warmpad` a `strings`
  `Ensemble(3–4)`.
- `subbass`: méně čistého sinu pod 60 Hz, víc 2./3. harmonické (slyšet i na
  malých reproduktorech).

## Ukázka

`songs/v2_showcase.py` přepsat „proti čipu“: místo 1/16 arpeggia pomalejší
rozložené akordy a legato linky, sekce přes ensemble, mastering „cinematic“.
Cíl měřitelně: sub ≤ 12 %, šířka ≥ 12 %, středy ≥ 25 %.

## Testy

- Ensemble: stereo, korelace L/R < 0.9, nástup rozmazaný (první nenulový
  vzorek kopií se liší), deterministické, klíč cache se mění.
- ADSR exp: decay konvexní (hodnota v polovině decaye pod lineární), končí
  nulou, `lin` = v1 hodnoty.
- Reverb conv: tail ≤ `tail()`, větší room zní déle, mix 0 průhledné,
  freeverb přepínač vrací v1 výstup.
- Metriky: sinus 40 Hz → sub ~100 %, šum L/R nezávislý → šířka ~50 %.
- Pravidla: sub přetížený / úzké stereo / tenké středy se spustí a nespustí.
- `compare` CLI vypíše tabulku pro dva wav.
- Mastering: každý styl vrací nové instance, zpracuje buffer bez NaN.
- Presety: stávající testy úrovní a odlišnosti dál platí.
- Showcase: metriky v cíli (měřeno ručně při psaní, ne v unittestu).
