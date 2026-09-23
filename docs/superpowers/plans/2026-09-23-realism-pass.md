# Realism Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Odstranit „16bitový / čipový“ charakter: sekce (ensemble), exponenciální obálky, konvoluční Reverb, oživené staré presety, mix metriky proti referenci, mastering.

**Architecture:** `Ensemble` žije v `modulation.py` a zpracuje ho `Patch.voice` (N kopií jádra hlasu). `ADSR` dostane `curve`. `Reverb` deleguje na konvoluci sdílenou s `Hall`. Nový `metrics.py` (čistě audio) používá report i CLI `compare`. `mastering.py` skládá existující efekty.

**Tech Stack:** numpy, scipy, unittest.

**Spec:** `docs/superpowers/specs/2026-09-23-realism-pass-design.md`

## Global Constraints

- Závislosti jen numpy + scipy. Vše v beatech. Každé nové pole v `key()`.
- Česky docstringy, chyby, report. Pravidla analýzy max 1–4 hlášení.
- Žádné per-sample Python cykly.
- API zpětně kompatibilní; zvuk starých songů se smí změnit.

## Review Focus

1. Ensemble + glide/vibrato/pitch env: kopie musí sdílet křivku, jen posunutou o detune — test v Task 1.
2. Ensemble + `--bars` / cache: stejný `rr_index` → stejný buffer (determinismus) — test v Task 1.
3. Exponenciální release musí skončit nulou (jinak klik na konci hlasu) — test v Task 2.
4. Konvoluční Reverb na dlouhé skladbě nesmí explodovat pamětí/časem (FFT na celé délce) — měření v Task 3.
5. `compare` na mono souboru / krátkém souboru nespadne — test v Task 4.

---

### Task 1: Ensemble

**Files:** Modify `paprika/modulation.py`, `paprika/patch.py`, `paprika/__init__.py`; Test `tests/test_ensemble.py`

**Interfaces:**
- `Ensemble(voices=4, detune=10.0, spread=0.8, smear=0.02, seed=0)`; `key()`; `layout(rr_index) -> list[(cents, delay_s, pan, phase_idx)]` deterministicky podle `seed` a `rr_index`.
- `Patch(..., ensemble=None)`; `Patch.voice` při `ensemble`: jádro hlasu (oscilátory + obálka + filtry) se spočítá pro každou kopii s `semis + cents/100` a `rr_index*voices + i` jako varianta fáze; mono kopie se panují equal-power, stereo kopie se balancují; posun o `delay_s`; součet × `1/sqrt(voices)`; délka + `smear`.
- Tests: stereo tvar; korelace < 0.9; nástupy kopií se liší (první vzorek nad prahem v L a R se liší o > 1 ms při smear 0.02); determinismus; `key` se liší s/bez ensemble; s vibratem konečné a ve výšce (FFT špička ±3 %).

### Task 2: Exponenciální ADSR

**Files:** Modify `paprika/patch.py`; Test `tests/test_patch.py` (nové třídy)

**Interfaces:** `ADSR(a, d, s, r, curve="exp")`, `CURVES = ("exp", "lin")`, neznámá → `ValueError`; `key()` obsahuje `curve`.
- exp: attack `1-(1-x)^2`, decay `s+(1-s)·exp(-5x)` (normalizované tak, aby na konci segmentu bylo přesně `s`), release `level·(exp(-5x)-exp(-5))/(1-exp(-5))` → přesně 0 na konci.
- Tests: `lin` = v1 hodnoty (bitově); exp decay v půlce pod lineárním; poslední vzorek release 0; sustain level drží; stávající ADSR testy platí pro obě křivky.

### Task 3: Konvoluční Reverb

**Files:** Modify `paprika/fx.py`, `paprika/fx_space.py` (sdílená konvoluce); Test `tests/test_fx.py`

**Interfaces:** `Reverb(room=0.6, mix=0.3, damp=0.4, width=1.0, name=None, algo="conv")`; `ALGOS = ("conv", "freeverb")`; conv: decay `0.3 + 4.7·room²`, predelay `0.01 + 0.03·room`, IR přes `fx_space._ir_cached(decay, predelay, damp, seed, sr)`, `width` míchá L/R IR (`L' = w1·L + w2·R`); `tail()` = `decay·1.1 + predelay`; freeverb = v1 kód.
- Tests: stávající Reverb testy (tail, bigger room, zero mix, finite) pro oba algoritmy; freeverb výstup = v1 (uložený referenční hash není potřeba — test porovná s `_freeverb_process` přímo); conv stereo korelace wet < 0.3.
- Změřit render `iron_requiem` (čas, paměť) — nesmí být > 1.5× pomalejší než před změnou.

### Task 4: Metriky, report, pravidla, `compare`

**Files:** Create `paprika/metrics.py`, `tests/test_metrics.py`; Modify `paprika/analyze.py`, `paprika/cli.py`

**Interfaces:**
- `metrics.BANDS7 = (("sub",0,60),("bass",60,250),("lowmid",250,500),("mid",500,2000),("himid",2000,4000),("pres",4000,8000),("air",8000,None))`
- `band_shares(audio, sr) -> dict[str, float]` (procenta, součet 100), `stereo_width(audio) -> float` (%), `correlation(audio) -> float|None`, `crest_db(audio)`, `loudness_range(audio, sr)`, `centroid(audio, sr)`, `summary(audio, sr) -> dict`.
- `TARGETS = {"sub": (0, 12), "mids": (20, 100), "width": (10, 60)}` (mids = lowmid+mid+himid).
- Report: `BANDS` řádek se 7 pásmy + `[ok]/[zkontroluj]`; `STEREO` řádek zůstává.
- Pravidla `_sub_heavy` (sub > 15 %), `_narrow` (šířka < 8 % a song > 8 taktů), `_thin_mids` (mids < 15 %) — WARN „mix“, každé max 1.
- CLI `compare a b [...]`: načte přes `reference.load_reference_audio`-like stereo loader (wav přes `wave`, ostatní přes ffmpeg), tabulka metrik + rozdíl proti prvnímu. Mono soubor → šířka 0, korelace „—“.
- Tests: sinus 40 Hz → sub > 90 %; nezávislý šum L/R → šířka 40–60 %; identické L=R → 0; pravidla spustí/nespustí; `compare` na dvou wav (tmp) vypíše „sub“ a oba názvy; mono wav nespadne.
- Úprava existujících testů reportu jen pokud testují starý 3-pásmový formát.

### Task 5: Mastering

**Files:** Create `paprika/mastering.py`, `tests/test_mastering.py`; Modify `paprika/__init__.py`

**Interfaces:** `mastering(style="cinematic") -> list[Fx]`, `MASTERING_STYLES = ("cinematic", "darkpop", "lofi")`; neznámý → `KeyError` se seznamem.
- Tests: nové instance při každém volání; každý styl zpracuje stereo šum bez NaN a peak < 2; `s.master.fx(*mastering())` render projde.

### Task 6: Presety proti čipu

**Files:** Modify `paprika/presets.py`; Test `tests/test_presets.py`

- v2 sekce s `Ensemble` podle specu; v1 melodické presety: drift + rr, `warmpad`/`strings` ensemble; `subbass` víc harmonických (2. harmonická 0.35, 3. 0.12, sine amp 0.8).
- Tests: stávající (úroveň < 1.0 přes rozsah, odlišnost, „not static“) platí; nový: `strings_ens`, `choir_ooh`, `warmpad` mají ensemble a stereo výstup; `subbass` má podíl energie pod 60 Hz na A1 (55 Hz) < 70 %.
- Doladit gainy (ensemble mění úroveň).

### Task 7: Showcase proti čipu + dokumentace

**Files:** Modify `songs/v2_showcase.py`, `README.md`, `CLAUDE.md`

- Showcase: pryč 1/16 arpeggio pluck → pomalé rozložené akordy (`rate=1/8`, `octaves=1`) na `harp`/`felt_piano` + `strings_short` staccato; mastering „cinematic“; 808 hlasitost tak, aby sub ≤ 12 %.
- Ověřit metrikami (`compare` proti oběma referencím): sub ≤ 12 %, šířka ≥ 12 %, středy ≥ 25 %.
- README: Ensemble, ADSR curve, Reverb algo, `compare`, `mastering`, nová pravidla, „jak se vyhnout čipovému zvuku“ (krátký seznam pro autora songů).
- CLAUDE.md: metrics.py, mastering.py, ensemble v cache klíči, `compare` v commands.
