"""Verdikt — dark ambient / cinematic psychological thriller score.

Oblouk: quiet unease -> sudden dread -> creeping tension -> grave realization.
D harmonická moll (D E F G A Bb C#) drží celou skladbu v napětí: zvětšená
sekunda Bb-C# a zmenšený septakord C#dim7 dělají tu "rozhodovací" barvu.
Tempo je jedno (72 BPM), gradace jde přes hustotu — celé noty v úvodu,
osminové ostinato uprostřed, těžký půlový puls ve vrcholu.

    python -m paprika show   songs/verdikt.py --bars 40:44
    python -m paprika render songs/verdikt.py
"""

from paprika import (ADSR, HP, Delay, HighPass, LP, LowPass, Noise, Patch,
                     Reverb, Saturate, Saw, Sine, Song, Tri, preset)

s = Song(bpm=72, key="D", scale="harmonic_minor", bars=64, title="verdikt")

BPB = 4


def b(bar, beat=0.0):
    """Takt (+ beat uvnitř taktu) -> pozice na timeline v beatech."""
    return bar * BPB + beat


# --------------------------------------------------------------------------
# nástroje
# --------------------------------------------------------------------------

# Sub drone: sinus + lehce rozladěná oktáva, pomalý nájezd, skoro žádné výšky.
DRONE = Patch(
    osc=[Sine(1.0, amp=1.0), Sine(2.005, amp=0.28), Tri(0.5, amp=0.20),
         Saw(1.004, amp=0.05)],
    adsr=ADSR(2.2, 2.5, 0.92, 4.0),
    filter=LP(340, env=0.12),
    gain=0.60,
    name="drone",
)

# Sbor: rozladěné pily (šum hlasů) + sudé harmonické (vokálnější barva),
# zavřený filtr a dlouhý attack = vzdálený mužský sbor, žádný text.
CHOIR = Patch(
    osc=[Saw(1.0, amp=0.34), Saw(1.007, amp=0.30), Saw(0.993, amp=0.30),
         Sine(2.0, amp=0.22), Sine(3.0, amp=0.08), Sine(0.5, amp=0.15)],
    adsr=ADSR(1.1, 1.6, 0.82, 2.6),
    filter=LP(1150, env=0.20),
    gain=0.40,
    name="choir",
)

# Hluboké smyčce: preset strings s níž posazeným filtrem a delším smykem.
LOWSTR = preset("strings")
LOWSTR.filter = LP(950, env=0.20)
LOWSTR.adsr = ADSR(0.9, 1.4, 0.85, 2.4)
LOWSTR.gain = 0.42

# Hlasitosti: drone a puls drží spodek, ale nesmí sežrat celé pásmo —
# všechno ostatní má highpass, aby zůstalo nad 130 Hz.
drone = s.track(DRONE, vol=0.34, name="drone")
drone.fx(LowPass(420), Reverb(room=0.90, mix=0.28, damp=0.75))

strings = s.track(LOWSTR, vol=0.52, pan=-0.15, name="strings")
strings.fx(HighPass(130), Reverb(room=0.88, mix=0.42, damp=0.55))

pad = s.track("warmpad", vol=0.36, pan=0.20, name="pad")
pad.fx(HighPass(240), LowPass(2600), Reverb(room=0.90, mix=0.45, damp=0.50))

choir = s.track(CHOIR, vol=0.46, name="choir")
choir.fx(HighPass(190), Reverb(room=0.92, mix=0.50, damp=0.50))

piano = s.track("epiano", vol=0.46, pan=-0.05, name="piano")
piano.fx(HighPass(160), Reverb(room=0.85, mix=0.44, damp=0.50),
         Delay(1 / 4, fb=0.32, mix=0.20))

bell = s.track("bell", vol=0.36, pan=0.25, name="bell")
bell.fx(HighPass(300), Reverb(room=0.95, mix=0.55, damp=0.35),
        Delay(3 / 8, fb=0.35, mix=0.20))

osti = s.track("pluck", vol=0.44, pan=0.30, name="osti")
osti.fx(HighPass(220), LowPass(2200), Reverb(room=0.80, mix=0.35),
        Delay(1 / 8, fb=0.28, mix=0.16))

# Vzduch: filtrovaný šum v pomalých nádeších. Jediný zdroj výšek v mixu.
AIR = Patch(
    osc=[Noise(1.0, amp=0.6, seed=4), Sine(8.0, amp=0.04)],
    adsr=ADSR(2.6, 3.0, 0.70, 3.2),
    filter=HP(4200),
    gain=0.45,
    name="air",
)

air = s.track(AIR, vol=0.85, pan=-0.3, chromatic=True, name="air")
air.fx(HighPass(3500), Reverb(room=0.95, mix=0.50, damp=0.25))

pulse = s.track("subbass", vol=0.28, name="pulse")
pulse.fx(LowPass(700), Saturate(1.6, mix=0.5))

dr = s.drums(vol=0.34)

# kořeny basu a ostinata pro každý akord
ROOT = {"Dm": "D1", "Dmmaj7": "D1", "Bb": "Bb1", "Gm": "G1", "A7": "A1",
        "C#dim7": "C#1"}
PULSE_ROOT = {"Dm": "D2", "Dmmaj7": "D2", "Bb": "Bb1", "Gm": "G1", "A7": "A1",
              "C#dim7": "C#2"}
CELL = {"Dm": "D3 A3 F3 A3", "Bb": "Bb2 F3 D3 F3", "Gm": "G2 D3 Bb2 D3",
        "A7": "A2 E3 C#3 E3", "C#dim7": "C#3 G3 E3 G3",
        "Dmmaj7": "D3 A3 F3 C#4"}


def cell(name):
    """Osminové ostinato na jeden takt (4 tóny dvakrát za sebou)."""
    return " ".join(f"{p}:0.5" for p in CELL[name].split() * 2)


# --------------------------------------------------------------------------
# A  takty 0-11 — quiet unease
# --------------------------------------------------------------------------

drone.note("D1", at=b(0), dur=32, vel=62)
# končí přesně na hraně taktu 12 — jinak by se D1 překrylo s C#1 sekce B
drone.note("D1", at=b(8), dur=16, vel=58)

pad.chord("Dm", bar=4, dur=16, octave=3, inv=1, vel=46)
pad.chord("Dm", bar=8, dur=16, octave=3, inv=2, vel=52)

strings.note("D2", at=b(6), dur=12, vel=44)

# klavír: jednotlivé tóny s velkými dírami, nic se zatím neopakuje
piano.note("D4", at=b(1, 2), dur=2, vel=58)
piano.note("F4", at=b(3, 0), dur=1.5, vel=52)
piano.note("A4", at=b(5, 2), dur=1, vel=60)
piano.note("D4", at=b(7, 0), dur=3, vel=54)
piano.note("E4", at=b(9, 2), dur=1, vel=56)
piano.note("F4", at=b(10, 3), dur=1, vel=50)
piano.note("C#5", at=b(11, 1), dur=3, vel=62)

# --------------------------------------------------------------------------
# B  takty 12-23 — sudden dread
# --------------------------------------------------------------------------

# zvon + malá sekunda C#/D o čtvrtinu později: to je ten úlek
bell.note("C#5", at=b(12), dur=6, vel=104)
bell.note("D5", at=b(12, 0.25), dur=6, vel=86)

DREAD = [("C#dim7", 12, 8), ("Bb", 14, 8), ("Gm", 16, 8), ("A7", 18, 8),
         ("C#dim7", 20, 4), ("Bb", 21, 4), ("Gm", 22, 4), ("A7", 23, 4)]

for name, bar, dur in DREAD:
    strings.chord(name, bar=bar, dur=dur, octave=3, vel=58)
    pad.chord(name, bar=bar, dur=dur, octave=4, inv=1, vel=40)
    drone.note(ROOT[name], at=b(bar), dur=dur, vel=62)

# klavír: dvoutónový motiv, pokaždé o stupeň jinde
for bar, (a, c) in zip((13, 15, 17, 19),
                       (("A4", "F4"), ("Bb4", "A4"),
                        ("G4", "Bb4"), ("C#5", "E5"))):
    piano.note(a, at=b(bar, 0), dur=1, vel=66 + (bar - 13) * 2)
    piano.note(c, at=b(bar, 1), dur=2, vel=60 + (bar - 13) * 2)

piano.note("D5", at=b(21, 2), dur=2, vel=70)
piano.note("C#5", at=b(23, 2), dur=2, vel=74)

# --------------------------------------------------------------------------
# C  takty 24-39 — creeping tension
# --------------------------------------------------------------------------

CPROG = ["Dm", "Dm", "Bb", "Bb", "Gm", "Gm", "A7", "A7"] * 2

for i, name in enumerate(CPROG):
    bar = 24 + i
    osti.notes(cell(name), bar=bar, vel=56 + i * 2)   # ostinato sílí 56 -> 86

for i in range(0, 16, 2):
    name = CPROG[i]
    bar = 24 + i
    strings.chord(name, bar=bar, dur=8, octave=3, vel=58 + i)
    pad.chord(name, bar=bar, dur=8, octave=3, inv=1, vel=42 + i)
    drone.note(ROOT[name], at=b(bar), dur=8, vel=64 + i)

# druhá půlka sekce: nastupuje těžký půlový puls v base
for i in range(8, 16):
    bar = 24 + i
    root = PULSE_ROOT[CPROG[i]]
    pulse.notes(f"{root}:1.6 _:0.4 {root}:1.6 _:0.4", bar=bar, vel=64 + i * 2)

# klavír: útržky nad ostinatem, směrem nahoru
piano.note("D5", at=b(26, 2), dur=2, vel=68)
piano.note("F5", at=b(28, 0), dur=1, vel=70)
piano.note("E5", at=b(28, 1), dur=3, vel=64)
piano.note("Bb4", at=b(30, 2), dur=2, vel=72)
piano.note("A4", at=b(33, 0), dur=2, vel=76)
piano.note("D5", at=b(35, 0), dur=1, vel=78)
piano.note("C#5", at=b(35, 1), dur=3, vel=74)
piano.note("E5", at=b(37, 2), dur=2, vel=80)
piano.note("F5", at=b(39, 0), dur=4, vel=86)

# vzdálené tomy jako tep, čtyři takty před vrcholem
dr.pattern("t.......t.......", bars=(36, 40), vel=0.55)

# --------------------------------------------------------------------------
# D  takty 40-55 — grave, heavy realization
# --------------------------------------------------------------------------

# osm taktů po dvou, pak osm po jednom — harmonie se zrychluje, tempo ne
DPROG = ([("Dmmaj7", 40, 8), ("Bb", 42, 8), ("Gm", 44, 8), ("A7", 46, 8)] +
         [(n, 48 + j, 4) for j, n in enumerate(
             ["Dm", "C#dim7", "Gm", "A7", "Dm", "C#dim7", "Gm", "A7"])])

for name, bar, dur in DPROG:
    heavy = 1 if bar >= 48 else 0
    choir.chord(name, bar=bar, dur=dur, octave=3, vel=74 + heavy * 8)
    strings.chord(name, bar=bar, dur=dur, octave=3, inv=1, vel=76 + heavy * 6)
    pad.chord(name, bar=bar, dur=dur, octave=4, inv=1, vel=48)
    piano.chord(name, bar=bar, dur=min(dur, 4), octave=4, vel=88 + heavy * 6)
    drone.note(ROOT[name], at=b(bar), dur=dur, vel=78)
    root = PULSE_ROOT[name]
    for k in range(int(dur // 4)):
        pulse.notes(f"{root}:1.6 _:0.4 {root}:1.6 _:0.4",
                    bar=bar + k, vel=96 + heavy * 6)

# kopák na první a třetí dobu — ne rytmus, jen váha
dr.pattern("K.......k.......", bars=(40, 48), vel=0.8)
dr.pattern("K.......k......t", bars=(48, 56), vel=0.95)

bell.note("D5", at=b(40), dur=8, vel=92)
bell.note("Bb4", at=b(44), dur=8, vel=84)
bell.note("C#5", at=b(48), dur=6, vel=96)
bell.note("D5", at=b(52), dur=6, vel=90)

# --------------------------------------------------------------------------
# E  takty 56-63 — doznívání, rozsudek platí
# --------------------------------------------------------------------------

choir.chord("Dm", bar=56, dur=16, octave=3, vel=64)
strings.chord("Dm", bar=56, dur=16, octave=3, inv=1, vel=56)
pad.chord("Dm", bar=56, dur=12, octave=3, inv=2, vel=40)
drone.note("D1", at=b(56), dur=28, vel=68)

piano.note("D4", at=b(56, 0), dur=4, vel=70)
piano.note("A3", at=b(58, 0), dur=4, vel=60)
piano.note("F3", at=b(60, 0), dur=4, vel=52)
piano.note("D3", at=b(62, 0), dur=6, vel=44)

bell.note("D5", at=b(60), dur=8, vel=70)

# --------------------------------------------------------------------------
# rozhýbání — bez toho zní všechno jako kvantizovaný stroj
# --------------------------------------------------------------------------

# vzduchové nádechy přes celou skladbu, sílí k vrcholu a pak mizí
for bar, vel, dur in [(2, 40, 10), (10, 52, 10), (18, 58, 10), (26, 64, 10),
                      (34, 72, 10), (40, 84, 10), (48, 90, 10), (56, 62, 12)]:
    air.note("D4", at=b(bar), dur=dur, vel=vel)

drone.humanize(timing=0.03, vel=0.06, seed=11)
air.humanize(timing=0.08, vel=0.10, seed=19)
strings.humanize(timing=0.04, vel=0.10, seed=3)
pad.humanize(timing=0.05, vel=0.08, seed=17)
choir.humanize(timing=0.06, vel=0.09, seed=23)
piano.humanize(timing=0.02, vel=0.14, seed=7)
bell.humanize(timing=0.015, vel=0.10, seed=5)
osti.humanize(timing=0.012, vel=0.16, seed=13)
pulse.humanize(timing=0.01, vel=0.08, seed=29)
dr.humanize(timing=0.008, vel=0.12, seed=31)

s.render("output/verdikt.wav")
