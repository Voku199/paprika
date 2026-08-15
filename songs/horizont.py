"""Horizont — instrumentální post-rock, cinematic, ambient-influenced.

Oblouk: sparse contemplative opening -> gradual build s clean kytarou ->
tension-driving střed se zkreslením -> climactic crescendo s vrstvenými leady
-> minimalistický dozvuk.

E moll (E F# G A B C D) drží introspektivní barvu, dur zvedy dělá loop
C - G - D ve vrcholu (VI - III - VII), kde se moll na chvíli otevře.
Tempo je jedno (84 BPM), gradace jde přes hustotu a zkreslení: čtvrťové
arpeggio v úvodu, osminové ostinato uprostřed, chugging power akordy
ve vrcholu, pak zpátky k jedné kytaře.

72 taktů @ 84 BPM = 3:26 hudby, s ocáskem reverbu ~3:32.

    python -m paprika show   songs/horizont.py --bars 44:48
    python -m paprika render songs/horizont.py
"""

from paprika import (ADSR, Delay, HighPass, LP, LowPass, Patch, Reverb,
                     Saturate, Saw, Sine, Song, Square, Tri, preset)

s = Song(bpm=84, key="E", scale="minor", bars=72, title="horizont")

BPB = 4


def b(bar, beat=0.0):
    """Takt (+ beat uvnitř taktu) -> pozice na timeline v beatech."""
    return bar * BPB + beat


def chug(root, fifth, n=8, dur=0.5):
    """Osminkové opakování power akordu na jeden takt."""
    return " ".join([f"({root} {fifth}):{dur}"] * n)


# --------------------------------------------------------------------------
# nástroje
# --------------------------------------------------------------------------

# Akustika: trojúhelník jako tělo struny, špetka pily za nehet, rychlý decay
# a skoro nulový sustain — drnknutí, které dozní samo.
ACOU = Patch(
    osc=[Tri(1.0, amp=0.90), Saw(1.0, amp=0.22), Sine(2.0, amp=0.18),
         Sine(3.01, amp=0.07)],
    adsr=ADSR(0.005, 1.25, 0.06, 0.55),
    filter=LP(2900, env=0.60),
    gain=0.55,
    name="acou",
)

# Clean elektrika: sinus + trojúhelník (skoro čistý tón), dlouhý sustain,
# zavřený filtr. Charakter dělá až dotted delay a reverb.
CLEAN = Patch(
    osc=[Sine(1.0, amp=1.0), Tri(1.0, amp=0.45), Saw(2.0, amp=0.10),
         Sine(3.0, amp=0.06)],
    adsr=ADSR(0.012, 1.9, 0.38, 0.95),
    filter=LP(2300, env=0.45),
    gain=0.50,
    name="clean",
)

# Zkreslená elektrika: tři rozladěné pily do saturace. Kvinta v oscilátorech
# (Square 1.5) přidá to typické "zeď", i když se hrají jen power akordy.
DIST = Patch(
    osc=[Saw(1.0, amp=0.55), Saw(1.006, amp=0.45), Saw(0.994, amp=0.45),
         Square(1.5, amp=0.14), Sine(0.5, amp=0.12)],
    adsr=ADSR(0.02, 0.7, 0.78, 0.65),
    filter=LP(3000, env=0.30),
    gain=0.40,
    name="dist",
)

# Lead: preset lead s níž posazeným filtrem, aby netrhal uši nad zkreslením.
LEAD = preset("lead")
LEAD.filter = LP(2600, env=0.40)
LEAD.adsr = ADSR(0.05, 0.5, 0.70, 0.8)
LEAD.gain = 0.50

# Stopy. Kytary jsou rozhozené do stran, bas a bicí zůstávají uprostřed —
# tam se stereo obraz drží pohromadě i po limiteru.
acou = s.track(ACOU, vol=0.72, pan=-0.35, name="acou")
acou.fx(HighPass(150), Reverb(room=0.88, mix=0.38, damp=0.45),
        Delay(1 / 8, fb=0.22, mix=0.14))

clean = s.track(CLEAN, vol=0.52, pan=0.32, name="clean")
clean.fx(HighPass(170), Reverb(room=0.90, mix=0.46, damp=0.40),
         Delay(3 / 16, fb=0.34, mix=0.26))          # tečkovaná osminka

# Zkreslené kytary mají highpass výš, než by se zdálo: pod 200 Hz nemají
# co dělat, tam patří bas. Jinak vrchol zežloutne do jedné basové kaše.
distL = s.track(DIST, vol=0.26, pan=-0.45, name="distL")
distL.fx(Saturate(5.0, mix=0.92), HighPass(210), LowPass(4200),
         Reverb(room=0.90, mix=0.34, damp=0.55))

distR = s.track(DIST, vol=0.24, pan=0.45, name="distR")
distR.fx(Saturate(4.0, mix=0.88), HighPass(230), LowPass(3800),
         Reverb(room=0.88, mix=0.30, damp=0.60))

lead = s.track(LEAD, vol=0.54, pan=0.12, name="lead")
lead.fx(HighPass(300), LowPass(5200), Reverb(room=0.94, mix=0.52, damp=0.35),
        Delay(1 / 4, fb=0.30, mix=0.22))

pad = s.track("warmpad", vol=0.38, pan=0.22, name="pad")
pad.fx(HighPass(280), LowPass(2500), Reverb(room=0.92, mix=0.50, damp=0.50))

strings = s.track("strings", vol=0.42, pan=-0.22, name="strings")
strings.fx(HighPass(220), LowPass(3200), Reverb(room=0.90, mix=0.46, damp=0.50))

bass = s.track("subbass", vol=0.21, name="bass")
bass.fx(HighPass(45), LowPass(800), Saturate(1.5, mix=0.4))

dr = s.drums(vol=0.38)

# kořeny basu pro každý akord skladby
ROOT = {"Em": "E2", "Em9": "E2", "Cmaj7": "C2", "G": "G1", "G6": "G1",
        "D": "D2", "Dsus4": "D2", "Am7": "A1", "Bm7": "B1"}
# power akordy (kořen, kvinta) pro zkreslené kytary
POWER = {"Em": ("E2", "B2"), "Em9": ("E2", "B2"), "Cmaj7": ("C3", "G3"),
         "G": ("G2", "D3"), "G6": ("G2", "D3"), "D": ("D3", "A3"),
         "Dsus4": ("D3", "A3"), "Am7": ("A2", "E3"), "Bm7": ("B2", "F#3")}


# --------------------------------------------------------------------------
# A  takty 0-11 — sparse contemplative opening
# --------------------------------------------------------------------------
# Jen akustika, po dvou taktech jeden akord. První čtyři takty čtvrťové
# arpeggio (hodně vzduchu mezi tóny), od taktu 4 osminové a nastupuje pad.

A_PROG = [("Em", 0), ("Cmaj7", 2), ("G", 4), ("Dsus4", 6),
          ("Em", 8), ("Am7", 10)]

for name, bar in A_PROG:
    rate = 1 / 4 if bar < 4 else 1 / 8
    acou.arp(name, bar=bar, pattern="updown", rate=rate, octaves=2, bars=2,
             octave=3, vel=52 + bar)
    if bar >= 4:
        pad.chord(name, bar=bar, dur=8, octave=4, inv=1, vel=38 + bar)

# osamocené tóny nad arpeggiem — melodie, která se zatím jen ptá
acou.note("B4", at=b(3, 2), dur=2, vel=62)
acou.note("D5", at=b(7, 2), dur=2, vel=66)
acou.note("E5", at=b(11, 0), dur=3, vel=70)


# --------------------------------------------------------------------------
# B  takty 12-27 — gradual build, clean guitar chords
# --------------------------------------------------------------------------
# Čtyřtaktová smyčka Em - Cmaj7 - G - D, jeden akord na takt. Přibývá bas,
# clean kytara a kartáče. Velocity roste přes celou sekci.

B_LOOP = ["Em", "Cmaj7", "G", "D"]

for i in range(16):
    bar = 12 + i
    name = B_LOOP[i % 4]
    grow = i * 2                                   # 0 -> 30 přes celou sekci

    acou.arp(name, bar=bar, pattern="up", rate=1 / 8, octaves=2,
             octave=3, vel=54 + grow // 2)
    clean.chord(name, bar=bar, dur=4, octave=3, inv=1, vel=40 + grow)
    pad.chord(name, bar=bar, dur=4, octave=4, inv=1, vel=42 + grow // 2)
    if bar >= 16:                                  # bas až se skladbou, ne hned
        bass.notes(f"{ROOT[name]}:4", bar=bar, vel=58 + grow)

    if bar >= 24:                                  # smyčce se přidají v půlce
        if i % 2 == 0:
            strings.chord(name, bar=bar, dur=8, octave=3, vel=52 + grow)

# G6 místo prostého G na taktu 22 — první durový záblesk
clean.clear(bars=(22, 23))
clean.chord("G6", bar=22, dur=4, octave=3, inv=1, vel=72)

# kartáče: tichý hi-hat na osminách, snare na trojku
dr.pattern("""
..h...h...h...h.
........s.......
""", bars=(16, 24), vel=0.42)

# od taktu 24 kopák a hustší puls, pořád ještě potichu
dr.pattern("""
..h.h.h...h.h.h.
........s.......
k.......k...k...
""", bars=(24, 28), vel=0.55)


# --------------------------------------------------------------------------
# C  takty 28-43 — tension-driving middle section
# --------------------------------------------------------------------------
# Osminové ostinato v clean kytaře, harmonie se posune do Am7 - Em - Cmaj7 -
# Bm7 (Bm7 je ten tah, který nikam nepustí). Od taktu 36 nastupuje zkreslení
# a tomy tlačí do vrcholu.

C1 = ["Am7", "Em", "Cmaj7", "Bm7"]
C2 = ["Em", "D", "Cmaj7", "Bm7"]

for i in range(16):
    bar = 28 + i
    name = (C1 if i < 8 else C2)[i % 4]
    grow = i * 2

    clean.arp(name, bar=bar, pattern="updown", rate=1 / 8, octaves=2,
              octave=3, vel=64 + grow)
    acou.arp(name, bar=bar, pattern="up", rate=1 / 8, octaves=1,
             octave=3, vel=46 + grow // 2)          # pod povrchem, jen textura
    pad.chord(name, bar=bar, dur=4, octave=4, inv=1, vel=52 + grow // 2)
    if i % 2 == 0:
        strings.chord(name, bar=bar, dur=8, octave=3, inv=1, vel=60 + grow)

    # Bas: půlové noty, od taktu 36 s osminovým odrazem na čtyřce. Noty končí
    # o zlomek beatu dřív — Cmaj7 a Bm7 mají kořeny C2 a B1 vedle sebe a
    # držené by se v base potkaly v malé sekundě.
    root = ROOT[name]
    if bar < 36:
        bass.notes(f"{root}:2 ~:1.9", bar=bar, vel=76 + grow)
    else:
        bass.notes(f"{root}:2 {root}:1.5 {root}:0.4", bar=bar, vel=84 + grow)

    # zkreslení: držené power akordy, pravá kytara o osminu posunutá
    if bar >= 36:
        r, f = POWER[name]
        distL.notes(f"({r} {f}):4", bar=bar, vel=72 + grow)
        distR.notes(f"_:0.5 ({r} {f}):3.5", bar=bar, vel=64 + grow)

# lead: čtyřtónový motiv, pokaždé o kus výš — první náznak melodie vrcholu
for bar, seq in ((32, "B4:1 D5:1 E5:2"), (34, "D5:1 E5:1 G5:2"),
                 (38, "E5:1 G5:1 A5:2"), (40, "G5:1 A5:1 B5:2"),
                 (42, "A5:1.5 B5:0.5 E5:2")):
    lead.melody(seq, bar=bar, vel=70 + (bar - 32) * 2)

# ride místo hi-hatu, backbeat na dvojce a čtyřce
dr.pattern("""
r.r.r.r.r.r.r.r.
....s.......s...
k.......k..k....
""", bars=(28, 36), vel=0.62)

dr.pattern("""
r.r.r.r.r.r.r.r.
....S.......S...
K.......k..k....
""", bars=(36, 42), vel=0.78)

# dva takty tomového náběhu — jediná věc, která hraje před vrcholem sólo
dr.pattern("""
K...t.t.K...t.t.
....S.......S...
""", bars=(42, 43), vel=0.85)
dr.pattern("""
K.t.t.T.t.T.tTtT
....S.......S..o
""", bars=(43, 44), vel=0.95)


# --------------------------------------------------------------------------
# D  takty 44-59 — climactic crescendo
# --------------------------------------------------------------------------
# Smyčka C - G - D - Em: tři durové akordy za sebou jsou to otevření, kvůli
# kterému celá skladba stoupala. Všechno hraje naráz, vrstvené leady nahoře,
# tomy dole.

D_LOOP = ["Cmaj7", "G", "D", "Em"]

for i in range(16):
    bar = 44 + i
    name = D_LOOP[i % 4]
    peak = 8 if bar >= 52 else 0                   # druhá půlka ještě o kus výš
    r, f = POWER[name]

    distL.notes(f"({r} {f}):4", bar=bar, vel=96 + peak)
    distR.notes(chug(r, f), bar=bar, vel=84 + peak)
    clean.arp(name, bar=bar, pattern="up", rate=1 / 8, octaves=2,
              octave=4, vel=78 + peak)
    acou.arp(name, bar=bar, pattern="up", rate=1 / 8, octaves=1,
             octave=3, vel=54)
    pad.chord(name, bar=bar, dur=4, octave=4, inv=1, vel=66 + peak)
    strings.chord(name, bar=bar, dur=4, octave=3, inv=1, vel=82 + peak)
    bass.notes(f"{ROOT[name]}:1.5 {ROOT[name]}:0.5 {ROOT[name]}:2",
               bar=bar, vel=98 + peak)

# hlavní téma vrcholu: dvoutaktová fráze, čtyřikrát, pokaždé s jiným koncem
THEME = [(44, "B4:1 E5:1 G5:1 A5:1 | B5:2 A5:1 G5:1"),
         (48, "B4:1 E5:1 G5:1 B5:1 | D6:2 B5:1 A5:1"),
         (52, "E5:1 G5:1 B5:1 D6:1 | E6:2 D6:1 B5:1"),
         (56, "D6:1 B5:1 A5:1 G5:1 | E5:4")]
for bar, seq in THEME:
    lead.melody(seq, bar=bar, vel=88 if bar < 52 else 100)

# druhá vrstva leadu o oktávu níž, jen v druhé půlce — zesílí tělo melodie
lead.melody("E5:1 G5:1 B5:1 D6:1 | E5:2 D5:1 B4:1", bar=52, vel=72)
lead.melody("B4:1 A4:1 G4:1 E4:1 | E4:4", bar=56, vel=68)

dr.pattern("""
K...t...K...t...
....S.......S...
o...............
""", bars=(44, 52), vel=0.9)

dr.pattern("""
K..tT...K..tT..t
....S.......S...
o.......o.......
""", bars=(52, 60), vel=1.0)


# --------------------------------------------------------------------------
# E  takty 60-71 — recede to minimalist finale
# --------------------------------------------------------------------------
# Zkreslení dozní na jednom drženém akordu, bicí utichnou, zbyde akustika
# a poslední Em9 — stejný tón, kterým to začalo, jen o poznání dál.

distL.notes("(E2 B2):8", bar=60, vel=84)
distR.notes("(E3 B3):8", bar=60, vel=70)
dr.pattern("K...t...t.......", bars=(60, 61), vel=0.7)
dr.hit("tom", at=b(61, 0), vel=70)
dr.hit("openhat", at=b(61, 2), vel=54)

for i, (name, bar) in enumerate([("Em", 60), ("Cmaj7", 62)]):
    strings.chord(name, bar=bar, dur=8, octave=3, inv=1, vel=76 - i * 12)
    pad.chord(name, bar=bar, dur=8, octave=4, inv=1, vel=60 - i * 10)
    clean.arp(name, bar=bar, pattern="up", rate=1 / 8, octaves=2, bars=2,
              octave=3, vel=62 - i * 12)
    acou.arp(name, bar=bar, pattern="up", rate=1 / 8, octaves=1, bars=2,
             octave=3, vel=54 - i * 8)
    bass.notes(f"{ROOT[name]}:4", bar=bar, vel=70 - i * 10)
    bass.notes(f"{ROOT[name]}:4", bar=bar + 1, vel=64 - i * 10)

# lead se stahuje po jednotlivých tónech, ať mezi vrcholem a koncem nezůstane
# hluché místo
lead.note("B5", at=b(60, 2), dur=3, vel=64)
lead.melody("B4:2 E5:2", bar=62, vel=58)

# poslední čtyři takty smyčky, každý průchod řidší
for name, bar, vel in [("Em", 64, 50), ("Cmaj7", 66, 42),
                       ("Em", 68, 36)]:
    acou.arp(name, bar=bar, pattern="up", rate=1 / 4, octaves=2, bars=2,
             octave=3, vel=vel)

pad.chord("Em9", bar=64, dur=16, octave=4, inv=1, vel=44)
strings.chord("Em", bar=66, dur=12, octave=3, inv=1, vel=40)

acou.note("E5", at=b(67, 2), dur=2, vel=52)
acou.note("B4", at=b(69, 2), dur=2, vel=46)
acou.chord("Em9", bar=70, dur=8, octave=3, vel=44)

lead.note("G5", at=b(66, 2), dur=3, vel=48)
lead.note("E5", at=b(70), dur=6, vel=44)

bass.note("C2", at=b(66), dur=3.9, vel=54)
bass.note("E2", at=b(70), dur=8, vel=52)


# --------------------------------------------------------------------------
# rozhýbání — bez toho to zní jako sekvencer, ne jako kapela
# --------------------------------------------------------------------------

acou.humanize(timing=0.018, vel=0.16, seed=3)
clean.humanize(timing=0.015, vel=0.14, seed=7)
distL.humanize(timing=0.012, vel=0.08, seed=11)
distR.humanize(timing=0.020, vel=0.10, seed=13)     # pravá kytara volnější
lead.humanize(timing=0.022, vel=0.12, seed=17)
pad.humanize(timing=0.05, vel=0.08, seed=19)
strings.humanize(timing=0.04, vel=0.09, seed=23)
bass.humanize(timing=0.010, vel=0.07, seed=29)
dr.humanize(timing=0.008, vel=0.13, seed=31)

s.render("output/horizont.wav")
