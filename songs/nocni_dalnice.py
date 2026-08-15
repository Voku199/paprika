"""Noční dálnice — synthwave, night-drive nálada.

A moll (A B C D E F G), 100 BPM. Smyčka Am7 - F - C - G (i-VI-III-VII) drží
melancholii mollu, ale oba durové akordy (F, C, G) jí dávají teplo — typický
synthwave trik: smutné akordy, ale hnací puls.

Oblouk: sparse intro (jen pad) -> build (bas, kartáče, náznak melodie) ->
první chorus (plný puls, lead melodie) -> breakdown (nádech) -> druhý,
větší chorus -> outro na tónice.

Bassline v chorusu je osminové pulzování, ne dlouhé tóny — spolu se
sidechainem od bicích dělá typické synthwave "dýchání".

64 taktů @ 100 BPM = 2:33 hudby, ocásek delaye/reverbu prodlouží na ~2:40.

    python -m paprika show   songs/nocni_dalnice.py --bars 24:28
    python -m paprika render songs/nocni_dalnice.py
"""

from paprika import Delay, HighPass, LowPass, Reverb, Saturate, Sidechain, Song

s = Song(bpm=100, key="A", scale="minor", bars=64, title="nocni_dalnice")

BPB = 4


def b(bar, beat=0.0):
    """Takt (+ beat uvnitř taktu) -> pozice na timeline v beatech."""
    return bar * BPB + beat


def pulse(note, n=8, dur=0.5):
    """Osminové opakování jednoho basového tónu na jeden takt."""
    return " ".join([f"{note}:{dur}"] * n)


LOOP = ["Am7", "F", "C", "G"]
ROOT = {"Am7": "A2", "F": "F2", "C": "C2", "G": "G2", "Em": "E2"}

# --------------------------------------------------------------------------
# nástroje
# --------------------------------------------------------------------------

pad = s.track("warmpad", vol=0.46, pan=0.0, name="pad")
pad.fx(HighPass(130), Reverb(room=0.86, mix=0.42, damp=0.42),
       Delay(1 / 4, fb=0.20, mix=0.12))

arp = s.track("pluck", vol=0.75, pan=-0.28, name="arp")
arp.fx(Delay(1 / 8, fb=0.32, mix=0.24), Reverb(room=0.55, mix=0.20, damp=0.5))

# Sub-bas drží pod 500 Hz, sidechain od bicích dělá pumpování na kicku.
# Tišší než by se čekalo — kick i bas soupeří o stejné pásmo pod 250 Hz,
# tak radši ustoupí a nechá mid/high nástrojům víc prostoru v mixu.
bass = s.track("subbass", vol=0.30, name="bass")
bass.fx(LowPass(500), Saturate(1.8, mix=0.35),
        Sidechain(source="drums", amount=0.55, attack=0.005, release=0.16))

lead = s.track("lead", vol=0.72, pan=0.18, name="lead")
lead.fx(HighPass(250), Reverb(room=0.75, mix=0.34, damp=0.40),
        Delay(3 / 16, fb=0.30, mix=0.24))

bell = s.track("bell", vol=0.85, pan=0.3, name="bell")
bell.fx(Reverb(room=0.90, mix=0.40, damp=0.30))

dr = s.drums(vol=0.62)


# --------------------------------------------------------------------------
# A  takty 0-7 — sparse intro, jen pad a pár zvonků
# --------------------------------------------------------------------------

pad.chord("Am7", bar=0, dur=16, octave=4, inv=1, vel=40)
pad.chord("F", bar=4, dur=16, octave=4, inv=1, vel=44)

arp.arp("F", bar=4, pattern="up", rate=1 / 8, octaves=1, bars=4,
        octave=4, vel=46)

bell.note("E5", at=b(2, 2), dur=2, vel=50)
bell.note("A5", at=b(6, 2), dur=2, vel=54)


# --------------------------------------------------------------------------
# B  takty 8-23 — build: bas, kartáče, postupně kick, náznak melodie
# --------------------------------------------------------------------------

for i in range(16):
    bar = 8 + i
    name = LOOP[i % 4]
    grow = i * 2                                    # 0 -> 30 přes sekci

    if i % 4 == 0:
        pad.chord(name, bar=bar, dur=4, octave=4, inv=1, vel=44 + grow // 2)
    arp.arp(name, bar=bar, pattern="updown", rate=1 / 16, octaves=2,
            octave=4, vel=48 + grow)
    bass.notes(f"{ROOT[name]}:4", bar=bar, vel=52 + grow)

dr.pattern("""
h.h.h.h.h.h.h.h.
k.......k.......
""", bars=(8, 16), vel=0.40)

dr.pattern("""
h.h.h.h.h.h.h.h.
k...k...k...k...
""", bars=(16, 20), vel=0.55)

dr.pattern("""
h.h.h.h.h.h.h.h.
k...k...k...k...
....c.......c...
""", bars=(20, 24), vel=0.65)

# náznak leadu na konci sekce — čtyři takty, potichu
lead.melody("A4:2 C5:2 | E5:2 A5:2 | G5:1 E5:1 C5:2 | A4:4", bar=20, vel=52)


# --------------------------------------------------------------------------
# C  takty 24-39 — první chorus, plný puls
# --------------------------------------------------------------------------

for i in range(16):
    bar = 24 + i
    name = LOOP[i % 4]

    if i % 4 == 0:
        pad.chord(name, bar=bar, dur=16, octave=4, inv=1, vel=64)
    arp.arp(name, bar=bar, pattern="up", rate=1 / 16, octaves=2,
            octave=4, vel=70)
    bass.notes(pulse(ROOT[name]), bar=bar, vel=92)

dr.pattern("K...K...K...K...", bars=(24, 40), vel=0.62)
dr.pattern("""
h.h.h.h.h.h.h.h.
....C.......C...
""", bars=(24, 40), vel=0.95)

CHORUS_LEAD = [
    (24, "A4:1 C5:1 E5:1 A5:1 | A5:1 G5:0.5 F5:0.5 E5:1 C5:1 | "
         "E5:1 D5:0.5 C5:0.5 E5:1 G5:1 | F5:1 D5:1 B4:2"),
    (28, "A4:1 C5:1 E5:1 A5:1 | A5:1 G5:0.5 F5:0.5 E5:1 C5:1 | "
         "E5:1 D5:0.5 C5:0.5 E5:1 G5:1 | A5:1 G5:1 E5:2"),
    (32, "C5:1 E5:1 A5:1 C6:1 | C6:1 A5:0.5 G5:0.5 E5:1 A5:1 | "
         "G5:1 F5:0.5 E5:0.5 G5:1 C6:1 | B5:1 A5:1 E5:2"),
    (36, "C6:1 B5:0.5 A5:0.5 G5:1 E5:1 | F5:1 E5:1 D5:1 C5:1 | "
         "B4:1 A4:1 E5:1 A5:1 | G5:1 E5:1 A4:2"),
]
for bar, seq in CHORUS_LEAD:
    lead.melody(seq, bar=bar, vel=82 + (bar - 24))

bell.note("A5", at=b(31, 3), dur=1, vel=56)
bell.note("C6", at=b(35, 3), dur=1, vel=60)


# --------------------------------------------------------------------------
# D  takty 40-47 — breakdown, nádech před druhým chorusem
# --------------------------------------------------------------------------

pad.chord("Am7", bar=40, dur=16, octave=4, inv=1, vel=42)
pad.chord("Em", bar=44, dur=12, octave=4, inv=1, vel=38)

arp.arp("Am7", bar=40, pattern="up", rate=1 / 8, octaves=1, bars=4,
        octave=4, vel=44)
arp.arp("Em", bar=44, pattern="up", rate=1 / 8, octaves=1, bars=3,
        octave=4, vel=40)

bass.notes("A2:4", bar=40, vel=48)
bass.notes("E2:4", bar=44, vel=44)

dr.pattern("h...h...h...h...", bars=(40, 46), vel=0.30)

# takt náběhu do druhého chorusu
dr.pattern("""
....t.t.....t.t.
""", bars=(46, 47), vel=0.55)
dr.pattern("""
K.t.t.T.t.T.tTtT
""", bars=(47, 48), vel=0.75)

bell.note("E5", at=b(45, 2), dur=2, vel=44)


# --------------------------------------------------------------------------
# E  takty 48-59 — druhý chorus, o kus výš a hlasitěji
# --------------------------------------------------------------------------

for i in range(12):
    bar = 48 + i
    name = LOOP[i % 4]

    if i % 4 == 0:
        pad.chord(name, bar=bar, dur=16, octave=4, inv=1, vel=74)
    arp.arp(name, bar=bar, pattern="up", rate=1 / 16, octaves=2,
            octave=4, vel=80)
    bass.notes(pulse(ROOT[name]), bar=bar, vel=104)

dr.pattern("K...K...K...K...", bars=(48, 60), vel=0.72)
dr.pattern("""
r.r.r.r.r.r.r.r.
....C.......C...
""", bars=(48, 60), vel=1.0)

FINAL_LEAD = [
    (48, "A4:1 C5:1 E5:1 A5:1 | A5:1 G5:0.5 F5:0.5 E5:1 C5:1 | "
         "E5:1 D5:0.5 C5:0.5 E5:1 G5:1 | F5:1 D5:1 B4:2"),
    (52, "C5:1 E5:1 A5:1 C6:1 | C6:1 A5:0.5 G5:0.5 E5:1 A5:1 | "
         "G5:1 F5:0.5 E5:0.5 G5:1 C6:1 | B5:1 A5:1 E5:2"),
    (56, "C6:1 B5:0.5 A5:0.5 G5:1 E5:1 | F5:1 E5:1 D5:1 C5:1 | "
         "B4:1 A4:1 E5:1 A5:1 | G5:1 E5:1 A4:2"),
]
for bar, seq in FINAL_LEAD:
    lead.melody(seq, bar=bar, vel=96)

bell.note("C6", at=b(51, 3), dur=1, vel=62)
bell.note("A5", at=b(55, 3), dur=1, vel=58)


# --------------------------------------------------------------------------
# F  takty 60-63 — outro, doznění na tónice
# --------------------------------------------------------------------------

pad.chord("Am7", bar=60, dur=16, octave=4, inv=1, vel=50)
arp.arp("Am7", bar=60, pattern="up", rate=1 / 8, octaves=2, bars=4,
        octave=4, vel=44)
bass.note("A2", at=b(60), dur=8, vel=50)
lead.note("A5", at=b(60), dur=8, vel=46)
bell.note("E5", at=b(61, 2), dur=2, vel=42)
bell.note("A5", at=b(63, 0), dur=4, vel=38)

dr.hit("kick", at=b(60, 0), vel=70)
dr.hit("openhat", at=b(60, 0), vel=50)
dr.hit("kick", at=b(62, 0), vel=44)


# --------------------------------------------------------------------------
# rozhýbání
# --------------------------------------------------------------------------

pad.humanize(timing=0.05, vel=0.08, seed=3)
arp.humanize(timing=0.012, vel=0.14, seed=7)
bass.humanize(timing=0.010, vel=0.08, seed=11)
lead.humanize(timing=0.016, vel=0.10, seed=13)
bell.humanize(timing=0.02, vel=0.10, seed=17)
dr.humanize(timing=0.008, vel=0.12, seed=19)

s.render("output/nocni_dalnice.wav")
