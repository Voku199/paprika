"""Výstřední/hravá nálada ve stylu Death Note (L's Theme) — cirkusové staccato,
harmonická moll pro exotické napětí, nesourodý pluck + epiano dialog.

    python -m paprika show   songs/death_note_eccentric.py --bars 0:8
    python -m paprika render songs/death_note_eccentric.py
"""

from paprika import Delay, HighPass, Reverb, Song

s = Song(bpm=144, key="A", scale="harmonic_minor", bars=16, title="death_note_eccentric")

pluck = s.track("pluck", vol=0.95, name="pluck")
pluck.fx(HighPass(200), Reverb(room=0.4, mix=0.18))

piano = s.track("epiano", vol=0.5, pan=0.2, name="piano")
piano.fx(HighPass(150), Delay(1 / 16, fb=0.15, mix=0.1))

bass = s.track("subbass", vol=0.44, name="bass")
dr = s.drums(vol=0.55)

prog = ["Am", "E", "Am", "Dm"]

# intro — staccato pluck ostinato, bas našlapuje pod tím
for bar, ch in enumerate(prog):
    pluck.arp(ch, bar=bar, pattern="up", rate=1 / 16, octaves=1, vel=72, gate=0.5)
bass.notes("A1:1 _:0.5 A1:0.5 _:1 A1:1 | E1:1 _:0.5 E1:0.5 _:1 E1:1 | "
           "A1:1 _:0.5 A1:0.5 _:1 A1:1 | D2:1 _:0.5 D2:0.5 _:1 D2:1",
           bar=0, vel=76)
dr.pattern("....c.......c...", bars=(2, 4), vel=0.5)

# téma — epiano melodie odpovídá plucku, staccato osminy
piano.melody("A4:0.5 C5:0.5 B4:0.5 A4:0.5 G#4:0.5 A4:0.5 E4:0.5 A4:0.5 | "
             "G#4:0.5 A4:0.5 E4:0.5 A4:0.5 G#4:0.5 F4:0.5 E4:0.5 A4:0.5 | "
             "A4:0.5 C5:0.5 D5:0.5 C5:0.5 B4:0.5 A4:0.5 G#4:0.5 A4:0.5 | "
             "C5:0.5 B4:0.5 A4:0.5 G#4:0.5 A4:0.5 B4:0.5 C5:0.5 A4:0.5",
             bar=4, vel=92)
for bar, ch in enumerate(prog):
    pluck.arp(ch, bar=4 + bar, pattern="down", rate=1 / 16, octaves=1, vel=68, gate=0.5)
bass.notes("A1:1 _:0.5 A1:0.5 _:1 A1:1 | E1:1 _:0.5 E1:0.5 _:1 E1:1 | "
           "A1:1 _:0.5 A1:0.5 _:1 A1:1 | D2:1 _:0.5 D2:0.5 _:1 D2:1",
           bar=4, vel=82)
dr.pattern("k.h.c.h.k.h.c.h.", bars=(4, 8), vel=0.6)

# variace — melodie se otočí, staccato osminy
piano.melody("E5:0.5 D5:0.5 C5:0.5 B4:0.5 A4:0.5 G#4:0.5 A4:0.5 E5:0.5 | "
             "B4:0.5 A4:0.5 G#4:0.5 A4:0.5 B4:0.5 C5:0.5 D5:0.5 C5:0.5 | "
             "A4:0.5 B4:0.5 C5:0.5 D5:0.5 E5:0.5 D5:0.5 C5:0.5 B4:0.5 | "
             "B4:0.5 A4:0.5 A4:0.5 G#4:0.5 A4:0.5 B4:0.5 C5:0.5 A4:0.5",
             bar=8, vel=96)
for bar, ch in enumerate(prog):
    pluck.arp(ch, bar=8 + bar, pattern="updown", rate=1 / 16, octaves=1, vel=74, gate=0.45)
bass.notes("A1:1 _:0.5 A1:0.5 _:1 A1:1 | E1:1 _:0.5 E1:0.5 _:1 E1:1 | "
           "A1:1 _:0.5 A1:0.5 _:1 A1:1 | D2:1 _:0.5 D2:0.5 _:1 D2:1",
           bar=8, vel=86)
dr.pattern("k.h.c.h.k.h.c.hh", bars=(8, 12), vel=0.7)

# outro — zpátky do intra, dozvuk
for bar, ch in enumerate(prog):
    pluck.arp(ch, bar=12 + bar, pattern="up", rate=1 / 16, octaves=1, vel=58, gate=0.5)
bass.notes("A1:1 _:0.5 A1:0.5 _:1 A1:1 | E1:1 _:0.5 E1:0.5 _:1 E1:1 | "
           "A1:1 _:0.5 A1:0.5 _:1 A1:1 | D2:1 _:0.5 D2:0.5 _:1 D2:1",
           bar=12, vel=68)
dr.pattern("....c.......c...", bars=(12, 16), vel=0.4)

pluck.humanize(timing=0.006, vel=0.1, seed=21)
piano.humanize(timing=0.01, vel=0.1, seed=23)
bass.humanize(timing=0.005, vel=0.06, seed=25)
dr.humanize(timing=0.004, vel=0.08, seed=27)

s.render("output/death_note_eccentric.wav")
