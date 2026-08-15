"""Romantická skladba na elektrické piano — referenční příklad pro Papriku.

    python -m paprika show   songs/romantic.py --bars 0:4
    python -m paprika render songs/romantic.py
"""

from paprika import Delay, HighPass, Reverb, Song

s = Song(bpm=72, key="C", scale="major", bars=32, title="romantic")

piano = s.track("epiano", vol=0.62, name="piano")
piano.fx(HighPass(120), Reverb(room=0.75, mix=0.35),
         Delay(1 / 8, fb=0.25, mix=0.15))

pads = s.track("warmpad", vol=0.32, pan=-0.2, name="pads")
# highpass uvolní basům místo pod 140 Hz
pads.fx(HighPass(140), Reverb(room=0.85, mix=0.4))

bass = s.track("subbass", vol=0.42, name="bass")
dr = s.drums(vol=0.62)

prog = ["Cmaj7", "Am7", "Fmaj7", "G7"]

# intro — arpeggia přes dvě kola akordů
for bar, ch in enumerate(prog * 2):
    piano.arp(ch, bar=bar, pattern="up", rate=1 / 8, octaves=2, vel=78)
    pads.chord(ch, bar=bar, dur=4, inv=1, octave=3, vel=70)

# verse — melodie, pod ní basová linka a bicí
piano.melody("E5:1 G5:1 A5:2 | G5:1 E5:1 D5:2 | C5:4 | _:2 G4:1 A4:1",
             bar=8, vel=104)
piano.melody("E5:1 G5:1 C6:2 | B5:1 G5:1 E5:2 | D5:3 ~:1 | C5:4",
             bar=12, vel=104)
for bar, ch in enumerate(prog * 2):
    pads.chord(ch, bar=8 + bar, dur=4, inv=1, octave=3, vel=64)

bass.notes("C2:4 | A1:4 | F1:4 | G1:4 | C2:4 | A1:4 | F1:4 | G1:4",
           bar=8, vel=96)

dr.pattern("k...h...s...h..h", bars=(12, 24))
dr.pattern("....h.......h...", bars=(8, 12), vel=0.7)

# refrén — akordy pod melodií, plus zdvojený bas
for bar, ch in enumerate(prog * 2):
    piano.chord(ch, bar=16 + bar, dur=4, octave=4, vel=88)
    pads.chord(ch, bar=16 + bar, dur=4, inv=2, octave=3, vel=72)
bass.notes("C2:2 C2:2 | A1:2 A1:2 | F1:2 F1:2 | G1:2 G1:2 | "
           "C2:2 C2:2 | A1:2 A1:2 | F1:2 F1:2 | G1:4",
           bar=16, vel=100)

# outro — jen arpeggia a pad, doznívání
for bar, ch in enumerate(prog * 2):
    piano.arp(ch, bar=24 + bar, pattern="updown", rate=1 / 8, octaves=2, vel=64)
    pads.chord(ch, bar=24 + bar, dur=4, inv=1, octave=3, vel=58)

piano.humanize(timing=0.015, vel=0.12, seed=7)
bass.humanize(timing=0.008, vel=0.08, seed=3)
dr.humanize(timing=0.006, vel=0.1, seed=5)

s.render("output/romantic.wav")
