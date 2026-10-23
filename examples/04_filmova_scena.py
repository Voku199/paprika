"""Filmová scéna — celý orchestr se sborem, d moll, 80 BPM, 16 taktů.

- 0-7   tichý začátek: harfa a křídlo, cella tremolo, víření tympánů
        roste do taktu 8,
- 8-15  téma v lesním rohu, houslová sekce, mužský sbor „aah“,
        údery tympánů na těžké doby, kontrabas.

    python -m paprika samples install all
    python -m paprika check  examples/04_filmova_scena.py
    python -m paprika render examples/04_filmova_scena.py
"""

from paprika import Hall, Song, Width, mastering

s = Song(bpm=80, key="D", scale="minor", bars=16, title="filmova_scena")
PROG = ["Dm", "Bb", "Gm", "A"]
s.harmony(" | ".join(["i", "VI", "iv", "V"] * 4))
s.section("napeti", bars=(0, 8)).section("tema", bars=(8, 16))

harp = s.track("concert_harp", vol=0.55, pan=0.3, fx="default")
piano = s.track("grand_piano", vol=1.6, pan=-0.2, fx=[Width(0.5), Hall("hall", mix=0.22)])
celli = s.track("celli.trem", vol=0.5, pan=0.25, fx="default")
violins = s.track("violins", vol=0.5, pan=-0.3, fx="default")
horn = s.track("horn", vol=0.65, pan=-0.1, fx="default")
choir = s.track("male_choir.aah", vol=0.45, fx="default")
bass = s.track("contrabass", vol=0.55, pan=0.3, fx="default")
roll = s.track("timpani.roll", vol=1.4, fx=[Width(0.5), Hall("hall", mix=0.3)])
hits = s.track("timpani", vol=0.7, fx="default")
s.master.fx(*mastering("cinematic"))

ROOT = {"Dm": "D", "Bb": "Bb", "Gm": "G", "A": "A"}
TIMP = {"Dm": "D3", "Bb": "Bb2", "Gm": "G2", "A": "A2"}  # pět kotlů: D#2-A3

# --- napětí (0-7) ---------------------------------------------------------
for bar, ch in enumerate(PROG * 2):
    harp.arp(ch, bar=bar, pattern="up", rate=1 / 4, octave=3, octaves=2, vel=62)
piano.melody("A4:3 F4:1 | D5:3 C5:1 | Bb4:2 A4:1 G4:1 | A4:4 | "
             "A4:3 F4:1 | F5:3 D5:1 | D5:2 Bb4:1 E5:1 | C#5:4", vel=84)
piano.pedal(bars=(0, 8), every=1)
celli.notes("D3:4 | D3:4 | D3:4 | C#3:4 | D3:4 | D3:4 | D3:4 | C#3:4", vel=70)
celli.automate("dyn", [(0, 0.4), (8, 1.1)], "s")
roll.note("A2", at=16, dur=16, vel=90)
roll.automate("dyn", [(4, 0.2), (8, 1.3)], "exp")

# --- téma (8-15) -----------------------------------------------------------
horn.melody("D4:2 A4:2 | F4:3 D4:1 | G4:2 Bb4:1 A4:1 | E4:4 | "
            "D4:2 A4:2 | D5:3 C5:1 | Bb4:1 A4:1 G4:1 F4:1 | E4:4", bar=8, vel=96)
for i, ch in enumerate(PROG * 2):
    bar = 8 + i
    violins.chord(ch, bar=bar, dur=4, octave=4, inv=1, vel=82)
    choir.chord(ch, bar=bar, dur=4, octave=3, vel=74)
    bass.notes(f"{ROOT[ch]}1:4" if ch != "Dm" else "D2:4", bar=bar, vel=84)
    celli.notes(f"{ROOT[ch]}2:4" if ch in ("Bb", "A") else f"{ROOT[ch]}3:4", bar=bar, vel=80)
    hits.notes(f"{TIMP[ch]}:2 _:2", bar=bar, vel=96)
violins.automate("dyn", [(8, 0.7), (14, 1.2), (16, 0.8)], "s")

for t, seed in ((harp, 1), (piano, 2), (horn, 3), (violins, 4), (choir, 5), (bass, 6), (hits, 7)):
    t.humanize(timing=0.012, vel=0.07, seed=seed)
