"""Smyčcový orchestr — sekce drží harmonii, sólové housle zpívají, D dur, 60 BPM.

Ukazuje plynulou dynamiku `automate("dyn", ...)` (jako modwheel ve filmových
knihovnách — jeden dlouhý tón udělá crescendo a vrstvy nahrávky se prolnou),
legato sólových houslí a pizzicato kontrabasu.

    python -m paprika samples install strings
    python -m paprika render examples/02_smycce.py
"""

from paprika import Song, mastering

s = Song(bpm=60, key="D", scale="major", bars=8, title="smycce")
s.harmony("D | Bm | G | A | D | F#m | G A | D")

violins = s.track("violins", vol=0.5, pan=-0.25, fx="default")
violas = s.track("violas", vol=0.5, pan=0.1, fx="default")
celli = s.track("celli", vol=0.6, pan=0.25, fx="default")
bass = s.track("contrabass.pizz", vol=0.6, pan=0.3, fx="default")
solo = s.track("violin_solo", vol=0.7, pan=-0.05, fx="default")
s.master.fx(*mastering("cinematic"))

violins.notes("F#4:4 | F#4:4 | G4:4 | E4:4 | F#4:4 | F#4:4 | G4:2 E4:2 | F#4:4", vel=70)
violas.notes("A3:4 | D4:4 | B3:4 | A3:4 | A3:4 | A3:4 | B3:2 C#4:2 | D4:4", vel=70)
celli.notes("D3:4 | B2:4 | G2:4 | A2:4 | D3:4 | A2:4 | G2:2 A2:2 | D3:4", vel=72)
bass.notes("D2:1 _:1 D2:1 _:1 | B1:1 _:1 B1:1 _:1 | G1:1 _:1 G1:1 _:1 | "
           "A1:1 _:1 A1:1 _:1 | D2:1 _:1 D2:1 _:1 | F#1:1 _:1 F#1:1 _:1 | "
           "G1:1 _:1 A1:1 _:1 | D2:2 _:2", vel=80)

# sekce: nádech od pianissima, vrchol v taktu 6, doznění
for t in (violins, violas, celli):
    t.automate("dyn", [(0, 0.45), (4, 0.8), (6, 1.25), (8, 0.6)], "s")

solo.melody("_:2 A4:1 D5:1 | F#5:3 E5:1 | D5:2 B4:1 D5:1 | C#5:3 A4:1 | "
            "D5:1 E5:1 F#5:1 A5:1 | A5:2 G5:1 F#5:1 | E5:1 D5:1 E5:1 C#5:1 | D5:4",
            vel=88)

for t, seed in ((violins, 1), (violas, 2), (celli, 3), (bass, 4), (solo, 5)):
    t.humanize(timing=0.012, vel=0.06, seed=seed)
