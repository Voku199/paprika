"""Hlas — ukázka znělých gest (fáze H3): hlasivky (LF puls s jitterem)
a trakt jako trubice. Bručení po tónech akordů, vzdech, smích, námaha,
pod tím sbor z hlasivek a traktu (preset choir_tract).

    python -m paprika check  songs/hlas.py
    python -m paprika show   songs/hlas.py --res 16
    python -m paprika render songs/hlas.py
"""

from paprika import Song, mastering

s = Song(bpm=76, key="D", scale="minor", bars=12, title="hlas")
s.harmony("i | VI | iv | V | i | VI | VII | V | i | iv | V | i")
s.section("bručení", (0, 4)).section("gesta", (4, 9)).section("závěr", (9, 12))

sbor = s.track("choir_tract", vol=0.45, name="sbor", fx="default")
bas = s.track("round_bass", vol=0.32, name="bas")
chords = ["Dm", "Bb", "Gm", "A", "Dm", "Bb", "C", "A", "Dm", "Gm", "A", "Dm"]
for bar, ch in enumerate(chords):
    sbor.chord(ch, bar=bar, dur=4, octave=4, vel=60 if bar < 4 else 68)
# bas nastoupí až po bručení: obojí sedí pod 250 Hz
bas.notes("D2:4 | Bb2:4 | C3:4 | A2:4 | D2:4 | G2:4 | A2:4 | D2:4", bar=4, vel=84)
bas.humanize(timing=0.01, vel=0.08, seed=2, corr=0.85)
sbor.humanize(timing=0.015, vel=0.08, seed=5, corr=0.85)

# zpěvačka zblízka: bručí na tónech akordů, pak vzdech, smích, námaha
hlas = s.human("hlas", voice="female", vol=1.0, pan=0.1)
for bar in range(4):
    hlas.hum(bar, beat=0.5, beats=3.0, vel=80)
hlas.sigh(4, beat=1.0, vel=90)
hlas.laugh(6, beat=0.5, calls=5, vel=85)
hlas.effort(8, beat=0.0, vel=95).effort(8, beat=2.0, vel=100)
hlas.hum(9, beat=0.5, beats=3.0, vel=72).hum(11, beats=3.5, pitch="D4", vel=70)
hlas.humanize(seed=6)

s.master.fx(*mastering("cinematic"))

s.render("output/hlas.wav")
