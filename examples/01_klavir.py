"""Koncertní křídlo — nejmenší skladba, 8 taktů, a moll, 72 BPM.

Levá ruka hraje rozložené akordy, pravá melodii, pedál se přešlapuje se
změnou harmonie. Nahrávka Salamander Grand Piano (16 dynamických vrstev):
velocity mění barvu, ne jen hlasitost.

    python -m paprika samples install piano     # jednou, ~1 GB
    python -m paprika show   examples/01_klavir.py
    python -m paprika render examples/01_klavir.py --play
"""

from paprika import Hall, Song, Width, mastering

s = Song(bpm=72, key="A", scale="minor", bars=8, title="klavir")
s.harmony("Am | F | C | G | Am | F | Dm E | Am")

# Salamander je nahraný na široce rozestavěné mikrofony; Width ho zúží,
# aby v mono (telefon, reproduktor) nezmizely výšky
PIANO_FX = [Width(0.5), Hall("hall", mix=0.22)]

left = s.track("grand_piano", vol=1.4, fx=PIANO_FX, name="leva")
right = s.track("grand_piano", vol=1.5, fx=PIANO_FX, name="prava")
s.master.fx(*mastering("cinematic"))

left.notes("A2:0.5 E3:0.5 A3:0.5 C4:0.5 E4:0.5 C4:0.5 A3:0.5 E3:0.5 | "
           "F2:0.5 C3:0.5 F3:0.5 A3:0.5 C4:0.5 A3:0.5 F3:0.5 C3:0.5 | "
           "C3:0.5 G3:0.5 C4:0.5 E4:0.5 G4:0.5 E4:0.5 C4:0.5 G3:0.5 | "
           "G2:0.5 D3:0.5 G3:0.5 B3:0.5 D4:0.5 B3:0.5 G3:0.5 D3:0.5 | "
           "A2:0.5 E3:0.5 A3:0.5 C4:0.5 E4:0.5 C4:0.5 A3:0.5 E3:0.5 | "
           "F2:0.5 C3:0.5 F3:0.5 A3:0.5 C4:0.5 A3:0.5 F3:0.5 C3:0.5 | "
           "D2:0.5 A2:0.5 D3:0.5 F3:0.5 E2:0.5 B2:0.5 E3:0.5 G#3:0.5 | "
           "(A2 E3 A3 C4):4", vel=70)
left.pedal(bars=(0, 8), every=1)

right.melody("E5:1.5 D5:0.5 C5:1 B4:1 | A4:2 C5:1 F5:1 | "
             "E5:1.5 D5:0.5 C5:1 E5:1 | D5:3 B4:1 | "
             "C5:1.5 B4:0.5 A4:1 E5:1 | F5:2 A5:1 F5:1 | "
             "F5:1 D5:1 E5:1 G#4:1 | A4:4", vel=92)
right.pedal(bars=(0, 8), every=1)

left.humanize(timing=0.012, vel=0.08, seed=1)
right.humanize(timing=0.015, vel=0.1, seed=2)
