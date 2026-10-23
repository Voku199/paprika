"""Sbor a soprán — ženský a mužský sbor, sólový soprán a křídlo, e moll, 60 BPM.

Hlasy jsou nahrávky zpěváků z VocalSet (CC BY 4.0), artikulace je
samohláska: `.aah`, `.oh`, `.ooh`, `.eh`. Sbor = šest hlasů na klávesu
rozestavěných do stran, sólo je mono a navazující tóny zpívá legato.

    python -m paprika samples install voices piano
    python -m paprika render examples/03_sbor.py
"""

from paprika import Hall, Song, Width, mastering

s = Song(bpm=60, key="E", scale="minor", bars=8, title="sbor")
s.harmony("Em | C | G | D | Em | C | Am B | Em")

women = s.track("female_choir.ooh", vol=0.45, pan=0.15, fx="default")
men = s.track("male_choir.ooh", vol=0.5, pan=-0.15, fx="default")
soprano = s.track("soprano.aah", vol=0.6, fx="default")
piano = s.track("grand_piano", vol=1.3, fx=[Width(0.5), Hall("hall", mix=0.22)])
s.master.fx(*mastering("cinematic"))

women.notes("(G4 B4):4 | (G4 C5):4 | (G4 B4):4 | (F#4 A4):4 | "
            "(G4 B4):4 | (G4 C5):4 | (A4 C5):2 (F#4 B4):2 | (G4 B4):4", vel=60)
men.notes("(E3 B3):4 | (C3 G3):4 | (D3 B3):4 | (D3 A3):4 | "
          "(E3 B3):4 | (C3 G3):4 | (A2 E3):2 (B2 D#3):2 | (E3 B3):4", vel=62)
women.automate("dyn", [(0, 0.6), (5, 1.15), (8, 0.7)], "s")
men.automate("dyn", [(0, 0.6), (5, 1.15), (8, 0.7)], "s")

# soprán nastoupí v taktu 2 a nese vrchol
soprano.melody("_:4 | _:4 | D5:2 E5:1 F#5:1 | A5:3 F#5:1 | "
               "G5:2 F#5:1 E5:1 | E5:3 G5:1 | F#5:1 E5:1 D#5:2 | E5:4", vel=84)

# piano jen rozložené akordy na doby
piano.notes("E2:1 B2:1 G3:1 B2:1 | C2:1 G2:1 E3:1 G2:1 | G1:1 D2:1 B2:1 D2:1 | "
            "D2:1 A2:1 F#3:1 A2:1 | E2:1 B2:1 G3:1 B2:1 | C2:1 G2:1 E3:1 G2:1 | "
            "A1:1 E2:1 B1:1 F#2:1 | (E2 B2 E3):4", vel=68)
piano.pedal(bars=(0, 8), every=1)

for t, seed in ((women, 1), (men, 2), (soprano, 3), (piano, 4)):
    t.humanize(timing=0.012, vel=0.06, seed=seed)
