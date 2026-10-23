"""Kontrola teorie — harmonický plán v římských číslicích a `check` bez renderu.

AI (ani člověk se sluchátky v šuplíku) neslyší, co napsal. `check` za
vteřinu porovná noty s plánem a nahlásí špatný bas, nerozvedené nevlastní
tóny, paralely a kadence po taktech.

    python -m paprika check  examples/05_kontrola_teorie.py
    python -m paprika render examples/05_kontrola_teorie.py

Zkus odkomentovat řádek CHYBA dole a pusť `check` znovu.
"""

from paprika import Hall, Song, Width, mastering

s = Song(bpm=66, key="A", scale="minor", bars=8, title="kontrola_teorie")
s.harmony("i | iv | VII | III | VI | iv | V7 | i")
s.section("A", bars=(0, 4)).section("B", bars=(4, 8))

piano = s.track("grand_piano", vol=1.5, fx=[Width(0.5), Hall("hall", mix=0.22)])
celli = s.track("celli", vol=0.6, pan=0.2, fx="default")
violin = s.track("violin_solo", vol=0.65, pan=-0.15, fx="default")
s.master.fx(*mastering("cinematic"))

for bar, ch in enumerate(["Am", "Dm", "G", "C", "F", "Dm", "E7", "Am"]):
    piano.chord(ch, bar=bar, dur=4, octave=3, inv=1, vel=70)
piano.pedal(bars=(0, 8), every=1)

celli.notes("A2:4 | D3:4 | G2:4 | C3:4 | F2:4 | D3:4 | E3:2 E2:2 | A2:4", vel=78)
violin.melody("E5:2 C5:1 B4:1 | A4:2 D5:2 | B4:1 C5:1 D5:2 | E5:4 | "
              "F5:2 C5:1 A4:1 | D5:2 F5:2 | E5:1 D5:1 B4:1 G#4:1 | A4:4", vel=86)

# CHYBA: bas Bb v taktu, kde plán říká VII (G dur) — check nahlásí WARN
# celli.clear(bars=(2, 3)).notes("Bb2:4", bar=2, vel=78)

for t, seed in ((piano, 1), (celli, 2), (violin, 3)):
    t.humanize(timing=0.012, vel=0.08, seed=seed)
