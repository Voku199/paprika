"""Dech — ukázka lidských zvuků: nádechy sólisty podle frází, tep, kroky,
šustění, blízký mikrofon a hudba, která dechu uhne v dynamické EQ.

    python -m paprika check  songs/dech.py
    python -m paprika show   songs/dech.py --bars 0:8 --res 16
    python -m paprika render songs/dech.py
"""

from paprika import DynEQ, Hall, Song, mastering

s = Song(bpm=76, key="D", scale="minor", bars=16, title="dech")
s.harmony("i | i | VI | VII | i | iv | V | V | i | VI | iv | V | i | VI | VII | i")
s.section("příchod", (0, 4)).section("téma", (4, 12)).section("závěr", (12, 16))

pad = s.track("strings_ens", vol=0.5, name="pad", fx="default")
# pad uhne ve 2–5 kHz jen tehdy, když zní nádech (klíčem je stopa "dech")
pad.fx(DynEQ(freq=3500.0, q=0.8, threshold=-45.0, range_db=4.0, source="dech"))
bass = s.track("round_bass", vol=0.38, name="bass")
cello = s.track("cello_solo", vol=0.62, pan=-0.1, name="cello",
                fx=[Hall("hall", mix=0.3)])

chords = ["Dm", "Dm", "Bb", "C", "Dm", "Gm", "A", "A",
          "Dm", "Bb", "Gm", "A", "Dm", "Bb", "C", "Dm"]
for bar, ch in enumerate(chords):
    pad.chord(ch, bar=bar, dur=4, octave=3, vel=64 if bar < 4 else 72)
bass.notes("D2:4 | G2:4 | A2:4 | A2:4 | D2:4 | Bb2:4 | G2:4 | A2:4 | "
           "D2:4 | Bb2:4 | C3:4 | D2:4", bar=4, vel=88)
bass.humanize(timing=0.01, vel=0.08, seed=5, corr=0.85)

# téma: pauzy mezi frázemi jsou místa na nádech
cello.notes("D4:2 F4:1 E4:1 | D4:3 _:1 | C#4:2 E4:1 A4:1 | A4:3 _:1", bar=4, vel=92)
cello.notes("F4:2 A4:1 D4:1 | D4:2 F4:2 | G4:3 _:1 | E4:2 A4:1 C#5:1", bar=8, vel=96)
cello.notes("D5:4 | _:1 D4:1 F4:1 Bb4:1 | G4:2 E4:2 | D4:3 _:1", bar=12, vel=88)
cello.humanize(timing=0.02, vel=0.1, seed=3, corr=0.85)

# dech violoncellisty: nádechy z frází, mlasknutí před prvním nástupem,
# vzdech po poslední notě
dech = s.human("dech", performer=cello, vol=0.2, pan=-0.1)
dech.breaths()
dech.lip_smack(3, beat=3.55, vel=60)
dech.exhale(15, beat=3.1, kind="sigh", depth=0.7, vel=80)
dech.humanize(seed=4)

# tělo: kroky ke stolu, tep, usednutí (šustění); s basem tep končí, jinak
# by se s ním pral pod 250 Hz
telo = s.human("telo", vol=0.45)
telo.steps((0, 3), every=1, surface="wood", vel=72)
telo.cloth(3, beat=0.0, beats=1.5, fabric="cotton", motion="fidget", vel=80)
telo.heartbeat((0, 4), every=1, vel=78)
telo.humanize(seed=8)

s.master.fx(*mastering("cinematic"))

s.render("output/dech.wav")
