"""Skica inspirovaná Kyomei (Hideki Taniuchi) — anime-baladový klavír.

Tempo, tónina a akordová smyčka vychází z `analyze-ref` nad originálem::

    python -m paprika analyze-ref Kyomei.mp3
    -> TEMPO ~92 BPM, KEY D minor, smyčka kolem Dm - F - Am - C

Melodie, aranžmá a struktura jsou moje — čistá syntéza nezní jako živý
klavír/smyčce (viz README), tohle je nálada, ne kopie.

    python -m paprika show   songs/kyomei.py --bars 0:8
    python -m paprika render songs/kyomei.py
"""

from paprika import Delay, HighPass, Reverb, Song

s = Song(bpm=92, key="D", scale="minor", bars=20, title="kyomei")

piano = s.track("epiano", vol=0.62, name="piano")
piano.fx(HighPass(110), Reverb(room=0.8, mix=0.4), Delay(1 / 8, fb=0.2, mix=0.12))

strings = s.track("strings", vol=0.34, pan=0.15, name="strings")
strings.fx(HighPass(160), Reverb(room=0.85, mix=0.45))

pads = s.track("warmpad", vol=0.28, pan=-0.15, name="pads")
pads.fx(HighPass(200), Reverb(room=0.85, mix=0.4))

bass = s.track("subbass", vol=0.4, name="bass")
dr = s.drums(vol=0.5)

prog = ["Dm", "F", "Am", "C"]

# intro (bar 0-3) — jen arpeggio, pad se pomalu rozjíždí
for bar, ch in enumerate(prog):
    piano.arp(ch, bar=bar, pattern="up", rate=1 / 8, octaves=2, vel=68)
pads.chord("Dm", bar=2, dur=8, inv=1, octave=3, vel=48)

# verse 1 (bar 4-7) — melodie nad arpeggiem, bas vstupuje
piano.melody("A4:1 F4:1 D5:2 | C5:1 A4:1 F4:2 | D5:1 C5:1 A4:2 | G4:2 F4:2",
             bar=4, vel=96)
for bar, ch in enumerate(prog):
    pads.chord(ch, bar=4 + bar, dur=4, inv=1, octave=3, vel=56)
bass.notes("D2:4 | F2:4 | A1:4 | C2:4", bar=4, vel=88)
dr.pattern("....h.......h...", bars=(6, 8), vel=0.55)

# verse 2 (bar 8-11) — bicí trochu zesílí, smyčce se šetří na klimax
piano.melody("A4:1 F4:1 D5:2 | C5:1 A4:1 F4:2 | D5:1 E5:1 F5:2 | E5:2 D5:2",
             bar=8, vel=100)
for bar, ch in enumerate(prog):
    pads.chord(ch, bar=8 + bar, dur=4, inv=1, octave=3, vel=60)
bass.notes("D2:4 | F2:4 | A1:4 | C2:4", bar=8, vel=92)
dr.pattern("k.......s.......", bars=(8, 12), vel=0.65)

# klimax (bar 12-15) — plné akordy místo arpeggia, smyčce naplno
for bar, ch in enumerate(prog):
    piano.chord(ch, bar=12 + bar, dur=4, octave=4, vel=104)
    strings.chord(ch, bar=12 + bar, dur=4, inv=1, octave=4, vel=74)
    pads.chord(ch, bar=12 + bar, dur=4, inv=1, octave=3, vel=66)
bass.notes("D2:2 D2:2 | F2:2 F2:2 | A1:2 A1:2 | C2:2 C2:2", bar=12, vel=104)
dr.pattern("k...h...s...h..h", bars=(12, 16), vel=0.85)

# outro (bar 16-19) — zpátky na arpeggio, doznívání
for bar, ch in enumerate(prog):
    piano.arp(ch, bar=16 + bar, pattern="downup", rate=1 / 8, octaves=2, vel=58)
    pads.chord(ch, bar=16 + bar, dur=4, inv=1, octave=3, vel=44)

piano.humanize(timing=0.014, vel=0.1, seed=11)
strings.humanize(timing=0.02, vel=0.08, seed=13)
bass.humanize(timing=0.008, vel=0.06, seed=17)
dr.humanize(timing=0.006, vel=0.08, seed=19)

s.render("output/kyomei.wav")
