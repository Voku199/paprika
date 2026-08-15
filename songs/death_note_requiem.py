"""Temná, chorální nálada ve stylu Death Note (Low of Solipsism / Kyrie) —
frygická tónina pro gregoriánský nádech, varhany jako náhrada sboru, pomalá
gradace do plných akordů s tomy.

    python -m paprika show   songs/death_note_requiem.py --bars 0:8
    python -m paprika render songs/death_note_requiem.py
"""

from paprika import Delay, HighPass, Reverb, Song

s = Song(bpm=72, key="D", scale="phrygian", bars=16, title="death_note_requiem")

organ = s.track("organ", vol=0.4, name="organ")
organ.fx(HighPass(180), Reverb(room=0.9, mix=0.5))

strings = s.track("strings", vol=0.38, pan=-0.1, name="strings")
strings.fx(HighPass(150), Reverb(room=0.9, mix=0.5))

bell = s.track("bell", vol=0.34, pan=0.25, name="bell")
bell.fx(Reverb(room=0.92, mix=0.55))

bass = s.track("subbass", vol=0.38, name="bass")
dr = s.drums(vol=0.5)

prog = ["Dm", "Eb", "Dm", "Gm"]

# intro — jen varhanní tón + zvonky, ticho okolo
organ.chord("Dm", bar=0, dur=8, inv=1, octave=3, vel=48)
bell.notes("D5:4 | _:4 | A4:4 | _:4", bar=0, vel=58)
bass.notes("D1:4 | D1:4 | D1:4 | D1:4", bar=0, vel=70)

# verse — smyčce se přidají, akordová smyčka se rozjíždí
for bar, ch in enumerate(prog):
    organ.chord(ch, bar=4 + bar, dur=4, inv=1, octave=3, vel=58)
    strings.chord(ch, bar=4 + bar, dur=4, inv=2, octave=3, vel=52)
bass.notes("D1:4 | Eb1:4 | D1:4 | G1:4", bar=4, vel=82)
dr.pattern("t.......t.......", bars=(4, 8), vel=0.55)

# gradace — plnější akordy, bicí zesilují
for bar, ch in enumerate(prog):
    organ.chord(ch, bar=8 + bar, dur=4, inv=0, octave=4, vel=72)
    strings.chord(ch, bar=8 + bar, dur=4, inv=1, octave=3, vel=68)
bell.notes("D5:2 A4:2 | Eb5:2 Bb4:2 | D5:2 F5:2 | C5:2 A4:2", bar=8, vel=64)
bass.notes("D1:2 D1:2 | Eb1:2 Eb1:2 | D1:2 D1:2 | G1:2 G1:2", bar=8, vel=94)
dr.pattern("t...t...s...t...", bars=(8, 12), vel=0.75)

# klimax — plný akord, nejtěžší tomy
for bar, ch in enumerate(prog):
    organ.chord(ch, bar=12 + bar, dur=4, inv=0, octave=4, vel=88)
    strings.chord(ch, bar=12 + bar, dur=4, inv=2, octave=4, vel=78)
bell.notes("D5:1 A4:1 D6:2 | Eb5:1 Bb4:1 Eb6:2 | D5:1 F5:1 D6:2 | C5:1 A4:1 C6:2",
           bar=12, vel=76)
bass.notes("D1:2 D1:2 | Eb1:2 Eb1:2 | D1:2 D1:2 | G1:2 G1:2", bar=12, vel=104)
dr.pattern("t.t.s.t.t.t.s.t.", bars=(12, 16), vel=0.9)

organ.humanize(timing=0.02, vel=0.06, seed=31)
strings.humanize(timing=0.02, vel=0.08, seed=33)
bell.humanize(timing=0.015, vel=0.08, seed=35)
bass.humanize(timing=0.01, vel=0.05, seed=37)
dr.humanize(timing=0.01, vel=0.1, seed=39)

s.render("output/death_note_requiem.wav")
