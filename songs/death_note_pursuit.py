"""Napjatá, hnací nálada ve stylu Death Note (Kyrie) — rychlý smyčcový
ostinato, pulzující bas, gradace do plné intenzity s ride/hi-hat honičkou.

    python -m paprika show   songs/death_note_pursuit.py --bars 0:8
    python -m paprika render songs/death_note_pursuit.py
"""

from paprika import HighPass, Reverb, Saturate, Song

s = Song(bpm=150, key="E", scale="minor", bars=16, title="death_note_pursuit")

strings = s.track("strings", vol=0.4, name="strings")
strings.fx(HighPass(180), Reverb(room=0.35, mix=0.18))

lead = s.track("lead", vol=0.4, pan=0.2, name="lead")
lead.fx(HighPass(250), Saturate(1.6, mix=0.4), Reverb(room=0.3, mix=0.15))

bass = s.track("subbass", vol=0.34, name="bass")
dr = s.drums(vol=0.58)

prog = ["Em", "C", "D", "Em"]

# intro — pulzující bas, smyčcový ostinato se rozjíždí potichu
for bar, ch in enumerate(prog):
    strings.arp(ch, bar=bar, pattern="up", rate=1 / 16, octaves=1, vel=54, gate=0.7)
for bar in range(0, 4):
    bass.notes("E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5", bar=bar,
               vel=70, dur=0.5)
dr.pattern("..r...r...r...r", bars=(0, 4), vel=0.4)

# rozjezd — smyčce sílí, bicí přidávají kick pulz
for bar, ch in enumerate(prog):
    strings.arp(ch, bar=4 + bar, pattern="up", rate=1 / 16, octaves=1, vel=72, gate=0.7)
for bar in range(4, 8):
    bass.notes("E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5", bar=bar,
               vel=88, dur=0.5)
dr.pattern("k.r.k.r.k.r.k.r.", bars=(4, 8), vel=0.65)

# napětí — lead vstupuje s krátkými stabs, bas i bicí houstnou
lead.notes("E4:0.25 _:0.75 G4:0.25 _:0.75 E4:0.25 _:0.75 G4:0.25 _:0.75 | "
           "E4:0.25 _:0.75 D4:0.25 _:0.75 E4:0.25 _:0.75 D4:0.25 _:0.75 | "
           "E4:0.25 _:0.75 G4:0.25 _:0.75 E4:0.25 _:0.75 G4:0.25 _:0.75 | "
           "E4:0.25 _:0.75 B3:0.25 _:0.75 E4:0.25 _:0.75 B3:0.25 _:0.75",
           bar=8, vel=100)
for bar, ch in enumerate(prog):
    strings.arp(ch, bar=8 + bar, pattern="updown", rate=1 / 16, octaves=1, vel=86, gate=0.75)
for bar in range(8, 12):
    bass.notes("E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5", bar=bar,
               vel=100, dur=0.5)
dr.pattern("k.s.k.k.s.k.k.s.", bars=(8, 12), vel=0.8)

# klimax — plná intenzita, lead vede, ride honí tempo
lead.notes("E4:0.25 G4:0.25 B4:0.25 G4:0.25 " * 4 + "| " +
           "E4:0.25 D4:0.25 C4:0.25 D4:0.25 " * 4 + "| " +
           "E4:0.25 G4:0.25 B4:0.25 D5:0.25 " * 4 + "| " +
           "C5:0.25 B4:0.25 G4:0.25 E4:0.25 " * 4,
           bar=12, vel=112)
for bar, ch in enumerate(prog):
    strings.arp(ch, bar=12 + bar, pattern="updown", rate=1 / 16, octaves=2, vel=100, gate=0.8)
for bar in range(12, 16):
    bass.notes("E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5 E1:0.5", bar=bar,
               vel=110, dur=0.5)
dr.pattern("k.srk.srk.srk.sr", bars=(12, 16), vel=0.95)

strings.humanize(timing=0.004, vel=0.08, seed=41)
lead.humanize(timing=0.005, vel=0.1, seed=43)
bass.humanize(timing=0.003, vel=0.05, seed=45)
dr.humanize(timing=0.003, vel=0.1, seed=47)

s.render("output/death_note_pursuit.wav")
