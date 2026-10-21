"""Divadlo — klidný kus pro koncertní křídlo a smyčce, d moll, 70 BPM.

Ukázka sampleru: všechno jsou nahrávky skutečných nástrojů (Salamander
Grand Piano, VSCO 2 CE), žádná syntéza. Čtyři části po osmi taktech:

- 0-7   sólo křídlo: levá ruka rozložené akordy, pravá téma A, pedál
        přešlapovaný se změnou harmonie,
- 8-15  téma B v pianu, nastoupí cella a violy (vol automatizace),
- 16-23 téma A přebírají sólové housle (legato), harfa, lesní roh,
        houslová sekce drží harmonii,
- 24-31 vrchol: houslová sekce hraje téma B legato, piano v oktávách,
        kontrabas; smyčce na konci utichnou a zbude piano.

Vyžaduje vzorky: python -m paprika samples install all

    python -m paprika check songs/divadlo.py    # harmonie, kadence, teorie
    python -m paprika render songs/divadlo.py
    python -m paprika compare output/divadlo.wav "songs_mp3/City Ruins - Rays of Light (NieR_Automata Original Soundtrack)Audio.mp3"
"""

from paprika import Hall, Song, Width, mastering, pitch

s = Song(bpm=70, key="D", scale="minor", bars=32, title="divadlo")

# akord na takt; takt 6 má dva akordy po půl taktu
PROG = ["Dm", "Bb", "Gm", "A", "Dm", "F", ("Gm", "A"), "Dm"]

# harmonický plán (i | VI | iv | V | i | III | iv V | i) a čtyři části:
# podle něj `check` hlídá bas a kadence a C# v dominantě A bere jako tón
# akordu, ne jako notu mimo d moll
PLAN = " | ".join(f"{ch[0]}:2 {ch[1]}:2" if isinstance(ch, tuple) else ch
                  for ch in PROG)
for part, name in enumerate(("A", "B", "A2", "vrchol")):
    s.harmony(PLAN, bar=part * 8).section(name, bars=(part * 8, part * 8 + 8))
ROOT = {"Dm": "D2", "Bb": "Bb1", "Gm": "G1", "A": "A1", "F": "F1"}
MINOR = {"Dm", "Gm"}

THEME_A = [
    "A4:2 D5:1 E5:1",
    "F5:2 E5:1 D5:1",
    "D5:2 Bb4:1 G4:1",
    "E5:1 C#5:1 A4:2",
    "A4:2 D5:1 F5:1",
    "A5:3 G5:1",
    "F5:1 E5:1 D5:1 C#5:1",
    "D5:4",
]
THEME_B = [
    "F5:2 E5:1 D5:1",
    "D5:2 C5:1 Bb4:1",
    "Bb4:2 A4:1 G4:1",
    "A4:3 _:1",
    "D5:1 E5:1 F5:1 G5:1",
    "A5:2 C6:1 A5:1",
    "G5:1 F5:1 E5:1 C#5:1",
    "D5:4",
]

# jednohlasé linky smyčců a rohu: výška na akord
CELLO = {"Dm": "D3", "Bb": "Bb2", "Gm": "G2", "A": "A2", "F": "F2"}
VIOLA = {"Dm": "A3", "Bb": "Bb3", "Gm": "Bb3", "A": "A3", "F": "A3"}
HORN = {"Dm": "D4", "Bb": "D4", "Gm": "D4", "A": "C#4", "F": "C4"}
VIOLINS = {"Dm": ("D4", "F4"), "Bb": ("D4", "F4"), "Gm": ("D4", "G4"),
           "A": ("C#4", "E4"), "F": ("C4", "F4")}

# ---------------------------------------------------------------------------
# stopy (orchestrální rozestavení: housle vlevo, cella a basy vpravo)
# ---------------------------------------------------------------------------
piano = s.track("grand_piano", vol=0.9, pan=-0.1, fx="default")
celli = s.track("celli", vol=0.55, pan=0.35, fx="default")
violas = s.track("violas", vol=0.4, pan=0.15, fx="default")
# sekce je nahraná rozestavenými mikrofony — Width ji zúží, ať drží v mono
violins = s.track("violins", vol=0.4, pan=-0.35,
                  fx=[Width(0.6), Hall("hall", mix=0.3)])
melody = s.track("violins", vol=0.55, pan=-0.3, mono=True, name="violins_melody",
                 fx=[Width(0.6), Hall("hall", mix=0.3)])
solo = s.track("violin_solo", vol=0.6, pan=-0.05, fx="default")
harp = s.track("concert_harp", vol=0.45, pan=-0.5, fx="default")
horn = s.track("horn", vol=0.35, pan=-0.15, fx="default")
bass = s.track("contrabass", vol=0.45, pan=0.45, fx="default")
s.master.fx(*mastering("cinematic"))
s.master.vol = 1.6            # +4 dB: klidný kus má jinak špičku jen -7 dBFS


def segments():
    """(takt, začátek v taktu, délka v beatech, akord) pro všech 32 taktů."""
    for bar in range(32):
        ch = PROG[bar % 8]
        if isinstance(ch, tuple):
            yield bar, 0.0, 2.0, ch[0]
            yield bar, 2.0, 2.0, ch[1]
        else:
            yield bar, 0.0, 4.0, ch


def arpeggio(ch, beats):
    """Levá ruka: rozložený akord nahoru a zpět po osminách."""
    r = pitch(ROOT[ch])
    third = 3 if ch in MINOR else 4
    up = [r, r + 7, r + 12, r + 12 + third, r + 24, r + 12 + third, r + 12, r + 7]
    return up[: int(beats * 2)]


# dynamika po částech: (levá ruka, melodie, smyčce)
DYN = [(50, 66, 0), (56, 74, 70), (48, 60, 76), (66, 100, 96)]

# ---------------------------------------------------------------------------
# piano
# ---------------------------------------------------------------------------
for bar, off, beats, ch in segments():
    lh, _, _ = DYN[bar // 8]
    at = bar * 4 + off
    for i, p in enumerate(arpeggio(ch, beats)):
        accent = 10 if i == 0 else (4 if i == 4 else 0)
        piano.note(p, at=at + i * 0.5, dur=0.5, vel=lh + accent)
    piano.pedal(bars=(bar + off / 4, bar + (off + beats) / 4))

for i, line in enumerate(THEME_A):
    piano.melody(line, bar=i, vel=DYN[0][1])          # sólo úvod
for i, line in enumerate(THEME_B):
    piano.melody(line, bar=8 + i, vel=DYN[1][1])
for i, line in enumerate(THEME_B):                    # vrchol: oktávy
    piano.melody(line, bar=24 + i, vel=DYN[3][1])
    low = piano.melody(line, bar=24 + i, vel=DYN[3][1] - 18).clips[-1]
    for n in low.notes:                               # dolní hlas o oktávu níž
        n.pitch -= 12
piano.humanize(timing=0.015, vel=0.08, seed=3)

# ---------------------------------------------------------------------------
# smyčce, harfa, roh
# ---------------------------------------------------------------------------
for bar, off, beats, ch in segments():
    if bar < 8:
        continue
    at = bar * 4 + off
    _, _, sv = DYN[bar // 8]
    celli.note(CELLO[ch], at=at, dur=beats, vel=sv)
    violas.note(VIOLA[ch], at=at, dur=beats, vel=sv - 6)
    if 16 <= bar < 24:
        for p in VIOLINS[ch]:
            violins.note(p, at=at, dur=beats, vel=sv - 10)
        r = pitch(ROOT[ch]) + 12
        third = 3 if ch in MINOR else 4
        run = [r + 12, r + 19, r + 24, r + 24 + third, r + 31, r + 36,
               r + 36 + third, r + 43][: int(beats * 2)]
        for i, p in enumerate(run):
            harp.note(p, at=at + i * 0.5, dur=0.5, vel=62 + (8 if i == 0 else 0))
    if bar >= 16:
        horn.note(HORN[ch], at=at, dur=beats, vel=sv - 8)
    if bar >= 24:
        bass.note(pitch(CELLO[ch]) - 12, at=at, dur=beats, vel=sv)

for i, line in enumerate(THEME_A):
    solo.melody(line, bar=16 + i, vel=84)
for i, line in enumerate(THEME_B):
    melody.melody(line, bar=24 + i, vel=100)

# nástup smyčců a závěrečné utišení: zbude jen piano
for t in (celli, violas):
    t.automate("vol", [(8, 0.0), (10, t.vol), (29, t.vol), (31, 0.0)], "s")
for t in (violins, melody, horn, bass):
    t.automate("vol", [(16, t.vol), (29, t.vol), (31, 0.0)], "s")
for t in (celli, violas, violins, melody, solo, horn, bass):
    t.humanize(timing=0.01, vel=0.06, seed=7)

s.render("output/divadlo.wav")
