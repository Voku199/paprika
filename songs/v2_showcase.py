"""Ukázka enginu v2 — temný cinematic → trap hybrid ve stylu Arcane, 96 BPM.

Tři části po osmi taktech, každá předvádí jiný kus enginu:

- intro (0-7)  — sbor „óó“ a felt piano v katedrále, taiko z cinematic
  sady, smyčce se pomalu zvednou (automatizace hlasitosti), reverzní činel
  vtáhne do buildu,
- build (8-15) — trap bicí s hi-hat rolly, temný pluck za rezonančním
  filtrem, který se otevírá (automatizace cutoffu), riser, půl doby ticha
  před dropem (automatizace masteru),
- drop (16-23) — 808 s glide (`>`), impact + braam, žesťové staccato,
  smyčcový stab přes rytmický gate, music box melodie (Cradles), bicí na
  sběrnici s kompresorem, sbor pumpuje sidechainem od bicích.

    python -m paprika show   songs/v2_showcase.py --bars 14:18
    python -m paprika render songs/v2_showcase.py --bars 14:18
    python -m paprika render songs/v2_showcase.py --play
"""

from paprika import (Compressor, Delay, Gate, Hall, HighPass, LowPass, Sidechain,
                     Song)

s = Song(bpm=96, key="D", scale="minor", bars=24, title="v2_showcase")

PROG = ["Dm", "Bb", "F", "C"]

# ---------------------------------------------------------------------------
# stopy
# ---------------------------------------------------------------------------
choir = s.track("choir_ooh", vol=0.42, fx="default")
piano = s.track("felt_piano", vol=0.85, pan=-0.15, fx="default")
strings = s.track("strings_ens", vol=0.4, pan=0.1, fx="default")
pluck = s.track("dark_pluck", vol=0.7, fx=[
    LowPass(300, res=0.45), Delay(3 / 16, fb=0.35, mix=0.2, ping_pong=True)])
bass = s.track("808", vol=0.24, fx="default")
brass = s.track("brass_section", vol=0.55, pan=-0.2,
                fx=[HighPass(160), Hall("hall", mix=0.25)])
stab = s.track("string_stab", vol=0.75, pan=0.2, fx=[
    Gate("x.x.xx.x", rate=1 / 16, depth=0.9), Hall("plate", mix=0.2)])
mbox = s.track("music_box", vol=0.6, pan=0.25, fx="default")
aah = s.track("choir_aah", vol=0.6, fx="default")

taiko = s.drums(kit="cinematic", vol=0.35, name="taiko")
trap = s.drums(kit="trap", vol=0.5, name="trap")
sfx = s.sfx(vol=0.32)

drums_bus = s.bus("drums_bus", vol=0.8).fx(
    Compressor(-16, 4, attack=0.005, release=0.12))
s.master.fx(Compressor(-10, 2, attack=0.02, release=0.2))
taiko.to(drums_bus)
trap.to(drums_bus)

# ---------------------------------------------------------------------------
# intro (0-7)
# ---------------------------------------------------------------------------
for bar, ch in enumerate(PROG * 4):
    choir.chord(ch, bar=bar, dur=4, octave=4, vel=72 if bar < 8 else 58)
for bar, ch in enumerate(PROG):
    piano.arp(ch, bar=bar, pattern="up", rate=1 / 8, octaves=1, octave=4, vel=55)
piano.melody(
    "A5:1 F5:0.5 E5:0.5 D5:1 A4:1 | Bb4:1.5 C5:0.5 D5:1 F5:1 | "
    "E5:1 C5:0.5 D5:0.5 E5:1 G5:1 | F5:3 E5:1",
    bar=4, vel=74)
for bar, ch in enumerate(PROG * 3):
    strings.chord(ch, bar=4 + bar, dur=4, inv=1, octave=3, vel=64)
strings.automate("vol", [(4, 0.0), (8, 0.4)], curve="s")
taiko.pattern("b.......b.......", bars=(2, 6), vel=0.7)
taiko.pattern("b.......b...b.b.", bars=(6, 8), vel=0.85)
sfx.reverse_swell(to_bar=8, beats=4)

# ---------------------------------------------------------------------------
# build (8-15)
# ---------------------------------------------------------------------------
for bar, ch in enumerate(PROG * 2):
    pluck.arp(ch, bar=8 + bar, pattern="updown", rate=1 / 16, octaves=2,
              octave=3, vel=90)
pluck.automate("lowpass.cutoff", [(8, 300), (16, 7000)], curve="exp")
trap.pattern("k.......k..k....\nh.h.h.h.h.h.h.h.", bars=(8, 12))
trap.pattern("k.......k..k....\n....c.......c...\nh.hhh.h.h.hhh.h.", bars=(12, 15))
trap.pattern("k.......k.......\n....c.......c...", bars=(15, 16))
trap.roll("h", bar=11, beat=3, beats=1, rate=1 / 32, vel=(50, 105))
trap.roll("s", bar=15, beat=2, beats=2, rate=1 / 32, vel=(40, 118))
sfx.riser(to_bar=16, bars=4, kind="both")
# půl doby ticha těsně před dropem
s.master.automate("vol", [(0, 1.0), (15.875, 0.0), (16, 1.0)], curve="step")

# ---------------------------------------------------------------------------
# drop (16-23)
# ---------------------------------------------------------------------------
sfx.impact(16).braam(16, "D2", bars=2).subdrop(20)
bass.notes(
    "D2:1.5 D2:0.5 _:1 >F2:1 | Bb1:2 _:1 >C2:1 | "
    "F1:1.5 F1:0.5 _:1 >A1:1 | C2:2 _:0.5 C2:0.5 >D2:1",
    bar=16, vel=110)
bass.repeat((16, 20), times=1)
for i, ch in enumerate(PROG * 2):
    bar = 16 + i
    brass.chord(ch, bar=bar, dur=0.5, octave=3, vel=104)
    brass.chord(ch, bar=bar, beat=1.5, dur=0.5, octave=3, vel=92)
    brass.chord(ch, bar=bar, beat=3, dur=1, octave=3, vel=98)
    stab.chord(ch, bar=bar, dur=4, inv=2, octave=4, vel=96)
    aah.chord(ch, bar=bar, dur=4, inv=1, octave=4, vel=70)
mbox.melody(
    "A5:1 F5:0.5 E5:0.5 D5:1 A5:1 | Bb5:1.5 C6:0.5 D6:1 F5:1 | "
    "E5:1 C6:0.5 D6:0.5 E6:1 G5:1 | F5:3 E5:1",
    bar=16, vel=86)
mbox.repeat((16, 20), times=1)
trap.pattern("k.....k...k.....\n....c.......c...\nh.hhh.h.h.hhh.h.", bars=(16, 24),
             swing=0.12)
trap.roll("h", bar=19, beat=3, beats=1, rate=1 / 32, vel=(55, 110))
trap.roll("h", bar=23, beat=2, beats=2, rate=1 / 32, vel=(55, 115))
taiko.pattern("b.......b.......", bars=(16, 24), vel=0.9)
aah.fx(Sidechain("drums_bus", amount=0.35))
choir.fx(Sidechain("drums_bus", amount=0.3))

# ---------------------------------------------------------------------------
# život
# ---------------------------------------------------------------------------
for i, t in enumerate((choir, piano, strings, pluck, bass, brass, stab, mbox, aah,
                       taiko, trap)):
    t.humanize(timing=0.01, vel=0.1, seed=100 + i)

s.render("output/v2_showcase.wav")
