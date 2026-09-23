"""Ukázka enginu v2 — temný cinematic → trap hybrid ve stylu Arcane, 96 BPM.

Tři části po osmi taktech:

- intro (0-7)  — sbor „óó“ (sekce přes Ensemble) a felt piano v katedrále,
  taiko z cinematic sady, smyčcová sekce se pomalu zvedne (automatizace),
  reverzní činel vtáhne do buildu,
- build (8-15) — trap hi-haty s rolly, staccato smyčce za filtrem, který se
  otevírá, cello linka, riser a půl doby ticha před dropem,
- drop (16-23) — 808 s glide (`>`), impact + braam, žesťová sekce, smyčcový
  stab přes rytmický gate, music box melodie (Cradles), bicí na sběrnici
  s kompresorem, sbor pumpuje sidechainem od bicích.

Proti „čipovému“ zvuku: žádná rychlá arpeggia, sekce místo sólových tónů,
HighPass na všem kromě basů, mastering „cinematic“. Kontrola:

    python -m paprika render  songs/v2_showcase.py
    python -m paprika compare songs_mp3/*.mp3 output/v2_showcase.wav
"""

from paprika import (Compressor, Delay, Gate, Hall, HighPass, LowPass, Sidechain,
                     Song, mastering)

s = Song(bpm=96, key="D", scale="minor", bars=24, title="v2_showcase")

PROG = ["Dm", "Bb", "F", "C"]
ROOT = {"Dm": "D3", "Bb": "Bb2", "F": "F2", "C": "C3"}

# ---------------------------------------------------------------------------
# stopy
# ---------------------------------------------------------------------------
choir = s.track("choir_ooh", vol=0.32, fx=[HighPass(140), Hall("cathedral", mix=0.35)])
piano = s.track("felt_piano", vol=1.0, pan=-0.2, fx=[HighPass(90), Hall("hall", mix=0.3)])
strings = s.track("strings_ens", vol=0.55, fx=[HighPass(110), Hall("hall", mix=0.3)])
spicc = s.track("strings_short", vol=0.9, pan=0.15, fx=[
    HighPass(150), LowPass(600, res=0.2), Delay(3 / 16, fb=0.25, mix=0.12, ping_pong=True),
    Hall("room", mix=0.2)])
cello = s.track("cello_solo", vol=0.6, pan=-0.25, fx=[HighPass(60), Hall("hall", mix=0.25)])
bass = s.track("808", vol=0.16, fx="default")
brass = s.track("brass_section", vol=0.6, pan=-0.1,
                fx=[HighPass(160), Hall("hall", mix=0.25)])
stab = s.track("string_stab", vol=0.6, pan=0.2, fx=[
    HighPass(180), Gate("x.x.xx.x", rate=1 / 16, depth=0.8), Hall("plate", mix=0.2)])
mbox = s.track("music_box", vol=0.6, pan=0.3, fx=[Hall("hall", mix=0.35)])
aah = s.track("choir_aah", vol=0.55, fx=[HighPass(160), Hall("cathedral", mix=0.3)])

taiko = s.drums(kit="cinematic", vol=0.3, name="taiko")
trap = s.drums(kit="trap", vol=0.34, name="trap")
sfx = s.sfx(vol=0.25)

drums_bus = s.bus("drums_bus", vol=0.8).fx(
    HighPass(35), Compressor(-16, 4, attack=0.005, release=0.12))
taiko.to(drums_bus)
trap.to(drums_bus)
s.master.fx(*mastering("cinematic"))

# ---------------------------------------------------------------------------
# intro (0-7)
# ---------------------------------------------------------------------------
for bar, ch in enumerate(PROG * 4):
    choir.chord(ch, bar=bar, dur=4, octave=4, vel=74 if bar < 8 else 60)
# rozložené akordy v klidných dobách, žádné strojové šestnáctiny
for bar, ch in enumerate(PROG):
    piano.notes(f"{ROOT[ch]}:1", bar=bar, vel=58)
    piano.chord(ch, bar=bar, beat=1, dur=3, octave=4, inv=1, vel=48)
piano.melody(
    "A5:1.5 F5:0.5 E5:1 D5:1 | Bb4:2 C5:1 D5:1 | "
    "E5:1.5 C5:0.5 D5:1 E5:1 | F5:3 E5:1",
    bar=4, vel=72)
for bar, ch in enumerate(PROG * 3):
    strings.chord(ch, bar=4 + bar, dur=4, inv=1, octave=3, vel=66)
strings.automate("vol", [(4, 0.0), (8, 0.55)], curve="s")
taiko.pattern("b.......b.......", bars=(2, 6), vel=0.7)
taiko.pattern("b.......b...b.b.", bars=(6, 8), vel=0.85)
sfx.reverse_swell(to_bar=8, beats=4)

# ---------------------------------------------------------------------------
# build (8-15)
# ---------------------------------------------------------------------------
for bar, ch in enumerate(PROG * 2):
    spicc.arp(ch, bar=8 + bar, pattern="up", rate=1 / 8, octaves=1, octave=4,
              vel=84, gate=0.6)
spicc.automate("lowpass.cutoff", [(8, 600), (16, 8000)], curve="exp")
cello.melody(
    "D3:2 F3:1 A3:1 | Bb2:3 C3:1 | F3:2 E3:1 D3:1 | C3:3 E3:1 | "
    "D3:2 F3:1 A3:1 | Bb3:2 A3:1 G3:1 | A3:2 G3:1 F3:1 | E3:2 C3:2",
    bar=8, vel=80)
trap.pattern("k.......k..k....\nh...h...h...h...", bars=(8, 12))
trap.pattern("k.......k..k....\n....c.......c...\nh.h.h.h.h.h.h.h.", bars=(12, 15),
             swing=0.1)
trap.pattern("k.......k.......\n....c.......c...", bars=(15, 16))
trap.roll("h", bar=11, beat=3, beats=1, rate=1 / 32, vel=(50, 100))
trap.roll("s", bar=15, beat=2, beats=2, rate=1 / 32, vel=(40, 115))
sfx.riser(to_bar=16, bars=4, kind="both")
# půl doby ticha těsně před dropem
s.master.automate("vol", [(0, 1.0), (15.875, 0.0), (16, 1.0)], curve="step")

# ---------------------------------------------------------------------------
# drop (16-23)
# ---------------------------------------------------------------------------
sfx.impact(16, size=0.8).braam(16, "D2", bars=2)
bass.notes(
    "D2:1.5 D2:0.5 _:1 >F2:1 | Bb1:2 _:1 >C2:1 | "
    "F2:1.5 F2:0.5 _:1 >A2:1 | C2:2 _:0.5 C2:0.5 >D2:1",
    bar=16, vel=110)
bass.repeat((16, 20), times=1)
for i, ch in enumerate(PROG * 2):
    bar = 16 + i
    brass.chord(ch, bar=bar, dur=0.75, octave=3, vel=104)
    brass.chord(ch, bar=bar, beat=1.5, dur=0.5, octave=3, vel=90)
    brass.chord(ch, bar=bar, beat=3, dur=1, octave=3, vel=98)
    stab.chord(ch, bar=bar, dur=4, inv=2, octave=4, vel=96)
    aah.chord(ch, bar=bar, dur=4, inv=1, octave=4, vel=72)
    strings.chord(ch, bar=bar, dur=4, inv=1, octave=3, vel=70)
mbox.melody(
    "A5:1 F5:0.5 E5:0.5 D5:1 A5:1 | Bb5:1.5 C6:0.5 D6:1 F5:1 | "
    "E5:1 C6:0.5 D6:0.5 E6:1 G5:1 | F5:3 E5:1",
    bar=16, vel=86)
mbox.repeat((16, 20), times=1)
trap.pattern("k.....k...k.....\n....c.......c...\nh.hhh.h.h.hhh.h.", bars=(16, 24),
             swing=0.12)
trap.roll("h", bar=19, beat=3, beats=1, rate=1 / 32, vel=(55, 105))
trap.roll("h", bar=23, beat=2, beats=2, rate=1 / 32, vel=(55, 110))
taiko.pattern("b.......b.......", bars=(16, 24), vel=0.85)
aah.fx(Sidechain("drums_bus", amount=0.3))
strings.fx(Sidechain("drums_bus", amount=0.25))

# ---------------------------------------------------------------------------
# život
# ---------------------------------------------------------------------------
for i, t in enumerate((choir, piano, strings, spicc, cello, bass, brass, stab, mbox,
                       aah, taiko, trap)):
    t.humanize(timing=0.012, vel=0.1, seed=100 + i)

s.render("output/v2_showcase.wav")
