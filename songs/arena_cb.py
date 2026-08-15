"""arena_cb — podkres pro web Laser Game České Budějovice.

Web má tmavě petrolejové pozadí (#002F3F) a chromově zelený akcent
(#6ABE45), dva týmy modrý/červený. Taktická, trochu sci-fi nálada
(radar, laser, aréna), ne cukrářský synthwave — tvrdší, chladnější.

D moll (D E F G A Bb C), 116 BPM. Smyčka Dm - Bb - F - C (i-VI-III-VII),
stejná logika jako u nocni_dalnice: mollové akordy nesou napětí, durové
uvnitř (Bb, F, C) dávají teplo, aby to nebylo jen temné.

Oblouk: intro (radar ping, jen pad + zvon) -> build (bas + hi-haty
nastupují) -> hlavní smyčka (plný puls, pluck jako "laser" beam,
lead občas prolétne) -> breakdown (nádech) -> druhá, širší smyčka ->
outro (dozvuk pingu).

Cíleně tišší a míň nabité než skladba na poslech — tohle běží na pozadí
webu, kde lidi čtou text a rezervují termín, ne poslouchají hudbu.
Proto: subbass hodně zkrocený sidechainem, žádný agresivní lead sólo,
hi-hat drží groove místo velkého kick+snare boje.

40 taktů @ 116 BPM = cca 1:23 hudby, ocásek delaye/reverbu o pár vteřin
prodlouží.

    python -m paprika show   songs/arena_cb.py --bars 8:12
    python -m paprika render songs/arena_cb.py
"""

from paprika import Delay, HighPass, LowPass, Reverb, Saturate, Sidechain, Song

s = Song(bpm=116, key="D", scale="minor", bars=40, title="arena_cb")

BPB = 4


def b(bar, beat=0.0):
    """Takt (+ beat uvnitř taktu) -> pozice na timeline v beatech."""
    return bar * BPB + beat


def pulse(note, n=8, dur=0.5):
    """Osminové opakování basového tónu na jeden takt."""
    return " ".join([f"{note}:{dur}"] * n)


LOOP = ["Dm", "Bb", "F", "C"]
ROOT = {"Dm": "D2", "Bb": "Bb1", "F": "F2", "C": "C2", "Am": "A2"}

# --------------------------------------------------------------------------
# nástroje
# --------------------------------------------------------------------------

pad = s.track("warmpad", vol=0.50, pan=0.0, name="pad")
pad.fx(HighPass(140), LowPass(900), Reverb(room=0.80, mix=0.38, damp=0.48),
       Delay(1 / 4, fb=0.18, mix=0.10))

# "laser beam" — ostrý pluck, mírně doprava, krátký delay pro pocit odrazu
arp = s.track("pluck", vol=0.90, pan=0.24, name="arp")
arp.fx(HighPass(500), Delay(1 / 8, fb=0.26, mix=0.20),
       Reverb(room=0.45, mix=0.16, damp=0.55))

# sub drží pod 450 Hz, sidechain od bicích ať kick dýchá
bass = s.track("subbass", vol=0.18, name="bass")
bass.fx(LowPass(450), Saturate(1.6, mix=0.30),
        Sidechain(source="drums", amount=0.60, attack=0.004, release=0.14))

# lead jen na pár nápadů, ne celou dobu — akcent, ne melodie na popředí
lead = s.track("lead", vol=0.62, pan=-0.20, name="lead")
lead.fx(HighPass(500), Delay(3 / 16, fb=0.24, mix=0.18),
        Reverb(room=0.60, mix=0.24, damp=0.45))

# "radar ping" — zvon vlevo, řídce, čistě dekorativní
bell = s.track("bell", vol=0.55, pan=-0.34, name="bell")
bell.fx(HighPass(1200), Reverb(room=0.85, mix=0.36, damp=0.32))

dr = s.drums(vol=0.62)


# --------------------------------------------------------------------------
# A  takty 0-3 — intro, jen radar ping a pad
# --------------------------------------------------------------------------

pad.chord("Dm", bar=0, dur=16, octave=4, inv=1, vel=36)

bell.note("A5", at=b(0, 0), dur=3, vel=48)
bell.note("D6", at=b(2, 0), dur=3, vel=44)


# --------------------------------------------------------------------------
# B  takty 4-11 — build: bas a hi-haty nastupují, arp naznačí puls
# --------------------------------------------------------------------------

for i in range(8):
    bar = 4 + i
    name = LOOP[i % 4]
    grow = i * 2

    if i % 4 == 0:
        pad.chord(name, bar=bar, dur=4, octave=4, inv=1, vel=40 + grow // 2)
    arp.arp(name, bar=bar, pattern="up", rate=1 / 8, octaves=1,
            octave=4, vel=42 + grow)
    bass.notes(f"{ROOT[name]}:4", bar=bar, vel=48 + grow)

dr.pattern("""
h.h.h.h.h.h.h.h.
k.......k.......
""", bars=(4, 8), vel=0.35)

dr.pattern("""
h.h.h.h.h.h.h.h.
k...k...k...k...
""", bars=(8, 12), vel=0.48)


# --------------------------------------------------------------------------
# C  takty 12-23 — hlavní smyčka, plný puls (ale zkrocený pro pozadí)
# --------------------------------------------------------------------------

for i in range(12):
    bar = 12 + i
    name = LOOP[i % 4]

    if i % 4 == 0:
        pad.chord(name, bar=bar, dur=16, octave=4, inv=1, vel=54)
    arp.arp(name, bar=bar, pattern="updown", rate=1 / 16, octaves=2,
            octave=4, vel=58)
    bass.notes(pulse(ROOT[name]), bar=bar, vel=74)

dr.pattern("k...k...k...k...", bars=(12, 24), vel=0.55)
dr.pattern("""
h.h.h.h.h.h.h.h.
....c.......c...
""", bars=(12, 24), vel=0.62)

LEAD_HITS = [
    (14, "D5:1 F5:1 A5:2"),
    (18, "F5:1 A5:1 D6:2"),
    (22, "A4:1 D5:1 F5:2"),
]
for bar, seq in LEAD_HITS:
    lead.melody(seq, bar=bar, vel=58)

bell.note("A5", at=b(15, 3), dur=1, vel=46)
bell.note("D6", at=b(21, 3), dur=1, vel=48)


# --------------------------------------------------------------------------
# D  takty 24-27 — breakdown, krátký nádech
# --------------------------------------------------------------------------

pad.chord("Dm", bar=24, dur=8, octave=4, inv=1, vel=34)
pad.chord("Am", bar=26, dur=6, octave=4, inv=1, vel=30)

arp.arp("Dm", bar=24, pattern="up", rate=1 / 8, octaves=1, bars=2,
        octave=4, vel=36)

bass.notes("D2:2", bar=24, vel=40)
bass.notes("A2:2", bar=26, vel=36)

dr.pattern("h...h...h...h...", bars=(24, 28), vel=0.28)

bell.note("D6", at=b(25, 2), dur=2, vel=40)


# --------------------------------------------------------------------------
# E  takty 28-35 — druhá smyčka, o trochu širší (víc pádu, ne víc hlasitosti)
# --------------------------------------------------------------------------

for i in range(8):
    bar = 28 + i
    name = LOOP[i % 4]

    if i % 4 == 0:
        pad.chord(name, bar=bar, dur=16, octave=4, inv=1, vel=58)
    arp.arp(name, bar=bar, pattern="updown", rate=1 / 16, octaves=2,
            octave=4, vel=62)
    bass.notes(pulse(ROOT[name]), bar=bar, vel=78)

dr.pattern("k...k...k...k...", bars=(28, 36), vel=0.58)
dr.pattern("""
r.r.r.r.r.r.r.r.
....C.......C...
""", bars=(28, 36), vel=0.66)

lead.melody("D5:1 F5:1 A5:1 D6:1 | A5:1 F5:1 D5:2", bar=32, vel=62)
bell.note("A5", at=b(31, 3), dur=1, vel=48)
bell.note("D6", at=b(35, 0), dur=2, vel=50)


# --------------------------------------------------------------------------
# F  takty 36-39 — outro, doznění pingu
# --------------------------------------------------------------------------

pad.chord("Dm", bar=36, dur=16, octave=4, inv=1, vel=40)
arp.arp("Dm", bar=36, pattern="up", rate=1 / 8, octaves=1, bars=2,
        octave=4, vel=36)
bass.note("D2", at=b(36), dur=4, vel=42)

dr.hit("kick", at=b(36, 0), vel=52)
dr.hit("openhat", at=b(36, 0), vel=40)

bell.note("A5", at=b(37, 2), dur=2, vel=42)
bell.note("D6", at=b(39, 0), dur=4, vel=36)


# --------------------------------------------------------------------------
# rozhýbání
# --------------------------------------------------------------------------

pad.humanize(timing=0.04, vel=0.08, seed=101)
arp.humanize(timing=0.010, vel=0.12, seed=103)
bass.humanize(timing=0.009, vel=0.07, seed=107)
lead.humanize(timing=0.014, vel=0.10, seed=109)
bell.humanize(timing=0.018, vel=0.09, seed=113)
dr.humanize(timing=0.007, vel=0.10, seed=127)

s.render("output/arena_cb.wav")
