"""arena_ingame — hudba do sálu při hře (ne web pozadí, viz arena_cb.py).

Jiný kontext než web: hráči běhaj, střílej, kolo trvá 5-10 min ze
sálových reproduktorů. Nesmí přebít pípání laser zbraní (to sedí kolem
1-4 kHz) a nesmí za 5-10 min omrzet ani unavit sluch.

Řešení: industrial techno. Málo melodie, žádný lead nahoře v pásmu
pípání zbraní. Energii nesou bicí + bas, "stab" akordy (organ,
zfiltrovaný nízko a saturovaný) jsou tmavé a krátké — vyplní groove,
ale nesoupeří s herními zvuky.

E moll (E F# G A B C D), 136 BPM. Smyčka Em - C - G - D (i-VI-III-VII),
stejný trik jako u ostatních skladeb — mollový základ, durové akordy
uvnitř dávají tahu bez ztráty temnoty.

Oblouk: krátký nájezd (2 takty, kolo právě začíná) -> hlavní groove
(32 taktů, konstantní energie, žádný breakdown — hra nesmí dostat
uprostřed kola pokles) -> tomový fill každých 8 taktů pro drobnou
variaci -> konec na tónice pro čistý loop point.

38 taktů @ 136 BPM = cca 1:07 hudby; v sále se pouští ve smyčce.

    python -m paprika show   songs/arena_ingame.py --bars 4:8
    python -m paprika render songs/arena_ingame.py
"""

from paprika import LowPass, Saturate, Sidechain, Song

s = Song(bpm=136, key="E", scale="minor", bars=38, title="arena_ingame")

BPB = 4


def b(bar, beat=0.0):
    """Takt (+ beat uvnitř taktu) -> pozice na timeline v beatech."""
    return bar * BPB + beat


def pulse(note, n=8, dur=0.5):
    """Osminové opakování basového tónu na jeden takt."""
    return " ".join([f"{note}:{dur}"] * n)


LOOP = ["Em", "C", "G", "D"]
ROOT = {"Em": "E2", "C": "C2", "G": "G2", "D": "D2"}

# --------------------------------------------------------------------------
# nástroje
# --------------------------------------------------------------------------

# hnací sub, agresivnější saturace a tvrdší sidechain než web verze —
# tohle jede ze sálových beden, ne z prohlížeče
bass = s.track("subbass", vol=0.38, name="bass")
bass.fx(LowPass(400), Saturate(2.4, mix=0.5),
        Sidechain(source="drums", amount=0.72, attack=0.003, release=0.11))

# "industrial stab" — organ zfiltrovaný hluboko, drží se pod pípáním zbraní
stab = s.track("organ", vol=0.60, pan=0.0, name="stab")
stab.fx(LowPass(700), Saturate(1.8, mix=0.4))

dr = s.drums(vol=0.82)
dr.fx(Saturate(1.4, mix=0.25))


# --------------------------------------------------------------------------
# A  takty 0-1 — krátký nájezd, kolo začíná
# --------------------------------------------------------------------------

stab.chord("Em", bar=0, dur=2, octave=3, inv=0, vel=60)
bass.notes("E2:2", bar=0, vel=70)

dr.pattern("""
k...............
............s...
""", bars=(0, 2), vel=0.70)


# --------------------------------------------------------------------------
# B  takty 2-33 — hlavní groove, konstantní energie, bez propadu
# --------------------------------------------------------------------------

for i in range(32):
    bar = 2 + i
    name = LOOP[i % 4]

    if i % 4 == 0:
        stab.chord(name, bar=bar, dur=1.5, octave=3, inv=0, vel=78)
        stab.chord(name, bar=bar, beat=2.5, dur=1, octave=3, inv=0, vel=62)
    bass.notes(pulse(ROOT[name]), bar=bar, vel=100)

dr.pattern("k...k...k...k...", bars=(2, 34), vel=0.85)
dr.pattern("""
..h...h...h...h.
....c.......c...
""", bars=(2, 34), vel=0.78)

# tomový fill každých 8 taktů — drobná variace, energie neklesá
for fill_bar in (9, 17, 25):
    dr.pattern("t.t.T.t.t.T.tTtT", bars=(fill_bar, fill_bar + 1), vel=0.82)

# poslední takt před koncem — o něco tvrdší nájezd na tóniku
dr.pattern("K.K.K.K.K.K.K.K.", bars=(33, 34), vel=0.95)


# --------------------------------------------------------------------------
# C  takty 34-37 — konec na tónice, čistý loop point
# --------------------------------------------------------------------------

stab.chord("Em", bar=34, dur=4, octave=3, inv=0, vel=82)
bass.notes(pulse("E2"), bar=34, vel=104)
bass.notes(pulse("E2"), bar=35, vel=100)
bass.note("E2", at=b(36), dur=8, vel=90)

dr.pattern("k...k...k...k...", bars=(34, 36), vel=0.88)
dr.pattern("""
..h...h...h...h.
....c.......c...
""", bars=(34, 36), vel=0.80)
dr.hit("kick", at=b(36, 0), vel=100)
dr.hit("ride", at=b(36, 0), vel=80)


# --------------------------------------------------------------------------
# rozhýbání
# --------------------------------------------------------------------------

bass.humanize(timing=0.006, vel=0.06, seed=201)
stab.humanize(timing=0.010, vel=0.10, seed=203)
dr.humanize(timing=0.005, vel=0.08, seed=207)

s.render("output/arena_ingame.wav")
