"""Hotové nástroje. Každý je obyčejný `Patch`, takže jde po vytvoření
doladit (`p = preset("lead"); p.filter.cutoff = 900`).

Presety mají tagy (`PRESET_TAGS`, `presets_by_tag("choir")`): classic (osm
presetů z v1), orchestral, choir, keys, darkpop, electronic, lofi.
Některé mají doporučené výchozí nastavení stopy (`PRESET_DEFAULTS`, např.
808 je mono s glide) a doporučený fx řetěz (`preset_fx`, připojí se přes
`s.track("808", fx="default")`).
"""

from __future__ import annotations

from .patch import (ADSR, BP, FM, LFO, LP, Additive, Formant, Noise, Patch,
                    PitchEnv, Pluck, Pulse, Saw, Sine, Square, SuperSaw, Tri)

__all__ = ["PRESETS", "PRESET_TAGS", "PRESET_DEFAULTS", "preset", "preset_names",
           "preset_fx", "presets_by_tag", "tag_names"]


def _epiano() -> Patch:
    # Rhodes: FM nosič s modulátorem, který rychle odezní, plus lehká oktáva.
    return Patch(
        osc=[FM(1.0, amp=1.0, mod=2.0, index=3.2, decay=0.45),
             Sine(1.0, amp=0.25),
             Sine(2.005, amp=0.10)],
        adsr=ADSR(0.004, 1.6, 0.22, 0.55),
        filter=LP(4200, env=0.25),
        gain=0.62,
        name="epiano",
    )


def _warmpad() -> Patch:
    return Patch(
        osc=[Saw(1.0, amp=0.5), Saw(1.006, amp=0.45), Saw(0.994, amp=0.45),
             Sine(0.5, amp=0.35)],
        adsr=ADSR(0.55, 1.2, 0.75, 1.6),
        filter=LP(1700, env=0.35),
        gain=0.5,
        name="warmpad",
    )


def _subbass() -> Patch:
    return Patch(
        osc=[Sine(1.0, amp=1.0), Sine(2.0, amp=0.18), Tri(1.0, amp=0.12)],
        adsr=ADSR(0.008, 0.25, 0.85, 0.16),
        filter=LP(600, env=0.4),
        gain=0.68,
        name="subbass",
    )


def _pluck() -> Patch:
    return Patch(
        osc=[Saw(1.0, amp=0.8), Square(1.002, amp=0.3), Sine(2.0, amp=0.15)],
        adsr=ADSR(0.003, 0.32, 0.0, 0.22),
        filter=LP(2600, env=0.85),
        gain=0.7,
        name="pluck",
    )


def _lead() -> Patch:
    return Patch(
        osc=[Square(1.0, amp=0.55), Saw(1.004, amp=0.45), Saw(0.996, amp=0.35),
             Sine(2.0, amp=0.12)],
        adsr=ADSR(0.02, 0.25, 0.65, 0.3),
        filter=LP(3200, env=0.45),
        gain=0.6,
        name="lead",
    )


def _strings() -> Patch:
    return Patch(
        osc=[Saw(1.0, amp=0.4), Saw(1.008, amp=0.4), Saw(0.992, amp=0.4),
             Saw(2.003, amp=0.15)],
        adsr=ADSR(0.28, 0.7, 0.8, 1.1),
        filter=LP(2400, env=0.3),
        gain=0.45,
        name="strings",
    )


def _bell() -> Patch:
    return Patch(
        osc=[FM(1.0, amp=1.0, mod=3.5, index=5.0, decay=0.25),
             Sine(2.76, amp=0.22),
             Sine(5.4, amp=0.08)],
        adsr=ADSR(0.002, 2.4, 0.0, 1.2),
        filter=None,
        gain=0.6,
        name="bell",
    )


def _organ() -> Patch:
    return Patch(
        osc=[Sine(1.0, amp=0.9), Sine(2.0, amp=0.5), Sine(3.0, amp=0.3),
             Sine(4.0, amp=0.18), Sine(6.0, amp=0.08)],
        adsr=ADSR(0.012, 0.05, 0.95, 0.08),
        filter=LP(5200),
        gain=0.42,
        name="organ",
    )


# --------------------------------------------------------------------------
# orchestral
# --------------------------------------------------------------------------

def _strings_ens() -> Patch:
    # sekce smyčců: unison rozladěných pil, formant „a“ dává dřevo těla,
    # vibrato nastoupí až po chvíli jako u živých hráčů
    return Patch(
        osc=[SuperSaw(voices=5, detune=0.18, spread=0.7, seed=1), Saw(2.0, amp=0.1)],
        adsr=ADSR(0.35, 0.5, 0.85, 1.2),
        filter=[LP(3200, res=0.1), Formant("a", gain=1.5)],
        vibrato=LFO(5.2, 0.12, delay=0.3),
        rr=3, gain=1.6, name="strings_ens")


def _strings_short() -> Patch:
    return Patch(
        osc=[SuperSaw(voices=5, detune=0.18, spread=0.6, seed=2), Saw(1.0, amp=0.3)],
        adsr=ADSR(0.008, 0.18, 0.0, 0.15),
        filter=LP(2600, env=0.6),
        rr=4, vel_to_cutoff=0.5, gain=0.7, name="strings_short")


def _pizz() -> Patch:
    return Patch(
        osc=[Pluck(decay=0.6, bright=0.35), Sine(1.0, amp=0.2)],
        adsr=ADSR(0.001, 0.4, 0.0, 0.2),
        rr=4, gain=0.8, name="pizz")


def _cello_solo() -> Patch:
    return Patch(
        osc=[Saw(1.0, amp=0.6), Saw(1.003, amp=0.5)],
        adsr=ADSR(0.12, 0.4, 0.8, 0.6),
        filter=Formant("o", voice="bass", gain=2.5),
        vibrato=LFO(5.0, 0.15, delay=0.25),
        drift=4.0, gain=1.6, name="cello_solo")


def _brass_section() -> Patch:
    # žesťový „bite“: filtr se otevře s obálkou a víc při silném úhozu
    return Patch(
        osc=[Saw(1.0, amp=0.5), Saw(1.004, amp=0.4), Saw(0.996, amp=0.4)],
        adsr=ADSR(0.06, 0.3, 0.75, 0.4),
        filter=LP(900, env=1.0, res=0.15),
        vel_to_cutoff=0.7, vibrato=LFO(5.5, 0.06, delay=0.4),
        gain=0.55, name="brass_section")


def _horn_swell() -> Patch:
    return Patch(
        osc=[Saw(1.0, amp=0.6), Square(0.5, amp=0.3, width=0.4)],
        adsr=ADSR(0.8, 0.5, 0.9, 1.0),
        filter=[LP(700, env=0.5), Formant("o", gain=2.0)],
        vibrato=LFO(4.8, 0.05, delay=0.6), gain=1.0, name="horn_swell")


def _harp() -> Patch:
    return Patch(
        osc=[Pluck(decay=2.5, bright=0.6), Additive(((2.0, 0.15, 0.4),))],
        adsr=ADSR(0.001, 2.0, 0.0, 1.0),
        rr=3, gain=0.45, name="harp")


def _timpani_tone() -> Patch:
    return Patch(
        osc=[Additive(((1.0, 1.0, 0.9), (1.5, 0.5, 0.5), (1.98, 0.35, 0.4),
                       (2.44, 0.2, 0.3))), Noise(amp=0.05, color="brown")],
        adsr=ADSR(0.002, 1.5, 0.0, 1.2),
        pitch_env=PitchEnv(1.0, 0.08), filter=LP(1800),
        gain=0.9, name="timpani_tone")


# --------------------------------------------------------------------------
# choir
# --------------------------------------------------------------------------

def _choir(vowel: str, name: str, extra_lp: float | None = None) -> Patch:
    filters = [Formant(vowel, gain=3.0)]
    if extra_lp:
        filters.append(LP(extra_lp))
    return Patch(
        osc=[SuperSaw(voices=4, detune=0.12, spread=0.6, seed=3),
             Noise(amp=0.03, color="pink")],
        adsr=ADSR(0.4, 0.5, 0.9, 1.2),
        filter=filters,
        vibrato=LFO(5.0, 0.1, delay=0.2), drift=6.0, rr=3,
        gain=1.0, name=name)


def _choir_ooh() -> Patch:
    return _choir("u", "choir_ooh")


def _choir_aah() -> Patch:
    return _choir("a", "choir_aah")


def _choir_mm() -> Patch:
    # zavřená ústa: „u“ a navíc zatemnit
    return _choir("u", "choir_mm", extra_lp=1200)


def _vox_chop() -> Patch:
    return Patch(
        osc=[Pulse(width=0.3), Saw(1.0, amp=0.5)],
        adsr=ADSR(0.005, 0.15, 0.3, 0.1),
        filter=Formant("a", morph=("a", "o", 0.15), gain=3.0),
        pitch_env=PitchEnv(-2.0, 0.05), gain=0.4, name="vox_chop")


# --------------------------------------------------------------------------
# keys
# --------------------------------------------------------------------------

def _felt_piano() -> Patch:
    # tlumené piano: harmonické s vlastním útlumem, lehce roztažené, úder
    # kladívka jako krátký hnědý šum
    return Patch(
        osc=[Additive(((1, 1.0, 1.8), (2, 0.45, 1.0), (3, 0.25, 0.6), (4, 0.12, 0.4),
                       (5, 0.08, 0.3), (6, 0.04, 0.2)), inharm=0.0004),
             Noise(amp=0.04, color="brown")],
        adsr=ADSR(0.002, 2.5, 0.0, 0.6),
        filter=LP(2500, env=0.3), vel_to_cutoff=0.6, rr=3,
        gain=0.9, name="felt_piano")


def _toy_piano() -> Patch:
    return Patch(
        osc=[Additive(((1, 1.0, 0.5), (2.95, 0.4, 0.2), (5.8, 0.2, 0.1)))],
        adsr=ADSR(0.001, 0.8, 0.0, 0.3), rr=2,
        gain=0.8, name="toy_piano")


def _music_box() -> Patch:
    return Patch(
        osc=[Additive(((1, 1.0, 1.2), (3.9, 0.3, 0.25), (8.1, 0.12, 0.1)))],
        adsr=ADSR(0.001, 1.5, 0.0, 0.8), rr=2,
        gain=0.8, name="music_box")


def _celesta() -> Patch:
    return Patch(
        osc=[FM(1.0, amp=1.0, mod=4.0, index=2.0, decay=0.3), Sine(2.0, amp=0.3)],
        adsr=ADSR(0.001, 1.2, 0.0, 0.9), rr=2,
        gain=0.6, name="celesta")


# --------------------------------------------------------------------------
# darkpop / trap
# --------------------------------------------------------------------------

def _808() -> Patch:
    return Patch(
        osc=[Sine(1.0), Sine(2.0, amp=0.08)],
        adsr=ADSR(0.002, 1.5, 0.6, 0.3),
        pitch_env=PitchEnv(7.0, 0.06), gain=0.85, name="808")


def _reese() -> Patch:
    return Patch(
        osc=[Saw(1.0, amp=0.5), Saw(1.012, amp=0.45), Saw(0.994, amp=0.45)],
        adsr=ADSR(0.01, 0.3, 0.9, 0.2),
        filter=LP(450, res=0.35, lfo=LFO(0.25, 0.8, "tri")),
        gain=0.55, name="reese")


def _dark_pluck() -> Patch:
    return Patch(
        osc=[Saw(1.0, amp=0.7), Square(0.5, amp=0.3)],
        adsr=ADSR(0.002, 0.25, 0.0, 0.2),
        filter=LP(500, env=0.9, res=0.5), vel_to_cutoff=0.8,
        gain=0.65, name="dark_pluck")


def _detuned_lead() -> Patch:
    return Patch(
        osc=[SuperSaw(voices=3, detune=0.3, spread=0.4, seed=4),
             Pulse(width=0.5, amp=0.3, pwm=LFO(0.4, 0.3))],
        adsr=ADSR(0.02, 0.2, 0.8, 0.3),
        filter=LP(2800, res=0.2), vibrato=LFO(5.8, 0.1, delay=0.35),
        gain=0.55, name="detuned_lead")


# --------------------------------------------------------------------------
# electronic
# --------------------------------------------------------------------------

def _supersaw() -> Patch:
    return Patch(
        osc=[SuperSaw(voices=7, detune=0.35, spread=1.0, seed=5)],
        adsr=ADSR(0.01, 0.3, 0.85, 0.5),
        filter=LP(6000), gain=0.5, name="supersaw")


def _acid() -> Patch:
    return Patch(
        osc=[Saw(1.0)],
        adsr=ADSR(0.002, 0.2, 0.2, 0.1),
        filter=LP(350, env=1.0, res=0.85, poles=4), vel_to_cutoff=1.0,
        gain=0.22, name="acid")


def _string_stab() -> Patch:
    # Moby: krátký smyčcový „stab“ s dozvukem
    return Patch(
        osc=[SuperSaw(voices=5, detune=0.2, spread=0.6, seed=6), Saw(2.0, amp=0.2)],
        adsr=ADSR(0.005, 0.25, 0.35, 0.35),
        filter=[LP(1800, env=0.6), Formant("e", gain=1.2)], rr=2,
        gain=2.5, name="string_stab")


def _pwm_pad() -> Patch:
    return Patch(
        osc=[Pulse(width=0.5, pwm=LFO(0.3, 0.35, "tri")),
             Pulse(1.004, width=0.5, pwm=LFO(0.23, 0.3, "tri", seed=1))],
        adsr=ADSR(0.8, 1.0, 0.8, 1.8),
        filter=LP(1600, lfo=LFO(0.1, 0.5)), gain=0.4, name="pwm_pad")


def _air_pad() -> Patch:
    return Patch(
        osc=[Noise(amp=0.6, color="pink"), Sine(1.0, amp=0.15)],
        adsr=ADSR(1.2, 1.0, 0.8, 2.5),
        filter=[BP(1200, res=0.7, lfo=LFO(0.07, 1.0)), Formant("i", gain=4.0)],
        gain=20.0, name="air_pad")


# --------------------------------------------------------------------------
# lofi / dreamy
# --------------------------------------------------------------------------

def _lofi_keys() -> Patch:
    return Patch(
        osc=[FM(1.0, amp=1.0, mod=1.0, index=1.8, decay=0.6), Sine(2.0, amp=0.15)],
        adsr=ADSR(0.003, 1.4, 0.25, 0.5),
        filter=LP(2200), drift=8.0, tremolo=LFO(4.5, 0.12),
        gain=0.6, name="lofi_keys")


def _glass_pad() -> Patch:
    return Patch(
        osc=[Additive(((1, 1.0, 0.0), (2, 0.4, 0.0), (4.02, 0.15, 0.0))),
             FM(1.0, amp=0.5, mod=3.5, index=0.8, decay=0.0)],
        adsr=ADSR(1.0, 1.0, 0.8, 2.0),
        vibrato=LFO(0.3, 0.05), gain=0.55, name="glass_pad")


def _shimmer_pad() -> Patch:
    return Patch(
        osc=[SuperSaw(voices=5, detune=0.1, spread=0.9, seed=7), Sine(2.0, amp=0.3),
             Sine(4.0, amp=0.1)],
        adsr=ADSR(1.5, 1.0, 0.85, 3.0),
        filter=LP(4000, lfo=LFO(0.12, 0.6)), gain=0.5, name="shimmer_pad")


def _round_bass() -> Patch:
    return Patch(
        osc=[Tri(1.0), Sine(1.0, amp=0.6)],
        adsr=ADSR(0.005, 0.3, 0.8, 0.2),
        filter=LP(700, env=0.3), rr=2, gain=0.55, name="round_bass")


PRESETS = {
    "epiano": _epiano,
    "warmpad": _warmpad,
    "subbass": _subbass,
    "pluck": _pluck,
    "lead": _lead,
    "strings": _strings,
    "bell": _bell,
    "organ": _organ,
    # orchestral
    "strings_ens": _strings_ens,
    "strings_short": _strings_short,
    "pizz": _pizz,
    "cello_solo": _cello_solo,
    "brass_section": _brass_section,
    "horn_swell": _horn_swell,
    "harp": _harp,
    "timpani_tone": _timpani_tone,
    # choir
    "choir_ooh": _choir_ooh,
    "choir_aah": _choir_aah,
    "choir_mm": _choir_mm,
    "vox_chop": _vox_chop,
    # keys
    "felt_piano": _felt_piano,
    "toy_piano": _toy_piano,
    "music_box": _music_box,
    "celesta": _celesta,
    # darkpop
    "808": _808,
    "reese": _reese,
    "dark_pluck": _dark_pluck,
    "detuned_lead": _detuned_lead,
    # electronic
    "supersaw": _supersaw,
    "acid": _acid,
    "string_stab": _string_stab,
    "pwm_pad": _pwm_pad,
    "air_pad": _air_pad,
    # lofi
    "lofi_keys": _lofi_keys,
    "glass_pad": _glass_pad,
    "shimmer_pad": _shimmer_pad,
    "round_bass": _round_bass,
}

_TAGGED = {
    "classic": ("epiano", "warmpad", "subbass", "pluck", "lead", "strings", "bell",
                "organ"),
    "orchestral": ("strings_ens", "strings_short", "pizz", "cello_solo",
                   "brass_section", "horn_swell", "harp", "timpani_tone"),
    "choir": ("choir_ooh", "choir_aah", "choir_mm", "vox_chop"),
    "keys": ("felt_piano", "toy_piano", "music_box", "celesta"),
    "darkpop": ("808", "reese", "dark_pluck", "detuned_lead"),
    "electronic": ("supersaw", "acid", "string_stab", "pwm_pad", "air_pad"),
    "lofi": ("lofi_keys", "glass_pad", "shimmer_pad", "round_bass"),
}

PRESET_TAGS: dict[str, tuple[str, ...]] = {}
for _tag, _names in _TAGGED.items():
    for _n in _names:
        PRESET_TAGS[_n] = PRESET_TAGS.get(_n, ()) + (_tag,)

# výchozí nastavení stopy, když ho autor nezadá sám
PRESET_DEFAULTS = {
    "808": {"mono": True, "glide": 0.12},
    "acid": {"mono": True, "glide": 0.08},
    "cello_solo": {"mono": True, "glide": 0.06},
    "detuned_lead": {"mono": True, "glide": 0.05},
}


def _default_fx(name: str) -> list:
    from .fx import Saturate
    from .fx_lofi import BitCrush, Tape
    from .fx_space import Chorus, Hall
    table = {
        "808": lambda: [Saturate(2.5, mix=0.7)],
        "reese": lambda: [Chorus(0.3, 4.0, 0.3)],
        "acid": lambda: [Saturate(2.0, mix=0.5)],
        "string_stab": lambda: [Hall("plate", mix=0.25)],
        "strings_ens": lambda: [Hall("hall", mix=0.3)],
        "choir_ooh": lambda: [Hall("cathedral", mix=0.35)],
        "choir_aah": lambda: [Hall("cathedral", mix=0.35)],
        "choir_mm": lambda: [Hall("hall", mix=0.3)],
        "felt_piano": lambda: [Hall("room", mix=0.25)],
        "lofi_keys": lambda: [Tape(0.5, 0.3, 1.3, 0.3), BitCrush(12, 2, 0.4)],
        "glass_pad": lambda: [Hall("cathedral", mix=0.35)],
        "shimmer_pad": lambda: [Chorus(), Hall("cathedral", mix=0.45)],
        "music_box": lambda: [Hall("room", mix=0.3)],
        "harp": lambda: [Hall("hall", mix=0.25)],
    }
    return table[name]() if name in table else []


def preset_fx(name: str) -> list:
    """Doporučený fx řetěz presetu — vždy nové instance efektů."""
    key = str(name).lower()
    if key not in PRESETS:
        raise KeyError(f"neznámý preset {name!r}; k dispozici: "
                       f"{', '.join(preset_names())}")
    return _default_fx(key)


def tag_names() -> list[str]:
    return list(_TAGGED)


def presets_by_tag(tag: str) -> list[str]:
    if tag not in _TAGGED:
        raise KeyError(f"neznámý tag {tag!r}; k dispozici: {', '.join(_TAGGED)}")
    return list(_TAGGED[tag])


def preset_names() -> list[str]:
    return sorted(PRESETS)


def preset(name: str) -> Patch:
    """Vrátí novou instanci presetu, takže úpravy neovlivní ostatní stopy."""
    key = str(name).lower()
    if key not in PRESETS:
        raise KeyError(f"neznámý preset {name!r}; k dispozici: "
                       f"{', '.join(preset_names())}")
    return PRESETS[key]()
