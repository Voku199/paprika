"""Kroky z vrstev: dopad, tření, úlomky, sání, výbava. Fyzika, žádné nahrávky.

Herní enginy skládají krok z vrstev (foot × shoe × surface × gear) a každá
vrstva má vlastní variaci. Tady je každá vrstva fyzikální model řízený
jedním průběhem — reakční silou podložky F(t) (GRF) během kontaktu
chodidla:

- **Dopad.** Bota narazí na podložku na dobu `τ` (Hertzův kontakt: tvrdá
  podrážka na kameni ~0.5 ms, bosá noha na trávě ~20 ms). Silový pulz
  půlsinus o jednotkové ploše má spektrum `cos(π f τ) / (1 − (2 f τ)²)`,
  takže měkký kontakt utlumí vysoké rezonance sám od sebe. Pulz budí
  rezonance podložky (`modes`), dupnutí nohy na pružné podložce (`body`)
  a krátký klik (`click`). Pata dopadne v 0, špička po `gap`.
  Rezonance mají při každém kroku jinou amplitudu a o pár procent jinou
  frekvenci — chodidlo dopadne pokaždé jinam.
- **Tření.** Smyk paty při dopadu a odval špičky při odrazu: filtrovaný
  šum, silnější s rychlostí (odval ~v^1.5).
- **Úlomky.** Zrna (PhISEM, Cook) s okamžitou četností
  `rate · max(0, dF/dt) + sustain · F`: štěrk, sníh a tráva praskají,
  když síla roste (dopad, odraz), sníh drtí i pod stálou vahou. Po
  odlepení dosedají kamínky ještě `tail` sekund.
- **Bláto** (`wet`): mlasknutí při dopadu, sání a bublinky po odlepení.
- **Sníh** (`squeak`): úzkopásmové vrzání stlačovaných krystalů.
- **Výbava** (`GEARS`): klíče, řetěz, zbroj — kousky kovu do sebe narazí
  s setrvačným zpožděním po dopadu a slaběji při odrazu; je na trupu, ve
  středu sterea.

Oblečení není ve zvuku kroku: `HumanTrack.steps(cloth=)` přidá vlastní
hity `cloth` ve fázi švihu nohy. Časování, síla a korelace kroků přicházejí
z `gait.plan`; tady se z nich jen dělá zvuk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import dsp, gait

__all__ = ["Surface", "SURFACES", "SHOES", "GEARS", "SURFACE_GAIN",
           "footstep", "grf", "pulse_gain", "contact_ms", "blend_name"]

TWO_PI = 2.0 * math.pi
CEILING = 0.95
CLICK_EDGE = 0.2      # kontakt hrany podrážky (klik) vůči celé ploše


@dataclass(frozen=True)
class Surface:
    """Podložka. Frekvence v Hz, časy v s, amplitudy relativní."""
    hardness: float            # 0..1: 1 = kámen, kov; 0 = bláto (delší kontakt)
    modes: tuple = ()          # ((Hz, τ, amp), ...) rezonance podložky
    body: tuple = (80.0, 0.03, 0.8)          # dupnutí: hmota nohy na podložce
    click: tuple = (2500.0, 0.5)             # (Hz, amp) krátký klik kontaktu
    friction: tuple = (3000.0, 1.0, 0.0)     # (Hz, Q, amp) tření podrážky
    debris: tuple | None = None
    # debris = (rate /s při max dF/dt, sustain /s při F = 1, (Hz, Q) hlavní
    # pásmo, (Hz, Q) tělo, amp, tail s)
    wet: float = 0.0           # 0..1 bláto: mlasknutí, sání, bublinky
    squeak: float = 0.0        # vrzání sněhu


SURFACES = {
    "stone": Surface(1.0, modes=((1800.0, 0.012, 0.25), (3400.0, 0.008, 0.35),
                                 (5200.0, 0.005, 0.25)),
                     body=(90.0, 0.025, 0.5), click=(3200.0, 1.2),
                     friction=(4500.0, 1.0, 0.25),
                     debris=(250.0, 0.0, (6000.0, 1.0), (2500.0, 0.8), 0.12, 0.0)),
    "wood": Surface(0.85, modes=((180.0, 0.06, 0.5), (420.0, 0.05, 0.6),
                                 (950.0, 0.03, 0.35), (2100.0, 0.015, 0.2)),
                    body=(110.0, 0.04, 0.6), click=(2000.0, 0.9),
                    friction=(3000.0, 1.2, 0.2)),
    "metal": Surface(1.0, modes=((380.0, 0.35, 0.35), (910.0, 0.28, 0.5),
                                 (1720.0, 0.2, 0.45), (2950.0, 0.14, 0.35),
                                 (4600.0, 0.09, 0.25)),
                     body=(100.0, 0.02, 0.4), click=(5000.0, 0.8),
                     friction=(5500.0, 1.0, 0.25),
                     debris=(200.0, 0.0, (7000.0, 1.0), (3000.0, 0.8), 0.08, 0.0)),
    "gravel": Surface(0.6, body=(80.0, 0.03, 0.6), click=(2500.0, 0.4),
                      friction=(2500.0, 0.8, 0.1),
                      debris=(2500.0, 600.0, (3200.0, 1.2), (1400.0, 0.9), 0.6, 0.12)),
    "dirt": Surface(0.4, body=(70.0, 0.04, 0.6), click=(1500.0, 0.3),
                    friction=(1800.0, 0.8, 0.15),
                    debris=(1200.0, 250.0, (1400.0, 0.9), (600.0, 0.8), 0.6, 0.04)),
    "grass": Surface(0.15, body=(75.0, 0.045, 0.7), click=(3000.0, 0.05),
                     friction=(5000.0, 0.7, 0.05),
                     debris=(4000.0, 1500.0, (6500.0, 0.9), (3000.0, 0.7), 0.35, 0.06)),
    "snow": Surface(0.2, body=(85.0, 0.035, 0.5), click=(2000.0, 0.1),
                    debris=(6000.0, 3000.0, (2200.0, 0.9), (1000.0, 1.0), 0.8, 0.03),
                    squeak=0.4),
    "mud": Surface(0.05, body=(60.0, 0.06, 0.8), click=(800.0, 0.05),
                   debris=(300.0, 80.0, (700.0, 1.0), (300.0, 0.8), 0.3, 0.05),
                   wet=1.0),
}

# bota: (pulz ms na tvrdé podložce, síla paty, odval, smyk paty)
SHOES = {
    "boot": (0.5, 1.0, 1.0, 1.0),
    "sneaker": (1.2, 0.7, 0.6, 0.5),
    "heels": (0.2, 1.2, 0.4, 0.3),
    "bare": (2.5, 0.6, 0.2, 0.2),
}

# výbava: (frekvence kousků Hz, rozsah τ s, srážek na dopad, amp)
GEARS = {
    "keys": ((2900.0, 3700.0, 4400.0, 5300.0, 6100.0, 7400.0, 8800.0),
             (0.03, 0.1), 3.0, 0.07),
    "chain": ((3500.0, 4200.0, 4900.0, 5800.0, 6600.0), (0.012, 0.03), 6.0, 0.06),
    "armor": ((650.0, 1100.0, 1700.0, 2400.0, 3300.0), (0.12, 0.3), 2.0, 0.08),
}

# pevný zisk podle povrchu: referenční chodec (75 kg, 1.4 m/s, boot, bez
# výbavy) má medián špičky podle povrchu — tráva je tišší než kámen, ale
# všechno pod CEILING i při běhu. Kalibrace: docs/superpowers/research/
# 2026-10-24-foley-kroky-pruzkum.md, sekce 7.
SURFACE_TARGET = {"stone": 0.45, "wood": 0.45, "metal": 0.5, "gravel": 0.45,
                  "dirt": 0.35, "grass": 0.3, "snow": 0.4, "mud": 0.4}
SURFACE_GAIN = {"stone": 0.6003, "wood": 0.3491, "metal": 0.388, "gravel": 0.4342,
                "dirt": 0.5001, "grass": 0.2955, "snow": 0.2359, "mud": 0.4493}


def blend_name(surface) -> tuple[str, str, float]:
    """Povrch hitu: jméno, nebo `(a, b, w)` = přechod z `a` do `b` s vahou `w`."""
    if isinstance(surface, str):
        return surface, surface, 0.0
    a, b, w = surface
    return str(a), str(b), float(w)


def contact_ms(shoe: str, surface: Surface, speed: float) -> float:
    """Doba kontaktu boty s podložkou v ms (Hertz: rychlejší dopad je kratší)."""
    base = SHOES[shoe][0]
    return base * (1.0 + 8.0 * (1.0 - surface.hardness)) * (float(speed) / gait.REF_SPEED) ** -0.2


def pulse_gain(freq, tau: float) -> np.ndarray:
    """|spektrum| půlsinusového pulzu délky `tau` s jednotkovou plochou."""
    x = 2.0 * np.asarray(freq, dtype=np.float64) * tau
    den = 1.0 - x * x
    near = np.abs(den) < 1e-6
    val = np.cos(0.5 * math.pi * x) / np.where(near, 1.0, den)
    return np.abs(np.where(near, math.pi / 4.0, val))


def grf(t: np.ndarray, contact: float, run: bool) -> np.ndarray:
    """Reakční síla podložky (maximum 1) během kontaktu `contact` s.

    Chůze: dva vrcholy (pata ve ~25 %, odraz ve ~75 % kontaktu) s údolím
    v půlce a rázem při dopadu paty. Běh: jeden vrchol a ráz při dopadu.
    """
    u = np.clip(t / contact, 0.0, 1.0)
    inside = (t >= 0) & (t <= contact)
    win = np.where(inside, np.sin(math.pi * u), 0.0)
    if run:
        f = win ** 1.5
    else:
        f = (np.exp(-((u - 0.25) / 0.13) ** 2) + 0.95 * np.exp(-((u - 0.75) / 0.13) ** 2)
             + 0.45 * np.exp(-((u - 0.5) / 0.2) ** 2)) * np.sqrt(win)
    spike = 0.5 * np.exp(-((t - 0.012) / 0.006) ** 2) * (t >= 0)
    f = f + spike
    peak = float(f.max())
    return f / peak if peak > 0 else f


def _decay(t: np.ndarray, t0: float, tau: float) -> np.ndarray:
    d = t - t0
    return np.where(d >= 0, np.exp(-np.maximum(d, 0.0) / max(1e-4, tau)), 0.0)


def _impact(t, sr, surf: Surface, t0: float, strength: float, tau: float,
            body_k: float, bright: float, rng) -> np.ndarray:
    """Pulz délky `tau` v čase `t0`: rezonance podložky + dupnutí + klik."""
    y = np.zeros_like(t)
    d = t - t0 - 0.5 * tau
    on = d >= 0
    modes = list(surf.modes) + [(surf.body[0] * body_k, surf.body[1], surf.body[2])]
    for i, (f, mtau, amp) in enumerate(modes):
        body = i == len(modes) - 1
        fr = f * (rng.uniform(0.95, 1.05) if body else rng.uniform(0.97, 1.03))
        a = amp * (1.0 if body else rng.lognormal(0.0, 0.35))
        g = float(pulse_gain(fr, tau))
        y += np.where(on, a * g * np.sin(TWO_PI * fr * d) * np.exp(-np.maximum(d, 0.0) / mtau), 0.0)
    cf, ca = surf.click
    cf = min(cf * bright * rng.uniform(0.9, 1.1), 0.4 * sr)
    if ca > 0:
        # klik dělá hrana podrážky: dotkne se dřív a kratší dobu než plocha
        tau_c = CLICK_EDGE * tau
        burst = rng.standard_normal(t.size) * _decay(t, t0, 0.0005 + 2.0 * tau_c)
        y += ca * float(pulse_gain(cf, tau_c)) * dsp.filt(burst, "bp", cf, sr, q=1.2)
    return strength * y


def _friction(t, sr, surf: Surface, shoe: str, speed: float, contact: float,
              bright: float, rng) -> np.ndarray:
    f, q, amp = surf.friction
    if amp <= 0:
        return np.zeros_like(t)
    _, _, scuff, grip = SHOES[shoe]
    v = float(speed) / gait.REF_SPEED
    # při běhu dopadá spíš přední část chodidla: smyk paty roste pomalu,
    # odval špičky rychleji, ale s mezí (víc tření podrážka nedá)
    slide = grip * v ** 0.5 * _decay(t, 0.0, 0.012)
    c = contact - 0.02
    roll = 0.6 * scuff * min(v, 2.0) ** 1.5 * np.exp(-((t - c) / 0.025) ** 2)
    noise = rng.standard_normal(t.size) * (slide + roll)
    return amp * dsp.filt(noise, "bp", min(f * bright, 0.4 * sr), sr, q=q).astype(np.float64)


def _grains(rate: np.ndarray, sr: int, rng, amp: np.ndarray) -> np.ndarray:
    p = np.clip(rate / sr, 0.0, 0.5)
    return np.where(rng.random(rate.size) < p, amp * rng.lognormal(0.0, 0.6, rate.size), 0.0)


def _debris(t, sr, surf: Surface, force, dforce, contact: float, level: float,
            bright: float, rng) -> np.ndarray:
    if surf.debris is None:
        return np.zeros_like(t)
    rate, sustain, (f1, q1), (f2, q2), amp, tail = surf.debris
    lam = rate * dforce + sustain * force
    if tail > 0:
        lam = lam + 0.08 * rate * _decay(t, contact, tail / 3.0)
    lam = lam * min(2.0, level) ** 0.5
    hits = _grains(lam, sr, rng, np.sqrt(0.3 + force))
    y = (dsp.filt(hits, "bp", min(f1 * bright, 0.4 * sr), sr, q=q1).astype(np.float64)
         + 0.5 * dsp.filt(hits, "bp", f2, sr, q=q2).astype(np.float64))
    if surf.squeak > 0:
        load = _grains(0.6 * sustain * force * (dforce > 0.05), sr, rng, np.ones(t.size))
        y += surf.squeak * dsp.filt(load, "bp", 1300.0 * rng.uniform(0.9, 1.1), sr,
                                    q=6.0).astype(np.float64)
    return amp * y


def _mud(t, sr, surf: Surface, contact: float, speed: float, rng) -> np.ndarray:
    if surf.wet <= 0:
        return np.zeros_like(t)
    w = surf.wet
    splat = dsp.filt(rng.standard_normal(t.size) * _decay(t, 0.0, 0.03), "bp",
                     500.0 * rng.uniform(0.8, 1.2), sr, q=0.8).astype(np.float64)
    d = t - contact
    on = d >= 0
    dur = rng.uniform(0.035, 0.06)
    f0, f1 = rng.uniform(420.0, 600.0), rng.uniform(130.0, 190.0)
    freq = f1 + (f0 - f1) * np.exp(-np.maximum(d, 0.0) / (dur / 2.5))
    phase = TWO_PI * np.cumsum(np.where(on, freq, 0.0)) / sr
    suck = (min(2.0, float(speed) / gait.REF_SPEED) * np.where(on, np.sin(phase), 0.0)
            * _decay(t, contact, dur / 2.0))
    bubbles = np.zeros_like(t)
    for _ in range(int(rng.poisson(3))):
        bt = contact + rng.uniform(0.01, 0.15)
        bf = rng.uniform(400.0, 1500.0)
        bubbles += (rng.uniform(0.05, 0.15) * np.sin(TWO_PI * bf * np.maximum(t - bt, 0.0))
                    * _decay(t, bt, rng.uniform(0.008, 0.025)))
    return w * (0.6 * splat + 0.5 * suck + bubbles)


def _gear(t, sr, gear: str | None, contact: float, level: float, speed: float,
          rng) -> np.ndarray:
    if gear is None:
        return np.zeros_like(t)
    freqs, (tau_lo, tau_hi), mean, amp = GEARS[gear]
    y = np.zeros_like(t)
    shake = level * (float(speed) / gait.REF_SPEED) ** 1.2
    for t_ev, k in ((rng.uniform(0.01, 0.04), mean), (contact, 0.5 * mean)):
        for _ in range(int(rng.poisson(k))):
            tc = t_ev + rng.uniform(0.0, 0.06)
            a = amp * shake * rng.lognormal(0.0, 0.5)
            for f in rng.choice(freqs, size=2, replace=False):
                fr = min(float(f) * rng.uniform(0.95, 1.05), 0.45 * sr)
                y += (a * np.sin(TWO_PI * fr * np.maximum(t - tc, 0.0))
                      * _decay(t, tc, rng.uniform(tau_lo, tau_hi)))
    return y


def _render(sr: int, surf: Surface, foot: int, weight: float, speed: float,
            level: float, gap_k: float, ratio_k: float, bright_k: float, pitch_k: float,
            shoe: str, gear, rng) -> tuple[np.ndarray, np.ndarray]:
    """Jeden krok na jedné podložce: (noha mono, výbava mono), bez zisku."""
    ph = gait.physics(weight, speed)
    bright = ph.bright * bright_k
    gap = float(np.clip(ph.gap * gap_k, 0.005, 0.2))
    tail = surf.debris[5] if surf.debris else 0.0
    n = int((ph.contact + max(0.15, tail + 0.08) + 0.05) * sr)
    t = np.arange(n) / sr
    tau = contact_ms(shoe, surf, speed) / 1000.0
    _, heel, _, _ = SHOES[shoe]
    body_k = ph.pitch * pitch_k
    foot_y = _impact(t, sr, surf, 0.0, heel, tau, body_k, bright, rng)
    foot_y += _impact(t, sr, surf, gap, ph.ratio * ratio_k * rng.uniform(0.9, 1.1),
                      tau * 1.5, body_k, bright, rng)
    force = grf(t, ph.contact, ph.run)
    dforce = np.maximum(np.gradient(force, 1.0 / sr), 0.0)
    dforce /= max(1e-9, float(dforce.max()))
    foot_y += _friction(t, sr, surf, shoe, speed, ph.contact, bright, rng)
    foot_y += _debris(t, sr, surf, force, dforce, ph.contact, level, bright, rng)
    foot_y += _mud(t, sr, surf, ph.contact, speed, rng)
    gear_y = _gear(t, sr, gear, ph.contact, level, speed, rng)
    return foot_y * ph.level * level, gear_y


def footstep(sr, surface, foot, weight, speed, level, gap, ratio, bright, pitch,
             shoe, gear, variant) -> np.ndarray:
    """Krok z vrstev jako `(m, 2)` float64 při velocity 127.

    `surface` je jméno, nebo `(a, b, w)` — přechod z `a` do `b` (oba
    povrchy znějí zároveň, `b` s vahou `w`, výkonově). `level`, `gap`,
    `ratio`, `bright`, `pitch` jsou korelované násobky z `gait.plan`.
    """
    a_name, b_name, w = blend_name(surface)
    for name in (a_name, b_name):
        if name not in SURFACES:
            raise ValueError(f"neznámý povrch {name!r}; k dispozici: {', '.join(SURFACES)}")
    if shoe not in SHOES:
        raise ValueError(f"neznámá bota {shoe!r}; k dispozici: {', '.join(SHOES)}")
    if gear is not None and gear not in GEARS:
        raise ValueError(f"neznámá výbava {gear!r}; k dispozici: {', '.join(GEARS)}")
    if not gait.MIN_KG <= float(weight) <= gait.MAX_KG:
        raise ValueError(f"váha chodce {weight!r} kg mimo {gait.MIN_KG:.0f}–{gait.MAX_KG:.0f}")
    if not gait.MIN_SPEED <= float(speed) <= gait.MAX_SPEED:
        raise ValueError(f"rychlost {speed!r} m/s mimo {gait.MIN_SPEED}–{gait.MAX_SPEED}")
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"váha přechodu povrchů {w!r} mimo 0..1")
    args = (foot, float(weight), float(speed), float(level), float(gap), float(ratio),
            float(bright), float(pitch), shoe)
    parts = []
    for name, k in ((a_name, math.sqrt(1.0 - w)), (b_name, math.sqrt(w))):
        if k <= 0.0:
            continue
        rng = np.random.default_rng(29 * 7919 + int(variant) * 31 + len(parts))
        y, g = _render(sr, SURFACES[name], *args, gear=gear if not parts else None, rng=rng)
        parts.append((k * SURFACE_GAIN[name] * y, g))
    m = max(p[0].size for p in parts)
    foot_y = np.zeros(m)
    gear_y = np.zeros(m)
    for y, g in parts:
        foot_y[:y.size] += y
        gear_y[:g.size] += g
    left, right = (1.0, 0.85) if int(foot) % 2 == 0 else (0.85, 1.0)
    out = np.stack([foot_y * left + gear_y, foot_y * right + gear_y], axis=1)
    edge = min(int(0.004 * sr), m // 2)
    if edge > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, edge))
        out[-edge:] *= ramp[::-1, None]
    peak = float(np.abs(out).max()) if out.size else 0.0
    return out * (CEILING / peak) if peak > CEILING else out
