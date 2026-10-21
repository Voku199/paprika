import os
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from paprika import Song, install, library, voices  # noqa: E402
from paprika.render import render_song  # noqa: E402
from paprika.sampler import clear_cache  # noqa: E402

SR = 44100


MALE_A = ((700, 110), (1150, 120), (2600, 200))
FEMALE_A = ((800, 400), (1400, 400), (2900, 500))    # širší: F5 má jen pár harmonických


def vowel(midi, seconds=2.0, vib=0.0, first=1, amp=0.5, formants=MALE_A):
    """Syntetická samohláska „a“: harmonické přes tři formanty."""
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(SR * seconds)) / SR
    ph = np.cumsum(f0 * 2 ** (vib * np.sin(2 * np.pi * 5.5 * t) / 12)) / SR
    x = sum(np.sin(2 * np.pi * k * ph) / k for k in range(first, 40))
    for fc, bw in formants:
        r = np.exp(-np.pi * bw / SR)
        th = 2 * np.pi * fc / SR
        x = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)
    return x / np.abs(x).max() * amp


def formant_peak(x):
    n = 32768
    mag = np.abs(np.fft.rfft(x[:n] * np.hanning(n)))
    c = np.fft.irfft(np.log(mag + 1e-9))
    c[40:-40] = 0
    env = np.fft.rfft(c).real
    fr = np.fft.rfftfreq(n, 1 / SR)
    band = (fr > 300) & (fr < 1500)
    return fr[band][np.argmax(env[band])]


def median_pitch(x):
    p, _, _ = voices.track_pitch(x, SR)
    return float(np.nanmedian(p))


class TestPitch(unittest.TestCase):
    def test_track_pitch_exact(self):
        for midi in (43, 48, 57):
            self.assertAlmostEqual(median_pitch(vowel(midi)), midi, delta=0.05)

    def test_find_notes_splits_on_silence_and_measures_vibrato(self):
        gap = np.zeros(int(0.4 * SR))
        x = np.concatenate([vowel(48, 1.5, vib=0.5), gap, vowel(60, 1.5), gap, vowel(65, 1.5)])
        p, _, hop = voices.track_pitch(x, SR)
        notes = voices.find_notes(p, hop)
        self.assertEqual([round(n[2]) for n in notes], [48, 60, 65])
        self.assertGreater(notes[0][3], notes[1][3])    # vibrato = větší rozptyl

    def test_spectral_octave_fixes_both_directions(self):
        x = vowel(48, first=2)       # slabý (tady chybějící) základní tón
        self.assertEqual(voices.spectral_octave(x, SR, 60.0), 48.0)
        self.assertEqual(voices.spectral_octave(x, SR, 48.0), 48.0)
        y = vowel(52)
        self.assertEqual(voices.spectral_octave(y, SR, 40.0), 52.0)

    def test_fix_long_tones_drops_strays(self):
        self.assertEqual(voices.fix_long_tones([48.2, 59.9, 77.0], "male3"), [48.2, 59.9, None])
        self.assertEqual(voices.fix_long_tones([60.0, 72.1], "male9"), [60.0, 72.1])


class TestPsola(unittest.TestCase):
    def test_shift_keeps_length_and_formants(self):
        x = vowel(48, vib=0.4)
        p, _, hop = voices.track_pitch(x, SR)
        for step in (-2, -6):
            y = voices.psola(x, SR, p, hop, step)
            self.assertEqual(y.size, x.size)
            self.assertAlmostEqual(median_pitch(y) - median_pitch(x), step, delta=0.1)
            # resampling by posunul formant o celý krok; PSOLA ho nechá do 15 %
            self.assertLess(abs(formant_peak(y) / formant_peak(x) - 1), 0.15)
            self.assertLess(voices.subharmonic_db(y, SR, 48 + step), -30)


def fake_vocalset(full):
    """Pět zpěváků, dlouhé tóny ve formátu VocalSet (3 tóny za sebou)."""
    gap = np.zeros(int(0.3 * SR))
    for singer, tones in (("male1", (48, 60, 65)), ("male8", (48, 60, 65)),
                          ("male9", (60, 65, 72)), ("female2", (60, 72, 77)),
                          ("female7", (60, 72, 77))):
        for take, amp in (("straight", 0.5), ("forte", 0.6), ("pp", 0.15)):
            for v in "aoue":
                f = FEMALE_A if singer.startswith("female") else MALE_A
                x = np.concatenate([gap] + [np.concatenate([vowel(t, 1.6, amp=amp, formants=f), gap])
                                            for t in tones])
                path = os.path.join(full, singer, "long_tones", take,
                                    f"m_long_{take}_{v}.wav")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                wavfile.write(path, SR, (x * 32767).astype(np.int16))


class TestBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        fake_vocalset(os.path.join(cls.root, "vocalset", "FULL"))
        cls.paths = voices.build(cls.root, log=lambda *a: None)
        cls.notes = voices.read_catalog(os.path.join(cls.root, "vocalset", "notes"))

    @classmethod
    def tearDownClass(cls):
        clear_cache()
        cls.tmp.cleanup()

    def test_catalog_has_sung_and_deepened_notes(self):
        sung = {round(v.pitch) for v in self.notes if v.singer == "male8" and not v.psola}
        self.assertEqual(sung, {48, 60, 65})
        deep = {round(v.pitch) for v in self.notes if v.singer == "male8" and v.psola}
        self.assertEqual(deep, {46, 44, 42})
        self.assertEqual(len(self.paths), len(voices.VOICES) * len(voices.VOWELS))

    def test_female_notes_reach_above_and_below(self):
        sung = {round(v.pitch) for v in self.notes if v.singer == "female2" and not v.psola}
        self.assertEqual(sung, {60, 72, 77})
        up = {round(v.pitch) for v in self.notes if v.singer == "female2" and v.psola > 0}
        self.assertEqual(up, {79, 81})
        down = {round(v.pitch) for v in self.notes if v.singer == "female2" and v.psola < 0}
        self.assertEqual(down, {58, 56, 54})

    def test_missing_singers_are_added_to_old_catalog(self):
        notes_dir = os.path.join(self.root, "vocalset", "notes")
        old = [v for v in self.notes if not v.singer.startswith("female")]
        voices._write_catalog(notes_dir, old)
        clear_cache()                     # Windows: namapované WAVy nejdou přepsat
        try:
            voices.build(self.root, log=lambda *a: None)
            again = voices.read_catalog(notes_dir)
            self.assertEqual(again[:len(old)], old)
            self.assertEqual({v.singer for v in again[len(old):]}, {"female2", "female7"})
        finally:
            voices._write_catalog(notes_dir, self.notes)

    def test_choir_stacks_singers_solo_does_not(self):
        choir = voices.plan_voice("male_choir", "a", self.notes)
        at60 = [r for r in choir if r["pitch_keycenter"] == 60 and r["hivel"] == 127]
        self.assertEqual(len(at60), 2)                    # male1 + male8
        self.assertNotEqual(at60[0]["pan"], at60[1]["pan"])
        solo = voices.plan_voice("basso", "a", self.notes)
        self.assertTrue(all("seq_length" not in r or r["seq_length"] <= 2 for r in solo))
        self.assertEqual({r["sample"].split("/")[0] for r in solo}, {"male8"})

    def test_instruments_play_in_tune(self):
        for name, pitches in (("basso.aah", (42, 48, 55, 65)), ("tenor.ooh", (58, 67)),
                              ("male_choir.oh", (45, 52, 62)), ("soprano.aah", (64, 76, 80)),
                              ("alto.ooh", (56, 67)), ("female_choir.eh", (62, 74))):
            self.assertTrue(library.is_installed(name, self.root))
            for p in pitches:
                s = Song(bpm=60, bars=1)
                t = s.track(library.instrument(name, self.root), chromatic=True)
                t.note(p, 0, 2.0, 100)
                audio = render_song(s, report=False, print_report=False).audio.mean(axis=1)
                got = median_pitch(audio[int(0.4 * SR):int(1.8 * SR)])
                self.assertAlmostEqual(got, p, delta=0.3, msg=f"{name} {p}")


class TestInstall(unittest.TestCase):
    def test_extract_vocalset_only_wanted_and_safe(self):
        with tempfile.TemporaryDirectory() as d:
            z = os.path.join(d, "v.zip")
            with zipfile.ZipFile(z, "w") as f:
                f.writestr("FULL/male8/long_tones/forte/a.wav", b"x")
                f.writestr("FULL/male8/excerpts/vibrato/b.wav", b"x")
                f.writestr("FULL/female1/long_tones/forte/c.wav", b"x")
                f.writestr("__MACOSX/FULL/._a.wav", b"x")
            n = install.extract_vocalset(z, os.path.join(d, "out"), ["male8"],
                                         ["long_tones/forte"])
            self.assertEqual(n, 1)
            self.assertTrue(os.path.isfile(os.path.join(d, "out", "FULL", "male8",
                                                        "long_tones", "forte", "a.wav")))
            bad = os.path.join(d, "bad.zip")
            with zipfile.ZipFile(bad, "w") as f:
                f.writestr("FULL/male8/../../../evil.wav", b"x")
            with self.assertRaises(install.InstallError):
                install.extract_vocalset(bad, os.path.join(d, "out2"), ["male8"],
                                         ["../../.."])

    def test_catalog_lists_voices(self):
        self.assertIn("voices", library.GROUPS)
        for name in ("male_choir", "basso", "tenor", "female_choir", "soprano", "alto"):
            inst = library.INSTRUMENTS[name]
            self.assertEqual(inst.library, "vocalset")
            self.assertTrue(inst.loops("aah"))
            self.assertEqual(inst.sfz("ooh"), f"vocalset/sfz/{name}.ooh.sfz")
        self.assertTrue(library.INSTRUMENTS["basso"].mono)
        self.assertFalse(library.INSTRUMENTS["male_choir"].mono)
        self.assertTrue(library.INSTRUMENTS["soprano"].mono)
        self.assertFalse(library.INSTRUMENTS["female_choir"].mono)


if __name__ == "__main__":
    unittest.main()
