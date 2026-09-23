import io
import os
import sys
import unittest
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paprika import Song  # noqa: E402
from paprika.cli import main  # noqa: E402
from paprika.presets import (PRESET_TAGS, preset, preset_fx,  # noqa: E402
                             preset_names, presets_by_tag, tag_names)

SR = 44100
NEW = [n for n in preset_names() if "classic" not in PRESET_TAGS[n]]
LOW = {"808", "round_bass", "cello_solo", "reese", "timpani_tone"}


def note_freq(name):
    return 55.0 if name in LOW else 261.6


def logspec(y, n=16384):
    m = y.mean(axis=1) if y.ndim == 2 else y
    m = np.pad(m[:n], (0, max(0, n - m.size)))
    return np.log10(np.abs(np.fft.rfft(m * np.hanning(n))) + 1e-5)


class TestPresetCatalog(unittest.TestCase):
    def test_count_and_tags(self):
        self.assertGreaterEqual(len(NEW), 26)
        for n in preset_names():
            self.assertTrue(PRESET_TAGS.get(n), n)
        self.assertIn("808", presets_by_tag("darkpop"))
        self.assertIn("choir", tag_names())
        self.assertIn("epiano", presets_by_tag("classic"))

    def test_unknown_tag(self):
        with self.assertRaises(KeyError) as cm:
            presets_by_tag("polka")
        self.assertIn("orchestral", str(cm.exception))

    def test_all_new_bounded_and_audible(self):
        for n in NEW:
            y = preset(n).voice(note_freq(n), 0.8, 100, SR)
            self.assertTrue(np.all(np.isfinite(y)), n)
            self.assertLess(np.abs(y).max(), 1.0, n)
            self.assertGreater(np.abs(y).max(), 0.03, n)

    def test_presets_are_distinct(self):
        specs = {n: logspec(preset(n).voice(261.6, 0.8, 100, SR)) for n in NEW}
        for i, a in enumerate(NEW):
            for b in NEW[i + 1:]:
                self.assertGreater(np.abs(specs[a] - specs[b]).mean(), 0.08, (a, b))

    def test_new_presets_are_not_static(self):
        """Nové presety mají aspoň jednu modulaci — proti „plastovému“ zvuku."""
        for n in NEW:
            p = preset(n)
            moving = (p.vibrato or p.tremolo or p.pitch_env or p.drift or p.rr > 1
                      or p.vel_to_cutoff
                      or any(getattr(f, "lfo", None) for f in p.filters)
                      or any(getattr(o, "pwm", None) for o in p.osc)
                      or any(type(o).__name__ in ("Pluck", "Additive", "SuperSaw")
                             for o in p.osc))
            self.assertTrue(moving, n)


class TestPresetDefaults(unittest.TestCase):
    def test_808_defaults_and_fx(self):
        s = Song(bars=1)
        t = s.track("808", fx="default")
        self.assertTrue(t.mono)
        self.assertGreater(t.glide, 0)
        self.assertTrue(t.fx_chain)

    def test_explicit_overrides_default(self):
        self.assertFalse(Song(bars=1).track("808", mono=False).mono)

    def test_preset_fx_fresh_instances(self):
        self.assertIsNot(preset_fx("808")[0], preset_fx("808")[0])

    def test_default_fx_only_when_asked(self):
        self.assertEqual(Song(bars=1).track("808").fx_chain, [])
        self.assertEqual(Song(bars=1).track("epiano", fx="default").fx_chain, [])

    def test_fx_default_needs_preset_name(self):
        from paprika import Patch
        with self.assertRaises(ValueError):
            Song(bars=1).track(Patch(), fx="default")


class TestCli(unittest.TestCase):
    def run_cli(self, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(list(args))
        return code, buf.getvalue()

    def test_presets_by_tag(self):
        code, out = self.run_cli("presets", "--tag", "choir")
        self.assertEqual(code, 0)
        self.assertIn("choir_ooh", out)
        self.assertNotIn("808", out)

    def test_presets_lists_kits(self):
        code, out = self.run_cli("presets")
        self.assertIn("cinematic", out)
        self.assertIn("riser", out)


if __name__ == "__main__":
    unittest.main()
