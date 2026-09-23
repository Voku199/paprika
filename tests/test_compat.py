import glob
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paprika.cli import load_song  # noqa: E402
from paprika.render import render_song  # noqa: E402


@unittest.skipUnless(os.environ.get("PAPRIKA_SLOW"),
                     "pomalý smoke test všech songů — spusť s PAPRIKA_SLOW=1")
class TestAllSongsRender(unittest.TestCase):
    """Každá skladba v songs/ se načte a první dva takty vyrenderují."""

    def test_every_song_renders_first_bars(self):
        paths = sorted(glob.glob(os.path.join(ROOT, "songs", "*.py")))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(song=os.path.basename(path)):
                song, _ = load_song(path)
                out = render_song(song, None, report=True, bars=(0, 2),
                                  print_report=False)
                self.assertTrue(np.all(np.isfinite(out.audio)))


class TestPublicApi(unittest.TestCase):
    def test_v2_exports(self):
        import paprika
        names = ("SuperSaw", "Pulse", "Pluck", "Additive", "BP", "Formant", "LFO",
                 "PitchEnv", "Filter", "EQ", "Compressor", "Gate", "Hall", "Chorus",
                 "Width", "BitCrush", "Tape", "Bus", "SfxTrack", "Automation",
                 "presets_by_tag", "tag_names", "preset_fx", "kit_names", "dsp")
        for name in names:
            self.assertTrue(hasattr(paprika, name), name)
            if name != "dsp":
                self.assertIn(name, paprika.__all__, name)

    def test_version(self):
        import paprika
        self.assertTrue(paprika.__version__.startswith("2."))


if __name__ == "__main__":
    unittest.main()
