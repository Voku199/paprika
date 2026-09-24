import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from paprika import library  # noqa: E402
from paprika.install import parse_sample_name, plan_regions, write_sfz  # noqa: E402
from paprika.presets import PRESETS  # noqa: E402
from paprika.sampler import clear_cache  # noqa: E402
from sample_fixture import make_vsco_violins  # noqa: E402


class TestNames(unittest.TestCase):
    def test_parse_real_vsco_names(self):
        cases = {
            "VlnEns_susVib_A2_v1.wav": (57, 1, 1),
            "LLVln_Pizz_A3_f_RR1.wav": (69, 6, 1),
            "Sum_SHTrumpet_stac_A#3_v1_rr2.wav": (70, 1, 2),
            "Tuba3_sus_A#0_v1_rr1_Mid.wav": (34, 1, 1),
            "PSBassoon_A#0_v1_1.wav": (34, 1, 1),
            "Xylo_Medium_C4_ff_01_far.wav": (72, 7, 1),
            "glock_medium_C5.wav": (84, 0, 1),
            "UR1_C1_pp_RR2.wav": (36, 2, 2),
            "Marimba_hit_Outrigger_B2_loud_01.wav": (59, 7, 1),
        }
        for name, want in cases.items():
            self.assertEqual(parse_sample_name(name, 12), want, name)
        self.assertEqual(parse_sample_name("LLVln_Pizz_A3_f_RR1.wav")[0], 57)
        self.assertIsNone(parse_sample_name("Rode_Blower.wav"))

    def test_plan_regions_splits(self):
        entries = [("a.wav", 60, 1, 1), ("b.wav", 60, 1, 2), ("c.wav", 64, 1, 1),
                   ("d.wav", 60, 2, 1), ("e.wav", 67, 2, 1)]
        rs = plan_regions(entries)
        low = [r for r in rs if r["lovel"] == 1]
        high = [r for r in rs if r["lovel"] == 64]
        self.assertEqual({r["hivel"] for r in low}, {63})
        self.assertEqual({r["hivel"] for r in high}, {127})
        for layer in (low, high):
            spans = sorted({(r["lokey"], r["hikey"]) for r in layer})
            for (_, a_hi), (b_lo, _) in zip(spans, spans[1:]):
                self.assertEqual(b_lo, a_hi + 1)
            self.assertEqual(spans[0][0], 58)
        rr = [r for r in low if r["pitch_keycenter"] == 60]
        self.assertEqual(sorted((r["seq_position"], r["seq_length"]) for r in rr),
                         [(1, 2), (2, 2)])


class TestLibrary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = self.tmp.name
        self.env = mock.patch.dict(os.environ, {"PAPRIKA_SAMPLES": self.root})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        clear_cache()
        self.tmp.cleanup()

    def test_root_from_env(self):
        self.assertEqual(library.samples_root(), os.path.abspath(self.root))

    def test_missing_library_message(self):
        with self.assertRaises(library.SamplesMissing) as cm:
            library.instrument("violins.pizz")
        msg = str(cm.exception)
        self.assertIn("python -m paprika samples install strings", msg)
        self.assertIn(os.path.abspath(self.root), msg)
        self.assertFalse(library.is_installed("violins"))

    def test_unknown_names(self):
        with self.assertRaises(KeyError) as cm:
            library.instrument("violins.legato")
        self.assertIn("pizz", str(cm.exception))
        with self.assertRaises(KeyError):
            library.instrument("kazoo")
        self.assertTrue(library.is_sampled("Violins.PIZZ"))
        self.assertFalse(library.is_sampled("epiano"))

    def test_catalog(self):
        self.assertIn("grand_piano", library.instrument_names())
        self.assertFalse(set(PRESETS) & set(library.INSTRUMENTS))
        for inst in library.INSTRUMENTS.values():
            self.assertIn(inst.group, library.GROUPS)
            self.assertTrue(inst.arts)
            self.assertTrue(library.instrument_fx(inst.name))
        self.assertTrue(library.instrument_defaults("violin_solo")["mono"])
        self.assertFalse(library.instrument_defaults("violins")["mono"])

    def test_generated_sfz_plays(self):
        make_vsco_violins(self.root)
        inst = library.INSTRUMENTS["violins"]
        for art in inst.arts:
            write_sfz(inst, art, self.root)
        s = library.instrument("violins")
        self.assertTrue(s.legato)
        self.assertEqual(s.range, (55, 66))       # kořeny 57, 60, 64 ± 2
        self.assertTrue(all(r.loop_mode == "loop_continuous"
                            and r.loop_end > r.loop_start for r in s.regions))
        pz = library.instrument("violins.pizz")
        self.assertFalse(pz.legato)
        self.assertEqual(pz.name, "violins.pizz")
        self.assertTrue(all(r.loop_mode == "one_shot" for r in pz.regions))
        r = s.select(60, 127)[0]
        v = s.voice(r, 60, 0.4, 44100) * s.gain(r, 127)
        rms = np.sqrt(np.mean(v[4410:13230].mean(axis=1) ** 2))
        self.assertAlmostEqual(20 * np.log10(rms), -18.0, delta=1.0)
        text = "\n".join(library.status_lines())
        self.assertIn("strings", text)
        self.assertIn("[chybí]", text)            # piano pořád chybí


if __name__ == "__main__":
    unittest.main()
