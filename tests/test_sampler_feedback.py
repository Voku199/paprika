import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import paprika  # noqa: E402
from paprika import Song  # noqa: E402
from paprika.analyze import _moving_patch, _sample_range  # noqa: E402
from paprika.cli import main  # noqa: E402
from paprika.install import InstallError  # noqa: E402
from paprika.sampler import Sampler, clear_cache  # noqa: E402
from sample_fixture import make_sfz_library  # noqa: E402


class TestFeedback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.sfz = make_sfz_library(cls.tmp.name, sr=44100)

    @classmethod
    def tearDownClass(cls):
        clear_cache()
        cls.tmp.cleanup()

    def test_sampled_is_not_static(self):
        self.assertTrue(_moving_patch(Sampler(self.sfz)))

    def test_out_of_range_warning(self):
        s = Song(bars=2)
        t = s.track(Sampler(self.sfz, name="x"))
        t.note("C4", 0, 1).note("C6", 4, 1).note("C1", 5, 1)
        warns = []
        _sample_range(s, warns)
        self.assertEqual(len(warns), 1)
        self.assertEqual(warns[0].level, "WARN")
        self.assertIn("C6", warns[0].text)
        self.assertIn("bar 1", warns[0].text)
        self.assertIn("mimo rozsah", warns[0].text)

    def test_in_range_is_quiet(self):
        s = Song(bars=1)
        s.track(Sampler(self.sfz, name="x")).note("E4", 0, 1)
        warns = []
        _sample_range(s, warns)
        self.assertEqual(warns, [])

    def test_public_api(self):
        for name in ("Sampler", "SampleError", "SamplesMissing", "instrument",
                     "instrument_names", "samples_root", "INSTRUMENTS"):
            self.assertIn(name, paprika.__all__)
            self.assertTrue(hasattr(paprika, name))


class TestCli(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_samples_status(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {"PAPRIKA_SAMPLES": d}):
            code, out, _ = self.run_cli("samples")
        self.assertEqual(code, 0)
        self.assertIn("strings", out)
        self.assertIn("[chybí]", out)
        self.assertIn("samples install", out)

    def test_samples_install_calls_installer(self):
        with mock.patch("paprika.install.install", return_value=["marimba"]) as m:
            code, out, _ = self.run_cli("samples", "install", "mallets")
        self.assertEqual(code, 0)
        m.assert_called_once_with("mallets")
        self.assertIn("marimba", out)

    def test_samples_install_error(self):
        with mock.patch("paprika.install.install",
                        side_effect=InstallError("stažení selhalo")):
            code, _, err = self.run_cli("samples", "install", "piano")
        self.assertEqual(code, 2)
        self.assertIn("stažení selhalo", err)

    def test_presets_lists_sampled(self):
        code, out, _ = self.run_cli("presets")
        self.assertEqual(code, 0)
        self.assertIn("sampled:", out)
        self.assertIn("grand_piano", out)


if __name__ == "__main__":
    unittest.main()
