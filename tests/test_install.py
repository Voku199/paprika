import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from paprika import install, library  # noqa: E402
from paprika.sampler import clear_cache  # noqa: E402


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def tar_with(path, name, data):
    with tarfile.open(path, "w:bz2") as t:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))


class TestDownload(unittest.TestCase):
    def test_download_and_skip(self):
        with tempfile.TemporaryDirectory() as d:
            dst = os.path.join(d, "x", "a.wav")
            with mock.patch.object(install, "_urlopen",
                                   return_value=FakeResp(b"abc")) as m:
                self.assertTrue(install.download("http://x/a.wav", dst, 3))
                self.assertFalse(install.download("http://x/a.wav", dst, 3))
                self.assertEqual(m.call_count, 1)
            with open(dst, "rb") as f:
                self.assertEqual(f.read(), b"abc")
            self.assertFalse(os.path.exists(dst + ".part"))

    def test_size_mismatch_and_network_error(self):
        with tempfile.TemporaryDirectory() as d:
            dst = os.path.join(d, "a.wav")
            with mock.patch.object(install, "_urlopen", return_value=FakeResp(b"ab")):
                with self.assertRaises(install.InstallError):
                    install.download("http://x/a", dst, 3)
            self.assertFalse(os.path.exists(dst))
            self.assertFalse(os.path.exists(dst + ".part"))
            with mock.patch.object(install, "_urlopen", side_effect=OSError("offline")):
                with self.assertRaises(install.InstallError) as cm:
                    install.download("http://x/a", dst, 3)
            self.assertIn("spusť instalaci znovu", str(cm.exception))

    def test_extract_and_reject_escape(self):
        with tempfile.TemporaryDirectory() as d:
            good = os.path.join(d, "good.tar.bz2")
            tar_with(good, "pkg/x.sfz", b"<region> sample=a.wav\n")
            install.extract(good, os.path.join(d, "out"))
            self.assertTrue(os.path.isfile(os.path.join(d, "out", "pkg", "x.sfz")))
            bad = os.path.join(d, "bad.tar.bz2")
            tar_with(bad, "../evil.txt", b"x")
            with self.assertRaises(install.InstallError):
                install.extract(bad, os.path.join(d, "out2"))
            self.assertFalse(os.path.exists(os.path.join(d, "evil.txt")))

    def test_vsco_files_filters_folders(self):
        tree = {"tree": [
            {"type": "blob", "path": "Strings/Harp/KSHarp_A2_mf.wav", "size": 10},
            {"type": "blob", "path": "Strings/Harp/sub/x.wav", "size": 10},
            {"type": "blob", "path": "Strings/Harp/readme.txt", "size": 1},
            {"type": "tree", "path": "Strings/Harp"}]}
        with mock.patch.object(install, "_urlopen",
                               return_value=FakeResp(json.dumps(tree).encode())):
            self.assertEqual(install.vsco_files(["Strings/Harp"]),
                             [("Strings/Harp/KSHarp_A2_mf.wav", 10)])

    def test_unknown_group(self):
        with self.assertRaises(install.InstallError) as cm:
            install.install("kazoo", root=".", log=lambda *a: None)
        self.assertIn("mallets", str(cm.exception))


class TestInstallOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = self.tmp.name

    def tearDown(self):
        clear_cache()
        self.tmp.cleanup()

    def test_mallets_end_to_end(self):
        buf = io.BytesIO()
        t = np.arange(22050) / 44100
        wavfile.write(buf, 44100, (8000 * np.sin(2 * np.pi * 523 * t)).astype(np.int16))
        payload = buf.getvalue()
        files = [{"type": "blob", "path": f"{folder}/{name}", "size": len(payload)}
                 for folder, name in (
                     ("Percussion/Marimba", "Marimba_hit_Outrigger_C4_loud_01.wav"),
                     ("Percussion/Glock", "glock_medium_C5.wav"),
                     ("Percussion/Xylo", "Xylo_Medium_C5_ff_01_far.wav"))]

        def fake(url, timeout=60.0):
            if "api.github.com" in url:
                return FakeResp(json.dumps({"tree": files}).encode())
            return FakeResp(payload)

        with mock.patch.object(install, "_urlopen", side_effect=fake):
            names = install.install("mallets", root=self.root, log=lambda *a: None)
        self.assertEqual(sorted(names), ["glockenspiel", "marimba", "xylophone"])
        self.assertTrue(library.is_installed("marimba", self.root))
        with open(os.path.join(self.root, install.MANIFEST), encoding="utf-8") as f:
            self.assertIn("mallets", json.load(f)["groups"])
        with open(os.path.join(self.root, "CREDITS.txt"), encoding="utf-8") as f:
            self.assertIn("Alexander Holm", f.read())

    def test_salamander_marker(self):
        """Rozbalení bez značky .complete se zopakuje, se značkou přeskočí."""
        archive = os.path.join(self.root, "_downloads",
                               os.path.basename(install.SALAMANDER_URL))
        os.makedirs(os.path.dirname(archive))
        inner = library.SALAMANDER_SFZ.split("/", 1)[1]
        tar_with(archive, inner, b"<region> sample=a.wav key=60\n")
        calls = []
        real_extract = install.extract

        def spy(a, dest):
            calls.append(a)
            real_extract(a, dest)

        with mock.patch.object(install, "SALAMANDER_SIZE", os.path.getsize(archive)), \
                mock.patch.object(install, "extract", side_effect=spy):
            install._install_salamander(self.root, lambda *a: None)
            install._install_salamander(self.root, lambda *a: None)
        self.assertEqual(len(calls), 1)
        self.assertTrue(os.path.isfile(os.path.join(self.root, library.SALAMANDER_SFZ)))


if __name__ == "__main__":
    unittest.main()
