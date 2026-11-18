# SPDX-License-Identifier: Apache-2.0
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import support
from yasakani import _native


class NativeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.default = self.root / "build" / "bin" / ("yskn.dll" if sys.platform == "win32" else "libyskn.so")
        self.default.parent.mkdir(parents=True)
        self.default.touch()
        patcher = mock.patch.object(_native, "__file__", str(self.root / "python" / "yasakani" / "_native.py"))
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ, {"YASAKANI_DLL": ""})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_explicit_path_takes_precedence(self):
        explicit = self.root / "explicit"
        explicit.touch()
        with mock.patch.dict(os.environ, {"YASAKANI_DLL": str(self.default)}):
            self.assertEqual(_native.find_library(str(explicit)), explicit.resolve())
            with self.assertRaises(_native.YasakaniError):
                _native.find_library(str(self.root / "missing"))

    def test_environment_path_takes_precedence(self):
        env = self.root / "environment"
        env.touch()
        with mock.patch.dict(os.environ, {"YASAKANI_DLL": str(env)}):
            self.assertEqual(_native.find_library(), env.resolve())
            env.unlink()
            with self.assertRaises(_native.YasakaniError):
                _native.find_library()

    def test_default_discovery(self):
        self.assertEqual(_native.find_library(), self.default.resolve())

    def test_load_library_from_current_directory(self):
        stub = support.stub_path()
        old = Path.cwd()
        try:
            os.chdir(stub.parent)
            for path in (stub.name, "./" + stub.name):
                with self.subTest(path=path):
                    self.assertEqual(_native.NativeLib(path).path, stub.resolve())
        finally:
            os.chdir(old)

    def test_pci_bus_order(self):
        """0.1.5 (issue #20): CUDA numbers the GPUs as nvidia-smi does, unless the user set CUDA_DEVICE_ORDER."""
        env = {}
        _native.pci_bus_order(env)
        self.assertEqual(env, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
        for kept in ("FASTEST_FIRST", "PCI_BUS_ID", ""):
            env = {"CUDA_DEVICE_ORDER": kept}
            _native.pci_bus_order(env)
            self.assertEqual(env["CUDA_DEVICE_ORDER"], kept)

    def test_set_before_the_library_loads(self):
        stub = support.stub_path()
        real, seen = _native.C.CDLL, []

        def load(path, *a, **k):
            seen.append(os.environ.get("CUDA_DEVICE_ORDER"))
            return real(path, *a, **k)
        with mock.patch.dict(os.environ):
            os.environ.pop("CUDA_DEVICE_ORDER", None)
            with mock.patch.object(_native.C, "CDLL", side_effect=load):
                _native.NativeLib(str(stub))
        self.assertEqual(seen, ["PCI_BUS_ID"])


class HipLibraryTests(unittest.TestCase):
    """The HIP engine library (the AMD-native backend) binds through the same C ABI.  YASAKANI_HIP_DLL names it; skipped when not
    given, so the suite never needs ROCm.  Loading only dlopens the library: no GPU is touched (that happens at yskn_open)."""

    def test_hip_dll_binds(self):
        p = os.environ.get("YASAKANI_HIP_DLL")
        if not p or not Path(p).is_file():
            self.skipTest("set YASAKANI_HIP_DLL to the HIP libyskn.so to check its bindings")
        lib = _native.NativeLib(p)
        for name in _native.SIGNATURES:
            self.assertIn(name, lib.fn)
        for name in ("yskn_set_visual", "yskn_prepare_wide", "yskn_verify_nll", "yskn_verify", "yskn_commit", "yskn_logits_rows"):
            self.assertIn(name, lib.fn)          # the optional exports the HIP build has too
        # HIP-specific extras (yskn_test_hip_prefill, yskn_kprof_dump) are not bound: their absence on CUDA must not matter
