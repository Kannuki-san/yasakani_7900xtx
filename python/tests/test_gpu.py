# SPDX-License-Identifier: Apache-2.0
"""0.1.3: which GPU, what 0.1.3 does on it, and the context that fits its memory (yasakani/gpu.py and the quickstart's copy of the rule).

nvidia-smi is never run for real here: its output is given as text, or a stand-in program prints it.  The memory sizes of the cards are
what nvidia-smi reports for them (MiB); the rule's numbers are those of the published model files.  The AMD tests (the HIP backend)
work the same way: a fake /sys/class/drm tree and rocm-smi CSV given as text, never a real GPU.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import support
from yasakani import gpu as G
from yasakani._native import YasakaniError

ROOT = Path(__file__).resolve().parent.parent
FASTEST = (10311344128, 1164969472, 1411010584)       # the published files: weights, draft model, its output head
QUALITY = (12365242368, 1164969472, 1411010584)

# index, name, compute capability, memory.total (MiB) -> what the quickstart does, whether the quality model fits
CARDS = [
    ("NVIDIA GeForce RTX 5090", "12.0", 32607, "tested", True),
    ("NVIDIA GeForce RTX 4090", "8.9", 24564, "other", True),
    ("NVIDIA GeForce RTX 3090 Ti", "8.6", 24564, "other", True),
    ("NVIDIA GeForce RTX 3090", "8.6", 24576, "other", True),
    ("NVIDIA GeForce RTX 5080", "12.0", 16303, "other", False),
    ("NVIDIA GeForce RTX 5070 Ti", "12.0", 16303, "other", False),
    ("NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, "other", False),
    ("NVIDIA GeForce RTX 4080 SUPER", "8.9", 16376, "other", False),
    ("NVIDIA GeForce RTX 4080", "8.9", 16376, "other", False),
    ("NVIDIA GeForce RTX 4070 Ti SUPER", "8.9", 16376, "other", False),
    ("NVIDIA GeForce RTX 4060 Ti", "8.9", 16380, "other", False),       # the 16 GB card
    ("NVIDIA GeForce RTX 4060 Ti", "8.9", 8188, "small", False),        # the 8 GB card of the same name
    ("NVIDIA GeForce RTX 5070", "12.0", 12227, "small", False),
    ("NVIDIA GeForce RTX 3060", "8.6", 12288, "small", False),
    ("NVIDIA GeForce RTX 5090 Laptop GPU", "12.0", 24463, "other", True),
    ("NVIDIA GeForce RTX 5090 D", "12.0", 32607, "other", True),
    ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "12.0", 97887, "other", True),
    ("NVIDIA RTX A6000", "8.6", 49140, "other", True),
    ("NVIDIA GeForce RTX 2080 Ti", "7.5", 11264, "old", False),
    ("NVIDIA GeForce GTX 1080 Ti", "6.1", 11264, "old", False),
    ("NVIDIA A100 80GB PCIe", "8.0", 81920, "old", True),
    ("NVIDIA H100 PCIe", "9.0", 81559, "nocode", True),
    ("NVIDIA B200", "10.0", 183359, "nocode", True),
]


def smi_line(i, name, cc, total, used=1000, uuid=None):
    return "%d, %s, %s, %s, %d, %d, 616.92" % (i, uuid or "GPU-%08d-0000-0000-0000-000000000000" % i, name, cc, total, used)


class FakeFile:
    def __init__(self, name, size):
        self.name, self._size = name, size

    def stat(self):
        return os.stat_result((0, 0, 0, 0, 0, 0, self._size, 0, 0, 0))


class FakePack:
    def __init__(self, sizes, max_ctx=262144):
        self.weights, self.draft, self.draft_head = (FakeFile(n, s) for n, s in zip(("w.yskn", "bdraft2_k8.yskn", "bdraft2_k8.shead"), sizes))
        self.max_ctx = max_ctx


def det_of(*rows, environ=None):
    gpus = G.parse("\n".join(rows) + "\n")
    return G.detect(0, environ=environ or {}, gpus=gpus)


def load_quickstart():
    """quickstart/quickstart.py of the repository (or the file YASAKANI_QUICKSTART names); None when it is not there (the engine zip)."""
    for p in (os.environ.get("YASAKANI_QUICKSTART"), ROOT.parent / "quickstart" / "quickstart.py"):
        if p and Path(p).is_file():
            spec = importlib.util.spec_from_file_location("yasakani_quickstart_under_test", str(p))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    return None


class ParseTests(unittest.TestCase):
    def test_rows(self):
        gs = G.parse(smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607, 2644) + "\n" + smi_line(1, "NVIDIA GeForce RTX 3090", "8.6", 24576, 15))
        self.assertEqual(len(gs), 2)
        self.assertEqual((gs[0].index, gs[0].compute_cap, gs[0].total_mib, gs[0].used_mib, gs[0].free_mib), (0, (12, 0), 32607, 2644, 29963))
        self.assertEqual((gs[1].name, gs[1].cc_text, gs[1].driver), ("NVIDIA GeForce RTX 3090", "8.6", "616.92"))
        self.assertTrue(gs[0].uuid.startswith("GPU-"))

    def test_missing_values_and_noise(self):
        gs = G.parse("0, [N/A], NVIDIA GeForce RTX 4090, [N/A], 24564, [N/A], 581.57\nNo devices were found\n\n")
        self.assertEqual(len(gs), 1)
        g = gs[0]
        self.assertEqual((g.uuid, g.compute_cap, g.used_mib, g.free_mib, g.arch), (None, None, None, None, "unknown"))
        self.assertEqual(G.parse("No devices were found\n"), [])
        self.assertEqual(G.parse(""), [])

    def test_query_falls_back_without_compute_cap(self):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd[1])
            if "compute_cap" in cmd[1]:
                return subprocess.CompletedProcess(cmd, 2, stdout='Field "compute_cap" is not a valid field to query.\n', stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="0, GPU-1, NVIDIA GeForce RTX 3090, 24576, 500, 470.10\n", stderr="")
        gs = G.query({"YASAKANI_NVIDIA_SMI": "/x/nvidia-smi"}, run=run)
        self.assertEqual(len(calls), 2)
        self.assertEqual((gs[0].name, gs[0].compute_cap, gs[0].total_mib), ("NVIDIA GeForce RTX 3090", None, 24576))

    def test_query_without_nvidia_smi(self):
        self.assertIsNone(G.query({"YASAKANI_NVIDIA_SMI": "none"}))
        self.assertIsNone(G.query({"YASAKANI_NVIDIA_SMI": ""}))

        def boom(cmd, **kw):
            raise FileNotFoundError(cmd[0])
        self.assertIsNone(G.query({"YASAKANI_NVIDIA_SMI": "/nonexistent/nvidia-smi"}, run=boom))
        with mock.patch.object(G.shutil, "which", return_value=None):
            self.assertIsNone(G.query({}))
        self.assertFalse(G.detect(0, environ={"YASAKANI_NVIDIA_SMI": "none"}).known)


class DecisionTableTests(unittest.TestCase):
    def test_cards(self):
        for name, cc, total, qs, quality in CARDS:
            g = G.parse(smi_line(0, name, cc, total))[0]
            with self.subTest(card=name, total=total):
                self.assertEqual(g.tested, qs == "tested")
                self.assertEqual(g.arch, {"old": "old", "nocode": "no-code"}.get(qs, "ok"))
                line = G.describe(g)
                if qs == "tested":
                    self.assertIn("tested and tuned", line)
                elif qs == "other":
                    self.assertIn("not optimized or tested", line)
                    self.assertEqual("fastest model only" in line, not quality)
                elif qs == "small":
                    self.assertIn("too small", line)
                else:
                    self.assertIn("no code", line)

    def test_untested_note(self):
        self.assertIsNone(G.untested_note(det_of(smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607))))
        self.assertIsNone(G.untested_note(G.Detected()))                         # nothing known: no note
        n = G.untested_note(det_of(smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16376)))
        self.assertIn("RTX 4080 (compute capability 8.9)", n)
        self.assertIn("not optimized or tested", n)
        self.assertIn("RTX 5090 only", n)
        n = G.untested_note(det_of(smi_line(0, "NVIDIA GeForce RTX 2080 Ti", "7.5", 11264)))
        self.assertIn("will probably not start", n)


class SelectTests(unittest.TestCase):
    TWO = [smi_line(0, "NVIDIA GeForce RTX 3090", "8.6", 24576, 300, "GPU-aaaa1111-0000"),
           smi_line(1, "NVIDIA GeForce RTX 5090", "12.0", 32607, 2000, "GPU-bbbb2222-0000")]

    def gpus(self):
        return G.parse("\n".join(self.TWO))

    def test_one_gpu(self):
        gs = self.gpus()[:1]
        self.assertEqual(G.select(gs, 0, {}), gs)
        self.assertEqual(G.select(gs, 1, {}), [])
        self.assertEqual(G.select([], 0, {}), [])

    def test_two_gpus(self):
        gs = self.gpus()
        self.assertEqual(len(G.select(gs, 0, {})), 2)                           # CUDA's default order: not known which
        self.assertEqual(G.select(gs, 1, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"})[0].index, 1)
        self.assertEqual(G.select(gs, 0, {"CUDA_VISIBLE_DEVICES": "GPU-bbbb2222"})[0].index, 1)     # a unique UUID prefix
        self.assertEqual(G.select(gs, 0, {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})[0].index, 1)
        self.assertEqual(len(G.select(gs, 0, {"CUDA_VISIBLE_DEVICES": "1"})), 2)
        self.assertEqual(G.select(gs, 1, {"CUDA_VISIBLE_DEVICES": "1"}), [])    # only one GPU is visible
        self.assertEqual(G.select(gs, 0, {"CUDA_VISIBLE_DEVICES": ""}), [])

    def test_two_gpus_plan_for_the_smaller(self):
        det = G.detect(0, environ={}, gpus=self.gpus())
        self.assertTrue(det.untested)                                          # one of them is not an RTX 5090
        w = det.worst()
        self.assertEqual((w.total_mib, w.free_mib, w.compute_cap), (24576, 24276, (8, 6)))
        plan = G.plan_context(FakePack(QUALITY), det)
        self.assertIsNone(plan.max_ctx)                                         # the quality model fits both at 262,144
        self.assertTrue(any("2 GPUs" in n for n in plan.notes))
        det = G.detect(0, environ={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, gpus=self.gpus())
        self.assertEqual(det.gpu.index, 0)
        det = G.detect(1, environ={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, gpus=self.gpus())
        self.assertTrue(det.tested)


# issue #20: an RTX A2000 12 GB at PCI 01:00.0 (nvidia-smi's GPU 0, the display) and an RTX 5060 Ti 16 GB at 02:00.0
REPORTED = [smi_line(0, "NVIDIA RTX A2000 12GB", "8.6", 12282, 826, "GPU-a2000000-0000"),
            smi_line(1, "NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, 400, "GPU-50600000-0000")]
PCI = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


class DeviceOrderTests(unittest.TestCase):
    """0.1.5: --device N is nvidia-smi's GPU N (the package sets CUDA_DEVICE_ORDER=PCI_BUS_ID), and the plan is made for that GPU."""

    def gpus(self):
        return G.parse("\n".join(REPORTED))

    def test_device_numbers(self):
        gs = self.gpus()
        self.assertEqual(G.select(gs, 0, PCI)[0].name, "NVIDIA RTX A2000 12GB")
        self.assertEqual(G.select(gs, 1, PCI)[0].name, "NVIDIA GeForce RTX 5060 Ti")
        self.assertEqual(G.select(gs, 2, PCI), [])
        self.assertTrue(G.detect(1, environ=PCI, gpus=gs).numbered)
        self.assertEqual(len(G.select(gs, 0, {})), 2)                          # CUDA's own order (0.1.4): not known which
        self.assertFalse(G.detect(0, environ={}, gpus=gs).numbered)
        self.assertTrue(G.detect(0, environ={}, gpus=gs[:1]).numbered)          # one GPU: always known
        det = G.detect(0, environ=dict(PCI, CUDA_VISIBLE_DEVICES="GPU-5060"), gpus=gs)
        self.assertEqual((det.gpu.name, det.numbered), ("NVIDIA GeForce RTX 5060 Ti", False))

    def test_plan_for_the_chosen_gpu(self):
        plan = G.plan_context(FakePack(FASTEST), G.detect(1, environ=PCI, gpus=self.gpus()))
        self.assertEqual((plan.max_ctx, plan.env, plan.free_mib), (98304, {"YSKN_BDR_SHEAD": "0"}, 15911))
        self.assertFalse(any("GPUs" in n for n in plan.notes))
        with self.assertRaises(YasakaniError) as cm:                           # GPU 0 is the A2000: it says so, and which GPU has room
            G.plan_context(FakePack(FASTEST), G.detect(0, environ=PCI, gpus=self.gpus()))
        msg = str(cm.exception)
        self.assertIn("(11456 of 12282 MiB free; it needs about 13906 MiB with a context of 16384 tokens)", msg)
        self.assertIn("That is nvidia-smi's GPU 0 (NVIDIA RTX A2000 12GB); GPU 1 (NVIDIA GeForce RTX 5060 Ti, 15911 MiB free) has room "
                      "for it: add --device 1.", msg)
        with self.assertRaises(YasakaniError) as cm:                           # CUDA_DEVICE_ORDER set to something else: why the numbers
            G.plan_context(FakePack(FASTEST), G.detect(0, environ={"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, gpus=self.gpus()))
        self.assertIn("This PC has 2 GPUs and CUDA's device 0 cannot be matched", str(cm.exception))
        self.assertNotIn("add --device", str(cm.exception))
        with self.assertRaises(YasakaniError) as cm:                           # named by UUID: no --device advice (it counts otherwise)
            G.plan_context(FakePack(FASTEST), G.detect(0, environ=dict(PCI, CUDA_VISIBLE_DEVICES="GPU-a2000"), gpus=self.gpus()))
        self.assertIn("That is nvidia-smi's GPU 0 (NVIDIA RTX A2000 12GB).  Close", str(cm.exception))

    def test_one_gpu_message_unchanged(self):
        for env in ({}, PCI):
            with self.assertRaises(YasakaniError) as cm:
                G.plan_context(FakePack(FASTEST), det_of(smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16376, 14000), environ=env))
            self.assertEqual(str(cm.exception), "the model does not fit in the free GPU memory (2376 of 16376 MiB free; it needs about 13906 "
                             "MiB with a context of 16384 tokens).  Close other programs that use the GPU, or give --max-ctx to try anyway.")

    def test_numbering_line(self):
        self.assertEqual(G.numbering(PCI), "--device N is GPU N above (PCI bus order)")
        self.assertIn("CUDA_VISIBLE_DEVICES=GPU-50600000", G.numbering(dict(PCI, CUDA_VISIBLE_DEVICES="GPU-50600000")))
        self.assertIn("CUDA_DEVICE_ORDER=FASTEST_FIRST", G.numbering({"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}))


class ContextRuleTests(unittest.TestCase):
    """The rule, with the numbers of the published files (yasakani/gpu.py: choose, plan_context)."""

    def test_choose(self):
        self.assertEqual(G.choose(16300 - 1000, *FASTEST, 262144), (65536, True))      # 16 GB that also drives the display
        self.assertEqual(G.choose(16311 - 0, *FASTEST, 262144), (65536, False))       # 16 GB without a display
        self.assertEqual(G.choose(16303 - 1500, *FASTEST, 262144), (49152, True))
        self.assertEqual(G.choose(16303 - 3000, *FASTEST, 262144)[0], 0)              # does not fit
        self.assertEqual(G.choose(24564 - 1000, *QUALITY, 262144), (262144, False))   # 24 GB: the quality model at full context
        self.assertEqual(G.choose(24564 - 4000, *QUALITY, 262144), (163840, False))
        self.assertEqual(G.choose(24564 - 1000, *FASTEST, 262144), (262144, False))
        self.assertEqual(G.choose(16376 - 800, *QUALITY, 262144)[0], 0)               # 16 GB: the quality model does not fit
        self.assertEqual(G.choose(12288 - 300, *FASTEST, 262144)[0], 0)               # 12 GB: nothing fits
        self.assertEqual(G.choose(20000, *FASTEST, 4096), (4096, False))              # a model whose own context is small

    def test_rule_is_safe_against_the_measured_overhead(self):
        """The chosen context must fit with the overhead measured on the RTX 5090 (2,090 / 2,291 MiB above the files)."""
        for sizes, measured in ((FASTEST, 2090), (QUALITY, 2291)):
            for total in (16303, 16376, 20470, 24564, 24576):
                for used in range(0, 6001, 250):
                    n, head_off = G.choose(total - used, *sizes, 262144)
                    if not n:
                        continue
                    files = sizes[0] + sizes[1] + (0 if head_off else sizes[2])
                    need = files / G.MIB + measured + n * G.KV_BYTES_PER_TOKEN / G.MIB
                    self.assertLessEqual(need + 500, total - used, (sizes, total, used, n, head_off))
                    self.assertEqual(n % G.CONTEXT_STEP if n < 262144 else 0, 0)

    def test_plan_unchanged_on_the_rtx_5090(self):
        for used in (0, 2644, 12000, 20000):                                   # however much else uses it
            plan = G.plan_context(FakePack(QUALITY), det_of(smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607, used)))
            self.assertEqual((plan.max_ctx, plan.env, plan.notes), (None, {}, []))
        plan = G.plan_context(FakePack(QUALITY), G.Detected())                    # nvidia-smi gave nothing
        self.assertEqual((plan.max_ctx, plan.env, plan.notes), (None, {}, []))

    def test_plan_16gb(self):
        plan = G.plan_context(FakePack(FASTEST), det_of(smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16376, 1000)))
        self.assertEqual(plan.max_ctx, 81920)
        self.assertEqual(plan.env, {"YSKN_BDR_SHEAD": "0"})
        self.assertTrue(any("bdraft2_k8.shead" in n and "3%" in n for n in plan.notes))
        self.assertTrue(any(n.startswith("context: 81920 tokens") for n in plan.notes))
        plan = G.plan_context(FakePack(FASTEST), det_of(smi_line(0, "NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, 0)))
        self.assertEqual((plan.max_ctx, plan.env), (65536, {}))

    def test_plan_24gb(self):
        plan = G.plan_context(FakePack(QUALITY), det_of(smi_line(0, "NVIDIA GeForce RTX 3090", "8.6", 24576, 1000)))
        self.assertEqual((plan.max_ctx, plan.env, plan.notes), (None, {}, []))
        plan = G.plan_context(FakePack(QUALITY), det_of(smi_line(0, "NVIDIA GeForce RTX 4090", "8.9", 24564, 4000)))
        self.assertEqual((plan.max_ctx, plan.env), (163840, {}))

    def test_plan_does_not_fit(self):
        with self.assertRaises(YasakaniError) as cm:
            G.plan_context(FakePack(QUALITY), det_of(smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16376, 500)))
        self.assertIn("does not fit", str(cm.exception))
        self.assertIn("use the fastest model", str(cm.exception))
        with self.assertRaises(YasakaniError) as cm:
            G.plan_context(FakePack(FASTEST), det_of(smi_line(0, "NVIDIA GeForce RTX 3060", "8.6", 12288, 300)))
        self.assertIn("--max-ctx", str(cm.exception))
        self.assertNotIn("fastest model", str(cm.exception))

    def test_plan_unknown_memory(self):
        det = det_of("0, GPU-1, NVIDIA GeForce RTX 4080, 8.9, 16376, [N/A], 581.57")
        self.assertEqual(G.plan_context(FakePack(FASTEST), det).max_ctx, None)

    def test_given_context(self):
        """--max-ctx: the context stays as given; the draft model's output head only where it fits beside it (plan_given_context)."""
        ti = det_of(smi_line(0, "NVIDIA GeForce RTX 5070 Ti", "12.0", 16303, 340))     # no display on the card: 340 MiB in use
        plan = G.plan_given_context(FakePack(FASTEST), ti, 16384)                    # the head fits beside 16,384 tokens
        self.assertEqual((plan.max_ctx, plan.env, plan.notes), (16384, {}, []))
        plan = G.plan_given_context(FakePack(FASTEST), ti, 98304)                    # it does not beside 98,304 (0.1.4 loaded it anyway)
        self.assertEqual((plan.max_ctx, plan.env), (98304, {"YSKN_BDR_SHEAD": "0"}))
        self.assertEqual(len(plan.notes), 1)
        self.assertIn("bdraft2_k8.shead", plan.notes[0])
        self.assertIn("beside 98304 tokens", plan.notes[0])
        self.assertNotIn("context: ", plan.notes[0])                                 # only the rule's own choice is reported that way
        plan = G.plan_given_context(FakePack(FASTEST), ti, 1048576)                  # far too long: still tried as given
        self.assertEqual((plan.max_ctx, plan.env), (1048576, {"YSKN_BDR_SHEAD": "0"}))
        rtx4090 = det_of(smi_line(0, "NVIDIA GeForce RTX 4090", "8.9", 24564, 4000))
        self.assertEqual(G.plan_given_context(FakePack(QUALITY), rtx4090, 163840).env, {})     # what plan_context chooses there
        self.assertEqual(G.plan_given_context(FakePack(QUALITY), rtx4090, 262144).env, {"YSKN_BDR_SHEAD": "0"})
        for det in (det_of(smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607, 20000)), G.Detected(),
                    det_of("0, GPU-1, NVIDIA GeForce RTX 4080, 8.9, 16376, [N/A], 581.57")):   # 30,000 MiB or more, or unknown: unchanged
            plan = G.plan_given_context(FakePack(QUALITY), det, 262144)
            self.assertEqual((plan.max_ctx, plan.env, plan.notes), (262144, {}, []))


class QuickstartTests(unittest.TestCase):
    """The quickstart's own copy of the decision and the rule (it runs before the package is installed)."""

    @classmethod
    def setUpClass(cls):
        cls.qs = load_quickstart()
        if cls.qs is None:
            raise unittest.SkipTest("quickstart/quickstart.py not found (set YASAKANI_QUICKSTART)")

    def test_version(self):
        self.assertEqual(self.qs.QS_VERSION, "0.1.5")
        self.assertIn("/v0.1.5/", self.qs.ENGINE_URL)

    def test_decision_table(self):
        for name, cc, total, qs, quality in CARDS:
            g = self.qs.parse_gpus(smi_line(0, name, cc, total))[0]
            with self.subTest(card=name, total=total):
                self.assertEqual(self.qs.gpu_verdict(g), qs)
                if qs in ("tested", "other"):
                    self.assertEqual(total >= self.qs.QUALITY_MIN_TOTAL_MIB, quality)
        g = self.qs.parse_gpus("0, GPU-1, NVIDIA GeForce RTX 4090, [N/A], 24564, 900, 470.10")[0]    # an old driver: cc unknown
        self.assertEqual(self.qs.gpu_verdict(g), "other")

    def test_same_as_package(self):
        for k in ("FULL_CONTEXT_MIN_TOTAL_MIB", "QUALITY_MIN_TOTAL_MIB", "MIN_TOTAL_MIB", "KV_BYTES_PER_TOKEN", "CONTEXT_STEP", "MIN_CONTEXT",
                  "HEAD_OFF_BELOW", "MARGIN_MIB"):
            self.assertEqual(getattr(G, k), getattr(self.qs, k), k)
        for name, cc, total, qs, quality in CARDS:
            g = G.parse(smi_line(0, name, cc, total))[0]
            self.assertEqual(g.tested, self.qs.gpu_verdict(self.qs.parse_gpus(smi_line(0, name, cc, total))[0]) == "tested", name)
        for model, sizes in (("speed", FASTEST), ("quality", QUALITY)):
            self.assertEqual(self.qs.MODEL_BYTES[model], sizes)
            self.assertEqual(self.qs.OVERHEAD_MIB[model], G.overhead_mib(sizes[0]))
            for free in range(9000, 32001, 37):
                self.assertEqual(self.qs.estimate_context(free, model), G.choose(free, *sizes, 262144), (model, free))

    def test_pick(self):
        qs = self.qs
        two = qs.parse_gpus(smi_line(0, "NVIDIA GeForce RTX 3060", "8.6", 12288, 300, "GPU-aaaa") + "\n"
                            + smi_line(1, "NVIDIA GeForce RTX 5090", "12.0", 32607, 2000, "GPU-bbbb"))
        g, v, pin = qs.pick_gpu(two, environ={})
        self.assertEqual((g["index"], v, pin), (1, "tested", "GPU-bbbb"))
        g, v, pin = qs.pick_gpu(two[:1], environ={})
        self.assertEqual((g["index"], v, pin), (0, "small", None))
        g, v, pin = qs.pick_gpu(two, environ={"CUDA_VISIBLE_DEVICES": "GPU-aaaa"})       # the user chose one
        self.assertEqual((g["index"], v, pin), (0, "small", None))
        g, v, pin = qs.pick_gpu(two, environ={"CUDA_VISIBLE_DEVICES": "0"})              # 0.1.5: numbers are nvidia-smi's (PCI order)
        self.assertEqual((g["index"], v, pin), (0, "small", None))
        g, v, pin = qs.pick_gpu(two, environ={"CUDA_VISIBLE_DEVICES": "1"})
        self.assertEqual((g["index"], v, pin), (1, "tested", None))
        g, v, pin = qs.pick_gpu(two, environ={"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"})   # not known which:
        self.assertEqual((g["index"], v, pin), (0, "small", None))                                                # the less able one
        three = qs.parse_gpus("\n".join([smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16376, 100, "GPU-c"),
                                         smi_line(1, "NVIDIA GeForce RTX 3090", "8.6", 24576, 900, "GPU-d"),
                                         smi_line(2, "NVIDIA GeForce RTX 2080 Ti", "7.5", 11264, 0, "GPU-e")]))
        g, v, pin = qs.pick_gpu(three, environ={})
        self.assertEqual((g["index"], v, pin), (1, "other", "GPU-d"))

    def test_query_fallback(self):
        def run(cmd, **kw):
            if "compute_cap" in cmd[1]:
                return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="0, GPU-1, NVIDIA GeForce RTX 3090, 24576, 500, 470.10\n", stderr="")
        gs = self.qs.query_gpus(run=run)
        self.assertEqual((gs[0]["name"], gs[0]["cc"], gs[0]["total"]), ("NVIDIA GeForce RTX 3090", None, 24576))

    def test_messages(self):
        T = self.qs.T
        self.assertIn("最適化も動作の確認もしていません（確かめたのは RTX 5090 だけです）", T["gpu_other"]["ja"])
        for key in ("no_smi", "no_gpu"):
            self.assertIn("AMD", T[key]["ja"])
            self.assertIn("Mac", T[key]["en"])
        self.assertNotIn("continue_q", T)
        for key, kw in (("gpu_is", dict(g="X", c="8.9", t=16376, d="1")), ("gpu_small", dict(t=8188)), ("gpu_old", dict(c="7.5")),
                        ("gpu_nocode", dict(c="9.0")), ("ctx_auto", dict(n="65,536")), ("ctx_short", dict(u=9000)),
                        ("gpu_pick", dict(n=2, g="X"))):
            for lang in ("ja", "en"):
                T[key][lang].format(**kw)


FAKE_SMI = r'''#!{python}
import json, os, sys
gpus = json.loads(os.environ["FAKE_GPUS"])
q = [a for a in sys.argv[1:] if a.startswith("--query-gpu=")]
if not q:
    sys.exit(0)
fields = q[0].split("=", 1)[1].split(",")
units = "nounits" not in " ".join(sys.argv)
known = set(gpus[0]) if gpus else set()
bad = [f for f in fields if f not in known]
if bad:
    print('Field "%s" is not a valid field to query.' % bad[0])
    sys.exit(2)
for g in gpus:
    print(", ".join(str(g[f]) + (" MiB" if units and f.startswith("memory.") else "") for f in fields))
'''


@unittest.skipUnless(support.tokenizer_dir() is not None and os.name == "posix",
                     "needs the Qwen3.8 tokenizer files (YASAKANI_TEST_MODEL_DIR) and a POSIX test machine")
class CliGpuTests(unittest.TestCase):
    """The command line with a stand-in nvidia-smi and the stub engine library."""

    @classmethod
    def setUpClass(cls):
        cls.dll = str(support.stub_path())
        cls.folder = support.make_real_folder("gpu")
        cls.addClassCleanup(shutil.rmtree, cls.folder.parent, True)
        with open(cls.folder / "bdraft2_k8.shead", "wb") as f:                # a sparse file of the published size: the rule sees it
            f.truncate(FASTEST[2])
        tmp = Path(tempfile.mkdtemp(prefix="fake_smi_"))
        cls.addClassCleanup(shutil.rmtree, tmp, True)
        cls.smi = tmp / "nvidia-smi"
        cls.smi.write_text(FAKE_SMI.replace("{python}", sys.executable), encoding="utf-8")
        cls.smi.chmod(0o755)

    def gpu(self, name, cc, total, used, index=0):
        return dict(index=index, uuid="GPU-%d" % index, name=name, compute_cap=cc, **{"memory.total": total, "memory.used": used},
                    driver_version="616.92")

    def run_cli(self, gpus, *args, timeout=60, stub=None):
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", YASAKANI_NVIDIA_SMI=str(self.smi), FAKE_GPUS=json.dumps(gpus))
        for k in [k for k in env if k.startswith(("YSKN_", "STUB_"))] + ["YASAKANI_DLL", "YASAKANI_MODEL_DIR", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"]:
            env.pop(k, None)
        env.update(stub or {})                                                   # knobs of the stub library (tests/stub/yskn_stub.c)
        return subprocess.run([sys.executable, "-m", "yasakani", *args], capture_output=True, text=True, timeout=timeout, cwd=str(ROOT),
                              env=env, input="/exit\n")

    def common(self):
        return ["--model-dir", str(self.folder), "--dll", self.dll]

    def test_fake_smi(self):
        r = subprocess.run([str(self.smi), "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True,
                           env=dict(os.environ, FAKE_GPUS=json.dumps([self.gpu("X", "8.9", 16376, 10)])))
        self.assertEqual(r.stdout.strip(), "X, 16376 MiB")

    def test_rtx_5090_unchanged(self):
        r = self.run_cli([self.gpu("NVIDIA GeForce RTX 5090", "12.0", 32607, 2644)], "chat", *self.common(), "--no-think")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(context 262144 tokens)", r.stderr)
        self.assertNotIn("note:", r.stderr)
        self.assertNotIn("context: ", r.stderr)

    def test_16gb_context_and_note(self):
        gpus = [self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 11418)]      # about 4,958 MiB free beside the placeholder weights
        r = self.run_cli(gpus, "chat", *self.common(), "--no-think")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NVIDIA GeForce RTX 4080 (compute capability 8.9)", r.stderr)
        self.assertIn("not optimized or tested", r.stderr)
        self.assertIn("bdraft2_k8.shead", r.stderr)
        self.assertIn("context: 98304 tokens", r.stderr)
        self.assertIn("(context 98304 tokens)", r.stderr)
        r = self.run_cli(gpus, "chat", *self.common(), "--no-think", "--max-ctx", "8192")       # given: no rule
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(context 8192 tokens)", r.stderr)
        self.assertNotIn("context: ", r.stderr)
        self.assertNotIn("bdraft2_k8.shead", r.stderr)                                    # the head fits beside 8,192 tokens
        self.assertIn("not optimized or tested", r.stderr)

    def test_given_context_leaves_out_the_head(self):
        gpus = [self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 11418)]                 # about 4,958 MiB free beside the placeholders
        r = self.run_cli(gpus, "chat", *self.common(), "--no-think", "--max-ctx", "98304")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("bdraft2_k8.shead", r.stderr)
        self.assertIn("is not loaded: it does not fit beside 98304 tokens", r.stderr)
        self.assertIn("(context 98304 tokens)", r.stderr)                               # the given context, unchanged
        self.assertNotIn("context: ", r.stderr)

    def test_does_not_fit(self):
        r = self.run_cli([self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 14000)], "chat", *self.common())
        self.assertEqual(r.returncode, 1)
        self.assertIn("does not fit in the free GPU memory", r.stderr)
        self.assertNotIn("loading the model", r.stderr)

    def test_no_nvidia_smi(self):
        env_gpus = [self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 14000)]
        env = dict(os.environ, PYTHONPATH=str(ROOT), YASAKANI_NVIDIA_SMI="none", FAKE_GPUS=json.dumps(env_gpus))
        r = subprocess.run([sys.executable, "-m", "yasakani", "chat", *self.common(), "--no-think"], capture_output=True, text=True,
                           timeout=60, cwd=str(ROOT), env=env, input="/exit\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(context 262144 tokens)", r.stderr)

    def test_info(self):
        r = self.run_cli([self.gpu("NVIDIA GeForce RTX 3090", "8.6", 24576, 500)], "info", *self.common())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("NVIDIA GeForce RTX 3090, 24576 MiB, 500 MiB", r.stdout)
        self.assertIn("GPU 0: compute capability 8.6 - runs since 0.1.3, but not optimized or tested", r.stdout)

    def test_two_gpus_device_numbers(self):
        """issue #20: --device N runs on nvidia-smi's GPU N, and the free-memory check looks at that GPU."""
        # the folder's weights are placeholders (only the head has its published size): the used memory stands in for the weights
        gpus = [self.gpu("NVIDIA RTX A2000 12GB", "8.6", 12282, 9500, 0), self.gpu("NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, 11353, 1)]
        log = self.folder.parent / "open_two.log"
        r = self.run_cli(gpus, "chat", *self.common(), "--no-think", "--device", "1", stub=dict(STUB_OPEN_LOG=str(log)))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NVIDIA GeForce RTX 5060 Ti (compute capability 12.0)", r.stderr)
        self.assertIn("context: 98304 tokens, chosen from the free GPU memory (4958 of 16311 MiB free)", r.stderr)
        self.assertEqual(log.read_text().splitlines(), ["device=1 order=PCI_BUS_ID visible=(unset)"])
        r = self.run_cli(gpus, "chat", *self.common(), "--device", "0", stub=dict(STUB_OPEN_LOG=str(log)))
        self.assertEqual(r.returncode, 1)
        self.assertIn("(2782 of 12282 MiB free", r.stderr)
        self.assertIn("add --device 1", r.stderr)
        self.assertNotIn("loading the model", r.stderr)
        r = self.run_cli(gpus, "chat", *self.common(), "--no-think", "--device", "0", "--max-ctx", "8192",
                         stub=dict(STUB_OPEN_LOG=str(log), CUDA_DEVICE_ORDER="FASTEST_FIRST"))
        self.assertEqual(r.returncode, 0, r.stderr)                            # a value the user set is kept
        self.assertEqual(log.read_text().splitlines()[-1], "device=0 order=FASTEST_FIRST visible=(unset)")
        r = self.run_cli(gpus, "info", *self.common())
        self.assertIn("--device N is GPU N above (PCI bus order)", r.stdout)
        r = self.run_cli(gpus[:1], "info", *self.common())
        self.assertNotIn("--device", r.stdout)                                   # one GPU: as before

    def test_selfcheck_exit_codes(self):
        ref = self.folder.parent / "ref_gpu.json"
        args = ["--model-dir", str(self.folder), "--dll", self.dll, "--max-ctx", "4096"]
        g5090 = [self.gpu("NVIDIA GeForce RTX 5090", "12.0", 32607, 2644)]
        g4080 = [self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 1000)]
        diff = dict(STUB_GEN_DIFF_AT="5")                                         # the generated output differs from id #6 on
        r = self.run_cli(g5090, "selfcheck", *args, "--write", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        d = json.loads(ref.read_text(encoding="utf-8"))
        for c in d["cases"]:                                                      # 0.1.4: the full output and the predictions
            self.assertEqual(len(c["ids"]), c["tokens"])
            self.assertEqual(len(c["forced"]["nll"]), c["prompt_tokens"] + c["tokens"] - 1)
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)                 # the same output and the same predictions
        self.assertIn("its predictions are also compared", r.stdout)
        self.assertIn("same most likely id at 100.0 %", r.stdout)
        self.assertIn("self-check passed: this installation reproduces the reference output.", r.stdout)
        r = self.run_cli(g5090, "selfcheck", *args, "--reference", str(ref), stub=dict(STUB_TF_NLL_SHIFT="0.05"))
        self.assertEqual(r.returncode, 0, r.stdout)                            # the RTX 5090: the output alone decides, as before
        self.assertNotIn("reads the reference output", r.stdout)
        r = self.run_cli(g5090, "selfcheck", *args, "--reference", str(ref), stub=diff)
        self.assertEqual(r.returncode, 1, r.stdout)                            # on the RTX 5090 a difference is a failure, as before
        self.assertIn("FAIL  the output differs", r.stdout)
        self.assertIn("self-check FAILED for 3 case(s).", r.stdout)
        self.assertNotIn("reads the reference output", r.stdout)
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(ref), stub=diff)
        self.assertEqual(r.returncode, 3, r.stdout)                            # elsewhere: the predictions on the reference output decide
        self.assertIn("DIFF  the output differs", r.stdout)
        self.assertIn("made on an RTX 5090", r.stdout)
        n = sum(c["prompt_tokens"] + c["tokens"] - 1 for c in d["cases"])
        self.assertIn("all %d positions: same most likely id at 100.0 %%" % n, r.stdout)
        self.assertIn("(generated output differs from id #6)", r.stdout)
        self.assertIn("passed within the tolerances", r.stdout)
        self.assertIn("Exit code 3", r.stdout)
        for knob, text in ((dict(STUB_TF_NLL_SHIFT="0.05"), "differs by 0.0500 on average"),        # a GPU that computes wrongly
                           (dict(STUB_TF_FLIP_EVERY="10"), "the most likely id is the same at only 9"),
                           (dict(STUB_TF_NAN="30"), "NaN")):
            r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(ref), stub=dict(diff, **knob))
            self.assertEqual(r.returncode, 1, (knob, r.stdout))
            self.assertIn("self-check FAILED: ", r.stdout)
            self.assertIn(text, r.stdout)
            self.assertIn("yasakani verify", r.stdout)
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(ref), stub=dict(STUB_TF_NLL_SHIFT="0.05"))
        self.assertEqual(r.returncode, 1, r.stdout)                            # the same output, but wrong predictions: a failure
        self.assertIn("differs by 0.0500 on average", r.stdout)
        old = self.folder.parent / "ref_gpu_old.json"                            # a reference file written before 0.1.4
        old.write_text(json.dumps(dict(d, cases=[{k: v for k, v in c.items() if k not in ("ids", "forced")} for c in d["cases"]])),
                       encoding="utf-8")
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(old))
        self.assertEqual(r.returncode, 0, r.stdout)                            # the same output; nothing more to compare with
        self.assertNotIn("also compared", r.stdout)
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(old), stub=diff)
        self.assertEqual(r.returncode, 1, r.stdout)                            # nothing to compare with: not passed
        self.assertIn("has no full output", r.stdout)
        d["cases"][2]["prompt_sha256"] = "1" * 64
        ref.write_text(json.dumps(d), encoding="utf-8")
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(ref), stub=diff)
        self.assertEqual(r.returncode, 1, r.stdout)                            # a tokenizer problem stays a failure
        self.assertIn("prompt ids differ", r.stdout)
        self.assertNotIn("reads the reference output", r.stdout)

    def test_selfcheck_consistency(self):
        """0.1.4: every GPU also checks its generated output against its own teacher-forced predictions (16- and 64-row passes)."""
        ref = self.folder.parent / "ref_cons.json"
        args = ["--model-dir", str(self.folder), "--dll", self.dll, "--max-ctx", "4096"]
        g5090 = [self.gpu("NVIDIA GeForce RTX 5090", "12.0", 32607, 2644)]
        g5080 = [self.gpu("NVIDIA GeForce RTX 5080", "12.0", 16303, 1000)]
        g4080 = [self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 1000)]
        r = self.run_cli(g4080, "selfcheck", *args, "--write", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("not known to be an RTX 5090", r.stdout)                  # --write elsewhere: a note, and the GPU in the file
        self.assertEqual(json.loads(ref.read_text(encoding="utf-8"))["gpu"], "NVIDIA GeForce RTX 4080")
        r = self.run_cli(g5090, "selfcheck", *args, "--write", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("not known to be an RTX 5090", r.stdout)
        self.assertEqual(json.loads(ref.read_text(encoding="utf-8"))["gpu"], "NVIDIA GeForce RTX 5090")
        self.assertNotIn("self-consistency", r.stdout)                          # --write checks nothing
        r = self.run_cli(g5090, "selfcheck", *args, "--reference", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("self-check passed: this installation reproduces the reference output.", r.stdout)
        self.assertIn("self-consistency (every GPU, no reference)", r.stdout)
        self.assertIn("every generated id is the most likely id (41 ids)", r.stdout)
        self.assertIn("note: this GPU does not use passes of more than 16 positions", r.stdout)
        self.assertIn("self-consistency passed.", r.stdout)
        wide = dict(STUB_TF_WIDE="1", STUB_GEN_MAX_ROWS="64", STUB_GEN_WIDE_PASSES="3")
        r = self.run_cli(g5090, "selfcheck", *args, "--reference", str(ref), stub=wide)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("3 of them wider than 16 positions, the widest 64", r.stdout)
        self.assertIn("the same predictions as in passes of 16, bit for bit", r.stdout)
        r = self.run_cli(g5080, "selfcheck", *args, "--reference", str(ref), stub=dict(wide, STUB_GEN_DIFF_AT="5"))
        self.assertEqual(r.returncode, 3, r.stdout)                            # another RTX 50: within the tolerances, consistent
        self.assertIn("passed within the tolerances", r.stdout)
        self.assertIn("self-consistency passed.", r.stdout)
        for gpus, knob, text in (
                (g5090, dict(STUB_TF_FLIP_EVERY="10"), "of the 41 generated ids are not the most likely id when this GPU reads"),
                (g5090, dict(wide, STUB_TF_WIDE_NLL_SHIFT="0.000001"), "passes of 64 positions do not give the same predictions"),
                (g5090, dict(STUB_GEN_MAX_ROWS="64"), "were used in the generation, but reading the text in passes of 64 failed"),
                (g5080, dict(wide, STUB_TF_WIDE_NLL_SHIFT="0.000001"), "passes of 64 positions do not give the same predictions")):
            r = self.run_cli(gpus, "selfcheck", *args, "--reference", str(ref), stub=knob)
            self.assertEqual(r.returncode, 1, (knob, r.stdout))
            self.assertIn("self-check FAILED: ", r.stdout)
            self.assertIn(text, r.stdout)
            self.assertIn("the same sums in the same order", r.stdout)


@unittest.skipUnless(support.tokenizer_dir() is not None and os.name == "posix",
                     "needs the Qwen3.8 tokenizer files (YASAKANI_TEST_MODEL_DIR) and a POSIX test machine")
class OpenStackTests(unittest.TestCase):
    """_open_stack passes the rule's context and engine setting to the Engine (in this process, the stub library)."""

    def test_head_off_reaches_the_engine(self):
        from yasakani import cli
        folder = support.make_real_folder("stack")
        self.addCleanup(shutil.rmtree, folder.parent, True)
        with open(folder / "bdraft2_k8.shead", "wb") as f:
            f.truncate(FASTEST[2])
        gpus = G.parse(smi_line(0, "NVIDIA GeForce RTX 5070 Ti", "12.0", 16303, 11300))
        args = cli.build_parser().parse_args(["chat", "--model-dir", str(folder), "--dll", str(support.stub_path())])
        saved = {k: v for k, v in os.environ.items() if k.startswith("YSKN_")}

        def restore():                                                           # the engine settings of this process as before
            for k in [k for k in os.environ if k.startswith("YSKN_")]:
                del os.environ[k]
            os.environ.update(saved)
        self.addCleanup(restore)
        with mock.patch.object(G, "query", return_value=gpus):
            pack, eng, fmt, runner = cli._open_stack(args, warm=False)
        try:
            self.assertEqual(eng.max_ctx, 98304)
            self.assertEqual(eng.profile.env["YSKN_BDR_SHEAD"], "0")
            self.assertEqual(os.environ["YSKN_BDR_SHEAD"], "0")
            self.assertTrue(args.gpu_detected.untested)
        finally:
            eng.close()

    def test_given_context_head_reaches_the_engine(self):
        """--max-ctx: the head is left out where it does not fit beside the given context, and loaded where it does."""
        from yasakani import cli
        folder = support.make_real_folder("given")
        self.addCleanup(shutil.rmtree, folder.parent, True)
        with open(folder / "bdraft2_k8.shead", "wb") as f:
            f.truncate(FASTEST[2])
        gpus = G.parse(smi_line(0, "NVIDIA GeForce RTX 5070 Ti", "12.0", 16303, 11300))     # about 5,000 MiB free
        saved = {k: v for k, v in os.environ.items() if k.startswith("YSKN_")}

        def restore():
            for k in [k for k in os.environ if k.startswith("YSKN_")]:
                del os.environ[k]
            os.environ.update(saved)
        self.addCleanup(restore)
        for ctx, head_off in (("98304", True), ("16384", False)):
            args = cli.build_parser().parse_args(["chat", "--model-dir", str(folder), "--dll", str(support.stub_path()), "--max-ctx", ctx])
            with mock.patch.object(G, "query", return_value=gpus):
                pack, eng, fmt, runner = cli._open_stack(args, warm=False)
            try:
                head = "0" if head_off else str(pack.draft_head)                     # "0" leaves it out; its path loads it
                self.assertEqual(eng.max_ctx, int(ctx))
                self.assertEqual(eng.profile.env["YSKN_BDR_SHEAD"], head)
                self.assertEqual(os.environ["YSKN_BDR_SHEAD"], head)
            finally:
                eng.close()


# ---- AMD (the HIP backend): a fake amdgpu sysfs tree and rocm-smi CSV, never a real GPU -------------------------------------------
AMD_TOTAL_MIB = 24560                     # RX 7900 XTX: sysfs mem_info_vram_total = 25753026560 bytes
AMD_USED_MIB = 1361                       # the desktop: mem_info_vram_used = 1427345408 bytes
RX7900XTX = (1, "0000:03:00.0", "0x1002", "0x744c", AMD_TOTAL_MIB * G.MIB, AMD_USED_MIB * G.MIB)
# no nvidia-smi (a path that cannot run: query() gives None on any machine) and no rocm-smi: the sysfs/PCI-id fallback alone
NO_SMI_ENV = {"YASAKANI_NVIDIA_SMI": "/nonexistent/nvidia-smi", "YASAKANI_ROCM_SMI": "none"}
ROCM_CSV = ("WARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status\n\n"
            "_amdgpu_device_initialize: amdgpu_get_auth (1) failed (-1)\n"
            "device,Card Series,Card Model,Card Vendor,Card SKU,Subsystem ID,Device Rev,Node ID,GUID,GFX Version\n"
            "card0,%s,0x744c,Advanced Micro Devices Inc. [AMD/ATI],D7020100,0x0e3b,0xc8,1,60133,%s\n")


def _no_smi(cmd, **kw):
    raise FileNotFoundError(cmd[0])


def make_sysfs(root, cards):
    """A fake /sys/class/drm: (card number, PCI address, vendor, device id, VRAM total bytes, VRAM used bytes); None leaves a file
    out.  cardN/device is a symlink to the PCI device, as in the real sysfs (cardN is probe order; the PCI address is the bus order)."""
    for n, bdf, vendor, dev, total, used in cards:
        pci = root / "devices" / bdf
        pci.mkdir(parents=True, exist_ok=True)
        for name, val in (("vendor", vendor), ("device", dev), ("mem_info_vram_total", total), ("mem_info_vram_used", used)):
            if val is not None:
                (pci / name).write_text(str(val))
        card = root / ("card%d" % n)
        card.mkdir(exist_ok=True)
        link = card / "device"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(pci)
    return root


def amd_det(root, cards=(RX7900XTX,), device=0, environ=None):
    make_sysfs(root, cards)
    return G.detect(device, environ=dict(NO_SMI_ENV, **(environ or {})), sysfs=str(root), run=_no_smi)


class AmdSysfsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="fake_sysfs_"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def query(self, environ=None, run=_no_smi):
        return G.amd_query(dict(NO_SMI_ENV, **(environ or {})), run=run, sysfs=str(self.root))

    def test_card(self):
        make_sysfs(self.root, [RX7900XTX])
        gs = self.query()
        self.assertEqual(len(gs), 1)
        g = gs[0]
        self.assertEqual((g.index, g.name, g.vendor, g.gfx), (0, "AMD Radeon RX 7900 XTX", "amd", "gfx1100"))
        self.assertEqual((g.total_mib, g.used_mib, g.free_mib), (24560, 1361, 23199))      # bytes -> MiB
        self.assertEqual((g.compute_cap, g.uuid, g.driver), (None, None, None))
        self.assertEqual((g.cc_text, g.arch, g.tested), ("gfx1100", "ok", True))

    def test_pci_order_and_vendor_filter(self):
        make_sysfs(self.root, [(5, "0000:83:00.0", "0x1002", "0x744c", AMD_TOTAL_MIB * G.MIB, 100 * G.MIB),
                               (2, "0000:03:00.0", "0x1002", "0x744c", 16384 * G.MIB, 0),
                               (0, "0000:00:02.0", "0x8086", "0x9a49", 0, 0),               # the Intel iGPU: not AMD
                               (7, "0000:84:00.0", "0x10de", "0x2684", 24564 * G.MIB, 0)])  # an NVIDIA card: not AMD
        conn = self.root / "card5-DP-1"                                                    # a connector points at the same device
        conn.mkdir()
        (conn / "device").symlink_to(self.root / "devices" / "0000:83:00.0")
        gs = self.query()
        self.assertEqual(len(gs), 2)                                                       # the connector is not a second card
        self.assertEqual((gs[0].index, gs[0].total_mib), (0, 16384))                       # PCI order, not cardN (probe) order
        self.assertEqual((gs[1].index, gs[1].total_mib), (1, 24560))

    def test_missing_files(self):
        make_sysfs(self.root, [(1, "0000:03:00.0", "0x1002", None, None, None),
                               (2, "0000:04:00.0", None, "0x744c", AMD_TOTAL_MIB * G.MIB, 0)])
        gs = self.query()
        self.assertEqual(len(gs), 1)                                                       # no vendor file: not known to be AMD
        g = gs[0]
        self.assertEqual((g.total_mib, g.used_mib, g.free_mib), (None, None, None))
        self.assertEqual((g.gfx, g.arch, g.tested, g.name), (None, "unknown", False, "AMD GPU"))

    def test_empty(self):
        self.assertEqual(self.query(), [])
        self.assertEqual(G.amd_query(NO_SMI_ENV, run=_no_smi, sysfs=str(self.root / "missing")), [])

    def test_rocm_smi(self):
        make_sysfs(self.root, [RX7900XTX])

        def run(cmd, **kw):
            if cmd[0] == "/x/rocm-smi":
                return subprocess.CompletedProcess(cmd, 0, stdout=ROCM_CSV % ("AMD Radeon RX 7900 XTX", "gfx1100"), stderr="")
            raise FileNotFoundError(cmd[0])
        g = self.query({"YASAKANI_ROCM_SMI": "/x/rocm-smi"}, run)[0]
        self.assertEqual((g.name, g.gfx, g.total_mib), ("AMD Radeon RX 7900 XTX", "gfx1100", 24560))

        def run_na(cmd, **kw):                                                             # Card Series N/A: the PCI-id name
            if cmd[0] == "/x/rocm-smi":
                return subprocess.CompletedProcess(cmd, 0, stdout=ROCM_CSV % ("N/A", "gfx1100"), stderr="")
            raise FileNotFoundError(cmd[0])
        g = self.query({"YASAKANI_ROCM_SMI": "/x/rocm-smi"}, run_na)[0]
        self.assertEqual((g.name, g.gfx), ("AMD Radeon RX 7900 XTX", "gfx1100"))

    def test_rocm_smi_unknown_device(self):
        make_sysfs(self.root, [(1, "0000:03:00.0", "0x1002", "0x9999", AMD_TOTAL_MIB * G.MIB, 0)])

        def run(cmd, **kw):
            if cmd[0] == "/x/rocm-smi":
                return subprocess.CompletedProcess(cmd, 0, stdout=ROCM_CSV % ("N/A", "gfx1100"), stderr="")
            raise FileNotFoundError(cmd[0])
        g = self.query({"YASAKANI_ROCM_SMI": "/x/rocm-smi"}, run)[0]
        self.assertEqual((g.name, g.gfx), ("AMD GPU 0x9999", "gfx1100"))                   # the gfx target from rocm-smi
        g = self.query()[0]                                                                # without rocm-smi: an unlisted chip
        self.assertEqual((g.name, g.gfx, g.arch, g.tested), ("AMD GPU 0x9999", None, "unknown", False))

    def test_rocm_smi_count_mismatch(self):
        make_sysfs(self.root, [RX7900XTX, (2, "0000:83:00.0", "0x1002", "0x744c", 16384 * G.MIB, 0)])

        def run(cmd, **kw):                                                                # one row for two cards: no enrichment
            if cmd[0] == "/x/rocm-smi":
                return subprocess.CompletedProcess(cmd, 0, stdout=ROCM_CSV % ("Some Other Card", "gfx900"), stderr="")
            raise FileNotFoundError(cmd[0])
        gs = self.query({"YASAKANI_ROCM_SMI": "/x/rocm-smi"}, run)
        self.assertEqual([(g.name, g.gfx) for g in gs], [("AMD Radeon RX 7900 XTX", "gfx1100")] * 2)

    def test_rocm_smi_failure(self):
        make_sysfs(self.root, [RX7900XTX])

        def boom(cmd, **kw):
            raise OSError(cmd[0])
        for run in (boom,
                    lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr=""),
                    lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="not a csv\n", stderr="")):
            g = self.query({"YASAKANI_ROCM_SMI": "/x/rocm-smi"}, run)[0]
            self.assertEqual((g.name, g.gfx, g.total_mib), ("AMD Radeon RX 7900 XTX", "gfx1100", 24560))

    def test_detect_fallback(self):
        det = amd_det(self.root)
        self.assertTrue(det.known)
        self.assertTrue(det.tested)
        self.assertTrue(det.numbered)                                                      # HIP counts in PCI order
        self.assertEqual((det.device, det.gpu.name, det.gpu.free_mib), (0, "AMD Radeon RX 7900 XTX", 23199))
        self.assertIsNone(G.untested_note(det))

    def test_detect_nvidia_wins(self):
        make_sysfs(self.root, [RX7900XTX])

        def run(cmd, **kw):
            if cmd[0] == "/x/nvidia-smi":
                return subprocess.CompletedProcess(cmd, 0, stdout=smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607) + "\n", stderr="")
            raise FileNotFoundError(cmd[0])
        det = G.detect(0, environ={"YASAKANI_NVIDIA_SMI": "/x/nvidia-smi", "YASAKANI_ROCM_SMI": "none"}, run=run, sysfs=str(self.root))
        self.assertEqual((det.gpu.vendor, det.gpu.name), ("nvidia", "NVIDIA GeForce RTX 5090"))   # the CUDA path is unchanged

    def test_detect_opt_out(self):
        for v in ("none", ""):                                                             # 0.1.3: "ask nothing" skips AMD too
            det = amd_det(self.root, environ={"YASAKANI_NVIDIA_SMI": v})
            self.assertFalse(det.known)

    def test_detect_gpus_given(self):
        def never(cmd, **kw):
            raise AssertionError("unexpected subprocess: %r" % (cmd,))
        g = G.GpuInfo(index=0, name="AMD Radeon RX 7900 XTX", total_mib=24560, used_mib=1361, vendor="amd", gfx="gfx1100")
        det = G.detect(0, environ={}, gpus=[g], run=never)
        self.assertTrue(det.tested)


class AmdDecisionTests(unittest.TestCase):
    def gpu(self, gfx="gfx1100", total=24560):
        return G.GpuInfo(index=0, name="AMD Radeon RX 7900 XTX", total_mib=total, used_mib=1361, vendor="amd", gfx=gfx)

    def det(self, g):
        return G.detect(0, environ={}, gpus=[g])

    def test_gfx1100_tested(self):
        g = self.gpu()
        self.assertTrue(g.tested)
        self.assertIn("tested and tuned", G.describe(g))
        self.assertIsNone(G.untested_note(self.det(g)))

    def test_other_gfx_no_code(self):
        g = self.gpu("gfx1101")
        self.assertEqual((g.tested, g.arch, g.cc_text), (False, "no-code", "gfx1101"))
        self.assertIn("built for gfx1100 only", G.describe(g))
        note = G.untested_note(self.det(g))
        self.assertIn("not optimized or tested there", note)
        self.assertIn("will probably not start", note)

    def test_unknown_gfx(self):
        g = self.gpu(None)
        self.assertEqual((g.tested, g.arch, g.cc_text), (False, "unknown", "?"))
        note = G.untested_note(self.det(g))
        self.assertIn("not optimized or tested there", note)
        self.assertNotIn("probably not start", note)
        self.assertIn("fastest model only", G.describe(self.gpu(None, 16384)))
        self.assertIn("too small", G.describe(self.gpu(None, 8192)))


class AmdSelectTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="fake_sysfs_"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_one_gpu(self):
        det = amd_det(self.root)
        gs = det.gpus
        self.assertEqual(G.select(gs, 0, {}), gs)
        self.assertEqual(G.select(gs, 1, {}), [])
        self.assertEqual(G.select(gs, 0, {"CUDA_VISIBLE_DEVICES": "1"}), gs)               # HIP does not read CUDA's variables
        self.assertEqual(G.select(gs, 0, {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}), gs)

    def test_two_gpus(self):
        cards = (RX7900XTX, (2, "0000:83:00.0", "0x1002", "0x744c", 16384 * G.MIB, 0))
        det = amd_det(self.root, cards)
        self.assertTrue(det.numbered)
        self.assertEqual(G.select(det.gpus, 1, {})[0].total_mib, 16384)                    # --device N is GPU N (PCI order)
        self.assertEqual(G.select(det.gpus, 2, {}), [])
        self.assertEqual(amd_det(self.root, cards, device=1).gpu.total_mib, 16384)
        self.assertEqual(amd_det(self.root, cards, device=0, environ={"CUDA_VISIBLE_DEVICES": "1"}).gpu.total_mib, 24560)


class AmdPlanTests(unittest.TestCase):
    """The context rule with AMD VRAM: the same numbers as on a 24 GB NVIDIA card (the rule is vendor-neutral)."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="fake_sysfs_"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def det(self, total=AMD_TOTAL_MIB, used=AMD_USED_MIB):
        return amd_det(self.root, [(1, "0000:03:00.0", "0x1002", "0x744c", total * G.MIB, used * G.MIB)])

    def test_24gb_full_context(self):
        for sizes in (FASTEST, QUALITY):                                                   # 23,199 MiB free: both fit at 262,144
            plan = G.plan_context(FakePack(sizes), self.det())
            self.assertEqual((plan.max_ctx, plan.env, plan.notes), (None, {}, []))

    def test_24gb_less_free(self):
        plan = G.plan_context(FakePack(QUALITY), self.det(used=3996))                      # 20,564 MiB free, as the RTX 4090 test
        self.assertEqual((plan.max_ctx, plan.env), (163840, {}))

    def test_16gb(self):
        plan = G.plan_context(FakePack(FASTEST), self.det(total=16384, used=1000))
        self.assertEqual((plan.max_ctx, plan.env), (81920, {"YSKN_BDR_SHEAD": "0"}))

    def test_does_not_fit(self):
        with self.assertRaises(YasakaniError) as cm:
            G.plan_context(FakePack(FASTEST), self.det(total=8192, used=300))
        self.assertIn("does not fit", str(cm.exception))
        self.assertIn("(7892 of 8192 MiB free", str(cm.exception))

    def test_given_context(self):
        plan = G.plan_given_context(FakePack(FASTEST), self.det(), 131072)                 # 128K: the verified HIP context
        self.assertEqual((plan.max_ctx, plan.env, plan.notes), (131072, {}, []))
        self.assertEqual(G.plan_given_context(FakePack(QUALITY), self.det(), 262144).env, {})
        plan = G.plan_given_context(FakePack(FASTEST), self.det(used=20000), 262144)       # 4,560 MiB free: the head is left out
        self.assertEqual((plan.max_ctx, plan.env), (262144, {"YSKN_BDR_SHEAD": "0"}))
        self.assertIn("bdraft2_k8.shead", plan.notes[0])

    def test_unknown_or_big_memory(self):
        make_sysfs(self.root, [(1, "0000:03:00.0", "0x1002", "0x744c", None, None)])
        det = G.detect(0, environ=NO_SMI_ENV, sysfs=str(self.root), run=_no_smi)
        self.assertEqual(G.plan_context(FakePack(FASTEST), det).max_ctx, None)             # unknown: unchanged
        self.assertEqual(G.plan_given_context(FakePack(FASTEST), det, 262144).env, {})
        self.assertEqual(G.plan_context(FakePack(QUALITY), self.det(total=32768)).max_ctx, None)   # 30,000 MiB or more: unchanged


@unittest.skipUnless(support.tokenizer_dir() is not None and os.name == "posix",
                     "needs the Qwen3.8 tokenizer files (YASAKANI_TEST_MODEL_DIR) and a POSIX test machine")
class CliAmdTests(unittest.TestCase):
    """The command line with a fake amdgpu sysfs tree and the stub engine library (the HIP path, no NVIDIA driver)."""

    @classmethod
    def setUpClass(cls):
        cls.dll = str(support.stub_path())
        cls.folder = support.make_real_folder("amd")
        cls.addClassCleanup(shutil.rmtree, cls.folder.parent, True)
        with open(cls.folder / "bdraft2_k8.shead", "wb") as f:                # a sparse file of the published size: the rule sees it
            f.truncate(FASTEST[2])
        cls.sysfs = Path(tempfile.mkdtemp(prefix="fake_sysfs_cli_"))
        cls.addClassCleanup(shutil.rmtree, cls.sysfs, True)

    def set_vram(self, total=AMD_TOTAL_MIB, used=AMD_USED_MIB, dev="0x744c"):
        make_sysfs(self.sysfs, [(1, "0000:03:00.0", "0x1002", dev, total * G.MIB, used * G.MIB)])

    def run_cli(self, *args, timeout=60):
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", YASAKANI_AMD_SYSFS=str(self.sysfs), **NO_SMI_ENV)
        for k in [k for k in env if k.startswith(("YSKN_", "STUB_"))] + ["YASAKANI_DLL", "YASAKANI_MODEL_DIR", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"]:
            env.pop(k, None)
        return subprocess.run([sys.executable, "-m", "yasakani", *args], capture_output=True, text=True, timeout=timeout, cwd=str(ROOT),
                              env=env, input="/exit\n")

    def common(self):
        return ["--model-dir", str(self.folder), "--dll", self.dll]

    def test_info(self):
        self.set_vram()
        r = self.run_cli("info", *self.common())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("AMD Radeon RX 7900 XTX, 24560 MiB, 1361 MiB", r.stdout)
        self.assertIn("GPU 0: gfx1100 - the chip the HIP backend is tested and tuned on", r.stdout)

    def test_chat_full_context(self):
        self.set_vram()
        r = self.run_cli("chat", *self.common(), "--no-think")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(context 262144 tokens)", r.stderr)
        self.assertNotIn("note:", r.stderr)                                                # gfx1100 is the tested chip
        self.assertNotIn("context: ", r.stderr)

    def test_chat_context_and_note(self):
        self.set_vram(used=AMD_TOTAL_MIB - 4958)                                           # about 4,958 MiB free beside the weights
        r = self.run_cli("chat", *self.common(), "--no-think")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("bdraft2_k8.shead", r.stderr)
        self.assertIn("context: 98304 tokens", r.stderr)
        self.assertIn("(context 98304 tokens)", r.stderr)
        self.assertNotIn("not optimized", r.stderr)

    def test_does_not_fit(self):
        self.set_vram(used=AMD_TOTAL_MIB - 2000)
        r = self.run_cli("chat", *self.common())
        self.assertEqual(r.returncode, 1)
        self.assertIn("does not fit in the free GPU memory", r.stderr)
        self.assertNotIn("loading the model", r.stderr)

    def test_given_context(self):
        self.set_vram()
        r = self.run_cli("chat", *self.common(), "--no-think", "--max-ctx", "131072")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(context 131072 tokens)", r.stderr)
        self.assertNotIn("context: ", r.stderr)

    def test_untested_card(self):
        self.set_vram(dev="0x9999")                                                        # a chip the fallback does not know
        r = self.run_cli("chat", *self.common(), "--no-think")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("the HIP backend runs on it, but it is not optimized or tested there", r.stderr)
        r = self.run_cli("info", *self.common())
        self.assertIn("GPU 0: ? - runs with the HIP backend", r.stdout)


if __name__ == "__main__":
    unittest.main()
