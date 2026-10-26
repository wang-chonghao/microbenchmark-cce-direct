"""Inventory, generator and conservative measurement checks for A5 migration."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .coverage_a5 import OUT, SNAPSHOT, ROOT, conversion, emit, generate, spec
from .raw_wrapper import wrap
from .analyze import analyze
from .archive import archive_dumps
from .test_suite import line


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((OUT / "manifest.json").read_text())

    def test_every_source_form_has_exactly_one_disposition(self):
        snapshot = json.loads(SNAPSHOT.read_text())
        expected = {(op, form) for op, v in snapshot["instructions"].items() for form in v["forms"]}
        rows = self.manifest["coverage"]
        self.assertEqual(len(expected), len(rows))
        self.assertEqual(expected, {(r["op"], r["form"]) for r in rows})
        self.assertEqual(snapshot["source_commit"], self.manifest["source_commit"])
        names = {c["name"] for c in self.manifest["cases"]}
        self.assertEqual(len(names), len(self.manifest["cases"]))
        self.assertEqual(names, {n for r in rows for n in r["cases"]})
        for row in rows:
            if row["status"] == "deferred":
                self.assertTrue(row["reason"])
                self.assertEqual(row["cases"], [])

    def test_no_duplicate_existing_forms(self):
        old = {(c["op"], c["form"]) for directory in ("a6_instruction", "a6_exu2")
               for c in json.loads((ROOT / "op_kernel" / directory / "manifest.json").read_text())["cases"]}
        new = {(c["op"], c["form"]) for c in self.manifest["cases"]}
        self.assertFalse(old & new)

    def test_sources_and_hashes_match_generator(self):
        for case in self.manifest["cases"]:
            with self.subTest(name=case["name"]):
                meta, text = emit(case["op"], case["form"], case["mode"])
                self.assertEqual(case, meta)
                self.assertEqual((OUT / case["source"]).read_text(), text)
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), case["source_sha256"])
                self.assertEqual(case["expected_target_count"], case["width"] * case["loop_iterations"])
                self.assertEqual(case["golden"], "not_implemented")

    def test_two_three_step_raw_chains(self):
        meta, code = emit("VADD", "int32", "forwarding")
        self.assertEqual(meta["width"], 6)
        self.assertEqual(meta["forwarding_edges"], [[0, 2], [1, 3], [2, 4], [3, 5]])
        for consumer in range(2, 6):
            self.assertIn(f"vadd(d{consumer}, d{consumer-2}, b{consumer}, m)", code)
        self.assertIn("d0 =", emit("VMADD", "fp32", "ii")[1])

    def test_eight_independent_ii_targets(self):
        meta, code = emit("VADD", "int32", "ii")
        self.assertEqual(meta["width"], 8)
        for i in range(8):
            self.assertIn(f"vadd(d{i}, a{i}, b{i}, m)", code)
        self.assertEqual(code.count("// TARGET"), 8)
        self.assertIn("vmuls(d0, a0, (int32_t)3, m)", emit("VMULS", "int32", "ii")[1])

    def test_compare_and_carry_outputs_are_observable(self):
        _, code = emit("VCMP_EQ", "int16", "ii")
        self.assertIn("vcmp_eq(p0, a0, b0, m)", code)
        self.assertIn("psts(p0, ubA", code)
        _, code = emit("VADDC", "int32", "ii")
        self.assertIn("vaddc(carry0, d0, a0, b0, m)", code)
        self.assertIn("psts(carry0, ubA", code)

    def test_alignment_state_is_local_to_operation_pair(self):
        for op in ("VLDAS", "VLDUS", "VSTUS", "VSTAS"):
            _, code = emit(op, "uint32", "ii")
            self.assertLess(code.index("// TARGET 0"), code.index("vector_align state0"))
            self.assertLess(code.index("vector_align state0"), code.index("// TARGET 1"))

    def test_raw_transport_does_not_add_conversions(self):
        _, code = emit("VADD", "fp16", "ii")
        self.assertIn("a0 = (vector_f16)ba0", code)
        self.assertNotIn("vcvt(", code)
        self.assertEqual(conversion("bf16_to_s32")[2], "m, ROUND_R, RS_ENABLE, PART_EVEN")
        with self.assertRaises(ValueError):
            spec("VLD", "s4x2")

    def test_ub_regions_cannot_overlap(self):
        for size in (0, 4097):
            with self.assertRaises(ValueError):
                wrap("invalid", "", size, 1)
            with self.assertRaises(ValueError):
                wrap("invalid", "", 1, size)

    def test_regeneration_preserves_hand_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = dict(source_commit="test", source_sha256="test",
                            instructions={"VCMP_EQ": {"forms": {"int32": {}}}})
            out = Path(tmp)
            manifest = generate(snapshot, out)
            path = out / manifest["cases"][0]["source"]
            path.write_text("hand edit")
            with self.assertRaisesRegex(ValueError, "Hand-edited"):
                generate(snapshot, out)
            self.assertEqual(path.read_text(), "hand edit")

    def test_explicit_compile_rejection_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = dict(source_commit="test", source_sha256="test",
                            instructions={"VCMP_EQ": {"forms": {"int32": {}}}})
            manifest = generate(snapshot, Path(tmp), {"VCMP_EQ/int32": "compiler diagnostic"})
            self.assertEqual(manifest["cases"], [])
            self.assertEqual(manifest["coverage"][0]["status"], "deferred")

    def test_new_memory_op_does_not_report_exu_ii(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "run/log_ca"
            logs.mkdir(parents=True)
            meta, _ = emit("VGATHER2", "fp32", "ii")
            count = meta["expected_target_count"]
            for suffix, delay in (("instr_popped_log", 0), ("instr_log", 6)):
                (logs / f"core0.veccore0.{suffix}.dump").write_text("".join(
                    line(10 + i + delay, i, 0x100 + i * 4, "RV_VGATHER2") for i in range(count)))
            record = dict(case=meta, returncode=0, compile_only=False)
            record["dump_archive"] = archive_dumps(root, root / "results", meta)
            result = analyze(root, record)
            self.assertEqual(result["status"], "not_collected")
            self.assertEqual(result["latency_min"], 6)
            self.assertIsNone(result["ii_min"])
            self.assertIsNone(result["memory_interval_min"])


if __name__ == "__main__":
    unittest.main()
