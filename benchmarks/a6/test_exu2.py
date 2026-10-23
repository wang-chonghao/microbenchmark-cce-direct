"""Structural tests for the EXU2 supplement; no simulator is required."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .exu2 import DEFAULT, SNAPSHOT, ROOT, capture, emit, generate


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((DEFAULT / "manifest.json").read_text())
        self.snapshot = json.loads(SNAPSHOT.read_text())
        self.baseline = json.loads((ROOT / "op_kernel/a6_instruction/manifest.json").read_text())

    def test_every_exu2_form_has_a_disposition(self):
        expected = {(op, form) for op, item in self.snapshot["instructions"].items() for form in item["forms"]}
        rows = [r for r in self.manifest["coverage"] if r["status"] != "native_stage_extra"]
        self.assertEqual(expected, {(r["op"], r["form"]) for r in rows})
        self.assertEqual(len(rows), len(expected))

    def test_no_repeated_baseline_probes(self):
        existing = {c["name"] for c in self.baseline["cases"]}
        names = [c["name"] for c in self.manifest["cases"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertFalse(existing & set(names))

    def test_checked_in_sources_match_generator(self):
        for case in self.manifest["cases"]:
            with self.subTest(case=case["name"]):
                meta, text = emit(case["op"], case["form"], case["mode"])
                self.assertEqual(meta, case)
                data = (DEFAULT / case["source"]).read_bytes()
                self.assertEqual(data, text.encode())
                self.assertEqual(hashlib.sha256(data).hexdigest(), case["source_sha256"])
                self.assertEqual(case["expected_target_count"], case["width"] * 2)

    def test_packed_float_transport_is_bits_not_numeric_conversion(self):
        meta, source = emit("VSQZ", "f8e4m3", "ii")
        self.assertEqual(meta["input_dtype"], "uint32")
        self.assertEqual(meta["semantic_input_dtype"], "f8e4m3")
        self.assertIn("(vector_f8e4m3)bits_a0", source)
        self.assertIn("(vector_u32)d0", source)
        self.assertNotIn("vcvt(", source)

    def test_merge_and_accumulator_initialization(self):
        meta, code = emit("VDUP_HIGHEST_MERGE", "fp32", "forwarding")
        self.assertTrue(meta["implicit_dst_read"])
        self.assertIn("d0 = (vector_f32)bits_b0", code)
        self.assertIn("vdup(d2, d0, mask, POS_HIGHEST, MODE_MERGING)", code)
        self.assertEqual(meta["target_aliases"], ["RV_VDUPM"])
        _, code = emit("VUSQZ", "int16", "ii")
        self.assertIn("vbr(d0, (uint16_t)0)", code)
        self.assertIn("vcmp_lt(count_mask0, a0, b0, m)", code)

    def test_compound_and_native_are_not_conflated(self):
        meta, code = emit("VINTLV", "f8e8m0", "ii")
        self.assertTrue(meta["compound"])
        self.assertIn("vintlvv2(e0, a0, b0, HIGHER)", code)
        native, _ = emit("VINTLVV2", "f8e8m0", "forwarding")
        self.assertFalse(native["compound"])
        self.assertEqual(native["expected_target_count"], 8)
        self.assertEqual(native["target_aliases"], ["RV_VINTLVV2"])

    def test_capture_uses_form_routing_and_not_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "isa.json"
            path.write_text(json.dumps({"instructions": {"X": {
                "eligible_exus": ["EXU2"], "forms": {
                    "fp32": {"latency": 999}, "fp16": {"eligible_exus": ["EXU0"]}}}}}))
            data = capture(path)
            self.assertEqual(data["instructions"], {"X": {"forms": {"fp32": {}}}})

    def test_regeneration_preserves_hand_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            snap = dict(source_sha256="test", instructions={"VLN": {"forms": {"fp16": {}}}})
            manifest = generate(snap, out, {"cases": []})
            path = out / manifest["cases"][0]["source"]
            path.write_text("hand edit")
            with self.assertRaisesRegex(ValueError, "Hand-edited"):
                generate(snap, out, {"cases": []})
            self.assertEqual(path.read_text(), "hand edit")


    def test_invalid_iteration_count(self):
        for count in (0, 5):
            with self.assertRaises(ValueError):
                emit("VINTLV", "fp32", "ii", iterations=count)


if __name__ == "__main__":
    unittest.main()
