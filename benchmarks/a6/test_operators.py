import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from benchmarks.a6.check_softmax import check
from run_a6_operators import DEFAULT_CASES, prepare_input


class OperatorTests(unittest.TestCase):
    def test_manifest_and_layout(self):
        manifest = json.loads((DEFAULT_CASES / "manifest.json").read_text())
        self.assertEqual(len(manifest["cases"]), 8)
        for case in manifest["cases"]:
            source = (DEFAULT_CASES / case["source"]).read_bytes()
            self.assertEqual(hashlib.sha256(source).hexdigest(), case["source_sha256"])
            self.assertLess(case["ub_reserved_bytes"], manifest["ub_budget_bytes"])
            regions = case["ub_regions"]
            for first, second in zip(regions, regions[1:]):
                self.assertLessEqual(first["end"], second["start"])
            self.assertEqual(source.count(b"mem_bar("), case["static_membar_count"])
            self.assertIn(b"nddma_out_to_ub", source)
            self.assertIn(b"scatter_ubuf_to_gm", source)

    def test_golden_rejects_corrupt_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp, out = root / "input.bin", root / "output.bin"
            inp.write_bytes(struct.pack("<1024f", *([0.0] * 1024)))
            out.write_bytes(struct.pack("<1024e", *([1 / 1024] * 1024)))
            self.assertTrue(check(inp, out, rows=1)["passed"])
            out.write_bytes(struct.pack("<1024e", *([0.0] * 1024)))
            self.assertFalse(check(inp, out, rows=1)["passed"])
            out.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                check(inp, out, rows=1)

    def test_inputs_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            prepare_input(path, 8, 12)
            original = (path / "input_input.bin").read_bytes()
            prepare_input(path, 8, 12)
            self.assertEqual(original, (path / "input_input.bin").read_bytes())
            self.assertEqual(len(original), 8 * 1024 * 4)


if __name__ == "__main__":
    unittest.main()
