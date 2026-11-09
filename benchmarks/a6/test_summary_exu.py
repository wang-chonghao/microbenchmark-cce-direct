"""Expose observed EXU IDs without inventing a hardware eligibility set."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_a6_bench import summarize


class SummaryExuTests(unittest.TestCase):
    def check_summary(self, streams, expected, cached=False):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run = out / "_work/test/r0"
            run.mkdir(parents=True)
            (run / "run_record.json").write_text(json.dumps({"case": {"name": "test"}}))
            result = dict(status="observed", golden="not_implemented", hardware_calibrated=False,
                          streams=streams, latency_min=6, ii_min=2, forwarding_min=None,
                          memory_interval_min=None)
            (run / "measurement.json").write_text(json.dumps(result))
            with patch("run_a6_bench.analyze", return_value=result) as analyze:
                self.assertEqual(summarize(out, quiet=True, use_cache=cached), 0)
                self.assertEqual(analyze.call_count, 0 if cached else 1)
            row = json.loads((out / "summary.json").read_text())[0]
            self.assertEqual(row["observed_exus"], expected)
            self.assertEqual(row["exu_evidence"], "observed" if expected else "unavailable")
            self.assertFalse(row["hardware_calibrated"])
            with (out / "summary.csv").open() as file:
                csv_row = next(csv.DictReader(file))
            self.assertEqual(csv_row["observed_exus"], "|".join(f"EXU{x}" for x in expected))
            self.assertEqual(csv_row["exu_evidence"], row["exu_evidence"])

    def test_union_is_sorted_and_deduplicated(self):
        self.check_summary([{"observed_target_exus": [2, 0]}, {"observed_target_exus": [2]}], [0, 2])

    def test_exu_zero_is_not_missing(self):
        self.check_summary([{"observed_target_exus": [0]}], [0])

    def test_missing_evidence_is_not_exu_zero(self):
        self.check_summary([{}], [])

    def test_old_cached_measurement_is_supported(self):
        self.check_summary([{"observed_target_exus": [2]}], [2], cached=True)


if __name__ == "__main__":
    unittest.main()
