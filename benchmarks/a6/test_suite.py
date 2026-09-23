"""Run with: python3 -m unittest benchmarks.a6.test_suite -v"""

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import sys
import unittest
from unittest.mock import patch

from .analyze import analyze, numeric_check, parse_events, parse_memory
from .generate import DEFAULT_CASES, generate, render
from .archive import archive_dumps


def line(cycle, ident, pc, op="RV_VADD", regs="Vd[2], Vn[0], Vm[1]"):
    return f"[info] [{cycle:08d}] (PC: 0x{pc:x}) RVECEX : (ID: {ident:06d}) {op} {regs}\n"


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logs = self.root / "run/log_ca"
        self.logs.mkdir(parents=True)
        self.prefix = "core0.veccore0"

    def dump(self, suffix, content):
        path = self.logs / (self.prefix + suffix)
        path.write_text(content)
        return path

    def record(self, mode="latency", width=1):
        meta, _ = render("VADD", "fp32", mode, width)
        return dict(case=meta, returncode=0, compile_only=False)

    def measure(self, record):
        with patch("benchmarks.a6.analyze.numeric_check", return_value=("pass", "")):
            return analyze(self.root, record)

    def test_pair_by_id_pc_and_opcode_not_line_order(self):
        self.dump(".instr_popped_log.dump", line(10, 3, 0x100))
        self.dump(".instr_log.dump", line(17, 9, 0x104) + line(16, 3, 0x100))
        r = self.measure(self.record())
        self.assertEqual(r["latency_min"], 6)

    def test_wrong_pc_never_pairs(self):
        self.dump(".instr_popped_log.dump", line(10, 3, 0x100))
        self.dump(".instr_log.dump", line(17, 3, 0x104))
        self.assertIsNone(self.measure(self.record())["latency_min"])

    def test_duplicate_done_is_ambiguous(self):
        self.dump(".instr_popped_log.dump", line(10, 3, 0x100))
        self.dump(".instr_log.dump", line(17, 3, 0x100) * 2)
        self.assertIsNone(self.measure(self.record())["latency_min"])

    def test_stars_only_is_missing(self):
        self.dump(".stars_log_task.dump", "hello\n")
        self.assertEqual(self.measure(self.record())["status"], "missing_logs")

    def test_no_global_ii_fallback(self):
        self.dump(".instr_popped_log.dump", line(10, 3, 0x100) + line(11, 4, 0x104, regs="Vd[5], Vn[3], Vm[4]"))
        self.assertIsNone(self.measure(self.record("ii", 2))["ii_min"])

    def ii_setup(self, exu0, exu1, dependent=False):
        self.dump(".instr_popped_log.dump", line(10, 3, 0x100) + line(12, 4, 0x104,
                  regs="Vd[5], Vn[2], Vm[4]" if dependent else "Vd[5], Vn[3], Vm[4]"))
        self.dump(".rvec.EXU.dump", f"[info] [0010] EXU instr_name RV_VADD instr_id 3 exu_id:{exu0}\n"
                  f"[info] [0012] EXU instr_name RV_VADD instr_id 4 exu_id:{exu1}\n")

    def test_same_exu_ii(self):
        self.ii_setup(0, 0)
        self.assertEqual(self.measure(self.record("ii", 2))["ii_min"], 2)

    def test_distinct_exu_not_ii(self):
        self.ii_setup(0, 1)
        self.assertIsNone(self.measure(self.record("ii", 2))["ii_min"])

    def test_missing_register_fields_not_assumed_independent(self):
        self.ii_setup(0, 0)
        self.dump(".instr_popped_log.dump", line(10, 3, 0x100, regs="") + line(12, 4, 0x104, regs=""))
        self.assertIsNone(self.measure(self.record("ii", 2))["ii_min"])

    def test_dependent_pair_not_ii(self):
        self.ii_setup(0, 0, True)
        self.assertIsNone(self.measure(self.record("ii", 2))["ii_min"])

    def test_raw_forwarding(self):
        self.dump(".instr_popped_log.dump", line(10, 3, 0x100) + line(15, 4, 0x104, regs="Vd[3], Vn[2], Vm[1]"))
        self.assertEqual(self.measure(self.record("forwarding"))["forwarding_min"], 5)

    def test_intervening_register_writer_breaks_raw(self):
        self.dump(".instr_popped_log.dump", line(10, 3, 0x100) +
                  line(11, 9, 0x104, "RV_VLDI", "Vd[2]") +
                  line(15, 4, 0x108, regs="Vd[3], Vn[2], Vm[1]"))
        self.assertIsNone(self.measure(self.record("forwarding"))["forwarding_min"])

    def test_optimized_target_rejected(self):
        self.dump(".instr_popped_log.dump", line(10, 3, 0x100))
        r = self.measure(self.record("forwarding"))
        self.assertIsNone(r["forwarding_min"])
        self.assertIn("Target count", r["reason"])

    def test_lsu_stage_parser_ignores_stalls_and_retirement(self):
        path = self.dump(".rvec.LSU.dump", "[info] [0010] [STALL.SEND] [LDU], instr_id: 2\n"
            "[info] [0011] [SEND_UB_RD.PORT_0] bytes_per_mask 1 instr_name RV_VLDI instr_id 2\n"
            "[info] [0017] SEND_RELEASE.LD instr_name RV_VLDI instr_id 2\n")
        events = parse_memory(path)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["cycle"], 11)

    def test_compile_only_is_not_measured(self):
        record = self.record()
        record["compile_only"] = True
        r = self.measure(record)
        self.assertEqual(r["status"], "compiled")
        self.assertIsNone(r["latency_min"])

    def test_fp16_chain_checks_intermediate_rounding(self):
        meta, _ = render("VADD", "fp16", "forwarding", 2)
        n = meta["dst_lanes"]
        for name, values in (("input_B.bin", [1.] * (2*n)), ("input_C.bin", [2.] * (2*n)),
                             ("output_A.bin", [3.] * n + [5.] * n)):
            (self.root / "run" / name).write_bytes(struct.pack("<" + "e" * len(values), *values))
        self.assertEqual(numeric_check(self.root, meta)[0], "pass")
        (self.root / "run/output_A.bin").write_bytes(b"\xcd" * (4*n))
        self.assertEqual(numeric_check(self.root, meta)[0], "fail")

    def test_manifest_covers_inventory_and_hashes_sources(self):
        inv = json.loads(Path(__file__).with_name("isa_inventory.json").read_text())
        manifest = json.loads((DEFAULT_CASES / "manifest.json").read_text())
        expected = {(o, f) for o, item in inv["instructions"].items() for f in item["forms"]}
        actual = {(c["op"], c["form"]) for c in manifest["coverage"] if not c["a6_native_extra"]}
        self.assertEqual(expected, actual)
        for c in manifest["cases"]:
            self.assertEqual(hashlib.sha256((DEFAULT_CASES / c["source"]).read_bytes()).hexdigest(), c["source_sha256"])
            self.assertLessEqual(c["width"] * 256, 0x4000)

    def test_conversion_has_no_fake_self_forwarding(self):
        manifest = json.loads((DEFAULT_CASES / "manifest.json").read_text())
        for c in manifest["coverage"]:
            if c["op"].startswith("VCVT_"):
                self.assertEqual(c["forwarding"], "not_applicable")

    def test_log_chain_uses_positive_domain_input(self):
        meta, source = render("VLN", "fp32", "forwarding", 2)
        self.assertIn("vlds(a0, ubC,", source)
        import math
        values = [math.log(2.)] * 64 + [math.log(math.log(2.))] * 64
        for name, data in (("input_B.bin", [.75] * 128), ("input_C.bin", [2.] * 128), ("output_A.bin", values)):
            (self.root / "run" / name).write_bytes(struct.pack("<" + "f" * len(data), *data))
        self.assertEqual(numeric_check(self.root, meta)[0], "pass")

    def test_unknown_opcode_template_rejected(self):
        with self.assertRaises(ValueError):
            render("NOT_A_REVIEWED_ISA", "fp32", "latency", 1)

    def test_subprocess_timeout(self):
        from run_a6_bench import run_command
        code = run_command([sys.executable, "-c", "import time; time.sleep(60)"],
                           self.root / "console.log", .1, os.environ.copy())
        self.assertEqual(code, 124)

    def test_compound_does_not_report_one_instruction_latency(self):
        meta, _ = render("VCMAX", "fp32", "latency", 1)
        self.dump(".instr_popped_log.dump", line(10, 3, 0x100, "RV_VCGMAXV2"))
        r = self.measure(dict(case=meta, returncode=0, compile_only=False))
        self.assertEqual(r["status"], "compound_api")
        self.assertIsNone(r["latency_min"])

    def test_archive_only_six_kinds_and_only_veccore0(self):
        for suffix in ("instr_popped_log", "instr_log", "rvec.EXU", "rvec.IDU", "rvec.ISU", "rvec.OOO"):
            self.dump("." + suffix + ".dump", "kept\n")
        unwanted = self.dump(".rvec.LSU.dump", "discarded")
        (self.logs / "core0.veccore1.instr_log.dump").write_text("discarded")
        config = self.logs / "config.json"
        config.write_text("{}")
        out = self.root / "results"
        record = archive_dumps(self.root, out, self.record()["case"])
        self.assertEqual(len(record["files"]), 6)
        self.assertEqual(record["missing_or_empty"], [])
        self.assertEqual(record["removed_files"], 2)
        self.assertFalse(unwanted.exists())
        self.assertTrue(config.exists())
        self.assertEqual(len(list(out.rglob("*.dump"))), 6)
        self.assertTrue(all(p.name.startswith("vadd_fp32_latency__") for p in out.rglob("*.dump")))

    def test_archive_can_be_reanalyzed(self):
        self.dump(".instr_popped_log.dump", line(10, 3, 0x100))
        self.dump(".instr_log.dump", line(16, 3, 0x100))
        record = self.record()
        record["dump_archive"] = archive_dumps(self.root, self.root / "results", record["case"])
        self.assertEqual(self.measure(record)["latency_min"], 6)

    def test_excluded_lsu_is_not_a_model_failure(self):
        meta, _ = render("VLDS", "fp32", "ii", 2)
        self.dump(".instr_popped_log.dump", line(10, 3, 0x100, "RV_VLDI") + line(11, 4, 0x104, "RV_VLDI"))
        self.dump(".instr_log.dump", line(16, 3, 0x100, "RV_VLDI") + line(17, 4, 0x104, "RV_VLDI"))
        record = dict(case=meta, returncode=0, compile_only=False)
        record["dump_archive"] = archive_dumps(self.root, self.root / "results", meta)
        result = self.measure(record)
        self.assertEqual(result["status"], "not_collected")
        self.assertIsNone(result["memory_interval_min"])
        self.assertEqual(result["latency_min"], 6)

    def test_archive_preserves_rotations_but_excludes_other_cores(self):
        self.dump(".instr_log.dump.1", "rotated")
        self.dump(".rvec.EXU.1.dump", "rotated")
        (self.logs / "core1.veccore0.instr_log.dump").write_text("core1")
        result = archive_dumps(self.root, self.root / "results", self.record()["case"])
        self.assertEqual(len(result["files"]), 2)
        self.assertFalse((self.logs / "core1.veccore0.instr_log.dump").exists())

    def test_archive_collision_prevents_pruning(self):
        original = self.dump(".instr_log.dump", "first")
        nested = self.logs / "nested"
        nested.mkdir()
        (nested / original.name).write_text("second")
        unwanted = self.dump(".rvec.LSU.dump", "unwanted")
        with self.assertRaises(FileExistsError):
            archive_dumps(self.root, self.root / "results", self.record()["case"])
        self.assertTrue(original.exists())
        self.assertTrue(unwanted.exists())

    def test_archive_does_not_follow_symlinks(self):
        outside = self.root / "unrelated"
        outside.mkdir()
        target = outside / "something.dump"
        target.write_text("untouched")
        (self.logs / "linked").symlink_to(outside, target_is_directory=True)
        (self.logs / "alias.dump").symlink_to(target)
        archive_dumps(self.root, self.root / "results", self.record()["case"])
        self.assertEqual(target.read_text(), "untouched")
        self.assertTrue((self.logs / "alias.dump").is_symlink())

    def test_keep_all_retains_extra_dumps(self):
        target = self.dump(".rvec.LSU.dump", "kept")
        record = archive_dumps(self.root, self.root / "results", self.record()["case"], keep_all=True)
        self.assertTrue(target.exists())
        self.assertEqual(record["removed_files"], 0)

    def test_batch_dual_packages_archive_and_reanalysis(self):
        from run_a6_bench import main
        cann = self.root / "compiler"
        cann.mkdir()
        (cann / "set_env.sh").write_text("")
        simulator = self.root / "debug_simulator"
        (simulator / "dav_9201/camodel").mkdir(parents=True)
        output = self.root / "results"

        def fake_model(command, console, timeout, env):
            self.assertEqual(env["CANN_HOME"], str(cann))
            self.assertEqual(env["FULL_SIMULATOR_HOME"], str(simulator))
            run = Path(command[command.index("--output") + 1])
            logs = run / "run/log_ca"
            logs.mkdir(parents=True)
            regs = ["Vd[2], Vn[0], Vm[1]", "Vd[5], Vn[3], Vm[4]",
                    "Vd[6], Vn[2], Vm[7]", "Vd[8], Vn[5], Vm[9]"]
            starts, dones = [], []
            for group in range(2):
                for i in range(4):
                    cycle = 10 + group * 30 + (i // 2) * 5
                    starts.append(line(cycle, group * 4 + i, 0x100 + i * 4, regs=regs[i]))
                    dones.append(line(cycle + 6, group * 4 + i, 0x100 + i * 4, regs=regs[i]))
            (logs / "core0.veccore0.instr_popped_log.dump").write_text("".join(starts))
            (logs / "core0.veccore0.instr_log.dump").write_text("".join(dones))
            for kind in ("EXU", "IDU", "ISU", "OOO", "LSU"):
                (logs / f"core0.veccore0.rvec.{kind}.dump").write_text("test\n")
            return 0

        argv = ["run_a6_bench.py", "--ops", "VADD", "--forms", "fp32", "--modes", "forwarding",
                "--repeats", "1", "--cann-home", str(cann), "--full-simulator-home", str(simulator),
                "--output", str(output)]
        with patch.object(sys, "argv", argv), patch("run_a6_bench.run_command", side_effect=fake_model), \
                patch("benchmarks.a6.analyze.numeric_check", return_value=("pass", "")), patch("builtins.print"):
            self.assertEqual(main(), 0)
        self.assertEqual(len(list((output / "VADD").rglob("*.dump"))), 6)
        self.assertFalse(list((output / "_work").rglob("*.dump")))
        with patch.object(sys, "argv", ["run_a6_bench.py", "--analyze-only", "--output", str(output)]), \
                patch("benchmarks.a6.analyze.numeric_check", return_value=("pass", "")), patch("builtins.print"):
            self.assertEqual(main(), 0)
        rows = json.loads((output / "summary.json").read_text())
        self.assertEqual(rows[0]["latency_min"], 6)
        self.assertEqual(rows[0]["forwarding_min"], 5)

    def test_compact_manifest_has_only_two_test_kinds(self):
        manifest = json.loads((DEFAULT_CASES / "manifest.json").read_text())
        for c in manifest["cases"]:
            self.assertIn(c["mode"], {"ii", "forwarding"})
            self.assertEqual(c["expected_target_count"], c["width"] * c["loop_iterations"])
            if c["mode"] == "ii":
                self.assertEqual(c["width"], 8)
            else:
                self.assertEqual(c["forwarding_edges"], [[0, 2], [1, 3]])
        self.assertFalse(list(DEFAULT_CASES.glob("*_latency.cce")))
        self.assertFalse(list(DEFAULT_CASES.glob("*_ii_w*.cce")))

    def test_unroll2_golden_tracks_two_chains_and_resets_each_group(self):
        meta, _ = render("VADD", "fp32", "forwarding", 2, iterations=2)
        n = meta["dst_lanes"]
        data = [3.] * (2*n) + [5.] * (2*n)
        for name, values in (("input_B.bin", [1.] * (8*n)), ("input_C.bin", [2.] * (8*n)),
                             ("output_A.bin", data * 2)):
            (self.root / "run" / name).write_bytes(struct.pack("<" + "f" * len(values), *values))
        self.assertEqual(numeric_check(self.root, meta)[0], "pass")

    def test_zero_timeout_waits_for_process_completion(self):
        from run_a6_bench import run_command
        code = run_command([sys.executable, "-c", "print('completed')"],
                           self.root / "console.log", 0, os.environ.copy())
        self.assertEqual(code, 0)
        self.assertIn("completed", (self.root / "console.log").read_text())


if __name__ == "__main__":
    unittest.main()
