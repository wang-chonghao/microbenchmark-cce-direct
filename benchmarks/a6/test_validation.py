"""Semantic inputs, operand-specific probes and conservative timing evidence."""

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import semantic
from .coverage_a5 import emit
from .archive import archive_dumps
from .analyze import analyze
from .test_suite import line
from .variants import generate, memory_pairs
from run_test import parse_kernel_manifest


class ValidationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.logs = self.root / "run/log_ca"
        self.logs.mkdir(parents=True)

    def test_semantic_float_inputs_are_normal_and_nonzero(self):
        for dtype in ("fp32", "fp16", "bf16"):
            data = semantic.input_bytes(dtype, 512)
            self.assertTrue(all(.5 <= v <= 1.25 for v in semantic.decode(data, dtype)))
        self.assertEqual(set(semantic.input_bytes("f8e4m3", 30)), {0x30, 0x38, 0x40})
        with self.assertRaisesRegex(ValueError, "No reviewed"):
            semantic.input_bytes("f4e1m2x2", 512)

    def test_profile_is_reproducible_and_hashes_exact_bytes(self):
        case, code = emit("VADD", "fp32", "ii")
        source = self.root / "kernel.cce"
        source.write_text(code)
        manifest = parse_kernel_manifest(source, 256)
        a = semantic.prepare_inputs(case, manifest, self.root / "inputs", 7)
        b = semantic.prepare_inputs(case, manifest, self.root / "inputs2", 7)
        c = semantic.prepare_inputs(case, manifest, self.root / "inputs3", 8)
        self.assertEqual(a, b)
        self.assertNotEqual(a["tensors"], c["tensors"])
        self.assertEqual((self.root / "inputs/input_B.bin").stat().st_size, manifest.tensors[1].elements * 4)

    def axpy_outputs(self, role):
        case, _ = emit("VAXPY", "fp32", "forwarding", dependency_role=role)
        b = (struct.pack("<64f", *([.5]*64)) + bytes([0xcd])*256) * 12
        c = (struct.pack("<64f", *([1.]*64)) + bytes([0xcd])*256) * 12
        values = [1.25, 1.25, 1.5, 1.5, 1.75, 1.75] if role == "old_vd" else [1.25, 1.25, 1.625, 1.625, 1.8125, 1.8125]
        a = b"".join(struct.pack("<64f", *([v]*64)) + bytes([0xcd])*256 for v in values*2)
        for name, data in (("input_B.bin", b), ("input_C.bin", c), ("output_A.bin", a)):
            (self.root / "run" / name).write_bytes(data)
        return case

    def test_old_destination_and_vn_have_different_references(self):
        for role in ("vn", "old_vd"):
            case = self.axpy_outputs(role)
            self.assertEqual(semantic.numeric_check(self.root, case)[0], "pass")
        case["dependency_role"] = "vn"
        self.assertEqual(semantic.numeric_check(self.root, case)[0], "fail")

    def test_reference_checks_last_valid_lane_and_not_padding(self):
        case = self.axpy_outputs("vn")
        path = self.root / "run/output_A.bin"
        data = bytearray(path.read_bytes())
        data[300] = 42
        path.write_bytes(data)
        self.assertEqual(semantic.numeric_check(self.root, case)[0], "pass")
        data[11*512+252:11*512+256] = struct.pack("<f", 99.)
        path.write_bytes(data)
        self.assertEqual(semantic.numeric_check(self.root, case)[0], "fail")

    def test_no_fictional_golden_for_complex_layouts(self):
        case, _ = emit("VGATHER2", "fp32", "ii")
        self.assertEqual(semantic.numeric_check(self.root, case)[0], "not_implemented")

    def test_contiguous_memory_checks_all_valid_bytes(self):
        case, _ = emit("VLDS", "uint32", "ii")
        data = semantic.input_bytes("uint32", 8192)
        for name in ("input_B.bin", "input_C.bin", "output_A.bin"):
            (self.root / "run" / name).write_bytes(data)
        self.assertEqual(semantic.numeric_check(self.root, case)[0], "pass")

    def test_runner_cannot_silently_ignore_input_profile(self):
        case, _ = emit("VADD", "fp32", "ii")
        result = analyze(self.root, dict(case=case, returncode=0, compile_only=False,
            input_profile={"tensors":{"B":{"sha256":"not-the-actual-input"}}}))
        self.assertEqual(result["status"], "failed")
        self.assertIn("does not match", result["reason"])

    def test_variant_sources_isolate_roles_and_widths(self):
        _, code = emit("VSUB", "fp32", "forwarding", dependency_role="vm")
        self.assertIn("vsub(d2, a2, d0, m)", code)
        _, code = emit("VAXPY", "fp32", "forwarding", dependency_role="old_vd")
        self.assertIn("d2 = d0;", code)
        self.assertIn("vaxpy(d2, a2, (float)0.5, m)", code)
        case, code = emit("VADD", "fp32", "ii", ii_width=16)
        self.assertEqual(case["width"], 16)
        self.assertEqual(code.count("// TARGET"), 16)
        self.assertIn("MB_TENSOR output A uint32 4096", code)
        with self.assertRaises(ValueError):
            emit("VADD", "fp32", "forwarding", dependency_role="old_vd")

    def test_variant_regeneration_does_not_replace_hand_edits(self):
        out = self.root / "variants"
        manifest = generate(out, ["VADD"], ["fp32"])
        self.assertEqual(len(manifest["cases"]), 4)
        path = out / manifest["cases"][0]["source"]
        path.write_text("hand edit")
        with self.assertRaisesRegex(ValueError, "Hand-edited"):
            generate(out, ["VADD"], ["fp32"])

    def test_archive_keeps_only_requested_lsu_extra(self):
        for name in ("core0.veccore0.rvec.LSU.dump", "core1.veccore0.rvec.LSU.dump", "core0.veccore0.rvec.OTHER.dump"):
            (self.logs / name).write_text("x")
        record = archive_dumps(self.root, self.root / "results", {"op":"X", "name":"x"}, keep_lsu=True)
        self.assertEqual(len(record["files"]), 1)
        self.assertEqual(record["files"][0]["kind"], "LSU")
        self.assertEqual(record["removed_files"], 2)

    def test_guards_required_means_missing_marker_is_failure(self):
        case, _ = emit("VADD", "fp32", "ii")
        r = analyze(self.root, dict(case=case, returncode=0, compile_only=False, guard_required=True))
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["guard_status"], "missing")

    def test_forwarding_report_keeps_operand_and_event_boundary(self):
        case, _ = emit("VADD", "fp32", "forwarding")
        case["expected_target_count"] = 2
        for suffix, delay in (("instr_popped_log",0), ("instr_log",6)):
            (self.logs / f"core0.veccore0.{suffix}.dump").write_text(
                line(10+delay,0,0x100) + line(20+delay,1,0x104,regs="Vd[4], Vn[3], Vm[2]"))
        r = analyze(self.root, dict(case=case, returncode=0, compile_only=False))
        self.assertEqual(r["forwarding_by_operand"], {"Vm":10})
        self.assertFalse(r["hardware_calibrated"])
        self.assertEqual(r["streams"][0]["forwarding"][0]["event_boundary"], "popped")

    def test_multi_port_request_is_not_a_single_instruction_interval(self):
        case, _ = emit("VGATHER2", "fp32", "ii")
        case["expected_target_count"] = 2
        for suffix, delay in (("instr_popped_log",0), ("instr_log",6)):
            (self.logs / f"core0.veccore0.{suffix}.dump").write_text(
                line(10+delay,0,0x100,"RV_VGATHER2") + line(20+delay,1,0x104,"RV_VGATHER2"))
        (self.logs / "core0.veccore0.rvec.LSU.dump").write_text(
            "[info] [00000010] [SEND_UB_RD.PORT_0] instr_name RV_VGATHER2 instr_id 0\n"
            "[info] [00000011] [SEND_UB_RD.PORT_1] instr_name RV_VGATHER2 instr_id 0\n"
            "[info] [00000012] [SEND_UB_RD.PORT_0] instr_name RV_VGATHER2 instr_id 1\n")
        r = analyze(self.root, dict(case=case, returncode=0, compile_only=False))
        self.assertIsNone(r["memory_interval_min"])
        self.assertFalse(r["hardware_calibrated"])

    def test_store_data_is_a_source_and_cross_pair_is_selected(self):
        case, _ = memory_pairs()[1]
        case["expected_target_count"] = 2
        for suffix, delay in (("instr_popped_log",0), ("instr_log",6)):
            (self.logs / f"core0.veccore0.{suffix}.dump").write_text(
                line(10+delay,0,0x100,"RV_VADD") +
                line(20+delay,1,0x104,"RV_VSTI",regs="Vd[2]"))
        r = analyze(self.root, dict(case=case, returncode=0, compile_only=False))
        self.assertEqual(r["forwarding_by_operand"], {"store_data":10})
        self.assertEqual(r["forwarding_min"], 10)
        self.assertFalse(r["hardware_calibrated"])

    def test_batch_semantic_input_golden_archive_and_repeat_stability(self):
        from run_a6_bench import main
        cases = self.root / "cases"
        generate(cases, ["VADD"], ["fp32"], (8,))
        cann = self.root / "compiler"
        cann.mkdir()
        (cann / "set_env.sh").write_text("")
        output = self.root / "results"

        def fake_model(command, console, timeout, env):
            run = Path(command[command.index("--output") + 1]) / "run"
            logs = run / "log_ca"
            logs.mkdir(parents=True)
            inputs = Path(env["MB_INPUT_DIR"])
            b, c = [(inputs / name).read_bytes() for name in ("input_B.bin", "input_C.bin")]
            (run / "input_B.bin").write_bytes(b)
            (run / "input_C.bin").write_bytes(c)
            result = bytearray([0xcd] * len(b))
            for offset in range(0, len(b), 512):
                x = struct.unpack("<64f", b[offset:offset+256])
                y = struct.unpack("<64f", c[offset:offset+256])
                result[offset:offset+256] = struct.pack("<64f", *[u+v for u,v in zip(x,y)])
            (run / "output_A.bin").write_bytes(result)
            (run / "guard_check.txt").write_text("pass: synthetic integration fixture\n")
            for suffix, delay in (("instr_popped_log",0), ("instr_log",6)):
                (logs / f"core0.veccore0.{suffix}.dump").write_text("".join(
                    line(10+2*i+delay,i,0x100+4*i,regs=f"Vd[{3*i}], Vn[{3*i+1}], Vm[{3*i+2}]") for i in range(16)))
            (logs / "core0.veccore0.rvec.EXU.dump").write_text("".join(
                f"[info] [{10+2*i:08d}] instr_name RV_VADD instr_id {i} exu_id: 0\n" for i in range(16)))
            return 0

        argv = ["run_a6_bench.py", "--cases", str(cases), "--modes", "ii", "--repeats", "3",
                "--cann-home", str(cann), "--output", str(output)]
        with patch.object(sys,"argv",argv), patch("run_a6_bench.run_command",side_effect=fake_model), \
                patch("builtins.print"):
            self.assertEqual(main(),0)
        rows = json.loads((output / "summary.json").read_text())
        self.assertEqual(len(rows),3)
        for row in rows:
            self.assertEqual(row["golden"],"pass")
            self.assertEqual(row["guard_status"],"pass")
            self.assertEqual(row["ii_min"],2)
            self.assertFalse(row["hardware_calibrated"])
        convergence = json.loads((output / "convergence.json").read_text())
        interval = next(row for row in convergence if row["metric"] == "ii_min")
        self.assertTrue(interval["stable_observed_minimum"])
        self.assertFalse(interval["hardware_calibrated"])


if __name__ == "__main__":
    unittest.main()
