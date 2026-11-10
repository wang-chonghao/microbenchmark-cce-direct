#!/usr/bin/env python3
"""Sequential A6 whole-operator experiments (not instruction calibration)."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import sys
import time

from benchmarks.a6.archive import archive_dumps
from benchmarks.a6.check_softmax import check
from run_a6_bench import run_command

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "op_kernel/a6_softmax_shapes"


def prepare_input(directory, rows, seed):
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    # Vary both lanes and rows, avoiding the default monotonic runner inputs.
    values = [rng.uniform(-6, 6) + row * 0.125
              for row in range(rows) for _ in range(1024)]
    (directory / "input_input.bin").write_bytes(struct.pack(f"<{len(values)}f", *values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--rows", type=int, nargs="+", choices=(8, 12, 16, 24))
    p.add_argument("--variants", nargs="+", choices=("no_membar", "three_loop"))
    p.add_argument("--list", action="store_true")
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--cann-home", default=os.environ.get("CANN_HOME"))
    p.add_argument("--full-simulator-home", default=os.environ.get("FULL_SIMULATOR_HOME", ""))
    p.add_argument("--timeout", type=float, default=0, help="Per-case seconds; 0 = unlimited")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-all-dumps", action="store_true")
    p.add_argument("--keep-lsu-dumps", action="store_true")
    args = p.parse_args()
    if not math.isfinite(args.timeout) or args.timeout < 0:
        p.error("timeout must be finite and nonnegative")
    case_root = args.cases.resolve()
    manifest = json.loads((case_root / "manifest.json").read_text())
    cases = [c for c in manifest["cases"]
             if (not args.rows or c["shape"][0] in args.rows)
             and (not args.variants or c["variant"] in args.variants)]
    if not cases:
        p.error("No selected cases")
    for c in cases:
        if c["family"] != "softmax" or c["shape"][1] != 1024:
            p.error("This runner currently supports the softmax shape suite only")
        source = case_root / c["source"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != c["source_sha256"]:
            p.error(f"Source hash mismatch: {source}")
    if args.list:
        for c in cases:
            print(f"{c['name']}: {c['shape']}, UB={c['ub_reserved_bytes']} bytes")
        return 0
    if not args.cann_home:
        p.error("Set CANN_HOME or pass --cann-home")
    out = (args.output or ROOT / "results" / ("a6_operators_" + time.strftime("%Y%m%d_%H%M%S"))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CANN_HOME=str(Path(args.cann_home).resolve()),
               FULL_SIMULATOR_HOME=args.full_simulator_home,
               SOC_VERSION="Ascend910_9691", NPU_TYPE="Ascend910_9691",
               CORE_ARCH="dav-920r1-vec", CORE_SIM_DIR="dav_9201", MB_GUARD_ENABLE="1",
               ACL_DEVICE_ID="0")
    records = []
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"OUTPUT={out}", flush=True)
    for c in cases:
        run = out / "_work" / c["name"]
        run.mkdir(parents=True)
        source = run / "kernel.cce"
        shutil.copyfile(case_root / c["source"], source)
        inputs = run / "inputs"
        prepare_input(inputs, c["shape"][0], args.seed)
        case_env = dict(env, MB_INPUT_DIR=str(inputs))
        if args.compile_only:
            cmd = [sys.executable, "-m", "benchmarks.a6.compile_probe",
                   "--kernel", str(source), "--output", str(run),
                   "--cann-home", env["CANN_HOME"]]
        else:
            cmd = [sys.executable, str(ROOT / "run_test.py"), "--kernel", str(source),
                   "--output", str(run), "--cann-home", env["CANN_HOME"],
                   "--soc-version", "Ascend910_9691", "--core-arch", "dav-920r1-vec",
                   "--core-sim-dir", "dav_9201", "--device-id", "0", "--block-dim", "1",
                   "--local-memory-size", "0", "--golden", "none"]
        record = dict(name=c["name"], shape=c["shape"], variant=c["variant"],
                      source_sha256=c["source_sha256"], command=cmd,
                      status="failed", golden=None)
        try:
            code = run_command(cmd, run / "console.log", args.timeout, case_env)
            record["returncode"] = code
            if args.compile_only:
                record["status"] = "compiled_only" if code == 0 else "failed"
            else:
                record["logs"] = archive_dumps(run, out / "dumps",
                    {"op": "softmax", "name": c["name"]},
                    keep_all=args.keep_all_dumps, keep_lsu=args.keep_lsu_dumps)
                if code == 0:
                    record["golden"] = check(run / "run/input_input.bin",
                                              run / "run/output_output.bin", c["shape"][0])
                    # The native runner fails on corrupted guards/input data.
                    record["status"] = "passed" if record["golden"]["passed"] else "golden_failed"
        except (OSError, ValueError) as exc:
            record["error"] = str(exc)
        (run / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        records.append(record)
        (out / "summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(f"[{len(records)}/{len(cases)}] {c['name']}: {record['status']}", flush=True)
    return int(any(r["status"] not in ("passed", "compiled_only") for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
