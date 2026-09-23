#!/usr/bin/env python3
"""Run A6 probes sequentially and archive selected veccore0 instruction logs."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from benchmarks.a6.generate import DEFAULT_CASES, ROOT
from benchmarks.a6.analyze import analyze
from benchmarks.a6.archive import archive_dumps


def run_command(command, output, timeout, env):
    with output.open("w") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            return 124


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--ops", nargs="+", help="ISA names, e.g. VADD VEXP (default: all)")
    p.add_argument("--forms", nargs="+", help="e.g. fp32 fp16")
    p.add_argument("--modes", nargs="+", choices=["latency", "ii", "forwarding"])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--compile-only", action="store_true", help="Kernel-only preflight; no model or runner")
    p.add_argument("--list", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip completed, identical run fingerprints")
    p.add_argument("--analyze-only", action="store_true", help="Reparse an existing --output")
    p.add_argument("--output", type=Path)
    p.add_argument("--cann-home", default=os.environ.get("CANN_HOME"))
    p.add_argument("--full-simulator-home", default=os.environ.get("FULL_SIMULATOR_HOME", ""),
                   help="External full-dump simulator root containing dav_9201 or Ascend910_9691")
    p.add_argument("--keep-all-dumps", action="store_true", help="Keep extra model dumps in _work (default: only six veccore0 kinds)")
    args = p.parse_args()
    if args.analyze_only:
        if not args.output:
            p.error("--analyze-only requires --output")
        return summarize(args.output, use_cache=False)
    if args.repeats < 1 or args.timeout <= 0:
        p.error("repeats and timeout must be positive")
    manifest = json.loads((args.cases / "manifest.json").read_text())
    selected = [c for c in manifest["cases"]
                if (not args.ops or c["op"] in [o.upper() for o in args.ops])
                and (not args.forms or c["form"] in args.forms)
                and (not args.modes or c["mode"] in args.modes)]
    if not selected:
        p.error("No matching cases")
    if args.list:
        for c in selected:
            print(c["name"])
        print(f"{len(selected)} cases")
        return 0
    if not args.cann_home or not (Path(args.cann_home) / "set_env.sh").is_file():
        p.error("Set CANN_HOME to an installed toolkit with set_env.sh")
    if args.resume and not args.output:
        p.error("--resume requires --output")
    if args.full_simulator_home:
        simulator = Path(args.full_simulator_home).expanduser().resolve()
        if not any((simulator / name / sub).is_dir() for name in ("dav_9201", "Ascend910_9691")
                   for sub in ("lib", "camodel")):
            p.error("FULL_SIMULATOR_HOME must contain dav_9201/{lib,camodel} or Ascend910_9691/{lib,camodel}")
    out = (args.output or ROOT / "results" / time.strftime("a6_%Y%m%d_%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    env = os.environ.copy()
    env.update(SOC_VERSION="Ascend910_9691", CORE_ARCH="dav-920r1-vec",
               CORE_SIM_DIR="dav_9201", NPU_TYPE="Ascend910_9691",
               CANN_HOME=str(Path(args.cann_home).resolve()))
    env["FULL_SIMULATOR_HOME"] = str(simulator) if args.full_simulator_home else ""
    env.setdefault("ARCH", platform.machine() + "-linux")
    env_record = {k: env.get(k, "") for k in ("CANN_HOME", "FULL_SIMULATOR_HOME", "CORE_ARCH",
                  "CORE_SIM_DIR", "SOC_VERSION", "NPU_TYPE", "CCEC_EXTRA_FLAGS", "LD_LIBRARY_PATH",
                  "LD_PRELOAD", "LOCAL_MEMORY_SIZE", "ARCH")}
    framework_hash = hashlib.sha256(b"".join((ROOT / f).read_bytes() for f in
        ("run_test.py", "runner/generic_runner.cpp", "benchmarks/a6/compile_probe.py"))).hexdigest()
    # Environment and source must match before reusing a completed experiment.
    for case in selected:
        source = args.cases / case["source"]
        contents = source.read_bytes()
        if hashlib.sha256(contents).hexdigest() != case["source_sha256"]:
            p.error(f"Source changed since generation: {source}; regenerate manifest first")
        repeats = 1 if args.compile_only else args.repeats
        for repeat in range(repeats):
            run = work / case["name"] / f"r{repeat}"
            fingerprint = hashlib.sha256(json.dumps(dict(case=case, env=env_record,
                framework_hash=framework_hash, keep_all_dumps=args.keep_all_dumps,
                compile_only=args.compile_only), sort_keys=True).encode()).hexdigest()
            attempts = sorted((work / case["name"]).glob(f"r{repeat}_attempt*"), key=lambda x: x.stat().st_mtime)
            if attempts and args.resume:
                run = attempts[-1]
            record_path = run / "run_record.json"
            if record_path.exists():
                old = json.loads(record_path.read_text())
                if args.resume and old["fingerprint"] == fingerprint and old["returncode"] == 0:
                    print(f"[SKIP] {case['name']} r{repeat}", flush=True)
                    continue
                if args.resume and old["fingerprint"] == fingerprint:
                    attempt = len(attempts) + 1
                    run = work / case["name"] / f"r{repeat}_attempt{attempt}"
                    record_path = run / "run_record.json"
                else:
                    p.error(f"Existing experiment {run}; choose a new --output (never overwrite dumps)")
            run.mkdir(parents=True, exist_ok=True)
            snapshot = run / "kernel.cce"
            snapshot.write_bytes(contents)
            if args.compile_only:
                cmd = [sys.executable, "-m", "benchmarks.a6.compile_probe", "--kernel", str(snapshot),
                       "--output", str(run), "--cann-home", env["CANN_HOME"]]
            else:
                cmd = [sys.executable, str(ROOT / "run_test.py"), "--kernel", str(snapshot),
                       "--output", str(run), "--cann-home", env["CANN_HOME"],
                       "--soc-version", env["SOC_VERSION"], "--core-arch", env["CORE_ARCH"],
                       "--core-sim-dir", env["CORE_SIM_DIR"], "--block-dim", "1", "--device-id", "0",
                       "--golden", "none"]
            record = dict(case=case, fingerprint=fingerprint, environment=env_record,
                          framework_hash=framework_hash, command=cmd,
                          compile_only=args.compile_only, returncode=None)
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"[RUN] {case['name']} r{repeat}", flush=True)
            t = time.monotonic()
            record["returncode"] = run_command(cmd, run / "console.log", args.timeout, env)
            record["wall_seconds"] = time.monotonic() - t
            if not args.compile_only:
                record["dump_archive"] = archive_dumps(run, out, case, args.keep_all_dumps)
                missing = record["dump_archive"]["missing_or_empty"]
                if missing:
                    print(f"[WARN] {case['name']} missing/empty dumps: {', '.join(missing)}", flush=True)
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            if record["returncode"] != 0:
                print(f"[FAIL] code={record['returncode']}: {run / 'debug.log'}", flush=True)
            summarize(out, quiet=True)
    return summarize(out)


def summarize(out, quiet=False, use_cache=True):
    rows = []
    latest = {}
    records = list(out.glob("*/r*/run_record.json")) + list(out.glob("_work/*/r*/run_record.json"))
    for path in sorted(records):
        record = json.loads(path.read_text())
        measurement = path.parent / "measurement.json"
        if use_cache and measurement.exists() and measurement.stat().st_mtime >= path.stat().st_mtime:
            result = json.loads(measurement.read_text())
        else:
            result = analyze(path.parent, record)
            measurement.write_text(json.dumps(result, indent=2) + "\n")
        row = dict(case=record["case"]["name"], repeat=path.parent.name, **result)
        rows.append(row)
        repeat, _, attempt = path.parent.name.partition("_attempt")
        key = (row["case"], repeat)
        rank = int(attempt or 0)
        if key not in latest or rank > latest[key][0]:
            latest[key] = (rank, row)
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = ["case", "repeat", "status", "golden", "latency_min", "ii_min", "forwarding_min", "memory_interval_min", "reason"]
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    if not quiet:
        print(f"[SUMMARY] {out / 'summary.csv'}")
    return int(any(r["status"] in {"failed", "missing_logs", "target_mismatch", "invalid_logs"}
                   for _, r in latest.values()))


if __name__ == "__main__":
    raise SystemExit(main())
