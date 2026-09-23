"""Parallel compiler preflight only; never starts multiple camodel processes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

from benchmarks.a6.generate import DEFAULT_CASES, ROOT
from run_a6_bench import run_command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cann-home", default=os.environ.get("CANN_HOME"), required=not os.environ.get("CANN_HOME"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--names", nargs="*")
    args = p.parse_args()
    if args.jobs < 1:
        p.error("jobs must be positive")
    cases = json.loads((args.cases / "manifest.json").read_text())["cases"]
    if args.names:
        cases = [c for c in cases if c["name"] in args.names]
    if not cases:
        p.error("No matching cases")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def check(case):
        dest = out / case["name"]
        dest.mkdir(exist_ok=False)
        source = dest / "kernel.cce"
        source.write_bytes((args.cases / case["source"]).read_bytes())
        cmd = [sys.executable, "-m", "benchmarks.a6.compile_probe", "--kernel", str(source),
               "--output", str(dest), "--cann-home", args.cann_home]
        code = run_command(cmd, dest / "console.log", 180, os.environ.copy())
        return dict(name=case["name"], returncode=code, source_sha256=case["source_sha256"])

    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(check, c) for c in cases]):
            row = future.result()
            rows.append(row)
            print(f"[{len(rows)}/{len(cases)}] {row['name']} exit={row['returncode']}", flush=True)
            (out / "compile_report.json").write_text(json.dumps(sorted(rows, key=lambda r: r['name']), indent=2))
    return int(any(r["returncode"] for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
