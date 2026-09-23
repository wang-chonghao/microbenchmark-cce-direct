"""Isolated kernel-only compilation worker; uses the same flags as run_test.py."""

import argparse
from pathlib import Path

import run_test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cann-home", required=True)
    p.add_argument("--arch", default=run_test.default_arch())
    p.add_argument("--core-arch", default="dav-920r1-vec")
    args = p.parse_args()
    args.total_count = 256
    args.kernel_name = run_test.parse_kernel_manifest(args.kernel, 256).kernel_name
    build = args.output / "build"
    build.mkdir(parents=True, exist_ok=True)
    run_test.compile_kernel(args, Path(args.cann_home), args.arch, build, args.output / "debug.log")


if __name__ == "__main__":
    main()
