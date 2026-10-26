"""Opt-in saturation and dependency-operand probes; never expands the default suite."""

import argparse
import hashlib
import json
from pathlib import Path

from .coverage_a5 import emit, BINARY, PLOGIC, ACC
from .raw_wrapper import wrap


def memory_pairs():
    jobs = []
    for op in ("VLDS_TO_VADD", "VADD_TO_VSTS"):
        name = op.lower() + "_fp32_forwarding"
        body = """        vector_bool m = pset_b32(PAT_ALL);
        vector_f32 a0, a1, d0, d1, one;
        vdup(one, 1.0f, m, MODE_ZEROING);
        vlds(a0, (__ubuf__ float*)ubB, 0, NORM);
        vlds(a1, (__ubuf__ float*)ubB, 128, NORM);
        vadd(d0, a0, one, m);
        vadd(d1, a1, one, m);
        vsts(d0, (__ubuf__ float*)ubA, 0, NORM_B32, m);
        vsts(d1, (__ubuf__ float*)ubA, 128, NORM_B32, m);"""
        producer = ["RV_VLDI", "RV_VLDS"] if op == "VLDS_TO_VADD" else ["RV_VADD"]
        consumer = ["RV_VADD"] if op == "VLDS_TO_VADD" else ["RV_VSTI", "RV_VSTS"]
        source = wrap(name, body, 256, 256, generator="variants.py")
        meta = dict(name=name, op=op, form="fp32", mode="forwarding", width=2, loop_iterations=1,
                    src_lanes=128, dst_lanes=128, input_dtype="uint32", output_dtype="uint32",
                    semantic_input_dtype="fp32", semantic_output_dtype="fp32", compound=False,
                    golden="semantic_reference", cross_forwarding=True, producer_aliases=producer,
                    consumer_aliases=consumer, target_aliases=producer+consumer, expected_target_count=4,
                    forwarding_edges=[], source=name+".cce", source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    note="Cross-unit popped gap only; not a calibrated bypass delay or cross-instruction II")
        jobs.append((meta, source))
    return jobs


def generate(out, ops, forms, widths=(8, 16), with_memory_pairs=False):
    jobs = []
    for op in ops:
        for form in forms:
            for width in widths:
                jobs.append(emit(op, form, "ii", ii_width=width))
            jobs.append(emit(op, form, "forwarding"))
            if op in BINARY | (PLOGIC - {"PNOT"}):
                jobs.append(emit(op, form, "forwarding", dependency_role="vm"))
            if op in ACC | {"VMOV_MERGE"}:
                jobs.append(emit(op, form, "forwarding", dependency_role="old_vd"))
    if with_memory_pairs:
        jobs += memory_pairs()
    out.mkdir(parents=True, exist_ok=True)
    old = out / "manifest.json"
    if old.exists():
        for case in json.loads(old.read_text())["cases"]:
            path = out / case["source"]
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != case["source_sha256"]:
                raise ValueError(f"Hand-edited probe: {path}")
    manifest = dict(schema_version=2, soc="Ascend910_9691", core_arch="dav-920r1-vec",
                    core_sim_dir="dav_9201", cases=[meta for meta, _ in jobs])
    for meta, code in jobs:
        path = out / meta["source"]
        if path.exists() and not old.exists():
            raise ValueError(f"Refusing to overwrite untracked source: {path}")
        path.write_text(code)
    old.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ops", nargs="+", default=["VADD", "VAXPY"])
    parser.add_argument("--forms", nargs="+", default=["fp32"])
    parser.add_argument("--ii-widths", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--with-memory-pairs", action="store_true", help="Add VLDS->VADD and VADD->VSTS observational probes")
    args = parser.parse_args()
    manifest = generate(args.output, [o.upper() for o in args.ops], args.forms, args.ii_widths, args.with_memory_pairs)
    print(f"Generated {len(manifest['cases'])} opt-in probes in {args.output}")


if __name__ == "__main__":
    main()
