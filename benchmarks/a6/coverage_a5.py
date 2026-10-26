"""Generate missing A6 probes using a pinned A5 instruction inventory."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .exu2 import VECTORS, emit as emit_exu2
from .generate import ROOT
from .raw_wrapper import wrap

SNAPSHOT = Path(__file__).with_name("a5_coverage_inventory.json")
OUT = ROOT / "op_kernel/a6_a5_coverage"
TYPES = dict(VECTORS)
TYPES.update(f4e1m2x2=("vector_f4e1m2x2", 8), f4e2m1x2=("vector_f4e2m1x2", 8),
             s4x2=("vector_s4x2", 8))
C = {"fp32": "float", "fp16": "half", "bf16": "bfloat16_t", "b32": "uint32_t",
     "f8e4m3": "float8_e4m3_t", "f8e5m2": "float8_e5m2_t", "f8e8m0": "float8_e8m0_t",
     "f4e1m2x2": "float4_e1m2x2_t", "f4e2m1x2": "float4_e2m1x2_t", "s4x2": "int4x2_t",
     "hif8": "hifloat8_t"}
C.update({t: t + "_t" for t in TYPES if t.startswith(("int", "uint"))})
BINARY = set("VADD VSUB VMUL VMAX VMIN VAND VOR VXOR VABSDIF VPRELU VSHL VSHR".split())
UNARY = set("VMOV VABS VNEG VNOT VRELU".split())
SCALAR = set("VADDS VMULS VMAXS VMINS VSHLS VSHRS VLRELU".split())
ACC = {"VMADD", "VMULA", "VAXPY"}
CARRY = {"VADDC", "VADDCS", "VSUBC", "VSUBCS"}
PLOGIC = {"PNOT", "PXOR", "PSEL", "PAND", "POR"}
MEMORY = {"VLDS", "VSTS", "VLDAS", "VLDUS", "VSTUS", "VSTAS", "VLD", "VST",
          "PLDS", "PSTS", "PLD", "PLDI", "PST", "PSTI", "VGATHER2", "VSCATTER", "VSSTB",
          "VLDA", "VLDU", "VLDUI", "VSLDB", "VGATHERB", "VGATHERB_UNMASKED", "VGATHER2_BC"}
STORE = {"VSTS", "VSTUS", "VSTAS", "VST", "PSTS", "PST", "PSTI", "VSCATTER", "VSSTB"}
PLOAD = {"PLDS", "PLD", "PLDI"}
PSTORE = {"PSTS", "PST", "PSTI"}
TRANS = {"f16": "fp16", "f32": "fp32", "s8": "int8", "s16": "int16", "s32": "int32",
         "u8": "uint8", "u16": "uint16", "u32": "uint32"}


def conversion(form):
    src, dst = form.split("_to_")
    a, b = TRANS.get(src, src), TRANS.get(dst, dst)
    if a not in TYPES or b not in TYPES:
        raise ValueError("64-bit/multi-register conversion needs a dedicated paired-register template")
    sa, sb = TYPES[a][1], TYPES[b][1]
    floating_a = src.startswith(("f", "bf", "hi"))
    floating_b = dst.startswith(("f", "bf", "hi"))
    if "s4x2" in form or "f4" in form or "hif" in form or (sa == sb and floating_a and floating_b):
        raise ValueError("Packed/equal-width float conversion needs reviewed A6 rounding/partition signature")
    part = "PART_EVEN" if max(sa, sb) == 2 * min(sa, sb) else "PART_P0"
    if floating_a and not floating_b:
        opts = "m, ROUND_R" + (", RS_ENABLE" if sa >= sb or src == "bf16" else "")
    elif not floating_a and floating_b:
        opts = "m, ROUND_R" if sa >= sb else "m"
    elif floating_a and floating_b:
        opts = "m, ROUND_R, RS_ENABLE" if sa > sb else "m"
    else:
        opts = "m, RS_ENABLE" if sa > sb else "m"
    if sa != sb:
        opts += ", " + part
    return a, b, opts


def spec(op, form):
    if op == "VCGADD":
        return form, form, False
    if op.startswith("VCVT_"):
        a, b, _ = conversion(form)
        return a, b, False
    if op in PLOAD | PSTORE | PLOGIC:
        return "uint8", "uint8", op in PLOGIC
    if op == "MOVVP":
        dtype = "uint" + form[1:] if form.startswith("b") else form
        return dtype, dtype, False
    if form not in TYPES:
        raise ValueError("64-bit or non-vector form requires reviewed register/address layout")
    if form == "s4x2":
        raise ValueError("Packed s4x2 cannot use the raw uint32 vector transport cast")
    if op in {"VBR", "VDUPS", "VDUPS_MERGE"} and form.startswith("f8"):
        raise ValueError("Local CCEC crashes lowering scalar float8 constants; needs a scalar-bit template")
    if op in {"VSHL", "VSHR"} and form.startswith("int"):
        raise ValueError("Local A6 header rejects this signed-vector shift signature; do not substitute unsigned semantics")
    if op in BINARY | UNARY | SCALAR | ACC | CARRY | {"VMULL", "VSEL", "VMOV_MERGE", "VTRC", "VDUPS", "VDUPS_MERGE", "VCI", "VBR"}:
        return form, form, op not in {"VDUPS", "VDUPS_MERGE", "VCI", "VBR"}
    if op.startswith(("VCMP_", "VCMPS_")):
        return form, form, False
    if op in MEMORY:
        if op == "VGATHER2" and TYPES[form][1] == 8:
            raise ValueError("8-bit gather has widened/packed lane layout; requires a dedicated probe")
        return form, form, False
    raise ValueError("A5 opcode/alias needs explicit A6 signature, lowering and side-effect review")


def emit(op, form, mode):
    if op == "VCGADD":
        return emit_exu2(op, form, mode)
    src, dst, raw = spec(op, form)
    if mode not in {"ii", "forwarding"} or (mode == "forwarding" and not raw):
        raise ValueError("Unsupported measurement mode")
    sv, bits = TYPES[src]
    dv, dbits = TYPES[dst]
    n = 8 if mode == "ii" else 6
    iterations = 1 if op in {"VCI", "VBR", "VDUPS"} else 2
    predicate = op in PLOGIC | PLOAD | PSTORE | {"MOVVP"} or op.startswith(("VCMP_", "VCMPS_"))
    compare = op.startswith(("VCMP_", "VCMPS_"))
    implicit = op in ACC | {"VMOV_MERGE", "VDUPS_MERGE"}
    memory = op in MEMORY
    name = f"{op.lower()}_{form}_{mode}"
    aliases = ["RV_" + op]
    if op == "VMOV_MERGE":
        aliases = ["RV_VSEL"]  # A6 header explicitly implements vmov merging with vsel.
    if op == "VDUPS_MERGE":
        aliases = ["RV_VDUPS"]
    if op == "VLDS":
        aliases = ["RV_VLDI", "RV_VLDS"]
    if op == "VSTS":
        aliases = ["RV_VSTI", "RV_VSTS"]
    if op == "VGATHERB_UNMASKED":
        aliases = ["RV_VGATHERB"]
    if op.startswith("VCVT_"):
        a, b, _ = conversion(form)
        af, bf = a.startswith(("fp", "bf", "hi")), b.startswith(("fp", "bf", "hi"))
        aliases = ["RV_VCVT_" + ("F" if af else "I") + "2" + ("F" if bf else "I")]
    code = [f"vector_bool m = pset_b{bits}(PAT_ALL);", f"vector_bool partial = pset_b{bits}(PAT_H);",
            "vector_bool out_mask = pset_b32(PAT_ALL);", "#pragma unroll(1)",
            f"for (uint16_t group = 0; group < {iterations}; ++group) {{"]
    for i in range(n):
        off = f"(group * {n} + {i}) * 128"
        code += [f"{dv} d{i};", f"{sv} a{i}, b{i};", f"vector_bool p{i}, q{i};"]
        # Preload vector inputs only for operations that actually consume them.
        if op not in {"VLDS", "VLD", "VLDAS", "VLDUS", "VLDA", "VLDU", "VLDUI", "VSLDB"} | PLOAD | PSTORE | PLOGIC:
            code += [f"vector_u32 ba{i}, bb{i};", f"vlds(ba{i}, ubB, {off}, NORM);",
                     f"vlds(bb{i}, ubC, {off}, NORM);", f"a{i} = ({sv})ba{i};", f"b{i} = ({sv})bb{i};"]
        if implicit:
            code += [f"d{i} = ({dv})b{i};"]
        if op in PLOGIC | PSTORE:
            code += [f"plds(p{i}, ubB, ({off}) * 4, NORM);", f"plds(q{i}, ubC, ({off}) * 4, NORM);"]
        if memory and op not in PLOAD | PSTORE:
            base = "ubA" if op in STORE else "ubB"
            code += [f"__ubuf__ {C[src]}* ptr{i} = (__ubuf__ {C[src]}*)({base} + {off});"]
        if op in {"PLD", "PST", "VLDA", "VLDU", "VST"}:
            code += [f"vector_address address{i} = vag_b{bits}((uint32_t)0);"]
        if op in {"VGATHER2", "VSCATTER"}:
            # Linear byte offsets, bounded to one 256-byte tile. The b8 scatter
            # uses only 128 lanes, matching its 128 x uint16 offset register.
            ib = max(16, bits)
            code += [f"vector_s{ib} signed_index{i};", f"vci(signed_index{i}, (int{ib}_t)0);",
                     f"vector_u{ib} index{i} = (vector_u{ib})signed_index{i};"]
            if bits > 8:
                code += [f"vmuls(index{i}, index{i}, (uint{ib}_t){bits // 8}, pset_b{ib}(PAT_ALL));"]
        if op in {"VGATHERB", "VGATHERB_UNMASKED", "VGATHER2_BC"}:
            code += [f"vector_u32 index{i};", f"vbr(index{i}, (uint32_t)0);"]
        if op == "VMULL":
            code += [f"{dv} hi{i};"]
        if op in CARRY:
            code += [f"vector_bool carry{i};"]
    for i in range(n):
        a = f"d{i-2}" if mode == "forwarding" and i >= 2 else f"a{i}"
        p = f"r{i-2}" if mode == "forwarding" and i >= 2 else f"p{i}"
        d, b = f"d{i}", f"b{i}"
        code += [f"// TARGET {i}: {op}/{form}"]
        if op in PLOGIC:
            code += [f"vector_bool r{i};"]
            args = f"r{i}, {p}, m" if op == "PNOT" else f"r{i}, {p}, q{i}, m"
            code += [f"{op.lower()}({args});"]
        elif compare:
            second = f"({C[src]})1" if op.startswith("VCMPS_") else b
            code += [f"{op.lower()}(p{i}, {a}, {second}, m);"]
        elif op == "MOVVP":
            code += [f"movvp(p{i}, {a}, 0);"]
        elif op in BINARY:
            code += [f"{op.lower()}({d}, {a}, {b}, m);"]
        elif op in UNARY:
            code += [f"vmov({d}, {a});" if op == "VMOV" else f"{op.lower()}({d}, {a}, m);"]
        elif op == "VSEL":
            code += [f"vsel({d}, {a}, {b}, partial);"]
        elif op == "VMOV_MERGE":
            code += [f"vmov({d}, {a}, partial, MODE_MERGING);"]
        elif op in SCALAR:
            scalar = 3 if op == "VMULS" else 1
            code += [f"{op.lower()}({d}, {a}, ({C[src]}){scalar}, m);"]
        elif op in ACC:
            rhs = f"({C[src]})0.5" if op == "VAXPY" else b
            code += [f"{op.lower()}({d}, {a}, {rhs}, m);"]
        elif op in CARRY:
            tail = ", partial, m" if op.endswith("CS") else ", m"
            code += [f"{op.lower()}(carry{i}, {d}, {a}, {b}{tail});"]
        elif op == "VMULL":
            code += [f"vmull({d}, hi{i}, {a}, {b}, m);"]
        elif op == "VTRC":
            code += [f"vtrc({d}, {a}, ROUND_R, m);"]
        elif op in {"VBR", "VCI"}:
            code += [f"{op.lower()}({d}, ({C[src]}){i+1});"]
        elif op in {"VDUPS", "VDUPS_MERGE"}:
            md = "MODE_MERGING" if implicit else "MODE_ZEROING"
            code += [f"vdup({d}, ({C[src]}){i+1}, partial, {md});"]
        elif op.startswith("VCVT_"):
            code += [f"vcvt({d}, {a}, {conversion(form)[2]});"]
        elif op in PLOAD | PSTORE:
            off = f"(group * {n} + {i}) * 512"
            base = "ubB" if op in PLOAD else "ubA"
            if op in {"PLD", "PST"}:
                code += [f"{op.lower()}(p{i}, {base} + ({off}) / 4, address{i}, NORM);"]
            else:
                code += [f"{op.lower()}(p{i}, {base}, {off}, NORM);"]
        elif op in {"VLDS", "VLD"}:
            code += [f"vlds({d}, ptr{i}, 0, NORM);" if op == "VLDS" else f"vld({d}, ptr{i}, NORM);"]
        elif op in {"VLDAS", "VLDUS"}:
            # Keep alignment state local to each load pair. Retaining eight
            # initialized align states across the whole body stresses a small
            # special-register bank and is not a pure memory-throughput probe.
            code += [f"vector_align state{i};", f"vldas(state{i}, ptr{i});",
                     f"vldus({d}, state{i}, ptr{i}, {2048 // bits}, POST_UPDATE);"]
        elif op in {"VLDA", "VLDU"}:
            code += [f"vector_align state{i};", f"vlda(state{i}, ptr{i}, address{i});",
                     f"vldu({d}, state{i}, address{i}, ptr{i}, {2048 // bits});"]
        elif op == "VLDUI":
            code += [f"vector_align state{i};", f"vldas(state{i}, ptr{i});",
                     f"vldui({d}, state{i}, ptr{i}, {2048 // bits}, POST_UPDATE);"]
        elif op == "VSLDB":
            code += [f"vsldb({d}, ptr{i}, 0, m);"]
        elif op == "VSTS":
            code += [f"vsts({a}, ptr{i}, 0, NORM_B{bits}, m);"]
        elif op == "VST":
            code += [f"vst({a}, ptr{i}, address{i}, NORM_B{bits}, m);"]
        elif op in {"VSTUS", "VSTAS"}:
            code += [f"vector_align state{i} = {{}};",
                     f"vstus(state{i}, {2048 // bits}, {a}, ptr{i}, POST_UPDATE);",
                     f"vstas(state{i}, ptr{i}, 0);"]
        elif op == "VGATHER2":
            code += [f"vgather2({d}, ptr{i}, index{i}, m);"]
        elif op in {"VGATHERB", "VGATHERB_UNMASKED"}:
            code += [f"vgatherb({d}, ptr{i}, index{i});"]
        elif op == "VGATHER2_BC":
            code += [f"vgather2_bc({d}, ptr{i}, index{i}, m);"]
        elif op == "VSCATTER":
            mask = "pset_b8(PAT_VL128)" if bits == 8 else "m"
            code += [f"vscatter({a}, ptr{i}, index{i}, {mask});"]
        elif op == "VSSTB":
            code += [f"vsstb({a}, ptr{i}, 0, pset_b{bits}(PAT_VL{256 // bits}));"]
        else:
            raise ValueError(f"No emitter for {op}")
    for i in ([0, 2, 4, 1, 3, 5] if mode == "forwarding" else range(n)):
        off = f"(group * {n} + {i}) * 128"
        if op in STORE:
            continue
        if predicate:
            r = f"r{i}" if op in PLOGIC else f"p{i}"
            code += [f"psts({r}, ubA, ({off}) * 4, NORM);"]
        else:
            code += [f"vector_u32 out{i} = (vector_u32)d{i};",
                     f"vsts(out{i}, ubA, {off}, NORM_B32, out_mask);"]
        if op in CARRY:
            code += [f"psts(carry{i}, ubA, (({off}) + 64) * 4, NORM);"]
        if op == "VMULL":
            code += [f"vector_u32 high{i} = (vector_u32)hi{i};",
                     f"vsts(high{i}, ubA, ({off}) + 64, NORM_B32, out_mask);"]
    code += ["}"]
    body = "\n".join("        " + s for s in code)
    size = n * iterations * 128
    cce = wrap(name, body, size, size)
    meta = dict(name=name, op=op, form=form, mode=mode, width=n,
                input_dtype="uint32", output_dtype="uint32", src_lanes=128, dst_lanes=128,
                semantic_input_dtype=src, semantic_output_dtype=dst,
                target_aliases=aliases, expected_target_count=n * iterations,
                compound=False, golden="not_implemented", memory_probe=memory,
                no_register_source=op in {"VCI", "VBR", "VDUPS"},
                note="A5 inventory migration; verify emitted opcode/count. Raw bits, no numeric golden.",
                loop_iterations=iterations, forwarding_edges=[[0, 2], [1, 3], [2, 4], [3, 5]] if mode == "forwarding" else [],
                layout="A0_A1_B0_B1_C0_C1" if mode == "forwarding" else "eight_independent_ops",
                implicit_dst_read=implicit, source=name + ".cce",
                source_sha256=hashlib.sha256(cce.encode()).hexdigest())
    return meta, cce


def capture(repo, ref):
    data = subprocess.check_output(["git", "-C", str(repo), "show", ref + ":configs/isa.json"])
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", ref]).decode().strip()
    isa = json.loads(data)
    return dict(source_ref=ref, source_commit=sha, source_sha256=hashlib.sha256(data).hexdigest(),
                instructions={op: {"forms": {f: {} for f in item.get("forms", {})}}
                              for op, item in isa["instructions"].items()})


def generate(snapshot, out, rejections=None):
    rejections = rejections or {}
    out.mkdir(parents=True, exist_ok=True)
    oldpath = out / "manifest.json"
    old = json.loads(oldpath.read_text()) if oldpath.exists() else {"cases": []}
    for case in old["cases"]:
        p = out / case["source"]
        if p.parent != out or p.suffix != ".cce":
            raise ValueError(f"Invalid generated source: {p}")
        if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() != case["source_sha256"]:
            raise ValueError(f"Hand-edited probe: {p}")
    existing = {(c["op"], c["form"]) for directory in ("a6_instruction", "a6_exu2")
                for c in json.loads((ROOT / "op_kernel" / directory / "manifest.json").read_text())["cases"]}
    cases, coverage = [], []
    for op, item in snapshot["instructions"].items():
        for form in item["forms"]:
            row = dict(op=op, form=form, cases=[])
            if (op, form) in existing:
                row["status"] = "existing_probe"
            else:
                try:
                    _, _, raw = spec(op, form)
                    if op + "/" + form in rejections:
                        raise ValueError(rejections[op + "/" + form])
                    generated = [emit(op, form, m) for m in (["ii", "forwarding"] if raw else ["ii"])]
                    for meta, source in generated:
                        (out / meta["source"]).write_text(source)
                        cases.append(meta)
                        row["cases"].append(meta["name"])
                    row.update(status="new_probe", forwarding="probe" if raw else "not_applicable_in_template")
                except ValueError as exc:
                    row.update(status="deferred", reason=str(exc))
            coverage.append(row)
    current = {c["source"] for c in cases}
    for case in old["cases"]:
        if case["source"] not in current:
            (out / case["source"]).unlink(missing_ok=True)
    manifest = dict(schema_version=2, soc="Ascend910_9691", core_arch="dav-920r1-vec",
                    core_sim_dir="dav_9201", isa_sha256=snapshot["source_sha256"],
                    source_commit=snapshot["source_commit"], cases=cases, coverage=coverage)
    oldpath.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-repo", type=Path)
    p.add_argument("--source-ref", default="tilesim/wch/VfSimulator-refactor")
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    if args.source_repo:
        SNAPSHOT.write_text(json.dumps(capture(args.source_repo, args.source_ref), indent=2) + "\n")
    rejection_path = Path(__file__).with_name("a5_coverage_deferred.json")
    rejections = json.loads(rejection_path.read_text()) if rejection_path.exists() else {}
    manifest = generate(json.loads(SNAPSHOT.read_text()), args.output, rejections)
    from collections import Counter
    print(f"Generated {len(manifest['cases'])} cases; coverage: {dict(Counter(r['status'] for r in manifest['coverage']))}")


if __name__ == "__main__":
    main()
