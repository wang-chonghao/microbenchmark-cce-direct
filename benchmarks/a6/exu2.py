"""Supplement DV121 EXU2 coverage without changing the original A6 suite.

The inventory records model routing, not measured hardware capabilities. Packed
types use raw uint32 tensor transport; vector casts preserve their input bits.
"""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.a6.generate import ROOT, TYPES, render

DEFAULT = ROOT / "op_kernel/a6_exu2"
SNAPSHOT = Path(__file__).with_name("exu2_inventory.json")
VECTORS = {k: (v[1], v[2]) for k, v in TYPES.items()}
VECTORS.update(int16=("vector_s16", 16), b32=("vector_u32", 32),
               f8e4m3=("vector_f8e4m3", 8), f8e5m2=("vector_f8e5m2", 8),
               f8e8m0=("vector_f8e8m0", 8), hif8=("vector_hif8", 8))
PACK = {"b32": ("uint32", "uint16"), "s16_to_u8": ("int16", "uint8"),
        "s32_to_u16": ("int32", "uint16"), "u16_to_u8": ("uint16", "uint8"),
        "u32_to_u16": ("uint32", "uint16")}
UNPACK = {"s16_to_s32": ("int16", "int32"), "s8_to_s16": ("int8", "int16"),
          "u16_to_u32": ("uint16", "uint32"), "u8_to_u16": ("uint8", "uint16")}
COMPOUND = {"VCMAX", "VCMIN", "VCADD", "VCGMAX", "VCGMIN", "VCGADD", "VCPADD",
            "VINTLV", "VDINTLV", "PINTLV", "PDINTLV"}
PREDICATE = {"PINTLV", "PDINTLV", "PINTLVV2", "PDINTLVV2", "PPACK", "PUNPACK"}
HIST = {"CHISTV2", "DHISTV2"}


def specification(op, form):
    src = dst = form
    if op == "VPACK":
        src, dst = PACK[form]
    elif op == "VUNPACK":
        src, dst = UNPACK[form]
    elif op in HIST:
        src, dst = "uint8", "uint16"
    elif op in PREDICATE:
        src = dst = "uint" + form[1:]
    elif op in {"VCADD", "VCGADD", "VCGADDV2"} and form in {"int16", "uint16"}:
        dst = form.replace("16", "32")
    if src not in VECTORS or dst not in VECTORS:
        raise ValueError(f"Unsupported types: {op}/{form}")
    # VUSQZ requires a zero initialized accumulator. A repeated RAW application
    # would not obey that API precondition, so only its independent probe is used.
    raw = src == dst and op not in COMPOUND | PREDICATE | HIST | {"VUSQZ"}
    return src, dst, raw


def emit(op, form, mode, iterations=2):
    if mode not in {"ii", "forwarding"} or not 1 <= iterations <= 4:
        raise ValueError("Expected ii/forwarding and 1..4 iterations (bounded UB output)")
    if op in {"VLN", "VSQRT", "VEXPDIF"}:
        return render(op, form, mode, 8, iterations=iterations)
    src, dst, raw = specification(op, form)
    if mode == "forwarding" and not raw:
        raise ValueError(f"No reviewed self-RAW template: {op}/{form}")
    sv, sb = VECTORS[src]
    dv, db = VECTORS[dst]
    n = 8 if mode == "ii" else 4
    pair = op in {"VINTLV", "VDINTLV", "PINTLV", "PDINTLV"}
    outputs = 2 if pair else 1
    name = f"{op.lower()}_{form}_{mode}"
    aliases = ["RV_" + op]
    if op.startswith("VDUP"):
        aliases = ["RV_VDUPM" if "HIGHEST" in op else "RV_VDUP"]
    if op == "VUNPACK":
        aliases = ["RV_VSUNPACK" if src.startswith("int") else "RV_VZUNPACK"]
    if op in {"VINTLV", "VDINTLV", "PINTLV", "PDINTLV"}:
        aliases = ["RV_" + op + "V2"]
    code = [f"vector_bool m = pset_b{sb}(PAT_ALL);",
            f"vector_bool mask = pset_b{sb}(PAT_H);",
            "vector_bool store_mask = pset_b32(PAT_ALL);",
            "#pragma unroll(1)", f"for (uint16_t group = 0; group < {iterations}; ++group) {{"]
    for i in range(n):
        off = f"(group * {n} + {i}) * 64"
        code += [f"vector_u32 bits_a{i}, bits_b{i};",
                 f"vlds(bits_a{i}, ubB, {off}, NORM);",
                 f"vlds(bits_b{i}, ubC, {off}, NORM);",
                 f"{sv} a{i} = ({sv})bits_a{i};",
                 f"{sv} b{i} = ({sv})bits_b{i};", f"{dv} d{i};"]
        if pair:
            code += [f"{dv} e{i};"]
        if op in PREDICATE:
            code += [f"vector_bool pa{i}, pb{i}, pd{i}, pe{i};",
                     f"vcmp_eq(pa{i}, a{i}, b{i}, m);",
                     f"vcmp_lt(pb{i}, a{i}, b{i}, m);"]
        if op == "VUSQZ":
            code += [f"vector_bool count_mask{i};", f"vcmp_lt(count_mask{i}, a{i}, b{i}, m);"]
        if op == "VSELR":
            # Bound all indices even for arbitrary runner inputs.
            iv = "vector_u" + str(sb)
            code += [f"{iv} index{i} = ({iv})bits_b{i};",
                     f"{iv} limit{i};", f"vbr(limit{i}, (uint{sb}_t){2048 // sb - 1});",
                     f"vand(index{i}, index{i}, limit{i}, m);"]
        if op in HIST | {"VUSQZ"}:
            code += [f"vbr(d{i}, (uint{db}_t)0);"]
        elif "MERGE" in op:
            code += [f"d{i} = ({dv})bits_b{i};"]
    for i in range(n):
        a = f"d{i-2}" if mode == "forwarding" and i >= 2 else f"a{i}"
        d, b = f"d{i}", f"b{i}"
        code += [f"// TARGET {i}: {op}"]
        if op.startswith("VDUP"):
            pos = "POS_HIGHEST" if "HIGHEST" in op else "POS_LOWEST"
            merge = "MODE_MERGING" if "MERGE" in op else "MODE_ZEROING"
            code += [f"vdup({d}, {a}, mask, {pos}, {merge});"]
        elif op in {"VINTLV", "VDINTLV"}:
            # The f8e8m0 two-output wrapper is absent in this compiler, but
            # both native halves are exposed. This is still a compound probe.
            code += [f"{op.lower()}v2({d}, {a}, {b}, LOWER);",
                     f"{op.lower()}v2(e{i}, {a}, {b}, HIGHER);"]
        elif op in {"VINTLVV2", "VDINTLVV2"}:
            code += [f"{op.lower()}({d}, {a}, {b}, LOWER);"]
        elif op in {"PINTLV", "PDINTLV"}:
            code += [f"{op.lower()}_{form}(pd{i}, pe{i}, pa{i}, pb{i});"]
        elif op in {"PPACK", "PUNPACK"}:
            code += [f"{op.lower()}(pd{i}, pa{i}, LOWER);"]
        elif op in {"PINTLVV2", "PDINTLVV2"}:
            code += [f"{op.lower()}_{form}(pd{i}, pa{i}, pb{i}, LOWER);"]
        elif op == "VPACK":
            code += [f"vpack({d}, {a}, LOWER, MODE_ZEROING);"]
        elif op == "VUNPACK":
            code += [f"vunpack({d}, {a}, LOWER);"]
        elif op == "VSELR":
            code += [f"vselr({d}, {a}, index{i});"]
        elif op == "VSQZ":
            code += [f"vsqz({d}, {a}, mask, MODE_NO_STORED);"]
        elif op == "VUSQZ":
            code += [f"vusqz({d}, count_mask{i});"]
        elif op in HIST:
            code += [f"{op.lower()}({d}, {a}, m, Bin_N0);"]
        elif op in {"VCMAXV2", "VCMINV2", "VCADDV2", "VPACKV2"}:
            code += [f"{op.lower()}({d}, {a});"]
        else:
            code += [f"{op.lower()}({d}, {a}, m);"]
    for i in ([0, 2, 1, 3] if mode == "forwarding" else range(n)):
        if op in PREDICATE:
            code += [f"vsel(d{i}, a{i}, b{i}, pd{i});"]
            if pair:
                code += [f"vsel(e{i}, a{i}, b{i}, pe{i});"]
        for j, reg in enumerate([f"d{i}", f"e{i}"] if pair else [f"d{i}"]):
            off = f"((group * {n} + {i}) * {outputs} + {j}) * 64"
            code += [f"vector_u32 out_{i}_{j} = (vector_u32){reg};",
                     f"vsts(out_{i}_{j}, ubA, {off}, NORM_B32, store_mask);"]
    code += ["}"]
    body = "\n".join("        " + s for s in code)
    insize, outsize = n * iterations * 64, n * iterations * outputs * 64
    cce = f'''// Generated by benchmarks/a6/exu2.py; raw-bit tensor transport.
// MB_KERNEL bench_{name}
// MB_TENSOR output A uint32 {outsize}
// MB_TENSOR input B uint32 {insize}
// MB_TENSOR input C uint32 {insize}
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#if __NPU_ARCH__ != 9201
#error "EXU2 probes require dav-920r1-vec"
#endif
extern "C" __global__ __aicore__ void bench_{name}(
    __gm__ uint32_t* A, __gm__ uint32_t* B, __gm__ uint32_t* C) {{
    __ubuf__ uint32_t* ubB = (__ubuf__ uint32_t*)get_imm(0);
    __ubuf__ uint32_t* ubC = (__ubuf__ uint32_t*)get_imm(0x4000);
    __ubuf__ uint32_t* ubA = (__ubuf__ uint32_t*)get_imm(0x8000);
    __ubuf__ uint32_t* offsets = (__ubuf__ uint32_t*)get_imm(0xc000);
    nddma_desc desc(nddma_desc::loop_desc({insize}, 1, 1));
    nddma_out_to_ub(ubB, B, 0, desc, 0, NEAREST_PADDING, 0);
    nddma_out_to_ub(ubC, C, 0, desc, 0, NEAREST_PADDING, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    __VEC_SCOPE__ {{
{body}
    }}
    offsets[0] = 0;
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
    scatter_ubuf_to_gm(A, ubA, 0, 0, offsets, 0, {outsize * 4}, 0, 1);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}}
'''
    meta = dict(name=name, op=op, form=form, mode=mode, width=n,
                input_dtype="uint32", output_dtype="uint32", src_lanes=64, dst_lanes=64,
                semantic_input_dtype=src, semantic_output_dtype=dst,
                target_aliases=aliases, expected_target_count=n * iterations,
                compound=op in COMPOUND, golden="not_implemented",
                note="Raw-bit transport; no numerical golden. Verify actual opcode and EXU routing.",
                loop_iterations=iterations, forwarding_edges=[[0, 2], [1, 3]] if mode == "forwarding" else [],
                layout="A0_A1_B0_B1_C0_C1" if mode == "forwarding" else "eight_independent_ops",
                implicit_dst_read="MERGE" in op or op in HIST | {"VUSQZ"},
                source=name + ".cce", source_sha256=hashlib.sha256(cce.encode()).hexdigest())
    return meta, cce


def capture(path):
    data = Path(path).read_bytes()
    isa = json.loads(data)
    return dict(source="configs/socs/DV121/isa.json", source_sha256=hashlib.sha256(data).hexdigest(),
                instructions={op: {"forms": {f: {} for f, spec in item.get("forms", {}).items()
                              if "EXU2" in spec.get("eligible_exus", item.get("eligible_exus", []))}}
                              for op, item in isa["instructions"].items()
                              if any("EXU2" in spec.get("eligible_exus", item.get("eligible_exus", []))
                                     for spec in item.get("forms", {}).values())})


def generate(snapshot, out, baseline):
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {"cases": []}
    for case in old["cases"]:
        path = out / case["source"]
        if path.parent != out or path.suffix != ".cce":
            raise ValueError(f"Invalid source path: {path}")
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != case["source_sha256"]:
            raise ValueError(f"Hand-edited probe: {path}")
    existing = {(c["op"], c["form"]) for c in baseline["cases"]}
    cases, coverage = [], []
    extras = set()
    def add(op, form):
        _, _, raw = specification(op, form)
        names = []
        for mode in (["ii", "forwarding"] if raw else ["ii"]):
            meta, cce = emit(op, form, mode)
            (out / meta["source"]).write_text(cce)
            cases.append(meta)
            names.append(meta["name"])
        return names, raw
    for op, item in snapshot["instructions"].items():
        for form in item["forms"]:
            if (op, form) in existing:
                coverage.append(dict(op=op, form=form, status="existing_probe", cases=[]))
                continue
            names, raw = add(op, form)
            coverage.append(dict(op=op, form=form, status="new_probe", cases=names,
                                 forwarding="probe" if raw else "not_applicable_in_this_template"))
            if op in {"VCMAX", "VCMIN", "VCADD"}:
                extras.add((op.replace("VC", "VCG", 1) + "V2", form))
                extras.add((op + "V2", specification(op, form)[1]))
            elif op in {"VCGMAX", "VCGMIN", "VCGADD"}:
                extras.add((op + "V2", form))
                extras.add(("VPACKV2", specification(op, form)[1]))
            elif op in {"VINTLV", "VDINTLV", "PINTLV", "PDINTLV", "VCPADD"}:
                extras.add((op + "V2", form))
    included = existing | {(c["op"], c["form"]) for c in cases}
    for op, form in sorted(extras - included):
        names, raw = add(op, form)
        coverage.append(dict(op=op, form=form, status="native_stage_extra", cases=names,
                             forwarding="probe" if raw else "not_applicable_in_this_template"))
    manifest = dict(schema_version=2, soc="Ascend910_9691", core_arch="dav-920r1-vec",
                    core_sim_dir="dav_9201", isa_sha256=snapshot["source_sha256"],
                    coverage=coverage, cases=cases)
    current = {c["source"] for c in cases}
    for case in old["cases"]:
        if case["source"] not in current:
            (out / case["source"]).unlink(missing_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isa", type=Path, help="Capture EXU2 routing from the DV121 ISA config")
    p.add_argument("--output", type=Path, default=DEFAULT)
    args = p.parse_args()
    if args.isa:
        SNAPSHOT.write_text(json.dumps(capture(args.isa), indent=2) + "\n")
    snapshot = json.loads(SNAPSHOT.read_text())
    baseline = json.loads((ROOT / "op_kernel/a6_instruction/manifest.json").read_text())
    manifest = generate(snapshot, args.output, baseline)
    print(f"Generated {len(manifest['cases'])} supplementary probes -> {args.output}")


if __name__ == "__main__":
    main()
