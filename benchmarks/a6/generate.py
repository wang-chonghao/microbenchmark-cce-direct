"""Generate reviewable CCE probes from an ISA inventory, not its A5 timings."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "op_kernel/a6_instruction"
TYPES = {
    "fp32": ("float", "vector_f32", 32),
    "fp16": ("half", "vector_f16", 16),
    "bf16": ("bfloat16_t", "vector_bf16", 16),
    "int32": ("int32_t", "vector_s32", 32),
    "uint32": ("uint32_t", "vector_u32", 32),
    "uint16": ("uint16_t", "vector_u16", 16),
    "uint8": ("uint8_t", "vector_u8", 8),
    "int8": ("int8_t", "vector_s8", 8),
}
BINARY = set("VADD VSUB VMUL VDIV VMAX VMIN VAND VOR VEXPDIF".split())
SCALAR = {"VADDS": "0.125", "VMULS": "0.5", "VMAXS": "0.75", "VMINS": "0.75",
          "VSHLS": "1", "VSHRS": "1"}
COMPOUND = {"VCMAX", "VCMIN", "VCADD", "VCGMAX"}
STORES = {"VSTS", "VSTUS", "VSTAS", "VSSTB"}
NO_RAW = STORES | {"VLDS", "VCI", "VBR", "VDUPS", "VPACK"}
NATIVE_REDUCTIONS = {"VCGMAXV2", "VCGMINV2", "VCGADDV2", "VCMAXV2", "VCMINV2", "VCADDV2"}
CONVERSIONS = {
    "VCVT_F16_TO_F32": ("fp16", "fp32", "m, PART_EVEN, MODE_ZEROING"),
    "VCVT_BF16_TO_F32": ("bf16", "fp32", "m, PART_EVEN, MODE_ZEROING"),
    "VCVT_F32_TO_F16": ("fp32", "fp16", "m, ROUND_R, RS_ENABLE, PART_EVEN"),
    "VCVT_F32_TO_BF16": ("fp32", "bf16", "m, ROUND_R, RS_ENABLE, PART_EVEN"),
    "VCVT_S32_TO_U8": ("int32", "uint8", "m, RS_ENABLE, PART_P0, MODE_ZEROING"),
    "VCVT_F32_TO_S32": ("fp32", "int32", "m, ROUND_R, RS_ENABLE"),
    "VCVT_S32_TO_F32": ("int32", "fp32", "m, ROUND_R"),
    "VMULSCVT": ("fp32", "fp16", ""),
}


def describe(op, form):
    supported = (BINARY | set(SCALAR) | COMPOUND | STORES | NO_RAW | set(CONVERSIONS)
                 | NATIVE_REDUCTIONS | {"VDUP", "VCMP_EQ", "VSEL", "VABS", "VSQRT", "VLN", "VEXP"})
    if op not in supported:
        raise ValueError(f"No reviewed A6 template for {op}/{form}")
    src = dst = {"b32": "uint32", "b16": "bf16"}.get(form, form)
    note = ""
    if op in CONVERSIONS:
        src, dst, _ = CONVERSIONS[op]
        note = "Different input/output types: no type-preserving self RAW probe here; bit-reinterpretation experiments need separate validation."
    if op == "VPACK":
        src, dst = "uint32", "uint16"
        note = "Bit packing u32 -> u16 (LOWER), not bf16 arithmetic."
    if op in {"VSHLS", "VSHRS"}:
        src = dst = "uint32" if form == "fp32" else "uint16"
        note = "ISA fp label denotes bit width here: unsigned integer shift, not floating arithmetic."
    if op in COMPOUND:
        note = "A6 API expands to multiple instructions; do not assign a single ISA latency."
    if op in NO_RAW:
        note += " No self vector-register RAW path in this form."
    return src, dst, note.strip()


def aliases(op):
    special = {
        "VLDS": ["RV_VLDI", "RV_VLDS"], "VSTS": ["RV_VSTI", "RV_VSTS"],
        "VDUP": ["RV_VDUP"], "VABS": ["RV_VABS", "RV_VABS_FP"],
        "VCVT_F16_TO_F32": ["RV_VCVT_F2F"], "VCVT_F32_TO_F16": ["RV_VCVT_F2F"],
        "VCVT_BF16_TO_F32": ["RV_VCVT_F2F"], "VCVT_F32_TO_BF16": ["RV_VCVT_F2F"],
        "VCVT_S32_TO_U8": ["RV_VCVT_I2I"], "VCVT_F32_TO_S32": ["RV_VCVT_F2I"],
        "VCVT_S32_TO_F32": ["RV_VCVT_I2F"],
    }
    return special.get(op, ["RV_" + op])


def operation(op, d, a, b, src, i):
    ctype = TYPES[src][0]
    if op in CONVERSIONS and op != "VMULSCVT":
        return f"vcvt({d}, {a}, {CONVERSIONS[op][2]});"
    if op == "VMULSCVT":
        return f"vmulscvt({d}, {a}, 0.5f, m, PART_EVEN);"
    if op == "VEXPDIF":
        return f"vexpdif({d}, {a}, {b}, m, PART_EVEN);"
    if op in BINARY:
        return f"{op.lower()}({d}, {a}, {b}, m);"
    if op in SCALAR:
        return f"{op.lower()}({d}, {a}, ({ctype}){SCALAR[op]}, m);"
    if op == "VDUP":
        return f"vdup({d}, {a}, m, POS_LOWEST, MODE_ZEROING);"
    if op == "VDUPS":
        return f"vdup({d}, ({ctype}){i + 1}, m, MODE_ZEROING);"
    if op == "VCI":
        return f"vci({d}, (int32_t){i + 1});"
    if op == "VBR":
        return f"vbr({d}, ({ctype}){i + 1});"
    if op == "VPACK":
        return f"vpack({d}, {a}, LOWER, MODE_ZEROING);"
    if op == "VCMP_EQ":
        return f"vcmp_eq(p{i}, {a}, {b}, m);"
    if op == "VSEL":
        return f"vsel({d}, {a}, {b}, select_mask);"
    if op in {"VCMAXV2", "VCMINV2", "VCADDV2"}:
        return f"{op.lower()}({d}, {a});"
    return f"{op.lower()}({d}, {a}, m);"


def render(op, form, mode, width, iterations=None):
    src, dst, note = describe(op, form)
    st, sv, sb = TYPES[src]
    dt, dv, db = TYPES[dst]
    sl, dl = 2048 // sb, 2048 // db
    count = 1 if mode == "latency" else (2 if mode == "forwarding" else width)
    name = f"{op.lower()}_{form}_{mode}" + (f"_w{width}" if mode == "ii" else "")
    if iterations is not None:
        if mode not in {"ii", "forwarding"} or not 1 <= iterations <= 8:
            raise ValueError("Loop probes require ii/forwarding and 1..8 iterations")
        count = 4 if mode == "forwarding" else 8
        name = f"{op.lower()}_{form}_{mode}"
    repeats = iterations or 1
    def offset(i, lanes):
        return str(i * lanes) if iterations is None else f"(group * {count} + {i}) * {lanes}"
    # Leave space for special store layouts and alignment tails; no UB aliasing.
    elements_in = repeats * count * sl
    elements_out = repeats * count * dl
    code = [f"vector_bool m = pset_b{sb}(PAT_ALL);",
            f"vector_bool out_mask = pset_b{db}(PAT_ALL);"]
    if op == "VSEL":
        code += [f"vector_bool select_mask = pset_b{sb}(PAT_H);"]
    prefix_count = len(code)
    for i in range(count):
        code += [f"{sv} a{i};", f"{dv} d{i};"]
        if op in BINARY or op in {"VCMP_EQ", "VSEL"}:
            code += [f"{sv} b{i};"]
        if op == "VCMP_EQ":
            code += [f"vector_bool p{i};"]
    for i in range(count):
        if op != "VLDS":
            # C is initialized in [1.69, 2.65], so log(log(C)) stays in-domain.
            source_buffer = "ubC" if op == "VLN" else "ubB"
            code += [f"vlds(a{i}, {source_buffer}, {offset(i, sl)}, NORM);"]
        if op in BINARY or op in {"VCMP_EQ", "VSEL"}:
            code += [f"vlds(b{i}, ubC, {offset(i, sl)}, NORM);"]
    for i in range(count):
        a = f"d{i-1}" if mode == "forwarding" and i else f"a{i}"
        if iterations is not None and mode == "forwarding":
            a = f"d{i-2}" if i >= 2 else f"a{i}"
        code += [f"// TARGET {i}: {op} ({mode})"]
        if op == "VLDS":
            code += [f"vlds(d{i}, ubB, {offset(i, sl)}, NORM);"]
        elif op == "VSTS":
            code += [f"vsts(a{i}, ubA, {offset(i, dl)}, NORM_B{db}, out_mask);"]
        elif op in {"VSTUS", "VSTAS"}:
            code += [f"vector_align align{i} = {{}};",
                     f"__ubuf__ {dt} *ptr{i} = ubA + {offset(i, dl)};",
                     f"vstus(align{i}, {dl}, a{i}, ptr{i}, POST_UPDATE);",
                     f"vstas(align{i}, ptr{i}, 0);"]
        elif op == "VSSTB":
            # Independent 256-byte tiles; one 32-byte block is active per tile.
            code += [f"vsstb(a{i}, ubA + {offset(i, dl)}, 0, pset_b16(PAT_VL16));"]
        else:
            code += [operation(op, f"d{i}", a, f"b{i}", src, i)]
    # Store every result, including intermediate RAW results, to prevent DCE.
    if op not in STORES:
        store_order = [0, 2, 1, 3] if iterations is not None and mode == "forwarding" else range(count)
        for i in store_order:
            if op == "VCMP_EQ":
                code += [f"vsel(d{i}, a{i}, b{i}, p{i});"]
            code += [f"vsts(d{i}, ubA, {offset(i, dl)}, NORM_B{db}, out_mask);"]
    if iterations is not None:
        # Explicit unroll2 for RAW: A0,A1,B0,B1,C0,C1; C stores both stages.
        code = code[:prefix_count] + ["#pragma unroll(1)",
            f"for (uint16_t group = 0; group < {iterations}; ++group) {{"] + [
                "    " + line for line in code[prefix_count:]] + ["}"]
    body = "\n".join("        " + line for line in code)
    cce = f'''// Generated by benchmarks/a6/generate.py. Edit the generator, then regenerate.
// {op}/{form}:{(' ' + note) if note else ''}
// MB_KERNEL bench_{name}
// MB_TENSOR output A {dst} {elements_out}
// MB_TENSOR input B {src} {elements_in}
// MB_TENSOR input C {src} {elements_in}
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#if __NPU_ARCH__ != 9201
#error "A6 probes require --cce-aicore-arch=dav-920r1-vec"
#endif
extern "C" __global__ __aicore__
void bench_{name}(__gm__ {dt}* A, __gm__ {st}* B, __gm__ {st}* C)
{{
    __ubuf__ {st}* ubB = (__ubuf__ {st}*)get_imm(0x0000);
    __ubuf__ {st}* ubC = (__ubuf__ {st}*)get_imm(0x4000);
    __ubuf__ {dt}* ubA = (__ubuf__ {dt}*)get_imm(0x8000);
    __ubuf__ uint32_t* offsets = (__ubuf__ uint32_t*)get_imm(0xc000);
    nddma_desc desc(nddma_desc::loop_desc({elements_in}, 1, 1));
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
    scatter_ubuf_to_gm((__gm__ uint32_t*)A, (__ubuf__ uint32_t*)ubA,
                       0, 0, offsets, 0, {repeats * count * 256}, 0, 1);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}}
'''
    meta = dict(name=name, op=op, form=form, mode=mode, width=count,
                input_dtype=src, output_dtype=dst, src_lanes=sl, dst_lanes=dl,
                target_aliases=aliases(op), expected_target_count=repeats * count,
                compound=op in COMPOUND, note=note,
                golden="pending", source=f"{name}.cce",
                source_sha256=hashlib.sha256(cce.encode()).hexdigest())
    if iterations is not None:
        meta.update(loop_iterations=iterations,
                    layout="A0_A1_B0_B1_C0_C1" if mode == "forwarding" else "eight_independent_ops",
                    forwarding_edges=[[0, 2], [1, 3]] if mode == "forwarding" else [])
    return meta, cce


def generate(isa, out, iterations=2):
    raw = json.loads(Path(isa).read_text(encoding="utf-8-sig"))
    out.mkdir(parents=True, exist_ok=True)
    previous = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {"cases": []}
    # Never replace or remove hand-edited generated sources silently.
    for item in previous["cases"]:
        path = out / item["source"]
        if path.parent != out or path.suffix != ".cce":
            raise ValueError(f"Invalid generated source path: {path}")
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != item["source_sha256"]:
            raise ValueError(f"Modified generated source: {path}; use another output directory")
    cases, coverage = [], []
    inventory = [(op, form, False) for op, item in raw["instructions"].items()
                 for form in item["forms"]]
    # Measure the native reduction stages separately from the compatibility APIs.
    inventory += [(op, form, True) for op in
                  ("VCGMAXV2", "VCGMINV2", "VCGADDV2", "VCMAXV2", "VCMINV2", "VCADDV2")
                  for form in ("fp32", "fp16")]
    inventory += [("VDUPS", form, True) for form in ("fp32", "fp16", "int8")]
    for op, form, extra in inventory:
        src, dst, note = describe(op, form)
        self_raw = op not in NO_RAW | set(CONVERSIONS) | {"VCMP_EQ"} | COMPOUND
        modes = [("ii", 8)]
        if self_raw:
            modes.append(("forwarding", 2))
        names = []
        for mode, width in modes:
            meta, cce = render(op, form, mode, width, iterations=iterations)
            (out / meta["source"]).write_text(cce, encoding="utf-8")
            cases.append(meta)
            names.append(meta["name"])
        coverage.append(dict(op=op, form=form, a6_native_extra=extra, cases=names,
                             forwarding="probe" if self_raw else "not_applicable",
                             note=note))
    current_sources = {c["source"] for c in cases}
    for item in previous["cases"]:
        if item["source"] not in current_sources:
            (out / item["source"]).unlink(missing_ok=True)
    manifest = dict(schema_version=2, isa_sha256=hashlib.sha256(Path(isa).read_bytes()).hexdigest(),
                    soc="Ascend910_9691", core_arch="dav-920r1-vec", core_sim_dir="dav_9201",
                    coverage=coverage, cases=cases)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isa", type=Path, default=Path(__file__).with_name("isa_inventory.json"))
    p.add_argument("--output", type=Path, default=DEFAULT_CASES)
    p.add_argument("--iterations", type=int, default=2, help="Loop trips within one VF; default 2, max 8")
    args = p.parse_args()
    if not 1 <= args.iterations <= 8:
        p.error("Loop iterations must be 1..8 (bounded UB use)")
    m = generate(args.isa, args.output, args.iterations)
    print(f"Generated {len(m['cases'])} probes; {len(m['coverage'])} forms -> {args.output}")


if __name__ == "__main__":
    main()
