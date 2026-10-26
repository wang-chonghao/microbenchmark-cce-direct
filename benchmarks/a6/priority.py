"""Deterministic breadth-first test order, independent of generated source order."""

import json
from pathlib import Path

from .generate import ROOT
from .semantic import FORMATS, FP8_NORMALS

# Engineering test priorities, not a claim about measured instruction frequency.
COMMON_OPS = (
    "VADD", "VMUL", "VSUB", "VMAX", "VMIN", "VEXP", "VDIV", "VLN", "VSQRT",
    "VEXPDIF", "VCMAX", "VCMAXV2", "VCADD", "VCADDV2", "VCMIN", "VCMINV2",
    "VCGMAX", "VCGMAXV2", "VCGADD", "VCGADDV2", "VCGMIN", "VCGMINV2",
    "VADDS", "VMULS", "VMAXS", "VMINS", "VABS", "VNEG", "VRELU",
    "VMADD", "VMULA", "VAXPY", "VDUP", "VDUPS", "VSEL", "VMOV",
    "VLDS", "VSTS", "VLD", "VST", "VLDUS", "VSTUS", "VLDAS", "VSTAS",
    "VGATHER2", "VSCATTER", "VLDS_TO_VADD", "VADD_TO_VSTS",
)
FORM_ORDER = ("fp32", "fp16", "bf16", "int32", "uint32", "b32",
              "int16", "uint16", "b16", "b8", "int8", "uint8",
              "f8e4m3", "f8e5m2", "f8e8m0", "f4e1m2x2", "f4e2m1x2")
PRIMARY = set(FORM_ORDER[:6])
SECONDARY = set(FORM_ORDER[6:12])
ALIASES = dict(f32="fp32", f16="fp16", s32="int32", u32="uint32",
               s16="int16", u16="uint16", s8="int8", u8="uint8")


def exu2_candidates():
    snapshot = json.loads(Path(__file__).with_name("exu2_inventory.json").read_text())
    pairs = {(op, form) for op, item in snapshot["instructions"].items() for form in item["forms"]}
    supplement = json.loads((ROOT / "op_kernel/a6_exu2/manifest.json").read_text())
    pairs.update((r["op"], r["form"]) for r in supplement["coverage"] if r["status"] == "native_stage_extra")
    return pairs


def form_priority(form):
    parts = [ALIASES.get(p, p) for p in form.split("_to_")]
    band = max(0 if p in PRIMARY else 1 if p in SECONDARY else 2 for p in parts)
    rank = max(FORM_ORDER.index(p) if p in FORM_ORDER else len(FORM_ORDER) for p in parts)
    return band, rank


def priority_key(case, exu2):
    op, form = case["op"], case["form"]
    high = op in COMMON_OPS or (op, form) in exu2
    band, form_rank = form_priority(form)
    unsupported_input = (case.get("input_dtype") == "uint32" and case.get("semantic_input_dtype")
                         and case["semantic_input_dtype"] not in set(FORMATS) | set(FP8_NORMALS) | {"bf16"})
    # Complete basic II coverage before adding RAW chains or wider stress variants.
    mode = 0 if case["mode"] == "ii" and case.get("width", 8) <= 8 else 1 if case["mode"] == "forwarding" else 2
    op_rank = COMMON_OPS.index(op) if op in COMMON_OPS else len(COMMON_OPS)
    return (bool(unsupported_input), band * 2 + (not high), mode, op_rank, op,
            form_rank, form, case.get("dependency_role", "vn"), case["name"])


def order_cases(cases, order="priority", exu2=None):
    if order == "manifest":
        return list(cases)
    if order != "priority":
        raise ValueError(f"Unknown execution order: {order}")
    pairs = exu2_candidates() if exu2 is None else exu2
    return sorted(cases, key=lambda case: priority_key(case, pairs))
