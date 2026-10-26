"""Semantic inputs and bounded references for raw-word timing probes.

Unsupported layouts/operations remain explicitly unverified. Padding and inactive
lanes are not numerical outputs; GM guards are checked separately by the runner.
"""

import hashlib
import json
import math
from pathlib import Path
import struct

FORMATS = {"fp32": "f", "fp16": "e", "int8": "b", "uint8": "B",
           "int16": "h", "uint16": "H", "int32": "i", "uint32": "I", "b32": "I"}
FP8_NORMALS = {"f8e4m3": (0x30, 0x38, 0x40), "f8e5m2": (0x38, 0x3c, 0x40),
               "f8e8m0": (0x7e, 0x7f, 0x80)}  # Exactly 0.5, 1, 2.
OPS = {"VADD", "VSUB", "VMUL", "VMAX", "VMIN", "VABSDIF", "VABS", "VNEG",
       "VRELU", "VMOV", "VADDS", "VMULS", "VMAXS", "VMINS", "VAXPY",
       "VAND", "VOR", "VXOR", "VNOT", "VLDS", "VLD", "VSTS", "VST",
       "VLDS_TO_VADD", "VADD_TO_VSTS"}


def encode(value, dtype):
    if dtype == "bf16":
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bits += 0x7fff + ((bits >> 16) & 1)
        return struct.pack("<H", (bits >> 16) & 0xffff)
    fmt = FORMATS[dtype]
    if fmt not in "fe":
        bits = struct.calcsize(fmt) * 8
        value = int(value) % (1 << bits)
        if fmt.islower() and value >= (1 << (bits - 1)):
            value -= 1 << bits
    return struct.pack("<" + fmt, value)


def decode(data, dtype):
    if dtype == "bf16":
        return [struct.unpack("<f", struct.pack("<I", x[0] << 16))[0]
                for x in struct.iter_unpack("<H", data)]
    return [v[0] for v in struct.iter_unpack("<" + FORMATS[dtype], data)]


def input_bytes(dtype, size, seed=0, tensor_index=0):
    if dtype not in FORMATS and dtype not in FP8_NORMALS and dtype != "bf16":
        raise ValueError(f"No reviewed semantic input encoding for {dtype}; raw bits are not a substitute")
    unit = 1 if dtype in FP8_NORMALS else len(encode(1, dtype))
    if size % unit:
        raise ValueError("Tensor byte size does not match semantic dtype")
    result = bytearray()
    for i in range(size // unit):
        n = (i + seed + tensor_index * 3) % 7
        if dtype in FP8_NORMALS:
            result.append(FP8_NORMALS[dtype][n % 3])
        else:
            value = .5 + n * .125 if dtype in {"fp32", "fp16", "bf16"} else 1 + n
            result.extend(encode(value, dtype))
    return bytes(result)


def prepare_inputs(case, manifest, directory: Path, seed=0):
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, seed=seed, semantic_dtype=case["semantic_input_dtype"], tensors={})
    for index, tensor in enumerate(manifest.tensors):
        if tensor.direction not in {"input", "inout"}:
            continue
        if tensor.dtype != "uint32":
            raise ValueError("Raw semantic input preparation requires uint32 transport")
        data = input_bytes(case["semantic_input_dtype"], tensor.elements * 4, seed, index)
        (directory / f"input_{tensor.name}.bin").write_bytes(data)
        report["tensors"][tensor.name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    (directory / "input_profile.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def reference_available(case):
    dtype = case.get("semantic_input_dtype")
    if case["op"] in {"VAND", "VOR", "VXOR", "VNOT"} and dtype in {"fp16", "fp32", "bf16"}:
        return False
    return (case["op"] in OPS and dtype in (set(FORMATS) | {"bf16"})
            and case.get("semantic_output_dtype") == dtype
            and case.get("src_lanes") == case.get("dst_lanes") == 128
            and case.get("dependency_role", "vn") in {"vn", "vm", "old_vd"})


def evaluate(op, x, y, old):
    functions = {"VADD": lambda: x+y, "VSUB": lambda: x-y, "VMUL": lambda: x*y,
                 "VMAX": lambda: max(x,y), "VMIN": lambda: min(x,y), "VABSDIF": lambda: abs(x-y),
                 "VABS": lambda: abs(x), "VNEG": lambda: -x, "VRELU": lambda: max(x,0),
                 "VMOV": lambda: x, "VADDS": lambda: x+1, "VMULS": lambda: x*3,
                 "VMAXS": lambda: max(x,1), "VMINS": lambda: min(x,1),
                 "VAXPY": lambda: old + .5*x, "VAND": lambda: int(x)&int(y),
                 "VOR": lambda: int(x)|int(y), "VXOR": lambda: int(x)^int(y), "VNOT": lambda: ~int(x)}
    if op in {"VLDS", "VLD", "VSTS", "VST"}:
        return x
    if op in {"VLDS_TO_VADD", "VADD_TO_VSTS"}:
        return x + 1
    return functions[op]()


def numeric_check(run, case):
    if not reference_available(case):
        return "not_implemented", "Semantic inputs supplied, but no reviewed reference for this opcode/layout"
    dtype, op = case["semantic_input_dtype"], case["op"]
    if op in {"VAND", "VOR", "VXOR", "VNOT"} and dtype in {"fp16", "fp32", "bf16"}:
        return "not_implemented", "Floating bitwise reference not implemented"
    try:
        a, b, c = [(run / "run" / name).read_bytes() for name in ("output_A.bin", "input_B.bin", "input_C.bin")]
        width, rounds = case["width"], case["loop_iterations"]
        if any(len(data) != width * rounds * 512 for data in (a,b,c)):
            return "fail", "Raw tensor size mismatch"
        edges = dict((consumer, producer) for producer, consumer in case.get("forwarding_edges", []))
        expected = []
        for block in range(width * rounds):
            start = block * 512
            x, y, actual = [decode(data[start:start+256], dtype) for data in (b,c,a)]
            old = list(y)
            lane = block % width
            if lane in edges:
                previous = expected[block - lane + edges[lane]]
                role = case.get("dependency_role", "vn")
                if role == "vn": x = previous
                elif role == "vm": y = previous
                elif role == "old_vd": old = previous
            values = [decode(encode(evaluate(op, u,v,w), dtype), dtype)[0] for u,v,w in zip(x,y,old)]
            expected.append(values)
            tolerance = {"fp32": 1e-6, "fp16": 2e-3, "bf16": 1e-2}.get(dtype, 0)
            for lane, (value, target) in enumerate(zip(actual,values)):
                if not math.isfinite(value) or abs(value-target) > tolerance * max(abs(target), .01):
                    return "fail", f"block={block} lane={lane}: actual={value}, expected={target}"
        return "pass", "Checked all 256 valid output bytes per 512-byte tile; padding is undefined"
    except (OSError, KeyError, ValueError, OverflowError, struct.error) as exc:
        return "fail", str(exc)
