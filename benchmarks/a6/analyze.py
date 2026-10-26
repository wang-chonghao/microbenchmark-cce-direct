"""Conservative dump measurements; missing evidence never becomes a zero timing."""

from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
from . import semantic
import re
import struct

CYCLE = re.compile(r"\[(\d+)\]")
ID = re.compile(r"\(ID:\s*(\d+)\)")
PC = re.compile(r"\(PC:\s*(0x[0-9a-f]+)\)", re.I)
OP = re.compile(r"\b(RV_[A-Z0-9_]+)\b", re.I)
REG = re.compile(r"\b([VP])([dnmg])\[(\d+)\]", re.I)
EXU = re.compile(r"instr_name\s+(RV_\w+).*?instr_id\s+(\d+).*?exu_id:\s*(\d+)", re.I)
MEM = re.compile(r"\[(SEND_UB_RD\.PORT_\d+|STU_IB_BUF\.ISSUE)\].*?instr_name\s+(RV_\w+).*?instr_id\s+(\d+)", re.I)


def parse_events(path):
    events = []
    with path.open(errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            cycle, ident, pc, op = CYCLE.search(line), ID.search(line), PC.search(line), OP.search(line)
            if not all((cycle, ident, pc, op)):
                continue
            src, dst, operands = [], [], []
            for bank, role, reg in REG.findall(line):
                (dst if role.lower() == "d" else src).append(bank.upper() + reg)
                if role.lower() != "d":
                    operands.append(dict(role=bank.upper() + role.lower(), reg=bank.upper() + reg))
            if op[1].upper().startswith(("RV_VST", "RV_VSST", "RV_PST")):
                operands += [dict(role="store_data", reg=r) for r in dst]
                src, dst = sorted(set(src + dst)), []
            events.append(dict(cycle=int(cycle[1]), id=int(ident[1]), pc=pc[1].lower(),
                               op=op[1].upper(), src=src, dst=dst, src_operands=operands, line=lineno, file=str(path)))
    return events


def parse_exu(path):
    events = []
    with path.open(errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            match, cycle = EXU.search(line), CYCLE.search(line)
            if match and cycle:
                events.append(dict(cycle=int(cycle[1]), op=match[1].upper(), id=int(match[2]),
                                   exu=int(match[3]), file=str(path), line=lineno))
    return events


def parse_memory(path):
    events = []
    with path.open(errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            match, cycle = MEM.search(line), CYCLE.search(line)
            if match and cycle:
                events.append(dict(cycle=int(cycle[1]), resource=match[1].upper(),
                                   op=match[2].upper(), id=int(match[3]), file=str(path), line=lineno))
    return events


def numeric_check(run, case):
    """Check ordinary arithmetic from actual saved inputs, including fp16 rounding."""
    if case.get("semantic_validation"):
        return semantic.numeric_check(run, case)
    op, dtype = case["op"], case["output_dtype"]
    funcs = {"VADD": lambda a, b: a + b, "VSUB": lambda a, b: a - b,
             "VMUL": lambda a, b: a * b, "VDIV": lambda a, b: a / b,
             "VADDS": lambda a, b: a + .125, "VMULS": lambda a, b: a * .5,
             "VMAX": max, "VMIN": min, "VMAXS": lambda a, b: max(a, .75),
             "VMINS": lambda a, b: min(a, .75), "VABS": lambda a, b: abs(a),
             "VEXP": lambda a, b: math.exp(a), "VLN": lambda a, b: math.log(a),
             "VSQRT": lambda a, b: math.sqrt(a),
             "VEXPDIF": lambda a, b: math.exp(a - b),
             "VLDS": lambda a, b: a, "VSTS": lambda a, b: a}
    if op not in funcs or dtype not in {"fp32", "fp16"}:
        return "not_implemented", "Timing probe only; semantic golden still requires review."
    fmt = "f" if dtype == "fp32" else "e"

    def read(name):
        return [v[0] for v in struct.iter_unpack("<" + fmt, (run / "run" / name).read_bytes())]

    try:
        a, b, c = read("output_A.bin"), read("input_B.bin"), read("input_C.bin")
        if op == "VLN":
            b = c
        n, width = case["dst_lanes"], case["width"]
        total = width * case.get("loop_iterations", 1)
        if len(a) != n * total or len(b) != n * total or len(c) != n * total:
            return "fail", "Binary output/input size mismatch"
        expected = []
        for i in range(total):
            source = expected[(i-1)*n:i*n] if case["mode"] == "forwarding" and i else b[i*n:(i+1)*n]
            if "forwarding_edges" in case:
                edges = {consumer: producer for producer, consumer in case["forwarding_edges"]}
                lane = i % width
                producer = i - lane + edges[lane] if lane in edges else None
                source = b[i*n:(i+1)*n] if producer is None else expected[producer*n:(producer+1)*n]
            expected += [struct.unpack("<" + fmt, struct.pack("<" + fmt, funcs[op](x, y)))[0]
                         for x, y in zip(source, c[i*n:(i+1)*n])]
        tolerance = 5e-3 if dtype == "fp16" else 1e-4
        for i, (actual, target) in enumerate(zip(a, expected)):
            if not math.isfinite(actual) or abs(actual-target) > tolerance * max(1, abs(target)):
                return "fail", f"A[{i}]: actual={actual}, expected={target}"
        return "pass", ""
    except (OSError, ValueError, OverflowError, struct.error, ZeroDivisionError) as exc:
        return "fail", str(exc)


def analyze(run: Path, record):
    case = record["case"]
    memory_probe = case.get("memory_probe", False) or case["op"] in {"VLDS", "VSTS", "VSTUS", "VSTAS", "VSSTB"}
    result = dict(status="failed", golden="not_run", latency_min=None, ii_min=None,
                  forwarding_min=None, memory_interval_min=None, reason="", streams=[])
    result.update(timing_quality="not_measured", hardware_calibrated=False,
                  guard_status="not_requested", forwarding_by_operand={},
                  measurement_semantics=dict(latency="done_minus_popped", ii="same_exu_issue_gap",
                      forwarding="dependent_popped_gap_not_proven_bypass", memory="named_lsu_request_gap"),
                  missing_calibration_evidence=["source_ready", "frontend_supply", "resource_stalls", "saturation"])
    if record["returncode"] != 0:
        result["reason"] = record.get("input_error") or f"Build/model exit={record['returncode']}; see console/debug/model.log"
        return result
    if record["compile_only"]:
        result.update(status="compiled", reason="No model executed; no timings measured")
        return result
    if record.get("guard_required"):
        path = run / "run/guard_check.txt"
        result["guard_status"] = "pass" if path.exists() and path.read_text().startswith("pass:") else "missing"
        if result["guard_status"] != "pass":
            result["reason"] = "Missing successful GM guard/input immutability check"
            return result
    for name, expected in (record.get("input_profile") or {}).get("tensors", {}).items():
        path = run / "run" / f"input_{name}.bin"
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            result["reason"] = f"Runner input {name} does not match the semantic input profile"
            return result
    result["golden"], reason = numeric_check(run, case)
    result["golden_detail"] = reason
    if result["golden"] == "fail":
        result["reason"] = reason
        return result
    archive = record.get("dump_archive")
    log_root = (run / archive["directory"]).resolve() if archive else run / "run"
    logs = sorted(log_root.rglob("*.instr_popped_log.dump"))
    if not logs:
        result.update(status="missing_logs", reason="No instruction popped dump (stars-only is insufficient)")
        return result
    for popped in logs:
        prefix = popped.name.removesuffix(".instr_popped_log.dump")
        done_path = popped.with_name(prefix + ".instr_log.dump")
        starts = parse_events(popped)
        histogram = dict(Counter(e["op"] for e in starts))
        if not starts:
            continue
        stream = dict(core=prefix, histogram=histogram, latency=[], ii=[], forwarding=[], memory_interval=[], issues=[])
        result["streams"].append(stream)
        if case["compound"]:
            stream["issues"].append("Compound API: use the native A6 stage probes; inspect histogram")
            continue
        targets = [e for e in starts if e["op"] in case["target_aliases"]]
        if case.get("implicit_dst_read"):
            # Merging/accumulating instructions also consume the old destination.
            # Include it even when the dump only prints Vd once.
            for event in targets:
                event["src"] = sorted(set(event["src"] + event["dst"]))
                event["src_operands"] += [dict(role="old_" + r[0] + "d", reg=r) for r in event["dst"]]
        target_ids_for_routing = {e["id"] for e in targets}
        stream["observed_target_exus"] = sorted({event["exu"]
            for path in popped.parent.glob(prefix + ".rvec.*EXU*.dump")
            for event in parse_exu(path)
            if event["id"] in target_ids_for_routing and event["op"] in case["target_aliases"]})
        if len(targets) != case["expected_target_count"]:
            stream["issues"].append(f"Target count {len(targets)} != {case['expected_target_count']}; optimization/lowering/log format mismatch")
            continue
        byid = defaultdict(list)
        for e in starts:
            byid[e["id"]].append(e)
        if any(len(byid[e["id"]]) != 1 for e in targets):
            stream["issues"].append("Repeated dynamic IDs; refusing ambiguous pairing")
            continue
        if done_path.exists():
            dones = defaultdict(list)
            for e in parse_events(done_path):
                dones[(e["id"], e["pc"], e["op"])].append(e)
            for start in targets:
                matches = dones[(start["id"], start["pc"], start["op"])]
                if len(matches) == 1 and matches[0]["cycle"] >= start["cycle"]:
                    stream["latency"].append(dict(cycles=matches[0]["cycle"] - start["cycle"],
                                                   start=start, done=matches[0]))
            if len(stream["latency"]) != len(targets):
                stream["issues"].append("Missing/ambiguous completion IDs")
        else:
            stream["issues"].append("Missing completion log")
        if case["mode"] == "forwarding":
            # Loop iterations reuse PCs. Dynamic decode IDs order the emitted stream;
            # legacy straight-line probes can also be ordered by static PC.
            last_writer = {}
            target_ids = {e["id"] for e in targets}
            for consumer in sorted(starts, key=lambda e: e["id"] if "loop_iterations" in case else int(e["pc"], 16)):
                if consumer["id"] in target_ids:
                    producers = {last_writer[r]["id"]: last_writer[r] for r in consumer["src"]
                                 if r in last_writer and last_writer[r]["id"] in target_ids}
                    for producer in producers.values():
                        if case.get("cross_forwarding") and (producer["op"] not in case["producer_aliases"]
                                or consumer["op"] not in case["consumer_aliases"]):
                            continue
                        delta = consumer["cycle"] - producer["cycle"]
                        if delta >= 0:
                            stream["forwarding"].append(dict(cycles=delta, producer=producer,
                                consumer=consumer, dependency_roles=sorted({o["role"] for o in consumer["src_operands"]
                                    if o["reg"] in producer["dst"]}),
                                other_source_readiness="unverified", event_boundary="popped"))
                if not consumer["op"].startswith(("RV_VST", "RV_VSST")):
                    for reg in consumer["dst"]:
                        last_writer[reg] = consumer
            if not stream["forwarding"]:
                stream["issues"].append("No confirmed vector-register RAW edge in emitted instructions")
        if case["mode"] == "ii":
            if memory_probe:
                resources = defaultdict(list)
                memory_logs = list(popped.parent.glob(prefix + ".rvec.LSU.dump"))
                if archive and archive.get("keep_all_dumps"):
                    original_prefix = prefix.split("__")[-1]
                    memory_logs += list((run / "run").rglob(original_prefix + ".rvec.LSU.dump"))
                all_memory_events = []
                for path in set(memory_logs):
                    for event in parse_memory(path):
                        resources[event["resource"]].append(event)
                        all_memory_events.append(event)
                global_counts = Counter(e["id"] for e in all_memory_events)
                if any(n > 1 for n in global_counts.values()):
                    stream["issues"].append("Multi-request/port instructions excluded; no instruction-level memory II claimed")
                target_ids = {e["id"] for e in targets}
                for events in resources.values():
                    events.sort(key=lambda e: e["cycle"])
                    for a, b in zip(events, events[1:]):
                        if a["id"] in target_ids and b["id"] in target_ids and global_counts[a["id"]] == global_counts[b["id"]] == 1:
                            delta = b["cycle"] - a["cycle"]
                            if delta > 0:
                                stream["memory_interval"].append(dict(cycles=delta, previous=a, current=b))
                stream["issues"].append("Memory interval uses named LSU request/issue resource, not EXU II")
                continue
            exu_logs = sorted(popped.parent.glob(prefix + ".rvec.*EXU*.dump"))
            if not exu_logs:
                stream["issues"].append("Missing EXU issue log: do not infer II from global popped intervals")
            by_exu = defaultdict(list)
            for path in exu_logs:
                for e in parse_exu(path):
                    by_exu[(path.name, e["exu"])].append(e)
            target_by_id = {e["id"]: e for e in targets}
            for _, events in by_exu.items():
                events.sort(key=lambda e: e["cycle"])
                for prev, cur in zip(events, events[1:]):
                    a, b = target_by_id.get(prev["id"]), target_by_id.get(cur["id"])
                    if a is None or b is None or prev["id"] == cur["id"]:
                        continue
                    if prev["op"] != a["op"] or cur["op"] != b["op"]:
                        continue
                    if not a["dst"] or not b["dst"]:
                        continue
                    if not case.get("no_register_source") and case["op"] not in {"VCI", "VBR", "VDUPS"} and (not a["src"] or not b["src"]):
                        continue
                    # Exclude RAW, WAR and WAW pairs, including predicate operands.
                    if set(a["dst"]) & set(b["src"] + b["dst"]) or set(b["dst"]) & set(a["src"]):
                        continue
                    delta = cur["cycle"] - prev["cycle"]
                    if delta > 0:
                        stream["ii"].append(dict(cycles=delta, previous=prev, current=cur))
            if not stream["ii"]:
                stream["issues"].append("No same-EXU independent adjacent issue samples")
    if case["compound"]:
        result.update(status="compound_api", reason="No single-instruction metrics; native-stage cases supplied")
        return result
    for key in ("latency", "ii", "forwarding", "memory_interval"):
        samples = [s["cycles"] for stream in result["streams"] for s in stream[key]]
        if samples:
            result[key + "_min"] = min(samples)
            result["timing_quality"] = "observed_uncalibrated"
    for stream in result["streams"]:
        for sample in stream["forwarding"]:
            # Keep simultaneous multi-operand dependence distinct from isolated paths.
            role = "+".join(sample["dependency_roles"]) or "unknown"
            old = result["forwarding_by_operand"].get(role)
            result["forwarding_by_operand"][role] = min(old, sample["cycles"]) if old is not None else sample["cycles"]
    needed = {"latency": "latency_min", "ii": "ii_min", "forwarding": "forwarding_min"}[case["mode"]]
    if case["mode"] == "ii" and memory_probe:
        needed = "memory_interval_min"
    result["status"] = "observed" if result[needed] is not None else "invalid_logs"
    if (needed == "memory_interval_min" and archive and not archive.get("keep_all_dumps") and not archive.get("keep_lsu_dumps")
            and result["latency_min"] is not None):
        result["status"] = "not_collected"
        result["reason"] = "Default six-dump policy excludes LSU; use --keep-all-dumps to measure memory intervals"
        return result
    issues = [i for s in result["streams"] for i in s["issues"]]
    result["reason"] = "; ".join(issues) or "Observed intervals, not automatically calibrated hardware constants"
    return result
