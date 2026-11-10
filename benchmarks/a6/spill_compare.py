"""Differential checks only: an original kernel is not a mathematical golden."""
import math
from pathlib import Path
import struct


def compare_bytes(reference, actual, regions, atol=1e-5, rtol=1e-3):
    if len(reference) != len(actual):
        raise ValueError('Output size mismatch')
    mismatch = nonfinite = checked = 0
    maximum = 0.0
    codes = {'float': 'f', 'half': 'e', 'int32_t': 'i'}
    for region in regions:
        code = codes[region['dtype']]
        start, size = region['gm_offset'], region['bytes']
        if start + size > len(actual) or size % struct.calcsize(code):
            raise ValueError('Invalid output region')
        lhs = struct.iter_unpack('<' + code, reference[start:start+size])
        rhs = struct.iter_unpack('<' + code, actual[start:start+size])
        for (expected,), (observed,) in zip(lhs, rhs):
            checked += 1
            if not math.isfinite(expected) or not math.isfinite(observed):
                nonfinite += 1
                mismatch += 1
                continue
            delta = abs(expected - observed)
            maximum = max(maximum, delta)
            equal = observed == expected if code == 'i' else delta <= atol + rtol * abs(expected)
            mismatch += not equal
    return dict(passed=mismatch == 0, elements=checked, mismatches=mismatch,
                nonfinite=nonfinite, max_abs_error=maximum, atol=atol, rtol=rtol,
                kind='A6 original versus A6 software; NOT independent golden')


def compare_pairs(cases, records, output):
    groups = {}
    statuses = {r['name']: r['status'] for r in records}
    for c in cases:
        groups.setdefault(c['row_id'], {})[c['role']] = c
    result = []
    for row, group in groups.items():
        if 'software' not in group:
            continue
        record = dict(row_id=row, status='not_compared')
        a, b = group.get('original'), group['software']
        if not a or any(statuses.get(c['name']) != 'executed_unvalidated' for c in (a, b)):
            record['reason'] = 'Both versions must complete in this batch'
        elif a['input_sha256'] != b['input_sha256'] or a['output_bytes'] != b['output_bytes']:
            record['reason'] = 'Input or output layout differs'
        elif [(o['gm_offset'], o['bytes'], o['dtype']) for o in a['outputs']] != [
                (o['gm_offset'], o['bytes'], o['dtype']) for o in b['outputs']]:
            record['reason'] = 'Output dtype/region mismatch'
        else:
            try:
                data = [(Path(output)/'_work'/c['name']/'run/output_output.bin').read_bytes() for c in (a,b)]
                record['comparison'] = compare_bytes(*data, a['outputs'])
                record['status'] = 'matched' if record['comparison']['passed'] else 'mismatch'
            except (OSError, ValueError, KeyError) as exc:
                record['reason'] = str(exc)
                record['status'] = 'error'
        result.append(record)
    return result
