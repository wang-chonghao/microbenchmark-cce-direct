"""独立验证 FP32 输入、FP16 输出的按行稳定 softmax；仅依赖 Python 标准库。"""

import argparse
import json
import math
from pathlib import Path
import struct


def read_values(path, code, count):
    data = Path(path).read_bytes()
    expected = count*struct.calcsize('<'+code)
    if len(data) != expected:
        raise ValueError(f'{path} 大小为 {len(data)}B，应为 {expected}B')
    return [item[0] for item in struct.iter_unpack('<'+code, data)]


def check(input_file, output_file, rows=8, cols=1024, atol=1e-6, rtol=1e-3):
    if rows <= 0 or cols <= 0:
        raise ValueError('rows 和 cols 必须为正整数')
    if not all(math.isfinite(v) and v >= 0 for v in (atol, rtol)):
        raise ValueError('容差必须为非负有限数')
    inputs = read_values(input_file, 'f', rows*cols)
    outputs = read_values(output_file, 'e', rows*cols)
    if not all(map(math.isfinite, inputs)):
        raise ValueError('输入包含 NaN 或 Inf')
    mismatches = []
    bad = 0
    max_abs = max_rel = 0.0
    row_sums = []
    for row in range(rows):
        values = inputs[row*cols:(row+1)*cols]
        maximum = max(values)
        exponent = [math.exp(value-maximum) for value in values]
        denominator = math.fsum(exponent)
        actual_row = outputs[row*cols:(row+1)*cols]
        row_sums.append(math.fsum(actual_row) if all(map(math.isfinite, actual_row)) else None)
        for column, (actual, numerator) in enumerate(zip(actual_row, exponent)):
            expected = numerator/denominator
            finite = math.isfinite(actual)
            absolute = abs(actual-expected) if finite else math.inf
            relative = absolute/max(expected, 1e-300)
            max_abs, max_rel = max(max_abs, absolute), max(max_rel, relative)
            if not finite or not (absolute <= atol or relative <= rtol):
                bad += 1
                if len(mismatches) < 8:
                    mismatches.append({'row': row, 'column': column,
                                       'actual': actual if finite else None, 'expected': expected})
    return {'passed': bad == 0, 'shape': [rows, cols], 'elements': rows*cols,
            'input_dtype': 'fp32', 'output_dtype': 'fp16',
            'golden': 'FP64 stable softmax per row, no scale',
            'mismatches': bad, 'max_abs_error': max_abs if math.isfinite(max_abs) else None,
            'max_relative_error': max_rel if math.isfinite(max_rel) else None,
            'row_sums': row_sums, 'first_mismatches': mismatches,
            'atol': atol, 'rtol': rtol, 'tolerance': 'abs <= atol OR rel <= rtol'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=8)
    parser.add_argument('--cols', type=int, default=1024)
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = check(args.input, args.output, args.rows, args.cols, args.atol, args.rtol)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
