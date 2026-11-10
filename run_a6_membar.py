#!/usr/bin/env python3
"""Compile, simulate, check, and extract the four controlled A6 membar probes."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import time

from run_a6_bench import run_command

ROOT = Path(__file__).resolve().parent
PROBES = ROOT/'op_kernel/a6_membar_timing_probe'
FLAGS = '--cce-simd-vf-fusion=false -mllvm -cce-aicore-vec-misched=0 -mllvm -cce-vf-stack-size=0'
DUMP = re.compile(r'.*\.dump(?:\.\d+)?(?:\.gz)?$')
CORE_LOG = re.compile(r'^(core0\.veccore0\..*?)(?:\.\d+)?\.(dump|log)(?:\.\d+)?(?:\.gz)?$')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extractor():
    spec = importlib.util.spec_from_file_location('a6_membar_extract', PROBES/'extract_membar_timing.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_golden(run):
    actual_input = (run/'run/input_input.bin').read_bytes()
    expected_input = (run/'inputs/input_input.bin').read_bytes()
    actual = (run/'run/output_output.bin').read_bytes()
    expected = expected_input * 8
    guard = (run/'run/guard_check.txt').read_text().startswith('pass:')
    return dict(passed=len(expected_input) == 256 and actual_input == expected_input and actual == expected and guard,
                input_unchanged=actual_input == expected_input, guard_passed=guard,
                output_bytes=len(actual), expected_bytes=len(expected),
                output_sha256=hashlib.sha256(actual).hexdigest(),
                expected_sha256=hashlib.sha256(expected).hexdigest(),
                kind='Exact byte equality: eight copies of 64 fp32 values; native GM guards on both sides')


def collect_logs(run, dest, keep_all=False):
    """Preserve core0 raw logs and normalize rotations for the upstream parser."""
    sources, discarded = [], []
    for base, dirs, files in os.walk(run/'run', followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(base)/d).is_symlink())
        for name in sorted(files):
            path = Path(base)/name
            if path.is_symlink():
                continue
            if name.startswith('core0.veccore0.') and (DUMP.fullmatch(name) or '.log' in name):
                sources.append(path)
            elif DUMP.fullmatch(name):
                discarded.append(path)
    if len({p.name for p in sources}) != len(sources):
        raise ValueError('Duplicate core0 log basenames in different directories; refusing to merge executions')
    raw = dest/'raw'
    raw.mkdir(parents=True, exist_ok=False)
    groups, receipts = defaultdict(list), []
    for path in sources:
        target = raw/path.name
        receipt = dict(original=str(path.relative_to(run)), sha256=sha(path), bytes=path.stat().st_size)
        path.rename(target)
        receipt['raw'] = str(target.relative_to(dest))
        receipts.append(receipt)
        match = CORE_LOG.fullmatch(target.name)
        if match:
            groups[match[1]+'.'+match[2]].append(target)
    maps = {}
    for name, paths in groups.items():
        line_count, ranges = 0, []
        with (dest/name).open('wb') as output:
            for path in sorted(paths):
                start = line_count+1
                opener = gzip.open if path.suffix == '.gz' else open
                with opener(path, 'rb') as stream:
                    for line in stream:
                        output.write(line if line.endswith(b'\n') else line+b'\n')
                        line_count += 1
                ranges.append(dict(raw=str(path.relative_to(dest)), first_line=start, last_line=line_count))
        maps[name] = dict(sha256=sha(dest/name), ranges=ranges)
    if not keep_all:
        for path in discarded:
            path.unlink()
    result = dict(raw_files=receipts, normalized=maps,
                  removed_other_dumps=0 if keep_all else len(discarded))
    (dest/'archive.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def verify_receipt(run, name, spec):
    receipt = json.loads((run/'receipt.json').read_text())
    expected = {
        'source_sha256': sha(PROBES/(name+'.cce')),
        'header_sha256': sha(PROBES/'probe_common.h'),
        'generated_sha256': sha(run/'kernel.cce'),
        'binary_sha256': sha(run/'build'/(spec['kernel']+'_mix.o')),
    }
    if any(receipt.get(k) != v for k, v in expected.items()) or sha(run/'probe_common.h') != expected['header_sha256']:
        raise ValueError('Source/header/binary differs from recorded probe contract')
    if receipt['flags'] != FLAGS or receipt['core_arch'] != 'dav-920r1-vec':
        raise ValueError('Unexpected compile contract')
    return receipt


def analyze_case(module, output, measurements, name, spec, args):
    run = output/'_work'/name
    receipt = verify_receipt(run, name, spec)
    golden = check_golden(run)
    (run/'golden.json').write_text(json.dumps(golden, indent=2)+'\n')
    if not golden['passed']:
        raise ValueError('Golden/GM guard check failed; no timing candidates accepted')
    custom = json.loads(args.patterns.read_text()) if args.patterns else None
    result = module.analyze(output/'logs'/name, spec, vf_index=args.vf_index,
                            control_unit=args.control_unit, custom=custom, trim=args.trim)
    result.update(case=name, golden=golden, build_receipt=receipt, source_contract_verified=True)
    module.write_report(measurements/name, result)
    required = ['issue', 'release', 'retire'] + (['consumer_start'] if spec['layout'] == 'interleaved' else [])
    complete = all(r[key] is not None for r in result['rows'] if r['steady'] for key in required)
    return dict(status='observed' if complete else 'timing_incomplete', calibrated=False,
                parameter_candidates=result['parameter_candidates'], metrics=result['metrics'],
                control_unit=result['control_unit'], golden=golden,
                serial_controller_compatible=result['serial_controller_compatible_in_observed_samples'])


def main():
    manifest = json.loads((PROBES/'experiments.json').read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--cases', nargs='+', choices=list(manifest['cases']))
    p.add_argument('--list', action='store_true')
    p.add_argument('--compile-only', action='store_true')
    p.add_argument('--analyze-only', action='store_true')
    p.add_argument('--analysis-output', type=Path)
    p.add_argument('--cann-home', default=os.environ.get('CANN_HOME'))
    p.add_argument('--full-simulator-home', default=os.environ.get('FULL_SIMULATOR_HOME', ''))
    p.add_argument('--timeout', type=float, default=0)
    p.add_argument('--keep-all-dumps', action='store_true')
    p.add_argument('--patterns', type=Path)
    p.add_argument('--control-unit', choices=('SU', 'SHQ', 'RVECSU', 'REVCSU'))
    p.add_argument('--vf-index', type=int)
    p.add_argument('--trim', type=int, default=1)
    args = p.parse_args()
    names = args.cases or list(manifest['cases'])
    if args.list:
        for name in names:
            print(name, manifest['cases'][name])
        return 0
    if not math.isfinite(args.timeout) or args.timeout < 0 or not 0 <= args.trim < 4:
        p.error('timeout must be finite and >=0; trim must be in [0,3]')
    if args.analyze_only and (not args.output or args.compile_only):
        p.error('--analyze-only requires --output and cannot use --compile-only')
    if not args.analyze_only and not args.cann_home:
        p.error('Set CANN_HOME or pass --cann-home')
    output = (args.output or ROOT/'results'/('a6_membar_'+time.strftime('%Y%m%d_%H%M%S'))).resolve()
    if args.analyze_only:
        if not output.is_dir():
            p.error('Existing result directory required')
    else:
        output.mkdir(parents=True, exist_ok=False)
    measurements = (args.analysis_output or output/('measurement_'+time.strftime('%Y%m%d_%H%M%S')
                    if args.analyze_only else 'measurement')).resolve()
    measurements.mkdir(parents=True, exist_ok=False)
    module = extractor()
    env = dict(os.environ, FULL_SIMULATOR_HOME=args.full_simulator_home,
               SOC_VERSION='Ascend910_9691', NPU_TYPE='Ascend910_9691', CORE_ARCH='dav-920r1-vec',
               CORE_SIM_DIR='dav_9201', ACL_DEVICE_ID='0', MB_GUARD_ENABLE='1', CCEC_EXTRA_FLAGS=FLAGS)
    if os.environ.get('CCEC_EXTRA_FLAGS'):
        print('[INFO] This controlled probe replaces CCEC_EXTRA_FLAGS with fixed no-fusion/no-misched/zero-VF-stack flags')
    results = {}
    print(f'OUTPUT={output}\nMEASUREMENT={measurements}', flush=True)
    for name in names:
        spec = manifest['cases'][name]
        record = dict(status='failed', calibrated=False)
        try:
            if not args.analyze_only:
                run = output/'_work'/name
                run.mkdir(parents=True)
                source = run/'kernel.cce'
                original = (PROBES/(name+'.cce')).read_bytes()
                header = (f'// MB_KERNEL {spec["kernel"]}\n// MB_TENSOR input input fp32 64\n'
                          '// MB_TENSOR output output fp32 512\n').encode()
                source.write_bytes(header+original)
                shutil.copyfile(PROBES/'probe_common.h', run/'probe_common.h')
                inputs = run/'inputs'
                inputs.mkdir()
                (inputs/'input_input.bin').write_bytes(struct.pack('<64f', *[(i-31)*.125 for i in range(64)]))
                case_env = dict(env, MB_INPUT_DIR=str(inputs))
                if args.compile_only:
                    cmd = [sys.executable, '-m', 'benchmarks.a6.compile_probe', '--kernel', str(source),
                           '--output', str(run), '--cann-home', args.cann_home]
                else:
                    cmd = [sys.executable, str(ROOT/'run_test.py'), '--kernel', str(source), '--output', str(run),
                           '--cann-home', args.cann_home, '--soc-version', 'Ascend910_9691', '--core-arch',
                           'dav-920r1-vec', '--core-sim-dir', 'dav_9201', '--device-id', '0', '--block-dim', '1',
                           '--local-memory-size', '0', '--golden', 'none']
                record['command'] = cmd
                code = run_command(cmd, run/'console.log', args.timeout, case_env)
                record['returncode'] = code
                if not args.compile_only:
                    record['archive'] = collect_logs(run, output/'logs'/name, args.keep_all_dumps)
                if code:
                    raise ValueError(f'Compiler/model exit={code}; see {run}/console.log')
                receipt = dict(source_sha256=hashlib.sha256(original).hexdigest(), header_sha256=sha(run/'probe_common.h'),
                    generated_sha256=sha(source), binary_sha256=sha(run/'build'/(spec['kernel']+'_mix.o')),
                    core_arch='dav-920r1-vec', soc='Ascend910_9691', flags=FLAGS,
                    cann_home=args.cann_home, full_simulator_home=args.full_simulator_home)
                (run/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
                if args.compile_only:
                    record['status'] = 'compiled_only'
                else:
                    record.update(analyze_case(module, output, measurements, name, spec, args))
            else:
                record.update(analyze_case(module, output, measurements, name, spec, args))
        except (OSError, ValueError, KeyError) as exc:
            record['error'] = str(exc)
        results[name] = record
        directions = {}
        for direction, cluster, interleave in [('VLD_VST','01_cluster_vld_vst','03_interleave_store'),
                                              ('VST_VLD','02_cluster_vst_vld','04_interleave_load')]:
            directions[direction] = {k:results.get(cluster, {}).get('parameter_candidates', {}).get(k)
                                     for k in ('release_latency','retire_latency','next_issue_delay')}
            directions[direction]['consumer_delay'] = results.get(interleave, {}).get('parameter_candidates', {}).get('consumer_delay')
        (measurements/'overview.json').write_text(json.dumps(dict(cases=results, directions=directions,
            calibrated=False, admission_delay=None, feedback_requires_bound_isolation=True), indent=2)+'\n')
        print(f'{name}: {record["status"]}' + (': '+record['error'] if 'error' in record else ''), flush=True)
    return int(any(r['status'] not in ('observed','compiled_only') for r in results.values()))


if __name__ == '__main__':
    raise SystemExit(main())
