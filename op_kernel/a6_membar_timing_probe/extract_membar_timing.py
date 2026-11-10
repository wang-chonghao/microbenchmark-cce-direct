"""Extract staged Membar observations, without guessing missing hardware events."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
RAW = re.compile(r'\[(\d+)\].*?\(PC:\s*(0x[0-9a-f]+)\)\s+(\w+)\s*:.*?'
                 r'\(ID:\s*(\d+)\)\s+(\w+)(.*)', re.I)
BARRIER = 'RV_SMEM_BAR'
LOADS = {'RV_VLD', 'RV_VLDI', 'RV_VLDS'}
STORES = {'RV_VST', 'RV_VSTI', 'RV_VSTS'}
CONTROL = r'\[(?P<unit>SHQ|SU|RVECSU|REVCSU)\]'
IDENTITY = r'\binstr_id\s*[:=]?\s*(?P<id>\d+)'
PATTERNS = {
    'issue': CONTROL + r'.*\b(?:ISU_ISSUE|SU_ISSUE|ISSUE)\b.*RV_SMEM_BAR.*' + IDENTITY,
    'retire': CONTROL + r'.*\b(?:ISU_BAR_RETIRE|SU_BAR_RETIRE|BAR_RETIRE)\b.*RV_SMEM_BAR.*' + IDENTITY,
    'receive': r'\[(?P<unit>SHQ|LDQ|SU|RVECSU|REVCSU)\].*\b(?:ISU_RECV|SU_RECV)\b.*' + IDENTITY,
    'release_vld_vst': r'<<consumer dec set_cnt>>\s+bar_type\s+VLD_VST,\s*isa_id\s+(?P<id>\d+),\s*unit_type\s+STU\b',
    'release_vst_vld': r'<<consumer dec set_cnt>>\s+bar_type\s+VST_VLD,\s*isa_id\s+(?P<id>\d+),\s*unit_type\s+LDU\b',
    'store_start': r'\[SHQ\].*\bISU_ISSUE\b.*instr_name\s+RV_VST(?:I|S)?\b.*' + IDENTITY,
    'load_start': r'\[SEND_UB_RD(?:\.PORT_\d+)?\].*instr_name\s+RV_VLD(?:I|S)?\b.*' + IDENTITY,
    'lsu_go': r'\[GO\.SMEM_BAR\].*instr_id\s*:\s*(?P<id>\d+)',
}


def lines(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', errors='replace') as stream:
        yield from enumerate(stream, 1)


def raw_events(path):
    result = []
    for number, line in lines(path):
        match = RAW.search(line)
        if match:
            cy, pc, unit, ident, op, detail = match.groups()
            result.append(dict(cycle=int(cy), pc=int(pc, 16), unit=unit.upper(), id=int(ident),
                               op=op.upper(), detail=detail.strip(), file=path.name, line=number))
    return result


def locate(root, name):
    choices = [root / name, root / (name + '.gz')]
    found = [p for p in choices if p.is_file()]
    if len(found) != 1:
        raise ValueError(f'Provide exactly one {name}[.gz] directly in {root}')
    return found[0]


def instruction_stream(root, vf_index=None):
    popped = raw_events(locate(root, 'core0.veccore0.instr_popped_log.dump'))
    done = raw_events(locate(root, 'core0.veccore0.instr_log.dump'))
    vfs = [e for e in popped if e['op'] == 'VF']
    if not vfs or vf_index is None and len(vfs) != 1:
        raise ValueError('Need one complete VF, or select one with --vf-index')
    index = 0 if vf_index is None else vf_index
    if not 0 <= index < len(vfs):
        raise ValueError('VF index out of range')
    vf = vfs[index]
    ends = [e for e in done if (e['id'], e['pc'], e['op']) == (vf['id'], vf['pc'], 'VF')]
    if len(ends) != 1:
        raise ValueError('Missing or ambiguous VF completion')
    geometry = re.search(r'addr:\s*(0x[0-9a-f]+),\s*instr_num:\s*(0x[0-9a-f]+|\d+)', vf['detail'], re.I)
    if not geometry:
        raise ValueError('VF code address/word count missing')
    first, words = int(geometry[1], 16), int(geometry[2], 0)
    scope = dict(index=index, start=vf['cycle'], end=ends[0]['cycle'], pc_start=first, pc_end=first+4*words)
    instructions = {}
    for label, events in (('popped', popped), ('completed', done)):
        for event in events:
            if not (scope['start'] <= event['cycle'] <= scope['end'] and
                    first <= event['pc'] < scope['pc_end'] and event['op'].startswith('RV_')):
                continue
            old = instructions.setdefault(event['id'], dict(id=event['id'], pc=event['pc'], op=event['op'],
                                                           detail=event['detail'], popped=[], completed=[]))
            if (old['pc'], old['op'], old['detail']) != (event['pc'], event['op'], event['detail']):
                # Addresses printed in details can differ between popped/done;
                # opcode/PC and direction, rather than scalar snapshots, identify an instruction.
                if (old['pc'], old['op']) != (event['pc'], event['op']):
                    raise ValueError('Instruction ID reused inside the selected VF')
            signature = (event['cycle'], event['unit'])
            if signature not in {(e['cycle'], e['unit']) for e in old[label]}:
                old[label].append(event)
    ordered = sorted(instructions.values(), key=lambda e: e['pc'])
    if len({e['pc'] for e in ordered}) != len(ordered):
        raise ValueError('Repeated instruction PC: probe was looped or VF scope is wrong')
    if not ordered or any(not e['completed'] for e in ordered):
        raise ValueError('Incomplete instruction stream/completion log')
    if any(e['op'].startswith('RV_VLOOP') for e in ordered):
        raise ValueError('Unexpected dynamic loop in a straight-line probe')
    return scope, ordered


def direction(inst):
    match = re.search(r'\btype:\s*(VLD_VST|VST_VLD)\b', inst['detail'])
    if not match:
        raise ValueError(f'Unknown barrier direction: {inst["detail"]}')
    return match[1]


def unit_name(unit):
    return 'SU' if unit in {'SU', 'RVECSU', 'REVCSU'} else unit


def detailed_events(root, scope, ordered, custom=None):
    patterns = dict(PATTERNS)
    for name, expression in (custom or {}).items():
        if name not in patterns:
            raise ValueError(f'Unknown event pattern: {name}')
        patterns[name] = expression
    compiled = {name: re.compile(expr) for name, expr in patterns.items()}
    if any('id' not in p.groupindex for p in compiled.values()):
        raise ValueError('Every event pattern must capture (?P<id>...)')
    if any('unit' not in compiled[name].groupindex for name in ('issue', 'retire', 'receive')):
        raise ValueError('Control event patterns must also capture (?P<unit>...)')
    known = {i['id']: i for i in ordered}
    events = defaultdict(lambda: defaultdict(list))
    paths = sorted(p for p in root.iterdir() if p.is_file() and p.name.startswith('core0.veccore0.') and
                   p.name.endswith(('.dump', '.dump.gz', '.log')) and
                   any(tag in p.name.upper() for tag in ('ISU', 'LSU', 'GSU', 'SU.', 'DEPMGR')))
    for path in paths:
        for number, line in lines(path):
            for name, regex in compiled.items():
                match = regex.search(line)
                if not match or int(match['id']) not in known:
                    continue
                captures = match.groupdict()
                time = captures.get('cycle') or next(iter(re.findall(r'\[(\d+)\]', line)), None)
                if time is None:
                    continue
                cy, ident = int(time), int(match['id'])
                if not scope['start'] <= cy <= scope['end']:
                    continue
                pc = captures.get('pc')
                if pc is None:
                    found = re.search(r'\bPC:?\s+(0x[0-9a-f]+|[0-9a-f]+)\b', line, re.I)
                    pc = found[1] if found else None
                if pc is not None and int(pc, 16) != known[ident]['pc']:
                    continue
                event = dict(cycle=cy, unit=unit_name(captures.get('unit', '')), file=path.name, line=number)
                events[ident][name].append(event)
    return events, [p.name for p in paths], patterns


def unique(events, stage, control=None):
    selected = [e for e in events.get(stage, []) if control is None or e['unit'] == control]
    values = {e['cycle'] for e in selected}
    return next(iter(values)) if len(values) == 1 else None


def distribution(values):
    values = [v for v in values if v is not None]
    return dict(count=len(values), minimum=min(values) if values else None,
                median=statistics.median(values) if values else None, maximum=max(values) if values else None,
                histogram=dict(sorted(Counter(values).items())))


def stable(metric, *, nonnegative=True):
    if metric['count'] >= 3 and metric['minimum'] == metric['maximum']:
        if not nonnegative or metric['minimum'] >= 0:
            return metric['minimum']
    return None


def analyze(root, spec, *, vf_index=None, control_unit=None, custom=None, trim=1):
    scope, ordered = instruction_stream(root, vf_index)
    barriers = [i for i in ordered if i['op'] == BARRIER]
    actual = dict(Counter(direction(i) for i in barriers))
    counts = dict(loads=sum(i['op'] in LOADS for i in ordered),
                  stores=sum(i['op'] in STORES for i in ordered), membars=actual)
    expected = {k: spec[k] for k in ('loads', 'stores', 'membars')}
    if counts != expected:
        raise ValueError(f'Unexpected compiled stream (removed/inserted instructions?): {counts} != {expected}')
    allowed = LOADS | STORES | {BARRIER, 'RV_PSET', 'RV_SEND'}
    extra = [i['op'] for i in ordered if i['op'] not in allowed and not i['op'].startswith('RV_S')]
    spill = [i['op'] for i in ordered if i['op'].startswith(('RV_SST', 'RV_SLD', 'RV_PST', 'RV_PLD'))]
    if extra or spill:
        raise ValueError(f'Unexpected compute/predicate/scalar spill: {extra + spill}')
    targets = [b for b in barriers if direction(b) == spec['direction']]
    if len(targets) != 8 or not 0 <= trim < 4:
        raise ValueError('Need eight measured barriers and trim in [0,3]')
    if spec['layout'] == 'cluster' and any(
        targets[0]['pc'] < i['pc'] < targets[-1]['pc'] and i['op'] in LOADS | STORES for i in ordered
    ):
        raise ValueError('Compiled cluster contains intervening memory operations')
    evidence, detail_files, patterns = detailed_events(root, scope, ordered, custom)
    units = {e['unit'] for b in targets for e in evidence[b['id']].get('issue', [])}
    control = unit_name(control_unit) if control_unit else next(iter(units)) if len(units) == 1 else None
    notes = []
    if control is None:
        notes.append('Control issue phase missing/ambiguous; provide SU/ISU logs or --control-unit / --patterns. Popped is NOT substituted.')
    release_stage = 'release_' + spec['direction'].lower()
    rows = []
    for index, b in enumerate(targets):
        ev = evidence[b['id']]
        issue = unique(ev, 'issue', control) if control else None
        retire = unique(ev, 'retire', control) if control else None
        release = unique(ev, release_stage)
        following = targets[index+1] if index+1 < len(targets) else None
        next_issue = unique(evidence[following['id']], 'issue', control) if following and control else None
        consumer_class = STORES if spec['direction'] == 'VLD_VST' else LOADS
        segment_end = next((x['pc'] for x in barriers if x['pc'] > b['pc']), scope['pc_end'])
        consumers = [i for i in ordered if b['pc'] < i['pc'] < segment_end and i['op'] in consumer_class]
        if spec['layout'] == 'interleaved' and len(consumers) != 1:
            raise ValueError('Compiled interleaving changed: expected exactly one consumer per measured barrier')
        consumer = consumers[0] if len(consumers) == 1 and spec['layout'] == 'interleaved' else None
        start = unique(evidence[consumer['id']], 'store_start' if consumer_class is STORES else 'load_start') if consumer else None
        producer_class = LOADS if spec['direction'] == 'VLD_VST' else STORES
        prior = [i for i in ordered if i['pc'] < b['pc'] and i['op'] in producer_class]
        producer_ready = issue is not None and all(max(e['cycle'] for e in i['completed']) <= issue for i in prior)
        delta = lambda a, c: a-c if a is not None and c is not None else None
        rows.append(dict(ordinal=index, id=b['id'], pc=hex(b['pc']), direction=spec['direction'],
            steady=trim <= index < len(targets)-trim, issue=issue, release=release, retire=retire,
            receive=unique(ev, 'receive', control) if control else None,
            consumer_id=consumer['id'] if consumer else None, consumer_start=start,
            next_issue=next_issue, producer_complete_before_issue=producer_ready,
            release_latency_observed=delta(release, issue), retire_latency_observed=delta(retire, issue),
            retire_tail_observed=delta(retire, release), consumer_delay_observed=delta(start, release),
            next_issue_delay_observed=delta(next_issue, retire), feedback_gap_upper_bound=delta(next_issue, start),
            overlapped_next=next_issue < retire if next_issue is not None and retire is not None else None))
    fields = ('release_latency_observed', 'retire_latency_observed', 'retire_tail_observed',
              'consumer_delay_observed', 'next_issue_delay_observed', 'feedback_gap_upper_bound')
    metrics = {name: distribution(r[name] for r in rows if r['steady'] and
               (r['producer_complete_before_issue'] or name not in ('release_latency_observed', 'retire_latency_observed')))
               for name in fields}
    overlap = any(r['overlapped_next'] is True for r in rows if r['steady'])
    order_errors = [r['ordinal'] for r in rows if r['steady'] and any(
        r[field] is not None and r[field] < 0 for field in
        ('release_latency_observed', 'retire_latency_observed', 'retire_tail_observed'))]
    observed_pairs = sum(r['overlapped_next'] is not None for r in rows if r['steady'])
    compatible = False if overlap or order_errors else True if observed_pairs >= 3 else None
    candidates = {}
    if spec['layout'] == 'cluster':
        for name in ('release_latency', 'retire_latency', 'next_issue_delay'):
            candidates[name] = stable(metrics[name + '_observed'])
        if overlap:
            candidates['next_issue_delay'] = None
            notes.append('Next barrier issues before previous retirement: current serial controller cannot be calibrated by a positive delay alone.')
    else:
        candidates['consumer_delay'] = stable(metrics['consumer_delay_observed'])
        notes.append('Consumer delay is a controlled-probe candidate, not proof of an intrinsic hardware constant; check readiness/ports/queue arrival.')
    notes += ['feedback_gap_upper_bound includes other issue constraints; it is NOT automatically lsu_start_to_next_issue.',
              'admission_delay is not independently identifiable here; no value is fabricated.',
              'Missing/ambiguous events remain null; raw RVECLD/RVECST/SU stages are preserved separately.']
    if order_errors:
        candidates = {name: None for name in candidates}
        notes.append('Selected stages violate issue <= release <= retire; do not fit the current controller to these events.')
    used_files = [locate(root, 'core0.veccore0.instr_popped_log.dump'),
                  locate(root, 'core0.veccore0.instr_log.dump'), *[root/name for name in detail_files]]
    return dict(schema_version=1, logs_directory=str(root.absolute()),
                log_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in used_files},
                vf=scope, instruction_counts=counts, control_unit=control,
                control_units_observed=sorted(units),
                direction=spec['direction'], layout=spec['layout'], rows=rows, metrics=metrics,
                parameter_candidates=candidates, next_barrier_overlaps_retirement=overlap,
                serial_controller_compatible_in_observed_samples=compatible, event_order_errors=order_errors,
                admission_delay=None, calibrated=False, notes=notes, event_patterns=patterns,
                detailed_files=detail_files, raw_instructions=ordered,
                evidence={str(k): dict(v) for k, v in evidence.items()})


def write_report(folder, result):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (folder / 'barriers.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['rows'][0]))
        writer.writeheader()
        writer.writerows(result['rows'])
    text = ['# Membar 日志提取', '', f"方向：`{result['direction']}`；控制阶段：`{result['control_unit']}`。",
            '', '这里只提供观测与候选，不自动修改 uarch。', '',
            '| 时间差 | 样本 | 最小 | 中位数 | 最大 |', '|---|---:|---:|---:|---:|']
    for name, stat in result['metrics'].items():
        text.append(f"| {name} | {stat['count']} | {stat['minimum']} | {stat['median']} | {stat['maximum']} |")
    text += ['', '候选：`' + json.dumps(result['parameter_candidates']) + '`。', '', *['- '+s for s in result['notes']]]
    (folder / 'README.md').write_text('\n'.join(text)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', type=Path, required=True, help='Case log directory, or parent containing four case directories')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', choices=json.loads((HERE/'experiments.json').read_text())['cases'])
    parser.add_argument('--vf-index', type=int)
    parser.add_argument('--control-unit', choices=('SU', 'SHQ', 'RVECSU', 'REVCSU'))
    parser.add_argument('--patterns', type=Path, help='JSON overrides for named event regexes; each must capture id')
    parser.add_argument('--trim', type=int, default=1)
    args = parser.parse_args()
    if args.out.absolute().is_relative_to(args.logs.absolute()):
        parser.error('--out must be outside the raw log directory')
    manifest = json.loads((HERE/'experiments.json').read_text())
    custom = json.loads(args.patterns.read_text()) if args.patterns else None
    names = [args.case] if args.case else list(manifest['cases'])
    reports = {}
    for name in names:
        logs = args.logs if args.case else args.logs/name
        result = analyze(logs, manifest['cases'][name], vf_index=args.vf_index,
                         control_unit=args.control_unit, custom=custom, trim=args.trim)
        result['case'] = name
        expected_sha = hashlib.sha256((HERE/(name+'.cce')).read_bytes()).hexdigest()
        result['expected_source_sha256'] = expected_sha
        result['source_contract_verified'] = False
        receipt = logs/'build.json'
        if receipt.exists():
            build = json.loads(receipt.read_text())
            source = logs/'source'/(name+'.cce')
            header = logs/'source/probe_common.h'
            if (build['target'] != manifest['target'] or not build.get('compiled') or build['vf_stack_budget'] != 0 or
                build['source_sha256'] != expected_sha or
                hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha or
                hashlib.sha256(header.read_bytes()).hexdigest() != build['header_sha256'] or
                build['header_sha256'] != hashlib.sha256((HERE/'probe_common.h').read_bytes()).hexdigest()):
                raise ValueError('Build snapshot does not match this probe contract')
            binary = logs/'kernel.o'
            if binary.exists() and hashlib.sha256(binary.read_bytes()).hexdigest() != build['binary_sha256']:
                raise ValueError('Kernel binary differs from build receipt')
            result['source_contract_verified'] = True
        write_report(args.out/name, result)
        reports[name] = dict(parameter_candidates=result['parameter_candidates'], metrics=result['metrics'],
                             overlap=result['next_barrier_overlaps_retirement'], control_unit=result['control_unit'],
                             serial_controller_compatible=result['serial_controller_compatible_in_observed_samples'])
        print(name, json.dumps(reports[name]['parameter_candidates']), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    directions = {}
    for direction in ('VLD_VST', 'VST_VLD'):
        cluster = '01_cluster_vld_vst' if direction == 'VLD_VST' else '02_cluster_vst_vld'
        interleave = '03_interleave_store' if direction == 'VLD_VST' else '04_interleave_load'
        fields = {name: reports.get(cluster, {}).get('parameter_candidates', {}).get(name)
                  for name in ('release_latency', 'retire_latency', 'next_issue_delay')}
        fields['consumer_delay'] = reports.get(interleave, {}).get('parameter_candidates', {}).get('consumer_delay')
        fields['feedback_gap_upper_bound'] = reports.get(interleave, {}).get('metrics', {}).get('feedback_gap_upper_bound')
        fields['serial_controller_compatible_in_observed_samples'] = reports.get(cluster, {}).get('serial_controller_compatible')
        directions[direction] = fields
    (args.out/'overview.json').write_text(json.dumps(dict(cases=reports, directions=directions, calibrated=False,
        admission_delay=None, feedback_requires_bound_isolation=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
