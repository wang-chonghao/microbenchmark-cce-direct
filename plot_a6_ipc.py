#!/usr/bin/env python3
"""Batch IPC plots from core0.veccore0 dumps, using only the standard library."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re

from benchmarks.a6.analyze import parse_events, parse_exu

LOG = re.compile(r'^(?P<prefix>.*core0\.veccore0)\.(?P<kind>instr_log|instr_popped_log|rvec\.EXU)'
                 r'(?:\.\d+)?\.dump(?:\.\d+)?$')
LABELS = {'done': 'instruction completion', 'popped': 'instruction popped (not EXU issue)',
          'exu-issue': 'EXU issue'}


def curve_color(label, index):
    """Bind softmax colors to variants, independently of discovery order."""
    if label.endswith('_three_loop'):
        return '#2166ac'
    if label.endswith(('_no_membar', '_giant_loop', '_gap2')):
        return '#2ca02c'
    return ['#087f8c', '#d1495b', '#7b4ab5', '#438a36'][index % 4]


def discover(root):
    groups = defaultdict(lambda: defaultdict(list))
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(base)/d).is_symlink())
        for name in sorted(files):
            match = LOG.fullmatch(name)
            path = Path(base)/name
            if match and not path.is_symlink():
                groups[(Path(base), match['prefix'])][match['kind']].append(path)
    return sorted(groups.items(), key=lambda item: (str(item[0][0]), item[0][1]))


def unique(events, exu=False):
    """Deduplicate rotation overlap; reject ID resets/multiple execution segments."""
    result, seen, duplicates = [], {}, 0
    for e in events:
        key = (e['id'], e['op'])
        value = (e['cycle'], e.get('exu') if exu else e['pc'])
        if key in seen:
            if seen[key] != value:
                raise ValueError(f'Ambiguous repeated dynamic ID/op: {key}; split launches before plotting')
            duplicates += 1
        else:
            seen[key] = value
            result.append(e)
    return result, duplicates


def measure(logs, metric, scope, window, max_cycles):
    routed = []
    if scope != 'all-rv' or metric == 'exu-issue':
        routed = [e for p in logs.get('rvec.EXU', []) for e in parse_exu(p)]
        if scope == 'compute':
            routed = [e for e in routed if e['op'].startswith('RV_V')]
        routed, _ = unique(routed, True)
        if not routed:
            raise ValueError('Missing/empty/unsupported EXU log; use --scope all-rv only for a different metric')
    route = {(e['id'], e['op']): e for e in routed}
    if metric == 'exu-issue':
        events, duplicates = routed, 0
    else:
        kind = 'instr_log' if metric == 'done' else 'instr_popped_log'
        events = [e for p in logs.get(kind, []) for e in parse_events(p)]
        if scope != 'all-rv':
            events = [e for e in events if (e['id'], e['op']) in route]
        events, duplicates = unique(events)
    if not events:
        raise ValueError('No matching instruction events; missing log or unsupported format')
    missing = len(set(route) - {(e['id'], e['op']) for e in events}) if scope != 'all-rv' else None
    counts = Counter(e['cycle'] for e in events)
    first, last = min(counts), max(counts)
    span = last - first + 1
    if span > max_cycles:
        raise ValueError(f'Cycle span {span} exceeds --max-cycles {max_cycles}')
    # Visualization only: drain a complete trailing window, without changing
    # measured duration/average. Incomplete logs must not imply execution ended.
    tail = window if not missing else 0
    if span + tail > max_cycles:
        raise ValueError('Event span plus window tail exceeds --max-cycles')
    per_cycle = [counts.get(c, 0) for c in range(first, last + 1)] + [0] * tail
    ipc, total = [], 0
    for i, n in enumerate(per_cycle):
        total += n
        if i >= window:
            total -= per_cycle[i-window]
        ipc.append(total / window)
    return dict(first_cycle=first, last_cycle=last, span_cycles=span, instructions=len(events),
                plot_last_cycle=last+tail, tail_padding_cycles=tail,
                average_ipc=len(events)/span, peak_cycle_ipc=max(per_cycle), peak_window_ipc=max(ipc),
                window=window, metric=metric, scope=scope, duplicates_removed=duplicates,
                missing_exu_matched_events=missing, status='partial' if missing else 'observed',
                histogram=dict(sorted(Counter(e['op'] for e in events).items())),
                observed_exus=sorted({e['exu'] for e in routed}), counts=per_cycle, ipc=ipc)


def plot_svg(path, title, curves, absolute=False):
    width, height, left, top, pw, ph = 1200, 440, 85, 100, 1080, 260
    xmax = max((c['plot_last_cycle'] if absolute else len(c['ipc'])-1) for _, c in curves)
    xmin = min(c['first_cycle'] for _, c in curves) if absolute else 0
    ymax = max(1.0, max(c['peak_window_ipc'] for _, c in curves) * 1.1)
    esc = html.escape
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="1200" height="440" fill="white"/>',
             '<g font-family="sans-serif" font-size="13" fill="#24292f">',
             '<text transform="translate(18,255) rotate(-90)" text-anchor="middle">instructions / cycle</text>',
             f'<text x="{left}" y="28" font-size="17">{esc(title[:120])}</text>']
    for j, (label, c) in enumerate(curves):
        color = curve_color(label, j)
        lines.append(f'<text x="{left}" y="{50+j*19}" fill="{color}">{esc(label[:100])} | avg={c["average_ipc"]:.4f} | N={c["instructions"]} | {c["status"]}</text>')
    for i in range(6):
        y, value = top+ph-i*ph/5, ymax*i/5
        lines.extend([f'<path d="M{left},{y} H{left+pw}" stroke="#e2e5e9"/>',
                      f'<text x="{left-10}" y="{y+4}" text-anchor="end">{value:.2f}</text>'])
        x, cycle = left+i*pw/5, xmin+(xmax-xmin)*i/5
        lines.append(f'<text x="{x}" y="{top+ph+24}" text-anchor="middle">{cycle:.0f}</text>')
    for j, (label, c) in enumerate(curves):
        # Keep the CSV at full resolution; SVG uses min/max envelopes when large.
        step = max(1, math.ceil(len(c['ipc'])/3000))
        points = []
        for start in range(0, len(c['ipc']), step):
            block = c['ipc'][start:start+step]
            indices = sorted({start+block.index(min(block)), start+block.index(max(block))})
            for i in indices:
                cycle = c['first_cycle']+i if absolute else i
                x = left+(cycle-xmin)*pw/max(1, xmax-xmin)
                y = top+ph-c['ipc'][i]*ph/ymax
                points.append(f'{x:.2f},{y:.2f}')
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{curve_color(label, j)}" stroke-width="1.4"/>')
    c = curves[0][1]
    axis = 'absolute model cycle' if absolute else 'cycles since each curve first selected event'
    lines.extend([f'<text x="{left}" y="405">{esc(axis)}; window={c["window"]}; zero tail for display only (not measured kernel end)</text>',
                  f'<text x="{left}" y="426">{esc(LABELS[c["metric"]])}; scope={esc(c["scope"])}; full-resolution CSV, SVG min/max envelope if &gt;3000 cycles</text>',
                  '</g></svg>'])
    path.write_text('\n'.join(lines))


def plot_png(path, title, curves, absolute=False):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.8), layout='constrained')
    for j, (label, c) in enumerate(curves):
        start = c['first_cycle'] if absolute else 0
        ax.plot(range(start, start+len(c['ipc'])), c['ipc'], color=curve_color(label, j),
                linewidth=1.1, label=f'{label}\navg={c["average_ipc"]:.4f}, N={c["instructions"]}, {c["status"]}')
    c = curves[0][1]
    ax.set_title(title, fontsize=13, wrap=True)
    axis = 'Absolute model cycle' if absolute else 'Cycles since each curve first selected event'
    ax.set_xlabel(f'{axis}\n{LABELS[c["metric"]]}; scope={c["scope"]}; window={c["window"]}\n'
                  'Zero tail is display padding, not measured kernel end', fontsize=9)
    ax.set_ylabel('Instructions / cycle')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc='upper right')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Batch results, dumps subtree, or one model log directory')
    p.add_argument('--output', type=Path, required=True, help='New plot directory (must not exist)')
    p.add_argument('--metric', choices=LABELS, default='done')
    p.add_argument('--scope', choices=('compute', 'exu', 'all-rv'), default='compute')
    p.add_argument('--window', type=int, default=10)
    p.add_argument('--max-cycles', type=int, default=2_000_000)
    p.add_argument('--absolute-cycle', action='store_true')
    p.add_argument('--compare', action='store_true', help='Overlay original/software or softmax variants when both are present')
    p.add_argument('--png', action='store_true', help='Also export PNG; requires matplotlib')
    args = p.parse_args()
    if args.window < 1 or args.max_cycles < 1:
        p.error('window and max-cycles must be positive')
    if args.metric == 'exu-issue' and args.scope == 'all-rv':
        p.error('EXU issue cannot include non-EXU RV instructions')
    if args.png:
        try:
            import matplotlib
        except ImportError:
            p.error('--png requires matplotlib: install it in this Python environment with python -m pip install matplotlib')
    root, out = args.input.resolve(), args.output.resolve()
    if not root.is_dir():
        p.error('input directory does not exist')
    groups = discover(root)
    if not groups:
        p.error('No core0.veccore0 logs found')
    out.mkdir(parents=True, exist_ok=False)
    records, paired, sections = [], defaultdict(list), []
    for (folder, prefix), logs in groups:
        relative = str(folder.relative_to(root))
        label = prefix.split('__')[0] if '__' in prefix else relative
        if label == '.':
            label = root.name
        key = relative + '/' + prefix
        slug = re.sub(r'[^\w.-]', '_', label)[:100] + '_' + hashlib.sha256(key.encode()).hexdigest()[:8]
        record = dict(name=label, directory=relative, logs={k:[str(f.relative_to(root)) for f in v] for k,v in logs.items()})
        try:
            data = measure(logs, args.metric, args.scope, args.window, args.max_cycles)
            record.update({k:v for k,v in data.items() if k not in ('counts', 'ipc')})
            record.update(svg=slug+'.svg', csv=slug+'.csv')
            with (out/record['csv']).open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['absolute_cycle', 'relative_cycle', 'instruction_count', 'window_ipc', 'is_tail_padding'])
                for i, (n, ipc) in enumerate(zip(data['counts'], data['ipc'])):
                    writer.writerow([data['first_cycle']+i, i, n, ipc, int(i >= data['span_cycles'])])
            plot_svg(out/record['svg'], label, [(label, data)], args.absolute_cycle)
            if args.png:
                record['png'] = slug+'.png'
                plot_png(out/record['png'], label, [(label, data)], args.absolute_cycle)
            sections.append(f'<h2>{html.escape(label)}</h2><a href="{record["csv"]}">CSV</a><img src="{record["svg"]}" alt="IPC plot"/>')
            if args.compare:
                match = re.fullmatch(r'(.*)_(original|software|no_membar|three_loop)', label)
                if match:
                    # Never silently collapse repeated experiments with the same label.
                    paired[match[1]].append((label, data))
        except (OSError, ValueError) as exc:
            record.update(status='error', error=str(exc))
            sections.append(f'<h2>{html.escape(label)}</h2><p>{html.escape(str(exc))}</p>')
        records.append(record)
        print(f'[{len(records)}/{len(groups)}] {label}: {record["status"]}', flush=True)
    comparisons = []
    for label, curves in paired.items():
        if len(curves) != 2 or len({name for name, _ in curves}) != 2:
            comparisons.append(dict(name=label, status='skipped', reason='Need exactly two distinct variants in this input'))
            continue
        name = 'compare_' + re.sub(r'[^\w.-]', '_', label)[:100] + '_' + hashlib.sha256(label.encode()).hexdigest()[:8] + '.svg'
        plot_svg(out/name, label + ' comparison', curves, args.absolute_cycle)
        sections.insert(0, f'<h2>{html.escape(label)} comparison</h2><img src="{name}" alt="IPC comparison"/>')
        comparisons.append(dict(name=label, status='plotted', svg=name))
        if args.png:
            png = name.removesuffix('.svg')+'.png'
            plot_png(out/png, label + ' comparison', curves, args.absolute_cycle)
            comparisons[-1]['png'] = png
    (out/'summary.json').write_text(json.dumps(dict(metric=args.metric, scope=args.scope,
        window=args.window, cases=records, comparisons=comparisons), indent=2)+'\n')
    (out/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>CAmodel IPC</title>'
        '<style>body{font-family:sans-serif;margin:24px;max-width:1250px}img{display:block;width:100%;height:auto}h2{font-size:18px}</style>'
        '</head><body><h1>CAmodel IPC</h1><p>Observed log events, not hardware peak IPC or proof of correctness. '
        'Each curve defaults to its own first selected event, not kernel entry. See summary.json for missing events.</p>'
        + '\n'.join(sections) + '</body></html>')
    return int(any(r['status'] != 'observed' for r in records))


if __name__ == '__main__':
    raise SystemExit(main())
