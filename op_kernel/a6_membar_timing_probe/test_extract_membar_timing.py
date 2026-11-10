import gzip
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from build import compiler_command
from extract_membar_timing import HERE, PATTERNS, analyze, detailed_events, direction, instruction_stream, unique

SPECS = json.loads((HERE/'experiments.json').read_text())['cases']


def fixture(root, name, *, control='SU', interval=None):
    spec = SPECS[name]
    program = [('RV_PSET', ''), ('RV_VLDI', '')]
    if spec['direction'] == 'VST_VLD':
        program += [('RV_SMEM_BAR', 'type: VLD_VST')]
        program += [('RV_VSTI', '')] * 8
    for _ in range(8):
        program.append(('RV_SMEM_BAR', 'type: '+spec['direction']))
        if spec['layout'] == 'interleaved':
            program.append(('RV_VSTI' if spec['direction'] == 'VLD_VST' else 'RV_VLDI', ''))
    if spec['direction'] == 'VLD_VST' and spec['layout'] == 'cluster':
        program += [('RV_VSTI', '')] * 8
    if spec['direction'] == 'VST_VLD':
        if spec['layout'] == 'cluster':
            program += [('RV_VLDI', '')] * 8
        program += [('RV_SMEM_BAR', 'type: VLD_VST')]
        program += [('RV_VSTI', '')] * 8
    program.append(('RV_SEND', ''))
    popped, done, isu, dep, lsu = [], [], [], [], []

    def raw(cy, pc, unit, ident, op, detail=''):
        return f'[info] [{cy:08d}] (PC: {hex(pc)}) {unit} : (Binary: 0x1) (ID: {ident}) {op} {detail}\n'

    popped.append(raw(0, 0x20, 'PUSHQ', 1, 'VF', f'addr: 0x100,instr_num: {hex(len(program))},'))
    done.append(raw(1000, 0x20, 'PUSHQ', 1, 'VF'))
    target_index, memory_index, last_target = 0, 0, None
    ids = []
    step = interval if interval is not None else 26 if name == '03_interleave_store' else 21 if spec['direction'] == 'VLD_VST' else 11
    for offset, (op, detail) in enumerate(program):
        pc, ident = 0x100 + 4*offset, offset + 10
        if op == 'RV_SMEM_BAR':
            target = detail == 'type: '+spec['direction']
            issue = 100+target_index*step if target else 30 if target_index == 0 else 500
            if target:
                ids.append(ident)
                target_index += 1
                last_target = issue
            release_offset, retire_offset = (18, 20) if 'VLD_VST' in detail else (7, 9)
            popped += [raw(10+offset, pc, 'RVECLD', ident, op, detail),
                       raw(issue+release_offset, pc, 'RVECST', ident, op, detail)]
            done.append(raw(issue+retire_offset, pc, 'RVECST', ident, op, detail))
            isu += [f'[info] [2] [{control}] [ISU_RECV] instr_name {op} instr_id {ident} PC {pc:x}\n',
                    f'[info] [{issue}] [{control}] [ISU_ISSUE] instr_name {op} instr_id {ident} PC {pc:x}\n',
                    f'[info] [{issue+retire_offset}] [{control}] [ISU_BAR_RETIRE] instr_name {op} instr_id {ident} PC {pc:x}\n']
            unit = 'STU' if 'VLD_VST' in detail else 'LDU'
            dep.append(f'[info] [{issue+release_offset}] <<consumer dec set_cnt>> bar_type {detail[6:]}, isa_id {ident}, unit_type {unit}\n')
        else:
            start = 4 if op == 'RV_PSET' else 20 if memory_index == 0 else 60+memory_index
            if spec['layout'] == 'interleaved' and last_target is not None and target_index and target_index <= 8:
                if op == ('RV_VSTI' if spec['direction'] == 'VLD_VST' else 'RV_VLDI'):
                    start = last_target + (22 if spec['direction'] == 'VLD_VST' else 8)
            if spec['direction'] == 'VST_VLD' and target_index == 8 and op == 'RV_VSTI':
                start = 610+memory_index
            unit = 'RVECST' if op == 'RV_VSTI' else 'RVECLD' if op == 'RV_VLDI' else 'RVECEX'
            popped.append(raw(start, pc, unit, ident, op, detail))
            done.append(raw(start+9, pc, unit, ident, op, detail))
            if op in ('RV_VLDI', 'RV_VSTI'):
                memory_index += 1
                queue = 'LDQ' if op == 'RV_VLDI' else 'SHQ'
                isu.append(f'[info] [2] [{queue}] [ISU_RECV] instr_name {op} instr_id {ident} PC {pc:x}\n')
                if op == 'RV_VSTI':
                    isu.append(f'[info] [{start}] [SHQ] [ISU_ISSUE] instr_name {op} instr_id {ident} PC {pc:x}\n')
                else:
                    lsu.append(f'[info] [PERF] [{start}] [SEND_UB_RD.PORT_0] req_id {ident} instr_name {op} instr_id {ident} PC {pc:x}\n')
    files = {'core0.veccore0.instr_popped_log.dump': popped, 'core0.veccore0.instr_log.dump': done,
             'core0.veccore0.rvec.ISU.dump': isu, 'core0.veccore0.rvec.simd.depmgr.dump': dep,
             'core0.veccore0.rvec.LSU.dump': lsu}
    for name, content in files.items():
        (root/name).write_text(''.join(content))
    return ids


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_all_source_contracts_are_straight_line_and_disjoint(self):
        for name, spec in SPECS.items():
            code = re.sub(r'//[^\n]*', '', (HERE/(name+'.cce')).read_text())
            self.assertNotRegex(code, r'\bfor\s*\(')
            self.assertEqual(len(re.findall(r'\bvlds\(', code)), spec['loads'])
            self.assertEqual(len(re.findall(r'\bvsts\(', code)), spec['stores'])
            for direction, count in spec['membars'].items():
                self.assertEqual(code.count('mem_bar('+direction+')'), count)

    def test_cluster_stages_not_earliest_popped_and_ports_are_deduplicated(self):
        fixture(self.root, '01_cluster_vld_vst', control='RVECSU')
        result = analyze(self.root, SPECS['01_cluster_vld_vst'])
        self.assertEqual(result['rows'][0]['issue'], 100)
        self.assertEqual(result['control_unit'], 'SU')
        self.assertEqual(result['parameter_candidates'], dict(release_latency=18, retire_latency=20, next_issue_delay=1))
        self.assertEqual(result['instruction_counts']['membars'], {'VLD_VST': 8})
        self.assertEqual(len(result['raw_instructions'][2]['popped']), 2)

    def test_other_direction_and_interleaved_deltas(self):
        for name, candidates in (
            ('02_cluster_vst_vld', dict(release_latency=7, retire_latency=9, next_issue_delay=2)),
            ('03_interleave_store', dict(consumer_delay=4)),
            ('04_interleave_load', dict(consumer_delay=1))):
            with self.subTest(name=name):
                fixture(self.root, name)
                result = analyze(self.root, SPECS[name])
                self.assertEqual(result['parameter_candidates'], candidates)
                self.assertIsNone(result['admission_delay'])
                if name.startswith(('03', '04')):
                    value = 4 if name.startswith('03') else 3
                    self.assertEqual(result['metrics']['feedback_gap_upper_bound']['minimum'], value)
                    self.assertNotIn('lsu_start_to_next_issue', result['parameter_candidates'])

    def test_missing_detail_is_not_fabricated_from_popped_or_done(self):
        fixture(self.root, '01_cluster_vld_vst')
        for name in ('ISU', 'LSU', 'simd.depmgr'):
            (self.root/f'core0.veccore0.rvec.{name}.dump').unlink()
        result = analyze(self.root, SPECS['01_cluster_vld_vst'])
        self.assertTrue(all(value is None for value in result['parameter_candidates'].values()))
        self.assertIsNone(result['rows'][2]['issue'])
        self.assertIsNone(result['serial_controller_compatible_in_observed_samples'])

    def test_missing_retirement_does_not_imply_serial_compatibility(self):
        fixture(self.root, '01_cluster_vld_vst')
        path = self.root/'core0.veccore0.rvec.ISU.dump'
        path.write_text(''.join(line for line in path.read_text().splitlines(True) if 'ISU_BAR_RETIRE' not in line))
        result = analyze(self.root, SPECS['01_cluster_vld_vst'])
        self.assertEqual(result['control_unit'], 'SU')
        self.assertIsNone(result['serial_controller_compatible_in_observed_samples'])

    def test_overlap_rejects_serial_positive_delay_candidate(self):
        fixture(self.root, '01_cluster_vld_vst', interval=5)
        result = analyze(self.root, SPECS['01_cluster_vld_vst'])
        self.assertTrue(result['next_barrier_overlaps_retirement'])
        self.assertEqual(result['metrics']['next_issue_delay_observed']['minimum'], -15)
        self.assertIsNone(result['parameter_candidates']['next_issue_delay'])

    def test_ambiguous_control_phases_require_explicit_unit(self):
        fixture(self.root, '01_cluster_vld_vst')
        path = self.root/'core0.veccore0.rvec.ISU.dump'
        text = path.read_text()
        path.write_text(text+text.replace('[SU]', '[SHQ]'))
        self.assertIsNone(analyze(self.root, SPECS['01_cluster_vld_vst'])['control_unit'])
        self.assertEqual(analyze(self.root, SPECS['01_cluster_vld_vst'], control_unit='SU')['parameter_candidates']['release_latency'], 18)

    def test_multiple_vfs_rejected_without_index(self):
        fixture(self.root, '01_cluster_vld_vst')
        path = self.root/'core0.veccore0.instr_popped_log.dump'
        path.write_text(path.read_text()+path.read_text().splitlines()[0]+'\n')
        with self.assertRaisesRegex(ValueError, 'one complete VF'):
            instruction_stream(self.root)

    def test_gzip_logs_and_incomplete_stream(self):
        fixture(self.root, '01_cluster_vld_vst')
        path = self.root/'core0.veccore0.instr_popped_log.dump'
        with gzip.open(str(path)+'.gz', 'wt') as stream:
            stream.write(path.read_text())
        path.unlink()
        self.assertEqual(analyze(self.root, SPECS['01_cluster_vld_vst'])['parameter_candidates']['release_latency'], 18)
        path = self.root/'core0.veccore0.instr_log.dump'
        path.write_text('\n'.join(path.read_text().splitlines()[1:]))
        with self.assertRaisesRegex(ValueError, 'completion'):
            analyze(self.root, SPECS['01_cluster_vld_vst'])

    def test_changed_compiled_counts_fail(self):
        fixture(self.root, '01_cluster_vld_vst')
        spec = dict(SPECS['01_cluster_vld_vst'], stores=9)
        with self.assertRaisesRegex(ValueError, 'Unexpected compiled stream'):
            analyze(self.root, spec)

    def test_zero_stack_and_target_are_explicit(self):
        command = compiler_command(Path('/compiler'), Path('/probe.cce'), Path('/probe.o'))
        self.assertIn('--cce-aicore-arch=dav-920r1-vec', command)
        self.assertIn('-cce-vf-stack-size=0', command)
        self.assertIn('-cce-aicore-vec-misched=0', command)

    def test_foreign_core_logs_do_not_pollute_measurements(self):
        fixture(self.root, '01_cluster_vld_vst')
        path = self.root/'core0.veccore0.rvec.ISU.dump'
        (self.root/'core1.veccore0.rvec.ISU.dump').write_text(path.read_text().replace('[100]', '[101]'))
        self.assertEqual(analyze(self.root, SPECS['01_cluster_vld_vst'])['parameter_candidates']['release_latency'], 18)

    def test_custom_event_pattern_and_conflicting_same_unit_event(self):
        ids = fixture(self.root, '01_cluster_vld_vst')
        path = self.root/'core0.veccore0.rvec.ISU.dump'
        text = path.read_text().replace('[ISU_ISSUE]', '[CUSTOM_ISSUE]')
        path.write_text(text)
        pattern = PATTERNS['issue'].replace('ISU_ISSUE|SU_ISSUE|ISSUE', 'CUSTOM_ISSUE')
        result = analyze(self.root, SPECS['01_cluster_vld_vst'], custom={'issue': pattern})
        self.assertEqual(result['parameter_candidates']['release_latency'], 18)
        path.write_text(text+f'[info] [999] [SU] [CUSTOM_ISSUE] instr_name RV_SMEM_BAR instr_id {ids[2]}\n')
        result = analyze(self.root, SPECS['01_cluster_vld_vst'], custom={'issue': pattern})
        self.assertIsNone(result['rows'][2]['issue'])

    def test_historical_a5_real_log_phase_alignment(self):
        root = HERE.parents[1]/'results/ub_address_dependency_experiment/a5_multi_membar_probe/clustered8/camodel'
        if not root.exists():
            self.skipTest('Optional archived A5 evidence is not installed')
        scope, ops = instruction_stream(root)
        events, _, _ = detailed_events(root, scope, ops)
        for kind, expected in (('VLD_VST', (18, 20, 1)), ('VST_VLD', (7, 9, 2))):
            bars = [i for i in ops if i['op'] == 'RV_SMEM_BAR' and direction(i) == kind]
            for index in range(1, 7):
                current = events[bars[index]['id']]
                issue = unique(current, 'issue', 'SHQ')
                release = unique(current, 'release_'+kind.lower())
                retire = unique(current, 'retire', 'SHQ')
                next_issue = unique(events[bars[index+1]['id']], 'issue', 'SHQ')
                self.assertEqual((release-issue, retire-issue, next_issue-retire), expected)

    def test_cli_four_case_overview_and_csv(self):
        logs = self.root/'logs'
        for name in SPECS:
            folder = logs/name
            folder.mkdir(parents=True)
            fixture(folder, name)
        output = self.root/'reports'
        process = subprocess.run([sys.executable, str(HERE/'extract_membar_timing.py'),
            '--logs', str(logs), '--out', str(output)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        overview = json.loads((output/'overview.json').read_text())
        self.assertEqual(overview['directions']['VLD_VST']['release_latency'], 18)
        self.assertEqual(overview['directions']['VST_VLD']['consumer_delay'], 1)
        self.assertIsNone(overview['admission_delay'])
        self.assertFalse(overview['calibrated'])
        for name in SPECS:
            result = json.loads((output/name/'summary.json').read_text())
            self.assertFalse(result['source_contract_verified'])
            self.assertTrue(result['log_sha256'])
            self.assertEqual(len((output/name/'barriers.csv').read_text(encoding='utf-8-sig').splitlines()), 9)


if __name__ == '__main__':
    unittest.main()
