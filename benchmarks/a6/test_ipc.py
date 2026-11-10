import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import plot_a6_ipc as ipc


def instruction(cycle, ident, op='RV_VADD'):
    return f'[info] [{cycle:08d}] (PC: 0x1000) (ID: {ident}) {op} Vd[0] Vn[1]\n'


def execution(cycle, ident, op='RV_VADD'):
    return f'[info] [PERF] [{cycle:010d}] EXU instr_name {op} type.F32 instr_id {ident} PC 1000 retire {cycle+4} exu_id:0\n'


def fixture(root, prefix='core0.veccore0'):
    root.mkdir(parents=True, exist_ok=True)
    logs = {}
    data = {
        'instr_log': instruction(10, 1)+instruction(10, 2)+instruction(12, 3)
                     +instruction(13, 99, 'RV_SEND')+instruction(14, 99, 'RV_SEND'),
        'instr_popped_log': instruction(6, 1)+instruction(7, 2)+instruction(8, 3),
        'rvec.EXU': execution(6, 1)+execution(7, 2)+execution(8, 3),
    }
    for kind, text in data.items():
        path = root/f'{prefix}.{kind}.dump'
        path.write_text(text)
        logs[kind] = [path]
    return logs


class IPCTests(unittest.TestCase):
    def test_cycles_and_zero_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = fixture(Path(tmp))
            result = ipc.measure(logs, 'done', 'compute', 2, 100)
            self.assertEqual(result['counts'], [2, 0, 1])
            self.assertEqual(result['ipc'], [1, 1, .5])
            self.assertEqual(result['average_ipc'], 1)
            self.assertEqual(result['status'], 'observed')
            self.assertEqual(ipc.measure(logs, 'exu-issue', 'compute', 1, 100)['first_cycle'], 6)
            self.assertEqual(ipc.measure(logs, 'popped', 'compute', 1, 100)['counts'], [1, 1, 1])

    def test_rotation_and_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = fixture(root)
            rotated = root/'core0.veccore0.instr_log.dump.1'
            rotated.write_text(instruction(10, 1))
            discovered = ipc.discover(root)
            self.assertEqual(len(discovered), 1)
            result = ipc.measure(discovered[0][1], 'done', 'compute', 2, 100)
            self.assertEqual(result['duplicates_removed'], 1)
            logs['instr_log'][0].write_text(instruction(10, 1))
            result = ipc.measure(logs, 'done', 'compute', 2, 100)
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['missing_exu_matched_events'], 2)

    def test_missing_and_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = fixture(Path(tmp))
            with self.assertRaises(ValueError):
                ipc.measure(logs, 'done', 'compute', 10, 1)
            with self.assertRaises(ValueError):
                ipc.measure({'instr_log':logs['instr_log']}, 'done', 'compute', 10, 100)
            logs['instr_log'][0].write_text(instruction(10, 1)+instruction(20, 1))
            with self.assertRaises(ValueError):
                ipc.measure(logs, 'done', 'compute', 10, 100)

    def test_batch_svg_csv_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'logs'
            out = Path(tmp)/'plots'
            for role in ('original', 'software'):
                name = 'test_'+role
                fixture(root/name, name+'__'+name+'__core0.veccore0')
            # Other cores must not affect statistics.
            (root/'core1.veccore0.instr_log.dump').write_text(instruction(1, 1))
            argv = ['plot_a6_ipc.py', '--input', str(root), '--output', str(out), '--compare']
            with patch('sys.argv', argv), patch('builtins.print'):
                self.assertEqual(ipc.main(), 0)
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(len(summary['cases']), 2)
            self.assertEqual(summary['comparisons'][0]['status'], 'plotted')
            for svg in out.glob('*.svg'):
                self.assertEqual(ET.parse(svg).getroot().tag, '{http://www.w3.org/2000/svg}svg')
            for csv in out.glob('*.csv'):
                self.assertEqual(len(csv.read_text().splitlines()), 4)
            self.assertTrue((out/'index.html').exists())


if __name__ == '__main__':
    unittest.main()
