import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import run_a6_operators

from benchmarks.a6.import_spill_report import adapt, calls
from benchmarks.a6.spill_compare import compare_bytes

ROOT = Path(__file__).resolve().parents[2] / 'op_kernel/a6_spill_operators'


class SpillTests(unittest.TestCase):
    def test_import_reproducible(self):
        m = json.loads((ROOT / 'manifest.json').read_text())
        self.assertEqual(len(m['cases']) + len(m['excluded']), 133)
        for c in m['cases']:
            with self.subTest(case=c['name']):
                original = (ROOT/c['name']/'original.cce').read_bytes().decode()
                generated, layout = adapt(original)
                self.assertEqual(generated.encode(), (ROOT/c['source']).read_bytes())
                self.assertEqual(hashlib.sha256(generated.encode()).hexdigest(), c['source_sha256'])
                data = (ROOT/c['input']).read_bytes()
                self.assertEqual(len(data), c['input_bytes'])
                self.assertEqual(hashlib.sha256(data).hexdigest(), c['input_sha256'])
                lo, hi = c['offsets_ub']
                for region in c['outputs']:
                    self.assertTrue(hi <= region['ub_start'] or lo >= region['ub_start']+region['bytes'])
                self.assertEqual(original.count('mem_bar('), generated.count('mem_bar('))

    def test_call_reader(self):
        self.assertEqual(calls('copy((float*)get_imm(0), x, 3)', 'copy')[0][2],
                         ['(float*)get_imm(0)', 'x', '3'])
        with self.assertRaises(ValueError):
            calls('copy(x', 'copy')
        with self.assertRaises(ValueError):
            adapt('void kernel() {}')

    def test_compare(self):
        r = [dict(dtype='float', gm_offset=0, bytes=8)]
        a = struct.pack('<2f', 1, 2)
        self.assertTrue(compare_bytes(a, a, r)['passed'])
        self.assertFalse(compare_bytes(a, struct.pack('<2f', 1, 3), r)['passed'])
        n = struct.pack('<2f', float('nan'), 2)
        self.assertFalse(compare_bytes(n, n, r)['passed'])

    def test_batch_continues_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'results'
            argv = ['run_a6_operators.py', '--suite', 'spill', '--families', 'swiglu',
                    '--cann-home', '/test/compiler', '--output', str(out)]
            def execute(command, console, timeout, env):
                run = console.parent
                if run.name == 'swiglu-u1_original':
                    return 1
                target = run/'run'
                target.mkdir()
                (target/'output_output.bin').write_bytes(bytes(12288))
                return 0
            with patch('sys.argv', argv), patch('run_a6_operators.run_command', side_effect=execute), patch('builtins.print'):
                self.assertEqual(run_a6_operators.main(), 1)
            rows = json.loads((out/'summary.json').read_text())
            self.assertEqual([r['status'] for r in rows], ['failed', 'executed_unvalidated', 'executed_unvalidated'])
            pairs = json.loads((out/'comparisons.json').read_text())
            self.assertEqual(pairs[0]['status'], 'matched')


if __name__ == '__main__':
    unittest.main()
