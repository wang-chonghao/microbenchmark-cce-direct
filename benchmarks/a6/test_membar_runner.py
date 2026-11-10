import gzip
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import run_a6_membar as runner


class MembarRunnerTests(unittest.TestCase):
    def test_byte_golden_and_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp)
            (r/'run').mkdir()
            (r/'inputs').mkdir()
            data = struct.pack('<64f', *range(64))
            (r/'inputs/input_input.bin').write_bytes(data)
            (r/'run/input_input.bin').write_bytes(data)
            (r/'run/output_output.bin').write_bytes(data*8)
            (r/'run/guard_check.txt').write_text('pass: guards intact')
            self.assertTrue(runner.check_golden(r)['passed'])
            (r/'run/output_output.bin').write_bytes(data*7)
            self.assertFalse(runner.check_golden(r)['passed'])
            (r/'run/output_output.bin').write_bytes(data*8)
            (r/'run/guard_check.txt').write_text('fail')
            self.assertFalse(runner.check_golden(r)['passed'])

    def test_archive_preserves_details_and_rotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp)/'work'
            src = r/'run/log_ca'
            src.mkdir(parents=True)
            (src/'core0.veccore0.instr_log.dump').write_text('first\n')
            (src/'core0.veccore0.instr_log.dump.1.gz').write_bytes(gzip.compress(b'rotated\n'))
            for kind in ['rvec.LSU','rvec.SU','rvec.simd.depmgr']:
                (src/f'core0.veccore0.{kind}.dump').write_text('detail\n')
            (src/'core1.veccore0.instr_log.dump').write_text('other core')
            (src/'config.json').write_text('{}')
            dest = Path(tmp)/'logs'
            archive = runner.collect_logs(r, dest)
            self.assertEqual(archive['removed_other_dumps'], 1)
            self.assertEqual((dest/'core0.veccore0.instr_log.dump').read_text(), 'first\nrotated\n')
            self.assertTrue((dest/'core0.veccore0.rvec.simd.depmgr.dump').exists())
            self.assertTrue((src/'config.json').exists())
            self.assertEqual(len(archive['raw_files']), 5)

    def test_compile_contract_and_continue_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'results'
            argv = ['run_a6_membar.py', '--compile-only', '--cann-home', '/compiler', '--output', str(out)]
            def compile_case(cmd, log, timeout, env):
                self.assertEqual(env['CCEC_EXTRA_FLAGS'], runner.FLAGS)
                self.assertEqual(env['ACL_DEVICE_ID'], '0')
                self.assertTrue((log.parent/'probe_common.h').exists())
                return 1
            with patch('sys.argv', argv), patch('run_a6_membar.run_command', side_effect=compile_case) as run, patch('builtins.print'):
                self.assertEqual(runner.main(), 1)
                self.assertEqual(run.call_count, 4)
            result = json.loads((out/'measurement/overview.json').read_text())
            self.assertFalse(result['calibrated'])
            self.assertEqual(len(result['cases']), 4)


if __name__ == '__main__':
    unittest.main()
