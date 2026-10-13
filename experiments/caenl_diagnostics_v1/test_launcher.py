"""CPU checks for the completion gate, pairing, bounded protocol and packaging."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import caenl_diagnostics_v1 as cli
import common


class LauncherContracts(unittest.TestCase):
    def test_active_campaign_waits_even_if_results_exist(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            cli.atomic(p/'STATUS.json', dict(status='RUNNING'))
            with patch.object(cli, 'process_active', return_value=True):
                self.assertEqual(cli.readiness(p), (False, 'WAITING_FOR_SOLIDIFICATION'))

    def test_blocked_campaign_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            cli.atomic(p/'STATUS.json', dict(status='BLOCKED', message='disk full'))
            with patch.object(cli, 'process_active', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'disk full'):
                    cli.readiness(p)
            self.assertEqual(cli.read(p/'STATUS.json')['status'], 'BLOCKED')

    def test_missing_worker_does_not_wait_forever(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            cli.atomic(p/'STATUS.json', dict(status='RUNNING'))
            with patch.object(cli, 'process_active', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'no active worker'):
                    cli.readiness(p)

    def test_only_complete_206_campaign_passes_gate(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            cli.atomic(p/'STATUS.json', dict(status='COMPLETE'))
            with patch.object(cli, 'process_active', return_value=False):
                self.assertEqual(cli.readiness(p), (False, 'WAITING_FOR_COMPLETION_RECORD'))
                cli.atomic(p/'COMPLETE.json', dict(status='COMPLETE', method_runs=205))
                with self.assertRaisesRegex(RuntimeError, 'counts'):
                    cli.readiness(p)
                cli.atomic(p/'COMPLETE.json', dict(status='COMPLETE', method_runs=206))
                cli.atomic(p/'RESULTS.json', {})
                self.assertEqual(cli.readiness(p), (True, 'READY'))

    def test_source_integrity_and_safe_paths(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'x.py').write_text('finite = True\n')
            h = cli.sha(p/'x.py')
            cli.verify_files(p, {'x.py':h})
            (p/'x.py').write_text('finite = False\n')
            with self.assertRaises(RuntimeError):
                cli.verify_files(p, {'x.py':h})
            with self.assertRaisesRegex(RuntimeError, 'Unsafe'):
                cli.verify_files(p, {'../x.py':h})

    def test_paired_units_and_exact_resolution(self):
        out = common.paired(np.ones(10), np.zeros(10))
        self.assertEqual(out['difference_pp'], 1.)
        self.assertEqual(out['exact_signflip_p'], 2/1024)
        null = common.paired(np.zeros(10), np.zeros(10))
        self.assertEqual(null['exact_signflip_p'], 1.)
        with self.assertRaises(ValueError):
            common.paired(np.zeros(10), np.zeros(9))
        self.assertEqual(common.holm([.01,.03,.2]), [.03,.06,.2])

    def test_finite_protocol_and_separate_primary_family(self):
        p = cli.read(cli.HERE/'protocol.json')
        self.assertEqual(len(p['seeds'])*len(p['methods']), 30)
        self.assertEqual(30*len(p['robustness']['radii']), 150)
        self.assertEqual(p['robustness']['primary'], 2)
        self.assertEqual(len(p['robustness']['primary_contrasts']), 3)
        self.assertEqual(p['queue_deadline_days'], 20)
        self.assertEqual(p['cost']['repetitions']*len(p['cost']['conditions']), 30)

    def test_package_has_hash_bound_numeric_evidence_and_no_models(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            common.atomic_json(p/'RESULTS.json', dict(status='COMPLETE'))
            common.atomic_npz(p/'vectors.npz', x=np.arange(12))
            (p/'FINAL_MODEL.pt').write_bytes(b'checkpoint remains outside exported review')
            archive, parts = cli.package(p)
            with zipfile.ZipFile(archive) as z:
                self.assertNotIn('FINAL_MODEL.pt', z.namelist())
                manifest = json.loads(z.read('REVIEW_MANIFEST.json'))
                self.assertEqual(manifest['RESULTS.json'], common.sha(p/'RESULTS.json'))
            self.assertEqual(len(parts), 1)
            with zipfile.ZipFile(p/'review'/parts[0]['name']) as z:
                manifest = json.loads(z.read('PART_MANIFEST.json'))
                self.assertEqual(manifest['vectors.npz'], common.sha(p/'vectors.npz'))
            self.assertEqual(common.sha(p/'review'/parts[0]['name']), parts[0]['sha256'])
            self.assertTrue((p/'FINAL_MODEL.pt').exists())


if __name__ == '__main__':
    unittest.main()
