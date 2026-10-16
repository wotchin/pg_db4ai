"""Fast harness regressions; no Docker, downloads or benchmark reruns required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess
import run
import export_transport


class HarnessTests(unittest.TestCase):
    def test_cli_timing_is_selected_by_marker(self):
        text='banner\nTime: 90 ms\n__BENCH__train_0\nMODEL CREATED\nTime: 12.5 ms\n__BENCH__predict_0\n42\nTime: 0.25 ms\n'
        self.assertEqual(run.timings(text), {'train_0': 12.5, 'predict_0': .25})

    def test_duplicate_marker_is_rejected(self):
        with self.assertRaises(ValueError):
            run.timings('__BENCH__train_0\nTime: 1 ms\n__BENCH__train_0\nTime: 2 ms')

    def test_shared_tolerance_is_positive(self):
        for name in ['linear_gd', 'linear_ngd', 'logistic_gd', 'ova_svm']:
            self.assertGreater(run.config(name, 42)['tolerance'], 0)

    def test_export_read_through_owner_then_runner_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'og_xgboost-predictions.csv'
            path.write_text('unreadable original placeholder')
            raw = '0,0.125\n1,0.875\n'
            with patch.object(export_transport.subprocess, 'run') as execute:
                execute.return_value = subprocess.CompletedProcess([], 0, raw, '')
                export_transport.collect_export(path, '/bench/case/' + path.name, 'og')
            self.assertEqual(path.read_text(), raw)
            self.assertEqual(execute.call_args.args[0],
                             ['docker', 'exec', '-u', 'omm', 'bench-og', 'cat', '--', '/bench/case/' + path.name])

    def test_failed_export_keeps_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.csv'
            path.write_text('original')
            with patch.object(export_transport.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, [])):
                with self.assertRaises(subprocess.CalledProcessError):
                    export_transport.collect_export(path, '/bench/predictions.csv', 'og')
            self.assertEqual(path.read_text(), 'original')

    def test_bootstrap_server_is_not_ready(self):
        with patch.object(export_transport.subprocess, 'run') as execute:
            execute.return_value = subprocess.CompletedProcess([], 0, 'bash\n', '')
            with self.assertRaisesRegex(RuntimeError, 'bootstrap'):
                export_transport.require_final_server('og')

    def test_final_server_is_ready(self):
        with patch.object(export_transport.subprocess, 'run') as execute:
            for engine, process in [('pg', 'postgres'), ('og', 'gaussdb')]:
                execute.return_value = subprocess.CompletedProcess([], 0, process+'\n', '')
                export_transport.require_final_server(engine)


if __name__ == '__main__':
    unittest.main()
