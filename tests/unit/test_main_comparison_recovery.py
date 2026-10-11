from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from run_qwen3_serial_replay import write_json
from specfit_main_recovery import prepare_recovery, prepare_reference_reuse, reused_stream, runtime_identity
from tests.unit import test_selection_refinement as selection_fixture


class MainComparisonRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.previous = self.root/'previous'/'formal-gpu0'
        self.previous.mkdir(parents=True)
        self.destination = self.root/'resumed'
        self.destination.mkdir()
        self.frozen = dict(protocol='test', git_commit='new', worker=0, smoke=False,
            method_order=['ar', 'eagle_no_cache', 'eagle_cache', 'specfit'], dataset_order=['gsm8k'],
            datasets={'gsm8k': dict(question_ids=[401, 402], total_tokens=512)},
            cost_model=selection_fixture.SelectionRefinementTest.model().to_dict())
        write_json(self.previous/'manifest.json', dict(self.frozen, git_commit='old'))
        self.identity = patch('specfit_main_recovery.runtime_identity', return_value={'runtime': 'same'})
        self.identity.start()
        self.addCleanup(self.identity.stop)
        self.verify = patch('specfit_main_recovery.verify_record')
        self.verify.start()
        self.addCleanup(self.verify.stop)

    def stream(self, method, *, reference, count=2, broken_cache=False):
        descs = []
        for i in range(count):
            row = dict(protocol='test', status='complete', dataset='gsm8k', method=method,
                question_id=401+i, ordinal=i, repetition=-1 if reference else 0,
                pre_cache_occupied=0 if i == 0 else 1,
                pre_cache_sha256='bad' if broken_cache and i else str(i), post_cache_sha256=str(i+1),
                output_token_ids=[1, 2], query_proofs=[], cycles=[])
            name=f'{"reference" if reference else "measure"}-gsm8k-{method}-q{401+i}.json.gz'
            descs.append(write_json(self.previous/name, row))
        return descs

    def prepare(self):
        return prepare_recovery(self.previous, self.destination, self.frozen, self.root/'source')

    def test_reuses_complete_references_but_restarts_unsealed_measurements(self):
        self.stream('ar', reference=True)
        self.stream('eagle_cache', reference=True, count=1)
        self.stream('ar', reference=False, count=1)
        recovery = self.prepare()
        self.assertEqual(len(recovery['reused']['references']), 2)
        self.assertEqual(recovery['reused']['measurements'], [])
        self.assertEqual(len(recovery['excluded_partial_measurements_or_references']), 2)
        self.assertEqual(len(list(self.previous.glob('*.json.gz'))), 4)
        self.assertEqual(len(list(self.destination.glob('*.json.gz'))), 2)

    def test_reuses_sealed_dataset_with_all_methods_and_binds_bytes(self):
        refs = sum((self.stream(m, reference=True) for m in ('ar', 'eagle_cache', 'specfit')), [])
        measured = sum((self.stream(m, reference=False) for m in self.frozen['method_order']), [])
        write_json(self.previous/'gsm8k-complete.json', dict(dataset='gsm8k', references=refs, measurements=measured))
        recovery = self.prepare()
        frozen = dict(self.frozen, recovery=recovery)
        self.assertEqual(len(recovery['reused']['measurements']), 8)
        self.assertEqual(len(reused_stream(self.destination, frozen, 'gsm8k', 'ar', False)), 2)
        target = self.destination/measured[0]['basename']
        target.write_bytes(target.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'file changed'):
            reused_stream(self.destination, frozen, 'gsm8k', 'ar', False)

    def test_rejects_partial_reuse_and_broken_cache_chain(self):
        refs = self.stream('ar', reference=True, broken_cache=True)
        with self.assertRaisesRegex(ValueError, 'cache continuity'):
            self.prepare()
        frozen = dict(self.frozen, recovery=dict(reused=dict(references=refs[:1])))
        with self.assertRaisesRegex(ValueError, 'partial recovery'):
            reused_stream(self.destination, frozen, 'gsm8k', 'ar', True)

    def test_rejects_changed_protocol_before_copying(self):
        self.stream('ar', reference=True)
        self.frozen['worker'] = 1
        with self.assertRaisesRegex(ValueError, 'protocol changed: worker'):
            self.prepare()
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_rejects_changed_runtime_or_invalid_record(self):
        self.stream('ar', reference=True)
        with patch('specfit_main_recovery.runtime_identity', side_effect=[{'x': 1}, {'x': 2}]):
            with self.assertRaisesRegex(ValueError, 'sources changed'):
                self.prepare()
        with patch('specfit_main_recovery.verify_record', side_effect=ValueError('invalid reference')):
            with self.assertRaisesRegex(ValueError, 'invalid reference'):
                self.prepare()

    def test_rejects_broken_seal_and_existing_destination_file(self):
        refs = self.stream('ar', reference=True)
        with (self.destination/refs[0]['basename']).open('xb') as handle:
            handle.write(b'preserve')
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual((self.destination/refs[0]['basename']).read_bytes(), b'preserve')
        write_json(self.previous/'gsm8k-complete.json', dict(dataset='gsm8k', references=refs[:1], measurements=[]))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.prepare()


class RuntimeIdentityTest(unittest.TestCase):
    def test_allows_only_orchestration_edits_and_rejects_inference_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'scripts').mkdir()
            (root/'src').mkdir()
            runner = root/'scripts/run_specfit_main_comparison.py'
            code = """def run(args):
    def question(): return 1
    try:
        for dataset in frozen['dataset_order']:
            run_dataset(dataset)
    finally:
        cleanup()
def validate(directory): return True
"""
            runner.write_text(code)
            first = runtime_identity(root)
            runner.write_text(code.replace('run_dataset(dataset)', 'reuse_or_run(dataset)'))
            self.assertEqual(first, runtime_identity(root))
            runner.write_text(code.replace('return 1', 'return 2'))
            self.assertNotEqual(first, runtime_identity(root))
            runner.write_text(code)
            (root/'src/runtime.py').write_text('changed = True\n')
            self.assertNotEqual(first, runtime_identity(root))


if __name__ == '__main__':
    unittest.main()

    def test_reference_reuse_copies_only_sealed_references(self):
        refs = sum((self.stream(m, reference=True) for m in ('ar', 'eagle_cache', 'specfit')), [])
        measured = self.stream('eagle_cache', reference=False)
        write_json(self.previous/'gsm8k-complete.json', dict(dataset='gsm8k', references=refs, measurements=measured))
        write_json(self.previous/'manifest.json', dict(self.frozen, git_commit='old', slots_per_layer=48,
            datasets={'gsm8k': dict(self.frozen['datasets']['gsm8k'], identity='data')}))
        frozen = dict(self.frozen, reference_methods=['eagle_cache', 'specfit'], slots_per_layer=16,
            datasets={'gsm8k': dict(self.frozen['datasets']['gsm8k'], identity='data')})
        reuse = prepare_reference_reuse(self.previous, self.destination, frozen)
        self.assertEqual([d['basename'] for d in reuse['reused']['references']], [d['basename'] for d in refs[2:]])
        self.assertEqual(reuse['reused']['measurements'], [])
        self.assertEqual(reuse['source_slots_per_layer'], 48)
        self.assertEqual(sorted(p.name for p in self.destination.glob('*.json.gz')), sorted(d['basename'] for d in refs[2:]))
        self.assertEqual(len(reused_stream(self.destination, dict(frozen, recovery=reuse), 'gsm8k', 'specfit', True)), 2)
        self.assertEqual(reused_stream(self.destination, dict(frozen, recovery=reuse), 'gsm8k', 'specfit', False), [])

    def test_reference_reuse_rejects_other_worker_or_dataset(self):
        refs = sum((self.stream(m, reference=True) for m in ('eagle_cache', 'specfit')), [])
        write_json(self.previous/'gsm8k-complete.json', dict(dataset='gsm8k', references=refs, measurements=[]))
        write_json(self.previous/'manifest.json', dict(self.frozen, git_commit='old', slots_per_layer=48,
            datasets={'gsm8k': dict(self.frozen['datasets']['gsm8k'], identity='data')}))
        frozen = dict(self.frozen, reference_methods=['eagle_cache', 'specfit'],
            datasets={'gsm8k': dict(self.frozen['datasets']['gsm8k'], identity='other')})
        with self.assertRaisesRegex(ValueError, 'dataset changed'):
            prepare_reference_reuse(self.previous, self.destination, frozen)
        frozen['datasets']['gsm8k']['identity'] = 'data'
        with self.assertRaisesRegex(ValueError, 'protocol/worker'):
            prepare_reference_reuse(self.previous, self.destination, dict(frozen, worker=1))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            prepare_reference_reuse(self.previous, self.destination, dict(frozen, reference_methods=['ar']))
