"""Recover complete streams into a new attempt without splicing cache histories."""
import ast
from hashlib import sha256
from pathlib import Path

from specfit.main_comparison import REFERENCE_METHODS, reference_method, verify_record
from specfit.selection_methods import SelectionCostModel
from specfit.serial_replay import file_identity
from run_qwen3_serial_replay import read_json


def runtime_identity(root):
    """Bind inference sources, including the runner outside its dataset loop."""
    excluded = {'run_specfit_main_comparison.py', 'control_specfit_main_comparison.py',
                'specfit_main_recovery.py'}
    files = {str(p.relative_to(root)): file_identity(p)['sha256']
             for folder in ('src', 'scripts') for p in (root/folder).rglob('*.py')
             if p.name not in excluded}
    tree = ast.parse((root/'scripts/run_specfit_main_comparison.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('run', 'validate'):
            if node.name == 'run':
                replaced = 0
                for child in ast.walk(node):
                    if isinstance(child, ast.Try):
                        for i, statement in enumerate(child.body):
                            if (isinstance(statement, ast.For)
                                    and isinstance(statement.target, ast.Name)
                                    and statement.target.id == 'dataset'
                                    and ast.unparse(statement.iter) == "frozen['dataset_order']"):
                                child.body[i] = ast.Pass()
                                replaced += 1
                if replaced != 1:
                    raise ValueError('cannot isolate dataset orchestration')
            files['runner_function:'+node.name] = sha256(ast.dump(node).encode()).hexdigest()
    if not all('runner_function:'+n in files for n in ('run', 'validate')):
        raise ValueError('missing inference/validation function')
    return files


def check_stream(rows, frozen, dataset, method, reference):
    expected = frozen['datasets'][dataset]['question_ids']
    if ([r['question_id'] for r in rows] != expected
            or any(r['dataset'] != dataset or r['method'] != method
                   or r['protocol'] != frozen['protocol'] or r['ordinal'] != i
                   or r['repetition'] != (-1 if reference else frozen['worker'])
                   for i, r in enumerate(rows))):
        raise ValueError('recovery stream identity/order changed')
    if rows[0]['pre_cache_occupied'] != 0 or any(
            a['post_cache_sha256'] != b['pre_cache_sha256'] for a, b in zip(rows, rows[1:])):
        raise ValueError('recovery stream cache continuity changed')


def prepare_recovery(previous, destination, frozen, source_root):
    """Reuse sealed datasets and complete references; rerun other measured streams."""
    previous = previous.resolve()
    old = read_json(previous/'manifest.json')
    for key, value in frozen.items():
        if key != 'git_commit' and old.get(key) != value:
            raise ValueError('recovery protocol changed: '+key)
    old_source = previous.parent/('source-'+old['git_commit'][:7])
    identities = runtime_identity(source_root)
    if runtime_identity(old_source) != identities:
        raise ValueError('recovery inference/validation sources changed')
    model = SelectionCostModel.from_dict(frozen['cost_model'])
    reusable = dict(references=[], measurements=[])
    ignored = []
    for dataset in frozen['dataset_order']:
        complete_path = previous/f'{dataset}-complete.json'
        sealed = read_json(complete_path) if complete_path.exists() else None
        if sealed is not None and sealed['dataset'] != dataset:
            raise ValueError('sealed dataset identity changed')
        references = {}
        groups = []
        for method in REFERENCE_METHODS:
            names = [f'reference-{dataset}-{method}-q{q}.json.gz'
                     for q in frozen['datasets'][dataset]['question_ids']]
            if sealed is not None:
                descs = [d for d in sealed['references'] if d['basename'] in names]
                if [d['basename'] for d in descs] != names:
                    raise ValueError('sealed reference coverage/order changed')
            elif all((previous/name).is_file() for name in names):
                descs = [file_identity(previous/name) for name in names]
            else:
                ignored.extend(name for name in names if (previous/name).exists())
                continue
            rows = [read_json(previous/d['basename'], d) for d in descs]
            check_stream(rows, frozen, dataset, method, True)
            for row in rows:
                verify_record(row, None, model, frozen['datasets'][dataset]['total_tokens'])
                references[method, row['question_id']] = row
            groups.append(('references', descs))
        if sealed is not None:
            measured = {}
            for method in frozen['method_order']:
                names = [f'measure-{dataset}-{method}-q{q}.json.gz'
                         for q in frozen['datasets'][dataset]['question_ids']]
                descs = [d for d in sealed['measurements'] if d['basename'] in names]
                if [d['basename'] for d in descs] != names:
                    raise ValueError('sealed measurement coverage/order changed')
                rows = [read_json(previous/d['basename'], d) for d in descs]
                check_stream(rows, frozen, dataset, method, False)
                for row in rows:
                    reference = references[reference_method(method), row['question_id']]
                    verify_record(row, reference, model, frozen['datasets'][dataset]['total_tokens'])
                    measured[method, row['question_id']] = row
                groups.append(('measurements', descs))
            for q in frozen['datasets'][dataset]['question_ids']:
                a, b = (measured[m, q] for m in ('eagle_no_cache', 'eagle_cache'))
                if any(a[k] != b[k] for k in ('output_token_ids', 'query_proofs', 'cycles')):
                    raise ValueError('sealed cache switch changed queries')
            if (len(sealed['references']) != len(references)
                    or len(sealed['measurements']) != len(measured)):
                raise ValueError('unexpected sealed records')
        else:
            ignored.extend(p.name for p in previous.glob(f'measure-{dataset}-*.json.gz'))
        for kind, descs in groups:
            for desc in descs:
                raw = (previous/desc['basename']).read_bytes()
                if len(raw) != desc['size_bytes'] or sha256(raw).hexdigest() != desc['sha256']:
                    raise ValueError('source artifact changed during recovery')
                with (destination/desc['basename']).open('xb') as handle:
                    handle.write(raw)
                reusable[kind].append(desc)
    return dict(source_directory=str(previous), source_manifest=file_identity(previous/'manifest.json'),
                source_git_commit=old['git_commit'], inference_source_identity=identities,
                policy='reuse_sealed_datasets_and_complete_references; rerun_other_measurements_from_stream_start',
                reused=reusable, excluded_partial_measurements_or_references=sorted(ignored))


def reused_stream(directory, frozen, dataset, method, reference):
    """Require all-or-none coverage before skipping a complete stream."""
    kind = 'references' if reference else 'measurements'
    prefix = ('reference' if reference else 'measure')+f'-{dataset}-{method}-'
    descs = [d for d in frozen.get('recovery', {}).get('reused', {}).get(kind, [])
             if d['basename'].startswith(prefix)]
    if not descs:
        return []
    expected = [prefix+f'q{q}.json.gz' for q in frozen['datasets'][dataset]['question_ids']]
    if [d['basename'] for d in descs] != expected:
        raise ValueError('partial recovery stream cannot preserve natural caches')
    for desc in descs:
        if file_identity(directory/desc['basename']) != desc:
            raise ValueError('recovered file changed')
    return descs


def prepare_reference_reuse(previous, destination, frozen):
    """Copy sealed same-query references only; every measured stream starts fresh."""
    previous = previous.resolve()
    old = read_json(previous/'manifest.json')
    if old['protocol'] != frozen['protocol'] or old['worker'] != frozen['worker'] or old['smoke']:
        raise ValueError('reference source protocol/worker changed')
    model = SelectionCostModel.from_dict(frozen['cost_model'])
    reused = []
    for dataset in frozen['dataset_order']:
        sealed = read_json(previous/f'{dataset}-complete.json')
        if (sealed['dataset'] != dataset
                or old['datasets'][dataset]['identity'] != frozen['datasets'][dataset]['identity']
                or old['datasets'][dataset]['total_tokens'] != frozen['datasets'][dataset]['total_tokens']):
            raise ValueError('sealed reference dataset changed')
        # Streams not listed are generated fresh by the new run.
        for method in frozen.get('reused_reference_methods', frozen['reference_methods']):
            if method not in frozen['reference_methods']:
                raise ValueError('reused reference method is not a reference method')
            names = [f'reference-{dataset}-{method}-q{q}.json.gz'
                     for q in frozen['datasets'][dataset]['question_ids']]
            descs = [d for d in sealed['references'] if d['basename'] in names]
            if [d['basename'] for d in descs] != names:
                raise ValueError('sealed reference coverage/order changed')
            rows = [read_json(previous/d['basename'], d) for d in descs]
            check_stream(rows, frozen, dataset, method, True)
            for row in rows:
                verify_record(row, None, model, frozen['datasets'][dataset]['total_tokens'])
            for desc in descs:
                raw = (previous/desc['basename']).read_bytes()
                if len(raw) != desc['size_bytes'] or sha256(raw).hexdigest() != desc['sha256']:
                    raise ValueError('source artifact changed during reference reuse')
                with (destination/desc['basename']).open('xb') as handle:
                    handle.write(raw)
                reused.append(desc)
    return dict(source_directory=str(previous), source_manifest=file_identity(previous/'manifest.json'),
                source_git_commit=old['git_commit'], source_slots_per_layer=old['slots_per_layer'],
                policy='reuse_sealed_references_only; measured outputs and query proofs must match them',
                reused=dict(references=reused, measurements=[]))
