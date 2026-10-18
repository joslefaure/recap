"""Run RECAP protocols on frozen vector banks."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
from threadpoolctl import threadpool_limits
from recap.data import load_manifest, load_bank, save_result
from recap.models import CONFIGS, MODELS
from recap.protocols import compression, recoverability, transfer, selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('protocol', choices=('compression', 'transfer', 'recoverability', 'selection', 'temporal', 'fresh'))
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--bank', type=Path, nargs='+', required=True)
    p.add_argument('--width', type=int)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--test-manifest', type=Path)
    p.add_argument('--test-bank', type=Path, nargs='+')
    p.add_argument('--supports', type=Path, default=Path('splits/clevr_fresh_supports.json'))
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = load_manifest(args.manifest)
    banks = [load_bank(path, manifest) for path in args.bank]
    width = args.width or max(MODELS[banks[0]['model']]['widths'])
    with threadpool_limits(limits=args.threads):
        if args.protocol == 'compression':
            if len(banks) != 1:
                p.error('Compression takes one bank')
            result, arrays = compression.run(banks[0], manifest)
        elif args.protocol in ('recoverability', 'temporal'):
            if len(banks) != 1:
                p.error('This protocol takes one bank')
            function = recoverability.nested if args.protocol == 'recoverability' else recoverability.temporal
            result, arrays = function(banks[0], manifest, width)
        elif args.protocol == 'selection':
            config = json.loads((CONFIGS/'selection.json').read_text())[manifest['task']]
            if len({b['model'] for b in banks}) != len(banks):
                p.error('Selection accepts one full-width bank per model')
            result, arrays = selection.run({b['model']: b for b in banks}, manifest, config)
        else:
            if not args.test_manifest or not args.test_bank:
                p.error('This protocol requires --test-manifest and --test-bank')
            test = load_manifest(args.test_manifest)
            tests = [load_bank(path, test) for path in args.test_bank]
            if args.protocol == 'transfer':
                if len(banks) != 2 or len(tests) != 2:
                    p.error('Transfer takes audio then video for both --bank and --test-bank')
                result, arrays = transfer.run(dict(zip(('audio', 'video'), banks)), manifest,
                                              dict(zip(('audio', 'video'), tests)), test, width)
            else:
                if len(banks) != 1 or len(tests) != 1:
                    p.error('Fresh scenes takes one original and one fresh bank')
                result, arrays = recoverability.fresh_scenes(banks[0], manifest, tests[0], test,
                                                              json.loads(args.supports.read_text()), width)
    result['configuration'] = {
        'manifest': args.manifest.name, 'banks': [path.name for path in args.bank],
        'test_manifest': args.test_manifest.name if args.test_manifest else None,
        'test_banks': [path.name for path in args.test_bank] if args.test_bank else None,
        'threads': args.threads,
        'versions': {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn')},
    }
    save_result(args.output, result, arrays)
    print(args.output/'results.json')


if __name__ == '__main__':
    main()
