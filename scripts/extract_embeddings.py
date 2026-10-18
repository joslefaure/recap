"""Extract one frozen checkpoint into a portable, ordered vector bank."""
import argparse
import json
import tempfile
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
import numpy as np
from PIL import Image
from recap.data import load_manifest
from recap.models import Encoder, MODELS, CONFIGS, truncate


def scene_text(scene):
    sentences = [f"Object {i+1} is a {o['size']} {o['color']} {o['material']} {o['shape']}."
                 for i, o in enumerate(scene['objects'])]
    for a, b in combinations(range(len(scene['objects'])), 2):
        for rel, opposite in (('left', 'right'), ('front', 'behind')):
            relations = scene['relationships']
            direction = rel if a in relations[rel][b] else opposite if a in relations[opposite][b] else None
            if direction is None:
                continue
            direction = 'in front of' if direction == 'front' else f'{direction} of' if direction in ('left', 'right') else direction
            sentences.append(f'Object {a+1} is {direction} object {b+1}.')
    return ' '.join(sentences)


def video_frames(row, root, scratch):
    if 'frame_pattern' in row:
        paths = [str(root / row['frame_pattern'].format(index=i)) for i in row['frame_numbers']]
        metadata = {'fps': row['fps'], 'total_num_frames': row['total_frames'], 'frames_indices': row['frame_indices']}
    else:
        import av
        with av.open(str(root/row['video']), metadata_errors='replace') as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            frames = list(container.decode(video=0))
        if not frames:
            raise ValueError(f"No frames: {row['id']}")
        indices = np.rint(np.linspace(0, len(frames)-1, min(len(frames), row['num_frames']))).astype(int)
        paths = []
        for i, index in enumerate(indices):
            path = Path(scratch)/f'frame_{i:02d}.png'
            Image.fromarray(frames[index].to_ndarray(format='rgb24')).save(path)
            paths.append(str(path))
        metadata = {'fps': fps, 'total_num_frames': len(frames), 'frames_indices': indices.tolist()}
    for p in paths:
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    return paths, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--cache-dir', type=Path, default=Path('embeddings'))
    parser.add_argument('--modality', choices=('text', 'image', 'audio', 'video'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--runtime-dir', type=Path)
    parser.add_argument('--vision-model', type=Path)
    parser.add_argument('--image-processor', type=Path)
    parser.add_argument('--width', type=int, help='Default: full width. Normalize after supported prefix truncation.')
    parser.add_argument('--reverse', action='store_true', help='Also encode the exact reversed frame sequence.')
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    modality = args.modality or manifest['modality']
    if modality not in MODELS[args.model]['modalities']:
        parser.error('This model has no native pathway for the requested modality')
    width = args.width or max(MODELS[args.model]['widths'])
    if width not in MODELS[args.model]['widths']:
        parser.error('Unsupported embedding width')
    if args.reverse and modality != 'video':
        parser.error('--reverse requires video')
    cache_name = manifest['task'].lower() + ('_'+manifest['split'] if 'split' in manifest else '')
    output = args.cache_dir / f"{cache_name}_{args.model}_{modality}_d{width}.npz"
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    encoder = Encoder(args.model, args.checkpoint, device=args.device, runtime_dir=args.runtime_dir,
                      vision_model=args.vision_model, image_processor=args.image_processor)
    scenes = {}
    if modality == 'text' and 'scenes' in manifest:
        scenes = {s['image_filename']: s for s in json.loads((args.data_root/manifest['scenes']).read_text())['scenes']}
    x, reverse = [], []
    for i, row in enumerate(manifest['items']):
        with tempfile.TemporaryDirectory(prefix='recap_frames_') as scratch:
            metadata = None
            if modality == 'video':
                value, metadata = video_frames(row, args.data_root, scratch)
            elif modality == 'text':
                value = row.get('text') or scene_text(scenes[row['id']])
            else:
                value = str(args.data_root/row[modality])
            x.append(encoder.encode(value, modality, metadata=metadata, instruction=manifest.get('instruction', '')))
            if args.reverse:
                reverse.append(encoder.encode(value[::-1], modality, metadata=metadata, instruction=manifest.get('instruction', '')))
        print(f"{i+1}/{len(manifest['items'])}", flush=True)
    arrays = {'x': truncate(np.stack(x), args.model, width), 'ids': manifest['ids'],
              'model': np.asarray(args.model), 'modality': np.asarray(modality)}
    if manifest['task'] not in ('AVE', 'CATER'):
        templates = json.loads((CONFIGS/'prompts.json').read_text())
        family = 'clevr' if manifest['task'].startswith('CLEVR') else modality
        if family not in templates:
            raise ValueError('No prompt family for this task')
        prompts = []
        for template in templates[family]:
            prompts.append([encoder.encode(template.format(label=label.replace('_', ' ') if manifest['task']=='HMDB51' else label),
                                           'text', role='query', instruction=manifest.get('instruction', '')) for label in manifest['labels']])
        arrays['prompts'] = truncate(np.asarray(prompts), args.model, width)
    if reverse:
        arrays['reverse'] = truncate(np.stack(reverse), args.model, width)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    metadata = {'model': args.model, **MODELS[args.model], 'width': width, 'task': manifest['task'],
                'modality': modality, 'batch_size': 1, 'seed': None, 'normalized_dtype': 'float32',
                'versions': {p: version(p) for p in ('numpy', 'torch', 'transformers', 'sentence-transformers')}}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
