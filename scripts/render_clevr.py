"""Run inside Blender 2.78c with the public CLEVR scene generator."""
# Blender 2.78c ships Python 3.5, so this file avoids newer Python syntax.
import argparse
import json
import os
import random
import sys


def main():
    args = sys.argv[sys.argv.index('--')+1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generator', required=True, help='Public clevr-dataset-gen checkout')
    parser.add_argument('--schedule', default='splits/clevr_render_schedule.json')
    parser.add_argument('--output', default='data/clevr_fresh')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=2000)
    options = parser.parse_args(args)
    with open(options.schedule) as handle:
        schedule = json.load(handle)['rows']
    if options.start < 0 or options.count < 1 or options.start+options.count > len(schedule):
        raise ValueError('Invalid schedule range')
    generator = os.path.abspath(os.path.join(options.generator, 'image_generation'))
    output = os.path.abspath(options.output)
    sys.path.insert(0, generator)
    import render_images
    original = render_images.render_scene

    def scheduled_scene(args, **kwargs):
        row = schedule[kwargs['output_index']]
        if os.path.exists(kwargs['output_image']) or os.path.exists(kwargs['output_scene']):
            raise IOError('Refusing to overwrite a rendered scene')
        kwargs['num_objects'] = row['object_count']
        random.seed(row['generation_seed'])
        original(args, **kwargs)
    render_images.render_scene = scheduled_scene
    # The upstream renderer resolves materials relative to image_generation.
    os.chdir(generator)
    args = render_images.parser.parse_args([
        '--start_idx', str(options.start), '--num_images', str(options.count),
        '--split', 'e3b', '--filename_prefix', 'CLEVR', '--width', '320', '--height', '240',
        '--render_num_samples', '512', '--min_pixels_per_object', '200', '--use_gpu', '0',
        '--output_image_dir', os.path.join(output, 'images'),
        '--output_scene_dir', os.path.join(output, 'scenes'),
        '--output_scene_file', os.path.join(output, 'scenes_{}_{}.json'.format(options.start, options.count)),
    ])
    render_images.main(args)


if __name__ == '__main__':
    main()
