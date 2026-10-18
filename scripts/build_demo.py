"""Rebuild the README artwork and small, static media companion."""
import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
INK, BLUE, TEAL, RUST = '#172f34', '#3865a3', '#087e77', '#b45739'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'svg.hashsalt': 'recap',
                     'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})


def overview(results):
    fig = plt.figure(figsize=(13.2, 4.25), facecolor='#f5f3ec')
    fig.text(.035, .92, 'ONE EMBEDDING. MORE THAN ONE WAY TO READ IT.', fontsize=12, weight='bold')
    width = results['objectnet_width_endpoints']['jina']
    ax = fig.add_axes([.065, .21, .245, .51], facecolor='#f5f3ec')
    for key, name, color in [('native_prompt', 'Text prompt', BLUE), ('ridge_alpha10', 'Fixed Ridge', TEAL)]:
        values = [100*width[key+'_ba_'+suffix] for suffix in ('native', 'minimum')]
        ax.plot([0, 1], values, 'o-', color=color, lw=2.5, ms=7, label=name)
        for x, y in enumerate(values):
            offset = 7 if (key == 'native_prompt') == (x == 0) else -10
            ax.annotate(f'{y:.1f}', (x, y), xytext=(0, offset), textcoords='offset points',
                        ha='center', va='bottom' if offset > 0 else 'top', color=color, fontsize=11)
    ax.set(xticks=[0, 1], xticklabels=['1,024D', '32D'], xlim=(-.18, 1.18), ylim=(0, 83), yticks=[0, 40, 80])
    ax.tick_params(length=0, labelsize=9)
    ax.grid(axis='y', alpha=.12)
    ax.legend(frameon=False, loc='upper right', fontsize=8)
    fig.text(.055, .79, '01 / COMPRESS', fontsize=10, weight='bold')
    fig.text(.055, .09, 'The winning reader changes.', fontsize=11, weight='bold')
    fig.text(.055, .035, 'Jina · ObjectNet · balanced accuracy (%)', fontsize=8, color='#56686b')

    ax = fig.add_axes([.40, .21, .235, .51], facecolor='#f5f3ec')
    score = results['natural_readers']['ESC-50']['gemma2']
    values = [100*score[k] for k in ('native_prompt_selected_inner', 'logistic_nested')]
    ax.barh([1, 0], values, height=.40, color=[BLUE, TEAL])
    for y, value in zip([1, 0], values):
        ax.text(value+2, y, f'{value:.1f}', va='center', fontsize=11, weight='bold')
    ax.set(yticks=[1, 0], yticklabels=['Prompt', 'Logistic'], xticks=[0, 50, 100], xlim=(0, 103), ylim=(-.7, 1.7))
    ax.tick_params(length=0, labelsize=9)
    ax.grid(axis='x', alpha=.12); ax.set_axisbelow(True)
    fig.text(.385, .79, '02 / CHANGE THE READER', fontsize=10, weight='bold')
    fig.text(.385, .09, f'Same audio vectors. +{values[1]-values[0]:.1f} points.', fontsize=11, weight='bold')
    fig.text(.385, .035, 'Gemma 2 · ESC-50 · balanced accuracy (%)', fontsize=8, color='#56686b')

    ax = fig.add_axes([.765, .21, .185, .51], facecolor='#f5f3ec')
    matrix = next(m['balanced_accuracy_matrix'] for m in results['ave']['models'] if m['name']=='Omni-Embed-Mini 2.3B')
    values = [100*matrix[1][1], 100*matrix[0][1]]
    ax.barh([1, 0], values, height=.40, color=[TEAL, RUST])
    for y, value in zip([1, 0], values):
        ax.text(value-3, y, f'{value:.1f}', va='center', ha='right', fontsize=11, weight='bold', color='white')
    ax.set(yticks=[1, 0], yticklabels=['Video-trained', 'Audio-trained'], xticks=[0, 50, 100], xlim=(0, 103), ylim=(-.7, 1.7))
    ax.tick_params(length=0, labelsize=9)
    ax.grid(axis='x', alpha=.12); ax.set_axisbelow(True)
    fig.text(.72, .79, '03 / CROSS A MODALITY', fontsize=10, weight='bold')
    fig.text(.72, .09, 'Shared space. Different decisions.', fontsize=11, weight='bold')
    fig.text(.72, .035, 'Omni Mini 2.3B · AVE video · balanced accuracy (%)', fontsize=8, color='#56686b')
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
    (ROOT/'docs/assets').mkdir(parents=True, exist_ok=True)
    path = ROOT/'docs/assets/overview.svg'
    fig.savefig(path, metadata={'Date': None, 'Creator': 'RECAP'})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def clean_audio(source, destination):
    # Preserve PCM samples; strip container metadata from the presentation copy.
    with wave.open(str(source), 'rb') as handle:
        params, samples = handle.getparams(), handle.readframes(handle.getnframes())
    with wave.open(str(destination), 'wb') as handle:
        handle.setparams(params)
        handle.writeframes(samples)


def waveform(path):
    with wave.open(str(path), 'rb') as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError('The selected demonstration expects mono, 16-bit PCM')
        signal = np.frombuffer(handle.readframes(handle.getnframes()), dtype='<i2') / 32768.
    chunks = np.array_split(signal, 240)
    peaks = np.array([np.abs(chunk).max() for chunk in chunks])
    bars = ''.join(f'<line x1="{i*4+2}" y1="{60-53*p:.2f}" x2="{i*4+2}" y2="{60+53*p:.2f}"/>'
                   for i, p in enumerate(peaks))
    text = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 120" role="img" '
            'aria-label="Waveform of the five-second sound; vertical height represents amplitude">'
            '<g stroke="#b7d7c9" stroke-width="2" stroke-linecap="round">'+bars+'</g></svg>')
    (ROOT/'demo/assets/waveform.svg').write_text(text+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, help='Prepared benchmark media; omit to rebuild numbers and overview only')
    args = parser.parse_args()
    results = json.loads((ROOT/'paper_results/results.json').read_text())
    examples = json.loads((ROOT/'demo/examples.json').read_text())
    overview(results)
    data = {'examples': examples, 'width': results['objectnet_width_endpoints']['jina'],
            'audio_scores': results['natural_readers']['ESC-50'],
            'transfer': next(m for m in results['ave']['models'] if m['name']=='Omni-Embed-Mini 2.3B'),
            'budget': {c['strategy']: c['score'] for c in results['selection']['ESC-50']['cells']
                       if c['cap']==512 and c['budget']==8}}
    (ROOT/'demo/data.js').write_text('window.RECAP = '+json.dumps(data, indent=2)+';\n')
    if args.data_root:
        assets = ROOT/'demo/assets'
        assets.mkdir(exist_ok=True)
        clean_audio(args.data_root/examples['audio']['source'], ROOT/'demo'/examples['audio']['asset'])
        waveform(ROOT/'demo'/examples['audio']['asset'])
        for example in examples['transfer']:
            output = ROOT/'demo'/example['asset']
            if example['target']=='audio':
                clean_audio(args.data_root/example['source']['audio'], output)
            else:
                with tempfile.TemporaryDirectory(prefix='recap_demo_') as directory:
                    for order, number in enumerate(example['source']['frame_numbers']):
                        path = args.data_root/example['source']['frame_pattern'].format(index=number)
                        with Image.open(path) as image:
                            image.convert('RGB').save(Path(directory)/f'{order:02d}.png')
                    subprocess.run(['ffmpeg','-v','error','-y','-framerate','5','-i',str(Path(directory)/'%02d.png'),
                                    '-vf','scale=640:-2','-c:v','libx264','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',
                                    '-map_metadata','-1',str(output)], check=True)
        for example in examples['images']:
            with Image.open(args.data_root/example['source']) as image:
                image = image.convert('RGB')
                image.thumbnail((720, 720))
                image.save(ROOT/'demo'/example['asset'], quality=88, optimize=True)
    print('Rebuilt README artwork and demo data'+(' / media' if args.data_root else ''))


if __name__=='__main__':
    main()
