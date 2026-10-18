"""Rebuild paper tables and figures from numerical evaluation outputs."""
import argparse
import json
import wave
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

BLUE, TEAL, ORANGE, INK, MUTED = '#3967af', '#087e77', '#c5633d', '#182b3c', '#667483'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})

LABELS = {'gemma2': 'Gemma 2', 'qwen2b': 'Qwen 2B', 'qwen8b': 'Qwen 8B',
          'jina': 'Jina', 'ovis': 'Ovis', 'omni23': 'Omni Mini 2.3B', 'omni09': 'Omni Mini 0.9B'}
READERS = ['native_prompt_selected_inner', 'ridge_nested', 'logistic_nested']


def score(rows, reader, field='balanced_accuracy'):
    values = [float(r[field]) for r in rows if r['reader'] == reader]
    if len(values) != 3:
        raise ValueError(f'Expected three seeds for {reader}; got {len(values)}')
    return mean(values)


def save(fig, name):
    for extension in ('pdf', 'svg', 'png'):
        options = {'dpi': 220} if extension == 'png' else {}
        fig.savefig(OUTPUT/'figures'/f'{name}.{extension}', bbox_inches='tight', facecolor='white', **options)
    plt.close(fig)


def tex_table(name, spec, header, rows):
    text = '\\begin{tabular}{'+spec+'}\n\\toprule\n'
    text += ' & '.join(header)+' \\\\\n\\midrule\n'
    text += ''.join(' & '.join(map(str, row))+' \\\\\n' for row in rows)
    text += '\\bottomrule\n\\end{tabular}\n'
    (OUTPUT/'tables'/f'{name}.tex').write_text(text)


def seed_range(v, k):
    vals = [100*r[k] for r in v['seed_gaps']]
    return f'[{min(vals):+.2f}, {max(vals):+.2f}]'


def make_tables(data):
    for task, models in data['natural_readers'].items():
        tex_table(task.lower().replace('-', '')+'_readers', 'lrrr', ['Checkpoint', 'Prompt', 'Ridge', 'Logistic'],
                  [[LABELS[m], *[f'{100*v[r]:.1f}' for r in READERS]] for m, v in models.items()])
    controls = []
    for task, models in data['natural_readers'].items():
        for model in models:
            rows = [r for r in data['controls'] if r['task']==task and r['model']==model]
            values = []
            for reader in ('prototype', 'ridge_alpha10', 'logistic_C1', 'native_bias_calibrated', 'ridge_shuffled'):
                if reader=='native_bias_calibrated' and any(r['reader']=='native_bias_calibrated_template0' for r in rows):
                    reader='native_bias_calibrated_template0'
                values.append(f'{100*score(rows, reader):.1f}' if any(r['reader']==reader for r in rows) else '--')
            controls.append([task, LABELS[model], *values])
    tex_table('natural_controls', 'llrrrrr', ['Task','Checkpoint','Prototype','Ridge 10','Logistic 1','Bias-cal.','Shuffled'], controls)
    width=data['objectnet_width_endpoints']
    tex_table('width_endpoints','lrrrr', ['Checkpoint / width','Prompt','Ridge','Prompt','Ridge'],
              [[LABELS[m]+f" ({v['native_dimension']}/{v['minimum_dimension']})",
                *[f'{100*v[k]:.1f}' for k in ['native_prompt_ba_native','ridge_alpha10_ba_native',
                                             'native_prompt_ba_minimum','ridge_alpha10_ba_minimum']]] for m,v in width.items()])
    tex_table('width_seed_ranges','lrrr',['Checkpoint','Full gap','Minimum gap','Interaction'],
              [[LABELS[m],*[seed_range(v,k) for k in ['gap_native','gap_minimum','gap_interaction_min_minus_native']]] for m,v in width.items()])
    effects=data['image_reader_effects']
    tex_table('task_effects','lrr',['Checkpoint','Country211','ObjectNet'],
              [[LABELS[m],*[f"{float(next(r['mean_delta_balanced_accuracy_pp'] for r in effects if r['model']==m and r['task']==t and r['reader']=='ridge_nested')):+.2f}" for t in ['Country211','ObjectNet']]] for m in ['jina','ovis','omni23']])
    selection_rows=[]
    for task, record in data['selection'].items():
        for cap in (512,2048):
            for budget in (1,2,4,8):
                vals=[next(c['score'] for c in record['cells'] if (c['cap'],c['budget'],c['strategy'])==(cap,budget,s))
                      for s in ('best_feasible_native_only','selector')]
                selection_rows.append([task,str(record['classes']),str(cap),str(budget),*[f'{v*100:.2f}' for v in vals],f'{100*(vals[1]-vals[0]):+.2f}'])
    tex_table('selection_all','lrrrrrr',['Task','Classes','Bytes','Groups/class','Native only','Select reader','Change'],selection_rows)
    if data.get('clevr'):
        clevr=data['clevr']
        crow=[]
        for m,d in [('gemma2',768),('gemma2',128),('qwen2b',2048),('qwen2b',64)]:
            row=[LABELS[m],str(d)]
            for modality in ['image','text']:
                rs=[r for r in clevr if r['model']==m and int(r['dimension'])==d and r['modality']==modality]
                row.extend([f"{float(next(r['macro_auroc'] for r in rs if r['reader']=='native:present')):.3f}",f"{score(rs,'ridge_nested','macro_auroc'):.3f}"])
            crow.append(row)
        tex_table('clevr_readers','lrrrrr',['Checkpoint','Width','Image prompt','Image Ridge','Text prompt','Text Ridge'],crow)
    if data.get('temporal'):
        temporal=data['temporal']
        temp=[]
        for m,d in [('gemma2',768),('gemma2',128),('qwen2b',2048),('qwen2b',64)]:
            r=next(r for r in temporal if r['model']==m and int(r['dimension'])==d and r['strategy']=='paired_both_views'
                   and r['method']=='ridge_alpha10' and r['metric']=='both_directions_correct')
            temp.append([LABELS[m],str(d),f"{100*float(r['estimate']):.1f}",f"[{100*float(r['ci_low_95']):.1f}, {100*float(r['ci_high_95']):.1f}]"])
        tex_table('temporal_joint','lrrr',['Checkpoint','Width','Both correct','95\% interval'],temp)


def make_plots(data):
    natural=data['natural_readers']
    AUDIO_PATHS=natural['ESC-50']
    width=data['objectnet_width_endpoints']
    ave=data['ave']
    selection=data['selection']
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    models = list(AUDIO_PATHS)
    y = np.arange(len(models))[::-1]
    for i, model in enumerate(models):
        a = natural['ESC-50'][model]['native_prompt_selected_inner']*100
        b = natural['ESC-50'][model]['logistic_nested']*100
        ax.plot([a,b],[y[i],y[i]],color='#cdd6df',lw=4,zorder=1)
        ax.scatter([a,b],[y[i],y[i]],color=[BLUE,TEAL],s=48,zorder=2)
        ax.text(a-1.8,y[i],f'{a:.1f}',ha='right',va='center',fontsize=8,color=BLUE)
        ax.text(b+1.8,y[i],f'{b:.1f}',ha='left',va='center',fontsize=8,color=TEAL)
    ax.set(yticks=y,yticklabels=[LABELS[m] for m in models],xlim=(10,104),ylim=(-.6,4.65),
           xlabel='Held-out balanced accuracy (%)')
    ax.scatter([],[],color=BLUE,label='Selected text prompt')
    ax.scatter([],[],color=TEAL,label='Trained logistic reader')
    ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(-.01,1.01),ncol=1,fontsize=7.5)
    ax.tick_params(axis='y',labelsize=7.5)
    ax.grid(axis='x',alpha=.15); ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.tight_layout();save(fig,'audio_readers')
    fig,axs=plt.subplots(1,4,figsize=(7.0,2.35),sharey=True)
    for ax,(m,v) in zip(axs,width.items()):
        for prefix,label,color in [('native_prompt','Selected prompt',BLUE),('ridge_alpha10','Fixed Ridge',TEAL)]:
            vals=[100*v[prefix+'_ba_native'],100*v[prefix+'_ba_minimum']]
            ax.plot([0,1],vals,'o-',color=color,lw=2,markersize=4.5,label=label)
            for x,t in enumerate(vals):
                suffix='_ba_native' if x==0 else '_ba_minimum'
                other='ridge_alpha10' if prefix=='native_prompt' else 'native_prompt'
                top = t > 100*v[other+suffix]
                offset = 4.2 if top else -7.5
                ax.text(x,t+offset,f'{t:.1f}',ha='center',fontsize=7.5,fontweight='bold' if m=='jina' else 'normal',color=color)
        ax.set_title(LABELS[m],fontsize=9,fontweight='bold',pad=6)
        ax.set(xticks=[0,1],xticklabels=[f"{v['native_dimension']:,}D",f"{v['minimum_dimension']}D"],
               xlim=(-.28,1.28),ylim=(0,92))
        ax.grid(axis='y',alpha=.16);ax.tick_params(axis='x',length=0,labelsize=8)
    axs[0].set_ylabel('Balanced accuracy (%)',fontsize=8.5)
    axs[0].legend(frameon=False,fontsize=7.2,loc='upper right')
    fig.tight_layout();save(fig,'width_changes_reader')
    fig,axs=plt.subplots(1,4,figsize=(7,2.32))
    for ax,m in zip(axs,ave['models']):
        mat=100*np.array(m['balanced_accuracy_matrix'])
        ax.imshow(mat,vmin=0,vmax=100,cmap='YlGnBu')
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',fontsize=12,fontweight='bold',color='white' if mat[i,j]>55 else INK)
        name=m['name'].replace('Jina v5 Omni Small','Jina').replace('Omni-Embed-Mini','Omni Mini').replace('Ovis Omni-Embed 3B','Ovis')
        ax.set_title(name,fontsize=8.5,fontweight='bold')
        ax.set(xticks=[0,1],xticklabels=['Audio','Video'],yticks=[0,1],yticklabels=['Audio','Video'],xlabel='Test input')
        ax.tick_params(length=0)
        for sp in ax.spines.values():sp.set_visible(False)
    axs[0].set_ylabel('Reader trained on')
    fig.tight_layout();save(fig,'reader_transfer')

    # Compare readers on identical target inputs. These are arithmetic contrasts
    # of preserved official-split matrix cells, not new fitted/scored experiments.
    penalties=[]
    for m in ave['models']:
        mat=m['balanced_accuracy_matrix']
        penalties.append({'checkpoint':m['name'],
                          'audio_target_pp':100*(mat[0][0]-mat[1][0]),
                          'video_target_pp':100*(mat[1][1]-mat[0][1])})
    fig,ax=plt.subplots(figsize=(3.45,2.55))
    names=['Jina','Omni Mini 2.3B','Ovis','Omni Mini 0.9B']
    y=np.arange(4)[::-1]
    for offset,k,label,color in [(.15,'audio_target_pp','Audio test inputs',BLUE),(-.15,'video_target_pp','Video test inputs',TEAL)]:
        vals=[m[k] for m in penalties]
        ax.barh(y+offset,vals,height=.24,color=color,label=label)
        for yi,v in enumerate(vals):ax.text(v+1,y[yi]+offset,f'{v:.1f}',va='center',fontsize=7.5,color=color)
    ax.set(yticks=y,yticklabels=names,xlim=(0,66),xlabel='Cross-modal reader penalty $P_{a\\to b}$ (pp)')
    ax.tick_params(axis='y',length=0,labelsize=7.5)
    ax.spines['left'].set_visible(False)
    ax.legend(frameon=False,fontsize=7.2,loc='lower left',bbox_to_anchor=(-.04,1.01))
    ax.grid(axis='x',alpha=.13)
    ax.set_axisbelow(True)
    fig.tight_layout();save(fig,'target_fixed_transfer')
    fig,axs=plt.subplots(1,2,figsize=(6.9,2.55))
    for ax,(task,data) in zip(axs,selection.items()):
        for strategy,color,label in [('best_feasible_native_only',BLUE,'Native-only selection (model + width + prompt)'),('selector',TEAL,'Joint selection (model + width + reader)')]:
            vals=[next(c['score']*100 for c in data['cells'] if (c['cap'],c['budget'],c['strategy'])==(512,b,strategy)) for b in [1,2,4,8]]
            ax.plot([1,2,4,8],vals,'o-',color=color,label=label,lw=2,markersize=4.5)
            ax.text(8.18,vals[-1],f'{vals[-1]:.1f}%',color=color,fontsize=8,fontweight='bold',va='center')
        ax.set_title(f"{task} ({data['classes']} classes, 512B cap)",fontsize=9,fontweight='bold',pad=6)
        ax.set(xlabel='Labeled source groups per class',xticks=[1,2,4,8],ylim=(42,100),xlim=(.6,9.55))
        ax.grid(alpha=.15)
    axs[0].set_ylabel('Held-out balanced accuracy (%)',fontsize=8.5)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.05),ncol=2,fontsize=8)
    fig.tight_layout();save(fig,'budget_selection')


def make_media(data, root):
    media=json.loads((Path(__file__).resolve().parents[1]/'configs/qualitative.json').read_text())
    width=data['objectnet_width_endpoints']
    fig=plt.figure(figsize=(10.6, 2.75))
    panel_xs = [.01, .345, .68]
    panel_titles = [
        '(a) Axis 1: Compression Stability',
        '(b) Axis 2: Cross-Modal Portability',
        '(c) Axis 3: Latent Label Recoverability'
    ]
    panel_subtitles = [
        'ObjectNet (969 images) · Jina v5 Omni',
        'AVE test video (402 clips) · Omni Mini 2.3B',
        'ESC-50 audio · Jina v5 Omni (1,024D)'
    ]
    for x, title, sub in zip(panel_xs, panel_titles, panel_subtitles):
        fig.add_artist(FancyBboxPatch((x, .03), .31, .94, boxstyle='round,pad=0.006,rounding_size=0.015',
                                      facecolor='#f8fafb', edgecolor='#d5dfe7', lw=1.0, transform=fig.transFigure, zorder=-1))
        fig.text(x+.016, .915, title, fontsize=9.5, fontweight='bold', color=INK, va='top')
        fig.text(x+.016, .835, sub, fontsize=7.8, color=MUTED, va='top')

    # Panel (a): Axis 1 — Compression rank reversal slopegraph
    ax_a=fig.add_axes([.060, .24, .235, .52])
    v=width['jina']
    for k,color,label in [('native_prompt',BLUE,'Selected prompt'),('ridge_alpha10',TEAL,'Fixed Ridge')]:
        vals=[100*v[k+'_ba_native'],100*v[k+'_ba_minimum']]
        ax_a.plot([0,1],vals,'o-',color=color,lw=2.2,markersize=5,label=label)
    ax_a.text(0, v['native_prompt_ba_native']*100+4.5, f"{v['native_prompt_ba_native']*100:.1f}%", ha='center', va='bottom', fontsize=8.2, fontweight='bold', color=BLUE)
    ax_a.text(0, v['ridge_alpha10_ba_native']*100-5.5, f"{v['ridge_alpha10_ba_native']*100:.1f}%", ha='center', va='top', fontsize=8.2, color=TEAL)
    ax_a.text(1, v['ridge_alpha10_ba_minimum']*100+4.5, f"{v['ridge_alpha10_ba_minimum']*100:.1f}%", ha='center', va='bottom', fontsize=8.2, fontweight='bold', color=TEAL)
    ax_a.text(1, v['native_prompt_ba_minimum']*100-5.5, f"{v['native_prompt_ba_minimum']*100:.1f}%", ha='center', va='top', fontsize=8.2, color=BLUE)
    ax_a.set(xticks=[0,1],xticklabels=['1,024D (Full)','32D (Min.)'],ylim=(-6,88),xlim=(-.28,1.28),yticks=[0,40,80])
    ax_a.set_ylabel('Balanced acc. (%)', fontsize=8, labelpad=2)
    ax_a.tick_params(labelsize=7.8)
    ax_a.grid(axis='y',alpha=.18)
    ax_a.set_facecolor('white')
    ax_a.legend(frameon=False,fontsize=7.5,loc='upper right')
    full_gap=100*(v['ridge_alpha10_ba_native']-v['native_prompt_ba_native'])
    min_gap=100*(v['ridge_alpha10_ba_minimum']-v['native_prompt_ba_minimum'])
    fig.text(.165, .075, rf'Rank inversion: $\Delta_{{\mathrm{{Ridge}}}}$ shifts ${full_gap:+.1f} \to {min_gap:+.1f}$ pp', fontsize=7.8, fontweight='bold', color=INK, ha='center')

    # Panel (b): Axis 2 — Cross-modal transfer on fixed target video inputs
    fp=root/media['horse']
    ax_b=fig.add_axes([.363, .44, .135, .34])
    ax_b.imshow(Image.open(fp));ax_b.set_xticks([]);ax_b.set_yticks([])
    for sp in ax_b.spines.values():
        sp.set_visible(True);sp.set_color('#ccd6e0');sp.set_linewidth(.8)
    fig.text(.507, .73, 'Target input:\nAVE test video\n("Horse")', fontsize=7.8, color=INK, va='top', fontweight='bold')
    fig.text(.507, .53, 'Same frozen\nvideo vectors,\ntwo readers:', fontsize=7.5, color=MUTED, va='top')

    fig.add_artist(FancyBboxPatch((.361, .245), .278, .125, boxstyle='round,pad=0.004,rounding_size=0.01',
                                  facecolor='#eaf7f5', edgecolor='#9ed3cc', lw=.8, transform=fig.transFigure))
    fig.text(.371, .308, 'Video-trained reader:', fontsize=8, color=INK, va='center')
    matrix=next(m['balanced_accuracy_matrix'] for m in data['ave']['models'] if m['name']=='Omni-Embed-Mini 2.3B')
    fig.text(.627, .308, f'{100*matrix[1][1]:.1f}% BA (✓)', fontsize=8.2, fontweight='bold', color=TEAL, ha='right', va='center')

    fig.add_artist(FancyBboxPatch((.361, .095), .278, .125, boxstyle='round,pad=0.004,rounding_size=0.01',
                                  facecolor='#fdf2ee', edgecolor='#e8bfb3', lw=.8, transform=fig.transFigure))
    fig.text(.371, .158, 'Audio-trained reader:', fontsize=8, color=INK, va='center')
    fig.text(.627, .158, f'{100*matrix[0][1]:.1f}% BA (✗)', fontsize=8.2, fontweight='bold', color=ORANGE, ha='right', va='center')

    # Panel (c): Axis 3 — Audio waveform + reader comparison on identical vector
    with wave.open(str(root/media['audio']),'rb') as w:
        audio=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)
        rate=w.getframerate()
    ax_c=fig.add_axes([.698, .48, .274, .28])
    step=max(1,len(audio)//3000)
    ax_c.plot(np.arange(0,len(audio),step)/rate,audio[::step]/32768,color=INK,lw=.6)
    ax_c.set_xlim(0,len(audio)/rate);ax_c.set_ylim(-1.05,1.05)
    ax_c.set_xticks([0, 2.5, 5.0]);ax_c.set_xticklabels(['0s', '2.5s', '5.0s'], fontsize=7)
    ax_c.set_yticks([])
    ax_c.spines['left'].set_visible(False);ax_c.spines['bottom'].set_color('#ccd6e0')
    ax_c.set_facecolor('white')
    ax_c.text(.98, .86, f"True label: \"{media['audio_truth']}\"", transform=ax_c.transAxes, ha='right', va='top', fontsize=7.5, fontweight='bold', color=INK)

    fig.add_artist(FancyBboxPatch((.696, .245), .278, .125, boxstyle='round,pad=0.004,rounding_size=0.01',
                                  facecolor='#eef3fa', edgecolor='#c5d5ea', lw=.8, transform=fig.transFigure))
    fig.text(.706, .308, 'Zero-shot text prompt:', fontsize=8, color=MUTED, va='center')
    fig.text(.962, .308, f"\"{media['audio_prompt_prediction']}\" (✗)", fontsize=8.2, fontweight='bold', color=BLUE, ha='right', va='center')

    fig.add_artist(FancyBboxPatch((.696, .095), .278, .125, boxstyle='round,pad=0.004,rounding_size=0.01',
                                  facecolor='#eaf7f5', edgecolor='#9ed3cc', lw=.8, transform=fig.transFigure))
    fig.text(.706, .158, 'Learned linear reader:', fontsize=8, color=INK, va='center')
    fig.text(.962, .158, f"\"{media['audio_reader_prediction']}\" (✓)", fontsize=8.2, fontweight='bold', color=TEAL, ha='right', va='center')

    save(fig,'teaser')

    # Paired image successes and regressions: uniform aspect framing for clean alignment.
    chosen=media['images']
    fig,axs=plt.subplots(1,4,figsize=(7.0,2.35))
    for ax,item in zip(axs,chosen):
        p=root/item['image']
        img = Image.open(p).convert('RGB')
        w_img, h_img = img.size
        side = max(w_img, h_img)
        canvas = Image.new('RGB', (side, side), (244, 247, 249))
        canvas.paste(img, ((side - w_img) // 2, (side - h_img) // 2))
        ax.imshow(canvas);ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True);sp.set_color('#dce3ea');sp.set_linewidth(.8)
        ax.set_title(f"{item['task']} · {item['true_label']}",fontsize=8,fontweight='bold',pad=5)
        ax.text(.5,-.08,f"Prompt: {item['native_predictions'][0]}\nRidge: {item['ridge_predictions'][0]}",
                transform=ax.transAxes,ha='center',va='top',fontsize=7.2)
    fig.tight_layout();save(fig,'image_reader_examples')



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('paper_results/results.json'))
    parser.add_argument('--output',type=Path,default=Path('outputs/paper'))
    parser.add_argument('--media-root',type=Path,help='Also render the teaser and qualitative image panel from local benchmark media')
    args=parser.parse_args()
    global OUTPUT
    OUTPUT=args.output
    OUTPUT.mkdir(parents=True,exist_ok=False)
    (OUTPUT/'figures').mkdir()
    (OUTPUT/'tables').mkdir()
    data=json.loads(args.results.read_text())
    make_tables(data)
    make_plots(data)
    if args.media_root:
        make_media(data,args.media_root)
    else:
        print('Rendered five quantitative figures; use --media-root for the teaser and image examples.')
    print(OUTPUT)


if __name__=='__main__':
    main()
