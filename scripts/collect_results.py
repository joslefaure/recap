"""Convert complete RECAP runs into the numerical input for make_figures.py."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

NAMES = {'prompt': 'native_prompt_selected_inner', 'ridge': 'ridge_nested', 'logistic': 'logistic_nested'}


def collect(root):
    docs = [json.loads(p.read_text()) for p in sorted(Path(root).glob('**/results.json'))]
    if not docs:
        raise ValueError('No protocol results found')
    data = {'natural_readers': {}, 'controls': [], 'objectnet_width_endpoints': {},
            'image_reader_effects': [], 'ave': {'models': []}, 'selection': {}, 'clevr': [], 'temporal': []}
    natural, endpoint, selection = defaultdict(list), {}, {}
    seen = set()
    for doc in docs:
        protocol = doc['protocol']
        identity = (protocol, doc.get('model'), doc.get('width'), doc.get('task'), doc.get('modality'))
        if protocol in ('recoverability', 'compression'):
            sample = doc['rows'][0]
            identity = (protocol, sample['model'], sample['task'], sample.get('modality'),
                        None if protocol == 'compression' else sample['width'])
        if identity in seen:
            raise ValueError(f'Duplicate run: {identity}')
        seen.add(identity)
        if protocol == 'recoverability':
            for r in doc['rows']:
                model, task = r['model'], r['task']
                if task == 'CLEVR':
                    reader = 'native:present' if r['reader'] == 'prompt' else NAMES.get(r['reader'],r['reader'])
                    if reader == 'native:present' and any(a['model']==model and a['dimension']==r['width'] and
                         a['modality']==r['modality'] and a['reader']==reader for a in data['clevr']):
                        continue
                    data['clevr'].append({'model': model, 'dimension': r['width'], 'modality': r['modality'],
                        'seed': 'fixed' if reader=='native:present' else r['seed'], 'reader': reader, 'macro_auroc': r['score']})
                elif r['reader'] in NAMES:
                    natural[task, model, r['reader']].append(r['score'])
                elif r['reader'] in ('prototype', 'ridge10'):
                    data['controls'].append({'task':task,'model':model,'reader': 'ridge_alpha10' if r['reader']=='ridge10' else r['reader'],
                                             'seed':r['seed'],'balanced_accuracy':r['score']})
        elif protocol == 'compression':
            rows=doc['rows'];m=rows[0]['model']
            if rows[0]['task'] != 'ObjectNet':
                continue
            widths=sorted({r['width'] for r in rows})
            values={'minimum_dimension':widths[0],'native_dimension':widths[-1]}
            for w,suffix in ((widths[0],'minimum'),(widths[-1],'native')):
                for r,key in [('prompt','native_prompt'),('ridge10','ridge_alpha10')]:
                    values[f'{key}_ba_{suffix}']=mean(a['score'] for a in rows if a['width']==w and a['reader']==r)
            values['seed_gaps']=[{'gap_native':e['full_gap'],'gap_minimum':e['minimum_gap'],
                                  'gap_interaction_min_minus_native':e['swing']} for e in doc['effects']]
            values['gap_interaction_min_minus_native']=mean(e['swing'] for e in doc['effects'])
            endpoint[m]=values
        elif protocol == 'transfer':
            names={'jina':'Jina v5 Omni Small','ovis':'Ovis Omni-Embed 3B','omni23':'Omni-Embed-Mini 2.3B','omni09':'Omni-Embed-Mini 0.9B'}
            if doc['model'] in names:
                data['ave']['models'].append({'key':doc['model'],'name':names[doc['model']],
                      'dimension':doc['width'],'balanced_accuracy_matrix':doc['matrix']})
        elif protocol == 'selection':
            task=doc['task'];cells=defaultdict(list)
            for r in doc['rows']:
                cells[r['cap'],r['groups_per_class'],r['strategy']].append(r['score'])
            values=[]
            for (cap,budget,strategy),scores in sorted(cells.items()):
                if len(scores) != 45:
                    raise ValueError('Selection cell needs 45 held-out evaluations')
                values.append({'cap':cap,'budget':budget,'strategy':'selector' if strategy=='joint' else 'best_feasible_native_only',
                               'score':mean(scores)})
            selection[task]={'classes':len(doc['classes']),'cells':values}
        elif protocol == 'temporal':
            interval=doc['joint']
            data['temporal'].append({'model':doc['model'],'dimension':doc['width'],'strategy':'paired_both_views',
                 'method':'ridge_alpha10','metric':'both_directions_correct','estimate':interval['estimate'],
                 'ci_low_95':interval['ci_low'],'ci_high_95':interval['ci_high']})
    for (task,model,reader),scores in natural.items():
        if len(scores) != 3:
            raise ValueError('Natural reader needs three split seeds at a single width')
        data['natural_readers'].setdefault(task,{}).setdefault(model,{})[NAMES[reader]]=mean(scores)
    for task in ('Country211','ObjectNet'):
        for model in ('jina','ovis','omni23'):
            v=data['natural_readers'].get(task,{}).get(model,{})
            if v:
                data['image_reader_effects'].append({'task':task,'model':model,'reader':'ridge_nested',
                    'mean_delta_balanced_accuracy_pp':100*(v['ridge_nested']-v['native_prompt_selected_inner'])})
    data['natural_readers']={t:v for t,v in data['natural_readers'].items() if t in ('ESC-50','HMDB51')}
    # Keep panel order identical to the paper.
    data['natural_readers']={t:{m:v[m] for m in ('gemma2','qwen2b','jina','ovis','omni23','omni09','qwen8b') if m in v}
                             for t,v in data['natural_readers'].items()}
    data['objectnet_width_endpoints']={m:endpoint[m] for m in ('jina','omni09','omni23','qwen8b') if m in endpoint}
    rank={'Jina v5 Omni Small':0,'Omni-Embed-Mini 2.3B':1,'Ovis Omni-Embed 3B':2,'Omni-Embed-Mini 0.9B':3}
    data['ave']['models'].sort(key=lambda v:rank[v['name']])
    data['selection']={t:selection[t] for t in ('ESC-50','HMDB51') if t in selection}
    required={'ESC-50':5,'HMDB51':7}
    for task,n in required.items():
        if len(data['natural_readers'].get(task,{})) != n or task not in data['selection']:
            raise ValueError(f'Missing main-paper runs for {task}')
    if len(endpoint)!=4 or len(data['ave']['models'])!=4 or len(data['image_reader_effects'])!=6:
        raise ValueError('Missing compression, transfer, or image recoverability runs')
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    data=collect(args.runs)
    with args.output.open('x') as handle:
        json.dump(data,handle,indent=2,allow_nan=False)
        handle.write('\n')


if __name__=='__main__':
    main()
