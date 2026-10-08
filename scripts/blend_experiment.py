"""Choose a small neural/tree blend on half-development, then freeze it."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from sklearn.model_selection import train_test_split
from referee.data import load_data,fixed_folds,concatenate
from scripts.experiment import report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',default='blend_v10')
    ap.add_argument('--train',default='data/competition/train.csv')
    ap.add_argument('--tree',default='relations_v5')
    ap.add_argument('--ranked',default='stack_v9')
    ap.add_argument('--neural',default='sequence_v6')
    ap.add_argument('--parameters')
    args=ap.parse_args()
    tree_folder=ROOT/'outputs'/args.tree
    tree=np.load(tree_folder/'validation.npz'); events=np.load(tree_folder/'turns.npz')
    neural=np.load(ROOT/'outputs'/args.neural/'validation.npz')
    ranked=np.load(ROOT/'outputs'/args.ranked/'validation.npz') if args.ranked else None
    np.testing.assert_array_equal(tree['idx'],neural['idx'])
    if ranked is not None:
        np.testing.assert_array_equal(tree['idx'],ranked['idx'])
    data=load_data(args.train); folds=fixed_folds(data)
    config=json.loads((tree_folder/'metrics.json').read_text()).get('configuration',{})
    validation_fold=config.get('validation_fold',0 if '/fold0' in args.tree else 1)
    training=np.flatnonzero((folds!=0)&(folds!=validation_fold)) if validation_fold else np.flatnonzero(folds!=0)
    extra=load_data('data/train.csv'); excluded=set(data['groups'][(folds==0)|(folds==validation_fold)])
    training=np.concatenate([training,np.array([i for i,g in enumerate(extra['groups']) if g not in excluded])+len(data['y'])])
    data=concatenate(data,extra)
    idx=tree['idx']; select,check=train_test_split(np.arange(len(idx)),test_size=.5,stratify=data['y'][idx],random_state=12)
    success=(neural['success_probabilities']>=.45).astype(int)
    choices=[]
    parameters=json.loads(Path(args.parameters).read_text()) if args.parameters else None
    for source in ('tree','ranked') if ranked is not None else ('tree',):
        for weight in (.25,.5,.75):
            if parameters and (source!=parameters['source'] or weight!=parameters['neural_weight']):
                continue
            base=tree if source=='tree' else ranked
            p=(1-weight)*base['label_probabilities']+weight*neural['label_probabilities']
            labels=p.argmax(1); turns=events['candidates'][np.arange(len(idx)),labels]
            score=report(data,idx[select],labels[select],success[select],turns[select],training)
            choices.append((score['score_proxy'],source,weight,p,labels,turns))
    _,source,weight,p,labels,turns=max(choices,key=lambda x:x[0])
    metrics=report(data,idx,labels,success,turns,training)
    metrics['check']=report(data,idx[check],labels[check],success[check],turns[check],training)
    metrics['configuration']={**vars(args),'source':source,'neural_weight':weight,'success_source':'neural','success_threshold':.45,'onset_source':'tree'}
    folder=ROOT/'outputs'/args.name;folder.mkdir(parents=True,exist_ok=True)
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    (folder/'parameters.json').write_text(json.dumps({'source':source,'neural_weight':weight,'success_threshold':.45,'success_source':'neural','onset_source':'tree'},indent=2),encoding='utf8')
    np.savez_compressed(folder/'validation.npz',idx=idx,label_probabilities=p,success_probabilities=neural['success_probabilities'],labels=labels,success=success,turns=turns,folds=folds)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
