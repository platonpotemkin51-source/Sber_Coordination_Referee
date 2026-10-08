"""Apply a frozen stack to a fold excluded from all of its training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
from referee.data import load_data,fixed_folds,concatenate,CODE_HASH
from referee.relations import relation_features,RELATIONS_HASH
from scripts.experiment import report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--experiment',default='stack_v9')
    ap.add_argument('--train',default='data/competition/train.csv')
    ap.add_argument('--fold',type=int,default=0)
    args=ap.parse_args()
    if args.fold not in (0,1):
        raise ValueError('The other folds were in this stack training')
    folder=ROOT/'outputs'/args.experiment
    bundle=joblib.load(folder/'model.joblib')
    if bundle['features_code_hash']!=CODE_HASH or bundle['relations_hash']!=RELATIONS_HASH:
        raise ValueError('Stack feature provenance mismatch')
    data=relation_features(load_data(args.train),args.train)
    folds=fixed_folds(data); idx=np.flatnonzero(folds==args.fold); training=np.flatnonzero(folds>1)
    p,ps=bundle['run_model'].predict(data,idx)
    candidates,evidence=bundle['turn_model'].predict(data,idx)
    ranked=bundle['ranker'].predict(data,idx,p,evidence,candidates)
    ep=evidence**2; ep/=ep.sum(1,keepdims=True)
    weight=bundle['ranker_weight']; p=(1-weight)*(.5*p+.5*ep)+weight*ranked
    labels=p.argmax(1); success=(ps>=.45).astype(int); onset=candidates[np.arange(len(idx)),labels]
    config=json.loads((folder/'metrics.json').read_text())['configuration']
    if config.get('extra_train'):
        extra=load_data(config['extra_train']); excluded=set(data['groups'][folds<2])
        training=np.concatenate([training,np.array([i for i,g in enumerate(extra['groups']) if g not in excluded])+len(data['y'])])
        metrics_data=dict(data); metrics_data['X']=data['X'][list(extra['X'].columns)]
        metrics_data['turn_X']=data['turn_X'][:,:len(extra['turn_columns'])]; metrics_data['turn_columns']=extra['turn_columns']
        data=concatenate(metrics_data,extra)
    metrics=report(data,idx,labels,success,onset,training)
    metrics.update(fold=args.fold,ranker_weight=weight,note='Frozen parameters chosen on development; additional evaluation on a previously opened fold.')
    target=folder/f'fold{args.fold}'; target.mkdir(parents=True,exist_ok=True)
    (target/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    np.savez_compressed(target/'validation.npz',idx=idx,label_probabilities=p,success_probabilities=ps,labels=labels,success=success,turns=onset,folds=folds)
    np.savez_compressed(target/'turns.npz',candidates=candidates,evidence=evidence)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
