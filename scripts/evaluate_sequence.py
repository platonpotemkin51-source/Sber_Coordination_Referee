"""Evaluate a frozen sequence checkpoint on an additional fold."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from referee.data import load_data,fixed_folds,concatenate,CODE_HASH
from referee.relations import relation_features,RELATIONS_HASH
from referee.sequence import SequenceNetwork,signed_log,predict
from scripts.experiment import report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--experiment',default='sequence_v6')
    ap.add_argument('--train',default='data/competition/train.csv')
    ap.add_argument('--fold',type=int,default=0)
    args=ap.parse_args()
    torch.set_num_threads(4)
    folder=ROOT/'outputs'/args.experiment
    checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    if checkpoint['features_code_hash']!=CODE_HASH or checkpoint['relations_hash']!=RELATIONS_HASH:
        raise ValueError('Sequence feature provenance mismatch')
    if args.fold not in (0,1):
        raise ValueError('Other folds were in training and cannot be evaluated honestly with this checkpoint')
    data=relation_features(load_data(args.train),args.train)
    folds=fixed_folds(data); valid=np.flatnonzero(folds==args.fold); training=np.flatnonzero(folds>1)
    stats=checkpoint['normalization']
    run=np.clip((signed_log(data['X'][checkpoint['run_columns']].to_numpy(dtype=np.float32))-stats['run_mean'])/stats['run_std'],-8,8)
    columns=[data['turn_columns'].index(c) for c in checkpoint['turn_columns']]
    turns=np.clip((signed_log(data['turn_X'][:,columns])-stats['turn_mean'])/stats['turn_std'],-8,8)
    network=SequenceNetwork(run.shape[1],turns.shape[1],checkpoint['width'])
    network.load_state_dict(checkpoint['state'])
    p,ps,candidates=predict(network,run,turns,data['offsets'],valid)
    labels=p.argmax(1); success=(ps>=.45).astype(int); onset=candidates[np.arange(len(valid)),labels]
    config=json.loads((folder/'metrics.json').read_text())['configuration']
    if config.get('extra_train'):
        extra=load_data(config['extra_train'])
        excluded=set(data['groups'][folds<2])
        training=np.concatenate([training,np.array([i for i,g in enumerate(extra['groups']) if g not in excluded])+len(data['y'])])
        # Metrics need run features only; turn features have already been used.
        metrics_data=dict(data); metrics_data['X']=data['X'][list(extra['X'].columns)]
        metrics_data['turn_X']=data['turn_X'][:,:len(extra['turn_columns'])]
        metrics_data['turn_columns']=extra['turn_columns']
        data=concatenate(metrics_data,extra)
    metrics=report(data,valid,labels,success,onset,training)
    metrics.update(checkpoint_epoch=checkpoint['epoch'],fold=args.fold,note='Additional evaluation of a frozen checkpoint; fold 0 was already opened in earlier model development.')
    target=folder/f'fold{args.fold}'; target.mkdir(parents=True,exist_ok=True)
    (target/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    np.savez_compressed(target/'validation.npz',idx=valid,label_probabilities=p,success_probabilities=ps,labels=labels,success=success,turns=onset,candidates=candidates,folds=folds)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
