"""Apply already selected calibration to a different saved validation fold."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from referee.data import load_data, fixed_folds, concatenate
from scripts.experiment import report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--experiment', required=True)
    ap.add_argument('--calibration', required=True)
    ap.add_argument('--train', default='data/competition/train.csv')
    args=ap.parse_args()
    folder=ROOT/'outputs'/args.experiment
    config=json.loads((folder/'metrics.json').read_text())['configuration']
    calibration=json.loads(Path(args.calibration).read_text())
    predictions=np.load(folder/'validation.npz')
    events=np.load(folder/'turns.npz')
    p=predictions['label_probabilities']
    evidence=events['evidence']
    oldweight=config.get('turn_weight',0)
    if oldweight:
        old=evidence**config.get('turn_power',2); old/=old.sum(axis=1,keepdims=True)
        p=np.clip((p-oldweight*old)/(1-oldweight),0,1)
    ep=evidence**calibration['turn_power']; ep/=ep.sum(axis=1,keepdims=True)
    weight=calibration['turn_weight']
    p=((1-weight)*p+weight*ep)*np.array(calibration.get('class_weights',[1]*7))
    labels=p.argmax(axis=1)
    success=(predictions['success_probabilities']>=calibration['success_threshold']).astype(int)
    turns=events['candidates'][np.arange(len(labels)),labels]
    data=load_data(args.train); folds=fixed_folds(data)
    validation_fold=config.get('validation_fold',1)
    excluded=(folds==0)|(folds==validation_fold)
    training=np.flatnonzero(~excluded)
    if config.get('extra_train'):
        extra=load_data(config['extra_train'])
        excluded_groups=set(data['groups'][excluded])
        selected=np.array([i for i,g in enumerate(extra['groups']) if g not in excluded_groups])+len(data['y'])
        training=np.concatenate([training,selected])
        data=concatenate(data,extra)
    result=report(data,predictions['idx'],labels,success,turns,training)
    result.update(calibration_file=args.calibration, validation_fold=validation_fold, note='Parameters selected on development; this is a repeat evaluation of a previously opened fold.')
    (folder/'applied_class_calibration.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
