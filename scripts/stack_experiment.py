"""Cross-fit detectors before learning which candidate is the primary fault."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold,train_test_split
from referee.data import load_data,fixed_folds,concatenate,CODE_HASH
from referee.relations import relation_features,RELATIONS_HASH
from referee.model import RunModel,TurnModel
from referee.stack import CandidateRanker
from scripts.experiment import report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',default='stack_v9')
    ap.add_argument('--train',default='data/competition/train.csv')
    ap.add_argument('--extra-train',default='data/train.csv')
    ap.add_argument('--validation-fold',type=int,default=1)
    ap.add_argument('--base-model',default='outputs/relations_v5/model.joblib')
    args=ap.parse_args()
    started=time.monotonic()
    data=relation_features(load_data(args.train),args.train)
    sources=[data['source_sha256']]
    folds=fixed_folds(data); valid=np.flatnonzero(folds==args.validation_fold)
    training=np.flatnonzero((folds!=0)&(folds!=args.validation_fold)) if args.validation_fold else np.flatnonzero(folds!=0)
    if args.extra_train:
        extra=relation_features(load_data(args.extra_train),args.extra_train)
        sources.append(extra['source_sha256'])
        excluded=set(data['groups'][(folds==0)|(folds==args.validation_fold)])
        training=np.concatenate([training,np.array([i for i,g in enumerate(extra['groups']) if g not in excluded])+len(data['y'])])
        data=concatenate(data,extra)
    folder=ROOT/'outputs'/args.name; folder.mkdir(parents=True,exist_ok=True)
    model_hash=hashlib.sha256((ROOT/'referee/model.py').read_bytes()).hexdigest()[:12]
    certificate={'sources':sources,'features_code_hash':CODE_HASH,'relations_hash':RELATIONS_HASH,'model_hash':model_hash,'training_groups_sha256':hashlib.sha256('\n'.join(data['groups'][training]).encode()).hexdigest(),'inner_seed':20261014,'inner_folds':3}
    signature=hashlib.sha256(json.dumps(certificate,sort_keys=True).encode()).hexdigest()[:16]
    oof_p=np.zeros((len(training),7)); oof_e=oof_p.copy(); oof_c=np.zeros((len(training),7),dtype=int)
    oof_success=np.zeros(len(training)); assigned=np.zeros(len(training),dtype=int)
    cv=StratifiedGroupKFold(3,shuffle=True,random_state=20261014)
    for inner,(fit,check) in enumerate(cv.split(data['X'].iloc[training],data['y'][training],data['groups'][training])):
        fitting,checking=training[fit],training[check]
        assert not (set(data['groups'][fitting]) & set(data['groups'][checking]))
        assert not (set(data['groups'][fitting]) & set(data['groups'][valid]))
        target=ROOT/'outputs/cache'/f'stack-oof-{signature}-{inner}.npz'
        if target.exists():
            cached=np.load(target)
            np.testing.assert_array_equal(cached['fitting'],fitting)
            np.testing.assert_array_equal(cached['checking'],checking)
            p,ps,candidates,evidence=(cached[k] for k in ('probabilities','success','candidates','evidence'))
        else:
            print(f'OOF {inner+1}/3: train={len(fitting)}, check={len(checking)}',flush=True)
            start=time.monotonic()
            model=RunModel(trees=850).fit(data,fitting)
            print(f'OOF {inner+1}/3: fitting onset detector',flush=True)
            turns=TurnModel().fit(data,fitting)
            p,ps=model.predict(data,checking)
            candidates,evidence=turns.predict(data,checking)
            np.savez_compressed(target,fitting=fitting,checking=checking,probabilities=p,success=ps,candidates=candidates,evidence=evidence)
            del model,turns
            print(f'OOF {inner+1}/3: complete in {time.monotonic()-start:.1f}s',flush=True)
        oof_p[check]=p; oof_e[check]=evidence; oof_c[check]=candidates; oof_success[check]=ps; assigned[check]+=1
    assert np.all(assigned==1),'Every meta-training run needs exactly one held-out prediction'
    print('Fitting candidate ranker on cross-fitted predictions',flush=True)
    ranker=CandidateRanker().fit(data,training,oof_p,oof_e,oof_c)
    base=joblib.load(args.base_model)
    if base.get('features_code_hash') != CODE_HASH or base.get('relations_hash') != RELATIONS_HASH:
        raise ValueError('Base model feature provenance mismatch')
    p,ps=base['run_model'].predict(data,valid)
    candidates,evidence=base['turn_model'].predict(data,valid)
    ranked=ranker.predict(data,valid,p,evidence,candidates)
    ep=evidence**2; ep/=ep.sum(1,keepdims=True); fused=.5*p+.5*ep
    select,check=train_test_split(np.arange(len(valid)),test_size=.5,stratify=data['y'][valid],random_state=12)
    success=(ps>=.45).astype(int)
    trials=[]
    for weight in (0,.25,.5,.75,1):
        probabilities=(1-weight)*fused+weight*ranked
        labels=probabilities.argmax(1); turns=candidates[np.arange(len(valid)),labels]
        selection=report(data,valid[select],labels[select],success[select],turns[select],training)
        trials.append((selection['score_proxy'],weight,probabilities,labels,turns))
    _,weight,probabilities,labels,turns=max(trials,key=lambda x:x[0])
    metrics=report(data,valid,labels,success,turns,training)
    metrics['check']=report(data,valid[check],labels[check],success[check],turns[check],training)
    pure_labels=ranked.argmax(1)
    metrics['pure_ranker']=report(data,valid,pure_labels,success,candidates[np.arange(len(valid)),pure_labels],training)
    metrics.update(configuration={**vars(args),'ranker_weight':weight},elapsed_seconds=time.monotonic()-started,oof_certificate=certificate)
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    np.savez_compressed(folder/'validation.npz',idx=valid,label_probabilities=probabilities,success_probabilities=ps,labels=labels,success=success,turns=turns,folds=folds,ranker_probabilities=ranked)
    np.savez_compressed(folder/'turns.npz',candidates=candidates,evidence=evidence)
    joblib.dump({**base,'ranker':ranker,'ranker_weight':weight,'success_threshold':.45,'oof_certificate':certificate},folder/'model.joblib',compress=3)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    main()
