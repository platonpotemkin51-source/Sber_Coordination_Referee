"""Joint classification, outcome and localization on run-level splits."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from referee.data import load_data,fixed_folds,concatenate,CODE_HASH
from referee.relations import relation_features,RELATIONS_HASH
from referee.sequence import SequenceNetwork,normalize_training,batch_arrays,predict
from scripts.experiment import report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',default='sequence_v6')
    ap.add_argument('--train',default='data/competition/train.csv')
    ap.add_argument('--extra-train',default='data/train.csv')
    ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--width',type=int,default=96)
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=ap.parse_args()
    start=time.monotonic()
    torch.set_num_threads(4); torch.manual_seed(42)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in this Python environment')
    device=torch.device(args.device)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    rng=np.random.default_rng(42)
    data=relation_features(load_data(args.train),args.train)
    folds=fixed_folds(data); valid=np.flatnonzero(folds==1); training=np.flatnonzero(folds>1)
    if args.extra_train:
        extra=relation_features(load_data(args.extra_train),args.extra_train)
        excluded=set(data['groups'][folds<2])
        selected=np.array([i for i,g in enumerate(extra['groups']) if g not in excluded])+len(data['y'])
        training=np.concatenate([training,selected]); data=concatenate(data,extra)
    select,check=train_test_split(valid,test_size=.5,stratify=data['y'][valid],random_state=12)
    run,turns,normalization=normalize_training(data,training)
    network=SequenceNetwork(run.shape[1],turns.shape[1],args.width).to(device)
    optimizer=torch.optim.AdamW(network.parameters(),lr=4e-4,weight_decay=.02)
    folder=ROOT/'outputs'/args.name; folder.mkdir(parents=True,exist_ok=True)
    best=-1; history=[]
    for epoch in range(1,args.epochs+1):
        epoch_start=time.monotonic(); network.train(); order=rng.permutation(training); losses=[]
        for start_batch in range(0,len(order),args.batch_size):
            selected=order[start_batch:start_batch+args.batch_size]
            batch=tuple(x.to(device) for x in batch_arrays(run,turns,data['offsets'],selected))
            selected_y=data['y'][selected].astype(np.int64)
            y=torch.from_numpy(selected_y).to(device)
            success=torch.from_numpy(data['success'][selected].astype(np.float32)).to(device)
            labels,ps,onset=network(*batch)
            loss=torch.nn.functional.cross_entropy(labels,y)+.5*torch.nn.functional.binary_cross_entropy_with_logits(ps,success)
            faulty=y>0
            if faulty.any():
                local=onset[faulty,:,y[faulty]-1]
                target=torch.from_numpy(data['fault'][selected][selected_y>0].astype(np.int64)).to(device)
                loss=loss+.3*torch.nn.functional.cross_entropy(local,target)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),2)
            optimizer.step(); losses.append(float(loss.detach()))
        p,ps,candidates=predict(network,run,turns,data['offsets'],select)
        labels=p.argmax(1); suc=(ps>=.45).astype(int); local=candidates[np.arange(len(select)),labels]
        metrics=report(data,select,labels,suc,local,training)
        log={'epoch':epoch,'loss':float(np.mean(losses)),'seconds':round(time.monotonic()-epoch_start,2),'selection_score_proxy':metrics['score_proxy']}
        history.append(log); print(json.dumps(log),flush=True)
        if metrics['score_proxy']>best:
            best=metrics['score_proxy']
            torch.save({'state':network.state_dict(),'normalization':normalization,'run_columns':list(data['X'].columns),'turn_columns':data['turn_columns'],'width':args.width,'epoch':epoch,'features_code_hash':CODE_HASH,'relations_hash':RELATIONS_HASH},folder/'model.pt')
    checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    network.load_state_dict(checkpoint['state'])
    p,ps,candidates=predict(network,run,turns,data['offsets'],valid)
    labels=p.argmax(1); success=(ps>=.45).astype(int); local=candidates[np.arange(len(valid)),labels]
    metrics=report(data,valid,labels,success,local,training)
    check_positions=np.flatnonzero(np.isin(valid,check))
    metrics['check']=report(data,valid[check_positions],labels[check_positions],success[check_positions],local[check_positions],training)
    metrics.update(configuration=vars(args),elapsed_seconds=time.monotonic()-start,selected_epoch=checkpoint['epoch'],history=history,features_code_hash=CODE_HASH,relations_hash=RELATIONS_HASH)
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    np.savez_compressed(folder/'validation.npz',idx=valid,label_probabilities=p,success_probabilities=ps,labels=labels,success=success,turns=local,folds=folds,candidates=candidates)
    print(json.dumps({k:v for k,v in metrics.items() if k!='history'},indent=2),flush=True)


if __name__=='__main__':
    main()
