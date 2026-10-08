"""Fixed development fold; fold zero stays excluded until final evaluation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
import pandas as pd

from referee.data import load_data,fixed_folds,concatenate,restrict_features,LABELS
from referee.model import RunModel,TurnModel
from baseline.features import parse_run
from baseline.localize import localize
from evaluation.metrics import macro_f1,binary_f1,fault_turn_hit_at_k,composite,f1_per_class


def report(data,idx,labels,success,turns,training=None):
    true=[LABELS[x] for x in data["y"][idx]]
    pred=[LABELS[x] for x in labels]
    # Threshold is defined on training data, not on validation features.
    nonvalidation=training if training is not None else np.setdiff1d(np.arange(len(data["y"])),idx)
    length=data["X"].n_messages.iloc[idx].to_numpy()
    cutoff=float(data["X"].n_messages.iloc[nonvalidation].median())
    hard=(length>cutoff) & ((data["X"].topo_mesh.iloc[idx]+data["X"].topo_blackboard.iloc[idx]).to_numpy()>0)
    scores={"macro_f1":macro_f1(true,pred),"robustness_f1":macro_f1(np.array(true)[hard],np.array(pred)[hard]) if hard.any() else 0,
            "success_f1":binary_f1(data["success"][idx],success),"fault_turn_hit2":fault_turn_hit_at_k(true,pred,data["fault"][idx],turns)}
    scores["score_proxy"]=composite(scores)
    scores["per_class_f1"]=f1_per_class(true,pred,LABELS)
    scores["robustness_proxy_n"]=int(hard.sum())
    return scores


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",default="data/competition/train.csv")
    ap.add_argument("--name",required=True)
    ap.add_argument("--mode",choices=["baseline","struct","text","ensemble"],default="struct")
    ap.add_argument("--turn-model",action="store_true")
    ap.add_argument("--trees",type=int,default=650)
    ap.add_argument("--leaves",type=int,default=31)
    ap.add_argument("--text-weight",type=float,default=.25)
    ap.add_argument("--validation-fold",type=int,default=1)
    ap.add_argument("--extra-train")
    ap.add_argument("--turn-weight",type=float,default=0)
    ap.add_argument("--turn-power",type=float,default=2)
    ap.add_argument("--success-threshold",type=float,default=.5)
    ap.add_argument("--profile",choices=["current","v2"],default="current")
    args=ap.parse_args()
    t=time.monotonic()
    destination=Path("outputs")/args.name
    destination.mkdir(parents=True,exist_ok=True)
    data=load_data(args.train)
    folds=fixed_folds(data)
    valid=np.flatnonzero(folds==args.validation_fold)
    training=np.flatnonzero((folds!=0)&(folds!=args.validation_fold)) if args.validation_fold != 0 else np.flatnonzero(folds!=0)
    if args.extra_train:
        extra=load_data(args.extra_train)
        excluded_groups=set(data["groups"][np.flatnonzero((folds==0)|(folds==args.validation_fold))])
        extra_training=np.array([i for i,g in enumerate(extra["groups"]) if g not in excluded_groups],dtype=int)+len(data["y"])
        training=np.concatenate([training,extra_training])
        data=concatenate(data,extra)
    data=restrict_features(data,args.profile)
    print(f"train={len(training)}, valid={len(valid)}, groups={len(set(data['groups']))}",flush=True)
    model=RunModel(args.mode,args.trees,args.leaves,args.text_weight).fit(data,training)
    probabilities,p_success=model.predict(data,valid)
    labels=probabilities.argmax(axis=1)
    success=(p_success>=args.success_threshold).astype(int)
    turn_model=None
    if args.turn_model:
        print("fitting turn model",flush=True)
        turn_model=TurnModel().fit(data,training)
        candidates,evidence=turn_model.predict(data,valid)
        ep=evidence**args.turn_power
        ep/=ep.sum(axis=1,keepdims=True)
        probabilities=(1-args.turn_weight)*probabilities+args.turn_weight*ep
        labels=probabilities.argmax(axis=1)
        turns=candidates[np.arange(len(valid)),labels]
        np.savez_compressed(destination/"turns.npz",candidates=candidates,evidence=evidence)
    else:
        df=pd.read_csv(args.train)
        turns=np.array([localize(parse_run(df.iloc[i].to_dict()),LABELS[label]) for i,label in zip(valid,labels)])
    scores=report(data,valid,labels,success,turns,training)
    scores.update(elapsed_seconds=time.monotonic()-t,configuration=vars(args),train_sha256=data["source_sha256"],validation_rows=len(valid),reserved_fold=0)
    print(json.dumps(scores,indent=2),flush=True)
    (destination/"metrics.json").write_text(json.dumps(scores,indent=2),encoding="utf-8")
    np.savez_compressed(destination/"validation.npz",idx=valid,label_probabilities=probabilities,success_probabilities=p_success,labels=labels,success=success,turns=turns,folds=folds)
    joblib.dump({"run_model":model,"turn_model":turn_model},destination/"model.joblib",compress=3)


if __name__ == "__main__":
    main()
