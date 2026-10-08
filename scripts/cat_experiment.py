"""Independent tabular algorithm evaluated on the same development fold."""
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
from catboost import CatBoostClassifier

from referee.data import load_data,fixed_folds,concatenate
from scripts.experiment import report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--name",default="cat_v2")
    ap.add_argument("--train",default="data/competition/train.csv")
    ap.add_argument("--extra-train",default="data/train.csv")
    ap.add_argument("--iterations",type=int,default=1600)
    ap.add_argument("--depth",type=int,default=7)
    ap.add_argument("--turn-experiment",default="structural_v2")
    args=ap.parse_args()
    start=time.monotonic()
    data=load_data(args.train)
    folds=fixed_folds(data)
    valid=np.flatnonzero(folds==1)
    training=np.flatnonzero(folds>1)
    if args.extra_train:
        extra=load_data(args.extra_train)
        excluded=set(data["groups"][folds<2])
        selected=np.array([i for i,g in enumerate(extra["groups"]) if g not in excluded])+len(data["y"])
        training=np.concatenate([training,selected])
        data=concatenate(data,extra)
    models=[]
    for target,loss in (("y","MultiClass"),("success","Logloss")):
        model=CatBoostClassifier(iterations=args.iterations,depth=args.depth,learning_rate=.04,l2_leaf_reg=5,
            loss_function=loss,thread_count=8,random_seed=42,verbose=200,allow_writing_files=False)
        model.fit(data["X"].iloc[training],data[target][training])
        models.append(model)
    p=models[0].predict_proba(data["X"].iloc[valid]); ps=models[1].predict_proba(data["X"].iloc[valid])[:,1]
    folder=Path("outputs")/args.name
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/"validation.npz",idx=valid,label_probabilities=p,success_probabilities=ps)
    joblib.dump(models,folder/"model.joblib",compress=3)
    if (Path("outputs")/args.turn_experiment/"turns.npz").exists():
        turn=np.load(Path("outputs")/args.turn_experiment/"turns.npz")
        evidence=turn["evidence"]**2
        evidence/=evidence.sum(axis=1,keepdims=True)
        labels=(.5*p+.5*evidence).argmax(axis=1)
        ft=turn["candidates"][np.arange(len(valid)),labels]
        result=report(data,valid,labels,(ps>=.4).astype(int),ft,training)
        result.update(elapsed_seconds=time.monotonic()-start,configuration=vars(args))
        (folder/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
