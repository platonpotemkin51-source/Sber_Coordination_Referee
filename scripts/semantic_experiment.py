"""Development evaluation of the local semantic encoder."""
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
from referee.data import load_data,fixed_folds,concatenate,CODE_HASH
from referee.semantic import semantic_features
from referee.model import RunModel,TurnModel
from scripts.experiment import report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--name",default="semantic_v4")
    ap.add_argument("--train",default="data/competition/train.csv")
    ap.add_argument("--extra-train",default="data/train.csv")
    ap.add_argument("--encoder",default="weights/encoder")
    ap.add_argument("--trees",type=int,default=850)
    args=ap.parse_args()
    start=time.monotonic()
    data=semantic_features(load_data(args.train),args.encoder)
    folds=fixed_folds(data)
    valid=np.flatnonzero(folds==1)
    training=np.flatnonzero(folds>1)
    if args.extra_train:
        extra=semantic_features(load_data(args.extra_train),args.encoder)
        excluded=set(data["groups"][folds<2])
        selected=np.array([i for i,g in enumerate(extra["groups"]) if g not in excluded])+len(data["y"])
        training=np.concatenate([training,selected])
        data=concatenate(data,extra)
    model=RunModel(trees=args.trees).fit(data,training)
    print("Fitting semantic turn model",flush=True)
    turn_model=TurnModel().fit(data,training)
    p,ps=model.predict(data,valid)
    candidates,evidence=turn_model.predict(data,valid)
    ep=evidence**2; ep/=ep.sum(axis=1,keepdims=True)
    p=.5*p+.5*ep
    labels=p.argmax(axis=1); success=(ps>=.45).astype(int)
    turns=candidates[np.arange(len(valid)),labels]
    metrics=report(data,valid,labels,success,turns,training)
    metrics.update(configuration=vars(args),elapsed_seconds=time.monotonic()-start,features_code_hash=CODE_HASH)
    folder=Path("outputs")/args.name; folder.mkdir(parents=True,exist_ok=True)
    (folder/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    np.savez_compressed(folder/"validation.npz",idx=valid,label_probabilities=p,success_probabilities=ps,labels=labels,success=success,turns=turns,folds=folds)
    np.savez_compressed(folder/"turns.npz",candidates=candidates,evidence=evidence)
    joblib.dump({"run_model":model,"turn_model":turn_model,"features_code_hash":CODE_HASH},folder/"model.joblib",compress=3)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__ == "__main__":
    main()
