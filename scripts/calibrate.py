"""Select a small fusion grid on half of development; report the other half."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from sklearn.model_selection import train_test_split
from referee.data import load_data,fixed_folds
from scripts.experiment import report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment",required=True)
    ap.add_argument("--train",default="data/competition/train.csv")
    args=ap.parse_args()
    folder=Path("outputs")/args.experiment
    metrics=json.loads((folder/"metrics.json").read_text())
    config=metrics["configuration"]
    data=load_data(args.train)
    prediction=np.load(folder/"validation.npz")
    turn=np.load(folder/"turns.npz")
    idx=prediction["idx"]
    candidates,evidence=turn["candidates"],turn["evidence"]
    probabilities=prediction["label_probabilities"]
    old_weight=config.get("turn_weight",0)
    if old_weight:
        old_ep=evidence**config.get("turn_power",2)
        old_ep/=old_ep.sum(axis=1,keepdims=True)
        probabilities=np.clip((probabilities-old_weight*old_ep)/(1-old_weight),0,1)
    calibration,check=train_test_split(np.arange(len(idx)),test_size=.5,stratify=data["y"][idx],random_state=12)
    folds=fixed_folds(data)
    training=np.flatnonzero((folds!=0)&(folds!=config.get("validation_fold",1)))
    trials=[]
    for power in (1,2,4):
        ep=evidence**power
        ep/=ep.sum(axis=1,keepdims=True)
        for weight in (.35,.5,.65,.8):
            labels=((1-weight)*probabilities+weight*ep).argmax(axis=1)
            turns=candidates[np.arange(len(idx)),labels]
            for threshold in (.35,.4,.45,.5,.55):
                success=(prediction["success_probabilities"]>=threshold).astype(int)
                score=report(data,idx[calibration],labels[calibration],success[calibration],turns[calibration],training)
                trials.append((score["score_proxy"],power,weight,threshold,labels,success,turns))
    best=max(trials,key=lambda t:t[0])
    _,power,weight,threshold,labels,success,turns=best
    result={"turn_power":power,"turn_weight":weight,"success_threshold":threshold,
            "selection_score_proxy":best[0],"check":report(data,idx[check],labels[check],success[check],turns[check],training),
            "development_full":report(data,idx,labels,success,turns,training),"note":"Half-development check was not used for this grid selection. This script does not evaluate the reserved fold."}
    (folder/"calibration.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
