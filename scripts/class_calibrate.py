"""Small class-bias calibration on half-development, with a separate check."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from sklearn.model_selection import train_test_split
from referee.data import load_data,fixed_folds,LABELS
from scripts.experiment import report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment",required=True)
    ap.add_argument("--train",default="data/competition/train.csv")
    args=ap.parse_args()
    folder=Path("outputs")/args.experiment
    config=json.loads((folder/"metrics.json").read_text())["configuration"]
    calibration=json.loads((folder/"calibration.json").read_text())
    data=load_data(args.train)
    pred=np.load(folder/"validation.npz"); turn=np.load(folder/"turns.npz")
    idx=pred["idx"]; p=pred["label_probabilities"]; e=turn["evidence"]
    oldweight=config.get("turn_weight",0)
    if oldweight:
        old=e**config.get("turn_power",2); old/=old.sum(axis=1,keepdims=True)
        p=np.clip((p-oldweight*old)/(1-oldweight),0,1)
    e=e**calibration["turn_power"]; e/=e.sum(axis=1,keepdims=True)
    weight=calibration["turn_weight"]
    p=(1-weight)*p+weight*e
    select,check=train_test_split(np.arange(len(idx)),test_size=.5,stratify=data["y"][idx],random_state=12)
    folds=fixed_folds(data); training=np.flatnonzero(folds>1)
    success=(pred["success_probabilities"]>=calibration["success_threshold"]).astype(int)
    def measure(weights,subset):
        labels=(p*weights).argmax(axis=1)
        turns=turn["candidates"][np.arange(len(idx)),labels]
        return report(data,idx[subset],labels[subset],success[subset],turns[subset],training)
    weights=np.ones(7)
    for _ in range(2):
        for c in range(7):
            options=[]
            for multiplier in (.75,.9,1,1.1,1.25,1.5):
                trial=weights.copy(); trial[c]=multiplier
                options.append((measure(trial,select)["score_proxy"],trial))
            weights=max(options,key=lambda x:x[0])[1]
    result={**calibration,"class_weights":weights.tolist(),"selection":measure(weights,select),"check":measure(weights,check),"development_full":measure(weights,np.arange(len(idx))),"classes":LABELS}
    (folder/"class_calibration.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
