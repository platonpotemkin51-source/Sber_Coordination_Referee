"""Offline inference or CPU training under the DSWorks run contract."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from referee.data import load_data,restrict_features,LABELS,CODE_HASH
from referee.model import RunModel,TurnModel


def infer(bundle,data,weights_directory=None,neural_device="auto"):
    indexes=np.arange(len(data["run_id"]))
    p,ps=bundle["run_model"].predict(data,indexes)
    run_probabilities=p.copy()
    if bundle.get("global_context"):
        from referee.context import global_context_features
        data=global_context_features(data,bundle["global_context"])
    candidates,evidence=bundle["turn_model"].predict(data,indexes)
    ep=evidence**bundle.get("turn_power",2)
    ep/=np.maximum(ep.sum(axis=1,keepdims=True),1e-12)
    w=bundle.get("turn_weight",.5)
    p=(1-w)*p+w*ep
    if bundle.get("ranker") is not None:
        ranked=bundle["ranker"].predict(data,indexes,run_probabilities,evidence,candidates)
        rank_weight=bundle.get("ranker_weight",.75)
        p=(1-rank_weight)*p+rank_weight*ranked
    if bundle.get("pair_selector") is not None:
        from referee.pair_selector import PAIR_HASH
        if bundle.get("pair_hash") != PAIR_HASH:
            raise RuntimeError("Pair selector and feature implementation do not match")
        if w != .5 or bundle.get("turn_power",2) != 2:
            raise ValueError("Pair selector expects the validated .5/2 detector fusion")
        paired=bundle["pair_selector"].predict(data,indexes,run_probabilities,evidence,candidates)
        pair_weight=bundle.get("pair_selector_weight",.5)
        p=(1-pair_weight)*p+pair_weight*paired
    if bundle.get("neural_success"):
        from referee.neural_success import success_probabilities
        directory=weights_directory or Path(__file__).resolve().parent/"weights"
        ps=success_probabilities(data,bundle["neural_success"],directory,neural_device)
    p*=np.array(bundle.get("class_weights",[1.]*7))
    y=p.argmax(axis=1)
    return pd.DataFrame({"run_id":data["run_id"],"label":[LABELS[c] for c in y],
        "success":(ps>=bundle.get("success_threshold",.4)).astype(int),
        "fault_turn":candidates[np.arange(len(y)),y]})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",required=True)
    ap.add_argument("--test",required=True)
    ap.add_argument("--output",default="predictions.csv")
    ap.add_argument("--weights",default=str(Path(__file__).resolve().parent/"weights"/"model.joblib"))
    ap.add_argument("--neural-device",choices=["auto","cpu","cuda"],default="auto")
    args=ap.parse_args()
    test_header=pd.read_csv(args.test,nrows=0)
    required={"goal","agents","messages","artifacts","shared_state","topology"}
    if not required.issubset(test_header.columns):
        test=pd.read_csv(args.test)
        pd.DataFrame({"run_id":test.run_id,"label":"clean","success":1,"fault_turn":-1}).to_csv(args.output,index=False)
        return
    if Path(args.weights).exists():
        bundle=joblib.load(args.weights)
        if bundle.get("features_code_hash") != CODE_HASH:
            raise RuntimeError("Model weights and feature extractor do not match")
    else:
        train=restrict_features(load_data(args.train),"v2")
        idx=np.arange(len(train["run_id"]))
        bundle={"run_model":RunModel(trees=850).fit(train,idx),"turn_model":TurnModel().fit(train,idx),"features_code_hash":CODE_HASH}
    test=load_data(args.test)
    if bundle.get("relations_hash"):
        from referee.relations import relation_features,RELATIONS_HASH
        if bundle["relations_hash"] != RELATIONS_HASH:
            raise RuntimeError("Model weights and relation extractor do not match")
        test=relation_features(test,args.test)
    infer(bundle,test,Path(args.weights).resolve().parent,args.neural_device).to_csv(args.output,index=False)
    print(f"Wrote {args.output}: {len(test['run_id'])} rows",flush=True)


if __name__ == "__main__":
    main()
