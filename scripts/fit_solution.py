"""Train reproducible offline weights on the supplied training files."""
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

from referee.data import load_data,concatenate,restrict_features,CODE_HASH
from referee.model import RunModel,TurnModel


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",default="data/competition/train.csv")
    ap.add_argument("--extra-train")
    ap.add_argument("--output",default="weights/model.joblib")
    ap.add_argument("--trees",type=int,default=850)
    ap.add_argument("--leaves",type=int,default=31)
    ap.add_argument("--turn-weight",type=float,default=.5)
    ap.add_argument("--turn-power",type=float,default=2)
    ap.add_argument("--success-threshold",type=float,default=.4)
    ap.add_argument("--profile",choices=["v2","current","v5"],default="v2")
    args=ap.parse_args()
    start=time.monotonic()
    data=load_data(args.train)
    sources=[data["source_sha256"]]
    relations_hash=None
    if args.profile == "v5":
        from referee.relations import relation_features,RELATIONS_HASH
        data=relation_features(data,args.train)
        relations_hash=RELATIONS_HASH
    if args.extra_train:
        extra=load_data(args.extra_train)
        sources.append(extra["source_sha256"])
        if args.profile == "v5":
            extra=relation_features(extra,args.extra_train)
        # Remove repeated run contents across sources, not repeated identifiers.
        if set(extra["groups"]) & set(data["groups"]):
            raise ValueError("Training sources contain duplicate run signatures; deduplicate first")
        data=concatenate(data,extra)
    data=restrict_features(data,"current" if args.profile == "v5" else args.profile)
    indexes=np.arange(len(data["y"]))
    model=RunModel("struct",args.trees,args.leaves).fit(data,indexes)
    print("Fitting fault onset classifier",flush=True)
    tm=TurnModel().fit(data,indexes)
    bundle={"run_model":model,"turn_model":tm,"features_code_hash":CODE_HASH,
            "turn_weight":args.turn_weight,"turn_power":args.turn_power,"success_threshold":args.success_threshold,
            "training_sources":sources,"configuration":vars(args)}
    if relations_hash:
        bundle["relations_hash"]=relations_hash
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,target,compress=3)
    metadata={"rows":len(data["y"]),"elapsed_seconds":time.monotonic()-start,"sources":sources,"configuration":vars(args),"features_code_hash":CODE_HASH}
    if relations_hash:
        metadata["relations_hash"]=relations_hash
    target.with_suffix(".json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(metadata,indent=2),flush=True)


if __name__ == "__main__":
    main()
