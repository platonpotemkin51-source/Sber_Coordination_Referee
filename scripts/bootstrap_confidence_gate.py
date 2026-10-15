"""Paired bootstrap for confidence-gated vs fixed word+char on check-half."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import v15_mix_audit as audit

ROOT=Path(__file__).resolve().parents[1]
HANDOFF=ROOT/"handoff/v15/assembled"
EXPORT=HANDOFF/"evaluation_export"
DEST=ROOT/"outputs/char-tfidf-v15-experiment"

def main():
    dev=audit.load_slice("development_fold1",None,HANDOFF/"outputs/pair_selector_v15/validation.npz")
    ids=dev["membership"].official_row_index.to_numpy()
    cm=pd.read_csv(EXPORT/"check_half_membership.csv"); ct=pd.read_csv(EXPORT/"check_half_truth.csv")
    cids=cm.official_row_index.to_numpy(); positions={int(x):i for i,x in enumerate(ids)}
    ix=np.asarray([positions[int(x)] for x in cids]); n=len(ix)
    check={"truth":ct,"membership":cm,"base":dev["base"][ix],"selector":dev["selector"][ix],
           "success":dev["success"][ix],"candidates":dev["candidates"][ix]}
    predictions=np.load(DEST/"confidence_gated_predictions.npz")
    dynamic=predictions["text_minus_base_threshold_dev"][ix]
    fixed=np.load(DEST/"predictions.npz")["word_char_blend_dev"][ix]
    rng=np.random.default_rng(20261013); deltas=[]
    for _ in range(2000):
        sample=rng.integers(0,n,n)
        s={k:(v.iloc[sample].reset_index(drop=True) if k in ("truth","membership") else v[sample]) for k,v in check.items()}
        deltas.append(audit.score_probability(s,dynamic[sample])["score_proxy"]-audit.score_probability(s,fixed[sample])["score_proxy"])
    d=np.asarray(deltas)
    result={"reps":len(d),"median_delta":float(np.median(d)),"mean_delta":float(np.mean(d)),
            "ci_2_5_97_5":np.quantile(d,[.025,.975]).tolist(),"positive_fraction":float(np.mean(d>0))}
    (DEST/"confidence_gate_bootstrap.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
