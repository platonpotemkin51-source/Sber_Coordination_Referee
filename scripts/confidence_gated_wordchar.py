"""Gate lexical probabilities by per-run text confidence/advantage."""
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

def norm(q):
    q=np.maximum(np.asarray(q,dtype=np.float64),1e-12)
    return q/q.sum(1,keepdims=True)

def main():
    dev=audit.load_slice("development_fold1",None,HANDOFF/"outputs/pair_selector_v15/validation.npz")
    fold=audit.load_slice("fold0",None,HANDOFF/"outputs/pair_selector_v15/fold0/validation.npz")
    pred=np.load(DEST/"predictions.npz")
    base,text=pred["baseline_dev"],pred["word_char_dev"]
    base0,text0=pred["baseline_fold0"],pred["word_char_fold0"]
    static=np.asarray([.1,.1,.5,.3,0.,0.,.2])
    ids=dev["membership"].official_row_index.to_numpy()
    ct=pd.read_csv(EXPORT/"check_half_truth.csv"); cm=pd.read_csv(EXPORT/"check_half_membership.csv")
    cids=cm.official_row_index.to_numpy(); pos={int(x):i for i,x in enumerate(ids)}
    cp=np.asarray([pos[int(x)] for x in cids]); sp=np.asarray([i for i,x in enumerate(ids) if int(x) not in set(cids)])
    selection={k:(v.iloc[sp].reset_index(drop=True) if k in ("truth","membership") else v[sp]) for k,v in dev.items()}
    check={"truth":ct,"membership":cm,**{k:v[cp] for k,v in dev.items() if k not in ("truth","membership")}}

    reports={}
    prediction_arrays={}
    specs={"probability_threshold":np.asarray([0.,.05,.1,.15,.2,.25,.3,.4,.5,.6]),
           "text_minus_base_threshold":np.asarray([-1.,-.3,-.2,-.1,-.05,0.,.05,.1,.2,.3])}
    for name,choices in specs.items():
        gates=np.full(7,choices[0])
        def weights(base_p,text_p,thresholds):
            if name=="probability_threshold":
                active=(text_p>=thresholds[None,:])
            else:
                active=((text_p-base_p)>=thresholds[None,:])
            return static[None,:]*active
        for _ in range(2):
            for cls in range(7):
                options=[]
                for value in choices:
                    trial=gates.copy(); trial[cls]=value
                    w=weights(base,text,trial)
                    q=norm((1-w)*base+w*text)
                    options.append((audit.score_probability(selection,q[sp])["score_proxy"],value))
                gates[cls]=max(options)[1]
        wd=weights(base,text,gates); w0=weights(base0,text0,gates)
        qd=norm((1-wd)*base+wd*text); q0=norm((1-w0)*base0+w0*text0)
        prediction_arrays[name+"_dev"]=qd
        prediction_arrays[name+"_fold0"]=q0
        reports[name]={"thresholds":gates.tolist(),
            "selection":audit.score_probability(selection,qd[sp]),
            "development":audit.score_probability(dev,qd),
            "check":audit.score_probability(check,qd[cp]),
            "fold0":audit.score_probability(fold,q0)}
    result={"static_gate":static.tolist(),"static":{"check":audit.score_probability(check,pred["word_char_blend_dev"][cp]),
        "fold0":audit.score_probability(fold,pred["word_char_blend_fold0"])},"dynamic":reports}
    (DEST/"confidence_gated_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez_compressed(DEST/"confidence_gated_predictions.npz",**prediction_arrays)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
