"""Check whether token-CNN outputs complement the stronger word+char model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import v15_mix_audit as audit

ROOT=Path(__file__).resolve().parents[1]
HANDOFF=ROOT/"handoff/v15/assembled"
EXPORT=HANDOFF/"evaluation_export"
DEST=ROOT/"outputs/cnn-wordchar-fusion"

def norm(q):
    q=np.maximum(np.asarray(q,dtype=np.float64),1e-12)
    return q/q.sum(1,keepdims=True)

def main():
    dev=audit.load_slice("development_fold1",None,HANDOFF/"outputs/pair_selector_v15/validation.npz")
    fold=audit.load_slice("fold0",None,HANDOFF/"outputs/pair_selector_v15/fold0/validation.npz")
    words=np.load(ROOT/"outputs/char-tfidf-v15-experiment/predictions.npz")
    cnn=np.load(ROOT/"outputs/token-cnn-v15/predictions.npz")
    base,base0=words["word_char_blend_dev"],words["word_char_blend_fold0"]
    text,text0=cnn["dev"],cnn["fold0"]
    ids=dev["membership"].official_row_index.to_numpy()
    ct=pd.read_csv(EXPORT/"check_half_truth.csv"); cm=pd.read_csv(EXPORT/"check_half_membership.csv")
    cids=cm.official_row_index.to_numpy(); pos={int(x):i for i,x in enumerate(ids)}
    cp=np.asarray([pos[int(x)] for x in cids]); sp=np.asarray([i for i,x in enumerate(ids) if int(x) not in set(cids)])
    selection={k:(v.iloc[sp].reset_index(drop=True) if k in ("truth","membership") else v[sp]) for k,v in dev.items()}
    check={"truth":ct,"membership":cm,**{k:v[cp] for k,v in dev.items() if k not in ("truth","membership")}}
    gates=np.zeros(7); choices=(0.,.1,.2,.3,.4,.5,.75,1.)
    for _ in range(2):
        for cls in range(7):
            options=[]
            for value in choices:
                trial=gates.copy(); trial[cls]=value
                q=norm((1-trial[None,:])*base+trial[None,:]*text)
                options.append((audit.score_probability(selection,q[sp])["score_proxy"],value))
            gates[cls]=max(options)[1]
    qd=norm((1-gates[None,:])*base+gates[None,:]*text)
    q0=norm((1-gates[None,:])*base0+gates[None,:]*text0)
    result={"gates_by_class":gates.tolist(),
        "word_char":{"selection":audit.score_probability(selection,base[sp]),"check":audit.score_probability(check,base[cp]),"fold0":audit.score_probability(fold,base0)},
        "cnn":{"selection":audit.score_probability(selection,text[sp]),"check":audit.score_probability(check,text[cp]),"fold0":audit.score_probability(fold,text0)},
        "fused":{"selection":audit.score_probability(selection,qd[sp]),"check":audit.score_probability(check,qd[cp]),"development":audit.score_probability(dev,qd),"fold0":audit.score_probability(fold,q0)}}
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez_compressed(DEST/"predictions.npz",dev=qd,fold0=q0,gates=gates)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
