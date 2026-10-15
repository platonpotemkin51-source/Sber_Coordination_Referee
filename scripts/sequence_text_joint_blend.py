"""Jointly tune sequence and word+char class gates over the frozen v15/v9 base."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import v15_mix_audit as audit

ROOT=Path(__file__).resolve().parents[1]
HANDOFF=ROOT/"handoff/v15/assembled"
EXPORT=HANDOFF/"evaluation_export"
OUT=HANDOFF/"outputs"
DEST=ROOT/"outputs/sequence-text-joint-blend"

def norm(q):
    q=np.maximum(np.asarray(q,dtype=np.float64),1e-12)
    return q/q.sum(1,keepdims=True)

def main():
    dev=audit.load_slice("development_fold1",None,OUT/"pair_selector_v15/validation.npz")
    fold=audit.load_slice("fold0",None,OUT/"pair_selector_v15/fold0/validation.npz")
    char=np.load(ROOT/"outputs/char-tfidf-v15-experiment/predictions.npz")
    text,text0=char["word_char_dev"],char["word_char_fold0"]
    seq=np.load(OUT/"sequence_v6/validation.npz")["label_probabilities"]
    seq0=np.load(OUT/"sequence_v6/fold0/validation.npz")["label_probabilities"]
    p15=np.load(OUT/"pair_selector_v15/validation.npz")["label_probabilities"]
    p150=np.load(OUT/"pair_selector_v15/fold0/validation.npz")["label_probabilities"]
    p9=np.load(OUT/"stack_v9/validation.npz")["label_probabilities"]
    p90=np.load(OUT/"stack_v9/fold0/validation.npz")["label_probabilities"]
    v9g=np.asarray([.25,.5,.75,1.,1.,1.,.5])
    base=norm((1-v9g)*p15+v9g*p9); base0=norm((1-v9g)*p150+v9g*p90)

    ids=dev["membership"].official_row_index.to_numpy()
    ctruth=pd.read_csv(EXPORT/"check_half_truth.csv"); cmem=pd.read_csv(EXPORT/"check_half_membership.csv")
    cids=cmem.official_row_index.to_numpy(); pos={int(x):i for i,x in enumerate(ids)}
    cp=np.asarray([pos[int(x)] for x in cids]); sp=np.asarray([i for i,x in enumerate(ids) if int(x) not in set(cids)])
    selection={k:(v.iloc[sp].reset_index(drop=True) if k in ("truth","membership") else v[sp]) for k,v in dev.items()}
    check={"truth":ctruth,"membership":cmem,**{k:v[cp] for k,v in dev.items() if k not in ("truth","membership")}}

    choices=(0.,.1,.2,.3,.4,.5,.6,.75,1.)
    pairs=[(s,t) for s in choices for t in choices if s+t<=1+1e-9]
    gates=np.zeros((7,2)); gates[:,0]=np.asarray([.5,0,1,.5,.25,.5,.5])
    def mix(b,s,t): return norm((1-s-t)*b+s*seq+t*text)
    def mix0(b,s,t): return norm((1-s-t)*b+s*seq0+t*text0)
    for _ in range(2):
        for cls in range(7):
            opts=[]
            for sg,tg in pairs:
                trial=gates.copy(); trial[cls]=[sg,tg]
                q=mix(base,trial[:,0][None,:],trial[:,1][None,:])
                opts.append((audit.score_probability(selection,q[sp])["score_proxy"],sg,tg))
            _,gates[cls,0],gates[cls,1]=max(opts)
    qd=mix(base,gates[:,0][None,:],gates[:,1][None,:])
    q0=mix0(base0,gates[:,0][None,:],gates[:,1][None,:])
    baseline_seq=norm((1-np.asarray([.5,0,1,.5,.25,.5,.5]))*base+np.asarray([.5,0,1,.5,.25,.5,.5])*seq)
    baseline_text=char["word_char_blend_dev"]
    result={"gates_sequence_text":gates.tolist(),
        "baseline_sequence":{"selection":audit.score_probability(selection,baseline_seq[sp]),"check":audit.score_probability(check,baseline_seq[cp]),"fold0":audit.score_probability(fold, norm((1-np.asarray([.5,0,1,.5,.25,.5,.5]))*base0+np.asarray([.5,0,1,.5,.25,.5,.5])*seq0))},
        "baseline_wordchar":{"selection":audit.score_probability(selection,baseline_text[sp]),"check":audit.score_probability(check,baseline_text[cp]),"fold0":audit.score_probability(fold,char["word_char_blend_fold0"])},
        "joint":{"selection":audit.score_probability(selection,qd[sp]),"development":audit.score_probability(dev,qd),"check":audit.score_probability(check,qd[cp]),"fold0":audit.score_probability(fold,q0)}}
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez_compressed(DEST/"predictions.npz",dev=qd,fold0=q0,gates=gates,base=base,base0=base0)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
