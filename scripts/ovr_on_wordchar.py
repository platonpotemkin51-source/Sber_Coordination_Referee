"""See if selected binary experts complement the word+char multiclass model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import v15_mix_audit as audit

ROOT=Path(__file__).resolve().parents[1]
HANDOFF=ROOT/"handoff/v15/assembled"
EXPORT=HANDOFF/"evaluation_export"
DEST=ROOT/"outputs/wordchar-ovr-experts"
CLASSES={"deadlock":audit.LABELS.index("deadlock"),"goal_drift":audit.LABELS.index("goal_drift")}

def norm(q):
    q=np.maximum(np.asarray(q,dtype=np.float64),1e-12)
    return q/q.sum(1,keepdims=True)

def inject(q,p,cls,a):
    target=(1-a)*q[:,cls]+a*p
    out=q.copy(); out*=((1-target)/np.maximum(1-q[:,cls],1e-12))[:,None]; out[:,cls]=target
    return norm(out)

def main():
    dev=audit.load_slice("development_fold1",None,HANDOFF/"outputs/pair_selector_v15/validation.npz")
    fold=audit.load_slice("fold0",None,HANDOFF/"outputs/pair_selector_v15/fold0/validation.npz")
    p=np.load(DEST/"predictions.npz")
    charpred=np.load(ROOT/"outputs/char-tfidf-v15-experiment/predictions.npz")
    base,base0=charpred["word_char_blend_dev"],charpred["word_char_blend_fold0"]
    ids=dev["membership"].official_row_index.to_numpy()
    checktruth=pd.read_csv(EXPORT/"check_half_truth.csv"); checkmem=pd.read_csv(EXPORT/"check_half_membership.csv")
    cids=checkmem.official_row_index.to_numpy(); pos={int(x):i for i,x in enumerate(ids)}
    cp=np.asarray([pos[int(x)] for x in cids]); sp=np.asarray([i for i,x in enumerate(ids) if int(x) not in set(cids)])
    selection={k:(v.iloc[sp].reset_index(drop=True) if k in ("truth","membership") else v[sp]) for k,v in dev.items()}
    check={"truth":checktruth,"membership":checkmem,**{k:v[cp] for k,v in dev.items() if k not in ("truth","membership")}}
    reports={"word_char_base":{"check":audit.score_probability(check,base[cp]),"fold0":audit.score_probability(fold,base0)}}
    for name,cls in CLASSES.items():
        opts=[]
        for alpha in (0.,.05,.1,.15,.2,.3,.4,.5,.75,1.):
            q=inject(base,p[f"{name}_prob_dev"],cls,alpha)
            q0=inject(base0,p[f"{name}_prob_fold0"],cls,alpha)
            opts.append({"alpha":alpha,"selection":audit.score_probability(selection,q[sp]),
                         "check":audit.score_probability(check,q[cp]),"fold0":audit.score_probability(fold,q0)})
        reports[name+"_alpha_sweep"]=opts
        best=max(opts,key=lambda x:x["selection"]["score_proxy"])
        base=inject(base,p[f"{name}_prob_dev"],cls,best["alpha"])
        base0=inject(base0,p[f"{name}_prob_fold0"],cls,best["alpha"])
        reports[name+"_selected"]=best
    reports["combined_selected"]={"check":audit.score_probability(check,base[cp]),"fold0":audit.score_probability(fold,base0)}
    (DEST/"ovr_on_wordchar.json").write_text(json.dumps(reports,indent=2),encoding="utf-8")
    print(json.dumps(reports,indent=2),flush=True)

if __name__=="__main__": main()
