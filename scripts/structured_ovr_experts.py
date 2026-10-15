"""Train binary structured-feature experts for deadlock and goal_drift."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT=Path(__file__).resolve().parents[1]
HANDOFF=ROOT/"handoff/v15/assembled"
sys.path.insert(0,str(HANDOFF)); sys.path.insert(0,str(ROOT/"scripts"))
from referee.data import concatenate,load_data
from referee.relations import relation_features
import v15_mix_audit as audit

EXPORT=HANDOFF/"evaluation_export"
OUTPUTS=HANDOFF/"outputs"
DEST=ROOT/"outputs/structured-ovr-experts"
TARGETS={"deadlock":audit.LABELS.index("deadlock"),"goal_drift":audit.LABELS.index("goal_drift")}

def norm(q):
    q=np.maximum(np.asarray(q,dtype=np.float64),1e-12)
    return q/q.sum(1,keepdims=True)

def inject(q,p,cls,a):
    target=(1-a)*q[:,cls]+a*p
    out=q.copy(); out*=((1-target)/np.maximum(1-q[:,cls],1e-12))[:,None]; out[:,cls]=target
    return norm(out)

def main():
    cache="/private/tmp/v15_oof_feature_cache"
    paths=[HANDOFF/"data/competition/train.csv",HANDOFF/"data/train.csv"]
    data=concatenate(*[relation_features(load_data(path,cache=cache),path,cache=cache) for path in paths])
    population=pd.read_csv(EXPORT/"population_and_truth.csv").set_index("global_index")
    train=population.index[population.meta_and_neural_training.to_numpy(bool)].to_numpy(dtype=int)
    xtrain=data["X"].iloc[train]; y=data["y"][train]
    dev=audit.load_slice("development_fold1",None,OUTPUTS/"pair_selector_v15/validation.npz")
    fold=audit.load_slice("fold0",None,OUTPUTS/"pair_selector_v15/fold0/validation.npz")
    did=dev["membership"].official_row_index.to_numpy(); fid=fold["membership"].official_row_index.to_numpy()
    xdev=data["X"].iloc[did]; xf=data["X"].iloc[fid]
    cp=pd.read_csv(EXPORT/"check_half_membership.csv").official_row_index.to_numpy()
    ct=pd.read_csv(EXPORT/"check_half_truth.csv"); cm=pd.read_csv(EXPORT/"check_half_membership.csv")
    pos={int(v):i for i,v in enumerate(did)}; cpos=np.asarray([pos[int(v)] for v in cp])
    spos=np.asarray([i for i,v in enumerate(did) if int(v) not in set(cp)])
    selection={k:(v.iloc[spos].reset_index(drop=True) if k in ("truth","membership") else v[spos]) for k,v in dev.items()}
    check={"truth":ct,"membership":cm,**{k:v[cpos] for k,v in dev.items() if k not in ("truth","membership")}}
    pred=np.load(ROOT/"outputs/char-tfidf-v15-experiment/predictions.npz")
    base,base0=pred["word_char_blend_dev"],pred["word_char_blend_fold0"]
    reports={}; arrays={"base_dev":base,"base_fold0":base0}
    for name,cls in TARGETS.items():
        binary=(y==cls).astype(np.uint8)
        variants=[]
        for pos_weight in (1.,2.,4.,8.):
            model=LGBMClassifier(objective="binary",n_estimators=600,learning_rate=.035,
                num_leaves=31,min_child_samples=80,reg_lambda=12,
                colsample_bytree=.85,subsample=.9,subsample_freq=1,
                scale_pos_weight=pos_weight,n_jobs=8,random_state=20261013,verbosity=-1)
            model.fit(xtrain,binary)
            pdev=model.predict_proba(xdev)[:,1]; pf=model.predict_proba(xf)[:,1]
            for alpha in (0.,.1,.2,.3,.4,.5,.65,.8,1.):
                q=inject(base,pdev,cls,alpha); qf=inject(base0,pf,cls,alpha)
                variants.append({"scale_pos_weight":pos_weight,"alpha":alpha,
                    "selection":audit.score_probability(selection,q[spos]),
                    "check":audit.score_probability(check,q[cpos]),
                    "development":audit.score_probability(dev,q),
                    "fold0_diagnostic_only":audit.score_probability(fold,qf),
                    "p_dev":pdev,"p_fold0":pf})
        best=max(variants,key=lambda r:r["selection"]["score_proxy"])
        reports[name]={k:v for k,v in best.items() if k not in ("p_dev","p_fold0")}
        arrays[name+"_prob_dev"]=best["p_dev"]; arrays[name+"_prob_fold0"]=best["p_fold0"]
        base=inject(base,best["p_dev"],cls,best["alpha"]); base0=inject(base0,best["p_fold0"],cls,best["alpha"])
        print(f"{name}: weight={best['scale_pos_weight']} alpha={best['alpha']} check={best['check']['score_proxy']:.5f}",flush=True)
    reports["combined"]={"check":audit.score_probability(check,base[cpos]),"fold0":audit.score_probability(fold,base0)}
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/"metrics.json").write_text(json.dumps(reports,indent=2),encoding="utf-8")
    np.savez_compressed(DEST/"predictions.npz",**arrays)
    print(json.dumps(reports,indent=2),flush=True)

if __name__=="__main__": main()
