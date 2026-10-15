"""Try binary word+char experts for the weak deadlock and goal_drift classes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff/v15/assembled"
sys.path.insert(0, str(HANDOFF))
sys.path.insert(0, str(ROOT / "scripts"))
from referee.data import concatenate, load_data
from referee.relations import relation_features
import v15_mix_audit as audit

EXPORT = HANDOFF / "evaluation_export"
OUTPUTS = HANDOFF / "outputs"
DEST = ROOT / "outputs/wordchar-ovr-experts"
LABELS = audit.LABELS
TARGETS = {"deadlock": LABELS.index("deadlock"), "goal_drift": LABELS.index("goal_drift")}


def norm(q):
    q = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    return q / q.sum(1, keepdims=True)


def inject(base, expert, cls, alpha):
    # Replace only the target-class marginal; preserve relative odds among others.
    target = (1-alpha)*base[:, cls] + alpha*expert
    other = np.maximum(1-base[:, cls], 1e-12)
    q = base.copy()
    q *= ((1-target)/other)[:, None]
    q[:, cls] = target
    return norm(q)


def main():
    cache = "/private/tmp/v15_oof_feature_cache"
    paths = [HANDOFF / "data/competition/train.csv", HANDOFF / "data/train.csv"]
    data = concatenate(*[relation_features(load_data(path, cache=cache), path, cache=cache) for path in paths])
    population = pd.read_csv(EXPORT / "population_and_truth.csv").set_index("global_index")
    train = population.index[population.meta_and_neural_training.to_numpy(bool)].to_numpy(dtype=int)
    y_train = data["y"][train]
    docs_train = [data["documents"][i] for i in train]
    docs_dev = [data["documents"][i] for i in audit.load_slice("development_fold1", None, OUTPUTS / "pair_selector_v15/validation.npz")["membership"].official_row_index]
    docs_0 = [data["documents"][i] for i in audit.load_slice("fold0", None, OUTPUTS / "pair_selector_v15/fold0/validation.npz")["membership"].official_row_index]
    bundle = joblib.load(ROOT / "outputs/char-tfidf-v15-experiment/model.joblib")
    word, char = bundle["word_vectorizer"], bundle["char_vectorizer"]
    x_train = hstack([word.transform(docs_train), char.transform(docs_train)], format="csr")
    x_dev = hstack([word.transform(docs_dev), char.transform(docs_dev)], format="csr")
    x_0 = hstack([word.transform(docs_0), char.transform(docs_0)], format="csr")

    dev = audit.load_slice("development_fold1", None, OUTPUTS / "pair_selector_v15/validation.npz")
    fold0 = audit.load_slice("fold0", None, OUTPUTS / "pair_selector_v15/fold0/validation.npz")
    pred = np.load(ROOT / "outputs/char-tfidf-v15-experiment/predictions.npz")
    base, base0 = pred["baseline_dev"], pred["baseline_fold0"]
    check_truth = pd.read_csv(EXPORT / "check_half_truth.csv")
    check_membership = pd.read_csv(EXPORT / "check_half_membership.csv")
    dev_ids = dev["membership"].official_row_index.to_numpy()
    check_ids = check_membership.official_row_index.to_numpy()
    position = {int(v):i for i,v in enumerate(dev_ids)}
    check_pos = np.asarray([position[int(v)] for v in check_ids])
    selection_pos = np.asarray([i for i,v in enumerate(dev_ids) if int(v) not in set(check_ids)])
    selection = {k:(v.iloc[selection_pos].reset_index(drop=True) if k in ("truth","membership") else v[selection_pos]) for k,v in dev.items()}
    check = {"truth":check_truth,"membership":check_membership,
             **{k:v[check_pos] for k,v in dev.items() if k not in ("truth","membership")}}

    alpha_values=(0.,.1,.2,.3,.4,.5,.65,.8,1.)
    reports={}
    pred_arrays={"base_dev":base,"base_fold0":base0}
    for label, cls in TARGETS.items():
        target=(y_train==cls).astype(np.uint8)
        candidates=[]
        for class_weight in (None,"balanced"):
            for c in (.5,1.,3.,8.):
                model=LogisticRegression(C=c,max_iter=500,class_weight=class_weight)
                model.fit(x_train,target)
                p_dev=model.predict_proba(x_dev)[:,1]
                p0=model.predict_proba(x_0)[:,1]
                for alpha in alpha_values:
                    q=inject(base,p_dev,cls,alpha)
                    candidates.append({"C":c,"class_weight":class_weight,"alpha":alpha,
                        "selection":audit.score_probability(selection,q[selection_pos]),
                        "check":audit.score_probability(check,q[check_pos]),
                        "development":audit.score_probability(dev,q),
                        "fold0_diagnostic_only":audit.score_probability(fold0,inject(base0,p0,cls,alpha)),
                        "p_dev":p_dev,"p0":p0})
        best=max(candidates,key=lambda row:row["selection"]["score_proxy"])
        reports[label]={k:v for k,v in best.items() if k not in ("p_dev","p0")}
        pred_arrays[f"{label}_prob_dev"]=best["p_dev"]
        pred_arrays[f"{label}_prob_fold0"]=best["p0"]
        base=inject(base,best["p_dev"],cls,best["alpha"])
        base0=inject(base0,best["p0"],cls,best["alpha"])
        print(f"{label}: alpha={best['alpha']}, check={best['check']['score_proxy']:.5f}, fold0={best['fold0_diagnostic_only']['score_proxy']:.5f}",flush=True)
    reports["combined_after_sequential_selection"]={
        "selection":audit.score_probability(selection,base[selection_pos]),
        "check":audit.score_probability(check,base[check_pos]),
        "development":audit.score_probability(dev,base),
        "fold0_diagnostic_only":audit.score_probability(fold0,base0)}
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/"metrics.json").write_text(json.dumps(reports,indent=2),encoding="utf-8")
    np.savez_compressed(DEST/"predictions.npz",**pred_arrays)
    print(json.dumps(reports,indent=2),flush=True)


if __name__=="__main__":
    main()
