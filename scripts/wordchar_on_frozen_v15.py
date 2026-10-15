"""Blend word+char directly with the frozen v15 outputs, preserving v15's base."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import v15_mix_audit as audit

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff/v15/assembled"
EXPORT = HANDOFF / "evaluation_export"
DEST = ROOT / "outputs/char-tfidf-v15-experiment"


def norm(q):
    q = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    return q / q.sum(1, keepdims=True)


def main():
    dev = audit.load_slice("development_fold1", None,
        HANDOFF / "outputs/pair_selector_v15/validation.npz")
    fold0 = audit.load_slice("fold0", None,
        HANDOFF / "outputs/pair_selector_v15/fold0/validation.npz")
    pv15 = np.load(HANDOFF / "outputs/pair_selector_v15/validation.npz")["label_probabilities"]
    pv150 = np.load(HANDOFF / "outputs/pair_selector_v15/fold0/validation.npz")["label_probabilities"]
    text_outputs = np.load(DEST / "predictions.npz")
    text, text0 = text_outputs["word_char_dev"], text_outputs["word_char_fold0"]

    dev_ids = dev["membership"].official_row_index.to_numpy()
    truth_check = pd.read_csv(EXPORT / "check_half_truth.csv")
    membership_check = pd.read_csv(EXPORT / "check_half_membership.csv")
    check_ids = membership_check.official_row_index.to_numpy()
    positions = {int(row): i for i, row in enumerate(dev_ids)}
    check_pos = np.asarray([positions[int(row)] for row in check_ids])
    selection_pos = np.asarray([i for i, row in enumerate(dev_ids) if int(row) not in set(check_ids)])
    selection = {k: (v.iloc[selection_pos].reset_index(drop=True) if k in ("truth", "membership") else v[selection_pos]) for k,v in dev.items()}
    check = {"truth": truth_check, "membership": membership_check,
             **{k:v[check_pos] for k,v in dev.items() if k not in ("truth", "membership")}}

    global_trials = []
    for alpha in (0., .1, .2, .3, .4, .5, .6, .75, 1.):
        q = norm((1-alpha)*pv15 + alpha*text)
        global_trials.append({"alpha":alpha,
            "selection":audit.score_probability(selection,q[selection_pos]),
            "check":audit.score_probability(check,q[check_pos]),
            "development":audit.score_probability(dev,q)})
    best_global = max(global_trials,key=lambda x:x["selection"]["score_proxy"])

    gates = np.full(7,best_global["alpha"])
    choices=(0.,.1,.2,.3,.4,.5,.75,1.)
    for _ in range(2):
        for cls in range(7):
            opts=[]
            for value in choices:
                trial=gates.copy(); trial[cls]=value
                q=norm((1-trial[None,:])*pv15+trial[None,:]*text)
                opts.append((audit.score_probability(selection,q[selection_pos])["score_proxy"],value))
            gates[cls]=max(opts)[1]

    qd=norm((1-gates[None,:])*pv15+gates[None,:]*text)
    q0=norm((1-gates[None,:])*pv150+gates[None,:]*text0)
    result={"global_trials":global_trials,"best_global":best_global,
        "class_gates":gates.tolist(),
        "classwise":{"selection":audit.score_probability(selection,qd[selection_pos]),
                     "check":audit.score_probability(check,qd[check_pos]),
                     "development":audit.score_probability(dev,qd),
                     "fold0":audit.score_probability(fold0,q0)},
        "frozen_v15":{"selection":audit.score_probability(selection,pv15[selection_pos]),
                      "check":audit.score_probability(check,pv15[check_pos]),
                      "development":audit.score_probability(dev,pv15),
                      "fold0":audit.score_probability(fold0,pv150)}}
    (DEST/"frozen_v15_blend.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez_compressed(DEST/"frozen_v15_blend.npz",dev=qd,fold0=q0,gates=gates)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
