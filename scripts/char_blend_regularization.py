"""Compare class gates with simpler and shrunken word+char blend weights."""
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
    pred = np.load(DEST / "predictions.npz")
    base, text = pred["baseline_dev"], pred["word_char_dev"]
    base0, text0 = pred["baseline_fold0"], pred["word_char_fold0"]
    gates = np.asarray([.1, .1, .5, .3, 0., 0., .2])

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

    trials = []
    for scalar in (0., .1, .2, .3, .4, .5, .6, .7, .8, 1.):
        for shrink in (0., .25, .5, .75, 1.):
            trial = scalar + shrink * (gates - scalar)
            qd = norm((1-trial[None,:])*base + trial[None,:]*text)
            q0 = norm((1-trial[None,:])*base0 + trial[None,:]*text0)
            trials.append({"scalar": scalar, "shrink": shrink, "gates": trial.tolist(),
                           "selection": audit.score_probability(selection,qd[selection_pos]),
                           "check": audit.score_probability(check,qd[check_pos]),
                           "development": audit.score_probability(dev,qd),
                           "fold0": audit.score_probability(fold0,q0)})
    best = max(trials, key=lambda x:x["selection"]["score_proxy"])
    deadlock_trials = []
    for value in (0., .1, .2, .3, .4, .5, .75, 1.):
        trial = gates.copy(); trial[3] = value
        qd = norm((1-trial[None,:])*base + trial[None,:]*text)
        q0 = norm((1-trial[None,:])*base0 + trial[None,:]*text0)
        deadlock_trials.append({"deadlock_gate": value,
            "selection": audit.score_probability(selection, qd[selection_pos]),
            "check": audit.score_probability(check, qd[check_pos]),
            "development": audit.score_probability(dev, qd),
            "fold0_diagnostic_only": audit.score_probability(fold0, q0)})
    best_deadlock = max(deadlock_trials, key=lambda x:x["selection"]["score_proxy"])
    result = {"best_selection_trial": best,
              "deadlock_gate_trials": deadlock_trials,
              "best_deadlock_gate_by_selection": best_deadlock,
              "full_class_gates": next(x for x in trials if x["scalar"]==0 and x["shrink"]==1),
              "trials": trials}
    (DEST / "regularization_sweep.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="trials"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
