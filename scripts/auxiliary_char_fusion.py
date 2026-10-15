"""Check whether independent auxiliary and word/char text gains combine."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import v15_mix_audit as audit

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff/v15/assembled"
EXPORT = HANDOFF / "evaluation_export"
DEST = ROOT / "outputs/auxiliary-char-fusion"


def norm(q):
    q = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    return q / q.sum(1, keepdims=True)


def main():
    dev = audit.load_slice("development_fold1", None,
        HANDOFF / "outputs/pair_selector_v15/validation.npz")
    fold0 = audit.load_slice("fold0", None,
        HANDOFF / "outputs/pair_selector_v15/fold0/validation.npz")
    aux = np.load(ROOT / "outputs/auxiliary-ensemble-search/predictions.npz")
    char = np.load(ROOT / "outputs/char-tfidf-v15-experiment/predictions.npz")
    p_aux, p_aux0 = aux["combined_dev"], aux["combined_fold0"]
    p_text, p_text0 = char["word_char_blend_dev"], char["word_char_blend_fold0"]

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

    gates = np.zeros(7)
    for _ in range(2):
        for cls in range(7):
            options = []
            for value in (0., .1, .2, .3, .4, .5, .6, .7, .8, 1.):
                candidate = gates.copy(); candidate[cls] = value
                q = norm((1-candidate[None,:])*p_aux + candidate[None,:]*p_text)
                options.append((audit.score_probability(selection, q[selection_pos])["score_proxy"], value))
            gates[cls] = max(options)[1]
    q_dev = norm((1-gates[None,:])*p_aux + gates[None,:]*p_text)
    q0 = norm((1-gates[None,:])*p_aux0 + gates[None,:]*p_text0)
    metrics = {
        "gates_text_over_aux_by_class": gates.tolist(),
        "auxiliary": {"selection": audit.score_probability(selection,p_aux[selection_pos]),
                      "check": audit.score_probability(check,p_aux[check_pos]),
                      "development": audit.score_probability(dev,p_aux),
                      "fold0": audit.score_probability(fold0,p_aux0)},
        "word_char": {"selection": audit.score_probability(selection,p_text[selection_pos]),
                      "check": audit.score_probability(check,p_text[check_pos]),
                      "development": audit.score_probability(dev,p_text),
                      "fold0": audit.score_probability(fold0,p_text0)},
        "fused": {"selection": audit.score_probability(selection,q_dev[selection_pos]),
                  "check": audit.score_probability(check,q_dev[check_pos]),
                  "development": audit.score_probability(dev,q_dev),
                  "fold0": audit.score_probability(fold0,q0)},
    }
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(DEST / "predictions.npz", dev=q_dev, fold0=q0, gates=gates)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
