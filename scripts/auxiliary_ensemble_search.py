"""Combine the two independent auxiliary signals found in the current search."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff/v15/assembled"
sys.path.insert(0, str(HANDOFF))
sys.path.insert(0, str(ROOT / "scripts"))
from referee.data import concatenate, load_data
from referee.relations import relation_features
from referee.stack import candidate_features
import v15_mix_audit as audit


OUTPUTS = HANDOFF / "outputs"
EXPORT = HANDOFF / "evaluation_export"
DEST = ROOT / "outputs/auxiliary-ensemble-search"
LABELS = audit.LABELS


def norm(q):
    q = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    return q / q.sum(1, keepdims=True)


def main():
    cache = "/private/tmp/v15_oof_feature_cache"
    paths = [HANDOFF / "data/competition/train.csv", HANDOFF / "data/train.csv"]
    pieces = [relation_features(load_data(path, cache=cache), path, cache=cache) for path in paths]
    data = concatenate(*pieces)
    dev = audit.load_slice("development_fold1", None, OUTPUTS / "pair_selector_v15/validation.npz")
    fold0 = audit.load_slice("fold0", None, OUTPUTS / "pair_selector_v15/fold0/validation.npz")
    auxiliary = joblib.load(ROOT / "outputs/oof-binary-candidate-experiment/model.joblib")
    lexical = joblib.load(ROOT / "outputs/tfidf-v15-experiment/model.joblib")

    binary_outputs, text_outputs = [], []
    for folder, slice_ in (("relations_v5", dev), ("control_relations_v5", fold0)):
        saved = np.load(OUTPUTS / folder / "validation.npz")
        turns = np.load(OUTPUTS / folder / "turns.npz")
        idx = slice_["membership"].official_row_index.to_numpy()
        np.testing.assert_array_equal(saved["idx"], idx)
        evidence = turns["evidence"]
        ep = norm(evidence ** 2)
        run_p = norm(np.maximum(2 * saved["label_probabilities"] - ep, 0))
        features = candidate_features(data, idx, run_p, evidence, turns["candidates"])
        binary = auxiliary["model"].predict_proba(features)[:, 1].reshape(len(idx), 7)
        binary_outputs.append(norm(binary))
        docs = [data["documents"][i] for i in idx]
        text = lexical["lexical"].predict_proba(lexical["vectorizer"].transform(docs))
        text_outputs.append(text)

    pair_gates = np.asarray([1, .5, 1, 0, .5, .5, .5])
    rank_gates = np.asarray([.25, .5, .75, 1, 1, 1, .5])
    seq_gates = np.asarray([.5, 0, 1, .5, .25, .5, .5])
    seq_prob = np.load(OUTPUTS / "sequence_v6/validation.npz")["label_probabilities"]
    seq0_prob = np.load(OUTPUTS / "sequence_v6/fold0/validation.npz")["label_probabilities"]
    rank = np.load(OUTPUTS / "stack_v9/validation.npz")["label_probabilities"]
    rank0 = np.load(OUTPUTS / "stack_v9/fold0/validation.npz")["label_probabilities"]
    base_dev = .25 * audit.predict(dev, np.full(7, .5)) + .75 * audit.predict(dev, pair_gates)
    base0 = .25 * audit.predict(fold0, np.full(7, .5)) + .75 * audit.predict(fold0, pair_gates)
    qv9 = norm((1 - rank_gates) * base_dev + rank_gates * rank)
    qv90 = norm((1 - rank_gates) * base0 + rank_gates * rank0)
    current = norm((1 - seq_gates) * qv9 + seq_gates * seq_prob)
    current0 = norm((1 - seq_gates) * qv90 + seq_gates * seq0_prob)

    check_truth = pd.read_csv(EXPORT / "check_half_truth.csv")
    check_membership = pd.read_csv(EXPORT / "check_half_membership.csv")
    dev_ids = dev["membership"].official_row_index.to_numpy()
    check_ids = check_membership.official_row_index.to_numpy()
    position = {int(value): i for i, value in enumerate(dev_ids)}
    check_pos = np.asarray([position[int(value)] for value in check_ids])
    check_set = set(check_ids.tolist())
    selection_pos = np.asarray([i for i, value in enumerate(dev_ids) if int(value) not in check_set])
    selection = {key: (value.iloc[selection_pos].reset_index(drop=True)
                       if key in ("truth", "membership") else value[selection_pos])
                 for key, value in dev.items()}
    check = {"truth": check_truth, "membership": check_membership,
             **{key: value[check_pos] for key, value in dev.items()
                if key not in ("truth", "membership")}}

    binary_dev, binary0 = binary_outputs
    text_dev, text0 = text_outputs
    choices = [(t, b) for t in (0., .25, .5, .75, 1.)
               for b in (0., .25, .5, .75, 1.) if t + b <= 1.]
    gates = np.zeros((7, 2))
    for _ in range(2):
        for cls in range(7):
            opts = []
            for text_gate, binary_gate in choices:
                trial = gates.copy()
                trial[cls] = [text_gate, binary_gate]
                q = norm((1 - trial.sum(1)) * current
                         + trial[:, 0] * text_dev + trial[:, 1] * binary_dev)
                opts.append((audit.score_probability(selection, q[selection_pos])["score_proxy"],
                             text_gate, binary_gate))
            _, gates[cls, 0], gates[cls, 1] = max(opts)

    q_dev = norm((1 - gates.sum(1)) * current
                 + gates[:, 0] * text_dev + gates[:, 1] * binary_dev)
    q0 = norm((1 - gates.sum(1)) * current0
              + gates[:, 0] * text0 + gates[:, 1] * binary0)
    result = {
        "gates_by_class_text_binary": gates.tolist(),
        "baseline": {
            "selection": audit.score_probability(selection, current[selection_pos]),
            "check": audit.score_probability(check, current[check_pos]),
            "development": audit.score_probability(dev, current),
            "fold0": audit.score_probability(fold0, current0),
        },
        "combined": {
            "selection": audit.score_probability(selection, q_dev[selection_pos]),
            "check": audit.score_probability(check, q_dev[check_pos]),
            "development": audit.score_probability(dev, q_dev),
            "fold0": audit.score_probability(fold0, q0),
        },
    }
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(DEST / "predictions.npz", baseline_dev=current,
                        baseline_fold0=current0, combined_dev=q_dev,
                        combined_fold0=q0, binary_dev=binary_dev,
                        binary_fold0=binary0, text_dev=text_dev,
                        text_fold0=text0, gates=gates)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
