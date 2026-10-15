"""Probe character n-grams for structured protocol text in the v15 handoff."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff/v15/assembled"
sys.path.insert(0, str(HANDOFF))
sys.path.insert(0, str(ROOT / "scripts"))
from referee.data import concatenate, load_data
from referee.relations import relation_features
import v15_mix_audit as audit

OUTPUTS = HANDOFF / "outputs"
EXPORT = HANDOFF / "evaluation_export"
DEST = ROOT / "outputs/char-tfidf-v15-experiment"
LABELS = np.asarray(audit.LABELS)


def norm(q):
    q = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    return q / q.sum(1, keepdims=True)


def mix(a, b, gates):
    return norm((1 - gates[None, :]) * a + gates[None, :] * b)


def main():
    cache = "/private/tmp/v15_oof_feature_cache"
    paths = [HANDOFF / "data/competition/train.csv", HANDOFF / "data/train.csv"]
    pieces = [relation_features(load_data(path, cache=cache), path, cache=cache) for path in paths]
    data = concatenate(*pieces)
    population = pd.read_csv(EXPORT / "population_and_truth.csv").set_index("global_index")
    training = population.index[population.meta_and_neural_training.to_numpy(bool)].to_numpy(dtype=int)
    y = data["y"][training]
    docs_train = [data["documents"][i] for i in training]
    print(f"Fit char n-gram models on {len(training)} training documents", flush=True)

    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=180000, sublinear_tf=True, dtype=np.float32)
    x_char = char.fit_transform(docs_train)
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=.98,
                           max_features=65000, sublinear_tf=True, dtype=np.float32)
    x_word = word.fit_transform(docs_train)
    fitted = {}
    for name, matrix, c in (("char", x_char, 4.), ("word_char", hstack([x_word, x_char], format="csr"), 2.)):
        model = LogisticRegression(C=c, max_iter=600, n_jobs=8)
        model.fit(matrix, y)
        fitted[name] = (model, char if name == "char" else (word, char))

    dev = audit.load_slice("development_fold1", None, OUTPUTS / "pair_selector_v15/validation.npz")
    fold0 = audit.load_slice("fold0", None, OUTPUTS / "pair_selector_v15/fold0/validation.npz")
    dev_idx = dev["membership"].official_row_index.to_numpy()
    fold0_idx = fold0["membership"].official_row_index.to_numpy()
    docs_dev = [data["documents"][i] for i in dev_idx]
    docs_0 = [data["documents"][i] for i in fold0_idx]
    p = {}
    for name, (model, vectorizers) in fitted.items():
        if name == "char":
            xd, x0 = char.transform(docs_dev), char.transform(docs_0)
        else:
            wd, w0 = word.transform(docs_dev), word.transform(docs_0)
            cd, c0 = char.transform(docs_dev), char.transform(docs_0)
            xd, x0 = hstack([wd, cd], format="csr"), hstack([w0, c0], format="csr")
        p[name] = (model.predict_proba(xd), model.predict_proba(x0))

    pair_gates = np.asarray([1, .5, 1, 0, .5, .5, .5])
    v9_gates = np.asarray([.25, .5, .75, 1, 1, 1, .5])
    seq_gates = np.asarray([.5, 0, 1, .5, .25, .5, .5])
    def current(split, idx, control=False):
        folder = "control_relations_v5" if control else "relations_v5"
        saved = np.load(OUTPUTS / folder / "validation.npz")
        np.testing.assert_array_equal(saved["idx"], idx)
        base = .25 * audit.predict(split, np.full(7, .5)) + .75 * audit.predict(split, pair_gates)
        rank_path = OUTPUTS / "stack_v9/fold0/validation.npz" if control else OUTPUTS / "stack_v9/validation.npz"
        seq_path = OUTPUTS / "sequence_v6/fold0/validation.npz" if control else OUTPUTS / "sequence_v6/validation.npz"
        rank, seq = np.load(rank_path)["label_probabilities"], np.load(seq_path)["label_probabilities"]
        return norm((1 - seq_gates) * norm((1 - v9_gates) * base + v9_gates * rank) + seq_gates * seq)

    current_dev, current_0 = current(dev, dev_idx), current(fold0, fold0_idx, True)
    check_truth = pd.read_csv(EXPORT / "check_half_truth.csv")
    check_membership = pd.read_csv(EXPORT / "check_half_membership.csv")
    check_ids = check_membership.official_row_index.to_numpy()
    pos = {int(row): i for i, row in enumerate(dev_idx)}
    check_pos = np.asarray([pos[int(row)] for row in check_ids])
    excluded = set(check_ids.tolist())
    selection_pos = np.asarray([i for i, row in enumerate(dev_idx) if int(row) not in excluded])
    selection = {k: (v.iloc[selection_pos].reset_index(drop=True) if k in ("truth", "membership") else v[selection_pos]) for k,v in dev.items()}
    check = {"truth": check_truth, "membership": check_membership,
             **{k:v[check_pos] for k,v in dev.items() if k not in ("truth", "membership")}}

    results = {}
    prediction_arrays = {"baseline_dev": current_dev, "baseline_fold0": current_0}
    for name, (pdv, p0) in p.items():
        gates = np.full(7, .25)
        for _ in range(2):
            for cls in range(7):
                opts = []
                for value in (0., .1, .2, .3, .4, .5, .75, 1.):
                    trial = gates.copy(); trial[cls] = value
                    q = mix(current_dev, pdv, trial)
                    opts.append((audit.score_probability(selection, q[selection_pos])["score_proxy"], value))
                gates[cls] = max(opts)[1]
        qd, q0 = mix(current_dev, pdv, gates), mix(current_0, p0, gates)
        results[name] = {"class_gates": gates.tolist(),
                         "selection": audit.score_probability(selection, qd[selection_pos]),
                         "development": audit.score_probability(dev, qd),
                         "check": audit.score_probability(check, qd[check_pos]),
                         "fold0": audit.score_probability(fold0, q0)}
        prediction_arrays[f"{name}_dev"] = pdv
        prediction_arrays[f"{name}_fold0"] = p0
        prediction_arrays[f"{name}_blend_dev"] = qd
        prediction_arrays[f"{name}_blend_fold0"] = q0
        print(f"{name}: check={results[name]['check']['score_proxy']:.5f}, fold0={results[name]['fold0']['score_proxy']:.5f}", flush=True)

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(DEST / "predictions.npz", **prediction_arrays)
    joblib.dump({"word_vectorizer": word, "char_vectorizer": char,
                 "models": {name: model for name, (model, _) in fitted.items()},
                 "results": results}, DEST / "model.joblib", compress=3)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
