"""Test whether lexical TF-IDF adds signal to the frozen v15 ensemble."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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
DEST = ROOT / "outputs/tfidf-v15-experiment"
LABELS = np.asarray(audit.LABELS)


def normalize(q):
    q = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    return q / q.sum(1, keepdims=True)


def mix(a, b, gates):
    return normalize((1 - gates[None, :]) * a + gates[None, :] * b)


def main():
    cache = "/private/tmp/v15_oof_feature_cache"
    official_path = HANDOFF / "data/competition/train.csv"
    demo_path = HANDOFF / "data/train.csv"
    official = relation_features(load_data(official_path, cache=cache), official_path, cache=cache)
    demo = relation_features(load_data(demo_path, cache=cache), demo_path, cache=cache)
    data = concatenate(official, demo)
    population = pd.read_csv(EXPORT / "population_and_truth.csv").set_index("global_index")
    training = population.index[population.meta_and_neural_training.to_numpy(bool)].to_numpy(dtype=int)
    if len(training) != 16000:
        raise ValueError(f"Expected 16,000 meta-training rows, got {len(training)}")

    print("Fit structured LGBM + lexical TF-IDF on the fixed 16k training population", flush=True)
    structured = LGBMClassifier(
        n_estimators=850, learning_rate=.04, num_leaves=31,
        min_child_samples=25, colsample_bytree=.85, subsample=.9,
        subsample_freq=1, reg_lambda=2, n_jobs=8, random_state=42, verbosity=-1,
    ).fit(data["X"].iloc[training], data["y"][training])
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_df=.98, max_features=65000,
        sublinear_tf=True, dtype=np.float32,
    )
    text_train = vectorizer.fit_transform([data["documents"][i] for i in training])
    lexical = LogisticRegression(C=3, max_iter=500).fit(text_train, data["y"][training])

    dev = audit.load_slice("development_fold1", None, OUTPUTS / "pair_selector_v15/validation.npz")
    fold0 = audit.load_slice("fold0", None, OUTPUTS / "pair_selector_v15/fold0/validation.npz")
    dev_idx = dev["membership"].official_row_index.to_numpy()
    fold0_idx = fold0["membership"].official_row_index.to_numpy()

    p_struct_dev = structured.predict_proba(data["X"].iloc[dev_idx])
    p_struct_0 = structured.predict_proba(data["X"].iloc[fold0_idx])
    p_text_dev = lexical.predict_proba(vectorizer.transform([data["documents"][i] for i in dev_idx]))
    p_text_0 = lexical.predict_proba(vectorizer.transform([data["documents"][i] for i in fold0_idx]))
    # Check class ordering in both estimators before blending.
    np.testing.assert_array_equal(structured.classes_, lexical.classes_)
    # The structured branch must reproduce the archived development detector.
    turn_dev = np.load(OUTPUTS / "relations_v5/turns.npz")
    relation_dev = np.load(OUTPUTS / "relations_v5/validation.npz")
    ep_dev = normalize(turn_dev["evidence"] ** 2)
    archived_run = 2 * relation_dev["label_probabilities"] - ep_dev
    np.testing.assert_allclose(p_struct_dev, archived_run, atol=2e-6, rtol=2e-5)

    # Reconstruct strongest fixed v15/v9/sequence candidate, then tune only
    # the lexical contribution on selection-half.
    pair_gates = np.asarray([1, .5, 1, 0, .5, .5, .5])
    v9_gates = np.asarray([.25, .5, .75, 1, 1, 1, .5])
    seq_gates = np.asarray([.5, 0, 1, .5, .25, .5, .5])
    base_dev = .25 * audit.predict(dev, np.full(7, .5)) + .75 * audit.predict(dev, pair_gates)
    base0 = .25 * audit.predict(fold0, np.full(7, .5)) + .75 * audit.predict(fold0, pair_gates)
    rank = np.load(OUTPUTS / "stack_v9/validation.npz")["label_probabilities"]
    rank0 = np.load(OUTPUTS / "stack_v9/fold0/validation.npz")["label_probabilities"]
    v15_v9 = normalize((1 - v9_gates) * base_dev + v9_gates * rank)
    v15_v90 = normalize((1 - v9_gates) * base0 + v9_gates * rank0)
    seq = np.load(OUTPUTS / "sequence_v6/validation.npz")["label_probabilities"]
    seq0 = np.load(OUTPUTS / "sequence_v6/fold0/validation.npz")["label_probabilities"]
    current = mix(v15_v9, seq, seq_gates)
    current0 = mix(v15_v90, seq0, seq_gates)

    check_truth = pd.read_csv(EXPORT / "check_half_truth.csv")
    check_membership = pd.read_csv(EXPORT / "check_half_membership.csv")
    check_ids = check_membership.official_row_index.to_numpy()
    pos = {int(value): i for i, value in enumerate(dev_idx)}
    check_pos = np.asarray([pos[int(value)] for value in check_ids])
    excluded = set(check_ids.tolist())
    selection_pos = np.asarray([i for i, value in enumerate(dev_idx) if int(value) not in excluded])
    selection = {key: (value.iloc[selection_pos].reset_index(drop=True)
                       if key in ("truth", "membership") else value[selection_pos])
                 for key, value in dev.items()}
    check = {"truth": check_truth, "membership": check_membership,
             **{key: value[check_pos] for key, value in dev.items()
                if key not in ("truth", "membership")}}

    results = []
    for alpha in (0., .1, .2, .35, .5, .75, 1.):
        qd = normalize((1 - alpha) * current + alpha * p_text_dev)
        q0 = normalize((1 - alpha) * current0 + alpha * p_text_0)
        results.append({"text_share": alpha,
            "selection": audit.score_probability(selection, qd[selection_pos]),
            "development": audit.score_probability(dev, qd),
            "check": audit.score_probability(check, qd[check_pos]),
            "fold0": audit.score_probability(fold0, q0)})
    best = max(results, key=lambda x: x["selection"]["score_proxy"])

    # Low-dimensional class-wise calibration for the lexical component.
    gates = np.full(7, best["text_share"])
    for _ in range(2):
        for cls in range(7):
            choices = []
            for value in (0., .25, .5, .75, 1.):
                trial = gates.copy(); trial[cls] = value
                q = mix(current, p_text_dev, trial)
                choices.append((audit.score_probability(selection, q[selection_pos])["score_proxy"], value))
            gates[cls] = max(choices)[1]
    qd_class = mix(current, p_text_dev, gates)
    q0_class = mix(current0, p_text_0, gates)
    results.append({"text_share": "class-wise", "class_gates": gates.tolist(),
        "selection": audit.score_probability(selection, qd_class[selection_pos]),
        "development": audit.score_probability(dev, qd_class),
        "check": audit.score_probability(check, qd_class[check_pos]),
        "fold0": audit.score_probability(fold0, q0_class)})

    DEST.mkdir(parents=True, exist_ok=True)
    report = {"cv_domain_auc": 0.65338,
              "structured_detector_match_max_abs_error": float(np.max(np.abs(p_struct_dev - archived_run))),
              "lexical_only": {
                  "development": audit.score_probability(dev, p_text_dev),
                  "check": audit.score_probability(check, p_text_dev[check_pos]),
                  "fold0": audit.score_probability(fold0, p_text_0),
              },
              "current_candidate": {
                  "development": audit.score_probability(dev, current),
                  "check": audit.score_probability(check, current[check_pos]),
                  "fold0": audit.score_probability(fold0, current0),
              },
              "blend_trials": results,
              "selection_best": best,
              "class_gates": gates.tolist()}
    (DEST / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump({"structured": structured, "vectorizer": vectorizer,
                 "lexical": lexical, "selected_text_gates": gates.tolist()},
                DEST / "model.joblib", compress=3)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
