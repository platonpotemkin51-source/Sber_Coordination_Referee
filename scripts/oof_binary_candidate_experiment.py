"""Compare binary candidate classification with the frozen LambdaRanker.

The candidate classifier is trained only on honest saved OOF detector outputs.
Run-content groups, rather than row identifiers, define its internal folds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff/v15/assembled"
sys.path.insert(0, str(HANDOFF))
sys.path.insert(0, str(ROOT / "scripts"))

from referee.data import concatenate, load_data
from referee.relations import relation_features
from referee.stack import candidate_features
import v15_mix_audit as audit


LABELS = np.asarray(audit.LABELS)
LABEL_COUNT = len(LABELS)
META = HANDOFF / "evaluation_export/meta_training_oof.csv"
POPULATION = HANDOFF / "evaluation_export/population_and_truth.csv"
OUTPUTS = HANDOFF / "outputs"
DEST = ROOT / "outputs/oof-binary-candidate-experiment"


def softmax(scores):
    scores = scores - scores.max(1, keepdims=True)
    result = np.exp(scores)
    return result / np.maximum(result.sum(1, keepdims=True), 1e-12)


def score_rows(data, rows, q, success, candidates, reference_training):
    y_idx = data["y"][rows]
    labels = LABELS[q.argmax(1)]
    truth = LABELS[y_idx]
    turns = candidates[np.arange(len(rows)), q.argmax(1)]
    from evaluation.metrics import binary_f1, composite, fault_turn_hit_at_k, macro_f1

    cutoff = np.median(data["X"].n_messages.iloc[reference_training])
    hard = ((data["X"].n_messages.iloc[rows].to_numpy() > cutoff)
            & ((data["X"].topo_mesh.iloc[rows].to_numpy()
                + data["X"].topo_blackboard.iloc[rows].to_numpy()) > 0))
    scores = {
        "macro_f1": macro_f1(truth, labels),
        "robustness_f1": macro_f1(truth[hard], labels[hard]) if hard.any() else 0.,
        "success_f1": binary_f1(data["success"][rows], success),
        "fault_turn_hit2": fault_turn_hit_at_k(
            truth, labels, data["fault"][rows], turns
        ),
        "robustness_proxy_n": int(hard.sum()),
    }
    scores["score_proxy"] = composite(scores)
    return scores


def normalize_probability_rows(values):
    values = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
    return values / values.sum(1, keepdims=True)


def fit_model(x, labels, class_counts):
    weights_per_run = len(labels) / (LABEL_COUNT * np.maximum(class_counts[labels], 1))
    candidate_targets = (np.arange(LABEL_COUNT)[None, :] == labels[:, None]).astype(np.uint8).ravel()
    model = LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=.035,
        num_leaves=31, min_child_samples=100, reg_lambda=12,
        colsample_bytree=.85, subsample=.9, subsample_freq=1,
        n_jobs=6, random_state=20261013, verbosity=-1,
    )
    model.fit(x, candidate_targets, sample_weight=np.repeat(weights_per_run, LABEL_COUNT))
    return model


def candidate_probabilities(model, features, rows):
    positive = model.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1]
    return normalize_probability_rows(positive.reshape(len(rows), LABEL_COUNT))


def main():
    cache = "/private/tmp/v15_oof_feature_cache"
    data_official = load_data(HANDOFF / "data/competition/train.csv", cache=cache)
    data_demo = load_data(HANDOFF / "data/train.csv", cache=cache)
    data_official = relation_features(
        data_official, HANDOFF / "data/competition/train.csv", cache=cache
    )
    data_demo = relation_features(data_demo, HANDOFF / "data/train.csv", cache=cache)
    data = concatenate(data_official, data_demo)

    oof = pd.read_csv(META)
    population = pd.read_csv(POPULATION).set_index("global_index")
    rows = oof.global_index.to_numpy(dtype=int)
    if len(rows) != 16000 or len(np.unique(rows)) != len(rows):
        raise ValueError("Expected 16,000 unique honest OOF rows")
    np.testing.assert_array_equal(
        population.loc[rows, "meta_and_neural_training"].to_numpy(), True
    )
    np.testing.assert_array_equal(population.loc[rows, "run_id"].to_numpy(), data["run_id"][rows])

    p = oof[[f"run_p_{name}" for name in LABELS]].to_numpy(dtype=np.float32)
    evidence = oof[[f"evidence_{name}" for name in LABELS]].to_numpy(dtype=np.float32)
    candidates = oof[[f"candidate_{name}" for name in LABELS]].to_numpy(dtype=np.int64)
    y = data["y"][rows]
    features = candidate_features(data, rows, p, evidence, candidates).reshape(len(rows), LABEL_COUNT, -1)
    x = features.reshape(-1, features.shape[-1])
    cached_success = {}
    for cache_path in sorted((OUTPUTS / "cache").glob("stack-oof-*.npz")):
        with np.load(cache_path) as cached:
            for global_index, success in zip(cached["checking"], cached["success"]):
                cached_success[int(global_index)] = float(success)
    if set(rows.tolist()) - set(cached_success):
        raise ValueError("Saved OOF success probabilities do not cover the meta-training rows")
    success_probability = np.asarray([cached_success[int(index)] for index in rows])

    print(f"OOF rows={len(rows)} candidate matrix={x.shape}", flush=True)
    # The rows come from official folds 2-4 and the additional source. Use
    # content hashes as groups, keeping an entire content family on one side.
    groups = population.loc[rows, "group_sha256"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261013)
    reports = []
    for fold, (fit_pos, valid_pos) in enumerate(cv.split(rows, y, groups)):
        fit_rows, valid_rows = rows[fit_pos], rows[valid_pos]
        counts = np.bincount(data["y"][fit_rows], minlength=LABEL_COUNT)
        model = fit_model(features[fit_pos].reshape(-1, features.shape[-1]), data["y"][fit_rows], counts)
        q = candidate_probabilities(model, features[valid_pos], valid_rows)
        baseline = .5 * p[valid_pos] + .5 * normalize_probability_rows(evidence[valid_pos] ** 2)
        success = (success_probability[valid_pos] >= .45).astype(int)
        reports.append({
            "fold": fold,
            "candidate_f1": score_rows(data, valid_rows, q, success, candidates[valid_pos], fit_rows),
            "baseline_f1": score_rows(data, valid_rows, baseline, success, candidates[valid_pos], fit_rows),
        })
        print(f"inner fold {fold}: candidate={reports[-1]['candidate_f1']['score_proxy']:.5f} "
              f"baseline={reports[-1]['baseline_f1']['score_proxy']:.5f}", flush=True)

    class_counts = np.bincount(y, minlength=LABEL_COUNT)
    model = fit_model(x, y, class_counts)

    dev = audit.load_slice(
        "development_fold1", None, OUTPUTS / "pair_selector_v15/validation.npz"
    )
    fold0 = audit.load_slice(
        "fold0", None, OUTPUTS / "pair_selector_v15/fold0/validation.npz"
    )
    pred_dev, pred_fold0 = [], []
    for split_name, split_rows in (("relations_v5", dev["membership"].official_row_index.to_numpy()),
                                   ("control_relations_v5", fold0["membership"].official_row_index.to_numpy())):
        detector_path = OUTPUTS / split_name / "validation.npz"
        turns_path = OUTPUTS / split_name / "turns.npz"
        saved, turns = np.load(detector_path), np.load(turns_path)
        np.testing.assert_array_equal(saved["idx"], split_rows)
        evidence_ext = turns["evidence"]
        ep = normalize_probability_rows(evidence_ext ** 2)
        run_p = 2 * saved["label_probabilities"] - ep
        run_p = np.clip(run_p, 0, None)
        run_p = normalize_probability_rows(run_p)
        ext_features = candidate_features(data, split_rows, run_p, evidence_ext,
                                          turns["candidates"]).reshape(len(split_rows), LABEL_COUNT, -1)
        prediction = candidate_probabilities(model, ext_features, split_rows)
        (pred_dev if split_name == "relations_v5" else pred_fold0).append(prediction)
    pred_dev, pred_fold0 = pred_dev[0], pred_fold0[0]

    # Reconstruct the strongest saved candidate and ask whether the new
    # classifier adds complementary information on the untouched check half.
    pair_gates = np.asarray([1, .5, 1, 0, .5, .5, .5])
    v9_gates = np.asarray([.25, .5, .75, 1, 1, 1, .5])
    seq_gates = np.asarray([.5, 0, 1, .5, .25, .5, .5])
    base_dev = .25 * audit.predict(dev, np.full(7, .5)) + .75 * audit.predict(dev, pair_gates)
    base0 = .25 * audit.predict(fold0, np.full(7, .5)) + .75 * audit.predict(fold0, pair_gates)
    old_rank = np.load(OUTPUTS / "stack_v9/validation.npz")["label_probabilities"]
    old_rank0 = np.load(OUTPUTS / "stack_v9/fold0/validation.npz")["label_probabilities"]
    old = (1 - v9_gates) * base_dev + v9_gates * old_rank
    old0 = (1 - v9_gates) * base0 + v9_gates * old_rank0
    old = normalize_probability_rows(old)
    old0 = normalize_probability_rows(old0)
    seq = np.load(OUTPUTS / "sequence_v6/validation.npz")["label_probabilities"]
    seq0 = np.load(OUTPUTS / "sequence_v6/fold0/validation.npz")["label_probabilities"]
    best = normalize_probability_rows((1 - seq_gates) * old + seq_gates * seq)
    best0 = normalize_probability_rows((1 - seq_gates) * old0 + seq_gates * seq0)

    check_membership = pd.read_csv(HANDOFF / "evaluation_export/check_half_membership.csv")
    check_truth = pd.read_csv(HANDOFF / "evaluation_export/check_half_truth.csv")
    dev_idx = dev["membership"].official_row_index.to_numpy()
    check_ids = check_membership.official_row_index.to_numpy()
    position = {int(value): i for i, value in enumerate(dev_idx)}
    check_pos = np.asarray([position[int(value)] for value in check_ids])
    selection_pos = np.asarray([i for i, value in enumerate(dev_idx) if int(value) not in set(check_ids)])
    selection = {key: (value.iloc[selection_pos].reset_index(drop=True) if key in ("truth", "membership") else value[selection_pos]) for key, value in dev.items()}
    check = {"truth": check_truth, "membership": check_membership,
             **{key: value[check_pos] for key, value in dev.items() if key not in ("truth", "membership")}}
    alt = np.full(7, .5)
    for _ in range(2):
        for cls in range(7):
            choices = []
            for value in (0., .25, .5, .75, 1.):
                trial = alt.copy(); trial[cls] = value
                q = normalize_probability_rows((1 - trial) * best + trial * pred_dev)
                choices.append((audit.score_probability(selection, q[selection_pos])["score_proxy"], value))
            alt[cls] = max(choices)[1]
    candidate = normalize_probability_rows((1 - alt) * best + alt * pred_dev)
    candidate0 = normalize_probability_rows((1 - alt) * best0 + alt * pred_fold0)
    results = {
        "baseline_ranker": {
            "selection": audit.score_probability(selection, best[selection_pos]),
            "check": audit.score_probability(check, best[check_pos]),
            "fold0": audit.score_probability(fold0, best0),
        },
        "binary_candidate_model": {
            "oof_inner_group_cv": reports,
            "selection": audit.score_probability(selection, candidate[selection_pos]),
            "development": audit.score_probability(dev, candidate),
            "check": audit.score_probability(check, candidate[check_pos]),
            "fold0": audit.score_probability(fold0, candidate0),
            "class_blend_gates": alt.tolist(),
        },
    }
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame({"run_id": check["truth"].run_id,
                  "label": LABELS[candidate[check_pos].argmax(1)],
                  "success": check["success"],
                  "fault_turn": check["candidates"][np.arange(len(check_pos)), candidate[check_pos].argmax(1)]
                  }).to_csv(DEST / "check_half.csv", index=False)
    joblib.dump({"model": model, "class_blend_gates": alt.tolist()}, DEST / "model.joblib", compress=3)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
