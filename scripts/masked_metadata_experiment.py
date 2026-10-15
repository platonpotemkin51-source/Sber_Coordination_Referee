"""Train-only masking augmentation for partially missing protocol metadata."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff/v15/assembled"
sys.path.insert(0, str(HANDOFF))
sys.path.insert(0, str(ROOT / "scripts"))
from referee.data import concatenate, load_data
from referee.relations import relation_features
import v15_mix_audit as audit

EXPORT = HANDOFF / "evaluation_export"
OUTPUTS = HANDOFF / "outputs"
DEST = ROOT / "outputs/masked-metadata-experiment"
LABELS = np.asarray(audit.LABELS)


def normalize(q):
    q = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    return q / q.sum(1, keepdims=True)


def main():
    cache = "/private/tmp/v15_oof_feature_cache"
    official_path = HANDOFF / "data/competition/train.csv"
    demo_path = HANDOFF / "data/train.csv"
    official = relation_features(load_data(official_path, cache=cache), official_path, cache=cache)
    demo = relation_features(load_data(demo_path, cache=cache), demo_path, cache=cache)
    data = concatenate(official, demo)
    population = pd.read_csv(EXPORT / "population_and_truth.csv").set_index("global_index")
    training = population.index[population.meta_and_neural_training.to_numpy(bool)].to_numpy(dtype=int)

    # One stochastic training-only copy: randomly hide intent/refs on a quarter
    # of messages and artifact subtasks on a quarter of artifacts.
    raw = pd.concat([pd.read_csv(official_path), pd.read_csv(demo_path)], ignore_index=True)
    augmented = raw.iloc[training].copy().reset_index(drop=True)
    rng = np.random.default_rng(20261013)
    touched_messages = touched_artifacts = total_messages = total_artifacts = 0
    for row in range(len(augmented)):
        messages = json.loads(augmented.at[row, "messages"])
        artifacts = json.loads(augmented.at[row, "artifacts"])
        for message in messages:
            total_messages += 1
            if rng.random() < .25:
                if message.get("intent") or message.get("refs"):
                    touched_messages += 1
                message["intent"] = ""
                message["refs"] = []
        for artifact in artifacts:
            total_artifacts += 1
            if rng.random() < .25:
                if artifact.get("subtask"):
                    touched_artifacts += 1
                artifact["subtask"] = ""
        augmented.at[row, "messages"] = json.dumps(messages, ensure_ascii=False)
        augmented.at[row, "artifacts"] = json.dumps(artifacts, ensure_ascii=False)

    augmented_path = Path("/private/tmp/v15_masked_meta_train.csv")
    augmented.to_csv(augmented_path, index=False)
    aug_raw = load_data(augmented_path, cache=cache)
    aug = relation_features(aug_raw, augmented_path, cache=cache)
    x_original = data["X"].iloc[training].to_numpy(dtype=np.float32)
    x_augmented = aug["X"].reindex(columns=data["X"].columns, fill_value=0).to_numpy(dtype=np.float32)
    x_train = np.concatenate([x_original, x_augmented])
    y_train = np.concatenate([data["y"][training], data["y"][training]])
    weights = np.r_[np.ones(len(training)), np.full(len(training), .5)]

    classifier = LGBMClassifier(
        n_estimators=850, learning_rate=.04, num_leaves=31,
        min_child_samples=25, colsample_bytree=.85, subsample=.9,
        subsample_freq=1, reg_lambda=2, n_jobs=8, random_state=42, verbosity=-1,
    ).fit(x_train, y_train, sample_weight=weights)

    dev = audit.load_slice("development_fold1", None, OUTPUTS / "pair_selector_v15/validation.npz")
    fold0 = audit.load_slice("fold0", None, OUTPUTS / "pair_selector_v15/fold0/validation.npz")
    dev_idx = dev["membership"].official_row_index.to_numpy()
    fold0_idx = fold0["membership"].official_row_index.to_numpy()
    p_aug_dev = classifier.predict_proba(data["X"].iloc[dev_idx])
    p_aug_0 = classifier.predict_proba(data["X"].iloc[fold0_idx])

    pair_gates = np.asarray([1, .5, 1, 0, .5, .5, .5])
    v9_gates = np.asarray([.25, .5, .75, 1, 1, 1, .5])
    seq_gates = np.asarray([.5, 0, 1, .5, .25, .5, .5])
    base_dev = .25 * audit.predict(dev, np.full(7, .5)) + .75 * audit.predict(dev, pair_gates)
    base_0 = .25 * audit.predict(fold0, np.full(7, .5)) + .75 * audit.predict(fold0, pair_gates)
    rank = np.load(OUTPUTS / "stack_v9/validation.npz")["label_probabilities"]
    rank0 = np.load(OUTPUTS / "stack_v9/fold0/validation.npz")["label_probabilities"]
    q_v15v9 = normalize((1 - v9_gates) * base_dev + v9_gates * rank)
    q_v15v90 = normalize((1 - v9_gates) * base_0 + v9_gates * rank0)
    seq = np.load(OUTPUTS / "sequence_v6/validation.npz")["label_probabilities"]
    seq0 = np.load(OUTPUTS / "sequence_v6/fold0/validation.npz")["label_probabilities"]
    current = normalize((1 - seq_gates) * q_v15v9 + seq_gates * seq)
    current0 = normalize((1 - seq_gates) * q_v15v90 + seq_gates * seq0)

    check_truth = pd.read_csv(EXPORT / "check_half_truth.csv")
    check_membership = pd.read_csv(EXPORT / "check_half_membership.csv")
    check_ids = check_membership.official_row_index.to_numpy()
    position = {int(value): i for i, value in enumerate(dev_idx)}
    check_pos = np.asarray([position[int(value)] for value in check_ids])
    check_set = set(check_ids.tolist())
    selection_pos = np.asarray([i for i, value in enumerate(dev_idx) if int(value) not in check_set])
    selection = {key: (value.iloc[selection_pos].reset_index(drop=True)
                       if key in ("truth", "membership") else value[selection_pos])
                 for key, value in dev.items()}
    check = {"truth": check_truth, "membership": check_membership,
             **{key: value[check_pos] for key, value in dev.items()
                if key not in ("truth", "membership")}}

    gates = np.full(7, .25)
    for _ in range(2):
        for cls in range(7):
            choices = []
            for value in (0., .25, .5, .75, 1.):
                trial = gates.copy(); trial[cls] = value
                q = normalize((1 - trial) * current + trial * p_aug_dev)
                choices.append((audit.score_probability(selection, q[selection_pos])["score_proxy"], value))
            gates[cls] = max(choices)[1]
    q_dev = normalize((1 - gates) * current + gates * p_aug_dev)
    q_0 = normalize((1 - gates) * current0 + gates * p_aug_0)
    results = {
        "masking": {"messages_rate": .25, "artifacts_rate": .25,
                    "masked_messages_with_metadata": touched_messages,
                    "total_messages": total_messages,
                    "masked_artifacts_with_subtask": touched_artifacts,
                    "total_artifacts": total_artifacts},
        "class_gates": gates.tolist(),
        "current": {"selection": audit.score_probability(selection, current[selection_pos]),
                    "check": audit.score_probability(check, current[check_pos]),
                    "fold0": audit.score_probability(fold0, current0)},
        "augmented_classifiers": {
            "selection": audit.score_probability(selection, q_dev[selection_pos]),
            "development": audit.score_probability(dev, q_dev),
            "check": audit.score_probability(check, q_dev[check_pos]),
            "fold0": audit.score_probability(fold0, q_0)},
    }
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    joblib.dump({"classifier": classifier, "class_gates": gates.tolist()}, DEST / "model.joblib", compress=3)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
