"""Choose the primary class between two plausible, cross-fitted candidates."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

PAIR_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def detector_fusion(probabilities, evidence):
    squared = np.asarray(evidence, dtype=np.float64) ** 2
    denominator = squared.sum(1, keepdims=True)
    if not np.isfinite(squared).all() or np.any(denominator <= 0):
        raise ValueError("Fault evidence must be finite with positive mass")
    return .5 * np.asarray(probabilities, dtype=np.float64) + .5 * squared / denominator


def top_two(probabilities):
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("Expected seven finite class probabilities per run")
    # Stable tie handling makes A/B deterministic without using labels or IDs.
    return np.argsort(-values, axis=1, kind="stable")[:, :2]


def pair_features(data, indexes, probabilities, evidence, candidates):
    """Observed features only; no y, success, fault_turn or run_id access."""
    fused = detector_fusion(probabilities, evidence)
    pairs = top_two(fused)
    count, local_width = len(indexes), data["turn_X"].shape[1]
    local = np.zeros((count, 2, 2 * local_width), dtype=np.float32)
    positions = np.full((count, 2), -1., dtype=np.float32)
    known = np.zeros((count, 2), dtype=np.float32)
    for row, index in enumerate(indexes):
        start, end = data["offsets"][index:index + 2]
        length = int(end - start)
        for member, label in enumerate(pairs[row]):
            turn = int(candidates[row, label])
            if label and 0 <= turn < length:
                window = data["turn_X"][start + max(0, turn - 2):start + min(length, turn + 3)]
                local[row, member, :local_width] = window.mean(0)
                local[row, member, local_width:] = window.max(0)
                positions[row, member] = turn / max(1, length - 1)
                known[row, member] = 1
    rows = np.arange(count)
    a, b = pairs[:, 0], pairs[:, 1]
    pa, pb = probabilities[rows, a], probabilities[rows, b]
    ea, eb = evidence[rows, a], evidence[rows, b]
    fa, fb = fused[rows, a], fused[rows, b]
    extras = np.column_stack([
        pa, pb, pa - pb, ea, eb, ea - eb, fa, fb, fa - fb,
        fa / np.maximum(fa + fb, 1e-12),
        np.log(np.maximum(fa, 1e-12) / np.maximum(fb, 1e-12)),
        positions[:, 0], positions[:, 1], positions[:, 0] - positions[:, 1],
        np.abs(positions[:, 0] - positions[:, 1]), known[:, 0], known[:, 1],
        -(fused * np.log(np.maximum(fused, 1e-12))).sum(1),
    ])
    features = np.concatenate([
        data["X"].iloc[indexes].to_numpy(dtype=np.float32),
        np.eye(7, dtype=np.float32)[pairs].reshape(count, 14),
        probabilities, evidence, fused,
        local.reshape(count, -1), local[:, 0] - local[:, 1], extras,
    ], axis=1).astype(np.float32)
    return features, pairs, fused


def redistribute_pair(original, pairs, probability_b):
    """Preserve the original top-two mass and every other class probability."""
    original = np.asarray(original, dtype=np.float64)
    probability_b = np.asarray(probability_b, dtype=np.float64)
    if probability_b.shape != (len(original),) or not np.isfinite(probability_b).all():
        raise ValueError("Expected one finite binary probability per run")
    probability_b = np.clip(probability_b, 0, 1)
    result = original.copy()
    rows = np.arange(len(result))
    mass = original[rows, pairs[:, 0]] + original[rows, pairs[:, 1]]
    result[rows, pairs[:, 0]] = mass * (1 - probability_b)
    result[rows, pairs[:, 1]] = mass * probability_b
    return result


class PairSelector:
    def fit(self, data, indexes, probabilities, evidence, candidates):
        self.run_columns = list(data["X"].columns)
        self.turn_columns = list(data["turn_columns"])
        features, pairs, _ = pair_features(data, indexes, probabilities, evidence, candidates)
        target = data["y"][indexes]
        eligible = (pairs == target[:, None]).any(1)
        binary_target = (pairs[:, 1] == target).astype(int)
        if len(np.unique(binary_target[eligible])) != 2:
            raise ValueError("Pair selector training needs both binary target classes")
        self.training_rows = int(len(indexes))
        self.eligible_training_rows = int(eligible.sum())
        self.target_b_share = float(binary_target[eligible].mean())
        self.classifier = LGBMClassifier(
            n_estimators=450, learning_rate=.04, num_leaves=15,
            min_child_samples=80, reg_lambda=10, colsample_bytree=.85,
            n_jobs=4, random_state=42, verbosity=-1,
        )
        self.classifier.fit(features[eligible], binary_target[eligible])
        return self

    def predict(self, data, indexes, probabilities, evidence, candidates):
        aligned = dict(data)
        aligned["X"] = data["X"].reindex(columns=self.run_columns, fill_value=0)
        columns = [data["turn_columns"].index(name) for name in self.turn_columns]
        aligned["turn_X"] = data["turn_X"][:, columns]
        aligned["turn_columns"] = self.turn_columns
        features, pairs, original = pair_features(aligned, indexes, probabilities, evidence, candidates)
        probability_b = self.classifier.predict_proba(features)[:, 1]
        return redistribute_pair(original, pairs, probability_b)
