"""Test whether OOF success probability improves primary-fault pair selection."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from referee.data import CODE_HASH, fixed_folds, load_data
from referee.pair_selector import PairSelector, detector_fusion, pair_features, redistribute_pair
from referee.relations import RELATIONS_HASH, relation_features
from scripts.experiment import report


class SuccessAwarePairSelector:
    def fit(self, data, indexes, probabilities, success_probabilities, evidence, candidates):
        self.run_columns = list(data["X"].columns)
        self.turn_columns = list(data["turn_columns"])
        features, pairs, _ = pair_features(data, indexes, probabilities, evidence, candidates)
        rows = np.arange(len(indexes))
        pa = probabilities[rows, pairs[:, 0]]
        pb = probabilities[rows, pairs[:, 1]]
        ea = evidence[rows, pairs[:, 0]]
        eb = evidence[rows, pairs[:, 1]]
        ps = np.asarray(success_probabilities, dtype=np.float32)
        extra = np.column_stack([ps, ps * (pa - pb), ps * (ea - eb)]).astype(np.float32)
        features = np.column_stack([features, extra])
        target = data["y"][indexes]
        eligible = (pairs == target[:, None]).any(1)
        binary_target = (pairs[:, 1] == target).astype(np.int8)
        self.eligible_training_rows = int(eligible.sum())
        self.classifier = LGBMClassifier(n_estimators=450, learning_rate=.04,
            num_leaves=15, min_child_samples=80, reg_lambda=10,
            colsample_bytree=.85, n_jobs=4, random_state=42, verbosity=-1)
        self.classifier.fit(features[eligible], binary_target[eligible])
        return self

    def predict(self, data, indexes, probabilities, success_probabilities, evidence, candidates):
        aligned = dict(data)
        aligned["X"] = data["X"].reindex(columns=self.run_columns, fill_value=0)
        keep = [data["turn_columns"].index(name) for name in self.turn_columns]
        aligned["turn_X"] = data["turn_X"][:, keep]
        aligned["turn_columns"] = self.turn_columns
        features, pairs, original = pair_features(aligned, indexes, probabilities, evidence, candidates)
        rows = np.arange(len(indexes))
        pa = probabilities[rows, pairs[:, 0]]
        pb = probabilities[rows, pairs[:, 1]]
        ea = evidence[rows, pairs[:, 0]]
        eb = evidence[rows, pairs[:, 1]]
        ps = np.asarray(success_probabilities, dtype=np.float32)
        extra = np.column_stack([ps, ps * (pa - pb), ps * (ea - eb)]).astype(np.float32)
        features = np.column_stack([features, extra])
        probability_b = self.classifier.predict_proba(features)[:, 1]
        return redistribute_pair(original, pairs, probability_b)


def load_oof(data, training, cache):
    probabilities = np.zeros((len(training), 7), dtype=np.float32)
    success = np.zeros(len(training), dtype=np.float32)
    evidence = np.zeros_like(probabilities)
    candidates = np.full((len(training), 7), -1, dtype=np.int32)
    assigned = np.zeros(len(training), dtype=np.int8)
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=20261014)
    for inner, (fit, check) in enumerate(cv.split(data["X"].iloc[training], data["y"][training], data["groups"][training])):
        fitting, checking = training[fit], training[check]
        matches = []
        for path in cache.glob(f"mw-*-{inner}.npz"):
            with np.load(path) as saved:
                if np.array_equal(saved["fitting"], fitting) and np.array_equal(saved["checking"], checking):
                    matches.append(path)
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one matching audited OOF cache for inner fold {inner}, found {len(matches)}")
        with np.load(matches[0]) as saved:
            p, ps, e, c = (saved[key] for key in ("probabilities", "success", "evidence", "turns"))
            np.testing.assert_array_equal(c[:, 0, 0], -np.ones(len(c), dtype=np.int32))
        probabilities[check], success[check], evidence[check] = p, ps, e
        candidates[check] = c[:, :, 0]
        assigned[check] += 1
    if not np.all(assigned == 1):
        raise ValueError("OOF coverage is incomplete")
    return probabilities, success, evidence, candidates


def score_mix(data, idx, training, fused, selected, success, candidates, weight):
    probabilities = (1 - weight) * fused + weight * selected
    labels = probabilities.argmax(1)
    turns = candidates[np.arange(len(idx)), labels]
    metrics = report(data, idx, labels, (success >= .45).astype(np.int8), turns, training)
    return metrics, probabilities, labels, turns


def evaluate(data, idx, training, fused, success, candidates, selector_outputs, args):
    base_labels = fused.argmax(1)
    base_turns = candidates[np.arange(len(idx)), base_labels]
    result = {"baseline_fused": report(data, idx, base_labels, (success >= .45).astype(np.int8), base_turns, training)}
    select, check = train_test_split(np.arange(len(idx)), test_size=.5,
        stratify=data["y"][idx], random_state=12)
    result["single_window_ranker"] = report(data, idx, args.single_labels,
        (success >= .45).astype(np.int8), args.single_turns, training)
    variants = {}
    for name, selected in selector_outputs.items():
        trials = []
        for weight in (0, .25, .5, .75, 1):
            _, _, labels, turns = score_mix(data, idx, training, fused, selected, success, candidates, weight)
            score = report(data, idx[select], labels[select], (success[select] >= .45).astype(np.int8), turns[select], training)["score_proxy"]
            trials.append((score, weight))
        weight = max(trials)[1]
        full, probabilities, labels, turns = score_mix(data, idx, training, fused, selected, success, candidates, weight)
        full["selection_half"] = report(data, idx[select], labels[select], (success[select] >= .45).astype(np.int8), turns[select], training)
        full["check_half"] = report(data, idx[check], labels[check], (success[check] >= .45).astype(np.int8), turns[check], training)
        full.update(weight=weight, selection_trials=[{"weight": w, "score": s} for s, w in trials])
        result[name] = full
        variants[name] = {"probabilities": probabilities, "labels": labels, "turns": turns, "weight": weight}
    return result, variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv.gz")
    parser.add_argument("--output", default="outputs/success-aware-pair")
    args = parser.parse_args()
    started = time.monotonic()
    data = relation_features(load_data(args.train), args.train)
    folds = fixed_folds(data)
    valid = np.flatnonzero(folds == 1)
    fold0 = np.flatnonzero(folds == 0)
    training = np.flatnonzero(folds > 1)
    cache = ROOT / "outputs" / "multiwindow-ranker" / "cache"
    oof_p, oof_ps, oof_e, oof_c = load_oof(data, training, cache)

    print("Training pair selectors on the same held-out OOF evidence", flush=True)
    standard = PairSelector().fit(data, training, oof_p, oof_e, oof_c)
    aware = SuccessAwarePairSelector().fit(data, training, oof_p, oof_ps, oof_e, oof_c)
    directory = ROOT / "outputs" / "multiwindow-ranker"
    dev = np.load(directory / "development-single_window.npz")
    fold = np.load(directory / "fold0-single_window.npz")
    p_valid, ps_valid, e_valid, c_valid = (dev[key] for key in ("raw_probabilities", "success", "evidence", "candidates"))
    p0, ps0, e0, c0 = (fold[key] for key in ("raw_probabilities", "success", "evidence", "candidates"))
    c_valid, c0 = c_valid[:, :, 0], c0[:, :, 0]
    fused_valid = detector_fusion(p_valid, e_valid)
    fused0 = detector_fusion(p0, e0)
    output_valid = {
        "pair_selector": standard.predict(data, valid, p_valid, e_valid, c_valid),
        "success_aware_pair_selector": aware.predict(data, valid, p_valid, ps_valid, e_valid, c_valid),
    }
    output0 = {
        "pair_selector": standard.predict(data, fold0, p0, e0, c0),
        "success_aware_pair_selector": aware.predict(data, fold0, p0, ps0, e0, c0),
    }
    single_labels, single_turns = dev["labels"], dev["turns"]
    valid_result, valid_variants = evaluate(data, valid, training, fused_valid, ps_valid, c_valid, output_valid,
        argparse.Namespace(single_labels=single_labels, single_turns=single_turns))
    previous_metrics = json.loads((directory / "development.json").read_text())
    valid_result["single_window_ranker"] = previous_metrics["single_window"]
    valid_result["configuration"] = vars(args)
    valid_result["oof_training_rows"] = int(len(training))
    valid_result["eligible_rows"] = {"standard": standard.eligible_training_rows, "success_aware": aware.eligible_training_rows}
    valid_result["elapsed_seconds"] = time.monotonic() - started

    fold_metrics = {"baseline_fused": report(data, fold0, fused0.argmax(1), (ps0 >= .45).astype(np.int8), c0[np.arange(len(fold0)), fused0.argmax(1)], training)}
    fold_metrics["single_window_ranker"] = report(data, fold0, fold["labels"], (ps0 >= .45).astype(np.int8), fold["turns"], training)
    for name, probability in output0.items():
        weight = valid_variants[name]["weight"]
        metrics, mixed, labels, turns = score_mix(data, fold0, training, fused0, probability, ps0, c0, weight)
        metrics.update(weight=weight, note="Fold 0 is a repeated control; selector weight was chosen on fold 1")
        fold_metrics[name] = metrics

    result = {"development": valid_result, "fold0": fold_metrics,
        "limitation": "Only the single 10k source in the clone; success probabilities are from the tree head, not the v15 neural success checkpoint."}
    outdir = ROOT / args.output
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (outdir / "metadata.json").write_text(json.dumps({"source_sha256": data["source_sha256"],
        "features_code_hash": CODE_HASH, "relations_hash": RELATIONS_HASH}, indent=2), encoding="utf-8")
    np.savez_compressed(outdir / "development.npz", idx=valid, raw_probabilities=p_valid, evidence=e_valid,
        base_probabilities=fused_valid, success=ps_valid, candidates=c_valid,
        standard=output_valid["pair_selector"], success_aware=output_valid["success_aware_pair_selector"])
    np.savez_compressed(outdir / "fold0.npz", idx=fold0, raw_probabilities=p0, evidence=e0,
        base_probabilities=fused0, success=ps0, candidates=c0,
        standard=output0["pair_selector"], success_aware=output0["success_aware_pair_selector"])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
