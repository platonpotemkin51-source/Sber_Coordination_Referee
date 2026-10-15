"""Compare direct and clean/fault hierarchical run classification on fixed groups."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from baseline.features import parse_run
from baseline.localize import localize
from evaluation.metrics import binary_f1, composite, fault_turn_hit_at_k, macro_f1, f1_per_class
from referee.data import LABELS, fixed_folds, load_data
from referee.model import RunModel
from referee.relations import relation_features


def hierarchical_probabilities(x_train, y_train, x_valid, trees: int, leaves: int):
    """Estimate clean probability separately, then conditional fault class."""
    binary = LGBMClassifier(n_estimators=trees, learning_rate=.04, num_leaves=leaves,
        min_child_samples=25, colsample_bytree=.85, subsample=.9, subsample_freq=1,
        reg_lambda=2, n_jobs=8, random_state=42, verbosity=-1)
    binary.fit(x_train, (y_train != 0).astype(np.int8))
    p_fault = binary.predict_proba(x_valid)[:, 1]

    faults = y_train != 0
    conditional = LGBMClassifier(n_estimators=trees, learning_rate=.04, num_leaves=leaves,
        min_child_samples=25, colsample_bytree=.85, subsample=.9, subsample_freq=1,
        reg_lambda=2, n_jobs=8, random_state=42, verbosity=-1)
    conditional.fit(x_train.loc[faults], y_train[faults] - 1)
    p_conditional = conditional.predict_proba(x_valid)
    out = np.zeros((len(x_valid), len(LABELS)), dtype=np.float64)
    out[:, 0] = 1 - p_fault
    out[:, 1:] = p_fault[:, None] * p_conditional
    return out


def score_variant(data, indices, probs, success, runs, folds, fold):
    truth = data["y"][indices]
    pred = probs.argmax(axis=1)
    turns = np.asarray([localize(runs[i], LABELS[c]) for i, c in zip(indices, pred)])
    names_true = [LABELS[i] for i in truth]
    names_pred = [LABELS[i] for i in pred]
    train = np.flatnonzero(folds != fold)
    lengths = data["X"].iloc[indices].n_messages.to_numpy()
    cutoff = float(data["X"].iloc[train].n_messages.median())
    topology = (data["X"].iloc[indices].topo_mesh + data["X"].iloc[indices].topo_blackboard).to_numpy()
    hard = (lengths > cutoff) & (topology > 0)
    scores = {
        "macro_f1": macro_f1(names_true, names_pred),
        "robustness_f1": macro_f1(np.asarray(names_true)[hard], np.asarray(names_pred)[hard]) if hard.any() else 0.0,
        "success_f1": binary_f1(data["success"][indices], success),
        "fault_turn_hit2": fault_turn_hit_at_k(names_true, names_pred, data["fault"][indices], turns),
        "robustness_proxy_n": int(hard.sum()),
        "per_class_f1": f1_per_class(names_true, names_pred, LABELS),
    }
    scores["score_proxy"] = composite(scores)
    return scores, pred, turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv.gz")
    parser.add_argument("--trees", type=int, default=850)
    parser.add_argument("--leaves", type=int, default=31)
    parser.add_argument("--output", default="outputs/hierarchical-cv/metrics.json")
    args = parser.parse_args()

    started = time.monotonic()
    raw = load_data(args.train)
    data = relation_features(raw, args.train)
    frame = pd.read_csv(args.train)
    runs = [parse_run(row) for row in frame.to_dict("records")]
    folds = fixed_folds(data)
    n = len(data["y"])
    oof = {name: {"probabilities": np.zeros((n, len(LABELS))), "success": np.zeros(n),
                  "predictions": np.zeros(n, dtype=np.int64), "turns": np.full(n, -1, dtype=np.int64)}
           for name in ("direct", "hierarchical")}
    fold_reports = []

    for fold in range(5):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        x_train = data["X"].iloc[train]
        x_valid = data["X"].iloc[valid]
        y_train = data["y"][train]
        print(f"fold={fold} train={len(train)} valid={len(valid)}", flush=True)

        direct = RunModel("struct", args.trees, args.leaves).fit(data, train)
        p_direct, p_success = direct.predict(data, valid)
        p_hier = hierarchical_probabilities(x_train, y_train, x_valid, args.trees, args.leaves)
        success_pred = (p_success >= .45).astype(np.int8)

        for name, p in (("direct", p_direct), ("hierarchical", p_hier)):
            scores, pred, turns = score_variant(data, valid, p, success_pred, runs, folds, fold)
            oof[name]["probabilities"][valid] = p
            oof[name]["success"][valid] = p_success
            oof[name]["predictions"][valid] = pred
            oof[name]["turns"][valid] = turns
            fold_reports.append({"fold": fold, "variant": name, **scores})
            print(json.dumps({"fold": fold, "variant": name, **scores}, ensure_ascii=False), flush=True)

    aggregate = {}
    for name in oof:
        pred = oof[name]["predictions"]
        y = data["y"]
        hard = np.zeros(n, dtype=bool)
        for fold in range(5):
            idx = np.flatnonzero(folds == fold)
            train = np.flatnonzero(folds != fold)
            cutoff = float(data["X"].iloc[train].n_messages.median())
            x = data["X"].iloc[idx]
            hard[idx] = ((x.n_messages.to_numpy() > cutoff) & ((x.topo_mesh + x.topo_blackboard).to_numpy() > 0))
        true_names = [LABELS[i] for i in y]
        pred_names = [LABELS[i] for i in pred]
        scores = {
            "macro_f1": macro_f1(true_names, pred_names),
            "robustness_f1": macro_f1(np.asarray(true_names)[hard], np.asarray(pred_names)[hard]) if hard.any() else 0.0,
            "success_f1": binary_f1(data["success"], (oof[name]["success"] >= .45).astype(np.int8)),
            "fault_turn_hit2": fault_turn_hit_at_k(true_names, pred_names, data["fault"], oof[name]["turns"]),
            "per_class_f1": f1_per_class(true_names, pred_names, LABELS),
        }
        scores["score_proxy"] = composite(scores)
        aggregate[name] = scores

    result = {"configuration": vars(args), "source_sha256": data["source_sha256"],
        "folds": {str(i): int((folds == i).sum()) for i in range(5)},
        "aggregate_oof": aggregate, "fold_results": fold_reports,
        "elapsed_seconds": time.monotonic() - started}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["aggregate_oof"], indent=2, ensure_ascii=False), flush=True)
    print(f"saved={target} elapsed={result['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
