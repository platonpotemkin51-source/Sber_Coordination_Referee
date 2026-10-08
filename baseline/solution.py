#!/usr/bin/env python3
"""Reference baseline for the Coordination Referee competition.

DSWorks (Contest) run contract:

    python solution.py --train train.csv --test test.csv --output predictions.csv

Pipeline:  run -> feature extraction -> graph features -> tabular matrix ->
gradient boosting (LightGBM, with a scikit-learn fallback) -> label + success,
plus a rule-based fault-turn localiser.

Trains and predicts on 10k train / 4k test in a few minutes on 4 vCPU, CPU only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import extract_features, parse_run          # noqa: E402
from localize import localize                             # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]


def log(msg: str) -> None:
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


_T0 = time.time()


RUN_COLUMNS = ["goal", "agents", "shared_state", "messages", "artifacts", "topology"]
FALLBACK = {"label": "clean", "success": 1, "fault_turn": -1}


def _write_fallback(test: pd.DataFrame, path: str, why: str) -> None:
    """Emit a valid predictions.csv when the input is not a full run table.

    The platform smoke step may hand the solution a reduced file (for example
    only ``run_id``). Producing a well-formed answer keeps that step green
    instead of failing the whole submission.
    """
    log("WARNING: %s -> writing constant predictions" % why)
    ids = test["run_id"] if "run_id" in test.columns else pd.Series(
        ["row_%d" % i for i in range(len(test))])
    pd.DataFrame({"run_id": ids, "label": FALLBACK["label"],
                  "success": FALLBACK["success"],
                  "fault_turn": FALLBACK["fault_turn"]}).to_csv(path, index=False)
    log("wrote %s (%d rows, fallback)" % (path, len(ids)))



def build_matrix(df: pd.DataFrame):
    runs, rows = [], []
    for rec in df.to_dict("records"):
        run = parse_run(rec)
        runs.append(run)
        rows.append(extract_features(run))
    X = pd.DataFrame(rows).fillna(0.0)
    return runs, X


def make_models(n_classes: int):
    """LightGBM if available, otherwise scikit-learn histogram boosting."""
    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            objective="multiclass", num_class=n_classes, n_estimators=600,
            learning_rate=0.05, num_leaves=63, min_child_samples=20,
            subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)
        suc = lgb.LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.05,
            num_leaves=63, subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)
        return clf, suc, "lightgbm"
    except Exception as exc:                                   # pragma: no cover
        print("lightgbm unavailable (%s), falling back to sklearn" % exc)
        from sklearn.ensemble import HistGradientBoostingClassifier
        mk = lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=63, random_state=42)
        return mk(), mk(), "sklearn"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()

    log("reading data")
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    log("train=%d test=%d" % (len(train), len(test)))

    missing_test = [c for c in RUN_COLUMNS if c not in test.columns]
    missing_train = [c for c in RUN_COLUMNS + ["label", "success"]
                     if c not in train.columns]
    if missing_test or missing_train:
        _write_fallback(test, args.output,
                        "input is not a full run table (test misses %s, train misses %s)"
                        % (missing_test or "nothing", missing_train or "nothing"))
        return

    log("extracting features (train)")
    _, Xtr = build_matrix(train)
    log("extracting features (test)")
    test_runs, Xte = build_matrix(test)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    log("feature matrix: %d features" % Xtr.shape[1])

    ytr = train["label"].map({l: i for i, l in enumerate(LABELS)}).values
    str_ = train["success"].values

    clf, suc, backend = make_models(len(LABELS))
    log("fitting fault classifier (%s)" % backend)
    clf.fit(Xtr, ytr)
    log("fitting success classifier")
    suc.fit(Xtr, str_)

    log("predicting")
    pred_label = [LABELS[i] for i in np.asarray(clf.predict(Xte)).astype(int)]
    pred_success = np.asarray(suc.predict(Xte)).astype(int)
    pred_turn = [localize(run, lab) for run, lab in zip(test_runs, pred_label)]

    out = pd.DataFrame({"run_id": test["run_id"], "label": pred_label,
                        "success": pred_success, "fault_turn": pred_turn})
    out.to_csv(args.output, index=False)
    log("wrote %s (%d rows)" % (args.output, len(out)))


if __name__ == "__main__":
    main()
